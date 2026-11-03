#!/usr/bin/env python3
"""Run one local checkpoint on the 250 safe prompts from XSTest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
DEFAULT_DATA = (
    REPO_ROOT
    / "controls"
    / "system_prompt_defense"
    / "xstest"
    / "xstest_safe.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--n-examples", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=15000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
    completed: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed[int(row["id"])] = row
            except Exception as exc:
                raise ValueError(
                    f"Invalid checkpoint row {path}:{line_number}: {exc}"
                ) from exc
    return completed


def split_reasoning(text: str) -> Tuple[str, str]:
    if "</think>" not in text:
        return "", text.strip()
    thinking, response = text.rsplit("</think>", 1)
    return thinking.replace("<think>", "", 1).strip(), response.strip()


def write_final(path: Path, rows_by_id: Dict[int, Dict[str, Any]]) -> None:
    rows = [rows_by_id[index] for index in sorted(rows_by_id)]
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    data = json.loads(args.data_file.read_text(encoding="utf-8"))
    data = data[: args.n_examples]
    if len(data) != args.n_examples:
        raise ValueError(
            f"{args.data_file} has {len(data)} selected prompts; "
            f"expected {args.n_examples}"
        )

    model_dir = args.output_dir / args.alias
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "xstest.jsonl"
    final_path = model_dir / "xstest_results.json"
    config = {
        "model": args.model,
        "alias": args.alias,
        "data_file": str(args.data_file),
        "n_examples": args.n_examples,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "system_prompt": None,
    }
    (model_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    completed = load_checkpoint(checkpoint_path)
    pending = [item for item in data if int(item["id"]) not in completed]
    if not pending:
        write_final(final_path, completed)
        print(f"{args.alias}: already complete ({len(completed)})", flush=True)
        return

    print(f"Loading {args.model}; pending={len(pending)}", flush=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in chunk
        ]
        input_lengths = [
            len(tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        ]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        rows: List[Dict[str, Any]] = []
        for item, output, input_tokens in zip(
            chunk,
            outputs,
            input_lengths,
        ):
            candidate = output.outputs[0]
            thinking, response = split_reasoning(candidate.text)
            rows.append(
                {
                    "id": int(item["id"]),
                    "prompt": item["prompt"],
                    "type": item["type"],
                    "label": item["label"],
                    "focus": item.get("focus", ""),
                    "response": response,
                    "thinking": thinking,
                    "full_text": candidate.text,
                    "finish_reason": str(candidate.finish_reason),
                    "input_tokens": input_tokens,
                    "output_tokens": len(candidate.token_ids),
                    "model": args.model,
                    "alias": args.alias,
                }
            )
        append_jsonl(checkpoint_path, rows)
        for row in rows:
            completed[int(row["id"])] = row
        print(
            f"{args.alias}: {len(completed)}/{args.n_examples}",
            flush=True,
        )

    write_final(final_path, completed)
    print(f"Wrote {final_path}", flush=True)


if __name__ == "__main__":
    main()
