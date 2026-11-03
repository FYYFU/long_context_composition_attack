#!/usr/bin/env python3
"""Run one safety-aligned model on the selected compositional-safety tasks.

The runner uses vLLM's offline API, the checkpoint's native chat template,
and an append-only JSONL checkpoint for safe resumption.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_DATA_DIR = Path(
    str(Path(__file__).resolve().parents[2] / 'paper/generated_tasks/gemini-2.0-flash-001')
)
DEFAULT_OUTPUT_DIR = Path(
    str(Path(__file__).resolve().parents[2] / 'robustness/safety_aligned_model_validation/results')
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--levels", default="0,1,2,3")
    parser.add_argument("--contexts", default="0k,16k")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=15000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-truncated",
        action="store_true",
        help="Regenerate checkpoint rows whose previous finish_reason is length",
    )
    return parser.parse_args()


def read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


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
                completed[int(row["sample_index"])] = row
            except Exception as exc:
                raise ValueError(
                    f"Invalid checkpoint row {path}:{line_number}: {exc}"
                ) from exc
    return completed


def split_reasoning(text: str) -> Tuple[str, str]:
    if "</think>" not in text:
        return "", text.strip()
    thinking, response = text.rsplit("</think>", 1)
    thinking = thinking.replace("<think>", "", 1).strip()
    return thinking, response.strip()


def build_result(
    *,
    sample_index: int,
    task: Dict[str, Any],
    full_text: str,
    finish_reason: str,
    input_tokens: int,
    output_tokens: int,
    model: str,
    alias: str,
) -> Dict[str, Any]:
    thinking, final_response = split_reasoning(full_text)
    return {
        "sample_index": sample_index,
        "original_task": {
            "input": task.get("input", ""),
            "question": task.get("question", ""),
            "target": task.get("target", ""),
            "original_query": task.get("original_query", ""),
            "level": task.get("level"),
            "context_length": task.get("context_length"),
            "n_facts": task.get("n_facts"),
        },
        "generated": {
            "thinking": thinking,
            "response": final_response,
            "full_text": full_text,
            "finish_reason": finish_reason,
        },
        "metadata": {
            "model": model,
            "alias": alias,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def write_final_json(path: Path, checkpoint: Dict[int, Dict[str, Any]]) -> None:
    ordered = [checkpoint[index] for index in sorted(checkpoint)]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def task_specs(levels: str, contexts: str) -> List[Tuple[int, str]]:
    parsed_levels = [int(value.strip()) for value in levels.split(",")]
    parsed_contexts = [value.strip() for value in contexts.split(",")]
    return [
        (level, context)
        for level in parsed_levels
        for context in parsed_contexts
    ]


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    output_model_dir = args.output_dir / f"{args.alias}-default"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model": args.model,
        "alias": args.alias,
        "data_dir": str(args.data_dir),
        "levels": args.levels,
        "contexts": args.contexts,
        "n_examples": args.n_examples,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    with (output_model_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    print(f"Loading {args.model}", flush=True)
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

    for level, context in task_specs(args.levels, args.contexts):
        input_path = args.data_dir / f"level_{level}_{context}.json"
        tasks = read_json(input_path)[: args.n_examples]
        if len(tasks) != args.n_examples:
            raise ValueError(
                f"{input_path} has {len(tasks)} selected tasks, expected "
                f"{args.n_examples}"
            )

        checkpoint_path = output_model_dir / f"level_{level}_{context}.jsonl"
        final_path = output_model_dir / f"level_{level}_{context}_results.json"
        if args.overwrite:
            checkpoint_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)

        completed = load_checkpoint(checkpoint_path)
        retry_indices = {
            index
            for index, row in completed.items()
            if args.retry_truncated
            and row.get("generated", {}).get("finish_reason") == "length"
        }
        pending = [
            (index, task)
            for index, task in enumerate(tasks)
            if index not in completed or index in retry_indices
        ]
        print(
            f"[{args.alias}] level={level} context={context}: "
            f"{len(completed)} complete, {len(pending)} pending"
            + (
                f" ({len(retry_indices)} truncated retries)"
                if retry_indices
                else ""
            ),
            flush=True,
        )

        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]
            messages = [
                [{"role": "user", "content": task["input"]}]
                for _, task in chunk
            ]
            formatted = [
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for message in messages
            ]
            input_lengths = [
                len(tokenizer.encode(prompt, add_special_tokens=False))
                for prompt in formatted
            ]
            for index, input_length in zip(
                [index for index, _ in chunk], input_lengths
            ):
                if input_length + args.max_new_tokens > args.max_model_len:
                    raise ValueError(
                        f"Sample {index} requires {input_length} input tokens + "
                        f"{args.max_new_tokens} output tokens, exceeding "
                        f"max_model_len={args.max_model_len}"
                    )

            outputs = llm.generate(formatted, sampling, use_tqdm=False)
            rows = []
            for (
                (sample_index, task),
                output,
                input_length,
            ) in zip(chunk, outputs, input_lengths):
                candidate = output.outputs[0]
                rows.append(
                    build_result(
                        sample_index=sample_index,
                        task=task,
                        full_text=candidate.text,
                        finish_reason=str(candidate.finish_reason),
                        input_tokens=input_length,
                        output_tokens=len(candidate.token_ids),
                        model=args.model,
                        alias=args.alias,
                    )
                )
            append_jsonl(checkpoint_path, rows)
            for row in rows:
                completed[int(row["sample_index"])] = row
            print(
                f"[{args.alias}] level={level} context={context}: "
                f"{len(completed)}/{len(tasks)}",
                flush=True,
            )

        if len(completed) != len(tasks):
            raise RuntimeError(
                f"{checkpoint_path} has {len(completed)}/{len(tasks)} results"
            )
        write_final_json(final_path, completed)
        print(f"Wrote {final_path}", flush=True)


if __name__ == "__main__":
    main()
