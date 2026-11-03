#!/usr/bin/env python3
"""Run GPT-oss through OpenRouter on matched Qwen-decomposition tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


HERE = Path(__file__).resolve().parent
DEFAULT_TASKS = HERE / "generated_tasks" / "qwen3-next-80b"
DEFAULT_RESULTS = HERE / "results" / "qwen3-next-80b"
_thread_local = threading.local()


def parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_csv_strings(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one value")
    return parsed


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def get_client(api_base_url: str, api_key: str, timeout: float) -> OpenAI:
    cache_key = (api_base_url, api_key, timeout)
    if getattr(_thread_local, "cache_key", None) != cache_key:
        _thread_local.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
            timeout=timeout,
        )
        _thread_local.cache_key = cache_key
    return _thread_local.client


def completed_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    generated = record.get("generated", {})
    return bool(str(generated.get("response", "")).strip())


def failure_record(
    task: dict[str, Any],
    error: Exception,
    *,
    model_path: str,
    provider: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Preserve a terminal per-run failure without aborting other samples."""
    record = result_record(
        task,
        "",
        "",
        model_path,
        provider,
        reasoning_effort,
    )
    record["error"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    return record


def load_existing(path: Path, expected: int) -> list[Any]:
    if not path.exists():
        return [None] * expected
    value = load_json(path)
    if not isinstance(value, list):
        raise TypeError(f"existing result is not a list: {path}")
    if len(value) != expected:
        raise ValueError(
            f"existing result length mismatch: {path}, "
            f"expected={expected}, found={len(value)}"
        )
    return value


def result_record(
    task: dict[str, Any],
    thinking: str,
    response: str,
    model_path: str,
    provider: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    full_text = (
        f"{thinking}\n</think>\n{response}" if thinking else response
    )
    original_task = {
        "input": task.get("input", ""),
        "question": task.get("question", ""),
        "target": task.get("target", ""),
        "original_query": task.get("original_query", ""),
        "level": task.get("level"),
        "context_length": task.get("context_length"),
        "n_facts": task.get("n_facts"),
        "decomposition_generator": task.get(
            "decomposition_generator", "Qwen3-Next-80B"
        ),
        "source_index": task.get("source_index"),
    }
    return {
        "original_task": original_task,
        "generated": {
            "thinking": thinking,
            "response": response,
            "full_text": full_text,
        },
        "inference_config": {
            "model": model_path,
            "api": "openrouter",
            "provider": provider,
            "reasoning_effort": reasoning_effort,
        },
    }


def infer_one(
    index: int,
    task: dict[str, Any],
    *,
    model_path: str,
    api_base_url: str,
    api_key: str,
    provider: str,
    reasoning_effort: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> tuple[int, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            client = get_client(api_base_url, api_key, timeout)
            extra_body: dict[str, Any] = {
                "provider": {"only": [provider]}
            }
            if reasoning_effort and reasoning_effort != "default":
                extra_body["reasoning_effort"] = reasoning_effort
            completion = client.chat.completions.create(
                model=model_path,
                messages=[
                    {"role": "user", "content": str(task["input"])}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            message = completion.choices[0].message
            thinking = (
                getattr(message, "reasoning_content", "")
                or getattr(message, "reasoning", "")
                or ""
            )
            response = message.content or ""
            if not response.strip():
                finish_reason = getattr(
                    completion.choices[0], "finish_reason", None
                )
                raise ValueError(
                    "model returned an empty final response "
                    f"(finish_reason={finish_reason}, "
                    f"reasoning_chars={len(thinking)})"
                )
            return (
                index,
                result_record(
                    task,
                    thinking,
                    response,
                    model_path,
                    provider,
                    reasoning_effort,
                ),
            )
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                delay = min(30.0, 2 ** (attempt - 1) + random.random())
                time.sleep(delay)
    raise RuntimeError(
        f"task index {index} failed after {max_retries} attempts: {last_error}"
    )


def run_condition(
    task_file: Path,
    output_file: Path,
    args: argparse.Namespace,
) -> None:
    tasks = load_json(task_file)
    if not isinstance(tasks, list):
        raise TypeError(f"task file is not a list: {task_file}")
    if args.max_examples is not None:
        tasks = tasks[: args.max_examples]

    if args.overwrite:
        results: list[Any] = [None] * len(tasks)
    else:
        results = load_existing(output_file, len(tasks))

    pending = [
        (index, task)
        for index, (task, record) in enumerate(zip(tasks, results))
        if not completed_record(record)
    ]
    if not pending:
        print(f"[skip] complete: {output_file}")
        return

    print(
        f"[run] {task_file.name}: total={len(tasks)}, "
        f"pending={len(pending)}, workers={args.num_workers}"
    )
    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                infer_one,
                index,
                task,
                model_path=args.model_path,
                api_base_url=args.api_base_url,
                api_key=args.api_key,
                provider=args.provider,
                reasoning_effort=args.reasoning_effort,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
            ): index
            for index, task in pending
        }
        try:
            for future in as_completed(futures):
                pending_index = futures[future]
                try:
                    index, record = future.result()
                    results[index] = record
                except Exception as error:
                    results[pending_index] = failure_record(
                        tasks[pending_index],
                        error,
                        model_path=args.model_path,
                        provider=args.provider,
                        reasoning_effort=args.reasoning_effort,
                    )
                    print(
                        f"  [warning] index {pending_index} failed; "
                        "recorded and continuing"
                    )
                completed_since_save += 1
                if completed_since_save >= args.save_every:
                    atomic_write_json(output_file, results)
                    done = sum(completed_record(item) for item in results)
                    attempted = sum(item is not None for item in results)
                    print(
                        f"  checkpoint: responses={done}/{len(results)}, "
                        f"attempted={attempted}/{len(results)}"
                    )
                    completed_since_save = 0
        except BaseException:
            atomic_write_json(output_file, results)
            for future in futures:
                future.cancel()
            raise

    atomic_write_json(output_file, results)
    incomplete = [
        index
        for index, record in enumerate(results)
        if not completed_record(record)
    ]
    if incomplete:
        print(
            f"[warning] {output_file}: {len(incomplete)} samples have no "
            f"final response: {incomplete[:20]}. Other conditions will "
            "continue. Rerunning the same command retries only these indices."
        )
    else:
        print(f"[complete] {output_file}: {len(results)} responses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-output-name", required=True)
    parser.add_argument(
        "--api-base-url", default="https://openrouter.ai/api/v1"
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENROUTER_API_KEY", "")
    )
    parser.add_argument("--provider", default="google-vertex")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--levels", type=parse_csv_ints, default=[1, 2, 3])
    parser.add_argument(
        "--contexts",
        type=parse_csv_strings,
        default=["0k", "16k", '64k'],
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16368)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Use a small value, e.g. 2, for a smoke test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set and --api-key was not provided"
        )
    model_dir = args.output_dir / args.model_output_name
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_path": args.model_path,
        "model_output_name": args.model_output_name,
        "api_base_url": args.api_base_url,
        "provider": args.provider,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "levels": args.levels,
        "contexts": args.contexts,
        "max_examples": args.max_examples,
    }
    atomic_write_json(model_dir / "inference_config.json", config)
    print(json.dumps(config, indent=2))

    for level in args.levels:
        for context in args.contexts:
            task_file = (
                args.tasks_dir / f"level_{level}_{context}.json"
            )
            if not task_file.exists():
                raise FileNotFoundError(f"missing task file: {task_file}")
            output_file = (
                model_dir / f"level_{level}_{context}_results.json"
            )
            run_condition(task_file, output_file, args)


if __name__ == "__main__":
    main()
