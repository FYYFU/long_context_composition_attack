#!/usr/bin/env python3
"""Run resumable GPT-oss inference against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_TASKS = REPO_ROOT / "data" / "tasks"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "results"
DEFAULT_CHECKPOINTS = (
    REPO_ROOT / "outputs" / "checkpoints" / "inference"
)
_thread_local = threading.local()


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one value")
    return result


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


def completed_response(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("generated", {}).get("completion_received") is True
        and isinstance(record.get("generated", {}).get("response"), str)
    )


def get_client(api_base_url: str, api_key: str, timeout: float) -> OpenAI:
    cache_key = (api_base_url, api_key, timeout)
    if getattr(_thread_local, "cache_key", None) != cache_key:
        _thread_local.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
            timeout=timeout,
            # infer_one owns retries and provider-aware backoff. Avoid
            # multiplying the long inference timeout with SDK-level retries.
            max_retries=0,
        )
        _thread_local.cache_key = cache_key
    return _thread_local.client


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def result_record(
    *,
    index: int,
    task: dict[str, Any],
    task_file: Path,
    model_id: str,
    model_name: str,
    api_base_url: str,
    provider: str,
    reasoning_effort: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    run_id: str,
    thinking: str,
    response: str,
    finish_reason: str | None,
    usage: Any,
    upstream_response: dict[str, Any],
) -> dict[str, Any]:
    usage_data: dict[str, Any] = {}
    if usage is not None:
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
        ):
            value = getattr(usage, key, None)
            if value is not None:
                usage_data[key] = value
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if reasoning_tokens is not None:
                usage_data["reasoning_tokens"] = reasoning_tokens

    prompt = str(task["input"])
    return {
        "sample_index": index,
        "original_task": {
            "original_query": task.get("original_query", ""),
            "question": task.get("question", ""),
            "target": task.get("target", ""),
            "level": task.get("level"),
            "context_length": task.get("context_length"),
            "n_facts": task.get("n_facts"),
            "source_file": str(task_file.resolve()),
            "source_index": index,
            "prompt_sha256": prompt_sha256(prompt),
        },
        "generated": {
            "thinking": thinking,
            "response": response,
            "finish_reason": finish_reason,
            "completion_received": True,
            "empty_final": not bool(response.strip()),
        },
        "usage": usage_data,
        "upstream_response": upstream_response,
        "inference_config": {
            "model_id": model_id,
            "model_name": model_name,
            "api_base_url": api_base_url,
            "provider": provider,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "run_id": run_id,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def infer_one(
    index: int,
    task: dict[str, Any],
    *,
    task_file: Path,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            client = get_client(
                args.api_base_url, args.api_key, args.timeout
            )
            extra_body: dict[str, Any] = {
                "reasoning_effort": args.reasoning_effort,
            }
            if args.provider:
                extra_body["provider"] = {"only": [args.provider]}
            completion = client.chat.completions.create(
                model=args.model_id,
                messages=[
                    {"role": "user", "content": str(task["input"])}
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body=extra_body,
            )
            choice = completion.choices[0]
            message = choice.message
            thinking = (
                getattr(message, "reasoning_content", "")
                or getattr(message, "reasoning", "")
                or ""
            )
            response = message.content or ""
            model_extra = getattr(completion, "model_extra", {}) or {}
            upstream_response = {
                "id": getattr(completion, "id", None),
                "model": getattr(completion, "model", None),
                "provider": (
                    getattr(completion, "provider", None)
                    or model_extra.get("provider")
                ),
                "created": getattr(completion, "created", None),
            }
            return (
                index,
                result_record(
                    index=index,
                    task=task,
                    task_file=task_file,
                    model_id=args.model_id,
                    model_name=args.model_name,
                    api_base_url=args.api_base_url,
                    provider=args.provider,
                    reasoning_effort=args.reasoning_effort,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    run_id=args.run_id,
                    thinking=thinking,
                    response=response,
                    finish_reason=choice.finish_reason,
                    usage=getattr(completion, "usage", None),
                    upstream_response=upstream_response,
                ),
            )
        except Exception as error:
            last_error = error
            if attempt < args.max_retries:
                status_code = getattr(error, "status_code", None)
                if status_code == 429:
                    delay = min(
                        180.0,
                        10.0 * (2 ** (attempt - 1))
                        + random.uniform(0.0, 5.0),
                    )
                    print(
                        f"  [throttle] sample={index} attempt={attempt}/"
                        f"{args.max_retries}; retrying in {delay:.1f}s"
                    )
                else:
                    delay = min(
                        60.0,
                        2 ** (attempt - 1) + random.random(),
                    )
                time.sleep(delay)
    raise RuntimeError(
        f"sample {index} failed after {args.max_retries} attempts: "
        f"{last_error}"
    )


def checkpoint_path(
    root: Path,
    run_id: str,
    model_dir: str,
    level: int,
    context: str,
    index: int,
) -> Path:
    return (
        root
        / run_id.replace("/", "-")
        / model_dir
        / f"level_{level}_{context}"
        / f"{index:04d}.json"
    )


def checkpoint_matches(
    record: Any,
    task: dict[str, Any],
    task_file: Path,
    args: argparse.Namespace,
) -> bool:
    if not completed_response(record):
        return False
    original_task = record.get("original_task", {})
    config = record.get("inference_config", {})
    expected_config = {
        "model_id": args.model_id,
        "model_name": args.model_name,
        "api_base_url": args.api_base_url,
        "provider": args.provider,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "run_id": args.run_id,
    }
    return (
        original_task.get("prompt_sha256")
        == prompt_sha256(str(task["input"]))
        and original_task.get("source_file") == str(task_file.resolve())
        and all(config.get(key) == value for key, value in expected_config.items())
    )


def load_checkpoint(
    path: Path,
    task: dict[str, Any],
    task_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return value if checkpoint_matches(value, task, task_file, args) else None


def run_condition(
    level: int,
    context: str,
    args: argparse.Namespace,
) -> None:
    task_file = args.tasks_dir / f"level_{level}_{context}.json"
    tasks = load_json(task_file)
    if not isinstance(tasks, list) or len(tasks) < args.samples:
        raise ValueError(
            f"{task_file}: need at least {args.samples} tasks"
        )
    tasks = tasks[: args.samples]

    model_dir = f"{args.model_name}-{args.reasoning_effort}"
    output_file = (
        args.output_dir
        / model_dir
        / f"level_{level}_{context}_results.json"
    )
    if output_file.exists():
        existing = load_json(output_file)
        if (
            isinstance(existing, list)
            and len(existing) == args.samples
            and all(
                checkpoint_matches(record, task, task_file, args)
                for record, task in zip(existing, tasks)
            )
        ):
            print(f"[skip] complete: {output_file}")
            return

    results: list[dict[str, Any] | None] = [None] * args.samples
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, task in enumerate(tasks):
        path = checkpoint_path(
            args.checkpoint_dir,
            args.run_id,
            model_dir,
            level,
            context,
            index,
        )
        record = load_checkpoint(path, task, task_file, args)
        if record is None:
            pending.append((index, task))
        else:
            results[index] = record

    print(
        f"[run] level={level} context={context}: "
        f"complete={args.samples - len(pending)}/{args.samples}, "
        f"workers={args.workers}"
    )
    failures: list[tuple[int, str]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    infer_one,
                    index,
                    task,
                    task_file=task_file,
                    args=args,
                ): index
                for index, task in pending
            }
            completed = args.samples - len(pending)
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result_index, record = future.result()
                    path = checkpoint_path(
                        args.checkpoint_dir,
                        args.run_id,
                        model_dir,
                        level,
                        context,
                        result_index,
                    )
                    atomic_write_json(path, record)
                    results[result_index] = record
                    completed += 1
                    if (
                        completed % args.log_every == 0
                        or completed == args.samples
                    ):
                        print(
                            f"  progress level={level} context={context}: "
                            f"{completed}/{args.samples}"
                        )
                except Exception as error:
                    failures.append((index, str(error)))
                    print(f"  [warning] sample={index}: {error}")

    missing = [
        index
        for index, record in enumerate(results)
        if not completed_response(record)
    ]
    if missing:
        error_file = output_file.with_name(
            output_file.stem + "_failures.json"
        )
        atomic_write_json(
            error_file,
            {
                "missing_indices": missing,
                "errors": [
                    {"sample_index": index, "error": error}
                    for index, error in failures
                ],
            },
        )
        raise RuntimeError(
            f"level={level} context={context}: "
            f"{len(missing)} incomplete samples; rerun to retry"
        )

    atomic_write_json(output_file, results)
    empty_finals = sum(
        not bool(str(record["generated"]["response"]).strip())
        for record in results
        if record is not None
    )
    print(
        f"[complete] {output_file}: {len(results)} completions, "
        f"empty_final={empty_finals}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        "--model-path",
        dest="model_id",
        required=True,
        help="OpenRouter model ID; --model-path is a compatibility alias.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--api-base-url", default="https://openrouter.ai/api/v1"
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENROUTER_API_KEY", "")
    )
    parser.add_argument("--provider", default="google-vertex")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS
    )
    parser.add_argument("--samples", type=int, default=520)
    parser.add_argument("--levels", type=parse_csv_ints, default=[0, 1, 2, 3])
    parser.add_argument(
        "--contexts",
        type=parse_csv_strings,
        default=["0k", "16k", "64k"],
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16368)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--condition-passes",
        type=int,
        default=3,
        help="Retry incomplete cells while preserving per-sample checkpoints.",
    )
    parser.add_argument(
        "--run-id",
        default="full520-openrouter-google-vertex-v1",
        help="Bump this identifier whenever the serving/runtime protocol changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set and --api-key was not provided"
        )
    config = {
        "model_id": args.model_id,
        "model_name": args.model_name,
        "api_base_url": args.api_base_url,
        "provider": args.provider,
        "tasks_dir": str(args.tasks_dir.resolve()),
        "samples": args.samples,
        "levels": args.levels,
        "contexts": args.contexts,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "run_id": args.run_id,
    }
    model_dir = (
        args.output_dir / f"{args.model_name}-{args.reasoning_effort}"
    )
    atomic_write_json(model_dir / "inference_config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    failures: list[str] = []
    for condition_pass in range(1, args.condition_passes + 1):
        failures = []
        print(
            f"[inference-pass] {condition_pass}/{args.condition_passes}"
        )
        for level in args.levels:
            for context in args.contexts:
                try:
                    run_condition(level, context, args)
                except Exception as error:
                    message = f"level={level} context={context}: {error}"
                    failures.append(message)
                    print(f"[condition-failed] {message}")
        if not failures:
            return
    raise RuntimeError(
        f"{len(failures)} condition(s) remain incomplete after "
        f"{args.condition_passes} passes; completed cells and per-sample "
        "checkpoints were preserved:\n" + "\n".join(failures)
    )


if __name__ == "__main__":
    main()
