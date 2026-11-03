#!/usr/bin/env python3
"""Run one configured model on the matched NIPS benign-reconstruction tasks."""

import argparse
import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ICML_BASELINE = REPO_ROOT / "controls" / "benign_baseline"
DEFAULT_TASKS = ICML_BASELINE / "generated_tasks"
DEFAULT_OUTPUT = HERE / "results"
SHARED_RUNNER = ICML_BASELINE / "run_benign.py"
LEVELS = (0, 1, 2, 3)
CONTEXT_LENGTHS = ("0k", "16k", "64k")

# Generation settings match the corresponding harmful experiment. Concurrency
# is intentionally conservative for non-GPT-oss/non-Gemini endpoints.
MODELS = {
    "qwen3-next-80b-thinking": {
        "model_path": "qwen/qwen3-next-80b-a3b-thinking",
        "api_provider": "openrouter",
        "reasoning_effort": "default",
        "temperature": 0.6,
        "top_p": 0.95,
        "batch_size": 2,
        "num_workers": 2,
    },
    "gemini-3-flash": {
        "model_path": "google/gemini-3-flash-preview",
        "api_provider": "openrouter",
        "reasoning_effort": "default",
        "temperature": 0.0,
        "top_p": 1.0,
        "batch_size": 20,
        "num_workers": 8,
    },
    "gemini-3-pro": {
        "model_path": "google/gemini-3-pro-preview",
        "api_provider": "openrouter",
        "reasoning_effort": "default",
        "temperature": 0.0,
        "top_p": 1.0,
        "batch_size": 20,
        "num_workers": 8,
    },
}


def load_shared_runner():
    spec = importlib.util.spec_from_file_location("icml_benign_runner", SHARED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared runner: {SHARED_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_name(config):
    base = config["model_path"].split("/")[-1]
    effort = config["reasoning_effort"]
    return base if effort is None else f"{base}-{effort}"


def result_is_complete(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, list) or len(data) != expected:
        return False
    return all(
        str(item.get("generated", {}).get("response", "")).strip()
        or str(item.get("generated", {}).get("thinking", "")).strip()
        for item in data
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--tasks-folder", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16368)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = MODELS[args.model]
    required_key = "OPENROUTER_API_KEY"
    if not os.environ.get(required_key):
        raise SystemExit(f"{required_key} is not set.")

    batch_size = args.batch_size or config["batch_size"]
    num_workers = args.num_workers or config["num_workers"]
    runner = load_shared_runner()
    model_dir = args.output_folder / output_name(config)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Model={config['model_path']} provider={config['api_provider']} "
        f"batch_size={batch_size} num_workers={num_workers}"
    )

    for level in LEVELS:
        for context_length in CONTEXT_LENGTHS:
            task_file = args.tasks_folder / f"level_{level}_{context_length}.json"
            output_file = model_dir / f"level_{level}_{context_length}_results.json"
            if not task_file.exists():
                raise FileNotFoundError(f"Missing task file: {task_file}")

            task_count = len(json.loads(task_file.read_text(encoding="utf-8")))
            expected = min(task_count, args.max_examples) if args.max_examples else task_count
            if not args.overwrite and result_is_complete(output_file, expected):
                print(f"[skip] Complete: {output_file}")
                continue

            print(f"[run] level={level}, context={context_length}, n={expected}")
            runner.process_tasks_with_model(
                task_file=str(task_file),
                model_path=config["model_path"],
                output_file=str(output_file),
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=args.max_tokens,
                batch_size=batch_size,
                api_provider=config["api_provider"],
                num_workers=num_workers,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_examples=args.max_examples,
                reasoning_effort=config["reasoning_effort"],
            )
            if not result_is_complete(output_file, expected):
                raise RuntimeError(
                    f"Incomplete or empty responses in {output_file}. "
                    "Rerun the same model to retry this condition."
                )

    print(f"All inference results are complete: {model_dir}")


if __name__ == "__main__":
    main()
