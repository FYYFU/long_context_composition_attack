#!/usr/bin/env python3
"""Run the NIPS benign-reconstruction control on GPT-oss-120b.

This reuses the exact task files and inference implementation from the earlier
ICML experiment, while keeping all new outputs under robustness.
"""

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
MODEL_PATH = "openai/gpt-oss-120b"
MODEL_OUTPUT_NAME = "gpt-oss-120b-medium"


def load_shared_runner():
    spec = importlib.util.spec_from_file_location("icml_benign_runner", SHARED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared runner: {SHARED_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    parser.add_argument("--tasks-folder", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16368)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rerun conditions even when a complete result file already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    runner = load_shared_runner()
    model_dir = args.output_folder / MODEL_OUTPUT_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

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
                model_path=MODEL_PATH,
                output_file=str(output_file),
                temperature=0.0,
                top_p=1.0,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                api_provider="openrouter",
                num_workers=args.num_workers,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_examples=args.max_examples,
                reasoning_effort="medium",
            )

            if not result_is_complete(output_file, expected):
                raise RuntimeError(
                    f"Incomplete or empty responses in {output_file}. "
                    "Rerun the script to retry this condition."
                )

    print(f"All inference results are complete: {model_dir}")


if __name__ == "__main__":
    main()
