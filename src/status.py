#!/usr/bin/env python3
"""Show compact process, inference, and judging progress."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUTPUTS = REPO_ROOT / "outputs"
MODELS = ("gpt-oss-20b", "gpt-oss-120b")
LEVELS = (0, 1, 2, 3)
CONTEXTS = ("0k", "16k", "64k")
SAMPLES = 520


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pid_status(path: Path) -> str:
    if not path.exists():
        return "not recorded"
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return f"running (pid={pid})"
    except (ValueError, ProcessLookupError):
        return "not running"
    except PermissionError:
        return f"running (pid={pid}, owned by another user)"


def inference_record_matches(
    record: Any, expected_config: dict[str, Any]
) -> bool:
    if not isinstance(record, dict):
        return False
    config = record.get("inference_config", {})
    return (
        record.get("generated", {}).get("completion_received") is True
        and isinstance(record.get("generated", {}).get("response"), str)
        and all(
            config.get(key) == value
            for key, value in expected_config.items()
        )
    )


def final_inference_count(
    path: Path, expected_config: dict[str, Any]
) -> int | None:
    if not path.exists():
        return None
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    count = sum(
        inference_record_matches(record, expected_config)
        for record in value
    )
    return count if count == len(value) else None


def final_judge_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    records = value.get("results", []) if isinstance(value, dict) else []
    return sum(
        isinstance(record, dict)
        and isinstance(record.get("score"), int)
        and record["score"] in (-1, 1, 2, 3, 4, 5)
        for record in records
    )


def checkpoint_count(path: Path) -> int:
    return sum(1 for _ in path.glob("*.json")) if path.exists() else 0


def inference_checkpoint_count(
    path: Path, expected_config: dict[str, Any]
) -> int:
    if not path.exists():
        return 0
    count = 0
    for checkpoint in path.glob("*.json"):
        try:
            record = load_json(checkpoint)
        except (OSError, json.JSONDecodeError):
            continue
        count += inference_record_matches(record, expected_config)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS)
    parser.add_argument(
        "--inference-api-base-url",
        default=os.environ.get(
            "API_BASE_URL", "https://openrouter.ai/api/v1"
        ),
    )
    parser.add_argument(
        "--inference-provider",
        default=os.environ.get("INFERENCE_PROVIDER", "google-vertex"),
    )
    parser.add_argument(
        "--model-20b",
        default=os.environ.get("MODEL_20B", "openai/gpt-oss-20b"),
    )
    parser.add_argument(
        "--model-120b",
        default=os.environ.get("MODEL_120B", "openai/gpt-oss-120b"),
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get(
            "RUN_ID", "full520-openrouter-google-vertex-v1"
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get(
            "JUDGE_MODEL", "google/gemini-2.0-flash-lite-001"
        ),
    )
    parser.add_argument(
        "--judge-provider",
        default=os.environ.get("JUDGE_PROVIDER", "google-vertex"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = args.outputs_dir
    artifacts = outputs / "artifacts"
    print(f"pipeline: {pid_status(artifacts / 'pipeline.pid')}")
    print()

    judge_slug = args.judge_model.replace("/", "-")
    judge_key = (
        judge_slug + "__" + args.judge_provider.replace("/", "-")
    )
    model_ids = {
        "gpt-oss-20b": args.model_20b,
        "gpt-oss-120b": args.model_120b,
    }
    for model in MODELS:
        model_dir = f"{model}-medium"
        expected_inference_config = {
            "model_id": model_ids[model],
            "model_name": model,
            "api_base_url": args.inference_api_base_url,
            "provider": args.inference_provider,
            "run_id": args.run_id,
        }
        print(model_dir)
        inference_total = 0
        judge_total = 0
        for level in LEVELS:
            cells = []
            for context in CONTEXTS:
                inference_file = (
                    outputs
                    / "results"
                    / model_dir
                    / f"level_{level}_{context}_results.json"
                )
                inference_final = final_inference_count(
                    inference_file, expected_inference_config
                )
                if inference_final is None:
                    inference_done = inference_checkpoint_count(
                        outputs
                        / "checkpoints"
                        / "inference"
                        / args.run_id.replace("/", "-")
                        / model_dir
                        / f"level_{level}_{context}",
                        expected_inference_config,
                    )
                else:
                    inference_done = inference_final
                inference_total += inference_done

                judge_file = (
                    outputs
                    / "evaluations"
                    / model_dir
                    / (
                        f"eval_level_{level}_{context}_results_"
                        f"{judge_slug}.json"
                    )
                )
                judge_final = final_judge_count(judge_file)
                if judge_final is None:
                    judge_done = checkpoint_count(
                        outputs
                        / "checkpoints"
                        / "judge"
                        / judge_key
                        / model_dir
                        / f"level_{level}_{context}"
                    )
                else:
                    judge_done = judge_final
                judge_total += judge_done
                cells.append(
                    f"{context}: infer {inference_done}/{SAMPLES}, "
                    f"judge {judge_done}/{SAMPLES}"
                )
            print(f"  L{level} | " + " | ".join(cells))
        expected = len(LEVELS) * len(CONTEXTS) * SAMPLES
        print(
            f"  total | infer {inference_total}/{expected}, "
            f"judge {judge_total}/{expected}"
        )
        print()


if __name__ == "__main__":
    main()
