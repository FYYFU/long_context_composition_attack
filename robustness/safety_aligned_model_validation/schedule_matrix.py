#!/usr/bin/env python3
"""Keep the selected GPUs occupied while running the model matrix."""

from __future__ import annotations
from pathlib import Path
import sys

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Python executable containing vLLM",
    )
    parser.add_argument(
        "--aligned-only",
        action="store_true",
        help="Skip the two original backbone checkpoints",
    )
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--retry-truncated", action="store_true")
    parser.add_argument(
        "--model-aliases",
        default="",
        help="Optional comma-separated aliases to run.",
    )
    return parser.parse_args()


def load_jobs(aligned_only: bool) -> List[Dict[str, str]]:
    with (ROOT / "models.json").open("r", encoding="utf-8") as handle:
        models = json.load(handle)["models"]
    if aligned_only:
        models = [model for model in models if model["alignment"] != "original"]
    return models


def has_truncated_rows(alias: str) -> bool:
    result_dir = ROOT / "results" / f"{alias}-default"
    for checkpoint_path in result_dir.glob("level_*.jsonl"):
        latest: Dict[int, Dict[str, object]] = {}
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[int(row["sample_index"])] = row
        if any(
            row.get("generated", {}).get("finish_reason") == "length"
            for row in latest.values()
        ):
            return True
    return False


def generation_is_complete(alias: str, n_examples: int) -> bool:
    result_dir = ROOT / "results" / f"{alias}-default"
    for level in (0, 1, 2, 3):
        for context in ("0k", "16k"):
            result_path = result_dir / f"level_{level}_{context}_results.json"
            if not result_path.exists():
                return False
            with result_path.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            if not isinstance(rows, list) or len(rows) != n_examples:
                return False
    return True


def main() -> None:
    args = parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    jobs = load_jobs(args.aligned_only)
    selected_aliases = {
        value.strip()
        for value in args.model_aliases.split(",")
        if value.strip()
    }
    if selected_aliases:
        jobs = [job for job in jobs if job["alias"] in selected_aliases]
    if args.retry_truncated:
        jobs = [job for job in jobs if has_truncated_rows(job["alias"])]
        print(
            "Models with truncated rows: "
            + (", ".join(job["alias"] for job in jobs) if jobs else "none"),
            flush=True,
        )
    else:
        jobs = [
            job
            for job in jobs
            if not generation_is_complete(job["alias"], args.n_examples)
        ]
        print(
            "Models requiring generation: "
            + (", ".join(job["alias"] for job in jobs) if jobs else "none"),
            flush=True,
        )
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    active: Dict[str, Dict[str, object]] = {}
    failures: List[str] = []

    def launch(gpu: str, job: Dict[str, str]) -> None:
        log_path = logs_dir / f"{job['alias']}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["HF_HOME"] = env.get(
            "SAFETY_HF_HOME",
            str(Path.home() / '.cache/huggingface'),
        )
        env["TRANSFORMERS_CACHE"] = env.get(
            "SAFETY_TRANSFORMERS_CACHE",
            str(Path.home() / '.cache/huggingface/hub'),
        )
        env["VLLM_CACHE_ROOT"] = env.get(
            "SAFETY_VLLM_CACHE_ROOT",
            str(Path.home() / '.cache/vllm'),
        )
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        command = [
            str(args.python),
            str(ROOT / "run_inference.py"),
            "--model",
            job["model_id"],
            "--alias",
            job["alias"],
            "--n-examples",
            str(args.n_examples),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.retry_truncated:
            command.append("--retry-truncated")
        print(
            f"Launching {job['alias']} on physical GPU {gpu}; log={log_path}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        active[gpu] = {
            "process": process,
            "job": job,
            "log_handle": log_handle,
            "log_path": log_path,
        }

    while jobs or active:
        while jobs and len(active) < len(gpus):
            free_gpu = next(gpu for gpu in gpus if gpu not in active)
            launch(free_gpu, jobs.pop(0))

        time.sleep(args.poll_seconds)
        for gpu, state in list(active.items()):
            process = state["process"]
            return_code = process.poll()
            if return_code is None:
                continue
            state["log_handle"].close()
            alias = state["job"]["alias"]
            if return_code == 0:
                print(f"Completed {alias} on GPU {gpu}", flush=True)
            else:
                failures.append(alias)
                print(
                    f"FAILED {alias} on GPU {gpu} with code {return_code}; "
                    f"see {state['log_path']}",
                    flush=True,
                )
            del active[gpu]

    if failures:
        print(f"Failed jobs: {', '.join(failures)}", file=sys.stderr)
        raise SystemExit(1)
    print("All model jobs completed.", flush=True)


if __name__ == "__main__":
    main()
