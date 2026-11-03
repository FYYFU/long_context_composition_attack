#!/usr/bin/env python3
"""Schedule the six XSTest over-refusal generation jobs across selected GPUs."""

from __future__ import annotations
from pathlib import Path
import sys

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DEFAULT_PYTHON = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=15)
    return parser.parse_args()


def load_jobs() -> List[Dict[str, str]]:
    payload = json.loads((PARENT / "models.json").read_text(encoding="utf-8"))
    return payload["models"]


def is_complete(alias: str) -> bool:
    path = ROOT / "results" / alias / "xstest_results.json"
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(rows, list) and len(rows) == 250
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    jobs = [job for job in load_jobs() if not is_complete(job["alias"])]
    print(
        "XSTest jobs: "
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
        env["HF_HOME"] = str(Path.home() / '.cache/huggingface')
        env["TRANSFORMERS_CACHE"] = str(Path.home() / '.cache/huggingface/hub')
        env["VLLM_CACHE_ROOT"] = (
            str(Path.home() / '.cache/vllm')
        )
        command = [
            str(args.python),
            str(ROOT / "run_xstest_local.py"),
            "--model",
            job["model_id"],
            "--alias",
            job["alias"],
        ]
        print(
            f"Launching {job['alias']} on physical GPU {gpu}",
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
                    f"FAILED {alias} on GPU {gpu}; see {state['log_path']}",
                    flush=True,
                )
            del active[gpu]

    if failures:
        raise SystemExit("Failed jobs: " + ", ".join(failures))
    print("All XSTest generation jobs completed.", flush=True)


if __name__ == "__main__":
    main()
