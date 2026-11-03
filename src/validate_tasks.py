#!/usr/bin/env python3
"""Validate and fingerprint the fixed 520-objective main-experiment tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data"
    / "source"
    / "advbench_transformed_api_gemini-2.0-flash-001_fixed.json"
)
DEFAULT_TASKS = REPO_ROOT / "data" / "tasks"
DEFAULT_MANIFEST = (
    REPO_ROOT / "outputs" / "artifacts" / "task_manifest.json"
)
LEVELS = (0, 1, 2, 3)
CONTEXTS = ("0k", "16k", "64k")
EXPECTED_CONTEXT_LENGTHS = {"0k": 0, "16k": 15700, "64k": 63700}
EXPECTED_SOURCE_SHA256 = (
    "5f0680b07b5543711370f30cf7d2edb2003c451a2ba4452dda58814c911732a6"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def original_query(item: dict[str, Any]) -> str:
    return str(
        item.get("original_query")
        or item.get("original_harmful_query")
        or ""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples", type=int, default=520)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_sha256 = sha256_file(args.source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"{args.source}: expected sha256={EXPECTED_SOURCE_SHA256}, "
            f"found {source_sha256}"
        )
    source = load_json(args.source)
    if not isinstance(source, list) or len(source) != args.samples:
        raise ValueError(
            f"source must contain exactly {args.samples} records; "
            f"found {len(source) if isinstance(source, list) else type(source)}"
        )

    queries = [original_query(item) for item in source]
    if any(not query for query in queries):
        raise ValueError("source contains an empty original query")
    if len(set(queries)) != args.samples:
        raise ValueError("source original queries are not unique")

    query_order_sha256 = hashlib.sha256(
        "\n".join(queries).encode("utf-8")
    ).hexdigest()
    files: list[dict[str, Any]] = []

    for level in LEVELS:
        level_key = f"level_{level}"
        expected_facts = level + 1
        for index, item in enumerate(source):
            decomposition = item.get(level_key)
            if not isinstance(decomposition, dict):
                raise ValueError(f"source index {index} missing {level_key}")
            facts = decomposition.get("facts")
            if not isinstance(facts, list) or len(facts) != expected_facts:
                raise ValueError(
                    f"source index {index} {level_key}: expected "
                    f"{expected_facts} facts"
                )

        for context in CONTEXTS:
            path = args.tasks_dir / f"level_{level}_{context}.json"
            tasks = load_json(path)
            if not isinstance(tasks, list) or len(tasks) != args.samples:
                raise ValueError(
                    f"{path}: expected {args.samples} tasks, found "
                    f"{len(tasks) if isinstance(tasks, list) else type(tasks)}"
                )

            for index, task in enumerate(tasks):
                if not isinstance(task, dict):
                    raise TypeError(f"{path}: task {index} is not an object")
                if original_query(task) != queries[index]:
                    raise ValueError(
                        f"{path}: original-query mismatch at index {index}"
                    )
                if task.get("level") != level:
                    raise ValueError(f"{path}: level mismatch at index {index}")
                if task.get("n_facts") != expected_facts:
                    raise ValueError(
                        f"{path}: n_facts mismatch at index {index}"
                    )
                if (
                    task.get("context_length")
                    != EXPECTED_CONTEXT_LENGTHS[context]
                ):
                    raise ValueError(
                        f"{path}: context_length mismatch at index {index}"
                    )
                prompt = str(task.get("input", ""))
                if not prompt.strip():
                    raise ValueError(f"{path}: empty input at index {index}")
                facts = source[index][level_key]["facts"]
                for fact in facts:
                    if str(fact) not in prompt:
                        raise ValueError(
                            f"{path}: missing fact at index {index}"
                        )

            files.append(
                {
                    "level": level,
                    "context": context,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "records": len(tasks),
                    "sha256": sha256_file(path),
                }
            )
            print(f"[ok] level={level} context={context}: {len(tasks)}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": args.samples,
        "levels": list(LEVELS),
        "contexts": list(CONTEXTS),
        "conditions": len(LEVELS) * len(CONTEXTS),
        "responses_per_model": (
            args.samples * len(LEVELS) * len(CONTEXTS)
        ),
        "source": {
            "path": str(args.source.resolve()),
            "bytes": args.source.stat().st_size,
            "sha256": source_sha256,
        },
        "query_order_sha256": query_order_sha256,
        "task_files": files,
    }
    atomic_write_json(args.manifest, manifest)
    print(f"[complete] manifest: {args.manifest}")
    print(f"[complete] query-order sha256: {query_order_sha256}")


if __name__ == "__main__":
    main()
