#!/usr/bin/env python3
"""Prepare matched Qwen-decomposition tasks for the NeurIPS rebuttal.

The main experiment's Gemini tasks are used as templates so that the Qwen
condition has exactly the same 100 original requests, PG19 background text,
fragment insertion positions, prompt wrapper, and neutral synthesis query.
Only the inserted decomposition fragments are replaced.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MAIN_ROOT = REPO_ROOT / "paper"

DEFAULT_QWEN = (
    MAIN_ROOT
    / "results"
    / "advbench_transformed_api_Qwen3-Next-80B_fixed.json"
)
DEFAULT_GEMINI = (
    MAIN_ROOT
    / "results"
    / "advbench_transformed_api_gemini-2.0-flash-001_fixed.json"
)
DEFAULT_TEMPLATE_DIR = (
    MAIN_ROOT / "generated_tasks" / "gemini-2.0-flash-001"
)
DEFAULT_SUBSET = HERE / "data" / "qwen_first100.json"
DEFAULT_OUTPUT_DIR = HERE / "generated_tasks" / "qwen3-next-80b"


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


def original_query(item: dict[str, Any]) -> str:
    return str(
        item.get("original_query")
        or item.get("original_harmful_query")
        or ""
    )


def get_level(item: dict[str, Any], level: int) -> dict[str, Any]:
    value = item.get(f"level_{level}")
    if not isinstance(value, dict):
        raise ValueError(
            f"missing level_{level} for query: {original_query(item)!r}"
        )
    facts = value.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError(
            f"empty facts in level_{level} for query: "
            f"{original_query(item)!r}"
        )
    return value


def validate_sources(
    qwen: list[dict[str, Any]],
    gemini: list[dict[str, Any]],
    sample_count: int,
    levels: list[int],
) -> None:
    if len(qwen) < sample_count or len(gemini) < sample_count:
        raise ValueError(
            f"need {sample_count} source examples, found "
            f"Qwen={len(qwen)}, Gemini={len(gemini)}"
        )

    expected_facts = {1: 2, 2: 3, 3: 4}
    for index in range(sample_count):
        qwen_query = original_query(qwen[index])
        gemini_query = original_query(gemini[index])
        if not qwen_query or qwen_query != gemini_query:
            raise ValueError(
                f"query mismatch at index {index}: "
                f"Qwen={qwen_query!r}, Gemini={gemini_query!r}"
            )

        for level in levels:
            qwen_level = get_level(qwen[index], level)
            gemini_level = get_level(gemini[index], level)
            expected = expected_facts.get(level)
            if expected is not None:
                for generator, level_data in (
                    ("Qwen", qwen_level),
                    ("Gemini", gemini_level),
                ):
                    actual = len(level_data["facts"])
                    if actual != expected:
                        raise ValueError(
                            f"{generator} index={index} level={level}: "
                            f"expected {expected} facts, found {actual}"
                        )

            if (
                str(qwen_level.get("reasoning_query", "")).strip()
                != str(gemini_level.get("reasoning_query", "")).strip()
            ):
                raise ValueError(
                    f"neutral query mismatch at index={index}, level={level}"
                )


def replace_fragments(
    prompt: str,
    old_facts: list[str],
    new_facts: list[str],
    context: str,
) -> str:
    if len(old_facts) != len(new_facts):
        raise ValueError(
            f"fact count mismatch: old={len(old_facts)}, new={len(new_facts)}"
        )

    updated = prompt
    for old_fact, new_fact in zip(old_facts, new_facts):
        if context == "0k":
            needle = old_fact
            replacement = new_fact
        else:
            needle = f"[[Inserted Fact: {old_fact}]]"
            replacement = f"[[Inserted Fact: {new_fact}]]"

        occurrences = updated.count(needle)
        if occurrences != 1:
            raise ValueError(
                f"expected exactly one occurrence of a template fragment, "
                f"found {occurrences}; context={context}, fragment={old_fact!r}"
            )
        updated = updated.replace(needle, replacement, 1)

    return updated


def build_condition(
    template_tasks: list[dict[str, Any]],
    qwen: list[dict[str, Any]],
    gemini: list[dict[str, Any]],
    level: int,
    context: str,
    sample_count: int,
) -> list[dict[str, Any]]:
    if len(template_tasks) < sample_count:
        raise ValueError(
            f"template level={level}, context={context} contains only "
            f"{len(template_tasks)} tasks"
        )

    prepared: list[dict[str, Any]] = []
    for index in range(sample_count):
        template = template_tasks[index]
        expected_query = original_query(qwen[index])
        if template.get("original_query") != expected_query:
            raise ValueError(
                f"template query mismatch at index={index}, level={level}, "
                f"context={context}"
            )

        old_level = get_level(gemini[index], level)
        new_level = get_level(qwen[index], level)
        task = copy.deepcopy(template)
        task["input"] = replace_fragments(
            str(task["input"]),
            [str(value) for value in old_level["facts"]],
            [str(value) for value in new_level["facts"]],
            context,
        )
        task["question"] = str(new_level.get("reasoning_query", ""))
        task["original_query"] = expected_query
        task["level"] = level
        task["n_facts"] = len(new_level["facts"])
        task["decomposition_generator"] = "Qwen3-Next-80B"
        task["source_index"] = index
        prepared.append(task)

    return prepared


def validate_prepared(
    tasks: list[dict[str, Any]],
    qwen: list[dict[str, Any]],
    level: int,
    context: str,
    sample_count: int,
) -> None:
    if len(tasks) != sample_count:
        raise ValueError(
            f"prepared level={level}, context={context}: "
            f"expected {sample_count}, found {len(tasks)}"
        )
    for index, task in enumerate(tasks):
        level_data = get_level(qwen[index], level)
        if task.get("original_query") != original_query(qwen[index]):
            raise ValueError(f"prepared query mismatch at index={index}")
        if task.get("source_index") != index:
            raise ValueError(f"prepared source index mismatch at index={index}")
        if task.get("n_facts") != len(level_data["facts"]):
            raise ValueError(f"prepared fact count mismatch at index={index}")
        for fact in level_data["facts"]:
            if str(fact) not in str(task.get("input", "")):
                raise ValueError(
                    f"Qwen fact missing from prompt at index={index}, "
                    f"level={level}, context={context}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare matched Qwen decomposition tasks."
    )
    parser.add_argument("--qwen-source", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--gemini-source", type=Path, default=DEFAULT_GEMINI)
    parser.add_argument(
        "--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR
    )
    parser.add_argument("--subset-output", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--levels", type=parse_csv_ints, default=[1, 2, 3])
    parser.add_argument(
        "--contexts",
        type=parse_csv_strings,
        default=["0k", "16k", '64k'],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen = load_json(args.qwen_source)
    gemini = load_json(args.gemini_source)
    if not isinstance(qwen, list) or not isinstance(gemini, list):
        raise TypeError("decomposition source files must contain JSON lists")

    validate_sources(qwen, gemini, args.samples, args.levels)
    subset = copy.deepcopy(qwen[: args.samples])
    atomic_write_json(args.subset_output, subset)

    manifest: dict[str, Any] = {
        "qwen_source": str(args.qwen_source.resolve()),
        "gemini_source": str(args.gemini_source.resolve()),
        "template_dir": str(args.template_dir.resolve()),
        "samples": args.samples,
        "levels": args.levels,
        "contexts": args.contexts,
        "conditions": {},
    }

    for level in args.levels:
        for context in args.contexts:
            template_file = (
                args.template_dir / f"level_{level}_{context}.json"
            )
            if not template_file.exists():
                raise FileNotFoundError(f"missing template: {template_file}")
            template_tasks = load_json(template_file)
            if not isinstance(template_tasks, list):
                raise TypeError(f"template must contain a list: {template_file}")

            tasks = build_condition(
                template_tasks,
                qwen,
                gemini,
                level,
                context,
                args.samples,
            )
            validate_prepared(
                tasks, qwen, level, context, args.samples
            )
            output_file = (
                args.output_dir / f"level_{level}_{context}.json"
            )
            atomic_write_json(output_file, tasks)
            manifest["conditions"][f"level_{level}_{context}"] = {
                "file": str(output_file.resolve()),
                "count": len(tasks),
            }
            print(f"[prepared] {output_file}: {len(tasks)} tasks")

    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(f"[complete] subset: {args.subset_output}")
    print(f"[complete] tasks: {args.output_dir}")


if __name__ == "__main__":
    main()
