"""
Step 2: Create task files from decomposed XSTest prompts.

Embeds decomposed facts into PG19 haystack at different context lengths.
Converts XSTest format to benign baseline format, then reuses create_benign_tasks.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from fire import Fire

# Reuse task creation from benign baseline
project_root = Path(__file__).resolve().parent.parent.parent.parent
benign_dir = project_root / "controls" / "benign_baseline"
sys.path.insert(0, str(benign_dir))
from create_task_benign import create_benign_tasks


def convert_xstest_to_benign_format(xstest_data):
    """Convert XSTest decomposed format to benign baseline format."""
    converted = []
    for item in xstest_data:
        converted.append({
            "id": item["id"],
            "original_harmful_query": "",  # XSTest safe prompts have no harmful counterpart
            "benign_query": item["original_query"],  # The XSTest prompt IS the benign query
            "domain": item.get("xstest_type", ""),
            "rationale": item.get("note", ""),
            "level_0": item.get("level_0"),
            "level_1": item.get("level_1"),
            "level_2": item.get("level_2"),
            "level_3": item.get("level_3"),
        })
    return converted


def create_xstest_tasks(
    dataset_path: str = None,
    level: int = 1,
    out_folder: str = None,
    number_of_samples: int = 250,
    context_lengths: str = "0,16000,64000",
):
    """
    Create task files from decomposed XSTest data.

    Args:
        dataset_path: Path to xstest_decomposed.json
        level: Reasoning level (0-3)
        out_folder: Output folder for task files
        number_of_samples: Max samples to use
        context_lengths: Comma-separated context lengths in tokens
    """
    script_dir = Path(__file__).resolve().parent
    if dataset_path is None:
        dataset_path = str(script_dir / "xstest_decomposed.json")
    if out_folder is None:
        out_folder = str(script_dir / "generated_tasks")

    # Parse context lengths
    if isinstance(context_lengths, str):
        msg_lengths = [int(x.strip()) for x in context_lengths.split(",")]
    else:
        msg_lengths = [int(x) for x in context_lengths]

    print(f"Creating tasks for Level {level}")
    print(f"Dataset: {dataset_path}")
    print(f"Context lengths: {msg_lengths}")

    # Load and convert format
    with open(dataset_path, "r", encoding="utf-8") as f:
        xstest_data = json.load(f)

    converted = convert_xstest_to_benign_format(xstest_data)

    # Write to temp file in benign format
    tmp_path = os.path.join(out_folder, "_xstest_converted_tmp.json")
    os.makedirs(out_folder, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    try:
        create_benign_tasks(
            dataset_path=tmp_path,
            level=level,
            out_folder=out_folder,
            number_of_samples=number_of_samples,
            message_lengths=msg_lengths,
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    Fire(create_xstest_tasks)
