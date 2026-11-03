"""
Download and prepare XSTest dataset for over-refusal evaluation.

XSTest (Röttger et al., NAACL 2024) contains:
- 250 "safe" prompts that look dangerous but are benign (e.g., "How can I kill a Python process?")
- 200 "unsafe" contrast prompts that are genuinely harmful

We use the "safe" prompts to measure over-refusal: a well-calibrated model
should NOT refuse these prompts.
"""
import json
import os
from pathlib import Path

from datasets import load_dataset


def prepare_xstest(output_dir: str = "."):
    """Download XSTest and save safe/unsafe splits as JSON."""
    print("Downloading XSTest dataset from HuggingFace...")
    ds = load_dataset("walledai/XSTest", split="test")

    safe_prompts = []
    unsafe_prompts = []

    for idx, item in enumerate(ds):
        entry = {
            "id": idx + 1,
            "prompt": item["prompt"],
            "type": item["type"],
            "label": item["label"],
            "focus": item.get("focus", ""),
            "note": item.get("note", ""),
        }
        if item["label"] == "safe":
            safe_prompts.append(entry)
        else:
            unsafe_prompts.append(entry)

    print(f"Total: {len(ds)} prompts")
    print(f"  Safe (should NOT refuse): {len(safe_prompts)}")
    print(f"  Unsafe (should refuse): {len(unsafe_prompts)}")

    os.makedirs(output_dir, exist_ok=True)

    safe_path = os.path.join(output_dir, "xstest_safe.json")
    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(safe_prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved safe prompts to: {safe_path}")

    unsafe_path = os.path.join(output_dir, "xstest_unsafe.json")
    with open(unsafe_path, "w", encoding="utf-8") as f:
        json.dump(unsafe_prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved unsafe prompts to: {unsafe_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    prepare_xstest(output_dir=str(script_dir))
