"""
Step 1: Download the first 100 standard HarmBench behaviors.

Usage:
    python download_harmbench.py
    python download_harmbench.py --max_samples 100
"""
import json
import os
from fire import Fire


def download_harmbench(
    max_samples: int = 100,
    output_file: str = "harmbench_standard_100.json",
):
    """
    Download standard HarmBench behaviors from HuggingFace.

    Args:
        max_samples: Select the first N standard behaviors.
        output_file: Output file path.
    """
    from datasets import load_dataset

    print("正在从 HuggingFace 下载 HarmBench 数据集...")

    # Option 1: Use walledai/HarmBench (standard subset).
    try:
        ds = load_dataset("walledai/HarmBench", "standard", trust_remote_code=True)
        # Use the test split or the first available split.
        if "test" in ds:
            data = ds["test"]
        elif "train" in ds:
            data = ds["train"]
        else:
            data = ds[list(ds.keys())[0]]

        behaviors = []
        for item in data:
            behaviors.append({
                "behavior": item.get("prompt", item.get("Behavior", "")),
                "semantic_category": item.get("SemanticCategory", item.get("semantic_category", "")),
                "behavior_id": item.get("BehaviorID", item.get("behavior_id", "")),
            })
        print(f"��� walledai/HarmBench 加载了 {len(behaviors)} 条 standard behaviors")

    except Exception as e:
        print(f"walledai/HarmBench 加载失败: {e}")
        print("尝试备选方案...")

        # Option 2: Use mariagrandury/harmbench.
        try:
            ds = load_dataset("mariagrandury/harmbench", "DirectRequest", split="test")
            # Filter for standard behaviors.
            standard_ds = ds.filter(lambda x: x.get("FunctionalCategory", "") == "standard")

            behaviors = []
            for item in standard_ds:
                behaviors.append({
                    "behavior": item.get("Behavior", ""),
                    "semantic_category": item.get("SemanticCategory", ""),
                    "behavior_id": item.get("BehaviorID", ""),
                })
            print(f"从 mariagrandury/harmbench 加载了 {len(behaviors)} 条 standard behaviors")

        except Exception as e2:
            print(f"mariagrandury/harmbench 也加载失败: {e2}")
            print("尝试直接从 GitHub CSV 下载...")

            # Option 3: Download the CSV directly from GitHub.
            import pandas as pd
            url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
            df = pd.read_csv(url)
            standard_df = df[df["FunctionalCategory"] == "standard"]

            behaviors = []
            for _, row in standard_df.iterrows():
                behaviors.append({
                    "behavior": row.get("Behavior", ""),
                    "semantic_category": row.get("SemanticCategory", ""),
                    "behavior_id": row.get("BehaviorID", ""),
                })
            print(f"从 GitHub CSV 加载了 {len(behaviors)} 条 standard behaviors")

    # Select the first N examples.
    behaviors = behaviors[:max_samples]
    print(f"���用前 {len(behaviors)} 条")

    # Print the category distribution.
    categories = {}
    for b in behaviors:
        cat = b["semantic_category"]
        categories[cat] = categories.get(cat, 0) + 1
    print(f"类别分布: {categories}")

    # Save.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(behaviors, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 已保存 {len(behaviors)} 条 HarmBench behaviors 到: {output_path}")


if __name__ == "__main__":
    Fire(download_harmbench)
