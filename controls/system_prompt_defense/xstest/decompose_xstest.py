"""
Step 1: Decompose XSTest safe prompts into Level 0-3 facts.

Reuses the same decomposition prompt templates as the benign baseline pipeline.
Input: xstest_safe.json (from prepare_xstest.py)
Output: xstest_decomposed.json (same format as benign_baseline_decomposed.json)
"""
import json
import os
import sys
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from fire import Fire

# Reuse decomposition logic from benign baseline
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root / "paper"))
sys.path.insert(0, str(project_root / "controls"))
from create_benign_baseline import (
    create_transformation_prompt,
    parse_generated_content,
    API_PROVIDER_CONFIG,
)
from utils.gpt_batch import GPTBatcher


def decompose_xstest(
    input_file: str = None,
    output_file: str = None,
    model: str = "google/gemini-2.0-flash-001",
    api_provider: str = "openrouter",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 131072,
    max_samples: Optional[int] = None,
    batch_size: int = 50,
    num_workers: int = 64,
    timeout: int = 120,
    max_retries: int = 3,
):
    """
    Decompose XSTest safe prompts into Level 0-3 facts.

    Args:
        input_file: Path to xstest_safe.json
        output_file: Output path for decomposed data
        model: Model for decomposition (Gemini recommended)
        api_provider: API provider
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max generation tokens
        max_samples: Limit number of samples (None = all)
        batch_size: Batch size for API calls
        num_workers: Parallel workers
        timeout: Timeout per request
        max_retries: Max retry attempts
    """
    script_dir = Path(__file__).resolve().parent
    if input_file is None:
        input_file = str(script_dir / "xstest_safe.json")
    if output_file is None:
        output_file = str(script_dir / "xstest_decomposed.json")

    # Load XSTest safe prompts
    with open(input_file, "r", encoding="utf-8") as f:
        xstest_data = json.load(f)

    if max_samples is not None:
        xstest_data = xstest_data[:max_samples]

    print(f"Loaded {len(xstest_data)} XSTest safe prompts")
    print(f"Model: {model}, API: {api_provider}")

    # Get API key
    config = API_PROVIDER_CONFIG[api_provider]
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise ValueError(f"{config['env_key']} not set")

    # Initialize results
    results = []
    for item in xstest_data:
        results.append({
            "id": item["id"],
            "original_query": item["prompt"],  # XSTest prompt = the query
            "xstest_type": item["type"],
            "xstest_label": item["label"],
            "focus": item.get("focus", ""),
            "note": item.get("note", ""),
            "level_0": None,
            "level_1": None,
            "level_2": None,
            "level_3": None,
        })

    # Setup batcher
    batcher_kwargs = {
        "api_key": api_key,
        "model_name": model,
        "system_prompt": None,
        "temperature": temperature,
        "num_workers": num_workers,
        "timeout_duration": timeout,
        "retry_attempts": max_retries,
        "api_base_url": config["base_url"],
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    if api_provider == "openrouter":
        batcher_kwargs["extra_body"] = {"provider": {"only": ["google-vertex"]}}

    batcher = GPTBatcher(**batcher_kwargs)

    # Process each level
    for level in [0, 1, 2, 3]:
        print(f"\n=== Decomposing Level {level} ===")

        prompts = [
            create_transformation_prompt(item["prompt"], level=level)
            for item in xstest_data
        ]

        all_completions = []
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            print(f"  Batch {batch_start // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1} ({len(batch_prompts)} samples)")

            try:
                completions = batcher.handle_message_list(batch_prompts)
                all_completions.extend(completions)
            except Exception as e:
                print(f"  Batch failed: {e}")
                all_completions.extend([None] * len(batch_prompts))

        # Parse results
        for idx, completion in enumerate(tqdm(all_completions, desc=f"Parsing Level {level}")):
            if completion is None:
                continue
            try:
                # Extract text from ChatCompletion object
                if hasattr(completion, "choices"):
                    generated_text = completion.choices[0].message.content or ""
                elif isinstance(completion, str):
                    generated_text = completion
                else:
                    generated_text = str(completion)
                generated_text = generated_text.strip()
                parsed = parse_generated_content(generated_text)
                level_key = f"level_{level}"
                level_data = parsed.get(level_key)
                if level_data:
                    results[idx][level_key] = level_data
            except Exception as e:
                print(f"\nWarning: Parse error at sample {idx}, level {level}: {e}")

        # Save intermediate
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Level {level} done, saved intermediate results")

    # Final stats
    stats = {lv: sum(1 for r in results if r[f"level_{lv}"] is not None) for lv in [0, 1, 2, 3]}
    print(f"\nDone! Processed {len(results)} XSTest safe prompts")
    for lv, count in stats.items():
        print(f"  Level {lv}: {count}/{len(results)} successful")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    Fire(decompose_xstest)
