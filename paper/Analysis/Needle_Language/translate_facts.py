"""
Translate AdvBench facts into different languages.
Evaluate how multilingual settings affect model safety.

Use Gemini 2.0 Flash Lite for translation.
Language selection:
- Chinese (zh): High-resource language.
- Russian (ru): Medium-resource language.
- Vietnamese (vi): Low-resource language.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from fire import Fire
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add the parent directory to the Python path.
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))
from utils.gpt_batch import GPTBatcher

# Supported language configuration.
LANGUAGE_CONFIG = {
    "zh": {
        "name": "Chinese",
        "native_name": "中文",
        "resource_level": "high",
    },
    "ru": {
        "name": "Russian", 
        "native_name": "Русский",
        "resource_level": "medium",
    },
    "vi": {
        "name": "Vietnamese",
        "native_name": "Tiếng Việt",
        "resource_level": "low",
    },
    "bn": {
        "name": "Bengali",
        "native_name": "বাংলা",
        "resource_level": "low",
    },
    "sw": {
        "name": "Swahili",
        "native_name": "Kiswahili",
        "resource_level": "low",
    },
    "ar": {
        "name": "Arabic",
        "native_name": "العربية",
        "resource_level": "high",
    },
    "pt": {
        "name": "Portuguese",
        "native_name": "Português",
        "resource_level": "high",
    },
}

# Translation prompt template.
TRANSLATION_PROMPT = """You are a professional translator. Translate the following English text to {target_language} ({native_name}).

**Important Instructions:**
1. Translate the text accurately and naturally
2. Preserve all technical terms, code snippets, and numbers exactly as they appear
3. Maintain the same level of formality and technical precision
4. Do NOT add any explanation or commentary
5. Only output the translated text, nothing else

**English Text:**
{text}

**{target_language} Translation:**"""


def create_translation_prompt(text: str, target_lang: str) -> str:
    """Create a translation prompt."""
    lang_config = LANGUAGE_CONFIG[target_lang]
    return TRANSLATION_PROMPT.format(
        target_language=lang_config["name"],
        native_name=lang_config["native_name"],
        text=text
    )


def translate_facts_batch(
    facts: List[str],
    target_lang: str,
    batcher: GPTBatcher,
) -> List[str]:
    """Translate facts in batches."""
    prompts = [create_translation_prompt(fact, target_lang) for fact in facts]
    
    try:
        results = batcher.handle_message_list(prompts)
        translated = []
        for i, result in enumerate(results):
            if result is None:
                # Keep the original text if translation fails.
                translated.append(facts[i])
            else:
                content = getattr(result.choices[0].message, 'content', '') or ''
                translated.append(content.strip())
        return translated
    except Exception as e:
        print(f"翻译批次失败: {e}")
        return facts  # Return the original text.


def translate_advbench_dataset(
    input_file: str = "../Needle_position/outputs/advbench_transformed_api_gemini-2.0-flash-001_fixed.json",
    output_dir: str = "./translated_datasets",
    target_languages: List[str] = None,
    model_name: str = "google/gemini-2.0-flash-lite-001",
    api_provider: str = "openrouter",
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    levels: List[int] = None,
    batch_size: int = 20,
    num_workers: int = 32,
    timeout: int = 60,
    max_retries: int = 3,
):
    """
    Translate facts in the AdvBench dataset into different languages.
    
    Args:
        input_file: Input AdvBench JSON file.
        output_dir: Output directory.
        target_languages: List of target languages (zh, ru, vi).
        model_name: Translation model name.
        api_provider: API provider.
        api_key: API key
        max_samples: Maximum number of samples.
        levels: List of levels to translate.
        batch_size: Batch size.
        num_workers: Number of concurrent workers.
        timeout: Timeout.
        max_retries: Maximum number of retries.
    """
    if target_languages is None:
        target_languages = ["zh", "ru", "vi"]
    
    if levels is None:
        levels = [0, 1, 2, 3]
    
    # API configuration.
    API_PROVIDER_CONFIG = {
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "env_key": "OPENROUTER_API_KEY",
        },
    }
    
    print("="*80)
    print("翻译 AdvBench Facts")
    print("="*80)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"目标语言: {target_languages}")
    print(f"翻译模型: {model_name}")
    print(f"Levels: {levels}")
    print("="*80)
    
    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data.
    print(f"\n正在加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples is not None:
        data = data[:max_samples]
    print(f"数据量: {len(data)}")
    
    # Get the API key.
    config = API_PROVIDER_CONFIG[api_provider]
    if api_key is None:
        api_key = os.environ.get(config["env_key"])
        if api_key is None:
            raise ValueError(f"{config['env_key']} 环境变量未设置")
    
    # Create the batcher.
    batcher = GPTBatcher(
        api_key=api_key,
        model_name=model_name,
        system_prompt=None,
        temperature=0,  # Use a low temperature for consistent translations.
        num_workers=num_workers,
        timeout_duration=timeout,
        retry_attempts=max_retries,
        api_base_url=config["base_url"],
        max_tokens=2048,
    )
    
    # Translate into each language.
    for lang in target_languages:
        print(f"\n{'='*60}")
        print(f"翻译到 {LANGUAGE_CONFIG[lang]['name']} ({LANGUAGE_CONFIG[lang]['native_name']})")
        print(f"{'='*60}")
        
        translated_data = []
        
        for item_idx, item in enumerate(tqdm(data, desc=f"翻译 {lang}")):
            translated_item = {
                "original_query": item["original_query"],
                "thinking": item.get("thinking"),
                "original_harmful_query": item.get("original_harmful_query", item["original_query"]),
                "target_language": lang,
                "language_name": LANGUAGE_CONFIG[lang]["name"],
            }
            
            # Translate facts for each level.
            for level in levels:
                level_key = f"level_{level}"
                if level_key not in item:
                    continue
                
                level_data = item[level_key]
                facts = level_data.get("facts", [])
                
                if not facts:
                    translated_item[level_key] = level_data.copy()
                    continue
                
                # Translate facts.
                translated_facts = translate_facts_batch(facts, lang, batcher)
                
                # Save translation results.
                translated_item[level_key] = {
                    "reasoning_level": level_data.get("reasoning_level", level),
                    "facts": translated_facts,
                    "facts_original": facts,  # Keep the original text for comparison.
                    "reasoning_query": level_data.get("reasoning_query", ""),
                    "reasoning_steps": level_data.get("reasoning_steps", ""),
                    "equivalence_check": level_data.get("equivalence_check", ""),
                    "thinking": level_data.get("thinking"),
                }
            
            translated_data.append(translated_item)
            
            # Save every 20 samples.
            if (item_idx + 1) % 20 == 0:
                output_file = os.path.join(output_dir, f"advbench_translated_{lang}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(translated_data, f, indent=2, ensure_ascii=False)
        
        # Save the final results.
        output_file = os.path.join(output_dir, f"advbench_translated_{lang}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, indent=2, ensure_ascii=False)
        
        print(f"已保存: {output_file}")
    
    print(f"\n{'='*80}")
    print("翻译完成!")
    print(f"{'='*80}")
    print(f"输出目录: {output_dir}")
    for lang in target_languages:
        print(f"  - advbench_translated_{lang}.json ({LANGUAGE_CONFIG[lang]['name']})")


def translate_single_language(
    input_file: str = "../Needle_position/outputs/advbench_transformed_api_gemini-2.0-flash-001_fixed.json",
    output_dir: str = "./translated_datasets",
    target_language: str = "zh",
    model_name: str = "google/gemini-2.0-flash-lite-001",
    api_provider: str = "openrouter",
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    levels: List[int] = None,
    batch_size: int = 20,
    num_workers: int = 32,
    timeout: int = 60,
    max_retries: int = 3,
):
    """
    Translate one language at a time for separate runs.
    """
    translate_advbench_dataset(
        input_file=input_file,
        output_dir=output_dir,
        target_languages=[target_language],
        model_name=model_name,
        api_provider=api_provider,
        api_key=api_key,
        max_samples=max_samples,
        levels=levels,
        batch_size=batch_size,
        num_workers=num_workers,
        timeout=timeout,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    Fire({
        "translate_all": translate_advbench_dataset,
        "translate": translate_single_language,
    })






