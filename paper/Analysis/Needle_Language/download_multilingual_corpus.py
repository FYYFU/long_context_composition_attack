"""
Download multilingual corpora for haystack background text.
Use Wikipedia or CC100 as the source of multilingual corpora.
"""
import os
import re
from datasets import load_dataset, Dataset
from fire import Fire


def filter_by_language(text: str, language: str, min_ratio: float = 0.7) -> bool:
    """
    Filter text to ensure it is mostly in the target language.
    
    Args:
        text: Text content.
        language: Target language code.
        min_ratio: Minimum proportion of target-language characters.
    
    Returns:
        bool: Whether the text passes the filter.
    """
    if not text or len(text) < 50:
        return False
    
    # Unicode ranges for each language.
    language_patterns = {
        "zh": r'[\u4e00-\u9fff]',  # Chinese characters.
        "ru": r'[\u0400-\u04ff]',  # Cyrillic letters (Russian).
        "vi": r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐa-zA-Z]',  # Vietnamese, including Latin letter variants.
        "bn": r'[\u0980-\u09ff]',  # Bengali characters.
        "sw": r'[a-zA-Z]',  # Swahili uses Latin letters; no special filtering is applied.
        "ar": r'[\u0600-\u06ff\u0750-\u077f]',  # Arabic characters.
        "pt": r'[a-zA-ZàáâãçéêíóôõúÀÁÂÃÇÉÊÍÓÔÕÚ]',  # Portuguese: Latin letters and diacritics.
    }
    
    if language not in language_patterns:
        return True  # Do not filter unsupported languages.
    
    pattern = language_patterns[language]
    
    # Calculate the proportion of target-language characters.
    target_chars = len(re.findall(pattern, text))
    # Exclude spaces and punctuation.
    content_chars = len(re.sub(r'[\s\d\.\,\!\?\:\;\-\"\'\(\)\[\]\{\}]', '', text))
    
    if content_chars == 0:
        return False
    
    ratio = target_chars / content_chars
    return ratio >= min_ratio


# Language configuration: Use the new wikimedia/wikipedia format or CC100.
LANGUAGE_CORPUS_CONFIG = {
    "en": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "text_field": "text",
        "description": "English Wikipedia",
    },
    "zh": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "text_field": "text",
        "description": "Chinese Wikipedia (中文维基百科)",
    },
    "ru": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.ru",
        "text_field": "text",
        "description": "Russian Wikipedia (Русская Википедия)",
    },
    "vi": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.vi",
        "text_field": "text",
        "description": "Vietnamese Wikipedia (Wikipedia tiếng Việt)",
    },
    "bn": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.bn",
        "text_field": "text",
        "description": "Bengali Wikipedia (বাংলা উইকিপিডিয়া)",
    },
    "sw": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.sw",
        "text_field": "text",
        "description": "Swahili Wikipedia (Wikipedia ya Kiswahili)",
    },
    "ar": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.ar",
        "text_field": "text",
        "description": "Arabic Wikipedia (ويكيبيديا العربية)",
    },
    "pt": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.pt",
        "text_field": "text",
        "description": "Portuguese Wikipedia (Wikipédia em português)",
    },
}

# Fallback configuration: Use the CC100 dataset.
CC100_CONFIG = {
    "zh": {"dataset": "cc100", "config": "zh-Hans", "text_field": "text"},
    "ru": {"dataset": "cc100", "config": "ru", "text_field": "text"},
    "vi": {"dataset": "cc100", "config": "vi", "text_field": "text"},
    "bn": {"dataset": "cc100", "config": "bn", "text_field": "text"},
    "sw": {"dataset": "cc100", "config": "sw", "text_field": "text"},
    "ar": {"dataset": "cc100", "config": "ar", "text_field": "text"},
    "pt": {"dataset": "cc100", "config": "pt", "text_field": "text"},
}


def download_corpus(
    language: str = "zh",
    output_dir: str = "./noise_datasets",
    max_samples: int = 50000,
    filter_language: bool = True,
    min_language_ratio: float = 0.5,
):
    """
    Download a corpus for the specified language.
    
    Args:
        language: Language code (en, zh, ru, vi).
        output_dir: Output directory.
        max_samples: Maximum number of samples.
        filter_language: Whether to filter text outside the target language.
        min_language_ratio: Minimum proportion of target-language characters (0-1).
    """
    if language not in LANGUAGE_CORPUS_CONFIG:
        raise ValueError(f"不支持的语言: {language}. 支持的语言: {list(LANGUAGE_CORPUS_CONFIG.keys())}")
    
    config = LANGUAGE_CORPUS_CONFIG[language]
    
    print("="*60)
    print(f"下载语料库: {config['description']}")
    print("="*60)
    print(f"数据集: {config['dataset']}")
    print(f"配置: {config['config']}")
    print(f"最大样本数: {max_samples}")
    print("="*60)
    
    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Stream the dataset to avoid downloading it in full.
    print("\n正在加载数据集 (streaming mode)...")
    
    dataset = None
    
    # Try loading Wikipedia.
    try:
        stream_dataset = load_dataset(
            config["dataset"],
            config["config"],
            split="train",
            streaming=True,
        )
        
        # Select the first max_samples examples.
        samples = []
        processed = 0
        skipped = 0
        print(f"正在采样 {max_samples} 条数据...")
        print(f"语言过滤: {'开启' if filter_language else '关闭'} (min_ratio={min_language_ratio})")
        
        for item in stream_dataset:
            if len(samples) >= max_samples:
                break
            
            processed += 1
            text = item.get(config["text_field"], "")
            
            # Filter out text that is too short.
            if not text or len(text) < 100:
                skipped += 1
                continue
            
            # Language filtering.
            if filter_language and not filter_by_language(text, language, min_language_ratio):
                skipped += 1
                continue
            
            samples.append({"text": text})
            
            if len(samples) % 500 == 0:
                print(f"  已采集: {len(samples)}, 已处理: {processed}, 跳过: {skipped}")
        
        print(f"采样完成: 采集 {len(samples)} 条, 处理 {processed} 条, 跳过 {skipped} 条")
        dataset = Dataset.from_list(samples)
        
    except Exception as e:
        print(f"Wikipedia 加载失败: {e}")
        
        # Try CC100 as a fallback.
        if language in CC100_CONFIG:
            print(f"\n尝试使用 CC100 数据集...")
            cc_config = CC100_CONFIG[language]
            try:
                stream_dataset = load_dataset(
                    cc_config["dataset"],
                    cc_config["config"],
                    split="train",
                    streaming=True,
                )
                
                samples = []
                processed = 0
                skipped = 0
                
                for item in stream_dataset:
                    if len(samples) >= max_samples:
                        break
                    
                    processed += 1
                    text = item.get(cc_config["text_field"], "")
                    
                    if not text or len(text) < 100:
                        skipped += 1
                        continue
                    
                    if filter_language and not filter_by_language(text, language, min_language_ratio):
                        skipped += 1
                        continue
                    
                    samples.append({"text": text})
                    
                    if len(samples) % 500 == 0:
                        print(f"  已采集: {len(samples)}, 已处理: {processed}")
                
                dataset = Dataset.from_list(samples)
            except Exception as e2:
                print(f"CC100 加载也失败: {e2}")
                raise RuntimeError(f"无法加载 {language} 语料库")
        else:
            raise
    
    print(f"数据集大小: {len(dataset)}")
    
    # Save.
    save_path = os.path.join(output_dir, f"wikipedia_{language}")
    print(f"\n保存到: {save_path}")
    dataset.save_to_disk(save_path)
    
    print(f"\n✓ 完成! 语料库已保存到: {save_path}")
    
    # Print samples.
    print("\n样本预览:")
    print("-"*60)
    if len(dataset) > 0:
        sample_text = dataset[0]["text"][:500]
        print(sample_text + "...")
    print("-"*60)
    
    return save_path


def download_all_languages(
    languages: list = None,
    output_dir: str = "./noise_datasets",
    max_samples: int = 50000,
):
    """
    Download corpora for all languages.
    """
    if languages is None:
        languages = ["zh", "ru", "vi"]
    
    for lang in languages:
        print(f"\n{'#'*60}")
        print(f"# 处理语言: {lang}")
        print(f"{'#'*60}")
        
        try:
            download_corpus(
                language=lang,
                output_dir=output_dir,
                max_samples=max_samples,
            )
        except Exception as e:
            print(f"错误: 下载 {lang} 语料库失败: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("所有语料库下载完成!")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    for lang in languages:
        print(f"  - wikipedia_{lang}/")


if __name__ == "__main__":
    Fire({
        "download": download_corpus,
        "download_all": download_all_languages,
    })


