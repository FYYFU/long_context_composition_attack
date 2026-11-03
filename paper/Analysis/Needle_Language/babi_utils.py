# Adapted from BABILong (https://github.com/booydar/babilong).
# Apache-2.0; see third_party/licenses/babilong.txt.
# Modified for decomposed AdvBench inputs and the context experiments.
"""
babi_utils for multilingual needle experiments
Support loading translated AdvBench data.
"""
import pandas as pd
import numpy as np
import re
import nltk
from torch.utils.data import Dataset


def compare_answers(target, output):
    target = target.lower()
    output = output.lower()
    output = output.split('.')[0]
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]
    if target in output:
        return True
    return False


class TaskDataset(Dataset):
    """
    Load translated AdvBench data.
    Support multilingual facts.
    """
    def __init__(self, dataset_path, level=1, max_n_facts=None, use_translated=True):
        """
        Args:
            dataset_path: Path to the translated JSON file.
            level: Difficulty level.
            max_n_facts: Maximum number of facts.
            use_translated: Whether to use translated facts; False uses the original English facts.
        """
        import json
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.level = level
        self.use_translated = use_translated
        level_names = {
            0: ['level_0'],
            1: ['level_1'],
            2: ['level_2'],
            3: ['level_3'],
            4: ['level_4'],
        }
        self.level_keys = level_names.get(level, [f"level_{level}"])
        
        # Filter the data.
        self.valid_indices = []
        for idx, item in enumerate(self.data):
            level_data = None
            for key in self.level_keys:
                if key in item and item[key] is not None:
                    level_data = item[key]
                    break
            
            if level_data is not None:
                context_fragments = level_data.get("facts", level_data.get("context_fragments", []))
                if max_n_facts is None or len(context_fragments) <= max_n_facts:
                    self.valid_indices.append(idx)
        
        # Get language information.
        if len(self.data) > 0:
            self.target_language = self.data[0].get("target_language", "en")
            self.language_name = self.data[0].get("language_name", "English")
        else:
            self.target_language = "en"
            self.language_name = "English"
        
        print(f"加载了 {len(self.valid_indices)} 个有效样本 (level {level}, language: {self.language_name})")

    def __getitem__(self, ind):
        actual_idx = self.valid_indices[ind]
        item = self.data[actual_idx]
        
        level_data = None
        for key in self.level_keys:
            if key in item and item[key] is not None:
                level_data = item[key]
                break
        
        if level_data is None:
            raise ValueError(f"无法找到 level {self.level} 的数据")
        
        # Choose between translated and original facts.
        if self.use_translated:
            context_fragments = level_data.get("facts", level_data.get("context_fragments", []))
        else:
            context_fragments = level_data.get("facts_original", level_data.get("facts", []))
        
        reasoning_query = level_data.get("reasoning_query", "")
        references = context_fragments.copy()
        
        sample = {
            'facts': context_fragments,
            'facts_original': level_data.get("facts_original", context_fragments),
            'question': reasoning_query,
            'answer': item.get("original_intent", item.get("original_query", "")),
            'references': references,
            'original_query': item.get("original_query", ""),
            'thinking': item.get("thinking", ""),
            'target': item.get("target", ""),
            'level': self.level,
            'target_language': self.target_language,
            'language_name': self.language_name,
        }
        
        return sample

    def __len__(self):
        return len(self.valid_indices)


def sum_lengths(sentences):
    return sum([len(s) for s in sentences])


def split_sentences_multilingual(text: str, language: str = "en") -> list:
    """
    Multilingual sentence segmentation.
    
    Args:
        text: Input text.
        language: Language code (en, zh, ru, vi, bn, sw, ar, pt).
    
    Returns:
        List of sentences.
    """
    if language == "zh":
        # Chinese: Split on Chinese punctuation.
        # Use a regular expression to split on periods, question marks, and exclamation marks.
        sentences = re.split(r'(?<=[。！？])\s*', text)
        # Filter out empty sentences.
        sentences = [s.strip() for s in sentences if s.strip()]
    elif language == "bn":
        # Bengali: Use Bengali punctuation (dari, U+0964) and common punctuation.
        # Bengali uses U+0964 as a full stop and may also use English punctuation.
        sentences = re.split(r'(?<=[।!?।])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    elif language == "ar":
        # Arabic: Use Arabic punctuation.
        # Arabic uses U+06D4 or . as a full stop, U+061F as a question mark, and ! as an exclamation mark.
        sentences = re.split(r'(?<=[.؟!۔])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    else:
        # English, Russian, Vietnamese, Swahili, and Portuguese: Use NLTK.
        try:
            sentence_tokenizer = nltk.PunktSentenceTokenizer()
            sentences = sentence_tokenizer.tokenize(text)
        except:
            # Fallback: Split on periods.
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


class SentenceSampler:
    """Background text sampler with multilingual support."""
    def __init__(self, dataset, tokenizer, min_sentence_len=10, max_sentence_len=None, 
                 shuffle=False, random_seed=42, language="en"):
        self.sample_ind = 0
        self.dataset = dataset
        self.sentences = []
        self.tokenizer = tokenizer
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.language = language  # Language code.
        self.shuffle = shuffle
        self.gen = np.random.default_rng(seed=random_seed)

    def get_sample(self, sample_size):
        sample = []
        total_len = 0
        while True:
            sentences = list(self.sentences)
            for i, sent in enumerate(sentences):
                tokenized = self.tokenizer.encode(' ' + sent, add_special_tokens=False)
                if not self.length_is_ok(tokenized):
                    continue
                total_len += len(tokenized)
                sample.append(tokenized)
                if total_len >= sample_size:
                    self.sentences = self.sentences[i+1:]
                    cutoff = total_len - sample_size
                    if cutoff > 0:
                        sample[-1] = sample[-1][:-cutoff]
                    return sample

            self.sentences = []
            self.sample_sentences_(sample_size)

    def sample_sentences_(self, sample_size):
        sentences = []
        max_iterations = 100
        iteration = 0
        
        while len(sentences) == 0 and iteration < max_iterations:
            iteration += 1
            try:
                text = self.next_sample_()
                if self.shuffle:
                    if len(text) == 0:
                        continue
                    text = text[self.gen.choice(len(text)):]
                    text = text[:sample_size * 10]
                
                # Use multilingual sentence segmentation.
                new_sentences = split_sentences_multilingual(text, self.language)
                sentences += new_sentences
                
                if self.shuffle and len(sentences) > 2:
                    sentences = sentences[1:-1]
            except Exception as e:
                print(f"警告: 获取样本时出错 (迭代 {iteration}): {str(e)}")
                if iteration >= max_iterations:
                    raise RuntimeError(f"无法从数据集获取样本，已尝试 {max_iterations} 次")
                import time
                time.sleep(0.5)
        
        if len(sentences) == 0:
            raise RuntimeError(f"无法从数据集获取有效句子，已尝试 {max_iterations} 次")
        
        self.sentences += sentences

    def next_sample_(self):
        if self.shuffle:
            self.total_tokens = 0
            dataset_len = len(self.dataset)
            sample_ind = self.gen.choice(dataset_len)
            item = self.dataset[int(sample_ind)]
            if 'text' in item:
                sample = item['text']
            elif isinstance(item, dict) and len(item) == 1:
                sample = list(item.values())[0]
            else:
                sample = str(item)
        else:
            dataset_len = len(self.dataset)
            item = self.dataset[int(self.sample_ind)]
            if 'text' in item:
                sample = item['text']
            elif isinstance(item, dict) and len(item) == 1:
                sample = list(item.values())[0]
            else:
                sample = str(item)
            self.sample_ind += 1
            self.sample_ind = self.sample_ind % dataset_len
        
        return sample

    def length_is_ok(self, tokenized):
        if self.max_sentence_len is not None and len(tokenized) > self.max_sentence_len:
            return False
        if self.min_sentence_len is not None and len(tokenized) < self.min_sentence_len:
            return False
        return True






