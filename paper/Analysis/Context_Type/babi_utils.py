# Adapted from BABILong (https://github.com/booydar/babilong).
# Apache-2.0; see third_party/licenses/babilong.txt.
# Modified for decomposed AdvBench inputs and the context experiments.
import pandas as pd
import numpy as np
import re
import nltk
from torch.utils.data import Dataset


def compare_answers(target, output):
    target = target.lower()
    output = output.lower()
    # take only the first sentence from output
    output = output.split('.')[0]
    # filter responses when model tries to generate examples
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]

    # we consider prediction correct if target is in output
    if target in output:
        return True

    return False


# babi task loader dataset for transformed AdvBench data
class TaskDataset(Dataset):
    def __init__(self, dataset_path, level=1, max_n_facts=None):

        import json
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.level = level
        level_names = {
            0: ['level_0'],
            1: ['level_1'],
            2: ['level_2'],
            3: ['level_3'],
            4: ['level_4'],
        }
        self.level_keys = level_names.get(level, [f"level_{level}"])
        
        # Keep only samples with data for the requested level.
        self.valid_indices = []
        for idx, item in enumerate(self.data):
            level_data = None
            for key in self.level_keys:
                if key in item and item[key] is not None:
                    level_data = item[key]
                    break
            
            if level_data is not None:
                # Support both the new facts field and the legacy context_fragments field.
                context_fragments = level_data.get("facts", level_data.get("context_fragments", []))
                if max_n_facts is None or len(context_fragments) <= max_n_facts:
                    self.valid_indices.append(idx)
        
        print(f"加载了 {len(self.valid_indices)} 个有效样本 (level {level})")

    def __getitem__(self, ind):
        """
        Retrieve a sample.
        
        Returns:
            dict: Contains the following fields:
                - 'facts': list of str, context fragments (facts)
                - 'question': str, reasoning query
                - 'answer': str, original intent used as the answer.
                - 'references': list of str, identical to facts for compatibility.
                - 'original_query': str, original query.
                - 'thinking': str, model reasoning trace.
                - 'target': str, target answer, if available.
        """
        actual_idx = self.valid_indices[ind]
        item = self.data[actual_idx]
        
        # Find level data, supporting both new and legacy field names.
        level_data = None
        for key in self.level_keys:
            if key in item and item[key] is not None:
                level_data = item[key]
                break
        
        if level_data is None:
            raise ValueError(f"无法找到 level {self.level} 的数据")
        
        # Support both the new facts field and the legacy context_fragments field.
        context_fragments = level_data.get("facts", level_data.get("context_fragments", []))
        reasoning_query = level_data.get("reasoning_query", "")
        
        # Use facts as references for compatibility with the original interface,
        # because every fact is relevant in this task.
        references = context_fragments.copy()
        
        sample = {
            'facts': context_fragments,
            'question': reasoning_query,
            'answer': item.get("original_intent", item.get("original_query", "")),
            'references': references,
            'original_query': item.get("original_query", ""),
            'thinking': item.get("thinking", ""),
            'target': item.get("target", ""),
            'level': self.level,
        }
        
        return sample

    def __len__(self):
        return len(self.valid_indices)


def sum_lengths(sentences):
    return sum([len(s) for s in sentences])


# sampler of background text
class SentenceSampler:
    def __init__(self, dataset, tokenizer, min_sentence_len=10, max_sentence_len=None, shuffle=False, random_seed=42):
        self.sample_ind = 0
        self.dataset = dataset
        self.sentences = []
        self.tokenizer = tokenizer
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.sentence_tokenizer = nltk.PunktSentenceTokenizer()
        self.shuffle = shuffle
        self.gen = np.random.default_rng(seed=random_seed)

    def get_sample(self, sample_size):
        sample = []
        total_len = 0
        while True:
            sentences = list(self.sentences)
            for i, sent in enumerate(sentences):  # add new sentence until sample_size is reached
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
            self.sample_sentences_(sample_size)  # appends new sentences, can be updated to just return new sentences

    def sample_sentences_(self, sample_size):
        sentences = []
        max_iterations = 100  # Prevent infinite loops.
        iteration = 0
        
        while len(sentences) == 0 and iteration < max_iterations:
            iteration += 1
            try:
                text = self.next_sample_()
                if self.shuffle:
                    if len(text) == 0:
                        continue
                    text = text[self.gen.choice(len(text)):]  # start from random position in text
                    text = text[:sample_size * 10]            # cut too long texts to speed up tokenization
                sentences += self.sentence_tokenizer.tokenize(text)
                if self.shuffle:
                    sentences = sentences[1:-1]
            except Exception as e:
                print(f"警告: 获取样本时出错 (迭代 {iteration}): {str(e)}")
                if iteration >= max_iterations:
                    raise RuntimeError(f"无法从数据集获取样本，已尝试 {max_iterations} 次")
                import time
                time.sleep(0.5)  # Wait before retrying.
        
        if len(sentences) == 0:
            raise RuntimeError(f"无法从数据集获取有效句子，已尝试 {max_iterations} 次")
        
        self.sentences += sentences

    def next_sample_(self):
        # Access the dataset by index.
        if self.shuffle:
            self.total_tokens = 0
            dataset_len = len(self.dataset)
            sample_ind = self.gen.choice(dataset_len)
            item = self.dataset[int(sample_ind)]
            # Support HuggingFace datasets with a 'text' field and other formats.
            if 'text' in item:
                sample = item['text']
            elif isinstance(item, dict) and len(item) == 1:
                # If there is only one field, use its value.
                sample = list(item.values())[0]
            else:
                # Try converting to a string.
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


# combined dataset for noisy babi QA
# it's recommended to use sample_size >= 1024
# and task_end_pct - task_start_pct >= 0.2 in order to
class NoiseInjectionDataset(Dataset):
    def __init__(self, task_dataset, noise_sampler, tokenizer,
                 task_start_pct=None,                   # left border of facts in sample, between 0 and 1
                 task_end_pct=None,                     # right border of facts in sample, between task_start_pct and 1
                 sample_size=1024,
                 mixed_length_ratio=0.0,                # used for mixed length curriculum, prob for shorter samples
                 random_seed=42):
        self.task_dataset = task_dataset
        self.noise_sampler = noise_sampler
        self.sample_size = sample_size
        self.mixed_length_ratio = mixed_length_ratio
        self.tokenizer = tokenizer
        self.task_start_pct = task_start_pct
        self.task_end_pct = task_end_pct
        if random_seed:
            self.gen = np.random.default_rng(seed=random_seed)

    def __getitem__(self, ind):
        sample = self.task_dataset[ind]
        facts = list(sample['facts'])
        facts_tok = self.tokenizer([' ' + f for f in facts], add_special_tokens=False)['input_ids']
        question_tok = self.tokenizer(sample['question'], add_special_tokens=False)['input_ids']
        answer_tok = self.tokenizer(sample['answer'], add_special_tokens=False)['input_ids']

        sample_size = self.get_sample_size()
        task_len = sum_lengths(facts_tok)
        background_text_len = sample_size - task_len
        background_text = self.noise_sampler.get_sample(background_text_len)
        sample['background_text'] = background_text

        if self.task_start_pct is None and self.task_end_pct is None:     # if fact position unspecified
            possible_positions = range(len(background_text) + 1) 
        else:
            task_start_ind = int(sample_size * self.task_start_pct)
            task_end_ind = int(sample_size * self.task_end_pct)
            total_facts_len = sum_lengths(facts_tok)

            possible_positions = []                                       # where can we insert facts?
            current_length = 0
            for i, text in enumerate(background_text):
                if (current_length >= task_start_ind) and (current_length < task_end_ind - total_facts_len):
                    possible_positions.append(i)
                current_length += len(text)

            if len(possible_positions) == 0:
                raise IndexError(f"Unable to insert facts in specified place: {self.task_start_pct, self.task_end_pct}."
                                 f"Total fact length: {total_facts_len}, "
                                 f"sentences length: {[len(t) for t in background_text]}. "
                                 f"Make the range wider or increase the sample size.")

        fact_positions = self.gen.choice(possible_positions, len(facts_tok))
        fact_positions.sort()
        sample['fact_positions'] = fact_positions                  # positions of facts between noise sentences

        updated_sample = [[] for _ in range(len(background_text) + 1)]
        for fact, pos in zip(facts_tok, fact_positions):
            updated_sample[pos].append(fact)

        for i, s in enumerate(background_text):
            updated_sample[i].append(s)

        flat = [i for s in updated_sample for i in s]
        tokens = [i for s in flat for i in s]

        sample['input_tokens'] = tokens
        sample['question_tokens'] = question_tok
        sample['target_tokens'] = answer_tok

        return sample

    def __len__(self):
        return len(self.task_dataset)

    def get_sample_size(self):
        if isinstance(self.sample_size, list):
            if self.gen.random() > self.mixed_length_ratio:
                return self.gen.choice(self.sample_size)
            return max(self.sample_size)
        else:
            return self.sample_size