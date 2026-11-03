"""
Step 4: Run model inference for the benign baseline.
Match run_api.py in the harmful pipeline.

Usage:
    python run_benign.py \
        --tasks_folder ./generated_tasks \
        --model_path google/gemini-2.0-flash-001 \
        --output_folder ./results \
        --api_provider openrouter \
        --batch_size 100 \
        --num_workers 64
"""
from pathlib import Path
import json
import os
import re
import sys
import glob
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from fire import Fire

# Reuse GPTBatcher from the original project.
PROJ_ROOT = str(Path(__file__).resolve().parents[2] / 'paper')
sys.path.insert(0, str(PROJ_ROOT))
from utils.gpt_batch import GPTBatcher

# API provider configuration.
API_PROVIDER_CONFIG = {
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "name": "Together AI"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "name": "OpenRouter",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "env_key": "SILICONFLOW_API_KEY",
        "name": "SiliconFlow"
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'env_key': 'OPENAI_API_KEY',
        'name': 'OpenAI',
    }
}


def split_thinking_response(response_text: str) -> Tuple[str, str]:
    """Separate the thinking and response sections."""
    patterns = [
        re.compile(r'</think>\s*', re.DOTALL),
        re.compile(r'<think>(.*?)</think>', re.DOTALL),
    ]
    for pattern in patterns:
        parts = pattern.split(response_text, 1)
        if len(parts) > 1:
            return parts[0].strip(), parts[1].strip()
    return "", response_text


def load_task_file(task_file: str) -> List[Dict[str, Any]]:
    """Load the task file."""
    with open(task_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    return tasks


def process_tasks_with_model(
    task_file: str,
    model_path: str,
    output_file: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 16368,
    batch_size: int = 100,
    api_provider: str = "openrouter",
    api_key: Optional[str] = None,
    num_workers: int = 64,
    timeout: int = 120,
    max_retries: int = 3,
    max_examples: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
):
    """Process a task file with the model, matching the harmful pipeline."""
    print(f"{'=' * 80}")
    print(f"处理任务文件: {task_file}")
    print(f"模型: {model_path}")
    print(f"API 服务商: {api_provider}")
    print(f"{'=' * 80}\n")

    print("正在加载任务...")
    tasks = load_task_file(task_file)
    print(f"共 {len(tasks)} 个任务")

    if max_examples is not None and max_examples > 0:
        tasks = tasks[:max_examples]
        print(f"使用前 {len(tasks)} 个样本")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    config = API_PROVIDER_CONFIG[api_provider]
    if api_key is None:
        api_key = os.environ.get(config["env_key"])
        if api_key is None:
            raise ValueError(f"{config['env_key']} 环境变量未设置")

    print(f"采样参数: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    print(f"批次大小: {batch_size}\n")

    prompts = [task['input'] for task in tasks]

    print("开始生成响应...")
    print(f"使用 GPTBatcher 批量处理 (batch_size={batch_size}, num_workers={num_workers})")
    results = []

    if api_provider == "openai":
        batcher_kwargs = {
            "api_key": api_key,
            "model_name": model_path,
            "system_prompt": None,
            "temperature": temperature,
            "num_workers": num_workers,
            "timeout_duration": timeout,
            "retry_attempts": max_retries,
            "api_base_url": config["base_url"],
            "max_completion_tokens": max_tokens,
            "top_p": top_p,
        }
    else:
        batcher_kwargs = {
            "api_key": api_key,
            "model_name": model_path,
            "system_prompt": None,
            "temperature": temperature,
            "num_workers": num_workers,
            "timeout_duration": timeout,
            "retry_attempts": max_retries,
            "api_base_url": config["base_url"],
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

    # OpenRouter: Use only google-vertex.
    if api_provider == "openrouter":
        try:
            if reasoning_effort is not None and reasoning_effort != "default":
                batcher_kwargs["extra_body"] = {
                    "provider": {"only": ["google-vertex"]},
                    "reasoning_effort": reasoning_effort
                }
            else:
                batcher_kwargs["extra_body"] = {
                    "provider": {"only": ["google-vertex"]},
                }
        except:
            pass

    batcher = GPTBatcher(**batcher_kwargs)

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="批次处理"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_tasks = tasks[batch_start:batch_end]

        try:
            completions = batcher.handle_message_list(batch_prompts)

            if completions is None:
                completions = [None] * len(batch_prompts)
            elif len(completions) != len(batch_prompts):
                if len(completions) < len(batch_prompts):
                    completions.extend([None] * (len(batch_prompts) - len(completions)))
                else:
                    completions = completions[:len(batch_prompts)]

            for task, completion in zip(batch_tasks, completions):
                if completion is None:
                    generated_text = ""
                    response = ""
                    thinking = ""
                else:
                    thinking = getattr(completion.choices[0].message, 'reasoning', '') or ''
                    response = getattr(completion.choices[0].message, 'content', '') or ''
                    generated_text = f"{thinking}\n</think>\n{response}"

                result_item = {
                    "original_task": {
                        "input": task.get("input", ""),
                        "question": task.get("question", ""),
                        "target": task.get("target", ""),
                        "original_query": task.get("original_query", ""),
                        "original_harmful_query": task.get("original_harmful_query", ""),
                        "level": task.get("level", 1),
                        "context_length": task.get("context_length", 0),
                        "n_facts": task.get("n_facts", 0),
                        "id": task.get("id", 0),
                        "domain": task.get("domain", ""),
                    },
                    "generated": {
                        "thinking": thinking,
                        "response": response,
                        "full_text": generated_text,
                    },
                }
                results.append(result_item)

        except Exception as e:
            print(f"警告: 批次 {batch_start // batch_size + 1} 处理失败: {str(e)}")
            for task in batch_tasks:
                result_item = {
                    "original_task": {
                        "input": task.get("input", ""),
                        "question": task.get("question", ""),
                        "target": task.get("target", ""),
                        "original_query": task.get("original_query", ""),
                        "original_harmful_query": task.get("original_harmful_query", ""),
                        "level": task.get("level", 1),
                        "context_length": task.get("context_length", 0),
                        "n_facts": task.get("n_facts", 0),
                    },
                    "generated": {
                        "thinking": "",
                        "response": "",
                        "full_text": "",
                    },
                }
                results.append(result_item)
            break

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n正在保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 完成! 已处理 {len(results)} 个任务")
    print(f"✓ 结果已保存到: {output_file}")


def run_all_tasks(
    tasks_folder: str = "./generated_tasks",
    model_path: str = "google/gemini-2.0-flash-001",
    output_folder: str = "./results",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 16368,
    batch_size: int = 100,
    task_pattern: str = "level_*.json",
    exclude_summary: bool = True,
    api_provider: str = "openrouter",
    api_key: Optional[str] = None,
    num_workers: int = 64,
    timeout: int = 120,
    max_retries: int = 3,
    max_examples: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
):
    """Run all task files."""
    pattern = os.path.join(tasks_folder, task_pattern)
    task_files = glob.glob(pattern)

    if exclude_summary:
        task_files = [f for f in task_files if 'summary' not in f]

    task_files.sort()

    print(f"找到 {len(task_files)} 个任务文件:")
    for f in task_files:
        print(f"  - {os.path.basename(f)}")
    print()

    model_name = model_path.split('/')[-1]
    if reasoning_effort is not None:
        model_output_folder = os.path.join(output_folder, model_name + f'-{reasoning_effort}')
    else:
        model_output_folder = os.path.join(output_folder, model_name)
    os.makedirs(model_output_folder, exist_ok=True)

    for task_file in task_files:
        task_name = os.path.basename(task_file).replace('.json', '')
        output_file = os.path.join(model_output_folder, f"{task_name}_results.json")

        if os.path.exists(output_file):
            print(f"跳过已存在的文件: {output_file}")
            continue

        # Process only 0k, 16k, and 64k, matching the harmful pipeline.
        if '0k' not in task_name and '16k' not in task_name and '64k' not in task_name:
            print(f"跳过 {task_name} 因为不是 0k, 16k, 64k")
            continue

        process_tasks_with_model(
            task_file=task_file,
            model_path=model_path,
            output_file=output_file,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            batch_size=batch_size,
            api_provider=api_provider,
            api_key=api_key,
            num_workers=num_workers,
            timeout=timeout,
            max_retries=max_retries,
            max_examples=max_examples,
            reasoning_effort=reasoning_effort,
        )


if __name__ == "__main__":
    Fire(run_all_tasks)
