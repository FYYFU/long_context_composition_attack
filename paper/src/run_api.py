"""
Run tasks in generated_tasks with different reasoning models to generate responses.
"""
import json
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to import utils.
import sys
# Use an absolute path so imports work from any working directory.
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from utils.gpt_batch import GPTBatcher

import re
import glob
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from fire import Fire
from asr import refusal_words

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
    """
    Separate thinking and response sections using the </think> marker.
    
    Args:
        response_text: Complete response text.
    
    Returns:
        tuple: (thinking_part, response_part)
    """
    # Try several possible markers.
    patterns = [
        re.compile(r'</think>\s*', re.DOTALL),
        re.compile(r'<think>(.*?)</think>', re.DOTALL),
    ]
    
    for pattern in patterns:
        parts = pattern.split(response_text, 1)
        if len(parts) > 1:
            thinking_part = parts[0].strip()
            response_part = parts[1].strip()
            return thinking_part, response_part
    
    # If no marker is found, return the original text as the response.
    return "", response_text


def check_is_safe(response_text: str) -> bool:
    """
    Check whether the response is safe by looking for refusal terms.
    
    Args:
        response_text: Response text.
    
    Returns:
        bool: True indicates a safe response that refused the harmful request.
    """
    response_lower = response_text.lower()
    return any(word.lower() in response_lower for word in refusal_words)


def load_task_file(task_file: str) -> List[Dict[str, Any]]:
    """Load the task file."""
    with open(task_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    return tasks


def process_tasks_with_model(
    task_file: str,
    model_path: str,
    output_file: str,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16368,
    batch_size: int = 10,
    api_provider: str = "openrouter",
    api_key: Optional[str] = None,
    num_workers: int = 64,
    timeout: int = 120,
    max_retries: int = 3,
    max_examples: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
):
    """
    Process a task file via the model API; only batch mode is supported.
    
    Args:
        task_file: Path to the task file.
        model_path: Model path or API model name.
        output_file: Output file path.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum number of generated tokens.
        batch_size: Batch size.
        api_provider: API provider ("openrouter", "together", "siliconflow").
        api_key: API key; read from the environment if None.
        num_workers: GPTBatcher thread pool size.
        timeout: GPTBatcher timeout in seconds.
        max_retries: Maximum retries for failed API calls.
        max_examples: Maximum samples per task file; None uses all samples.
    """
    print(f"{'='*80}")
    print(f"处理任务文件: {task_file}")
    print(f"模型: {model_path}")
    print(f"API 服务商: {api_provider}")
    print(f"{'='*80}\n")
    
    # Load tasks.
    print("正在加载任务...")
    tasks = load_task_file(task_file)
    print(f"共 {len(tasks)} 个任务")
    
    # Limit the sample count.
    if max_examples is not None and max_examples > 0:
        tasks = tasks[:max_examples]
        print(f"使用前 {len(tasks)} 个样本")
    
    # Create the output directory.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get the API key.
    config = API_PROVIDER_CONFIG[api_provider]
    if api_key is None:
        api_key = os.environ.get(config["env_key"])
        if api_key is None:
            raise ValueError(f"{config['env_key']} 环境变量未设置，请设置后重试")
    
    print(f"采样参数: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    print(f"批次大小: {batch_size}\n")
    
    # Extract all prompts.
    prompts = [task['input'] for task in tasks]

    # import ipdb; ipdb.set_trace()
    # Generate responses in batches.
    print("开始生成响应...")
    print(f"使用 GPTBatcher 批量处理 (batch_size={batch_size}, num_workers={num_workers})")
    results = []
    
    # Create a GPTBatcher instance.
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
    
    if api_provider == "openrouter":
        # Try adding extra_body if GPTBatcher supports it.
        try:
            if reasoning_effort is not None and reasoning_effort != "default":
                batcher_kwargs["extra_body"] = {
                    "provider": {
                        "only": ["google-vertex"],
                    },
                    "reasoning_effort": reasoning_effort
                }
                print(f'reasoning_effort: {reasoning_effort}')
                print(batcher_kwargs["extra_body"])
            else:
                batcher_kwargs["extra_body"] = {
                    "provider": {
                        "only": ["google-vertex"],
                    },
                }
        except:
            pass  # Ignore this option if GPTBatcher does not support it.
    
    batcher = GPTBatcher(**batcher_kwargs)
    
    # Process prompts in batches.
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="批次处理"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_tasks = tasks[batch_start:batch_end]
        
        try:
            # Process batches with GPTBatcher.
            completions = batcher.handle_message_list(batch_prompts)
            
            # Check that the returned results are valid.
            if completions is None:
                completions = [None] * len(batch_prompts)
            elif len(completions) != len(batch_prompts):
                # Pad or truncate results when the returned count does not match.
                if len(completions) < len(batch_prompts):
                    completions.extend([None] * (len(batch_prompts) - len(completions)))
                else:
                    completions = completions[:len(batch_prompts)]
            
            # Process results.
            for task, completion in zip(batch_tasks, completions):
                # Check whether the completion is None.
                if completion is None:
                    generated_text = ""
                    response = ""
                    thinking = ""
                else:
                    thinking = getattr(completion.choices[0].message, 'reasoning', '') or ''
                    response = getattr(completion.choices[0].message, 'content', '') or ''
                    generated_text = f"{thinking}\n</think>\n{response}"

                # Check safety.
                is_safe = check_is_safe(response)
                
                # Build the result item.
                result_item = {
                    "original_task": {
                        "input": task.get("input", ""),
                        "question": task.get("question", ""),
                        "target": task.get("target", ""),
                        "original_query": task.get("original_query", ""),
                        "level": task.get("level", 1),
                        "context_length": task.get("context_length", 0),
                        "n_facts": task.get("n_facts", 0),
                    },
                    "generated": {
                        "thinking": thinking,
                        "response": response,
                        "full_text": generated_text,
                        "safe": is_safe,
                    },
                }
                
                results.append(result_item)
                
        except Exception as e:
            print(f"警告: 批次 {batch_start//batch_size + 1} 处理失败: {str(e)}")
            # If a batch fails, fill in empty results for the failed samples.
            for task in batch_tasks:
                result_item = {
                    "original_task": {
                        "input": task.get("input", ""),
                        "question": task.get("question", ""),
                        "target": task.get("target", ""),
                        "original_query": task.get("original_query", ""),
                        "level": task.get("level", 1),
                        "context_length": task.get("context_length", 0),
                        "n_facts": task.get("n_facts", 0),
                    },
                    "generated": {
                        "thinking": "",
                        "response": "",
                        "full_text": "",
                        "safe": False,
                    },
                }
                results.append(result_item)
            break
            
        
        # Save intermediate results after each batch.
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save the final results.
    print(f"\n正在保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Summary statistics.
    safe_count = sum(1 for r in results if r["generated"]["safe"])
    unsafe_count = len(results) - safe_count
    safe_rate = safe_count / len(results) if len(results) > 0 else 0
    
    print(f"\n✓ 完成! 已处理 {len(results)} 个任务")
    print(f"✓ 结果已保存到: {output_file}")
    print(f"✓ 统计信息: Safe={safe_count} ({safe_rate:.2%}), Unsafe={unsafe_count} ({1-safe_rate:.2%})")


def run_all_tasks(
    tasks_folder: str = "./generated_tasks",
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    output_folder: str = "./results",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16368,
    batch_size: int = 10,
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
    """
    Run all task files in tasks_folder; only batch mode is supported.
    
    Args:
        tasks_folder: Path to the task folder.
        model_path: Model path or API model name.
        output_folder: Path to the output folder.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum number of generated tokens.
        batch_size: Batch size.
        task_pattern: Pattern for matching task files.
        exclude_summary: Whether to exclude summary files.
        api_provider: API provider ("openrouter", "together", "siliconflow").
        api_key: API key; read from the environment if None.
        num_workers: GPTBatcher thread pool size.
        timeout: GPTBatcher timeout in seconds.
        max_retries: Maximum retries for failed API calls.
        max_examples: Maximum samples per task file; None uses all samples.
    """
    # Find all task files.
    pattern = os.path.join(tasks_folder, task_pattern)
    task_files = glob.glob(pattern)
    
    if exclude_summary:
        task_files = [f for f in task_files if 'summary' not in f]
    
    task_files.sort()
    
    print(f"找到 {len(task_files)} 个任务文件:")
    for f in task_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # Create the output folder.
    model_name = model_path.split('/')[-1]  # Extract the model name from the path.
    if reasoning_effort is not None:
        model_output_folder = os.path.join(output_folder, model_name + f'-{reasoning_effort}')
    else:
        model_output_folder = os.path.join(output_folder, model_name)
    os.makedirs(model_output_folder, exist_ok=True)
    
    # Process each task file.
    for task_file in task_files:
        task_name = os.path.basename(task_file).replace('.json', '')
        output_file = os.path.join(model_output_folder, f"{task_name}_results.json")
        
        # Skip files whose output already exists.
        if os.path.exists(output_file):
            print(f"跳过已存在的文件: {output_file}")
            continue

        if '0k'not in task_name and '16k' not in task_name and '64k' not in task_name:
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
        
        print("\n" + "="*80 + "\n")
    
    print(f"✓ 所有任务处理完成! 结果保存在: {model_output_folder}")


if __name__ == "__main__":
    Fire(run_all_tasks)

