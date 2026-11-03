"""
Run tasks in generated_tasks using gpt-oss-20b through the OpenAI API.
Run inference through the vLLM API server.
"""
import json
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to import modules from it.
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import glob
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from fire import Fire
from openai import OpenAI
from asr import refusal_words


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
    api_base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    reasoning_effort: str = "medium",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16368,
    batch_size: int = 10,
    n_examples: Optional[int] = None,
):
    """
    Process a task file through the OpenAI API.
    
    Args:
        task_file: Path to the task file.
        model_path: Model path or name.
        output_file: Output file path.
        api_base_url: API server URL.
        api_key: API key.
        reasoning_effort: Reasoning effort setting (low/medium/high).
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum number of generated tokens.
        batch_size: Batch size.
        n_examples: Process the first n examples; None processes all examples.
    """
    print(f"{'='*80}")
    print(f"处理任务文件: {task_file}")
    print(f"模型: {model_path}")
    print(f"API 地址: {api_base_url}")
    print(f"{'='*80}\n")
    
    # Load tasks.
    print("正在加载任务...")
    tasks = load_task_file(task_file)
    original_count = len(tasks)
    
    # If n_examples is specified, select only the first n examples.
    if n_examples is not None and n_examples > 0:
        tasks = tasks[:n_examples]
        print(f"共 {original_count} 个任务，选择前 {n_examples} 个进行处理")
    else:
        print(f"共 {len(tasks)} 个任务")
    
    # Create the output directory.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize the OpenAI client.
    print("\n正在初始化 OpenAI 客户端...")
    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key
    )
    
    print(f"推理参数: reasoning_effort={reasoning_effort}, temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    print(f"批次大小: {batch_size}\n")
    
    # Extract all prompts.
    prompts = [task['input'] for task in tasks]
    
    # Generate responses in batches.
    print("开始生成响应...")
    results = []
    
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="批次处理"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_tasks = tasks[batch_start:batch_end]
        
        # Generate responses in batches through the OpenAI API.
        batch_messages = []
        for prompt in batch_prompts:
            batch_messages.append([
                {"role": "user", "content": prompt}
            ])
        
        # Make individual API calls; although this endpoint does not batch requests, they can run concurrently.
        batch_outputs = []
        for messages in batch_messages:
            try:
                result = client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                batch_outputs.append(result)
            except Exception as e:
                print(f"错误: {e}")
                # Create empty results to preserve index alignment.
                batch_outputs.append(None)
        
        # Process results.
        for task, output in zip(batch_tasks, batch_outputs):
            if output is None:
                # Handle errors.
                thinking = ""
                response = ""
                generated_text = ""
            else:
                # Get thinking and response.
                message = output.choices[0].message
                thinking = getattr(message, 'reasoning_content', '') or ''
                response = message.content or ''
                # Build the complete generated text using the same separator format as the original code.
                if thinking:
                    # Use </think> as the separator to match the original code.
                    generated_text = f"{thinking}\n</think>\n{response}"
                else:
                    generated_text = response
            
            # Check safety.
            is_safe = check_is_safe(response)
            
            # Build the result item using the same format as run_tasks.py.
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
    model_path: str = "openai/gpt-oss-20b",
    output_folder: str = "./results",
    api_base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    reasoning_effort: str = "medium",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16368,
    batch_size: int = 10,
    task_pattern: str = "level_*.json",
    exclude_summary: bool = True,
    n_examples: Optional[int] = None,
):
    """
    Run all task files in tasks_folder.
    
    Args:
        tasks_folder: Path to the task folder.
        model_path: Model path or name.
        output_folder: Path to the output folder.
        api_base_url: API server URL.
        api_key: API key.
        reasoning_effort: Reasoning effort setting (low/medium/high).
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum number of generated tokens.
        batch_size: Batch size.
        task_pattern: Pattern for matching task files.
        exclude_summary: Whether to exclude summary files.
        n_examples: Process the first n examples; None processes all examples.
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
    model_name = model_path.split('/')[-1] + f'-{reasoning_effort}' # Extract the model name from the path.
    model_output_folder = os.path.join(output_folder, model_name)
    os.makedirs(model_output_folder, exist_ok=True)
    
    # Process each task file.
    for task_file in task_files:
        task_name = os.path.basename(task_file).replace('.json', '')
        output_file = os.path.join(model_output_folder, f"{task_name}_results.json")
        
        if 'level_0' in task_file:
            print(f"Skipping {task_file} because it is level 0")
            continue
        
        if os.path.exists(output_file):
            print(f"Skipping {task_file} because it already exists")
            continue
        
        process_tasks_with_model(
            task_file=task_file,
            model_path=model_path,
            output_file=output_file,
            api_base_url=api_base_url,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            batch_size=batch_size,
            n_examples=n_examples,
        )
        
        print("\n" + "="*80 + "\n")
    
    print(f"✓ 所有任务处理完成! 结果保存在: {model_output_folder}")


if __name__ == "__main__":
    Fire(run_all_tasks)
