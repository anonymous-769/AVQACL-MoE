import dashscope
import os
import argparse
import json
import csv
from multiprocessing.pool import Pool
import numpy as np
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from pathlib import Path
import glob

# 清理问题文本的函数
def clean_question(raw_question):
    """清理问题文本，移除无关标签和换行符"""
    cleaned = re.sub(r'^<video>\s*<audio>\s*', '', raw_question)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    return cleaned.strip()

# 从JSON文件中提取数据的函数
def extract_data_from_json(json_path):
    """从单个JSON文件中提取数据"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []
    
    extracted = []
    for item in data:
        sample_id = item["id"]
        
        human_question = None
        label = None
        
        for conv in item["conversations"]:
            if conv["from"] == "human":
                raw_question = conv["value"]
                human_question = clean_question(raw_question)
            elif conv["from"] == "gpt":
                label = conv["value"]
        
        predicted = item["text"]
        
        extracted.append({
            "id": sample_id,
            "question": human_question,
            "label": label,
            "predicted": predicted
        })
    
    return extracted

# 可选：控制并发速率
SLEEP_BETWEEN_CALLS = 0  # 秒，设为 0 禁用

def parse_args():
    parser = argparse.ArgumentParser(description="Question-answer evaluation using Qwen")
    parser.add_argument("--json_dir", required=True, help="Directory containing task JSON files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results.")
    parser.add_argument("--api_key", required=True, help="DashScope API key.")
    parser.add_argument("--num_tasks", type=int, default=2, help="Number of parallel processes.")
    args = parser.parse_args()
    return args

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def call_qwen_api(prompt, api_key_value):
    """调用 Qwen-turbo API，带重试机制"""
    if SLEEP_BETWEEN_CALLS > 0:
        time.sleep(SLEEP_BETWEEN_CALLS)
    
    response = dashscope.Generation.call(
        model='qwen-flash-2025-07-28',
        messages=[{'role': 'user', 'content': prompt}],
        api_key=api_key_value,
        temperature=0.01,
    )
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} {response.message}")
    return response

def extract_json(s):
    """从字符串中提取第一个合法的 JSON 对象"""
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[\s\S]*\}', s)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass

    depth = 0
    start = -1
    for i, char in enumerate(s):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(s[start:i+1])
                except json.JSONDecodeError:
                    continue
    return None

def annotate(prediction_set, caption_files, output_dir, api_key_value):
    """评估QA对并保存结果"""
    for file in caption_files:
        key = os.path.splitext(file)[0]
        if key not in prediction_set:
            print(f"[Process-{os.getpid()}] Warning: Key '{key}' not found in prediction_set. Skipping.")
            continue

        qa_set = prediction_set[key]
        question = qa_set['question']
        answer = qa_set['answer']
        pred = qa_set['predicted']
        
        if answer.strip().lower() == pred.strip().lower():
            response_dict = {
                "score": 5,
                "pred": "yes",
                "reason": "Exact match (case-insensitive)"
            }
            result_qa_pair = [response_dict, qa_set]
            output_file = os.path.join(output_dir, f"{key}.json")
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(result_qa_pair, f, ensure_ascii=False, indent=2)
            print(f"[Process-{os.getpid()}] Skipped API call for exact match: '{key}'")
            continue
            
        prompt = (
            "You are an expert evaluator designed to assess the correctness of predicted answers "
            "in question-answer pairs. Your task is to compare the predicted answer with the ground truth answer "
            "and determine if they match meaningfully. Follow these instructions precisely:\n\n"
            "## CRITERIA\n"
            "1. Focus on semantic equivalence\n"
            "2. Accept synonyms, paraphrases, equivalent expressions\n"
            "3. Scoring (1-5):\n"
            "   5 = Strong match (minor wording differences)\n"
            "   4 = incomplete match but correct or similar elements\n"
            "   3 = Very Weak relevance\n"
            "   2 = Incorrect but related\n"
            "   1 = Completely wrong\n"
            "## Score 5 EXAMPLES OF VALID MATCHES\n"
            "- 'Shooting' and 'Machine gun fire' →  (specific instance of general category)\n"
            "- 'Fire engine whistle' and 'Fire engine siren' →  (equivalent meaning)\n"
            "- 'The sound of motorcycles' and 'motorcycle' → (equivalent meaning but concise answer)\n"
            "- 'pump' and 'machine', 'train' and 'metro', 'wind' and 'tornado' → (pump is a type of machine, metro is a type of train)\n"
            "- 'car horn' and 'vehicle sound' → (specific instance of general category)\n\n"
            "## KEY CONSIDERATIONS (set 'yes')\n"
            "- Accept specific types/instances of categories and functionally equivalent terms\n"
            "- Accept components/parts of answers and vice versa\n"
            "- Accept concise answer ('Birdsong' and 'bird', 'Police car siren' and 'police car')\n\n"
            "## OUTPUT REQUIREMENTS\n"
            "- ONLY valid JSON with keys: 'score'(1-5), 'pred'('yes'/'no'), 'reason'\n"
            "- NO additional text outside JSON\n\n"

            "## EVALUATION TASK\n"
            f"Question: {question}\n"
            f"Ground Truth Answer: {answer}\n"
            f"Predicted Answer: {pred}\n\n"

            "Your JSON response:"
        )

        try:
            response = call_qwen_api(prompt, api_key_value)

            text = response.output.get("text", "").strip()
            if not text:
                print(f"[Process-{os.getpid()}] No response text for '{key}'.")
                continue

            response_dict = extract_json(text)
            if not response_dict:
                print(f"[Process-{os.getpid()}] Failed to extract JSON for '{key}': {repr(text)}")
                continue

            if "score" not in response_dict or "pred" not in response_dict:
                print(f"[Process-{os.getpid()}] Invalid format for '{key}': missing 'score' or 'pred'. Got: {response_dict}")
                continue

            try:
                response_dict['score'] = int(response_dict['score'])
            except (TypeError, ValueError):
                print(f"[Process-{os.getpid()}] Invalid score type for '{key}': {response_dict['score']}")
                continue

            result_qa_pair = [response_dict, qa_set]
            output_file = os.path.join(output_dir, f"{key}.json")
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(result_qa_pair, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[Process-{os.getpid()}] Error processing '{key}': {e}")

def process_task(json_dir, output_dir, api_key, num_tasks):
    """处理单个任务"""
    # 创建临时目录
    task_name = os.path.basename(json_dir)
    tmp_dir = os.path.join(output_dir, f"{task_name}_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 查找所有以"task"开头的JSON文件
    task_files = glob.glob(os.path.join(json_dir, "task*.json"))
    if not task_files:
        print(f"No task files found in {json_dir}")
        return
    
    for task_file in task_files:
        # 提取任务编号（如task0, task1等）
        task_name = os.path.basename(task_file)
        task_num = task_name.split('_')[0]  # 如 "task0"
        
        # 设置输出路径
        output_json = os.path.join(output_dir, f"eval_metrices_{task_num}.json")
        if os.path.exists(output_json):
            print(f"Output file {output_json} already exists. Skipping task {task_name}.")
            continue

        print(f"Processing task: {task_name} -> {output_json}")
        
        # 读取JSON文件
        try:
            data = extract_data_from_json(task_file)
        except Exception as e:
            print(f"Error processing {task_file}: {e}")
            continue
        
        if not data:
            print(f"No valid data in {task_file}")
            continue
        
        # 输入校验
        required_keys = {'id', 'question', 'label', 'predicted'}
        new_pred_contents = []
        for idx, sample in enumerate(data):
            if not sample:
                continue
            missing = required_keys - set(sample.keys())
            if missing:
                continue
            if not sample['question'] or not sample['label'] or not sample['predicted']:
                continue

            video_id = str(sample['id'])
            new_sample = {
                'video_name': video_id,
                'question': sample['question'],
                'answer': sample['label'],
                'predicted': sample['predicted']
            }
            new_pred_contents.append(new_sample)

        if not new_pred_contents:
            print(f"No valid samples in {task_file}")
            continue

        # 构建 prediction_set
        prediction_set = {
            item['video_name']: {
                'question': item['question'],
                'answer': item['answer'],
                'predicted': item['predicted']
            }
            for item in new_pred_contents
        }

        # 生成文件列表
        caption_files = [f"{item['video_name']}.json" for item in new_pred_contents]

        max_retries = 10
        retry_count = 0

        # 主循环：处理未完成的文件
        while retry_count < max_retries:
            try:
                completed_files = set(f for f in os.listdir(tmp_dir) if f.endswith(".json"))
                incomplete_files = [f for f in caption_files if f not in completed_files]

                if len(incomplete_files) == 0:
                    print(f"🎉 All files processed for {task_name}.")
                    break

                print(f"🔁 Handling {retry_count + 1}th: {len(incomplete_files)} files remaining.")

                num_processes = min(num_tasks, len(incomplete_files))
                if num_processes == 0:
                    break

                chunk_size = (len(incomplete_files) + num_processes - 1) // num_processes
                all_parts = [
                    incomplete_files[i:i+chunk_size]
                    for i in range(0, len(incomplete_files), chunk_size)
                ]
                
                task_args = [
                    (prediction_set, part, tmp_dir, api_key)
                    for part in all_parts
                ]

                with Pool(processes=num_processes) as pool:
                    pool.starmap(annotate, task_args)

                retry_count += 1

            except KeyboardInterrupt:
                print("\n🛑 Process interrupted by user.")
                break
            except Exception as e:
                print(f"❌ Error in main loop (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(2)

        if retry_count >= max_retries:
            print(f"⚠️ Warning: Maximum retries ({max_retries}) exceeded for {task_name}.")

        # 合并结果
        combined_contents = {}
        success_count = 0
        for file_name in os.listdir(tmp_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(tmp_dir, file_name)
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        content = json.load(f)
                        key = os.path.splitext(file_name)[0]
                        combined_contents[key] = content
                        success_count += 1
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        # 保存最终结果
        try:
            with open(output_json, "w", encoding='utf-8') as f:
                json.dump(combined_contents, f, ensure_ascii=False, indent=2)
            print(f"✅ Task {task_name} completed! Results saved to {output_json}")
        except Exception as e:
            print(f"❌ Failed to save result for {task_name}: {e}")
            continue

        # 计算指标
        score_sum = 0
        count = 0
        yes_count = 0
        no_count = 0

        for key, result in combined_contents.items():
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                count += 1
                score = result[0].get('score', 0)
                if isinstance(score, (int, float)):
                    score_sum += score
                pred = str(result[0].get('pred', '')).strip().lower()
                if pred == "yes":
                    yes_count += 1
                elif pred == "no":
                    no_count += 1

        average_score = score_sum / count if count > 0 else 0
        accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0

        print(f"📊 Evaluation Summary for {task_name}:")
        print(f"   Total evaluated samples: {count}")
        print(f"   Yes count: {yes_count}")
        print(f"   No count: {no_count}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Average score: {average_score:.4f}")
        
        # 清理临时文件
        for file_name in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, file_name))

def main():
    args = parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理指定目录下的所有任务
    process_task(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        api_key=args.api_key,
        num_tasks=args.num_tasks
    )


if __name__ == "__main__":
    main()