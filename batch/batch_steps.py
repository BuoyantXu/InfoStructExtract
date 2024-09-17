import csv
import json
import os
import shutil
from glob import glob

import pandas as pd
from tqdm import tqdm
from zhipuai import ZhipuAI

from schema.prompts import prompt_user_extractor, prompt_system_extractor
from schema.schema import Object
from schema.utils import format_json_response
from settings import *


# Zhipu AI batch API mode
def create_batch_prompt(custom_id: str, text: str, scheme: Object = None,
                        prompt_user: str = prompt_user_extractor,
                        prompt_system: str = prompt_system_extractor) -> dict:
    prompt_system = prompt_system if not scheme else scheme.prompt_system
    prompt_user = prompt_user if not scheme else scheme.prompt_user
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v4/chat/completions",
        "body": {
            "model": "glm-4-flash",
            "messages": [
                {
                    "role": "system",
                    "content": prompt_system
                },
                {
                    "role": "user",
                    "content": prompt_user.format(text=text)
                }
            ],
            "temperature": 0
        }
    }


def create_batch_prompts(text_dict: dict) -> list:
    if isinstance(text_dict, pd.core.series.Series):
        text_dict = text_dict.to_dict()
    batch_prompts = []
    for custom_id, text in text_dict.items():
        custom_id = str(custom_id)
        if len(custom_id) < 6:
            custom_id = str(custom_id).zfill(12)
        elif len(custom_id) >= 65:
            raise ValueError("Custom ID should be less than 65 characters.")
        batch_prompts.append(create_batch_prompt(custom_id, text))
    return batch_prompts


def write_jsonl_files(batch_prompts: list, batch_input_dir: str = "batch/batch_input",
                      max_requests_per_file: int = 50000,
                      max_file_size: int = 100 * 1024 * 1024):
    if not os.path.exists(batch_input_dir):
        os.makedirs(batch_input_dir)

    file_index = 0
    current_file_requests = []
    current_file_size = 0

    for prompt in batch_prompts:
        prompt_str = json.dumps(prompt, ensure_ascii=False) + '\n'
        prompt_size = len(prompt_str.encode('utf-8'))

        if len(current_file_requests) >= max_requests_per_file or current_file_size + prompt_size > max_file_size:
            with open(f"{batch_input_dir}/batch_input_{file_index}.jsonl", 'w', encoding='utf-8') as f:
                f.writelines(current_file_requests)
            file_index += 1
            current_file_requests = []
            current_file_size = 0

        current_file_requests.append(prompt_str)
        current_file_size += prompt_size

    if current_file_requests:
        with open(f"{batch_input_dir}/batch_input_{file_index}.jsonl", 'w', encoding='utf-8') as f:
            f.writelines(current_file_requests)


# send_batch
def send_batch(file_path, zhipu_key=batch_key):
    client = ZhipuAI(api_key=zhipu_key)

    with open(file_path, "rb") as f:
        result = client.files.create(file=f, purpose="batch")

    file_name = os.path.basename(file_path).split(".")[0]
    create = client.batches.create(
        input_file_id=result.id,
        endpoint="/v4/chat/completions",
        auto_delete_input_file=True,
        metadata={"description": file_name}
    )

    batch_id = create.id
    return batch_id


def save_uploaded_file(file_path, batch_id, key):
    with open("batch/batch_id.csv", 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([file_path, batch_id, key])


def load_uploaded_files():
    if os.path.exists("batch/batch_id.csv"):
        uploaded_files = {"uploaded_files": [], "batch_ids": {}}
        with open("batch/batch_id.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                file_path, batch_id, key = row
                uploaded_files["uploaded_files"].append(file_path)
                uploaded_files["batch_ids"][file_path] = batch_id
        return uploaded_files
    return {"uploaded_files": [], "batch_ids": {}}


# format json batch response with custom_id
def format_json_batch(data, scheme: Object = None):
    content = data['response']['body']['choices'][0]['message']['content']
    custom_id = data['custom_id']

    try:
        result = format_json_response(content, scheme)
    except json.JSONDecodeError as e:
        print(f"JSON解码错误: {e}")
        return None

    if isinstance(result, dict):
        result["custom_id"] = custom_id
    return result


# step 1: create batches
def step_create_batches(text_dict: dict):
    batch_prompts = create_batch_prompts(text_dict)
    write_jsonl_files(batch_prompts, batch_input_dir="batch/batch_input")


# step 2: upload batches
def step_upload_batches(batch_input_dir: str = "batch/batch_input", key=batch_key):
    paths_jsonl = glob(f"{batch_input_dir}/*.jsonl")
    uploaded_files = load_uploaded_files()
    files_to_upload = [f for f in paths_jsonl if f not in uploaded_files["uploaded_files"]]

    upload_count = 0
    for path in tqdm(files_to_upload):
        batch_id = send_batch(path)
        save_uploaded_file(path, batch_id, key)
        upload_count += 1
        print(f"Uploaded {upload_count} files. Current file: {path}")


# step 3: download batches
def step_download_output(key=batch_key):
    batch_ids = []
    with open("batch/batch_id.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            file_path, batch_id, key = row
            batch_ids.append(batch_id)

    paths_download = glob(r"batch/batch_output/*.jsonl")
    file_names = [os.path.basename(path).split(".")[0] for path in paths_download]
    client = ZhipuAI(api_key=key)
    for batch_id in batch_ids:
        batch_job = client.batches.retrieve(batch_id)
        if batch_job.output_file_id and (batch_id not in file_names):
            content = client.files.content(batch_job.output_file_id)
            # 使用write_to_file方法把返回结果写入文件
            content.write_to_file(f"batch/batch_output/{batch_id}.jsonl")
            print(f"Downloaded file: {batch_id}.jsonl")


# step 4: merge output
def step_merge_output():
    paths_output = glob(r"batch/batch_output/*.jsonl")
    messages = []
    for path in tqdm(paths_output):
        # read jsonl file
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        messages_chunk = [format_json_batch(r) for r in data]
        messages_chunk = [message for message in messages_chunk if message]
        messages.extend(messages_chunk)

    with open("batch/processed/output.json", 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)
    df = pd.DataFrame(messages)
    df.to_pickle("batch/processed/output.pkl")


# remove batch files
def remove_files(path):
    if os.path.exists(path):
        files = glob(path + "*")
        for file in files:
            try:
                if os.path.isfile(file) or os.path.islink(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
            except PermissionError as e:
                print(f"PermissionError: {e}")
            except Exception as e:
                print(f"Error: {e}")


def remove_batch_files(mode="IN"):
    if mode == "IN":
        remove_files("batch/batch_input/")
        remove_files(path="batch/batch_id.csv")
    elif mode == "IO":
        remove_files(path="batch/batch_input/")
        remove_files(path="batch/batch_id.csv")
        remove_files(path="batch/batch_output/")
