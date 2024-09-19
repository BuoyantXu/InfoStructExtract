import csv
import json
import os
import pickle
import shutil
from glob import glob

import pandas as pd
from tqdm import tqdm
from zhipuai import ZhipuAI

from schema.prompts import prompt_user_extractor, prompt_system_extractor
from schema.schema import Object
from schema.utils import format_json_response
from settings import batch_key, batch_model


# Zhipu AI batch API mode
def create_batch_prompt(custom_id: str, text: str, prefix: str = "", schema: Object = None,
                        prompt_user: str = prompt_user_extractor,
                        prompt_system: str = prompt_system_extractor,
                        model: str = batch_model) -> dict:
    prompt_system = prompt_system if not schema else schema.prompt_system
    prompt_user = prompt_user if not schema else schema.prompt_user
    return {
        "custom_id": prefix + "_split_" + custom_id,
        "method": "POST",
        "url": "/v4/chat/completions",
        "body": {
            "model": model,
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


def create_batch_prompts(text_dict: dict, schema: Object = None, prefix: str = "", model: str = batch_model) -> list:
    if isinstance(text_dict, pd.core.series.Series):
        text_dict = text_dict.to_dict()
    batch_prompts = []
    for custom_id, text in text_dict.items():
        custom_id = str(custom_id)
        if len(custom_id) < 6:
            custom_id = str(custom_id).zfill(12)
        elif len(custom_id) >= 65:
            raise ValueError("Custom ID should be less than 65 characters.")
        batch_prompts.append(create_batch_prompt(custom_id, text, prefix=prefix, schema=schema, model=model))
    return batch_prompts


def write_jsonl_files(batch_prompts: list, batch_input_dir: str = "batch/batch_input", prefix: str = "",
                      max_requests_per_file: int = 50000,
                      max_file_size: int = 95 * 1024 * 1024):
    if not os.path.exists(batch_input_dir):
        os.makedirs(batch_input_dir)

    file_index = 0
    current_file_requests = []
    current_file_size = 0

    for prompt in batch_prompts:
        prompt_str = json.dumps(prompt, ensure_ascii=False) + '\n'
        prompt_size = len(prompt_str.encode('utf-8'))

        if len(current_file_requests) >= max_requests_per_file or current_file_size + prompt_size > max_file_size:
            with open(f"{batch_input_dir}/batch_input_{prefix}_{file_index}.jsonl", 'w', encoding='utf-8') as f:
                f.writelines(current_file_requests)
            file_index += 1
            current_file_requests = []
            current_file_size = 0

        current_file_requests.append(prompt_str)
        current_file_size += prompt_size

    if current_file_requests:
        with open(f"{batch_input_dir}/batch_input_{prefix}_{file_index}.jsonl", 'w', encoding='utf-8') as f:
            f.writelines(current_file_requests)


# send batch
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


# step 1: create batches (single file with multiple lines)
def step_create_batches(text_dict: dict, schema: Object = None, prefix: str = "", model: str = batch_model):
    batch_prompts = create_batch_prompts(text_dict, schema=schema, prefix=prefix, model=model)
    write_jsonl_files(batch_prompts, batch_input_dir="batch/batch_input", prefix=prefix)


# create batch files and pandas DataFrame chunks pickle files
def step_create_batches_chunks(schema: Object, paths_chunk_pkl_files: list, chunk_size: int = 100, text_column: str = ""):
    """
    将大量从原始文本数据中分割出来的chunk数据.pkl文件，按照顺序生成处理后的chunk数据.pkl文件，用prefix列标记文件，标签标记每一行。
    output.jsonl文件中的custom_id相对应，用于后续与结果的匹配 (custom_id = prefix + custom_id)
    :param schema:
    :param paths_chunk_pkl_files:
    :param chunk_size:
    :param text_column:
    :return:
    """
    progress_file = "batch/batch_chunks/progress.json"

    # 如果存在进度文件，读取进度
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
        start_index = progress_data.get("index", 0)
    else:
        start_index = 0

    # 遍历文件路径列表，并以每120个文件为一个chunk进行处理
    for i in tqdm(range(start_index, len(paths_chunk_pkl_files), chunk_size)):
        prefix = str(i // chunk_size + 1)
        # 获取当前chunk的文件路径
        chunk_paths = paths_chunk_pkl_files[i:i + chunk_size]

        # 读取所有当前chunk的文件并合并到一个DataFrame中
        combined_df = pd.concat([pd.read_pickle(path) for path in chunk_paths], ignore_index=True)

        # 生成batch文件到batch/batch_chunks目录下
        step_create_batches(combined_df[text_column], schema=schema, prefix=prefix)

        # str(combined_df.index).zfill(12)
        combined_df["prefix"] = prefix
        combined_df["custom_id"] = combined_df["prefix"] + "_" + combined_df.index.astype(str).str.zfill(12)

        # 保存当前chunk的DataFrame到batch/batch_chunks目录下
        combined_df.to_pickle(f'batch/batch_chunks/page_text_chunk_{prefix}.pkl')

        # 记录进度
        with open(progress_file, "w") as f:
            json.dump({"index": i + chunk_size}, f)


# step 2: upload batches
def step_upload_batches(batch_input_dir: str = "batch/batch_input", key: str = ""):
    if not key:
        key = batch_key
    paths_jsonl = glob(f"{batch_input_dir}/*.jsonl")
    uploaded_files = load_uploaded_files()
    files_to_upload = [f for f in paths_jsonl if f not in uploaded_files["uploaded_files"]]

    upload_count = 0
    for path in tqdm(files_to_upload):
        batch_id = send_batch(path, zhipu_key=key)
        save_uploaded_file(path, batch_id, key)
        upload_count += 1
        print(f"Uploaded {upload_count} files. Current file: {path}")


# step 3: download batches
def step_download_output():
    batch_ids = []
    with open("batch/batch_id.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            file_path, batch_id, key = row
            batch_ids.append((batch_id, key))

    paths_download = glob(r"batch/batch_output/*.jsonl")
    file_names = [os.path.basename(path).split(".")[0] for path in paths_download]
    num_files = len(batch_ids)
    n_not_downloaded = 0
    for batch_id, key in batch_ids:
        client = ZhipuAI(api_key=key)
        batch_job = client.batches.retrieve(batch_id)
        if batch_job.output_file_id and (batch_id not in file_names):
            content = client.files.content(batch_job.output_file_id)
            # 使用write_to_file方法把返回结果写入文件
            content.write_to_file(f"batch/batch_output/{batch_id}.jsonl")
            print(f"Downloaded file: {batch_id}.jsonl")
        else:
            n_not_downloaded += 1
    print(f"Downloaded {num_files - n_not_downloaded} files. {n_not_downloaded} files not finished.")


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
    with open("batch/processed/output.pkl", 'wb') as f:
        pickle.dump(messages, f)


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
    remove_files("batch/batch_input/")
    remove_files(path="batch/batch_id.csv")
    if mode == "IO":
        remove_files(path="batch/batch_output/")
    elif mode == "ALL":
        remove_files(path="batch/batch_output/")
        remove_files(path="batch/processed/")
