import os
import json
from tqdm import tqdm

# 指定train.jsonl文件路径
jsonl_file_path = "./dataset/splited/train.jsonl"

# 创建一个文件夹来存储提取的信息
output_folder = "./dataset/ropgen/origin"
os.makedirs(output_folder, exist_ok=True)

# 打开train.jsonl文件并逐行处理
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in tqdm(jsonl_file, desc='to_ropgen', ncols=100):
        data = json.loads(line)
        func = data.get("func")
        target = data.get("target")
        idx = data.get("idx")
        if func is not None and target is not None and idx is not None:
            target_folder = os.path.join(output_folder, str(target))
            os.makedirs(target_folder, exist_ok=True)
            file_path = os.path.join(target_folder, f"{idx}.c")
            with open(file_path, 'w') as output_file:
                output_file.write(func)

