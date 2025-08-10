import json
from pathlib import Path

# 定义文件夹路径和文件名
folder_path = Path("/root/arxiv/dataset")
filename = "arxiv_classification_dataset_350012.json"

# 拼接完整文件路径
file_path = folder_path / filename

# 确保文件夹存在
if not folder_path.exists():
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"文件夹 '{folder_path}' 不存在，已自动创建。")
else:
    print(f"文件夹 '{folder_path}' 已存在。")

# 读取原始JSON数据
with open('/root/data/newdataset/arxiv_classification_dataset_350012.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建新的JSONL格式数据
with open(file_path, 'w', encoding='utf-8') as f:
    for item in data:
        # 创建新的数据项，使用简化的instruction
        new_item = {
            "instruction": "你是个优秀的论文分类师",
            "input": item["input"],
            "output": item["output"]
        }
        # 写入JSONL格式
        f.write(json.dumps(new_item, ensure_ascii=False) + '\n')

print(f"转换完成，已生成{file_path}文件")
