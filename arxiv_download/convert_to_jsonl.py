import json

# 读取原始JSON数据
with open('/tmp/code/dataset/arxiv_classification_dataset_20000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建新的JSONL格式数据
with open('new_sftdata_20000.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        # 创建新的数据项，使用简化的instruction
        new_item = {
            "instruction": "你是个优秀的论文分类师",
            "input": item["input"],
            "output": item["output"]
        }
        # 写入JSONL格式
        f.write(json.dumps(new_item, ensure_ascii=False) + '\n')

print("转换完成，已生成new_sftdata_20000.jsonl文件")