import json

input_path = "/tmp/code/newdataset/arxiv.jsonl"          # 上一步筛选后的数据
output_path = "/tmp/code/newdataset/arxiv_high.jsonl"  # 输出高质量数据

count = 0

def extract_category_mapping():
    """定义类别到选项的映射"""
    category_to_option = {
        'quant-ph': 'A',
        'physics.chem-ph': 'B', 
        'physics.atom-ph': 'C',
        'cond-mat.soft': 'D',
        'cs.RO': 'E',
        'cs.CL': 'F',
        'cs.SE': 'G',
        'cs.IR': 'H',
        'hep-th': 'I',
        'hep-ph': 'J',
        'physics.optics': 'K',
        'cs.AI': 'L',
        'cs.CV': 'M',
        'nucl-th': 'N',
        'astro-ph': 'O',
        'math.PR': 'P',
        'cs.OS': 'Q',
        'eess.SP': 'R',
        'math.OC': 'S',
        'math.DS': 'T',
        'math.DG': 'U',
        'math.MP': 'V',
        'cs.MM': 'W',
        'stat.ME': 'X',
        'math.CO': 'Y',
        'cs.NE': 'Z'
    }
    return category_to_option

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        try:
            record = json.loads(line)

            # 获取更新日期和摘要
            update_date = record.get("update_date", "")
            abstract = record.get("abstract", "")
            
            category_mapping = extract_category_mapping()
            
            # 提取基本信息
            paper_id = record.get('id', '')
            title = record.get('title', '').replace('\n', ' ').strip()
            authors = record.get('authors', '')
            categories = record.get('categories', '')
            
            category_list = categories.split()
            
            # 检查类别是否在我们的目标列表中
            target_category = category_list[0] if category_list else ''
            if target_category not in category_mapping:
                print(f"跳过非目标类别论文 {paper_id}: {target_category}")
                continue
    
            # 检查是否包含多个类别（用空格分隔）
            #if len(category_list) > 1:
            #    print(f"跳过多类别论文 {paper_id}: {categories}")
            #    continue

            # 过滤条件，这里根据自己的模型参数修改
            if len(abstract) >= 300 and len(abstract)<=4096:
                if update_date and int(update_date[:4]) >= 2020:
                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                    count += 1

        except json.JSONDecodeError:
            continue  # 跳过格式错误的行

print(f"高质量筛选完成，共保留 {count} 条记录到 {output_path}")