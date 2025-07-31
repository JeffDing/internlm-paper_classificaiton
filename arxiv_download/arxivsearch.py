import arxiv
import json
import re
import time
import random
from tqdm import tqdm

def extract_category_mapping():
    """定义类别到选项的映射"""
    category_to_option = {
        'A': 'quant-ph',
        'B': 'physics.chem-ph', 
        'C': 'physics.atom-ph',
        'D': 'cond-mat.soft',
        'E': 'cs.RO',
        'F': 'cs.CL',
        'G': 'cs.SE',
        'H': 'cs.IR',
        'I': 'hep-th',
        'J': 'hep-ph',
        'K': 'physics.optics',
        'L': 'cs.AI',
        'M': 'cs.CV',
        'N': 'nucl-th',
        'O': 'astro-ph',
        'P': 'math.PR',
        'Q': 'cs.OS',
        'R': 'eess.SP',
        'S': 'math.OC',
        'T': 'math.DS',
        'U': 'math.DG',
        'V': 'math.MP',
        'W': 'cs.MM',
        'X': 'stat.ME',
        'Y': 'math.CO',
        'Z': 'cs.NE'
    }
    return category_to_option


def get_category_options_text():
    """生成选项文本"""
    options = [
        "A. quant-ph", "B. physics.chem-ph", "C. physics.atom-ph", "D. cond-mat.soft",
        "E. cs.RO", "F. cs.CL", "G. cs.SE", "H. cs.IR", "I. hep-th", "J. hep-ph",
        "K. physics.optics", "L. cs.AI", "M. cs.CV", "N. nucl-th", "O. astro-ph",
        "P. math.PR", "Q. cs.OS", "R. eess.SP", "S. math.OC", "T. math.DS",
        "U. math.DG", "V. math.MP", "W. cs.MM", "X. stat.ME", "Y. math.CO", "Z. cs.NE"
    ]
    return "\n".join(options)


# 固定instruction
INSTRUCTION = "你是一个论文分类专家。"

def format_authors(authors):
    """将作者列表格式化为要求的字符串格式"""
    formatted = []
    for author in authors:
        # 提取姓氏和首字母
        parts = author.name.split()
        if len(parts) == 0:
            continue
            
        last_name = parts[-1]
        if len(parts) > 1:
            initials = '.'.join([n[0] for n in parts[:-1]]) + '.'
        else:
            initials = ''
        formatted.append(f"{initials} {last_name}".strip())
    return ", ".join(f"'{author}'" for author in formatted)

def format_abstract(abstract):
    """清理摘要文本，移除换行符和多余空格"""
    return re.sub(r'\s+', ' ', abstract).replace('\n', ' ').strip()

def extract_page_info(comment):
    """从评论中提取页数信息"""
    if not comment:
        return "unknown"
    
    # 尝试匹配页数模式
    page_match = re.search(r'(\d+)\s*pages?', comment)
    if page_match:
        return f"{page_match.group(1)} pages"
    
    return "unknown"

def extract_figure_info(comment):
    """从评论中提取图数信息"""
    if not comment:
        return "unknown"
    
    # 尝试匹配图数模式
    fig_match = re.search(r'(\d+)\s*figures?', comment, re.IGNORECASE)
    if fig_match:
        return f"{fig_match.group(1)} figures"
    
    return "unknown"

def fetch_papers(num_per_category=1000):
    """从arXiv获取论文数据并格式化为指定格式"""
    dataset = []
    CATEGORY_MAP = extract_category_mapping()
    OPTIONS_STRING = get_category_options_text()
    total_papers = len(CATEGORY_MAP) * num_per_category
    
    with tqdm(total=total_papers, desc="Collecting papers") as pbar:
        for letter, arxiv_cat in CATEGORY_MAP.items():
            # 为每个类别创建进度条
            cat_pbar = tqdm(total=num_per_category, desc=f"Processing {arxiv_cat}", leave=False)
            
            # 分批次获取论文以避免API限制
            start_index = 0  # 添加这行，定义起始索引
            batch_size = 100  # 添加这行，定义批次大小
            papers_collected = 0
            
            while papers_collected < num_per_category:
                try:
                    # 计算本次需要获取的数量
                    remaining = num_per_category - papers_collected
                    current_batch = min(batch_size, remaining)
                    
                    # 创建客户端和搜索对象
                    client = arxiv.Client()
                    
                    # 执行搜索 - 使用正确的参数
                    search = arxiv.Search(
                        query=f"cat:{arxiv_cat}",
                        max_results=current_batch,
                        sort_by=arxiv.SortCriterion.SubmittedDate
                    )
                    
                    # 使用客户端获取结果
                    results = list(client.results(search))
                    
                    for result in results:
                        # 格式化作者
                        authors_str = format_authors(result.authors)
                        
                        # 格式化摘要
                        abstract_clean = format_abstract(result.summary)
                        
                        # 提取附加信息
                        page_info = extract_page_info(result.comment)
                        figure_info = extract_figure_info(result.comment)
                        doi = result.doi if result.doi else "not available"
                        
                        # 构建input字符串
                        input_text = (
                            f"Based on the title '{result.title}', authors {authors_str}, "
                            f"and abstract '{abstract_clean}'. Additional info: "
                            f"{page_info}, {figure_info} DOI: {doi}"
                            f"{OPTIONS_STRING}"
                        )
                        
                        # 添加到数据集
                        dataset.append({
                            "instruction": INSTRUCTION,
                            "input": input_text,
                            "output": letter
                        })
                        
                        papers_collected += 1
                        cat_pbar.update(1)
                        pbar.update(1)
                    
                    # 更新索引
                    start_index += current_batch
                    
                    # 礼貌性延迟
                    time.sleep(random.uniform(1.0, 3.0))
                    
                except Exception as e:  # 修改这里，使用通用异常处理
                    tqdm.write(f"Unexpected error for {arxiv_cat}: {str(e)}. Skipping batch...")
                    time.sleep(random.uniform(5.0, 10.0))
            
            cat_pbar.close()
    
    return dataset

# 生成样本
print("Starting data collection for papers...")
dataset = fetch_papers(num_per_category=400)

file_name = f"/root/data/newdataset/arxiv_classification_dataset_{len(dataset)}.json"
# 保存为JSON文件
with open(file_name, "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"已成功生成 {len(dataset)} 条符合格式的样本！")
print(f"数据集已保存到 arxiv_classification_dataset_{len(dataset)}.json")