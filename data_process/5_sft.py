import json
import random

input_file = "/tmp/code/dataset/arxiv-full.jsonl"   # 20000条原始数据文件路径
output_file = "/tmp/code/newdata/arxiv_sft.jsonl"

# 类别对应选项映射
label_map = {
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

options_text = (
    "\nA. quant-ph", "\nB. physics.chem-ph", "\nC. physics.atom-ph", "\nD. cond-mat.soft",
    "\nE. cs.RO", "\nF. cs.CL", "\n. cs.SE", "\nH. cs.IR", "\nI. hep-th", "\nJ. hep-ph",
    "\nK. physics.optics", "\nL. cs.AI", "\nM. cs.CV", "\nN. nucl-th", "\nO. astro-ph",
    "\nP. math.PR", "\nQ. cs.OS", "\nR. eess.SP", "\nS. math.OC", "\nT. math.DS",
    "\nU. math.DG", "\nV. math.MP", "\nW. cs.MM", "\nX. stat.ME", "Y. math.CO\n", "\nZ. cs.NE"
)

# 读取所有数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 随机抽样1000条
random.seed(42)
sampled = random.sample(data, 1000)

with open(output_file, 'w', encoding='utf-8') as f_out:
    count = 0
    for item in sampled:
        # 多类别时取最后一个类别（通常以空格分割）
        categories_str = item.get("categories", "").strip()
        if not categories_str:
            continue
        last_category = categories_str.split()[-1]

        if last_category not in label_map:
            continue

        title = item.get("title", "").replace("\n", " ").strip()
        authors = item.get("authors", "").replace("\n", " ").strip()
        abstract = item.get("abstract", "").replace("\n", " ").strip()
        if not title or not authors or not abstract:
            continue

        human_text = (
            f"Based on the title '{title}', authors '{authors}', and abstract '{abstract}', "
            f"please determine the scientific category of this paper.{options_text}"
        )

        finetune_sample = {
            "system": "你是个优秀的论文分类师",
            "conversation": [
                {
                    "human": human_text,
                    "assistant": label_map[last_category]
                }
            ]
        }

        f_out.write(json.dumps(finetune_sample, ensure_ascii=False) + "\n")
        count += 1

print(f"转换完成，共生成{count}条微调数据，保存到 {output_file}")