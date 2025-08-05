import json
import os
import argparse

def convert_to_swift_format(input_file, output_file):
    """
    将 Alpaca 格式的数据转换为 Swift 格式

    输入格式 (Alpaca):
    {
        "instruction": "根据论文的标题、作者和摘要，确定该论文的科学类别。",
        "input": "Based on the title...",
        "output": "D"
    }

    输出格式 (Swift):
    {
        "system": "你是个优秀的论文分类师",
        "conversation": [
            {
                "human": "Based on the title...",
                "assistant": "D"
            }
        ]
    }
    """
    print(f"反向转换数据: {input_file} -> {output_file}")

    converted_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                # 检查数据结构
                if "instruction" not in data or "input" not in data or "output" not in data:
                    print(f"警告: 数据缺少必要字段: {data}")
                    continue

                # 构建 Swift 格式数据
                swift_data = {
                    "system": data["instruction"],
                    "conversation": [
                        {
                            "human": data["input"],
                            "assistant": data["output"],
                        }
                    ],
                }
                converted_data.append(swift_data)

            except json.JSONDecodeError:
                print(f"警告: 无法解析JSON行: {line}")
            except Exception as e:
                print(f"处理行时发生错误: {str(e)}")

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"转换完成! 共转换 {len(converted_data)} 条数据")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换数据到Swift格式")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件路径 (Alpaca格式)",
    )
    parser.add_argument("--output", type=str, required=True, help="输出文件路径 (Swift格式)")

    args = parser.parse_args()
    convert_to_swift_format(args.input, args.output)