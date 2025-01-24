"""0115用于删除txt中不正确的路径，避免手动"""
"""修改path和re.match后面的部分"""
"""直接一键替换"""
import re
from pathlib import Path

def clean_description_paths(txt_path):
    # 提取basename
    file_path = Path(txt_path)
    parts = [part for part in file_path.parts if re.match(r'corporate-commitments-to-nature-have-evolved-since-2022', part)]
    if not parts:
        raise ValueError("The basename '---' could not be found in the provided path.")
    basename = parts[0]

    # 构建正则表达式模式匹配需要替换的部分
    pattern = re.compile(r'({}_\d+)(/{}_\d+/)+'.format(re.escape(basename), re.escape(basename)))

    # 读取文件并处理每一行
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        if line.startswith('img_description_path='):
            # 使用正则表达式替换掉多余的重复部分
            cleaned_line = pattern.sub(r'\1/', line)
        else:
            cleaned_line = line
        cleaned_lines.append(cleaned_line)

    # 将清理后的行写回到文件中
    with open(txt_path, 'w') as file:
        for line in cleaned_lines:
            file.write(line)
    print("clear up！")

# 给定的具体路径
# file_path = '/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/global-materials-perspective-2024/find_result重复.txt'
# file_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/19 新发展阶段工业绿色低碳发展路径研究报告（2023年）2023年）/find_result.txt"
# file_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/corporate-commitments-to-nature-have-evolved-since-2022r/find_result.txt"
# file_path="/home/ubuntu/moe/PaddleOCR_m/constuctor/result/pdf_test/new_week_in_charts/corporate-commitments-to-nature-have-evolved-since-2022r/find_result.txt"
file_path="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/new_week_in_charts/corporate-commitments-to-nature-have-evolved-since-2022/find_result.txt"

# 调用函数进行清理
clean_description_paths(file_path)