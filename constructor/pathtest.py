import os

path = "./constructor/result/fortune-or-fiction-final-v3_4/glm_result/cur/response.json/rsp.txt"

# moe/PaddleOCR_m/constructor/result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_4/glm_result/cur/response.json/rsp.txt

# spilt_path = os.path.split(path)
# print(f"spilt_path 分割路径与文件名:{spilt_path}\n")

# dirname=os.path.dirname(path)
# print(f"dirname 返回所在路径:{dirname}\n")

# basename= os.path.basename(path)
# print(f"basename 返回文件名 : {basename}\n")



# ------------------------
import glob
import os
import re

base_path = "./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/"

# 递归查找多级目录
pattern = os.path.join(base_path, 'fortune-or-fiction-final-v3_*', 'glm_result', 'cur', 'response.json', 'rsp.txt')

# 获取所有匹配项
items = glob.glob(pattern, recursive=True)

# 定义一个函数用于提取路径中的数字部分以便排序
def extract_number(file_path):
    match = re.search(r'fortune-or-fiction-final-v3_(\d+)', file_path)
    if match:
        return int(match.group(1))
    else:
        return -1  # 如果没有匹配到，则返回-1，这通常意味着路径不符合预期格式

# 对所有匹配项按照提取出的数字进行排序
sorted_items = sorted(items, key=extract_number)

# 打印排序后的结果
for item in sorted_items:
    print(item)
