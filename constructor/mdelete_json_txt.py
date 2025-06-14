"""本脚本用于删除glmresult下一个层级的目录"""
import os
import shutil
import logging

import re
import glob

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_subfolders.log"),
        logging.StreamHandler()
    ]
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_json_files.log"),
        logging.StreamHandler()
    ]
)

def process_subfolders(base_path):
    """
    遍历指定的父文件夹下的所有子文件夹，并处理其中的 glm_result 文件夹。

    参数:
        base_path (str): 父文件夹的路径。
    """
    try:
        # 检查提供的路径是否有效
        if not os.path.isdir(base_path):
            logging.error(f"The provided path {base_path} is not a valid directory.")
            return
        
        # 遍历所有的子文件夹
        for subfolder in os.listdir(base_path):
            subfolder_path = os.path.join(base_path, subfolder)
            
            # 检查是否是文件夹
            if os.path.isdir(subfolder_path):
                glm_result_path = os.path.join(subfolder_path, "glm_result")
                
                # 如果 glm_result 文件夹存在
                if os.path.exists(glm_result_path) and os.path.isdir(glm_result_path):
                    # 处理 'cur', 'next', 'prev' 目录
                    for dir_name in ['cur', 'next', 'prev']:
                        dir_path = os.path.join(glm_result_path, dir_name, "response.json")# 注意文件夹名是response.txt还是.json

                        # 确认 response.json 存在并且是一个文件（根据实际情况调整）
                        if os.path.isfile(dir_path):
                            try:
                                # 移动文件到上一级目录
                                new_file_path = os.path.join(glm_result_path, dir_name, os.path.basename(dir_path))
                                shutil.move(dir_path, new_file_path)
                                logging.info(f"Moved file: {dir_path} to {new_file_path}")
                                
                                # 创建空的 response.json 文件夹
                                os.makedirs(dir_path, exist_ok=True)
                            except Exception as e:
                                logging.error(f"Failed to move or create directory for {dir_path}: {e}")
                        elif os.path.isdir(dir_path):
                            # 获取 response.json 文件夹中的所有文件
                            try:
                                for filename in os.listdir(dir_path):
                                    file_path = os.path.join(dir_path, filename)
                                    
                                    # 如果是文件，移动到上一级目录
                                    if os.path.isfile(file_path):
                                        shutil.move(file_path, os.path.join(glm_result_path, dir_name, filename))
                                        logging.info(f"Moved file: {file_path} to {os.path.join(glm_result_path, dir_name, filename)}")
                                        
                                # 删除空的 response.json 文件夹
                                os.rmdir(dir_path)
                                logging.info(f"Removed empty directory: {dir_path}")
                            except Exception as e:
                                logging.error(f"Error processing directory {dir_path}: {e}")
                else:
                    logging.warning(f"glm_result folder does not exist in {subfolder_path}")
            else:
                logging.warning(f"{subfolder_path} is not a directory.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing folders: {e}")

def rm_json(base_path):
    """遍历路径下的文件，处理glm_sub/sub/下的重复扩展名文件，
    即将名为[x,x,x,x]_0.jpg.json的文件重命名为[x,x,x,x]_0.jpg"""
    
    # 定义目标文件的正则表达式，匹配 .json 后缀的文件
    pattern_json = re.compile(r"(\[.*\]_0\.jpg)\.json$")
    
    # 使用 glob 查找所有符合条件的文件
    file_paths = glob.glob(os.path.join(base_path, '_*', 'glm_result', 'sub', '*_0.jpg.json'))
    
    if not file_paths:
        print("没有找到符合条件的文件。请检查路径和匹配规则。")
        return

    # 遍历找到的所有文件
    for file_path in file_paths:
        # 从文件路径中提取文件名
        file_name = os.path.basename(file_path)
        
        # 使用正则表达式检查文件是否符合命名规则
        match = pattern_json.match(file_name)
        if match:
            # 提取文件名（去掉 .json 后缀）
            new_name = match.group(1)
            
            # 生成新的文件路径
            new_file_path = os.path.join(os.path.dirname(file_path), new_name)
            
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"Renamed: {file_path} -> {new_file_path}")
        else:
            print(f"文件不符合命名规则: {file_path}")

def rm_txt(base_path):
    """遍历路径下的文件，处理glm_sub/sub/下的重复扩展名文件，
    即将名为[x,x,x,x]_0.jpg.txt的文件重命名为[x,x,x,x]_0.jpg"""
    
    # 定义目标文件的正则表达式，匹配 .json 后缀的文件
    pattern_txt = re.compile(r"(\[.*\]_0\.jpg)\.txt$")
    
    # 使用 glob 查找所有符合条件的文件
    file_paths = glob.glob(os.path.join(base_path, '_*', 'glm_result', 'sub', '*_0.jpg.json'))
    
    if not file_paths:
        print("没有找到符合条件的文件。请检查路径和匹配规则。")
        return

    # 遍历找到的所有文件
    for file_path in file_paths:
        # 从文件路径中提取文件名
        file_name = os.path.basename(file_path)
        
        # 使用正则表达式检查文件是否符合命名规则
        match = pattern_txt.match(file_name)
        if match:
            # 提取文件名（去掉 .txt 后缀）
            new_name = match.group(1)
            
            # 生成新的文件路径
            new_file_path = os.path.join(os.path.dirname(file_path), new_name)
            
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"Renamed: {file_path} -> {new_file_path}")
        else:
            print(f"文件不符合命名规则: {file_path}")




if __name__ == "__main__":
    # 定义父文件夹路径
    # base_path = "/home/ubuntu/moe/PaddleOCR_m/constructor/result/2021中国医疗AI行业研究报告"
    # base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/result/0116删除了一层文件夹/2024全球医疗行业展望deloitte"
    base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/12 互联网行业可持续信息披露发展报告（2023年）"

    
    # 调用函数处理子文件夹
    process_subfolders(base_path)

    rm_json(base_path)

    rm_txt(base_path)
    
# # 定义父文件夹路径
# base_path = "./week_in_charts/fortune-or-fiction-final-v3"

# # 遍历所有的子文件夹
# for subfolder in os.listdir(base_path):
#     subfolder_path = os.path.join(base_path, subfolder)
    
#     # 检查是否是文件夹
#     if os.path.isdir(subfolder_path):
#         glm_result_path = os.path.join(subfolder_path, "glm_result")
        
#         # 如果 glm_result 文件夹存在
#         if os.path.exists(glm_result_path):
#             # 处理 'cur', 'next', 'prev' 目录
#             for dir_name in ['cur', 'next', 'prev']:
#                 dir_path = os.path.join(glm_result_path, dir_name, "response.json")
                
#                 # 确认 response.json 文件夹存在
#                 if os.path.exists(dir_path) and os.path.isdir(dir_path):
#                     # 获取 response.json 文件夹中的所有文件
#                     for filename in os.listdir(dir_path):
#                         file_path = os.path.join(dir_path, filename)
                        
#                         # 如果是文件，移动到上一级目录
#                         if os.path.isfile(file_path):
#                             shutil.move(file_path, os.path.join(glm_result_path, dir_name, filename))
                    
#                     # 删除空的 response.json 文件夹
#                     os.rmdir(dir_path)