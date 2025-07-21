

import os
import re

def info_aggrev(file_path, result_path, target_file_name, target_folder):
    """信息聚合器
    递归查找目录下的文件，并将目标文件（rsp.txt）的内容提取出来，并生成文件名包含页码的结果。
    
    args:
        file_path (str): 当前目录路径，用户提供的根目录
        result_path (str): 输出 路径，聚合后的结果存储路径
        target_file_name (str): 目标文件夹名称，用于查找特定目录下的 rsp.txt 文件
        target_folder (str): 用户指定的目标文件夹名称（如 cur, prev, next, sub 等）
    """
    # 用来存储 rsp.txt 文件的内容
    rsp_txt_content = []

    # 页码信息
    start_page = None
    end_page = None

    # 遍历目录结构
    for root, dirs, files in os.walk(file_path):
        print(f"正在检查目录: {root}")

        # 查找目标 rsp.txt 文件
        rsp_txt_path = os.path.join(root, target_file_name)
        if os.path.isfile(rsp_txt_path):
            try:
                # 读取 rsp.txt 文件的内容
                with open(rsp_txt_path, 'r', encoding='utf-8') as rsp_file:
                    content = rsp_file.read()
                    rsp_txt_content.append(content)
                print(f"成功读取 {rsp_txt_path}")
                
                # 获取页码（通过正则表达式从文件夹名中提取页码）
                folder_name = os.path.basename(root)
                # 使用正则表达式匹配文件夹名中的数字部分（页码）
                match = re.search(r'_(\d+)$', folder_name)
                if match:
                    page_number = int(match.group(1))  # 提取数字作为页码
                    if start_page is None:  # 初始化起始页码
                        start_page = page_number
                    end_page = page_number  # 更新终止页码
            except Exception as e:
                print(f"无法读取文件 {rsp_txt_path}: {e}")

        # 处理用户指定的目标文件夹
        if target_folder in dirs:
            target_folder_path = os.path.join(root, target_folder)
            if os.path.exists(target_folder_path):
                for sub_root, sub_dirs, sub_files in os.walk(target_folder_path):
                    sub_rsp_txt_path = os.path.join(sub_root, target_file_name)
                    if os.path.isfile(sub_rsp_txt_path):
                        try:
                            with open(sub_rsp_txt_path, 'r', encoding='utf-8') as sub_rsp_file:
                                sub_content = sub_rsp_file.read()
                                rsp_txt_content.append(sub_content)
                            print(f"成功读取 {sub_rsp_txt_path}")
                        except Exception as e:
                            print(f"无法读取文件 {sub_rsp_txt_path}: {e}")

    # 如果找到了 rsp.txt 文件，保存到指定路径
    if rsp_txt_content:
        # 确保保存路径存在
        if not os.path.exists(result_path):
            os.makedirs(result_path)  # 创建路径

        # 构建文件名：aggrev_basename_起始页码_终止页码.txt
        basename = os.path.basename(file_path)  # 获取文件夹名作为 basename
        if start_page is None or end_page is None:
            print("未找到页码信息，文件名将不包含页码。")
            result_txt_path = os.path.join(result_path, f"{basename}_aggregated.txt")
        else:
            result_txt_path = os.path.join(result_path, f"{basename}_aggrev_{start_page}_{end_page}.txt")

        try:
            # 保存所有 rsp.txt 内容到一个新的大文件中
            with open(result_txt_path, 'w', encoding='utf-8') as result_file:
                for content in rsp_txt_content:
                    result_file.write(content + "\n")
            print(f"成功保存所有 rsp.txt 内容到 {result_txt_path}")
        except Exception as e:
            print(f"无法保存文件 {result_txt_path}: {e}")
    else:
        print(f"未找到任何 {target_file_name} 文件")

def global_info(file_path, basename,result_path):
    """
    获取文件名，用info_aggrev()函数整合全局信息，写入txt文件
    获取的文件名从
    """
    pass

def local_info(file_path, basename,result_path):
    """
    获取文件名，用info_aggrev()函数整合局部信息，写入txt文件
    接收的参数：
    方法1：begin-end *此方法已弃用
    方法2：访问navigator运行后的局部信息文件窗口，根据用户输入的路径直接执行，后面并到方法里
    """
    pass

if __name__ == "__main__":
    # file_path后面不用加斜杠，不然会导致无法正常读取basenamename！！！
    # file_path = "./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3"
    # file_path="./result/1202result_txt/week_in_charts/americas-small-businesses-time-to-think-big"
    # file_path="./result/1202result_txt/week_in_charts/Toward-action-A-G20-agenda-for-sustainable-and-inclusive-growth"
    # result_path = "./result/aggrev_result/global/"

    # local_path="./result/window_result/americas-small-businesses-time-to-think-big_4---1--7/"
    # local_path="./result/window_result/americas-small-businesses-time-to-think-big_10---7--13"
    local_path='./result/class_aggrev_result/fortune-or-fiction-final-v3_4---3--5'
    result_local_path = "./result/aggrev_result/local/"

    target = "rsp.txt"
    target_folder = "cur"  # 用户指定的目标文件夹名称（例如：cur, prev, next, sub）

    # 执行信息聚合
    # info_aggrev(file_path, result_path, target, target_folder)

    # 执行局部信息聚合
    info_aggrev(local_path,result_local_path,target,target_folder)



# import os
# import re

# def info_aggrev(file_path, result_path, target_file_name, target_folder):
#     """信息聚合器
#     递归查找目录下的文件，并将目标文件（rsp.txt）的内容提取出来，并生成文件名包含页码的结果。
    
#     args:
#         file_path (str): 当前目录路径，用户提供的根目录
#         result_path (str): 输出路径，聚合后的结果存储路径
#         target_file_name (str): 目标文件夹名称，用于查找特定目录下的 rsp.txt 文件
#         target_folder (str): 用户指定的目标文件夹名称（如 cur, prev, next, sub 等）
#     """
#     # 用来存储 rsp.txt 文件的内容
#     rsp_txt_content = []

#     # 页码信息
#     start_page = None
#     end_page = None

#     # 遍历目录结构
#     for root, dirs, files in os.walk(file_path):
#         print(f"正在检查目录: {root}")

#         # 查找目标 rsp.txt 文件
#         rsp_txt_path = os.path.join(root, target_file_name)
#         if os.path.isfile(rsp_txt_path):
#             try:
#                 # 读取 rsp.txt 文件的内容
#                 with open(rsp_txt_path, 'r', encoding='utf-8') as rsp_file:
#                     content = rsp_file.read()
#                     rsp_txt_content.append(content)
#                 print(f"成功读取 {rsp_txt_path}")
                
#                 # 获取页码（通过正则表达式从文件夹名中提取页码）
#                 folder_name = os.path.basename(root)
#                 # 使用正则表达式匹配文件夹名中的数字部分（页码）
#                 match = re.search(r'_(\d+)$', folder_name)
#                 if match:
#                     page_number = int(match.group(1))  # 提取数字作为页码
#                     if start_page is None:  # 初始化起始页码
#                         start_page = page_number
#                     end_page = page_number  # 更新终止页码
#             except Exception as e:
#                 print(f"无法读取文件 {rsp_txt_path}: {e}")

#         # 处理用户指定的目标文件夹
#         if target_folder in dirs:
#             target_folder_path = os.path.join(root, target_folder)
#             if os.path.exists(target_folder_path):
#                 for sub_root, sub_dirs, sub_files in os.walk(target_folder_path):
#                     sub_rsp_txt_path = os.path.join(sub_root, target_file_name)
#                     if os.path.isfile(sub_rsp_txt_path):
#                         try:
#                             with open(sub_rsp_txt_path, 'r', encoding='utf-8') as sub_rsp_file:
#                                 sub_content = sub_rsp_file.read()
#                                 rsp_txt_content.append(sub_content)
#                             print(f"成功读取 {sub_rsp_txt_path}")
#                         except Exception as e:
#                             print(f"无法读取文件 {sub_rsp_txt_path}: {e}")

#     # 如果找到了 rsp.txt 文件，保存到指定路径
#     if rsp_txt_content:
#         # 确保保存路径存在
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)  # 创建路径

#         # 构建文件名：aggrev_basename_起始页码_终止页码.txt
#         basename = os.path.basename(file_path)  # 获取文件夹名作为 basename
#         if start_page is None or end_page is None:
#             print("未找到页码信息，文件名将不包含页码。")
#             result_txt_path = os.path.join(result_path, f"{basename}_aggregated.txt")
#         else:
#             result_txt_path = os.path.join(result_path, f"{basename}_aggrev_{start_page}_{end_page}.txt")

#         try:
#             # 保存所有 rsp.txt 内容到一个新的大文件中
#             with open(result_txt_path, 'w', encoding='utf-8') as result_file:
#                 for content in rsp_txt_content:
#                     result_file.write(content + "\n")
#             print(f"成功保存所有 rsp.txt 内容到 {result_txt_path}")
#         except Exception as e:
#             print(f"无法保存文件 {result_txt_path}: {e}")
#     else:
#         print(f"未找到任何 {target_file_name} 文件")

# # 示例目录路径
# file_path = "./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/"
# result_path = "./result/aggrev_result"
# target = "rsp.txt"
# target_folder = "cur"  # 用户指定的目标文件夹名称（例如：cur, prev, next, sub）

# # 执行信息聚合
# info_aggrev(file_path, result_path, target, target_folder)
