import os
import re

def info_aggrev(file_path, result_path, target_file_name, target_folder):
    """信息聚合器
    递归查找指定目录下的目标文件夹（如 cur）中的 rsp.txt 文件，并将其内容提取出来。
    
    args:
        file_path (str): 当前目录路径，用户提供的根目录
        result_path (str): 输出 路径，聚合后的结果存储路径
        target_file_name (str): 目标文件名称（例如 rsp.txt）
        target_folder (str): 用户指定的目标文件夹名称（例如 cur）
    """
    # 用来存储 rsp.txt 文件的内容
    rsp_txt_content = []

    # 页码信息
    start_page = None
    end_page = None

    # 用于存储路径
    paths = []

    # 遍历目录结构
    for root, dirs, files in os.walk(file_path):
        print(f"正在检查目录: {root}")

        # 检查是否存在目标文件夹（例如 cur）
        if target_folder in dirs:
            target_folder_path = os.path.join(root, target_folder)
            # 构造目标文件路径
            target_file_path = os.path.join(target_folder_path, target_file_name)

            # 检查是否是文件且是 rsp.txt 文件
            if os.path.isfile(target_file_path):
                try:
                    # 读取 rsp.txt 文件的内容
                    with open(target_file_path, 'r', encoding='utf-8') as rsp_file:
                        content = rsp_file.read()
                        rsp_txt_content.append(content)
                    print(f"成功读取 {target_file_path}")
                    
                    # 获取页码（通过正则表达式从文件夹名中提取页码）
                    folder_name = os.path.basename(root)
                    # 使用正则表达式匹配文件夹名中的数字部分（页码）
                    match = re.search(r'_(\d+)$', folder_name)
                    if match:
                        page_number = int(match.group(1))  # 提取数字作为页码
                        if start_page is None:  # 初始化起始页码
                            start_page = page_number
                        end_page = page_number  # 更新终止页码

                    # 存储目录路径用于排序
                    paths.append(root)
                except Exception as e:
                    print(f"无法读取文件 {target_file_path}: {e}")

    # 排序路径
    def extract_number(file_path):
        """从路径中提取数字部分"""
        match = re.search(r'_(\d+)', file_path)
        if match:
            return int(match.group(1))
        else:
            return -1  # 如果没有匹配到，则返回-1

    # 按照数字部分排序路径
    sorted_paths = sorted(paths, key=extract_number)

    # 根据排序后的路径依次处理文件内容
    sorted_rsp_txt_content = []
    for path in sorted_paths:
        # 只在目标文件夹下查找 rsp.txt 文件
        target_folder_path = os.path.join(path, target_folder)
        target_file_path = os.path.join(target_folder_path, target_file_name)
        if os.path.isfile(target_file_path):
            try:
                with open(target_file_path, 'r', encoding='utf-8') as rsp_file:
                    sorted_rsp_txt_content.append(rsp_file.read())
                print(f"成功读取并排序 {target_file_path}")
            except Exception as e:
                print(f"无法读取文件 {target_file_path}: {e}")

    # 如果找到了 rsp.txt 文件，保存到指定路径
    if sorted_rsp_txt_content:
        # 确保保存路径存在
        if not os.path.exists(result_path):
            os.makedirs(result_path)  # 创建路径

        # 获取正确的 basename
        # 使用正则表达式从 file_path 中提取最后一个文件夹的名称
        basename_match = re.search(r'([^/\\]+)(?=/?)$', file_path)
        if basename_match:
            basename = basename_match.group(1)
        else:
            # 如果没有匹配到有效的文件夹名，则使用默认值
            basename = os.path.basename(file_path)

        print(f"提取到的 basename: {basename}")

        # 构建文件名：aggrev_basename_起始页码_终止页码.txt
        if start_page is None or end_page is None:
            print("未找到页码信息，文件名将不包含页码。")
            result_txt_path = os.path.join(result_path, f"{basename}_aggregated.txt")
        else:
            result_txt_path = os.path.join(result_path, f"{basename}_aggrev_{start_page}_{end_page}.txt")

        try:
            # 保存所有 rsp.txt 内容到一个新的大文件中
            with open(result_txt_path, 'w', encoding='utf-8') as result_file:
                for content in sorted_rsp_txt_content:
                    result_file.write(content + "\n")
            print(f"成功保存所有 rsp.txt 内容到 {result_txt_path}")
        except Exception as e:
            print(f"无法保存文件 {result_txt_path}: {e}")
    else:
        print(f"未找到任何 {target_file_name} 文件")


if __name__ == "__main__":
    # file_path = "./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/"
    file_path = "./result/1202result_txt/week_in_charts/gbar-2024-attaining-escape-velocity-f/"
    result_path = "./result/aggrev_result"
    target = "rsp.txt"
    target_folder = "cur"  # 用户指定的目标文件夹名称（例如：cur, prev, next, sub）

    # 执行信息聚合
    info_aggrev(file_path, result_path, target, target_folder)
