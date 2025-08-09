"""找到ppocr识别出来的图片数量

修改 root_directory的内容后运行"""


import os
import re

# ---------------------------- 统计文件夹下的文件数量 ----------------------------

def count_files_in_img_directories(root_path):
    """
    遍历 root_path 下的所有目录，找到名为 'img' 的文件夹并统计其内部文件的数量
    """
    total_file_count = 0
    img_dir_counts = []

    for root, dirs, files in os.walk(root_path):
        # 当前目录名是否为 'img'
        if os.path.basename(root).lower() == 'img':
            file_count = len([f for f in files if os.path.isfile(os.path.join(root, f))])
            img_dir_counts.append((root, file_count))
            total_file_count += file_count
            # 避免进一步递归进入当前的 'img' 文件夹
            dirs[:] = []

    return img_dir_counts, total_file_count

# ---------------------------- 统计PP识别结果的图片数量 ----------------------------

def generate_paths(root_dir, exclude_pattern):
    """
    遍历根目录及其子目录，生成符合条件的图片路径
    """
    paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg') and not file.endswith(exclude_pattern):
                img_path = os.path.join(root, file)
                paths.append(img_path)
                # print(f"img_path='{img_path}'")  

    return paths  # 返回路径列表

def count_lines_in_file(file_path):
    """
    统计文件的总行数
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")
        return None

# ---------------------------- 主执行部分 ----------------------------

if __name__ == "__main__":
    # 根目录路径
    # root_directory = "/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/12 互联网行业可持续信息披露发展报告（2023年）"
    # root_directory="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/new_week_in_charts"
    root_directory="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/new_week_in_charts/corporate-commitments-to-nature-have-evolved-since-2022"
    # 统计 'img' 文件夹下的文件数量
    print(f"Starting to count files in img folders within {root_directory}...")
    img_dirs_with_counts, total_count = count_files_in_img_directories(root_directory)

    if not img_dirs_with_counts:
        print("No img folders found.")
    else:
        for dir_path, count in img_dirs_with_counts:
            print(f"Found {count} files in {dir_path}.")
        print(f"\nTotal number of files found in all img folders: {total_count}")

    # 生成图片路径
    # result = generate_paths('/home/ubuntu/moe/PaddleOCR_m/constructor/result/rm_result/McKinsey', '_result.jpg')
    result = generate_paths(root_directory,'_result.jpg')


    # 结果保存路径
    result_path = os.path.join(root_directory, "_count.txt")

    # 确保目录存在，不存在则创建
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # 将路径写入文件
    with open(result_path, 'w', encoding='utf-8') as f:
        for img_path in result:
            f.write(f"img_path='{img_path}'\n")
    print('图片路径写入成功！')

    # 统计文本文件的总行数
    lines_count = count_lines_in_file(result_path)
    if lines_count is not None:
        print(f"文件 {result_path} 的总行数是: {lines_count}")

    # ---------------------------- OCR准确率计算 ----------------------------

    # OCR_accuracy = lines_count / total_count if total_count else 0
    # print(f"OCR识别准确率是: {OCR_accuracy}")
    
