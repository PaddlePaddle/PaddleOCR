import glob
import os
import re

# 该脚本定义了FindPath类，用于导航并执行寻找局部上下文的内容
# 该脚本用于搜索文件夹下的目录，找到指定文件夹内容,并提取出basename
#提取到的basename用于后续window搜索结果里的子文件夹命名-已弃用

# 设想：后期实现
# 用于批量的实现，即通过命令行指定文件夹的路径和名称，获取文件夹路径结构
# 针对路径解析，每次对_后的数字+1，即可实现对整个文件的批量处理


class FindPath:
    def __init__(self, base_dir, target_folder_name, output_result):
        """
        初始化FindPath对象。
        
        args:
        base_dir (str): 初始目录。
        target_folder_name (str): 目标文件夹名称。
        output_result (callable): 输出结果的处理方式或位置。
        """
    
        self.base_dir = base_dir
        self.target_folder_name = target_folder_name
        self.output_result = output_result

    def find_target_file(self):
        """找到指定的文件夹内容
        returns:
        list: 找到的文件路径列表。
        """
        # 构建匹配模式
        pattern = os.path.join(self.base_dir, '**', self.target_folder_name, '**')

        # 使用glob进行匹配
        matching_files = glob.glob(pattern, recursive=True)
        
        # 对路径进行排序
        sorted_paths = self.sort_paths_by_number(matching_files)

        # 如果需要输出结果到某个地方，可以在这里实现
        if callable(self.output_result):
            self.output_result(sorted_paths)
        else:
            print("Sorted paths:")
            for path in sorted_paths:
                print(path)

        return sorted_paths

    def sort_paths_by_number(self, paths):
        """
        按照路径中的数字部分对路径列表进行排序。
        
        args:
        paths (list): 路径列表。
        
        returns:
        list: 排序后的路径列表。
        """
        def extract_number(file_path):
            """
            从路径中提取数字部分。
            
            args:
            file_path (str): 文件路径。
            
            returns:
            int: 提取的数字。
            """
            match = re.search(rf'{self.target_folder_name}_(\d+)', file_path)
            if match:
                return int(match.group(1))
            else:
                return -1  # 如果没有匹配到，则返回-1
        
        return sorted(paths, key=extract_number)

    def extract_basename_from_path(self, path):
        """
        从给定路径中提取直接父目录或文件所在目录的basename。
        
        args:
        path (str): 路径字符串。
        
        returns:
        str or None: 如果找到，则返回basename，否则返回None。
        """
        # 获取路径的父目录
        parent_dir = os.path.dirname(path)
        # 获取父目录的basename
        basename = os.path.basename(parent_dir)
        
        # 如果父目录的basename不包含目标文件夹名，则尝试获取路径本身的basename
        if self.target_folder_name not in basename:
            basename = os.path.basename(path)

        return basename if self.target_folder_name in basename else None

def count_page():
    """根据接收的文件夹名获得最大的页数并输出"""

    pass

# base_dir = "./result/1202result_txt/week_in_charts/"
# target_folder_name = "fortune-or-fiction-final-v3"

import argparse

if __name__ == "__main__":
    # 设置命令行解析器
    parser = argparse.ArgumentParser(description="Find and sort target folders in a base directory.")
    
    # 添加命令行参数
    parser.add_argument('base_dir', type=str, help='The base directory to start the search from.')
    parser.add_argument('target_folder_name', type=str, help='The name of the target folder to find.')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Optional output file path to save the results.')

    # 解析命令行参数
    args = parser.parse_args()

    def print_output(result):
        for path in result:
            print(path)
            basename = fp.extract_basename_from_path(path)
            if basename:
                print(f"Extracted basename: {basename}")

        # 如果指定了输出文件，则将结果写入文件
        if args.output:
            with open(args.output, 'w') as f:
                for path in result:
                    f.write(f"{path}\n")
                    basename = fp.extract_basename_from_path(path)
                    if basename:
                        f.write(f"Extracted basename: {basename}\n")

    try:
        # 创建FindPath实例并执行搜索
        fp = FindPath(args.base_dir, args.target_folder_name, print_output)
        sorted_paths = fp.find_target_file()
        
        if not sorted_paths:
            print("No matching files found.")
    except FileNotFoundError as e:
        print(f"Error: The specified directory does not exist - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")