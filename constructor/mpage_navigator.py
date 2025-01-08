import os
import shutil
import re
import sys


# 初始化方法：初始化PageNavigator对象，接收文件路径和结果路径。
# 私有方法：提取页码，返回页码列表。



class PageNavigator:
    def __init__(self, file_path, result_path):
        """
        初始化PageNavigator对象。
        
        args:
            file_path: 文件路径
            result_path: 结果路径
        """
        self.file_path = file_path
        self.result_path = result_path
        self.folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]
        self.folder_page_numbers = self._extract_page_numbers()

    def _extract_page_numbers(self):
        """从文件夹名中提取页码，并排序"""
        folder_page_numbers = []
        for folder in self.folders:
            page_number = self.get_page_number(folder)
            if page_number is not None:
                folder_page_numbers.append((folder, page_number))
        folder_page_numbers.sort(key=lambda x: x[1])
        return folder_page_numbers

    @staticmethod
    def get_page_number(folder_name):
        """从文件夹名中提取页码，假设文件夹名形式为[文件名 _编号]"""
        match = re.search(r'_(\d+)$', folder_name)
        if match:
            return int(match.group(1))
        return None
    


    def window(self, target_page, radius=None):
        """
        返回目标页码和当前页码之间的页码列表，并保存符合条件的文件夹到指定路径。

        args：
            target_page: 目标页码
            radius: 搜索半径，默认为1，正整数
        
        return:
            符合条件的文件夹名称列表，并保存符合条件的文件夹到指定路径。 
        """

        # 类型转换与参数检查
        try:
            target_page = int(target_page) #保证目标页码为整数类型
        except ValueError:
            print("Invalid! 目标页码必须为整数.")
            sys.exit(1)

        if radius is None:
            try:
                radius_input = int(input("请输入搜索半径(正整数): ")or "1")
                radius = radius_input
            except ValueError:
                print("Invalid! 将使用默认搜索半径为1.")

        if radius < 0:
            print("Error!搜索半径不能为负数.")
            sys.exit(1)

        total_pages = len(self.folder_page_numbers)
        print(f"文件总数为: {total_pages}")

        min_page = max(0, target_page - radius)
        max_page = target_page + radius

        result_folders = [
            folder for folder, page_number in self.folder_page_numbers
            if min_page <= page_number <= max_page
        ]

        result_folder_name = f"{os.path.basename(self.file_path)}_{target_page}---{min_page}--{max_page}"
        result_path = os.path.join(self.result_path, result_folder_name)

        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        for folder in result_folders:
            source_folder = os.path.join(self.file_path, folder)
            dest_folder = os.path.join(result_path, folder)
            shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
            print(f"Copied {folder} to {dest_folder}")

        return result_folders
    

# file_path = './result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3'
# result_path = './result/class_aggrev_result/'

if __name__ == "__main__":
    import argparse

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file_path', type=str, help='The path to the directory containing folders with page numbers')
    parser.add_argument('result_path', type=str, help='The path where result directories will be saved')
    parser.add_argument('--target_page', type=int, default=None, help='The target page number for the window method')
    parser.add_argument('--radius', type=int, default=1, help='The radius for the window method')

    # 解析命令行参数
    args = parser.parse_args()

    # 检查并初始化 PageNavigator 对象
    if not os.path.isdir(args.file_path):
        print(f"Error: The specified file_path '{args.file_path}' does not exist or is not a directory.")
        sys.exit(1)

    if not os.path.isdir(args.result_path):
        print(f"Warning: The specified result_path '{args.result_path}' does not exist. It will be created.")

    navigator = PageNavigator(args.file_path, args.result_path)


    # 如果提供了目标页码，则调用 window 方法
    if args.target_page is not None:
        try:
            result_folders = navigator.window(args.target_page, args.radius)
            print(f"Processed folders: {result_folders}")
        except Exception as e:
            print(f"An error occurred while processing the window: {e}")
    else:
        print("No target_page provided. Skipping window method.")
    