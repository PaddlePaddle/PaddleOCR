"""成功的脚本，但是不具备鲁棒性"""

import os
import shutil
import re

def get_page_number(folder_name):
    """从文件夹名中提取页码，假设文件夹名形式为【文件名 _编号】 """
    match = re.search(r'_(\d+)$', folder_name)
    if match:
        return int(match.group(1))
    return None

# def navigator(file_path, target_page, radius, result_path):
#     """
#     给定文件夹路径，目标页码，设定搜索半径（默认为1，最大不超过min{当前页码-1，总页码-当前页码}），
#     返回目标页码和当前页码之间的页码列表，并保存符合条件的文件夹到指定路径。
    
#     :param file_path: 查找的根路径
#     :param target_page: 目标页码
#     :param radius: 搜索半径
#     :param result_path: 结果保存的目标文件夹路径
#     """
#     # 1. 遍历文件夹，获取所有文件夹名（假设文件夹名以_编号结尾）
#     folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]
    
#     # 2. 提取所有文件夹的页码并排序
#     folder_page_numbers = []
#     for folder in folders:
#         page_number = get_page_number(folder)
#         if page_number is not None:
#             folder_page_numbers.append((folder, page_number))
    
#     # 3. 按照页码排序文件夹
#     folder_page_numbers.sort(key=lambda x: x[1])

#     # 4. 获取文件夹总数
#     total_pages = len(folder_page_numbers)
#     print(f"文件夹总页数：{total_pages}")

#     # 5. 确定目标页码范围：目标页码 ± 搜索半径
#     min_page = max(0, target_page - radius)  # 页码不能小于0
#     max_page = target_page + radius
    
#     # 6. 筛选出符合范围的文件夹
#     result_folders = []
#     for folder, page_number in folder_page_numbers:
#         if min_page <= page_number <= max_page:
#             result_folders.append(folder)

#     # 7. 创建目标文件夹，如果不存在
#     if not os.path.exists(result_path):
#         os.makedirs(result_path)

#     # 8. 复制符合条件的文件夹到目标路径
#     for folder in result_folders:
#         source_folder = os.path.join(file_path, folder)
#         destination_folder = os.path.join(result_path, folder)
#         shutil.copytree(source_folder, destination_folder)
#         print(f"复制文件夹: {folder} 到 {destination_folder}")

#     # 返回搜索结果
#     return result_folders

def navigator(file_path, target_page, radius, result_path):
        """
    给定文件夹路径，目标页码，设定搜索半径（默认为1，最大不超过min{当前页码-1，总页码-当前页码}），
    返回目标页码和当前页码之间的页码列表，并保存符合条件的文件夹到指定路径。
    
    新增：由于目标页码和搜索半径需要用用户手动输入，每次生成的内容都不一样，因此动态生成保存路径
    args:
        file_path (str): 文件夹路径
        target_page (int): 目标页码
        radius (int): 搜索半径，确保为正
        result_path (str): 保存结果的路径（文件夹名成动态指定）
    return:
        file_list_path
    """
        # target_page类型转换
        target_page=int(target_page)

        # 检查radius
        if radius < 0:
            raise ValueError("Invalid! Radius must be positive.")

        # 遍历文件夹，获取所有文件夹名
        folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]

        # 提取所有文件夹的页码并排序
        folder_page_numbers= []
        for folder in folders:
             page_number = get_page_number(folder)
             if page_number is not None:
                  folder_page_numbers.append((folder,page_number))
            
        # 按照页码排序文件夹
        folder_page_numbers.sort(key=lambda x: x[1])

        # 获取文件夹总数
        total_pages = len(folder_page_numbers)
        print(f"文件总数为: {total_pages}")


        # 确定目标页码范围：目标页码±搜索半径+修正指数α
        min_page=max(0,target_page-radius) #保证搜索范围为正整数
        max_page=target_page+radius

        # 筛选出符合范围的文件夹
        # 此处逻辑尚需修改，可能存在溢出或者遗漏
        # 需要修改保证鲁棒性
        # 如果用户没有输入，则默认为1
        result_folders = []
        for folder , page_number in folder_page_numbers:
             if min_page <= page_number <= max_page:
                  result_folders.append(folder)
                
        # 动态生成结果保存路径
        # *****文件夹名称为动态生成
        # 创建动态文件夹
        result_folder_name=f"{os.path.basename(file_path)}_{target_page}---{min_page}--{max_page}"
        result_path = os.path.join(result_path, result_folder_name)
        
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        # 将窗口内的文件cv到结果文件夹
        for folder in result_folders:
             source_folder = os.path.join(file_path, folder)
             dest_folder = os.path.join(result_path, folder)
            #  使用dirs_exist_ok=True参数，如果目标文件夹已经存在，则不会报错
             shutil.copytree(source_folder, dest_folder,dirs_exist_ok=True)
             print(f"Copied {folder} to {dest_folder}")
            
        return result_folders
             

        # 





             


# 示例调用
if __name__ == '__main__':

    # file_path = './result/1202result_txt/week_in_charts/americas-small-businesses-time-to-think-big'
    # t_page = input("Please input the target page: {}")
    # target_page = 4
    # radius = 2
    # result_path = 'output_results'

    # 获取用户输入信息，包括目标页码、半径
    # file_path = input("Please input the file path: ")
    file_path = './result/1202result_txt/week_in_charts/americas-small-businesses-time-to-think-big'
    target_page = int(input("Please input the target page: "))
    radius = int(input("Please input the radius: "))
    result_path = "./result/window_result/"
    print("正在执行.......")

    try:
        results = navigator(file_path, target_page, radius, result_path)
        print(f"符合条件的文件夹：{results}")
        print("执行完毕！")
    except Exception as e:
        print(f"发生错误: {e}")
        

