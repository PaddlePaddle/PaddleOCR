
# 该脚本用于删除文件下缺失or不全的glm_result 文件夹
import os
import shutil
import logging

logging.basicConfig(filename = 'delete_glm_result_folders.log',level = logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def delete_glm_result_folders(base_path):
    """
    遍历 base_path 下的所有子目录，批量删除子目录下的 glm_result 文件夹
    """
    print(f"正在处理路径: {base_path}")
    base_path = os.path.abspath(base_path)  # 转换为绝对路径，确保正确解析
    print(f"转换后的绝对路径: {base_path}")

    # folders_to_delete = []

    # 获取一级子目录列表
    top_level_dirs = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    for dir_name in top_level_dirs:
        folder_to_check = os.path.join(base_path, dir_name)
        print(f"\n检查文件夹：{folder_to_check}")

        # 将确认放在外层循环中
        user_input = input("确认删除此文件夹下的所有二级子目录中的glm_result文件夹吗？(y/n)")
        if user_input.lower()!= "y":
            print(f"跳过文件夹{folder_to_check}")
            continue
        
        folders_to_delete = []

        # 遍历一级子目录下的二级子目录
        for sub_dir in os.listdir(folder_to_check):
            sub_folder_path = os.path.join(folder_to_check, sub_dir)
            if os.path.isdir(sub_folder_path):
                # 检查二级子目录下是否包含glm_result文件夹
                for item in os.listdir(sub_folder_path):
                    item_path = os.path.join(sub_folder_path, item)
                    if os.path.isdir(item_path) and item == "glm_result":
                        folders_to_delete.append(item_path)
                        print(f"发现文件夹: {item_path}") # 调试输出，检查文件夹是否被找到

        # 执行删除操作
        for folder in folders_to_delete:
            try:
                shutil.rmtree(folder)
                print(f"已删除文件夹: {folder}")
                print(f"================={folder}=======================")
                logging.info(f"已删除文件夹: {folder}")

            except Exception as e:
                print(f"删除文件夹 {folder} 时出错: {e}")
                logging.error(f"删除文件夹 {folder} 时出错: {e}")



        
                # # 删除并记录日志
                # for folder in folders_to_delete:
                #     try:
                #         shutil.rmtree(folder_path)  # 删除整个文件夹及其内容
                #         print(f"已删除文件夹: {folder_path}")
                #         logging.info(f"已删除文件夹: {folder_path}")
                #     except Exception as e:
                #         print(f"删除文件夹 {folder_path} 时出错: {e}")
                #         logging.error(f"删除文件夹 {folder_path} 时出错: {e}")

if __name__ == '__main__':
    # 示例调用
    # base_path = '/home/ubuntu/moe/PaddleOCR_m/constructor/mini/pp/'  # 需要遍历的根目录路径
    base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/new_week_in_charts"
    # moe/PaddleOCR_m/constructor/result/try
    # moe/PaddleOCR_m/constructor/result/1120result/try
    # moe/PaddleOCR_m/constructor/result/1120result/try
    # ./constructor/window/source/2024年中国医疗大健康产业发展白皮书1
    delete_glm_result_folders(base_path)
