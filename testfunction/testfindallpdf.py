import os
import glob

def read_pdf_files(folder_path):
    print("start")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                # 在这里进行你想要的操作，比如读取文件内容
                print("Reading file:", file_path)
                # 你可以使用适当的PDF库来读取PDF文件内容
    print("end")

# 指定要遍历的文件夹路径
folder_path = './pdf'

# 调用函数进行遍历和读取
read_pdf_files(folder_path)
