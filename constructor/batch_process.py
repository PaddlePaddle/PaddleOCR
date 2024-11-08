"""这是成功的脚本，更改输入的pdf文件路径
输出结果如下：
-pdf每一页转为的图像
-该页图像
    --该页图像的ocr结果
    --该页图像
        ---该页图像中截取出的图像
        ---res0.txt

+ 处理pdf文件时，不仅识别出图表内容，截取图表，同时记录图表在原pdf文件中的位置，并返回本身页和上下页
因此处理流程修改如下：
1. 读取pdf转换为img；
2. 识别img中的图表内容
3. 记录图表的位置信息（包括页码
4. 保存识别结果和相关页面


"""

import os
import cv2
from paddleocr import PPStructure, draw_structure_result, save_structure_res
from PIL import Image
from paddle.utils import try_import #"Lazy imports for heavy dependencies."
import py3nvml.py3nvml as nvml
import time
import fitz
import pymupdf
import numpy as np

"""该脚本用于识别图片形式的文件中的图表
首先判断文件类型，是pdf则进行转换
然后对图像进行PaddleOCR的识别，
识别结果的形式为单页识别，
当前的结果目录结构为：
-文件名_page_number
--文件名_page_number
---版式识别结果 （.jpg)
---版式识别内容(txt)
--文件名_page_number OCR： 识别结果
文件名_page_number.png： 转换为的png文件
--------------------------------------
希望保存目录结构为
文件名
-img（保存所有转换好的img文件）
-文件名_0
--版式识别结果 （图表）
--版式识别内容 （txt）


"""

import os
import cv2
import numpy as np
import time
from paddleocr import PPStructure, draw_structure_result, save_structure_res
from paddle.utils import try_import
from PIL import Image
import fitz  # PyMuPDF
# import nvml
from pynvml import *

# 初始化NVML
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)

# 初始化表格引擎
table_engine = PPStructure(show_log=True)

def Pdf2Img(pdf_path, dpi=200):
    """将pdf文件转化为PaddleOCR可处理的图片文件并保存在指定路径
    args:
    pdf_path(str): pdf文件路径
    dpi(int): 控制图像分辨率
    """
    images = []
    with fitz.open(pdf_path) as pdf:
        for pg in range(pdf.page_count):
            page = pdf[pg]
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # 转换为指定dpi
            pm = page.get_pixmap(matrix=mat, alpha=False)
            
            # 判断图片，如果大于2000pixels，不放大图片
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            
            # 将PIL图像转换为OpenCV图像格式
            img = Image.frombytes('RGB', [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            images.append((pg, img))  # 返回页码和图像
    return images

def PaddleImages(file_folder, save_folder, font_path, batch_size):
    """
    批量处理图片函数

    args:
    file_folder(str): 文件夹路径
    save_folder(str): 结果路径
    font_path(str): 字体路径
    batch_size(int): 批处理大小
    """
    os.makedirs(save_folder, exist_ok=True)
    process_count = 0  # 处理图片的计数器

    # 获取文件夹中的所有PDF文件
    pdf_files = [os.path.join(file_folder, f) for f in os.listdir(file_folder) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        print(f'Processing PDF: {pdf_file}')
        
        # 获取PDF文件的基本名称
        file_base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        pdf_save_folder = os.path.join(save_folder, file_base_name)
        os.makedirs(pdf_save_folder, exist_ok=True)
        
        # 将PDF文件转换为图像
        images = Pdf2Img(pdf_file)
        print(f'Converted images count: {len(images)}')

        # 保存转换的图像
        for pg, img in images:
            img_path = os.path.join(pdf_save_folder, 'img', f'{file_base_name}_{pg}.png')
            os.makedirs(os.path.join(pdf_save_folder, 'img'), exist_ok=True)
            cv2.imwrite(img_path, img)
        
        # 分批处理图像
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for pg, img in batch:
                start_time = time.time()  # 开始记录处理时间
                
                img_name = f'{file_base_name}_{pg}'
                result_folder = os.path.join(pdf_save_folder, img_name)
                os.makedirs(result_folder, exist_ok=True)
                
                # 识别图像
                result = table_engine(img)
                
                # 记录显存使用情况
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                used_mem = mem_info.used / (1024 * 1024)  # 转换成 MB
                print(f"Used Memory (MB): {used_mem}")
                
                # 保存识别后的数据
                save_structure_res(result, result_folder, img_name)
                
                # 打印结果
                for line in result:
                    line.pop('img', None)
                    print(line)
                
                # 提取并保存图表位置信息
                chart_positions = []
                for item in result:
                    if 'type' in item and item['type'] == 'table':
                        chart_positions.append({
                            'page': pg,
                            'bbox': item['bbox'],
                            'text': item.get('text', '')
                        })
                
                chart_positions_path = os.path.join(result_folder, f'{img_name}_chart_positions.txt')
                with open(chart_positions_path, 'w') as f:
                    for position in chart_positions:
                        f.write(f"Page: {position['page']}, Bounding Box: {position['bbox']}, Text: {position['text']}\n")
                
                # 保存当前页和上下页
                with fitz.open(pdf_file) as pdf:
                    current_page = pdf[pg]
                    prev_page = pdf[pg - 1] if pg > 0 else None
                    next_page = pdf[pg + 1] if pg < pdf.page_count - 1 else None
                    
                    # 保存当前页
                    mat = fitz.Matrix(200 / 72, 200 / 72)  # 转换为指定dpi
                    pm = current_page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                    img.save(os.path.join(result_folder, f'{img_name}_current_page.png'))
                    
                    # 保存上一页
                    if prev_page:
                        pm = prev_page.get_pixmap(matrix=mat, alpha=False)
                        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                        img.save(os.path.join(result_folder, f'{img_name}_prev_page.png'))
                    
                    # 保存下一页
                    if next_page:
                        pm = next_page.get_pixmap(matrix=mat, alpha=False)
                        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                        img.save(os.path.join(result_folder, f'{img_name}_next_page.png'))
                
                # 加载图像并绘制识别结果
                image = Image.open(os.path.join(pdf_save_folder, 'img', f'{file_base_name}_{pg}.png')).convert('RGB')
                im_show = draw_structure_result(image, result, font_path=font_path)
                
                # 保存识别结果的图像
                result_img_path = os.path.join(result_folder, f'{img_name}_result.jpg')
                im_show = Image.fromarray(im_show)
                im_show.save(result_img_path)
                
                process_time = time.time() - start_time  # 结束记录处理时间
                
                process_count += 1
                print(f'Processed and saved: {result_img_path}, Time taken: {process_time:.2f}s')
    
    print("Batch processing complete.")
    print(f'Total processed number is: {process_count}')

if __name__ == "__main__":
    file_folder = './constructor/source/others/'
    save_folder = './constructor/result_batch/others/'
    font_path = 'doc/fonts/simfang.ttf'  # 字体位置
    batch_size = 10  # 根据显存调整批次大小

    PaddleImages(file_folder, save_folder, font_path, batch_size)





# # 成功的批量处理文件夹下的PDFs==========begin===========
# #初始化NVML
# nvml.nvmlInit()
# handle=nvml.nvmlDeviceGetHandleByIndex(0)

# # 初始化 table_engine
# table_engine = PPStructure(show_log=True)

# def Pdf2Img(pdf_path, dpi=200):
#     """将pdf文件转化为PaddleOCR可处理的图片文件并保存在指定路径
#     args:
#     pdf_path(str): pdf文件路径
#     dpi(int): 控制图像分辨率    """

#     images=[]
    
#     with fitz.open(pdf_path) as pdf:
#         for pg in range(pdf.page_count):
#             page=pdf[pg]
#             mat=fitz.Matrix(dpi/72,dpi/72) # 转换为指定dpi
#             pm=page.get_pixmap(matrix=mat,alpha=False)
            
#             #判断图片，如果大于2000pixels，不放大图片
#             if pm.width>2000 or pm.height >2000:
#                 pm=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
            
#             #将PIL图像转换为OpenCV图像格式
#             img=Image.frombytes('RGB',[pm.width,pm.height],pm.samples)
#             img=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             images.append(img)
#     return images

# def PaddleImages(file_folder, save_folder, font_path, batch_size):
#     """
#     批量处理图片函数

#     args:
#     file_folder(str): 文件夹路径
#     save_folder(str): 结果路径
#     font_path(str): 字体路径
#     batch_size(int): 批处理大小
#     """
#     os.makedirs(save_folder, exist_ok=True)
#     process_count = 0  # 处理图片的计数器

#     # 获取文件夹中的所有PDF文件
#     pdf_files = [os.path.join(file_folder, f) for f in os.listdir(file_folder) if f.lower().endswith('.pdf')]

#     for pdf_file in pdf_files:
#         print(f'Processing PDF: {pdf_file}')
        
#         # 获取PDF文件的基本名称
#         file_base_name = os.path.splitext(os.path.basename(pdf_file))[0]
#         pdf_save_folder = os.path.join(save_folder, file_base_name)
#         os.makedirs(pdf_save_folder, exist_ok=True)
        
#         # 将PDF文件转换为图像
#         images = Pdf2Img(pdf_file)
#         print(f'Converted images count: {len(images)}')

#         img_paths = [os.path.join(pdf_save_folder, f'{file_base_name}_{i}.png') for i in range(len(images))]
        
#         # 保存转换的图像
#         for idx, img in enumerate(images):
#             cv2.imwrite(img_paths[idx], img)
        
#         # 分批处理图像
#         for i in range(0, len(img_paths), batch_size):
#             batch = img_paths[i:i + batch_size]
            
#             for img_path in batch:
#                 start_time = time.time()  # 开始记录处理时间
                
#                 img_name = os.path.basename(img_path)
#                 base_filename = os.path.splitext(img_name)[0]
#                 result_folder = os.path.join(pdf_save_folder, base_filename)
#                 os.makedirs(result_folder, exist_ok=True)
                
#                 # 读取图片
#                 img = cv2.imread(img_path)
                
#                 # 识别图像
#                 result = table_engine(img)
                
#                 # 记录显存使用情况
#                 mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
#                 used_mem = mem_info.used / (1024 * 1024)  # 转换成 MB
#                 print(f"Used Memory (MB): {used_mem}")
                
#                 # 保存识别后的数据
#                 save_structure_res(result, result_folder, base_filename)
                
#                 # 打印结果
#                 for line in result:
#                     line.pop('img', None)
#                     print(line)
                
#                 # 加载图像并绘制识别结果
#                 image = Image.open(img_path).convert('RGB')
#                 im_show = draw_structure_result(image, result, font_path=font_path)
                
#                 # 保存识别结果的图像
#                 result_img_path = os.path.join(result_folder, f'{base_filename}_result.jpg')
#                 im_show = Image.fromarray(im_show)
#                 im_show.save(result_img_path)
                
#                 process_time = time.time() - start_time  # 结束记录处理时间
                
#                 process_count += 1
#                 print(f'Processed and saved: {result_img_path}, Time taken: {process_time:.2f}s')
    
#     print("Batch processing complete.")
#     print(f'Total processed number is: {process_count}')

# # 成功的批量处理文件夹下的PDFs==========end===========


# def PaddleImages(file_path, save_folder,  font_path, batch_size):
#     """
#     批量处理图片函数

#     args:
#     img_folder(str): 图片路径
#     save_folder(str): 结果路径
#     result_folder(str): 结果文件夹路径
#     font_path(str): 字体路径
#     batch_size(int): 批处理大小

#     """
#     os.makedirs(save_folder, exist_ok=True)
#     # os.makedirs(result_folder, exist_ok=True)

#     process_count = 0  # 处理图片的计数器



#     #判断是否是pdf
#     if file_path.lower().endswith('.pdf'):
#         images=Pdf2Img(file_path)
#         print(f'Converted images count:{len(images)}')


#         file_base_name=os.path.splitext(os.path.basename(file_path))[0]
#         img_paths=[os.path.join(save_folder,f'{file_base_name}_{i}.png')
#                    for i in range(len(images))]
        
#         #保存转换的图像
#         for idx, img in enumerate(images):
#             cv2.imwrite(img_paths[idx],img)
#     else:
#         img_paths = [file_path] if os.path.isfile(file_path) else [os.path.join(file_path, img_name) for img_name in os.listdir(file_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]


#     # 分批处理图片
#     for i in range(0, len(img_paths), batch_size):
#         batch = img_paths[i:i + batch_size]
        
#         for img_path in batch:
#             start_time = time.time()  # 开始记录处理时间
            
#             img_name = os.path.basename(img_path)
#             base_filename=os.path.splitext(img_name)[0]
#             result_folder=os.path.join(save_folder,base_filename)
#             os.makedirs(result_folder,exist_ok=True)
            
#             # 读取图片
#             img = cv2.imread(img_path)
            
#             # 识别图像
#             result = table_engine(img)
            
#             # 记录显存使用情况
#             mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
#             used_mem = mem_info.used / (1024 * 1024)  # 转换成 MB
#             print(f"Used Memory (MB): {used_mem}")
            
#             # 保存识别后的数据
#             save_structure_res(result, result_folder, base_filename)
            
#             # 打印结果
#             for line in result:
#                 line.pop('img', None)
#                 print(line)
            
#             # 加载图像并绘制识别结果
#             image = Image.open(img_path).convert('RGB')
#             im_show = draw_structure_result(image, result, font_path=font_path)
            
           
#             # 保存识别结果的图像
#             # os.makedirs(result_img_path,exist_ok=True) # 结果路径不存在则创建
#             result_img_path = os.path.join(result_folder, f'{base_filename}_result.jpg')
            
#             # 去重和保存结果
#             im_show = Image.fromarray(im_show)
#             im_show.save(result_img_path)
            
#             process_time = time.time() - start_time  # 结束记录处理时间
            
#             process_count += 1
#             print(f'Processed and saved: {result_img_path}, Time taken: {process_time:.2f}s')
    
#     print("Batch processing complete.")
#     print(f'Total processed number is: {process_count}')

if __name__ == "__main__":

    file_folder = './constructor/source/medical/'
    # moe/PaddleOCR_m/constructor/source/medical
    save_folder= './constructor/result_single/medical/'

    # img_folder = 'data/ChartQA_Dataset/train/png'
    # save_folder = './output_train'
    # result_folder = './result_train'
    
    # file_path='./source/medical/'
    # file_path='./constructor/source/caict/caict白皮书合集/'
    # moe/PaddleOCR_m/constructor/source/caict/caict白皮书合集
    # PaddleOCR/constructor/source/medical/2021中国医疗AI行业研究报告.pdf
    # PaddleOCR/constructor/source/medical/2024全球医疗行业展望deloitte.pdf
    # save_folder='./constructor/result_single/caict/caict白皮书合集/'
    # result_folder='.constructor/result/medical_result/2021中国医疗AI行业研究报告'
    
    # file_path='./source/ChartQA Dataset/test/png/'
    # /home/ubuntu/moe/PaddleOCR/constructor/source/ChartQADataset/test/png
    # save_folder='./result/ChartQA_Dataset_res/test'
    # result_folder='./result/ChartQA_Dataset_result/test'
    
    font_path = 'doc/fonts/simfang.ttf' # 字体位置
    batch_size = 10  # 根据显存调整批次大小

   
    PaddleImages(file_folder , save_folder, font_path, batch_size)















# ####版面分析+表格识别
# import os
# import cv2
# from paddleocr import PPStructure,draw_structure_result,save_structure_res

# from PIL import Image
# # 批量处理，走多进程

# #初始化table_engine
# table_engine = PPStructure(show_log=True)

# #路径
# img_folder='data/ChartQA_Dataset/train/train_mini'

# save_folder = './output'
# result_folder='./result'
# # img_path = 'data/ChartQA_Dataset/train/png/233.png'


# os.makedirs(save_folder,exist_ok=True)
# os.makedirs(save_folder,exist_ok=True)

# process_count=0

# #定义结果的路径
# font_path = 'doc/fonts/simfang.ttf' # PaddleOCR下提供字体包

# for img_name in os.listdir(img_folder):
#     if img_name.endswith(('.png','.jpg','.jpeg')):
#      img_path=os.path.join(img_folder,img_name)
#      #读取图片
#      img = cv2.imread(img_path)
     
#      #开始识别
#      result = table_engine(img)
     
#      base_filename=os.path.basename(img_path).split('.')[0]
#      save_structure_res(result, save_folder,base_filename)
     
#      # 打印结果
#      for line in result:
#         line.pop('img',None)
#         print(line)
    
#     #加载图像并识别
#     image = Image.open(img_path).convert('RGB')
#     im_show = draw_structure_result(image, result,font_path=font_path)
    
#     #保存结果
#     result_img_path=os.path.join(result_folder,f'{os.path.splitext(img_name)[0]}_result.jpg')
    
#     #去重
#     im_show = Image.fromarray(im_show)
#     # reslut_img_dir=os.path.dirname(result_img_path)
#     # os.makedirs(reslut_img_dir,exist_ok=True)
#     im_show.save(result_img_path)
    
#     # print(f'Processed and saved:{img_name}')
#     process_count+=1
#     print(f'Procesed and saved :{result_img_path}')
    
# print("Batch processing complete.")
# print('Total process number is:',{process_count})


# # 版面分析
# import os
# import cv2
# from paddleocr import PPStructure,save_structure_res

# table_engine = PPStructure(table=False, ocr=False, show_log=True)

# save_folder = './output'
# img_path = 'ppstructure/docs/table/1.png'
# img = cv2.imread(img_path)
# result = table_engine(img)
# save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

# for line in result:
#     line.pop('img')
#     print(line)



# import os
# import cv2
# from paddleocr import PPStructure,save_structure_res

# ocr_engine = PPStructure(table=False, ocr=True, show_log=True)

# save_folder = './output'
# img_path = 'ppstructure/docs/recovery/UnrealText.pdf'
# result = ocr_engine(img_path)
# for index, res in enumerate(result):
#     save_structure_res(res, save_folder, os.path.basename(img_path).split('.')[0], index)

# for res in result:
#     for line in res:
#         line.pop('img')
#         print(line)


# import os
# import cv2
# import numpy as np
# from paddleocr import PPStructure,save_structure_res
# from paddle.utils import try_import
# from PIL import Image

# ocr_engine = PPStructure(table=False, ocr=True, show_log=True)

# save_folder = './output'
# img_path = 'ppstructure/docs/recovery/UnrealText.pdf'

# fitz = try_import("fitz")
# imgs = []
# with fitz.open(img_path) as pdf:
#     for pg in range(0, pdf.page_count):
#         page = pdf[pg]
#         mat = fitz.Matrix(2, 2)
#         pm = page.get_pixmap(matrix=mat, alpha=False)

#         # if width or height > 2000 pixels, don't enlarge the image
#         if pm.width > 2000 or pm.height > 2000:
#             pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

#         img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         imgs.append(img)

# for index, img in enumerate(imgs):
#     result = ocr_engine(img)
#     save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0], index)
#     for line in result:
#         line.pop('img')
#         print(line)