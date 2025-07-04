# 本临时脚本用于对1105的batch_paddle_images.py中出现的错误
# AttributeError: 'Image' object has no attribute 'shape'. Did you mean: 'save'?
# 回退到上一版本进行调试


# --------------baidu开发者中心-----------begin
# 下面是示例代码，由于官网教程更新过快，找不到原来的，所以用了另一版
# https://developer.baidu.com/article/details/2792650


# # # # 打开PDF文件，批量处理文件------begin
# # # 将pdf的每一页转化为图像文件并保存到指定路径
import fitz  # PyMuPDF库, pip install PyMuPDF 
from PIL import Image
import paddleocr
import os
import glob
from paddleocr import PaddleOCR, draw_ocr

"""以下是将pdf转化为图片的代码，可以批量对文件夹中的文件进行处理
成功版本
"""
# 初始化PaddleOCR模型
# ocr = paddleocr.PaddleOCR(use_gpu=True)

# # 获得PDF文件夹路径
# pdf_folder='./constructor/source/caict/caict白皮书合集/'
# # PaddleOCR/constructor/source/caict/caict白皮书合集
# # 获取该目录下所有pdf后缀的文件
# pdf_files = glob.glob(os.path.join(pdf_folder, '*.pdf'))

# # 遍历所有pdf文件
# for pdf_path in pdf_files:
#     # 打开pdf
#     pdf_doc=fitz.open(pdf_path)

#     # 获取不包括扩展名的文件名称
#     pdf_basename=os.path.splitext(os.path.basename(pdf_path))[0]
    
#     # pdf2image保存路径
#     convert_img_folder =f'./constructor/result/pdf2img/caict/caict白皮书合集/{pdf_basename}/'
#     os.makedirs(convert_img_folder, exist_ok=True)

#     # 遍历pdf中的每一页
#     for page_index in range(len(pdf_doc)):
#         page = pdf_doc[page_index]

#         # 将PDF页面转换为图像
#         # img = page.get_image(zoom=0.2, scale=1.5)  # 根据需要调整zoom和scale参数
#         pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 调整缩放比例
#         # 缩放比例说明，fitz.Matrix中的参数取决于如下因素：
#             # 输出图像的清晰度（更高的缩放比例生成更高的图像，增加处理时间）、
#             # 目标用途（高缩放比例适合较高分辨率），
#             # 同时也需要考虑性能和设备
        
#         # 构建保存的图像路径和文件名
#         img_filename=f'{pdf_basename}_page_{page_index+1}.jpg'
#         img_path=os.path.join(convert_img_folder, img_filename)

        
#         # 使用PIL库保存图像
#         img = Image.frombytes('RGB',[pix.width,pix.height], pix.samples)
#         img.save(img_path)
#         print(f'Image saved to {img_path}')
#     print(f'All pages of {pdf_basename} have been converted to image and saved to path {convert_img_folder}')
# # 打开PDF文件，批量处理文件------end

"""以上是将pdf转化为图片的代码，可以批量对文件夹中的文件进行处理
成功版本 end
"""

"""将单个文件夹下的图片用ppocr批量处理"""
import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

def process_images(input_dir, output_dir):
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入目录下的所有图片文件
    for file in os.listdir(input_dir):
        type_tuple=('png', 'jpg', 'jpeg')

        if file.lower().endswith(type_tuple):
            # 构建输入文件路径
            img_path = os.path.join(input_dir, file)
            
            # 文字识别
            result = ocr.ocr(img_path, cls=True)
            
            # 处理识别结果
            image = Image.open(img_path).convert('RGB')
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result] # 提取的是float

            
            # 绘制OCR结果
            im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/simfang.ttf')
            im_show = Image.fromarray(im_show)
            
            # 保存处理后的图片
            output_path = os.path.join(output_dir, file)
            im_show.save(output_path) 
            
            print(f"处理完成并保存至: {output_path}")

# 指定输入和输出目录
input_dir = './constructor/result/pdf2img/mckinsey/2023麦肯锡中国消费者报告/'
# PaddleOCR/constructor/result/pdf2img/mckinsey/2023麦肯锡中国消费者报告
output_dir = './constructor/result/tmp_pp_result/'

# 开始处理
process_images(input_dir, output_dir)


# # 打开PDF文件，处理单个文件------begin
# pdf_path = './constructor/source/McKinsey/2023麦肯锡中国消费者报告.pdf'
# pdf_doc = fitz.open(pdf_path)

# # 获取不包括扩展名的文件名称
# pdf_basename=os.path.splitext(os.path.basename(pdf_path))[0]

# # 保存后的pdf2img文件路径
# convert_img_folder =f'./constructor/result/pdf2img/mckinsey/{pdf_basename}/'

# # 路径不存在则创建
# if not os.path.exists(convert_img_folder):
#     os.makedirs(convert_img_folder, exist_ok=True)

# # 遍历PDF中的每一页
# for page_index in range(len(pdf_doc)):
#     page = pdf_doc[page_index]

#     # 将PDF页面转换为图像
#     # img = page.get_image(zoom=0.2, scale=1.5)  # 根据需要调整zoom和scale参数
#     pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 调整缩放比例
    
#     # 构建保存的图像路径和文件名
#     img_filename=f'{pdf_basename}_page_{page_index+1}.jpg'
#     img_path=os.path.join(convert_img_folder, img_filename)

#     # img_path = f'{pdf_path}_{page_index}.jpg'  # 保存图像的路径和文件名
    
#     # 使用PIL库保存图像
#     img = Image.frombytes('RGB',[pix.width,pix.height], pix.samples)
#     img.save(img_path)
#     print(f'Image saved to {img_path}')
# # 打开PDF文件，处理单个文件------end

# # 先运行上面的脚本保存到指定位置


#  --------------baidu开发者中心-----------end

# 以下为官方教程内容
# ---------------官方教程------------begin
# https://paddlepaddle.github.io/PaddleOCR/v2.9/ppocr/blog/whl.html#21
# paddleocr whl包会自动下载ppocr轻量级模型作为默认模型

# 读取图像文件
# img_path = './constructor/result/pdf2img/mckinsey/2023麦肯锡中国消费者报告/2023麦肯锡中国消费者报告_page_4.jpg'  # 替换为实际的图像文件路径

# 使用PaddleOCR对单个图像进行文本识别
# import cv2

# from paddleocr import PaddleOCR, draw_ocr

# # Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# # 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
# ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
# # img_path = 'PaddleOCR/doc/imgs/11.jpg'
# # 读取图像文件
# img_path = './constructor/result/pdf2img/mckinsey/2023麦肯锡中国消费者报告/2023麦肯锡中国消费者报告_page_4.jpg' 

# output_dir = './constructor/result/tmp_result/'
# os.makedirs(output_dir, exist_ok=True)

# result = ocr.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)

# # 显示结果
# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]

# # 绘制OCR结果
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)

# # 保存图片到指定路径
# output_path=os.path.join(output_dir, 'result.jpg')
# im_show.save(output_path)
# print(f"结果保存在:{output_path}")

# ---------------官方教程------------end

# from paddleocr import PaddleOCR, draw_ocr
# # 初始化模型
# ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

# # 根文件夹路径
# root_folder='./constructor/result/pdf2img/mckinsey/'

# # 处理后的根文件夹路径
# save_root_folder='./constructor/result/tmp_pp_result/'
# os.makedirs(save_root_folder, exist_ok=True)

# def ImageProcessor(img_path, save_folder):
#     # 读取图像文件
#     result = ocr.ocr(img_path, cls=True)
    
#     # 显示结果
#     image = Image.open(img_path).convert('RGB')
#     boxes = [line[0] for line in result]
#     txts = [line[1][0] for line in result]
#     scores = [line[1][1] for line in result]
#     im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/simfang.ttf')
#     im_show = Image.fromarray(im_show)
    
#     # 构建保存的图像路径
#     img_basename = os.path.splitext(os.path.basename(img_path))[0]
#     ocr_result_path = os.path.join(save_folder, f'{img_basename}_ocr_result.jpg')
    
#     # 保存OCR结果图像
#     im_show.save(ocr_result_path)
#     print(f'OCR result saved to {ocr_result_path}')

# sub_folders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

# for sub_folder in sub_folders:
#     # 获取子文件夹名称
#     folder_name = os.path.basename(sub_folder)
    
#     # 获取该子文件夹下的所有图像文件
#     img_files = glob.glob(os.path.join(sub_folder, '*.jpg'))
    
#     # 创建保存处理后图像的文件夹
#     save_folder = os.path.join(save_root_folder, folder_name)
#     os.makedirs(save_folder, exist_ok=True)
    
#     # 遍历每个图像文件
#     for img_file in img_files:
#         ImageProcessor(img_file, save_folder)
    