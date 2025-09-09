import os
import cv2
import numpy as np
import time
from paddleocr import PPStructure, draw_structure_result, save_structure_res
from paddle.utils import try_import
from PIL import Image
import fitz  # PyMuPDF
from pynvml import *
import json
from ppstructure.table.predict_table import to_excel
from copy import deepcopy

# 该脚本用于批量处理pdf文件，先转化为img然后用pp处理
# 给出ppstructure识别结果，毗邻页面以及ocr识别结果
# --------1120 begin
# 有子图及识别结果（表现不太好）会丢失，需要再做处理
# 没有识别出表格
# 以上问题都是没有使用自带的save_structure_res导致的
# 使用了该函数就无法保存前后页以及筛选后的txt内容
# --------1120 --end solved


# 初始化NVML
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)

# 初始化表格引擎
process_engine = PPStructure(show_log=True)

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



def SaveStructureRes(result, result_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(result_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    # res_cp = result.copy()
    res_cp= deepcopy(result)
    
    # save res
    with open(
        os.path.join(excel_save_folder, "res_{}.txt".format(img_idx)),
        "w",
        encoding="utf8",
    ) as f:
        for region in res_cp:
            roi_img = region.pop("img", None)
            f.write("{}\n".format(json.dumps(region)))

            if (
                region["type"].lower() == "table"
                and len(region["res"]) > 0
                and "html" in region["res"]
            ):
                excel_path = os.path.join(
                    excel_save_folder, "{}_{}.xlsx".format(region["bbox"], img_idx)
                )
                to_excel(region["res"]["html"], excel_path)
            elif region["type"].lower() == "figure":
                img_path = os.path.join(
                    excel_save_folder, "{}_{}.jpg".format(region["bbox"], img_idx)
                )
                if roi_img is not None:
                    cv2.imwrite(img_path, roi_img)

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
                result = process_engine(img)
                
                # 记录显存使用情况
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                used_mem = mem_info.used / (1024 * 1024)  # 转换成 MB
                print(f"Used Memory (MB): {used_mem}")
                
                # 保存识别后的数据
                SaveStructureRes(result, result_folder, img_name)
                
                # 打印结果
                for line in result:
                    line.pop('img', None)
                    print(line)
                
                # 提取并保存图表位置信息
                chart_positions = []   
                save_type = ['table', 'figure', 'figure_caption'] # 增加了需要保存的类型，将需要的图像提取出来 
                for item in result:
                    if item['type'] in save_type:
                        chart_positions.append({
                            'page':pg,
                            'type':item['type'],
                            'bbox':item['bbox'],
                            'text':item.get('res', [])
                        })
                # for item in result:
                #     if 'type' in item and item['type'] == 'table':
                #         chart_positions.append({
                #             'page': pg,
                #             'bbox': item['bbox'],
                #             'text': item.get('text', '')
                #         })

                # 打印图表位置信息
                print(f"Chart Positions for Page {pg}:{chart_positions}")
                
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

def ProcessSingelPDF(pdf_path, save_folder, font_path, batch_size):
    """
    单个PDF处理函数, 当输入为单个PDF文件时使用
    args:
    pdf_path(str): PDF文件路径
    save_folder(str): 识别结果保存路径
    font_path(str): 字体路径
    batch_size(int): 批处理大小
    """
    file_folder=os.path.dirname(pdf_path)
    PaddleImages(file_folder, save_folder, font_path, batch_size)

if __name__ == "__main__":
    # file_folder = './constructor/source/medical/'
    # moe/PaddleOCR_m/constructor/source/medical

    # pdf_path='./constructor/source/McKinsey/'
    pdf_path='./constructor/source/others/'
    # moe/PaddleOCR_m/constructor/source/medical/2021中国医疗AI行业研究报告.pdf
    # moe/PaddleOCR_m/constructor/source/others/艾瑞咨询-中国医疗科技行业研究报告-221212.pdf

    save_folder = './constructor/result/1120result/others/'
    font_path = 'doc/fonts/simfang.ttf'  # 字体位置
    batch_size = 50  # 根据显存调整批次大小

    # PaddleImages(file_folder, save_folder, font_path, batch_size)
    ProcessSingelPDF(pdf_path, save_folder, font_path, batch_size)






