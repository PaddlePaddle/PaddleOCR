import os
import cv2
import numpy as np
import logging
import datetime
from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
from PIL import Image

# 將 PDF 轉換為圖像
def pdf_to_image(pdf_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(pdf_folder):
        pages = convert_from_path(
            os.path.join(pdf_folder, filename),
            dpi=300,
            grayscale=True
        )

        for i, page in enumerate(pages):
            output_path = os.path.join(output_folder, f'{filename}_page_{i+1}.jpg')
            page.save(output_path, 'JPEG')

def preprocess_image(image):
    # 將圖像轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 將灰階圖像進行二值化處理，將黑色設置為255（白色），將其他顏色設置為0（黑色）
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

    # 將二值化圖像轉換為三通道圖像
    binary_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 過濾原始彩色圖像，僅保留黑色區域
    filtered_image = cv2.bitwise_and(image, binary_image)

    # 將其他區域轉換為白色
    filtered_image[np.where((filtered_image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    return filtered_image


# 在進行 OCR 之前預處理圖像
def preprocess_ocr_images(img_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(img_folder):
        img_path = os.path.join(img_folder, filename)

        # 讀取彩色圖像
        image = cv2.imread(img_path)

        # 預處理圖像
        preprocessed_image = preprocess_image(image)

        # 儲存預處理後的圖像
        output_path = os.path.join(output_folder, f'{filename}')
        cv2.imwrite(output_path, preprocessed_image)


# 在進行 OCR 之前的預處理步驟
def preprocess_before_ocr(img_folder, output_folder):
    # 將圖像預處理，僅保留黑色區域
    preprocess_ocr_images(img_folder, output_folder)

    # 進行 OCR
    perform_ocr(output_folder, outputlog_folder, outputimg_folder, outputfile_folder)



# 進行 OCR
def perform_ocr(img_folder, outputlog_folder, pdfoutputimg_folder, outputimg_folder, outputfile_folder):
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 創建 log 檔案處理器
    log_filename = datetime.datetime.now().strftime("%Y-%m-%d.log")
    log_path = os.path.join(outputlog_folder, log_filename)

    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.ERROR)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.error('-------------log start------------------')

    # 進行 OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    for filename in os.listdir(pdfoutputimg_folder):
        img_path = os.path.join(pdfoutputimg_folder, filename)

        # 預處理圖像，僅保留黑色區域
        processed_image = preprocess_image(img_path)

        result = ocr.ocr(processed_image, cls=True)

        logger.error(filename)
        logger.error('--------------filestart-----------------')

        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)
                print_obj = str(line)
                logger.error(print_obj)
        logger.error('--------------fileend----------------')

        # 繪製 OCR 結果
        result = result[0]
        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save(os.path.join(outputimg_folder, f'{filename}_result.jpg'))

        # 將結果儲存到檔案
        righttop_location = [topright[2] for topright in boxes]
        righttop_order = sorted(righttop_location, reverse=True)

        output_path = os.path.join(outputfile_folder, f'{filename}.txt')
        with open(output_path, 'w') as f:
            for righttop in righttop_order:
                for line in result:
                    if line[0][2] == righttop:
                        f.write(line[1][0]+'\n')
                        break



def main():
    pdf_folder = './pdf/'
    img_folder = './img'
    pdfoutputimg_folder = './pdftoimg'
    outputfile_folder = './outputfile'
    outputlog_folder = './outputlog'
    outputimg_folder = './outputimg'

    # 將 PDF 轉換為圖像
    pdf_to_image(pdf_folder, pdfoutputimg_folder)

    # 在進行 OCR 之前的預處理步驟
    preprocess_before_ocr(img_folder, outputimg_folder, outputfile_folder)

    # 進行 OCR
    perform_ocr(img_folder, outputlog_folder, pdfoutputimg_folder, outputimg_folder, outputfile_folder)

if __name__ == '__main__':
    main()

