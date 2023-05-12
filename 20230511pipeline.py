from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
import os, cv2, numpy as np
import logging
import datetime

pdf_folder = './pdf/'
img_folder = './img'
pdfoutputimg_folder = './pdftoimg'
outputfile_folder = './outputfile'
outputlog_folder = './outputlog'
outputimg_folder = './outputimg'

file_names = os.listdir(img_folder)

if not os.path.exists(outputimg_folder):
    os.makedirs(outputimg_folder)

if not os.path.exists(outputfile_folder):
    os.makedirs(outputfile_folder)


# 將 PDF 轉換為圖像並獲取寬度和高度
def pdf_to_image_with_size(pdf_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(pdf_folder):
        pages = convert_from_path(
            os.path.join(pdf_folder, filename),
            dpi=300,
            grayscale=False
        )

        for i, page in enumerate(pages):
            output_path = os.path.join(output_folder, f'{filename}_page_{i+1}.jpg')
            page.save(output_path, 'JPEG')

            width, height = page.size
            print(f'{filename}_page_{i+1} width: {width}, height: {height}')


# 預處理圖像，僅保留黑色區域
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result = cv2.bitwise_and(image, image, mask=threshold)
    result[threshold == 0] = 255  # 將其餘部分轉換為白色
    return result


# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# create logFileHandler
log_filename = datetime.datetime.now().strftime("%Y-%m-%d.log")  # -%M-%S
log_path = os.path.join(outputlog_folder, log_filename)

# 指定log路径和檔名
log_dir = os.path.dirname(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

file_handler = logging.FileHandler(log_path)  # log_path)

file_handler.setLevel(logging.ERROR)
# 將 FileHandler 添加到 logger
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.error('-------------log start------------------')


# do ocr
def perform_ocr(img_folder, outputlog_folder, pdfoutputimg_folder, outputimg_folder, outputfile_folder):
    for filename in os.listdir(pdfoutputimg_folder):
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')

        img_path = os.path.join(pdfoutputimg_folder, filename)
        image = cv2.imread(img_path)

        # 預處理圖像
        processed_image = preprocess_image(image)

        # 
        # 偵測圖像顯示的黑色區塊
        contours, _ = cv2.findContours(cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 只保留黑色區塊，其餘轉為白色
        mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            mask[y:y+h, x:x+w] = 255

        processed_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)

        # # 執行 OCR
        # result = ocr.ocr(processed_image, cls=True)

        # logger.error(filename)
        # logger.error('--------------filestart-----------------')

        # for idx in range(len(result)):
        #     res = result[idx]
        #     for line in res:
        #         print(line)
        #         # record log
        #         print_obj = str(line)
        #         logger.error(print_obj)
        # logger.error('---------------fileend----------------')

        # 執行 OCR
        result = ocr.ocr(processed_image, cls=True)

        logger.error(filename)
        logger.error('--------------filestart-----------------')

        for idx, res in enumerate(result):
            if len(res) > 0:
                for line in res:
                    print(line)
                    # record log
                    print_obj = str(line)
                    logger.error(print_obj)
            else:
                print(f'No result found for index {idx} in {filename}')
                logger.error(f'No result found for index {idx} in {filename}')

        logger.error('---------------fileend----------------')


        # 繪製結果
        result = result[0]
        im_show = draw_ocr(processed_image, [line[0] for line in result], [line[1][0] for line in result], [line[1][1] for line in result])
        im_show = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(outputimg_folder, f'{filename}_result.jpg'), im_show)

        # 寫入檔案
        righttop_location = [topright[2] for topright in result]
        righttop_order = [sorted(righttop_location, reverse=True)]

        output_path = os.path.join(outputfile_folder, f'{filename}.txt')
        with open(output_path, 'w') as f:
            for righttop in righttop_order[0]:
                for line in result:
                    if line[0][2] == righttop:
                        f.write(line[1][0] + '\n')
                        break
                else:
                    continue

# 主函式
def main():
    pdf_to_image_with_size(pdf_folder, pdfoutputimg_folder)
    perform_ocr(img_folder, outputlog_folder, pdfoutputimg_folder, outputimg_folder, outputfile_folder)

# 執行主函式
if __name__ == '__main__':
    main()

