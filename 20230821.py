from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
import os, cv2, shutil ,numpy as np
import logging
import datetime
from PIL import Image

pdf_folder = './pdf/'
img_folder = './img'

pdfoutputimg_folder_cover = './pdftoimg_cover'
pdfoutputimg_folder_note = './pdftoimg_note'
pdfoutputimg_folder_main = './pdftoimg_main'
pdfoutputimg_folder_appendix = './pdftoimg_appendix'

outputfile_folder = './outputfile'
outputlog_folder = './outputlog'
outputimg_folder = './outputimg'

# 將來源資料夾中的符合條件的檔案重新命名並移動到目標資料夾
def rename_and_move_files(src_folder, dst_folder, compare_file_end, compare_string, replace_string):

    for file in os.listdir(src_folder):
        file_extension = os.path.splitext(file)[1]
        file_name = os.path.splitext(file)[0]

        if src_folder == pdfoutputimg_folder_cover and not file.endswith(compare_file_end):
            if 'cover' in file_name:
                for i in range(1, 53):
                    compare_page_string = f"_page_{i}_cover"
                    if i <= 8:
                        # 執行處理 page1-8 的檔案
                        print(f"Processing file: {file}")
                        process_file(file, src_folder, dst_folder, compare_string, replace_string)
            else:
                # 不處理 page9-10 的檔案
                print(f"Skipping file: {file}")

        elif src_folder == pdfoutputimg_folder_main and file.endswith(compare_file_end):
            process_file(file, src_folder, dst_folder, compare_string, replace_string)


# 進行檔名比對處理
def process_file(file, src_folder, dst_folder, compare_string, replace_string):
    # 取得副檔名
    file_extension = os.path.splitext(file)[1]
    file_name = os.path.splitext(file)[0]

    # 如果檔名有包比字串，則進行檔名部份更換的處裡
    new_file_name = file_name.replace(compare_string, replace_string) + file_extension
    new_file_path = os.path.join(dst_folder, new_file_name)

    # 確認移動位置該檔案是否已存在，若不存在再進行檔案搬移
    if not os.path.exists(new_file_path):
        old_file_path = os.path.join(src_folder, file)
        shutil.move(old_file_path, new_file_path)


# 向內移動movePixel個 再往外移動 碰到紅線角則停止
def walkFind(image, red_mask, x, y, w, h, movePixel=100):
    x1 = x + movePixel
    x2 = x + w - movePixel
    x3 = x + movePixel
    x4 = x + w - movePixel
    y1 = y + movePixel
    y2 = y + movePixel
    y3 = y + h - movePixel
    y4 = y + h - movePixel

    # cv2.circle(image, (x1, y1), 10, (255, 0, 0), 5)
    # cv2.circle(image, (x2, y2), 10, (255, 0, 0), 5)
    # cv2.circle(image, (x3, y3), 10, (255, 0, 0), 5)
    # cv2.circle(image, (x4, y4), 10, (255, 0, 0), 5)

    #左上
    conti = 1
    while(conti):
        if red_mask[y1-1][x1-1] == 0 and y1 >= y and x1 >= x:
            y1 -= 1
            x1 -= 1
        elif red_mask[y1][x1-1] == 0 and y1 >= y and x1 >= x:
            x1 -= 1
        elif red_mask[y1-1][x1] == 0 and y1 >= y and x1 >= x:
            y1 -= 1
        else:
            conti = 0

    #右上

    conti = 1
    while(conti):
        if red_mask[y2-1][x2+1] == 0 and y2 >= y and x2 <= x + w:
            y2 -= 1
            x2 += 1
        elif red_mask[y2][x2+1] == 0 and y2 >= y and x2 <= x + w:
            x2 += 1
        elif red_mask[y2-1][x2] == 0 and y2 >= y and x2 <= x + w:
            y2 -= 1
        else:
            conti = 0
    
    #左下
    conti = 1
    while(conti):
        if red_mask[y3+1][x3-1] == 0 and y3 <= y + h and x3 >= x:
            y3 += 1
            x3 -= 1
        elif red_mask[y3][x3-1] == 0 and y3 <= y + h and x3 >= x:
            x3 -= 1
        elif red_mask[y3+1][x3] == 0 and y3 <= y + h and x3 >= x:
            y3 += 1
        else:
            conti = 0

    #右下
    conti = 1
    while(conti):
        if red_mask[y4+1][x4+1] == 0 and y4 <= y + h and x4 <= x + w:
            y4 += 1
            x4 += 1
        elif red_mask[y4][x4+1] == 0 and y4 <= y + h and x4 <= x + w:
            x4 += 1
        elif red_mask[y4+1][x4] == 0 and y4 <= y + h and x4 <= x + w:
            y4 += 1
        else:
            conti = 0

    x1 = min(x1, x3)
    y1 = min(y1, y2)
    x4 = max(x2, x4)
    y4 = max(y4, y3)

    return x1, y1, x4-x1, y4-y1

# 剪切圖片
def crop_img(img_path):

    # 讀取圖片
    ori_img = cv2.imread(img_path)

    if ori_img is None or ori_img.size == 0:
        return None

    # 檢查圖像尺寸
    if ori_img.shape[0] == 0 or ori_img.shape[1] == 0:
        return None

    # 將影像轉換為HSV顏色空間
    hsv_image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)

    # 定義紅色的HSV閾值範圍
    lower_red = np.array([30, 40, 200])  # 下限閾值
    upper_red = np.array([90, 100, 255]) # 上限閾值

    # 使用inRange函式根據閾值範圍創建遮罩
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到最大的輪廓
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 若有找到最大的方框
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # 切割最大的方框區域，否則使用原圖
    if max_contour is not None:
        # x, y, w, h = cv2.boundingRect(max_contour)
        x, y, w, h = walkFind(ori_img, red_mask, x, y, w, h)
        max_box_region = ori_img[y:y+h, x:x+w].copy()
        if(max_box_region.size > 1500000):
            ori_img = max_box_region

    return ori_img

def create_folders():
    if not os.path.exists(outputimg_folder):
        os.makedirs(outputimg_folder)

    if not os.path.exists(outputfile_folder):
        os.makedirs(outputfile_folder)

# 依據pdf每頁大小進行分類並轉為jpg檔案
def convert_pdf_to_images():
    for root, dirs, files in os.walk(pdf_folder):
        for filename in files:
            if filename.endswith('.pdf'):
                file_path = os.path.join(root, filename)
                pages = convert_from_path(file_path, dpi=300)

                for i, page in enumerate(pages):
                    filename = os.path.splitext(filename)[0]
                    if page.size[0] < 2000:
                        output_path = os.path.join(pdfoutputimg_folder_note, f'{filename}_page_{i+1}_note.jpg')
                        page.save(output_path, 'JPEG')
                    elif page.size[0] > 2000 and page.size[0] < 2400:
                        output_path = os.path.join(pdfoutputimg_folder_main, f'{filename}_page_{i+1}_main.jpg')
                        page.save(output_path, 'JPEG')
                    elif page.size[0] > 2400 and page.size[0] < 3000:
                        output_path = os.path.join(pdfoutputimg_folder_cover, f'{filename}_page_{i+1}_cover.jpg')
                        page.save(output_path, 'JPEG')
                    else:
                        output_path = os.path.join(pdfoutputimg_folder_appendix, f'{filename}_page_{i+1}.jpg')
                        page.save(output_path, 'JPEG')

def process_files():
    rename_and_move_files(pdfoutputimg_folder_cover, pdfoutputimg_folder_main, '_page_1_cover.jpg', 'cover', 'main')
    rename_and_move_files(pdfoutputimg_folder_main, pdfoutputimg_folder_cover, '_page_1_main.jpg', 'main', 'cover')

    for filename in os.listdir(pdfoutputimg_folder_main):
        img_path = os.path.join(pdfoutputimg_folder_main, filename)
        aftercrop_img = crop_img(img_path)
        if aftercrop_img is not None:
            cv2.imwrite(img_path, aftercrop_img)
        else:
            print("Error: Failed to crop image: {}".format(img_path))


def perform_ocr():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    for filename in os.listdir(pdfoutputimg_folder_main):
        for i in range(1, 9):
            page_string = f"_page_{i}_main.jpg"
            if page_string in filename:
                # 執行處理 page1-8 的檔案

                ocr = PaddleOCR(use_angle_cls=True, lang='ch')
                img_path = os.path.join(pdfoutputimg_folder_main, filename)

                result = ocr.ocr(img_path, cls=True)

                logger.error(filename)
                logger.error('--------------filestart-----------------')

                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        logger.error(str(line))
                logger.error('---------------fileend----------------')

                filename = os.path.splitext(filename)[0]
                result = result[0]
                image = Image.open(img_path).convert('RGB')
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                im_show = draw_ocr(image, boxes, txts, scores,font_path='./doc/fonts/chinese_cht.ttf')
                im_show = Image.fromarray(im_show)
                im_show.save(os.path.join(outputimg_folder, f'{filename}_result.jpg'))

                righttop_location = [topright[2] for topright in boxes]
                righttop_order = [sorted(righttop_location, reverse=True)]

                outputresultfile = filename.split("_page")[0]
                output_path = os.path.join(outputfile_folder, f'{outputresultfile}_result.txt')
                if os.path.exists(output_path):
                    mode = 'a'
                else:
                    mode = 'w'

                with open(output_path, mode, encoding='UTF-8') as f:
                    for righttop in righttop_order[0]:
                        for line in result:
                            if line[0][2] == righttop:
                                f.write(line[1][0] + '\n')
                                break
                        else:
                            continue


def main():
    create_folders()
    convert_pdf_to_images()
    process_files()
    perform_ocr()

if __name__ == "__main__":
    main()  