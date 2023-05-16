from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
import os, cv2, numpy as np
import logging
import datetime

def fill_incomplete_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 调整填充范围的大小
        padding = 1  # 缩小填充范围的像素值
        x += padding
        y += padding
        w -= 2 * padding
        h -= 2 * padding

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

        # 檢查區域是否包含有效像素
        if np.count_nonzero(binary[y:y+h, x:x+w]) > 0:
            # 計算輪廓內部區域的平均像素值
            roi = image[y:y+h, x:x+w]
            mean_val = np.mean(roi, axis=(0, 1)).astype(np.uint8)

            # 填充輪廓內部區域
            cv2.fillPoly(image, pts=[contour], color=mean_val.tolist())

    return image


def color_filter(img_path, filename):
    input_image = cv2.imread(img_path)
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 50], dtype=np.uint8)  # 灰色的下界
    upper_gray = np.array([180, 50, 220], dtype=np.uint8)  # 灰色的上界

    mask = cv2.inRange(hsv, lower_gray, upper_gray)  # 通过范围过滤得到遮罩

    non_gray_pixels = cv2.bitwise_not(mask)  # 非灰色像素的遮罩

    output_image = cv2.bitwise_and(input_image, input_image, mask=mask)  # 过滤出灰色像素
    output_image[np.where(non_gray_pixels)] = [255, 255, 255]  # 将非灰色像素填充为白色

    # 调整对比度
    alpha = 2  # 调整对比度的比例因子
    beta = 0  # 调整对比度的偏移量
    adjusted = cv2.convertScaleAbs(output_image, alpha=alpha, beta=beta)

    # 对黑色部分进行加深
    adjusted[adjusted == 0] = adjusted[adjusted == 0] * 0.3 - 200

    # 自定义模糊
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv2.filter2D(adjusted, -1, kernel)

    # 锐化滤波器
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
    sharpened_image = cv2.filter2D(dst, -1, sharpening_kernel)

    # 填补不完整区域
    filled_image = fill_incomplete_regions(sharpened_image)
    cv2.imwrite(test_folder+"/_4"+filename, filled_image)
    # cv2.namedWindow("filter", cv2.WINDOW_NORMAL)
    # cv2.imshow("filter", filled_image)
    # cv2.waitKey(0)


pdf_folder = './pdf/'
img_folder = './img'
pdfoutputimg_folder_all = './pdftoimg_all'
pdfoutputimg_folder_cover = './pdftoimg_cover'
pdfoutputimg_folder_note = './pdftoimg_note'
pdfoutputimg_folder_main = './pdftoimg_main'
pdfoutputimg_folder_appendix = './pdftoimg_appendix'
outputfile_folder = './outputfile'
outputlog_folder = './outputlog'
outputimg_folder = './outputimg'
test_folder = './testoutput'

# do ocr (stage1以本文為主)
for filename in os.listdir(pdfoutputimg_folder_main):
    img_path = pdfoutputimg_folder_main+'/'+filename #'PaddleOCR/doc/imgs_en/img_12.jpg'
    color_filter(img_path, filename)
    # print(filename)
    # print(img_folder+'/'+filename)

    ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory
    # img_path = './img/4-1.jpeg'

    # cv2.imread(img_path)
    # cv_img = cv2.cvtColor(np.asarray(img_path), cv2.COLOR_RGB2BGR)
    # gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # ret, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #prepare output ocr result 
    result = ocr.ocr(img_path, cls=True)
    # 去掉副檔名
    filename = os.path.splitext(filename)[0]

    # draw result
    from PIL import Image
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    print('---------------------------')
    # print(boxes)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(test_folder, f'{filename}_result_4.jpg'))


    # save to file
    righttop_location = [topright[2] for topright in boxes]
    # right to left order [right][top]
    righttop_order = [sorted(righttop_location,reverse=True)]

    # 搭配 with 寫入檔案
    output_path = os.path.join(test_folder, f'{filename}_4.txt')
    with open(output_path, 'w') as f:
        # get each box righttop(x, y)
        for righttop in righttop_order[0]:
            for line in result:
                # print(righttop)
                
                # check if match the righttop(x, y)
                if line[0][2] == righttop:
                    # write file
                    f.write(line[1][0]+'\n')
                    break
            else:
                continue