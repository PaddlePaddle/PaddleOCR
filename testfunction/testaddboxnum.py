import cv2
import numpy as np
import os
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

img_path = 'pdftoimg_main/__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg'
filename = '__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg'
test_folder = './testoutput'


def crop_ed_img(img_path):
    lower = np.array([30, 40, 200])
    upper = np.array([90, 100, 255])
    ori_img = cv2.imread(img_path)
    output = cv2.inRange(ori_img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    output = cv2.dilate(output, kernel)
    output = cv2.erode(output, kernel)
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        max_box_region = ori_img[y:y+h, x:x+w].copy()
        ori_img_copy = max_box_region

    cv2.imwrite(test_folder + "/_10_crop_ori_" + filename, ori_img_copy)

    return ori_img_copy, test_folder + "/_10_crop_ori_" + filename


def draw_ocr_numbers(image, boxes):
    for i, box in enumerate(boxes):
        box = box.astype(int).reshape((-1, 1, 2))
        cv2.putText(
            image,
            str(i + 1),
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 0, 255),  # 編號顏色為紅色
            thickness=1
        )
    return image


def do_ocr(crop_img, cropimg_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', font_path='./doc/fonts/chinese_cht.ttf')

    #draw result
    result = ocr.ocr(crop_img, cls=True)
    result = result[0]

    image = np.array(Image.open(cropimg_path).convert('RGB'))
    boxes = np.array([line[0] for line in result])

    # print(boxes)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    im_show = draw_ocr_numbers(image, boxes)  # 加上編號
    temp_ocr_path = test_folder + "/_10_crop_ocrnum_" + filename
    cv2.imwrite(temp_ocr_path, im_show)
    ocr_step1_img = cv2.imread(temp_ocr_path)

    ocr_step1_img = draw_ocr(ocr_step1_img, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
    ocr_step1_img = Image.fromarray(ocr_step1_img)
    ocr_step1_img.save(test_folder + '/dect_ori3_' + filename)


crop_img, cropimg_path = crop_ed_img(img_path)
do_ocr(crop_img, cropimg_path)


# def do_ocr(crop_img, cropimg_path):
#     ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 需要运行一次以下载和加载模型到内存中

#     result = ocr.ocr(crop_img, cls=True)

#     result = result[0]
#     image = np.array(Image.open(cropimg_path).convert('RGB'))  # 将image转换为NumPy数组
#     boxes = np.array([line[0] for line in result])  # 将boxes转换为NumPy数组
#     txts = [line[1][0] for line in result]
#     scores = [line[1][1] for line in result]
#     im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
#     im_show = Image.fromarray(im_show)
#     im_show.save(test_folder + '/dect_ori2_' + filename)