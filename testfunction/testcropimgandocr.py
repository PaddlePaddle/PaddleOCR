import cv2
import numpy as np
import numpy as np
import os
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

img_path = 'pdftoimg_main/__w__CP069-000-000345-000-000-000_001_page_3_cover.jpg'
filename = '__w__CP069-000-000345-000-000-000_001_page_3_cover.jpg'
test_folder = './testoutput'

def crop_ed_img(img_path):

    lower = np.array([30, 40, 200])
    upper = np.array([90, 100, 255])
    # 讀取圖片
    ori_img = cv2.imread(img_path)

    output = cv2.inRange(ori_img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    output = cv2.dilate(output, kernel)
    output = cv2.erode(output, kernel)
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的方框
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # 切割最大的方框區域
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        max_box_region = ori_img[y:y+h, x:x+w].copy()
        ori_img_copy = max_box_region

    cv2.imwrite(test_folder+"/_10_crop_"+filename, ori_img_copy)

    return test_folder+"/_10_crop_"+filename



def seg_img(cropimg_path):
    crop_img = cv2.imread(cropimg_path)
    hsvColor = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    thH = [0, 255]
    thS = [0, 255]
    thV = [0, 190]

    seg_image = cv2.inRange(hsvColor, np.array([thH[0], thS[0], thV[0]]), np.array([thH[1], thS[1], thV[1]]))

    cv2.imwrite(test_folder+"/_10_seg_"+filename, seg_image)

    return test_folder+"/_10_seg_"+filename



def erosion_dilation_img(seg_img_path):
    seg_image = cv2.imread(seg_img_path)

    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    e_img = cv2.erode(seg_image, kernel)     # 先侵蝕，將白色小圓點移除

    d_img = cv2.dilate(e_img, kernel)    # 再膨脹，白色小點消失
    edimg_output_path = test_folder+'/'+filename
    cv2.imwrite(edimg_output_path, seg_image)

    return seg_image, edimg_output_path



def do_ocr(ed_img, edimg_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory

    result = ocr.ocr(ed_img, cls=True)

    #draw result
    result = result[0]
    image = Image.open(edimg_path).convert('RGB')
    boxes = [line[0] for line in result]
    print('---------------------------')
    # print(boxes)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(test_folder+'/dect_'+filename)



cropimg_path = crop_ed_img(img_path)
seg_img_path = seg_img(cropimg_path)
ed_img, edimg_path = erosion_dilation_img(seg_img_path)
do_ocr(ed_img, edimg_path)