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


    return ori_img_copy


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
    im_show.save(test_folder+'/dect_clahe'+filename)


crop_img = crop_ed_img(img_path)
cropimg_path = test_folder+"/_10_crop_ori_"+filename
cv2.imwrite(cropimg_path, crop_img)


# Reading the image from the present directory
image = cv2.imread(cropimg_path)

# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit = 3)
final_img = clahe.apply(image_bw) + 10
 
# Ordinary thresholding the same image
_, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
 
# Showing all the three images
cv2.imshow("ordinary threshold", ordinary_img)
# cv2.imshow("CLAHE image", final_img)



clahe_img_path = test_folder+'/clahe_'+filename
cv2.imwrite(clahe_img_path, final_img)

do_ocr(final_img, clahe_img_path)