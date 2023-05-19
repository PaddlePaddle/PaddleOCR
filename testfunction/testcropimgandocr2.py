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

def modify_contrast_and_brightness2(img, brightness=0 , contrast=0):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果
    '''
    Algorithm of Brightness Contrast transformation
    The formula is:
        y = [x - 127.5 * (1 - B)] * k + 127.5 * (1 + B);

        x is the input pixel value
        y is the output pixel value
        B is brightness, value range is [-1,1]
        k is used to adjust contrast
            k = tan( (45 + 44 * c) / 180 * PI );
            c is contrast, value range is [-1,1]
    '''
    import math
    
    # ---------------------- 減少對比度 (白黑都接近灰，分不清楚) ---------------------- #
    
    # brightness = 0
    # contrast = -100

    # B = brightness / 255.0
    # c = contrast / 255.0 
    # k = math.tan((45 + 44 * c) / 180 * math.pi)

    # img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
      
    # # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    # img = np.clip(img, 0, 255).astype(np.uint8)

    # print("減少對比度 (白黑都接近灰，分不清楚): ")
    # show_img(img)
    
    # ---------------------- 增加對比度 (白的更白，黑的更黑) ---------------------- #
    
    brightness = 0
    contrast = +200
    
    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((18 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
      
    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img
    # print("增加對比度 (白的更白，黑的更黑): ")

def do_ocr(crop_img, cropimg_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory

    result = ocr.ocr(crop_img, cls=True)

    #draw result
    result = result[0]
    image = Image.open(cropimg_path).convert('RGB')
    boxes = [line[0] for line in result]
    print('---------------------------')
    # print(boxes)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(test_folder+'/dect_ori_imp0_'+filename)



crop_img = crop_ed_img(img_path)
cropimg_path = test_folder+"/_10_crop_ori_"+filename
cv2.imwrite(cropimg_path, crop_img)


imp_img = modify_contrast_and_brightness2(crop_img)
impimg_path = test_folder+"/_10_crop_ori_imp0_"+filename
cv2.imwrite(impimg_path, imp_img)


do_ocr(imp_img, impimg_path)