from paddleocr import PaddleOCR,draw_ocr
import os
import cv2
from PIL import Image
# import numpy as np
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.

img_folder = './img'
binimg_folder = './binimg'
outputfile_folder = './outputfile'
outputimg_folder = './outputimg'

file_names = os.listdir(img_folder)

if not os.path.exists(binimg_folder):
    os.makedirs(binimg_folder)

if not os.path.exists(outputimg_folder):
    os.makedirs(outputimg_folder)

if not os.path.exists(outputfile_folder):
    os.makedirs(outputfile_folder)

for filename in os.listdir(img_folder):
    # print(filename)
    # print(img_folder+'/'+filename)

    ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory
    img_path = img_folder+'/'+filename #'PaddleOCR/doc/imgs_en/img_12.jpg'
    # img_path = './img/4-1.jpeg'

    img = Image.open(img_path)
    
    cv_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
    # gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # ret, bin_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    # ret, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    result = ocr.ocr(cv_img, cls=True)

    # binimg_path = binimg_folder+'/'+filename+'_bin.jpg'

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    # save binarization image
    # cv2.imshow('2', bin_img)
    # cv2.imwrite(binimg_path, bin_img)
    # result.save(os.path.join(binimg_folder, f'{filename}_bin.jpg'))

    # draw result
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    print('---------------------------')
    print(boxes)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(outputimg_folder, f'{filename}_result1.jpg'))


    # save to file
    righttop_location = [topright[2] for topright in boxes]
    # right to left order [right][top]
    righttop_order = [sorted(righttop_location,reverse=True)]

    # 搭配 with 寫入檔案
    output_path = os.path.join(outputfile_folder, f'{filename}_result1.txt')
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
