from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
import os, cv2, numpy as np
import logging
import datetime

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

def color_filter(img_path, filename):
    input_image = cv2.imread(img_path)
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 50], dtype=np.uint8)  # 灰色的下界
    upper_gray = np.array([180, 50, 220], dtype=np.uint8)  # 灰色的上界

    mask = cv2.inRange(hsv, lower_gray, upper_gray)  # 通過範圍過濾得到遮罩

    non_gray_pixels = cv2.bitwise_not(mask)  # 非灰色像素的遮罩

    output_image = cv2.bitwise_and(input_image, input_image, mask=mask)  # 過濾出灰色像素
    output_image[np.where(non_gray_pixels)] = [255, 255, 255]  # 將非灰色像素填充為白色

    cv2.imwrite(test_folder+"/_5"+filename, output_image)

    return output_image, test_folder+"/_4"+filename
    # cv2.namedWindow("filter", cv2.WINDOW_NORMAL)
    # cv2.imshow("filter", filled_image)
    # cv2.waitKey(0)


# do ocr (stage1以笨文為主)
for filename in os.listdir(pdfoutputimg_folder_main):
    img_path = pdfoutputimg_folder_main+'/'+filename #'PaddleOCR/doc/imgs_en/img_12.jpg'
    binaryimg, binaryimg_path = color_filter(img_path, filename)
    # print(filename)
    # print(img_folder+'/'+filename)

    ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory
    # img_path = './img/4-1.jpeg'

    # cv2.imread(img_path)
    # cv_img = cv2.cvtColor(np.asarray(img_path), cv2.COLOR_RGB2BGR)
    # gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # ret, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #prepare output ocr result 
    result = ocr.ocr(binaryimg, cls=True)
    # 去掉副檔名
    filename = os.path.splitext(filename)[0]

    # draw result
    from PIL import Image
    result = result[0]
    image = Image.open(binaryimg_path).convert('RGB')
    boxes = [line[0] for line in result]
    print('---------------------------')
    # print(boxes)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(test_folder, f'{filename}_result_5.jpg'))


    # save to file
    righttop_location = [topright[2] for topright in boxes]
    # right to left order [right][top]
    righttop_order = [sorted(righttop_location,reverse=True)]

    # 搭配 with 寫入檔案
    output_path = os.path.join(test_folder, f'{filename}_5.txt')
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