from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import os
from PIL import Image

# pdfoutputimg_folder_main = './pdftoimg_main'
# pdfoutputimg_binary_folder_main = './pdftoimg_binary_main'
test_folder = './testoutput'
img_path = './pdftoimg_main/__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg'
filename = '__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg'

def seg_img(img_path, filename):
    img = cv2.imread(img_path)
    hsvColor = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    thH = [0, 255]
    thS = [0, 255]
    thV = [0, 190]

    seg_image = cv2.inRange(hsvColor, np.array([thH[0], thS[0], thV[0]]), np.array([thH[1], thS[1], thV[1]]))

    cv2.imwrite(test_folder+"/_10"+filename, seg_image)

    return seg_image, test_folder+"/_10"+filename


seg_image, seg_img_path = seg_img(img_path, filename)
seg_image = cv2.imread(seg_img_path)

seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

e_img = cv2.erode(seg_image, kernel)     # 先侵蝕，將白色小圓點移除

d_img = cv2.dilate(e_img, kernel)    # 再膨脹，白色小點消失
edimg_output_path = test_folder+'/'+filename
cv2.imwrite(edimg_output_path, seg_image)



ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory

result = ocr.ocr(d_img, cls=True)

#draw result
result = result[0]
image = Image.open(edimg_output_path).convert('RGB')
boxes = [line[0] for line in result]
print('---------------------------')
# print(boxes)
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
#im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
im_show = Image.fromarray(im_show)
im_show.save(test_folder+'/dect_ori_'+filename)


# cv2.namedWindow("oxxostudio3", cv2.WINDOW_NORMAL)
# cv2.imshow('oxxostudio3', img)   # 膨脹後的影像

# cv2.namedWindow("oxxostudio1", cv2.WINDOW_NORMAL)
# cv2.imshow('oxxostudio1', img)   # 原始影像
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# img = cv2.erode(img, kernel)     # 先侵蝕，將白色小圓點移除
# cv2.namedWindow("oxxostudio2", cv2.WINDOW_NORMAL)
# cv2.imshow('oxxostudio2', img)   # 侵蝕後的影像

# img = cv2.dilate(img, kernel)    # 再膨脹，白色小點消失
# cv2.namedWindow("oxxostudio3", cv2.WINDOW_NORMAL)
# cv2.imshow('oxxostudio3', img)   # 膨脹後的影像

# cv2.waitKey(0)                   # 按下 q 鍵停止
# cv2.destroyAllWindows()

# # do ocr (stage1以笨文為主)
# for filename in os.listdir(pdfoutputimg_folder_main):

#     img_path = pdfoutputimg_folder_main+'/'+filename
#     binaryimg, binaryimg_path = seg_img(img_path, filename)
#     # print(filename)
#     # print(img_folder+'/'+filename)

#     ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory
#     # img_path = './img/4-1.jpeg'

#     # cv2.imread(img_path)
#     # cv_img = cv2.cvtColor(np.asarray(img_path), cv2.COLOR_RGB2BGR)
#     # gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#     # ret, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#     #prepare output ocr result 
#     result = ocr.ocr(binaryimg, cls=True)
#     # 去掉副檔名
#     filename = os.path.splitext(filename)[0]

#     # draw result
#     from PIL import Image
#     result = result[0]
#     image = Image.open(binaryimg_path).convert('RGB')
#     boxes = [line[0] for line in result]
#     print('---------------------------')
#     # print(boxes)
#     txts = [line[1][0] for line in result]
#     scores = [line[1][1] for line in result]
#     #im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
#     im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
#     im_show = Image.fromarray(im_show)
#     im_show.save(os.path.join(test_folder, f'{filename}_result_8.jpg'))


#     # save to file
#     righttop_location = [topright[2] for topright in boxes]
#     # right to left order [right][top]
#     righttop_order = [sorted(righttop_location,reverse=True)]

#     # 搭配 with 寫入檔案
#     output_path = os.path.join(test_folder, f'{filename}_8.txt')
#     with open(output_path, 'w') as f:
#         # get each box righttop(x, y)
#         for righttop in righttop_order[0]:
#             for line in result:
#                 # print(righttop)
                
#                 # check if match the righttop(x, y)
#                 if line[0][2] == righttop:
#                     # write file
#                     f.write(line[1][0]+'\n')
#                     break
#             else:
#                 continue
