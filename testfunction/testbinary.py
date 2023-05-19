from paddleocr import PaddleOCR, draw_ocr
import cv2
import os
import numpy as np

def color_filter(img_path, filename):
    input_image = cv2.imread(img_path)
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv_image.shape

    for i in range(height):
        for j in range(width):
            hsv_pixel = hsv_image[i, j]

            # if not (((hsv_pixel[0] > 0 and hsv_pixel[0] < 8) or (hsv_pixel[0] > 120 and hsv_pixel[0] < 180))):
            if hsv_pixel[2] < 65:
                # 保留黑色像素
                pass
            else:
                # 将非黑色像素转换为其他颜色
                hsv_image[i, j] = [0, 0, 255]

    # output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    # cv2.imshow("Original Image", input_image)
    # cv2.imshow("Filtered Image", output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(test_folder+"/_1"+filename, hsv_image)
    return hsv_image, test_folder+"/_1"+filename

def adjust_contrast(image_path, alpha, beta, output_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 调整对比度
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 对黑色部分进行加深
    adjusted[adjusted == 0] = adjusted[adjusted == 0] * 0.3 -200

    # 保存调整后的图像
    cv2.imwrite(output_path, adjusted)
    return adjusted, output_path

def gama_transfer(img, power1, output_path):
    if len(img.shape) == 3:
         img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = 255*np.power(img/255,power1)
    img = np.around(img)
    img[img>255] = 255
    out_img = img.astype(np.uint8)
    # 保存调整后的图像
    cv2.imwrite(output_path, out_img)

pdfoutputimg_folder_main = './pdftoimg_main'
pdfoutputimg_binary_folder_main = './pdftoimg_binary_main'
test_folder = '../testoutput'

# for filename in os.listdir(pdfoutputimg_folder_main):
#     # 读取输入图像
#     input_image = cv2.imread(pdfoutputimg_folder_main+'/'+filename)

#     # 进行颜色过滤处理并显示结果
#     binaryimg, binaryimg_path = color_filter(img_path, filename)

#     # print(adjusted_image)
#     # 调用函数进行对比度调整
#     img, img_path = adjust_contrast(binaryimg, 1.5, -50, pdfoutputimg_binary_folder_main+"/"+f'{filename}.jpg')
#     print("img_path="+img_path)
#     # 进行颜色过滤处理并显示结果
#     gama_transfer(img, 5, pdfoutputimg_binary_folder_main+"/"+f'{filename}_1.jpg')
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
    im_show.save(os.path.join(test_folder, f'{filename}_result_1.jpg'))


    # save to file
    righttop_location = [topright[2] for topright in boxes]
    # right to left order [right][top]
    righttop_order = [sorted(righttop_location,reverse=True)]

    # 搭配 with 寫入檔案
    output_path = os.path.join(test_folder, f'{filename}_1.txt')
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

