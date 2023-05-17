import cv2
import os
import numpy as np

<<<<<<< HEAD
def color_filter(img_path, filename):
    input_image = cv2.imread(img_path)
=======
def color_filter(input_image):
>>>>>>> cbd04d1fbbfe8f7d1f0ed5ccd24503c692676c55
    #灰度图像处理
    GrayImage = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

    #反二进制阈值化处理
    r, b = cv2.threshold(GrayImage, 65, 255, cv2.THRESH_BINARY_INV)
    # hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    # height, width, _ = hsv_image.shape

    # for i in range(height):
    #     for j in range(width):
    #         hsv_pixel = hsv_image[i, j]

    #         # if not (((hsv_pixel[0] > 0 and hsv_pixel[0] < 8) or (hsv_pixel[0] > 120 and hsv_pixel[0] < 180))):
    #         if hsv_pixel[2] < 65:
    #             # 保留黑色像素
    #             pass
    #         else:
    #             # 将非黑色像素转换为其他颜色
    #             hsv_image[i, j] = [0, 0, 255]

    # output_image = cv2.cvtColor(GrayImage, cv2.COLOR_HSV2BGR)
    # cv2.imshow("Original Image", input_image)
    # cv2.imshow("Filtered Image", output_image)
<<<<<<< HEAD
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(test_folder+"/_3"+filename, b)

    return b, test_folder+"/_3"+filename

def adjust_contrast(binaryimg_path, alpha, beta, output_path):
    # 读取图像
    img = cv2.imread(binaryimg_path, cv2.IMREAD_GRAYSCALE)
=======
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./output.jpg', b)
    return './output.jpg'

def adjust_contrast(image_path, alpha, beta, output_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
>>>>>>> cbd04d1fbbfe8f7d1f0ed5ccd24503c692676c55

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

pdfoutputimg_folder_main = '../pdftoimg_main'
pdfoutputimg_binary_folder_main = '../pdftoimg_binary_main'
<<<<<<< HEAD
test_folder = '../testoutput'

for filename in os.listdir(pdfoutputimg_folder_main):

    img_path = pdfoutputimg_folder_main+'/'+filename #'PaddleOCR/doc/imgs_en/img_12.jpg'
    binaryimg, binaryimg_path = color_filter(img_path, filename)

    # print(adjusted_image)
    # 调用函数进行对比度调整
    img, img_path = adjust_contrast(binaryimg_path, 10, -50, test_folder+"/"+f'{filename}_3.jpg')
    print("img_path="+img_path)
    # 进行颜色过滤处理并显示结果
    gama_transfer(img, 1.5, test_folder+"/"+f'{filename}_3.jpg')
=======

for filename in os.listdir(pdfoutputimg_folder_main):
    # 读取输入图像
    input_image = cv2.imread(pdfoutputimg_folder_main+'/'+filename)

    # 进行颜色过滤处理并显示结果
    adjusted_image = color_filter(input_image)
    print("adjusted_image="+adjusted_image)

    # print(adjusted_image)
    # 调用函数进行对比度调整
    img, img_path = adjust_contrast(adjusted_image, 10, -50, pdfoutputimg_binary_folder_main+"/"+f'{filename}.jpg')
    print("img_path="+img_path)
    # 进行颜色过滤处理并显示结果
    gama_transfer(img, 1.5, pdfoutputimg_binary_folder_main+"/"+f'{filename}_2.jpg')
>>>>>>> cbd04d1fbbfe8f7d1f0ed5ccd24503c692676c55

