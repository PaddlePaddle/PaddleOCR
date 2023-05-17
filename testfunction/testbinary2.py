import cv2
import numpy as np

def color_filter(input_image):
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

    output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
<<<<<<< HEAD
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", input_image)
    cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
=======
    cv2.imshow("Original Image", input_image)
>>>>>>> cbd04d1fbbfe8f7d1f0ed5ccd24503c692676c55
    cv2.imshow("Filtered Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取输入图像
<<<<<<< HEAD
input_image = cv2.imread("../pdftoimg_main/__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg")
=======
input_image = cv2.imread("../pdftoimg_main\__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg")
>>>>>>> cbd04d1fbbfe8f7d1f0ed5ccd24503c692676c55

# 进行颜色过滤处理并显示结果
color_filter(input_image)


