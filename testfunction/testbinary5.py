import cv2
import numpy as np

def color_filter(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 50], dtype=np.uint8)  # 灰色的下界
    upper_gray = np.array([180, 50, 220], dtype=np.uint8)  # 灰色的上界

    mask = cv2.inRange(hsv, lower_gray, upper_gray)  # 通過範圍過濾得到遮罩

    non_gray_pixels = cv2.bitwise_not(mask)  # 非灰色像素的遮罩

    output_image = cv2.bitwise_and(input_image, input_image, mask=mask)  # 過濾出灰色像素
    output_image[np.where(non_gray_pixels)] = [255, 255, 255]  # 將非灰色像素填充為白色

    cv2.namedWindow("filter", cv2.WINDOW_NORMAL)  # 創建可調整大小的視窗
    cv2.imshow("filter", output_image)
    cv2.waitKey(0)

input_image = cv2.imread('../pdftoimg_main\__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg')
color_filter(input_image)