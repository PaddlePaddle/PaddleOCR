import cv2
import os
import numpy as np

def get_hsv_values(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv_image.shape

    hsv_values = []
    for i in range(height):
        for j in range(width):
            hsv_pixel = hsv_image[i, j]
            hsv_values.append(hsv_pixel)

    return hsv_values

input_image = cv2.imread("./pdftoimg_main/__w__CP066-000-000099-000-000-000_001_KJfQV0p_page_2_main.jpg")


# 獲得每個點的HSV值
hsv_values = get_hsv_values(input_image)

# 打印第一個點的HSV值
# print(hsv_values)

# 搭配 with 寫入檔案
output_path = os.path.join("./", 'hsv.txt')
with open(output_path, 'w') as f:
    # get each box righttop(x, y)
    for hsv in hsv_values:
        if hsv[2] < 65 :
            # 將ndarray轉換為字符串
            array_str = np.array2string(hsv, separator=', ')
            f.write(array_str+'\n')