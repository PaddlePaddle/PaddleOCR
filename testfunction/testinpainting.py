# import numpy as np
# import cv2

# def inpainting(f, method=2):
#     nr, nc = f.shape[:2]
#     mask = np.zeros([nr, nc], dtype='uint8')
#     corners = []  # 存儲角落座標
#     for x in range(nr):
#         for y in range(nc):
#             # 檢查紅色的範圍
#             if f[x, y, 2] >= 250 and f[x, y, 2] <= 255 and f[x, y, 0] <= f[x, y, 2] and f[x, y, 1] <= f[x, y, 2]:
#                 mask[x, y] = 255
#                 corners.append((x, y))
#                 if len(corners) == 3:
#                     break
#         if len(corners) == 3:
#             break
    
#     if method == 1:
#         g = cv2.inpaint(f, mask, 3, cv2.INPAINT_NS)
#     else:
#         g = cv2.inpaint(f, mask, 3, cv2.INPAINT_TELEA)
    
#     # 繪製方框
#     if len(corners) == 3:
#         x1, y1 = corners[0]
#         x2, y2 = corners[1]
#         x3, y3 = corners[2]
#         cv2.rectangle(g, (x1, y1), (x2, y3), (0, 0, 255), 2)
    
#     return g

# def main():
#     img1 = cv2.imread("pdftoimg_appendix/__w__CP066-000-000170-000-000-000_001_Iyls18x_page_2.jpg", -1)
#     img2 = inpainting(img1, 1)
#     cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
#     cv2.imshow("Original", img1)
#     cv2.namedWindow("Inpainting", cv2.WINDOW_NORMAL)
#     cv2.imshow("Inpainting", img2)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# main()

import cv2
import numpy as np

# 讀取圖片
image = cv2.imread("pdftoimg_appendix/__w__CP066-000-000170-000-000-000_001_Iyls18x_page_2.jpg", -1)

# 將圖片轉換為HSV色彩空間
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義紅色範圍的HSV值
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
upper_red2 = np.array([170, 255, 255])

# 創建遮罩，標記符合紅色範圍的區域
mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
mask2 = cv2.inRange(hsv_image, lower_red, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 使用形態學操作對遮罩進行擴張和腐蝕，以去除噪點
kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
mask = cv2.erode(mask, kernel, iterations=1)

# 在遮罩上找到白色區域的水平投影
projection = np.sum(mask, axis=0)

# 找到最左邊和最右邊的非零值索引
left_index = np.argmax(projection > 0)
right_index = len(projection) - np.argmax(projection[::-1] > 0)

# 切割圖片
cropped_image = image[:, left_index:right_index]

# 顯示結果
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow('Image', image)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.imshow('Mask', mask)
cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
