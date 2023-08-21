import cv2
import numpy as np

# 讀取影像
image = cv2.imread('./pdftoimg_main/__w__CP066-000-000170-000-000-000_001_Iyls18x_page_2_main.jpg') #圖片名稱


# 將影像轉換為HSV顏色空間
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義紅色的HSV閾值範圍
lower_red = np.array([0, 100, 100])  # 下限閾值
upper_red = np.array([10, 255, 255]) # 上限閾值

# 使用inRange函式根據閾值範圍創建遮罩
red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

# 尋找紅色物體的輪廓
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的輪廓
largest_contour = max(contours, key=cv2.contourArea)

# 在原始影像上繪製最大方框
x, y, w, h = cv2.boundingRect(largest_contour)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 縮小圖片成原本的20%
smaller_image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

# 顯示結果
cv2.imshow('Largest Red Object Detection', smaller_image)
cv2.waitKey(0)
cv2.destroyAllWindows()