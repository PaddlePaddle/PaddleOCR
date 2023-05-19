import cv2
import numpy as np

img_path = 'pdftoimg_main/__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg'

lower = np.array([30, 40, 200])
upper = np.array([90, 100, 255])

# 讀取圖片
img = cv2.imread(img_path)

output = cv2.inRange(img, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
output = cv2.dilate(output, kernel)
output = cv2.erode(output, kernel)
contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的方框
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# 切割最大的方框區域並取代原圖
if max_contour is not None:
    x, y, w, h = cv2.boundingRect(max_contour)
    max_box_region = img[y:y+h, x:x+w].copy()
    img = max_box_region

cv2.imshow('oxxostudio', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
