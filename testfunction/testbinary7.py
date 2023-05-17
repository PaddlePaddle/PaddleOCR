import cv2
import numpy as np

src = cv2.imread('../pdftoimg_main/__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg')
cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.imshow("input", src)

"""
提取圖中的灰色部分
"""
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
low_hsv = np.array([0, 0, 46])  # Adjust the lower threshold for gray color
high_hsv = np.array([180, 43, 220])  # Adjust the upper threshold for gray color
mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("test",600, 800)
cv2.imshow("test", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# img = cv2.imread('../pdftoimg_main/__w__CP071-000-000038-000-000-000_001_page_2_cover.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
# output1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# img_gray2 = cv2.medianBlur(img_gray, 5)   # 模糊化
# output2 = cv2.adaptiveThreshold(img_gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# cv2.namedWindow("oxxostudio1", cv2.WINDOW_NORMAL)
# cv2.imshow('oxxostudio1', output1)
# cv2.namedWindow("oxxostudio2", cv2.WINDOW_NORMAL)
# cv2.imshow('oxxostudio2', output2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()