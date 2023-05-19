import cv2

import numpy as np

img = cv2.imread('image.jpg')
type(img)

img.shape
(1080, 1920, 3)

ltop = (100, 100)
rtbm = (200, 200)

img_cap = img[ltop[1]:rtbm[1], ltop[0]: rtbm[0]]

cv2.imshow('Image', img_cap)
# cv2.imwrite('output.jpg', img_cap)
cv2.waitKey(0)
cv2.destroyAllWindows()