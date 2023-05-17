import cv2
import numpy as np

input_img = '__w__CP069-000-000310-000-000-000_001_page_3_cover.jpg'
img = cv2.imread(input_img)
hsvColor = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def mouseHSV(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsH = hsvColor[y,x,0]
        colorsS = hsvColor[y,x,1]
        colorsV = hsvColor[y,x,2]
        colors = hsvColor[y,x]
        print("Red: ",colorsH)
        print("Green: ",colorsS)
        print("Blue: ",colorsV)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)


window_name = 'mouseHSV'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name,600, 800)
cv2.setMouseCallback(window_name, mouseHSV)


thH = [5, 190]
thS = [6, 190]
thV = [0, 200]

#inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);

seg_image = cv2.inRange(hsvColor, np.array([thH[0], thS[0], thV[0]]), np.array([thH[1], thS[1], thV[1]]))


#upper = np.array([hsvColor[0][0][0] + 10, hsvColor[0][0][1] + 10, hsvColor[0][0][2] + 40])
#lower = np.array([hsvColor[0][0][0] - 10, hsvColor[0][0][1] - 10, hsvColor[0][0][2] - 40])

#seg_image= cv2.inRange(hsvColor, (upper, lower))
cv2.imshow(window_name, seg_image)
cv2.imwrite('matytest.jpg', seg_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



