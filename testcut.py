import cv2
import numpy as np

imgPath = './pdftoimg_main/__w__CP066-000-000170-000-000-000_001_Iyls18x_page_2_main.jpg'
outputPath = "./output.png"

# 向內移動movePixel個 再往外移動 碰到紅線角則停止
def walkFind(image, red_mask, x, y, w, h, movePixel=200):
    x1 = x + movePixel
    x2 = x + w - movePixel
    x3 = x + movePixel
    x4 = x + w - movePixel
    y1 = y + movePixel
    y2 = y + movePixel
    y3 = y + h - movePixel
    y4 = y + h - movePixel

    # cv2.circle(image, (x1, y1), 10, (255, 0, 0), 5)
    # cv2.circle(image, (x2, y2), 10, (255, 0, 0), 5)
    # cv2.circle(image, (x3, y3), 10, (255, 0, 0), 5)
    # cv2.circle(image, (x4, y4), 10, (255, 0, 0), 5)

    #左上
    conti = 1
    while(conti):
        if red_mask[y1-1][x1-1] == 0 and y1 >= y and x1 >= x:
            y1 -= 1
            x1 -= 1
        elif red_mask[y1][x1-1] == 0 and y1 >= y and x1 >= x:
            x1 -= 1
        elif red_mask[y1-1][x1] == 0 and y1 >= y and x1 >= x:
            y1 -= 1
        else:
            conti = 0

    #右上

    conti = 1
    while(conti):
        if red_mask[y2-1][x2+1] == 0 and y2 >= y and x2 <= x + w:
            y2 -= 1
            x2 += 1
        elif red_mask[y2][x2+1] == 0 and y2 >= y and x2 <= x + w:
            x2 += 1
        elif red_mask[y2-1][x2] == 0 and y2 >= y and x2 <= x + w:
            y2 -= 1
        else:
            conti = 0
    
    #左下
    conti = 1
    while(conti):
        if red_mask[y3+1][x3-1] == 0 and y3 <= y + h and x3 >= x:
            y3 += 1
            x3 -= 1
        elif red_mask[y3][x3-1] == 0 and y3 <= y + h and x3 >= x:
            x3 -= 1
        elif red_mask[y3+1][x3] == 0 and y3 <= y + h and x3 >= x:
            y3 += 1
        else:
            conti = 0

    #右下
    conti = 1
    while(conti):
        if red_mask[y4+1][x4+1] == 0 and y4 <= y + h and x4 <= x + w:
            y4 += 1
            x4 += 1
        elif red_mask[y4][x4+1] == 0 and y4 <= y + h and x4 <= x + w:
            x4 += 1
        elif red_mask[y4+1][x4] == 0 and y4 <= y + h and x4 <= x + w:
            y4 += 1
        else:
            conti = 0

    x1 = min(x1, x3)
    y1 = min(y1, y2)
    x4 = max(x2, x4)
    y4 = max(y4, y3)

    return x1, y1, x4-x1, y4-y1

if __name__ == "__main__":
    # 讀取影像
    image = cv2.imread(imgPath) #圖片名稱

    # 將影像轉換為HSV顏色空間
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義紅色的HSV閾值範圍
    lower_red = np.array([0, 75, 75])  # 下限閾值
    upper_red = np.array([10, 255, 255]) # 上限閾值

    # 使用inRange函式根據閾值範圍創建遮罩
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    # cv2.imwrite("./HSV.png", red_mask)

    # 尋找紅色物體的輪廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的輪廓
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros(red_mask.shape)
    for i in range(len(largest_contour)):
        mask[largest_contour[i][0][1]][largest_contour[i][0][0]] = 255
    # cv2.imwrite("./mask.png", mask)

    # 在原始影像上繪製最大方框
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)
    
    x, y, w, h = walkFind(image, red_mask, x, y, w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (209, 206, 0), 10)

    cv2.imwrite(outputPath, image)