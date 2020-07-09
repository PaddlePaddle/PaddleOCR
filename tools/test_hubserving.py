#!usr/bin/python
# -*- coding: utf-8 -*-

import requests
import json
import cv2
import base64
import time

def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

start = time.time()
# 发送HTTP请求
data = {'images':[cv2_to_base64(open("./doc/imgs/11.jpg", 'rb').read())]}
headers = {"Content-type": "application/json"}
# url = "http://127.0.0.1:8866/predict/ocr_det"
# url = "http://127.0.0.1:8866/predict/ocr_rec"
url = "http://127.0.0.1:8866/predict/ocr_system"
r = requests.post(url=url, headers=headers, data=json.dumps(data))
end = time.time()

# 打印预测结果
print(r.json()["results"])
print("time cost: ", end - start)
