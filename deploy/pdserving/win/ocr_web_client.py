# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-

import requests
import json
import cv2
import base64
import os, sys
import time


def cv2_to_base64(image):
    #data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(image).decode(
        'utf8')  #data.tostring()).decode('utf8')


headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:9292/ocr/prediction"

test_img_dir = "../../../doc/imgs/"
for idx, img_file in enumerate(os.listdir(test_img_dir)):
    with open(os.path.join(test_img_dir, img_file), 'rb') as file:
        image_data1 = file.read()

    image = cv2_to_base64(image_data1)
    for i in range(1):
        data = {
            "feed": [{
                "image": image
            }],
            "fetch": ["save_infer_model/scale_0.tmp_1"]
        }
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        print(r.json())

test_img_dir = "../../../doc/imgs/"
print("==> total number of test imgs: ", len(os.listdir(test_img_dir)))
