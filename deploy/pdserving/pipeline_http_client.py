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

import numpy as np
import requests
import json
import base64
import os

import argparse


def str2bool(v):
    return v.lower() in ("true", "t", "1")


parser = argparse.ArgumentParser(description="args for paddleserving")
parser.add_argument("--image_dir", type=str, default="../../doc/imgs/")
parser.add_argument("--det", type=str2bool, default=True)
parser.add_argument("--rec", type=str2bool, default=True)
args = parser.parse_args()


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


url = "http://127.0.0.1:9998/ocr/prediction"
test_img_dir = args.image_dir

for idx, img_file in enumerate(os.listdir(test_img_dir)):
    with open(os.path.join(test_img_dir, img_file), 'rb') as file:
        image_data1 = file.read()
    # print file name
    print('{}{}{}'.format('*' * 10, img_file, '*' * 10))

    image = cv2_to_base64(image_data1)

    data = {"key": ["image"], "value": [image]}
    r = requests.post(url=url, data=json.dumps(data))
    result = r.json()
    print("erro_no:{}, err_msg:{}".format(result["err_no"], result["err_msg"]))
    # check success
    if result["err_no"] == 0:
        ocr_result = result["value"][0]
        if not args.det:
            print(ocr_result)
        else:
            try:
                for item in eval(ocr_result):
                    # return transcription and points
                    print("{}, {}".format(item[0], item[1]))
            except Exception as e:
                print(ocr_result)
                print("No results")
                continue

    else:
        print(
            "For details about error message, see PipelineServingLogs/pipeline.log"
        )
print("==> total number of test imgs: ", len(os.listdir(test_img_dir)))
