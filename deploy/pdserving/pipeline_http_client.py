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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse
parser = argparse.ArgumentParser(description="args for paddleserving")
parser.add_argument("--image_dir", type=str, default="../../doc/imgs/")
parser.add_argument('--num_thread', type=int, default=35)
args = parser.parse_args()


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

def handle(filename):
    with open(filename, 'rb') as file:
        image_data1 = file.read()

    image = cv2_to_base64(image_data1)

    data = {"key": ["image"], "value": [image]}
    r = requests.post(url=url, data=json.dumps(data))
    return r.json()


url = "http://127.0.0.1:9998/ocr/prediction"
test_img_dir = args.image_dir

start = time.time()
with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
    futures = [executor.submit(handle, os.path.join(test_img_dir, img_file)) for idx, img_file in enumerate(os.listdir(test_img_dir))]
    for future in as_completed(futures):
        print(future.result())

print("==> total number of test imgs: ", len(os.listdir(test_img_dir)))
print('==> total time cost:', time.time() - start)
