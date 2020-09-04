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

from paddle_serving_client import Client
from paddle_serving_app.reader import OCRReader
import cv2
import sys
import numpy as np
import os
from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, URL2Image, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes
if sys.argv[1] == 'gpu':
    from paddle_serving_server_gpu.web_service import WebService
elif sys.argv[1] == 'cpu':
    from paddle_serving_server.web_service import WebService
import time
import re
import base64


class OCRService(WebService):
    def init_rec(self):
        self.ocr_reader = OCRReader()

    def preprocess(self, feed=[], fetch=[]):
        img_list = []
        for feed_data in feed:
            data = base64.b64decode(feed_data["image"].encode('utf8'))
            data = np.fromstring(data, np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img_list.append(im)
        max_wh_ratio = 0
        for i, boximg in enumerate(img_list):
            h, w = boximg.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        _, w, h = self.ocr_reader.resize_norm_img(img_list[0],
                                                  max_wh_ratio).shape
        imgs = np.zeros((len(img_list), 3, w, h)).astype('float32')
        for i, img in enumerate(img_list):
            norm_img = self.ocr_reader.resize_norm_img(img, max_wh_ratio)
            imgs[i] = norm_img
        feed = {"image": imgs.copy()}
        fetch = ["ctc_greedy_decoder_0.tmp_0", "softmax_0.tmp_0"]
        return feed, fetch

    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        rec_res = self.ocr_reader.postprocess(fetch_map, with_score=True)
        res_lst = []
        for res in rec_res:
            res_lst.append(res[0])
        res = {"res": res_lst}
        return res


ocr_service = OCRService(name="ocr")
ocr_service.load_model_config("ocr_rec_model")
ocr_service.init_rec()
if sys.argv[1] == 'gpu':
    ocr_service.set_gpus("0")
    ocr_service.prepare_server(workdir="workdir", port=9292, device="gpu", gpuid=0)
    ocr_service.run_debugger_service(gpu=True)
elif sys.argv[1] == 'cpu':
    ocr_service.prepare_server(workdir="workdir", port=9292, device="cpu")
    ocr_service.run_debugger_service()
ocr_service.run_web_service()
