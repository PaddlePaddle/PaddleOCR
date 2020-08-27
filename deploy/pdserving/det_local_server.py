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
import cv2
import sys
import numpy as np
import os
from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from paddle_serving_app.reader import DBPostProcess, FilterBoxes
if sys.argv[1] == 'gpu':
    from paddle_serving_server_gpu.web_service import WebService
elif sys.argv[1] == 'cpu':
    from paddle_serving_server.web_service import WebService
import time
import re
import base64


class OCRService(WebService):
    def init_det(self):
        self.det_preprocess = Sequential([
            ResizeByFactor(32, 960), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                (2, 0, 1))
        ])
        self.filter_func = FilterBoxes(10, 10)
        self.post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "min_size": 3
        })

    def preprocess(self, feed=[], fetch=[]):
        data = base64.b64decode(feed[0]["image"].encode('utf8'))
        data = np.fromstring(data, np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self.ori_h, self.ori_w, _ = im.shape
        det_img = self.det_preprocess(im)
        _, self.new_h, self.new_w = det_img.shape
        return {"image": det_img[np.newaxis, :].copy()}, ["concat_1.tmp_0"]

    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        det_out = fetch_map["concat_1.tmp_0"]
        ratio_list = [
            float(self.new_h) / self.ori_h, float(self.new_w) / self.ori_w
        ]
        dt_boxes_list = self.post_func(det_out, [ratio_list])
        dt_boxes = self.filter_func(dt_boxes_list[0], [self.ori_h, self.ori_w])
        return {"dt_boxes": dt_boxes.tolist()}


ocr_service = OCRService(name="ocr")
ocr_service.load_model_config("ocr_det_model")
ocr_service.init_det()
if sys.argv[1] == 'gpu':
    ocr_service.set_gpus("0")
    ocr_service.prepare_server(workdir="workdir", port=9292, device="gpu", gpuid=0)
    ocr_service.run_debugger_service(gpu=True)
elif sys.argv[1] == 'cpu':
    ocr_service.prepare_server(workdir="workdir", port=9292)
    ocr_service.run_debugger_service()
ocr_service.init_det()
ocr_service.run_web_service()
