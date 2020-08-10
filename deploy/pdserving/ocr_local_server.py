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
from paddle_serving_app.local_predict import Debugger
import time
import re
import base64


class OCRService(WebService):
    def init_det_debugger(self, det_model_config):
        self.det_preprocess = Sequential([
            ResizeByFactor(32, 960), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                (2, 0, 1))
        ])
        self.det_client = Debugger()
        if sys.argv[1] == 'gpu':
            self.det_client.load_model_config(
                det_model_config, gpu=True, profile=False)
        elif sys.argv[1] == 'cpu':
            self.det_client.load_model_config(
                det_model_config, gpu=False, profile=False)
        self.ocr_reader = OCRReader()

    def preprocess(self, feed=[], fetch=[]):
        data = base64.b64decode(feed[0]["image"].encode('utf8'))
        data = np.fromstring(data, np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        ori_h, ori_w, _ = im.shape
        det_img = self.det_preprocess(im)
        _, new_h, new_w = det_img.shape
        det_img = det_img[np.newaxis, :]
        det_img = det_img.copy()
        det_out = self.det_client.predict(
            feed={"image": det_img}, fetch=["concat_1.tmp_0"])
        filter_func = FilterBoxes(10, 10)
        post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "min_size": 3
        })
        sorted_boxes = SortedBoxes()
        ratio_list = [float(new_h) / ori_h, float(new_w) / ori_w]
        dt_boxes_list = post_func(det_out["concat_1.tmp_0"], [ratio_list])
        dt_boxes = filter_func(dt_boxes_list[0], [ori_h, ori_w])
        dt_boxes = sorted_boxes(dt_boxes)
        get_rotate_crop_image = GetRotateCropImage()
        img_list = []
        max_wh_ratio = 0
        for i, dtbox in enumerate(dt_boxes):
            boximg = get_rotate_crop_image(im, dt_boxes[i])
            img_list.append(boximg)
            h, w = boximg.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        if len(img_list) == 0:
            return [], []
        _, w, h = self.ocr_reader.resize_norm_img(img_list[0],
                                                  max_wh_ratio).shape
        imgs = np.zeros((len(img_list), 3, w, h)).astype('float32')
        for id, img in enumerate(img_list):
            norm_img = self.ocr_reader.resize_norm_img(img, max_wh_ratio)
            imgs[id] = norm_img
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
ocr_service.prepare_server(workdir="workdir", port=9292)
ocr_service.init_det_debugger(det_model_config="ocr_det_model")
if sys.argv[1] == 'gpu':
    ocr_service.run_debugger_service(gpu=True)
elif sys.argv[1] == 'cpu':
    ocr_service.run_debugger_service()
ocr_service.run_web_service()
