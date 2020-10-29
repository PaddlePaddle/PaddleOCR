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
import time
import re
import base64
from clas_local_server import TextClassifierHelper
from det_local_server import TextDetectorHelper
from rec_local_server import TextRecognizerHelper
from tools.infer.predict_system import TextSystem, sorted_boxes
from paddle_serving_app.local_predict import Debugger
import copy
from params import read_params

global_args = read_params()

if global_args.use_gpu:
    from paddle_serving_server_gpu.web_service import WebService
else:
    from paddle_serving_server.web_service import WebService


class TextSystemHelper(TextSystem):
    def __init__(self, args):
        self.text_detector = TextDetectorHelper(args)
        self.text_recognizer = TextRecognizerHelper(args)
        self.use_angle_cls = args.use_angle_cls
        if self.use_angle_cls:
            self.clas_client = Debugger()
            self.clas_client.load_model_config(
                global_args.cls_server_dir, gpu=True, profile=False)
            self.text_classifier = TextClassifierHelper(args)
        self.det_client = Debugger()
        self.det_client.load_model_config(
            global_args.det_server_dir, gpu=True, profile=False)
        self.fetch = ["save_infer_model/scale_0.tmp_0", "save_infer_model/scale_1.tmp_0"]

    def preprocess(self, img):
        feed, fetch, self.tmp_args = self.text_detector.preprocess(img)
        fetch_map = self.det_client.predict(feed, fetch)
        outputs = [fetch_map[x] for x in fetch]
        dt_boxes = self.text_detector.postprocess(outputs, self.tmp_args)
        if dt_boxes is None:
            return None, None
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        self.dt_boxes = dt_boxes
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            feed, fetch, self.tmp_args = self.text_classifier.preprocess(
                img_crop_list)
            fetch_map = self.clas_client.predict(feed, fetch)
            outputs = [fetch_map[x] for x in self.text_classifier.fetch]
            for x in fetch_map.keys():
                if ".lod" in x:
                    self.tmp_args[x] = fetch_map[x]
            img_crop_list, _ = self.text_classifier.postprocess(outputs,
                                                                self.tmp_args)
        feed, fetch, self.tmp_args = self.text_recognizer.preprocess(
            img_crop_list)
        return feed, self.fetch, self.tmp_args

    def postprocess(self, outputs, args):
        return self.text_recognizer.postprocess(outputs, args)


class OCRService(WebService):
    def init_rec(self):
        self.text_system = TextSystemHelper(global_args)

    def preprocess(self, feed=[], fetch=[]):
        # TODO: to handle batch rec images
        data = base64.b64decode(feed[0]["image"].encode('utf8'))
        data = np.fromstring(data, np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        feed, fetch, self.tmp_args = self.text_system.preprocess(im)
        return feed, fetch

    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        outputs = [fetch_map[x] for x in self.text_system.fetch]
        for x in fetch_map.keys():
            if ".lod" in x:
                self.tmp_args[x] = fetch_map[x]
        rec_res = self.text_system.postprocess(outputs, self.tmp_args)
        res = []
        for i in range(len(rec_res)):
            tmp_res = {
                "text_region": self.text_system.dt_boxes[i].tolist(),
                "text": rec_res[i][0],
                "confidence": float(rec_res[i][1])
            }
            res.append(tmp_res)
        return res


if __name__ == "__main__":
    ocr_service = OCRService(name="ocr")
    ocr_service.load_model_config(global_args.rec_server_dir)
    ocr_service.init_rec()
    if global_args.use_gpu:
        ocr_service.prepare_server(
            workdir="workdir", port=9292, device="gpu", gpuid=0)
    else:
        ocr_service.prepare_server(workdir="workdir", port=9292, device="cpu")
    ocr_service.run_debugger_service()
    ocr_service.run_web_service()
