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
from tools.infer.predict_cls import TextClassifier
from params import read_params

global_args = read_params()
if global_args.use_gpu:
    from paddle_serving_server_gpu.web_service import WebService
else:
    from paddle_serving_server.web_service import WebService


class TextClassifierHelper(TextClassifier):
    def __init__(self, args):
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.rec_batch_num
        self.label_list = args.label_list
        self.cls_thresh = args.cls_thresh
        self.fetch = [
            "save_infer_model/scale_0.tmp_0", "save_infer_model/scale_1.tmp_0"
        ]

    def preprocess(self, img_list):
        args = {}
        img_num = len(img_list)
        args["img_list"] = img_list
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))
        args["indices"] = indices
        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        predict_time = 0
        beg_img_no, end_img_no = 0, img_num
        norm_img_batch = []
        max_wh_ratio = 0
        for ino in range(beg_img_no, end_img_no):
            h, w = img_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            norm_img = self.resize_norm_img(img_list[indices[ino]])
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        norm_img_batch = np.concatenate(norm_img_batch)
        feed = {"image": norm_img_batch.copy()}
        return feed, self.fetch, args

    def postprocess(self, outputs, args):
        prob_out = outputs[0]
        label_out = outputs[1]
        indices = args["indices"]
        img_list = args["img_list"]
        cls_res = [['', 0.0]] * len(label_out)
        if len(label_out.shape) != 1:
            prob_out, label_out = label_out, prob_out
        for rno in range(len(label_out)):
            label_idx = label_out[rno]
            score = prob_out[rno][label_idx]
            label = self.label_list[label_idx]
            cls_res[indices[rno]] = [label, score]
            if '180' in label and score > self.cls_thresh:
                img_list[indices[rno]] = cv2.rotate(img_list[indices[rno]], 1)
        return img_list, cls_res


class OCRService(WebService):
    def init_rec(self):
        self.text_classifier = TextClassifierHelper(global_args)

    def preprocess(self, feed=[], fetch=[]):
        img_list = []
        for feed_data in feed:
            data = base64.b64decode(feed_data["image"].encode('utf8'))
            data = np.fromstring(data, np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img_list.append(im)
        feed, fetch, self.tmp_args = self.text_classifier.preprocess(img_list)
        return feed, fetch

    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        outputs = [fetch_map[x] for x in self.text_classifier.fetch]
        for x in fetch_map.keys():
            if ".lod" in x:
                self.tmp_args[x] = fetch_map[x]
        _, rec_res = self.text_classifier.postprocess(outputs, self.tmp_args)
        res = []
        for i in range(len(rec_res)):
            res.append({
                "direction": rec_res[i][0],
                "confidence": float(rec_res[i][1])
            })
        return res


if __name__ == "__main__":
    ocr_service = OCRService(name="ocr")
    ocr_service.load_model_config(global_args.cls_server_dir)
    ocr_service.init_rec()
    if global_args.use_gpu:
        ocr_service.prepare_server(
            workdir="workdir", port=9292, device="gpu", gpuid=0)
    else:
        ocr_service.prepare_server(workdir="workdir", port=9292, device="cpu")
    ocr_service.run_debugger_service()
    ocr_service.run_web_service()
