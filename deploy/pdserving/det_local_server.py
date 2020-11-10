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
from tools.infer.predict_det import TextDetector
from params import read_params

global_args = read_params()
if global_args.use_gpu:
    from paddle_serving_server_gpu.web_service import WebService
else:
    from paddle_serving_server.web_service import WebService


class TextDetectorHelper(TextDetector):
    def __init__(self, args):
        super(TextDetectorHelper, self).__init__(args)
        if self.det_algorithm == "SAST":
            self.fetch = [
                "bn_f_border4.output.tmp_2", "bn_f_tco4.output.tmp_2",
                "bn_f_tvo4.output.tmp_2", "sigmoid_0.tmp_0"
            ]
        elif self.det_algorithm == "EAST":
            self.fetch = ["sigmoid_0.tmp_0", "tmp_2"]
        elif self.det_algorithm == "DB":
            self.fetch = ["save_infer_model/scale_0.tmp_0"]

    def preprocess(self, img):
        img = img.copy()
        im, ratio_list = self.preprocess_op(img)
        if im is None:
            return None, 0
        return {
            "image": im.copy()
        }, self.fetch, {
            "ratio_list": [ratio_list],
            "ori_im": img
        }

    def postprocess(self, outputs, args):
        outs_dict = {}
        if self.det_algorithm == "EAST":
            outs_dict['f_geo'] = outputs[0]
            outs_dict['f_score'] = outputs[1]
        elif self.det_algorithm == 'SAST':
            outs_dict['f_border'] = outputs[0]
            outs_dict['f_score'] = outputs[1]
            outs_dict['f_tco'] = outputs[2]
            outs_dict['f_tvo'] = outputs[3]
        else:
            outs_dict['maps'] = outputs[0]
        dt_boxes_list = self.postprocess_op(outs_dict, args["ratio_list"])
        dt_boxes = dt_boxes_list[0]
        if self.det_algorithm == "SAST" and self.det_sast_polygon:
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes,
                                                         args["ori_im"].shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, args["ori_im"].shape)
        return dt_boxes


class DetService(WebService):
    def init_det(self):
        self.text_detector = TextDetectorHelper(global_args)

    def preprocess(self, feed=[], fetch=[]):
        data = base64.b64decode(feed[0]["image"].encode('utf8'))
        data = np.fromstring(data, np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        feed, fetch, self.tmp_args = self.text_detector.preprocess(im)
        return feed, fetch

    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        outputs = [fetch_map[x] for x in fetch]
        det_res = self.text_detector.postprocess(outputs, self.tmp_args)
        res = []
        for i in range(len(det_res)):
            res.append({"text_region": det_res[i].tolist()})
        return res

if __name__ == "__main__":
    ocr_service = DetService(name="ocr")
    ocr_service.load_model_config(global_args.det_server_dir)
    ocr_service.init_det()
    if global_args.use_gpu:
        ocr_service.prepare_server(
            workdir="workdir", port=9292, device="gpu", gpuid=0)
    else:
        ocr_service.prepare_server(workdir="workdir", port=9292, device="cpu")
    ocr_service.run_debugger_service()
    ocr_service.run_web_service()
