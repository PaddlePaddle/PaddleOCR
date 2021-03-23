# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
try:
    from paddle_serving_server_gpu.web_service import WebService, Op
except ImportError:
    from paddle_serving_server.web_service import WebService, Op

import logging
import numpy as np
import cv2
import base64
# from paddle_serving_app.reader import OCRReader
from ocr_reader import OCRReader, DetResizeForTest
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes

_LOGGER = logging.getLogger()


class DetOp(Op):
    def init_op(self):
        self.det_preprocess = Sequential([
            DetResizeForTest(), Div(255),
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

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        data = base64.b64decode(input_dict["image"].encode('utf8'))
        data = np.fromstring(data, np.uint8)
        # Note: class variables(self.var) can only be used in process op mode
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self.im = im
        self.ori_h, self.ori_w, _ = im.shape

        det_img = self.det_preprocess(self.im)
        _, self.new_h, self.new_w = det_img.shape
        print("det image shape", det_img.shape)
        return {"x": det_img[np.newaxis, :].copy()}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, log_id):
        print("input_dicts: ", input_dicts)
        det_out = fetch_dict["save_infer_model/scale_0.tmp_1"]
        ratio_list = [
            float(self.new_h) / self.ori_h, float(self.new_w) / self.ori_w
        ]
        dt_boxes_list = self.post_func(det_out, [ratio_list])
        dt_boxes = self.filter_func(dt_boxes_list[0], [self.ori_h, self.ori_w])
        out_dict = {"dt_boxes": dt_boxes, "image": self.im}

        print("out dict", out_dict["dt_boxes"])
        return out_dict, None, ""


class RecOp(Op):
    def init_op(self):
        self.ocr_reader = OCRReader(
            char_dict_path="../../ppocr/utils/ppocr_keys_v1.txt")

        self.get_rotate_crop_image = GetRotateCropImage()
        self.sorted_boxes = SortedBoxes()

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        im = input_dict["image"]
        dt_boxes = input_dict["dt_boxes"]
        dt_boxes = self.sorted_boxes(dt_boxes)
        feed_list = []
        img_list = []
        max_wh_ratio = 0
        for i, dtbox in enumerate(dt_boxes):
            boximg = self.get_rotate_crop_image(im, dt_boxes[i])
            img_list.append(boximg)
            h, w = boximg.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        _, w, h = self.ocr_reader.resize_norm_img(img_list[0],
                                                  max_wh_ratio).shape

        imgs = np.zeros((len(img_list), 3, w, h)).astype('float32')
        for id, img in enumerate(img_list):
            norm_img = self.ocr_reader.resize_norm_img(img, max_wh_ratio)
            imgs[id] = norm_img
        print("rec image shape", imgs.shape)
        feed = {"x": imgs.copy()}
        return feed, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, log_id):
        rec_res = self.ocr_reader.postprocess(fetch_dict, with_score=True)
        res_lst = []
        for res in rec_res:
            res_lst.append(res[0])
        res = {"res": str(res_lst)}
        return res, None, ""


class OcrService(WebService):
    def get_pipeline_response(self, read_op):
        det_op = DetOp(name="det", input_ops=[read_op])
        rec_op = RecOp(name="rec", input_ops=[det_op])
        return rec_op


uci_service = OcrService(name="ocr")
uci_service.prepare_pipeline_config("config.yml")
uci_service.run_service()
