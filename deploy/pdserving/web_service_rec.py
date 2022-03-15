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
from paddle_serving_server.web_service import WebService, Op

import logging
import numpy as np
import cv2
import base64
# from paddle_serving_app.reader import OCRReader
from ocr_reader import OCRReader, DetResizeForTest, ArgsParser
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose

_LOGGER = logging.getLogger()


class RecOp(Op):
    def init_op(self):
        self.ocr_reader = OCRReader(
            char_dict_path="../../ppocr/utils/ppocr_keys_v1.txt")

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        raw_im = base64.b64decode(input_dict["image"].encode('utf8'))
        data = np.fromstring(raw_im, np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        feed_list = []
        max_wh_ratio = 0
        ## Many mini-batchs, the type of feed_data is list.
        max_batch_size = 6  # len(dt_boxes)

        # If max_batch_size is 0, skipping predict stage
        if max_batch_size == 0:
            return {}, True, None, ""
        boxes_size = max_batch_size
        rem = boxes_size % max_batch_size

        h, w = im.shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        _, w, h = self.ocr_reader.resize_norm_img(im, max_wh_ratio).shape
        norm_img = self.ocr_reader.resize_norm_img(im, max_batch_size)
        norm_img = norm_img[np.newaxis, :]
        feed = {"x": norm_img.copy()}
        feed_list.append(feed)
        return feed_list, False, None, ""

    def postprocess(self, input_dicts, fetch_data, data_id, log_id):
        res_list = []
        if isinstance(fetch_data, dict):
            if len(fetch_data) > 0:
                rec_batch_res = self.ocr_reader.postprocess(
                    fetch_data, with_score=True)
                for res in rec_batch_res:
                    res_list.append(res[0])
        elif isinstance(fetch_data, list):
            for one_batch in fetch_data:
                one_batch_res = self.ocr_reader.postprocess(
                    one_batch, with_score=True)
                for res in one_batch_res:
                    res_list.append(res[0])

        res = {"res": str(res_list)}
        return res, None, ""


class OcrService(WebService):
    def get_pipeline_response(self, read_op):
        rec_op = RecOp(name="rec", input_ops=[read_op])
        print("rec op:", rec_op)
        return rec_op


uci_service = OcrService(name="ocr")
FLAGS = ArgsParser().parse_args()
uci_service.prepare_pipeline_config(yml_dict=FLAGS.conf_dict)
uci_service.run_service()