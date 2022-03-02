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
from tools.infer.predict_rec import TextRecognizer
from params import read_params

global_args = read_params()

if global_args.use_gpu:
    from paddle_serving_server_gpu.web_service import WebService
else:
    from paddle_serving_server.web_service import WebService


class TextRecognizerHelper(TextRecognizer):
    def __init__(self, args):
        super(TextRecognizerHelper, self).__init__(args)
        if self.loss_type == "ctc":
            self.fetch = ["save_infer_model/scale_0.tmp_0", "save_infer_model/scale_1.tmp_0"]

    def preprocess(self, img_list):
        img_num = len(img_list)
        args = {}
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))
        args["indices"] = indices
        predict_time = 0
        beg_img_no = 0
        end_img_no = img_num
        norm_img_batch = []
        max_wh_ratio = 0
        for ino in range(beg_img_no, end_img_no):
            h, w = img_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            if self.loss_type != "srn":
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            else:
                norm_img = self.process_image_srn(img_list[indices[ino]],
                                                  self.rec_image_shape, 8, 25,
                                                  self.char_ops)
                encoder_word_pos_list = []
                gsrm_word_pos_list = []
                gsrm_slf_attn_bias1_list = []
                gsrm_slf_attn_bias2_list = []
                encoder_word_pos_list.append(norm_img[1])
                gsrm_word_pos_list.append(norm_img[2])
                gsrm_slf_attn_bias1_list.append(norm_img[3])
                gsrm_slf_attn_bias2_list.append(norm_img[4])
                norm_img_batch.append(norm_img[0])
        norm_img_batch = np.concatenate(norm_img_batch, axis=0).copy()
        feed = {"image": norm_img_batch.copy()}
        return feed, self.fetch, args

    def postprocess(self, outputs, args):
        if self.loss_type == "ctc":
            rec_idx_batch = outputs[0]
            predict_batch = outputs[1]
            rec_idx_lod = args["save_infer_model/scale_0.tmp_0.lod"]
            predict_lod = args["save_infer_model/scale_1.tmp_0.lod"]
            indices = args["indices"]
            rec_res = [['', 0.0]] * (len(rec_idx_lod) - 1)
            for rno in range(len(rec_idx_lod) - 1):
                beg = rec_idx_lod[rno]
                end = rec_idx_lod[rno + 1]
                rec_idx_tmp = rec_idx_batch[beg:end, 0]
                preds_text = self.char_ops.decode(rec_idx_tmp)
                beg = predict_lod[rno]
                end = predict_lod[rno + 1]
                probs = predict_batch[beg:end, :]
                ind = np.argmax(probs, axis=1)
                blank = probs.shape[1]
                valid_ind = np.where(ind != (blank - 1))[0]
                if len(valid_ind) == 0:
                    continue
                score = np.mean(probs[valid_ind, ind[valid_ind]])
                rec_res[indices[rno]] = [preds_text, score]
        elif self.loss_type == 'srn':
            char_num = self.char_ops.get_char_num()
            preds = rec_idx_batch.reshape(-1)
            elapse = time.time() - starttime
            predict_time += elapse
            total_preds = preds.copy()
            for ino in range(int(len(rec_idx_batch) / self.text_len)):
                preds = total_preds[ino * self.text_len:(ino + 1) *
                                    self.text_len]
                ind = np.argmax(probs, axis=1)
                valid_ind = np.where(preds != int(char_num - 1))[0]
                if len(valid_ind) == 0:
                    continue
                score = np.mean(probs[valid_ind, ind[valid_ind]])
                preds = preds[:valid_ind[-1] + 1]
                preds_text = self.char_ops.decode(preds)
                rec_res[indices[ino]] = [preds_text, score]
        else:
            for rno in range(len(rec_idx_batch)):
                end_pos = np.where(rec_idx_batch[rno, :] == 1)[0]
                if len(end_pos) <= 1:
                    preds = rec_idx_batch[rno, 1:]
                    score = np.mean(predict_batch[rno, 1:])
                else:
                    preds = rec_idx_batch[rno, 1:end_pos[1]]
                    score = np.mean(predict_batch[rno, 1:end_pos[1]])
                preds_text = self.char_ops.decode(preds)
                rec_res[indices[rno]] = [preds_text, score]
        return rec_res


class OCRService(WebService):
    def init_rec(self):
        self.text_recognizer = TextRecognizerHelper(global_args)

    def preprocess(self, feed=[], fetch=[]):
        # TODO: to handle batch rec images
        img_list = []
        for feed_data in feed:
            data = base64.b64decode(feed_data["image"].encode('utf8'))
            data = np.fromstring(data, np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img_list.append(im)
        feed, fetch, self.tmp_args = self.text_recognizer.preprocess(img_list)
        return feed, fetch

    def postprocess(self, feed={}, fetch=[], fetch_map=None):
        outputs = [fetch_map[x] for x in self.text_recognizer.fetch]
        for x in fetch_map.keys():
            if ".lod" in x:
                self.tmp_args[x] = fetch_map[x]
        rec_res = self.text_recognizer.postprocess(outputs, self.tmp_args)
        res = []
        for i in range(len(rec_res)):
            res.append({
                "text": rec_res[i][0],
                "confidence": float(rec_res[i][1])
            })
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
