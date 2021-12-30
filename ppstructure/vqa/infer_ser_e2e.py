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

import os
import sys
import json
import cv2
import numpy as np
from copy import deepcopy
from PIL import Image

import paddle
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForTokenClassification
from paddlenlp.transformers import LayoutLMModel, LayoutLMTokenizer, LayoutLMForTokenClassification

# relative reference
from utils import parse_args, get_image_file_list, draw_ser_results, get_bio_label_maps

from utils import pad_sentences, split_page, preprocess, postprocess, merge_preds_list_with_ocr_info

MODELS = {
    'LayoutXLM':
    (LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForTokenClassification),
    'LayoutLM':
    (LayoutLMTokenizer, LayoutLMModel, LayoutLMForTokenClassification)
}


def trans_poly_to_bbox(poly):
    x1 = np.min([p[0] for p in poly])
    x2 = np.max([p[0] for p in poly])
    y1 = np.min([p[1] for p in poly])
    y2 = np.max([p[1] for p in poly])
    return [x1, y1, x2, y2]


def parse_ocr_info_for_ser(ocr_result):
    ocr_info = []
    for res in ocr_result:
        ocr_info.append({
            "text": res[1][0],
            "bbox": trans_poly_to_bbox(res[0]),
            "poly": res[0],
        })
    return ocr_info


class SerPredictor(object):
    def __init__(self, args):
        self.args = args
        self.max_seq_length = args.max_seq_length

        # init ser token and model
        tokenizer_class, base_model_class, model_class = MODELS[
            args.ser_model_type]
        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path)
        self.model = model_class.from_pretrained(args.model_name_or_path)
        self.model.eval()

        # init ocr_engine
        from paddleocr import PaddleOCR

        self.ocr_engine = PaddleOCR(
            rec_model_dir=args.rec_model_dir,
            det_model_dir=args.det_model_dir,
            use_angle_cls=False,
            show_log=False)
        # init dict
        label2id_map, self.id2label_map = get_bio_label_maps(
            args.label_map_path)
        self.label2id_map_for_draw = dict()
        for key in label2id_map:
            if key.startswith("I-"):
                self.label2id_map_for_draw[key] = label2id_map["B" + key[1:]]
            else:
                self.label2id_map_for_draw[key] = label2id_map[key]

    def __call__(self, img):
        ocr_result = self.ocr_engine.ocr(img, cls=False)

        ocr_info = parse_ocr_info_for_ser(ocr_result)

        inputs = preprocess(
            tokenizer=self.tokenizer,
            ori_img=img,
            ocr_info=ocr_info,
            max_seq_len=self.max_seq_length)

        if self.args.ser_model_type == 'LayoutLM':
            preds = self.model(
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"])
        elif self.args.ser_model_type == 'LayoutXLM':
            preds = self.model(
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                image=inputs["image"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"])
            preds = preds[0]

        preds = postprocess(inputs["attention_mask"], preds, self.id2label_map)
        ocr_info = merge_preds_list_with_ocr_info(
            ocr_info, inputs["segment_offset_id"], preds,
            self.label2id_map_for_draw)
        return ocr_info, inputs


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # get infer img list
    infer_imgs = get_image_file_list(args.infer_imgs)

    # loop for infer
    ser_engine = SerPredictor(args)
    with open(
            os.path.join(args.output_dir, "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        for idx, img_path in enumerate(infer_imgs):
            save_img_path = os.path.join(
                args.output_dir,
                os.path.splitext(os.path.basename(img_path))[0] + "_ser.jpg")
            print("process: [{}/{}], save result to {}".format(
                idx, len(infer_imgs), save_img_path))

            img = cv2.imread(img_path)

            result, _ = ser_engine(img)
            fout.write(img_path + "\t" + json.dumps(
                {
                    "ser_resule": result,
                }, ensure_ascii=False) + "\n")

            img_res = draw_ser_results(img, result)
            cv2.imwrite(save_img_path, img_res)
