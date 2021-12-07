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

# relative reference
from utils import parse_args, get_image_file_list, draw_ser_results, get_bio_label_maps, build_ocr_engine

from utils import pad_sentences, split_page, preprocess, postprocess, merge_preds_list_with_ocr_info


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


@paddle.no_grad()
def infer(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # init token and model
    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)
    model = LayoutXLMForTokenClassification.from_pretrained(
        args.model_name_or_path)
    model.eval()

    label2id_map, id2label_map = get_bio_label_maps(args.label_map_path)
    label2id_map_for_draw = dict()
    for key in label2id_map:
        if key.startswith("I-"):
            label2id_map_for_draw[key] = label2id_map["B" + key[1:]]
        else:
            label2id_map_for_draw[key] = label2id_map[key]

    # get infer img list
    infer_imgs = get_image_file_list(args.infer_imgs)

    ocr_engine = build_ocr_engine(args.ocr_rec_model_dir,
                                  args.ocr_det_model_dir)

    # loop for infer
    with open(os.path.join(args.output_dir, "infer_results.txt"), "w") as fout:
        for idx, img_path in enumerate(infer_imgs):
            print("process: [{}/{}]".format(idx, len(infer_imgs), img_path))

            img = cv2.imread(img_path)

            ocr_result = ocr_engine.ocr(img_path, cls=False)

            ocr_info = parse_ocr_info_for_ser(ocr_result)

            inputs = preprocess(
                tokenizer=tokenizer,
                ori_img=img,
                ocr_info=ocr_info,
                max_seq_len=args.max_seq_length)

            outputs = model(
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                image=inputs["image"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"])

            preds = outputs[0]
            preds = postprocess(inputs["attention_mask"], preds, id2label_map)
            ocr_info = merge_preds_list_with_ocr_info(
                ocr_info, inputs["segment_offset_id"], preds,
                label2id_map_for_draw)

            fout.write(img_path + "\t" + json.dumps(
                {
                    "ocr_info": ocr_info,
                }, ensure_ascii=False) + "\n")

            img_res = draw_ser_results(img, ocr_info)
            cv2.imwrite(
                os.path.join(args.output_dir,
                             os.path.splitext(os.path.basename(img_path))[0] +
                             "_ser.jpg"), img_res)

    return


if __name__ == "__main__":
    args = parse_args()
    infer(args)
