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
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForRelationExtraction

# relative reference
from utils import parse_args, get_image_file_list, draw_re_results
from infer_ser_e2e import SerPredictor


def make_input(ser_input, ser_result, max_seq_len=512):
    entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}

    entities = ser_input['entities'][0]
    assert len(entities) == len(ser_result)

    # entities
    start = []
    end = []
    label = []
    entity_idx_dict = {}
    for i, (res, entity) in enumerate(zip(ser_result, entities)):
        if res['pred'] == 'O':
            continue
        entity_idx_dict[len(start)] = i
        start.append(entity['start'])
        end.append(entity['end'])
        label.append(entities_labels[res['pred']])
    entities = dict(start=start, end=end, label=label)

    # relations
    head = []
    tail = []
    for i in range(len(entities["label"])):
        for j in range(len(entities["label"])):
            if entities["label"][i] == 1 and entities["label"][j] == 2:
                head.append(i)
                tail.append(j)

    relations = dict(head=head, tail=tail)

    batch_size = ser_input["input_ids"].shape[0]
    entities_batch = []
    relations_batch = []
    for b in range(batch_size):
        entities_batch.append(entities)
        relations_batch.append(relations)

    ser_input['entities'] = entities_batch
    ser_input['relations'] = relations_batch

    ser_input.pop('segment_offset_id')
    return ser_input, entity_idx_dict


class SerReSystem(object):
    def __init__(self, args):
        self.ser_engine = SerPredictor(args)
        self.tokenizer = LayoutXLMTokenizer.from_pretrained(
            args.re_model_name_or_path)
        self.model = LayoutXLMForRelationExtraction.from_pretrained(
            args.re_model_name_or_path)
        self.model.eval()

    def __call__(self, img):
        ser_result, ser_inputs = self.ser_engine(img)
        re_input, entity_idx_dict = make_input(ser_inputs, ser_result)

        re_result = self.model(**re_input)

        pred_relations = re_result['pred_relations'][0]
        # 进行 relations 到 ocr信息的转换
        result = []
        used_tail_id = []
        for relation in pred_relations:
            if relation['tail_id'] in used_tail_id:
                continue
            used_tail_id.append(relation['tail_id'])
            ocr_info_head = ser_result[entity_idx_dict[relation['head_id']]]
            ocr_info_tail = ser_result[entity_idx_dict[relation['tail_id']]]
            result.append((ocr_info_head, ocr_info_tail))

        return result


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # get infer img list
    infer_imgs = get_image_file_list(args.infer_imgs)

    # loop for infer
    ser_re_engine = SerReSystem(args)
    with open(
            os.path.join(args.output_dir, "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        for idx, img_path in enumerate(infer_imgs):
            save_img_path = os.path.join(
                args.output_dir,
                os.path.splitext(os.path.basename(img_path))[0] + "_re.jpg")
            print("process: [{}/{}], save_result to {}".format(
                idx, len(infer_imgs), save_img_path))

            img = cv2.imread(img_path)

            result = ser_re_engine(img)
            fout.write(img_path + "\t" + json.dumps(
                {
                    "result": result,
                }, ensure_ascii=False) + "\n")

            img_res = draw_re_results(img, result)
            cv2.imwrite(save_img_path, img_res)
