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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.visual import draw_ser_results
from ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps
import tools.program as program


def to_tensor(data):
    import numbers
    from collections import defaultdict
    data_dict = defaultdict(list)
    to_tensor_idxs = []
    for idx, v in enumerate(data):
        if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
            if idx not in to_tensor_idxs:
                to_tensor_idxs.append(idx)
        data_dict[idx].append(v)
    for idx in to_tensor_idxs:
        data_dict[idx] = paddle.to_tensor(data_dict[idx])
    return list(data_dict.values())


class SerPredictor(object):
    def __init__(self, config):
        global_config = config['Global']

        # build post process
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)

        # build model
        self.model = build_model(config['Architecture'])

        load_model(
            config, self.model, model_type=config['Architecture']["model_type"])

        from paddleocr import PaddleOCR

        self.ocr_engine = PaddleOCR(use_angle_cls=False, show_log=False)

        # create data ops
        transforms = []
        for op in config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                op[op_name]['ocr_engine'] = self.ocr_engine
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = [
                    'input_ids', 'labels', 'bbox', 'image', 'attention_mask',
                    'token_type_ids', 'segment_offset_id', 'ocr_info',
                    'entities'
                ]

            transforms.append(op)
        global_config['infer_mode'] = True
        self.ops = create_operators(config['Eval']['dataset']['transforms'],
                                    global_config)
        self.model.eval()

    def __call__(self, img_path):
        with open(img_path, 'rb') as f:
            img = f.read()
            data = {'image': img}
        batch = transform(data, self.ops)
        batch = to_tensor(batch)
        preds = self.model(batch)
        post_result = self.post_process_class(
            preds,
            attention_masks=batch[4],
            segment_offset_ids=batch[6],
            ocr_infos=batch[7])
        return post_result, batch


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    os.makedirs(config['Global']['save_res_path'], exist_ok=True)

    ser_engine = SerPredictor(config)

    infer_imgs = get_image_file_list(config['Global']['infer_img'])
    with open(
            os.path.join(config['Global']['save_res_path'],
                         "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        for idx, img_path in enumerate(infer_imgs):
            save_img_path = os.path.join(
                config['Global']['save_res_path'],
                os.path.splitext(os.path.basename(img_path))[0] + "_ser.jpg")
            logger.info("process: [{}/{}], save result to {}".format(
                idx, len(infer_imgs), save_img_path))

            result, _ = ser_engine(img_path)
            result = result[0]
            fout.write(img_path + "\t" + json.dumps(
                {
                    "ocr_info": result,
                }, ensure_ascii=False) + "\n")
            img_res = draw_ser_results(img_path, result)
            cv2.imwrite(save_img_path, img_res)
