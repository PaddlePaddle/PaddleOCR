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
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle
from paddle.jit import to_static

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.visual import draw_rectangle
from tools.infer.utility import draw_boxes
import tools.program as program
import cv2


@paddle.no_grad()
def main(config, device, logger, vdl_writer):
    global_config = config['Global']

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))

    model = build_model(config['Architecture'])
    algorithm = config['Architecture']['algorithm']

    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Encode' in op_name:
            continue
        if op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    os.makedirs(save_res_path, exist_ok=True)

    model.eval()
    with open(
            os.path.join(save_res_path, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        for file in get_image_file_list(config['Global']['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)

            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds, [shape_list])

            structure_str_list = post_result['structure_batch_list'][0]
            bbox_list = post_result['bbox_batch_list'][0]
            structure_str_list = structure_str_list[0]
            structure_str_list = [
                '<html>', '<body>', '<table>'
            ] + structure_str_list + ['</table>', '</body>', '</html>']
            bbox_list_str = json.dumps(bbox_list.tolist())

            logger.info("result: {}, {}".format(structure_str_list,
                                                bbox_list_str))
            f_w.write("result: {}, {}\n".format(structure_str_list,
                                                bbox_list_str))

            if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
                img = draw_rectangle(file, bbox_list)
            else:
                img = draw_boxes(cv2.imread(file), bbox_list)
            cv2.imwrite(
                os.path.join(save_res_path, os.path.basename(file)), img)
            logger.info('save result to {}'.format(save_res_path))
        logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main(config, device, logger, vdl_writer)
