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
import tools.program as program
import cv2


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

    load_model(config, model)

    # create data ops
    transforms = []
    use_padding = False
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        if op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image']
        if op_name == "ResizeTableImage":
            use_padding = True
            padding_max_len = op['ResizeTableImage']['max_len']
        transforms.append(op)

    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    model.eval()
    for file in get_image_file_list(config['Global']['infer_img']):
        logger.info("infer_img: {}".format(file))
        with open(file, 'rb') as f:
            img = f.read()
            data = {'image': img}
        batch = transform(data, ops)
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        preds = model(images)
        post_result = post_process_class(preds)
        res_html_code = post_result['res_html_code']
        res_loc = post_result['res_loc']
        img = cv2.imread(file)
        imgh, imgw = img.shape[0:2]
        res_loc_final = []
        for rno in range(len(res_loc[0])):
            x0, y0, x1, y1 = res_loc[0][rno]
            left = max(int(imgw * x0), 0)
            top = max(int(imgh * y0), 0)
            right = min(int(imgw * x1), imgw - 1)
            bottom = min(int(imgh * y1), imgh - 1)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            res_loc_final.append([left, top, right, bottom])
        res_loc_str = json.dumps(res_loc_final)
        logger.info("result: {}, {}".format(res_html_code, res_loc_final))
    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main(config, device, logger, vdl_writer)
