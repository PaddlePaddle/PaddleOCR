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
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import math
import time
import traceback
import paddle

import tools.infer.utility as utility
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from test.utility import parse_args

logger = get_logger()


class TableStructurer(object):
    def __init__(self, args):
        pre_process_list = [{
            'ResizeTableImage': {
                'max_len': args.structure_max_len
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'PaddingTableImage': None
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image']
            }
        }]
        postprocess_params = {
            'name': 'TableLabelDecode',
            "character_type": args.structure_char_type,
            "character_dict_path": args.structure_char_dict_path,
        }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'structure', logger)

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        starttime = time.time()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        preds['structure_probs'] = outputs[1]
        preds['loc_preds'] = outputs[0]

        post_result = self.postprocess_op(preds)

        structure_str_list = post_result['structure_str_list']
        res_loc = post_result['res_loc']
        imgh, imgw = ori_im.shape[0:2]
        res_loc_final = []
        for rno in range(len(res_loc[0])):
            x0, y0, x1, y1 = res_loc[0][rno]
            left = max(int(imgw * x0), 0)
            top = max(int(imgh * y0), 0)
            right = min(int(imgw * x1), imgw - 1)
            bottom = min(int(imgh * y1), imgh - 1)
            res_loc_final.append([left, top, right, bottom])

        structure_str_list = structure_str_list[0][:-1]
        structure_str_list = ['<html>', '<body>', '<table>'] + structure_str_list + ['</table>', '</body>', '</html>']

        elapse = time.time() - starttime
        return (structure_str_list, res_loc_final), elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    table_structurer = TableStructurer(args)
    count = 0
    total_time = 0
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        structure_res, elapse = table_structurer(img)

        logger.info("result: {}".format(structure_res))

        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    main(parse_args())
