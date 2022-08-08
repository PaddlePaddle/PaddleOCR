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
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time

import tools.infer.utility as utility
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppstructure.utility import parse_args
from picodet_postprocess import PicoDetPostProcess

logger = get_logger()


class LayoutPredictor(object):
    def __init__(self, args):
        pre_process_list = [{
            'Resize': {
                'size': [800, 608]
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image']
            }
        }]
        # postprocess_params = {
        #     'name': 'LayoutPostProcess',
        #     "character_dict_path": args.layout_dict_path,
        # }

        self.preprocess_op = create_operators(pre_process_list)
        # self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'layout', logger)

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]

        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        img = img.copy()

        preds, elapse = 0, 1
        starttime = time.time()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()

        # outputs = []
        # for output_tensor in self.output_tensors:
        #     output = output_tensor.copy_to_cpu()
        #     outputs.append(output)
        np_score_list, np_boxes_list = [], []
        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        for out_idx in range(num_outs):
            np_score_list.append(
                self.predictor.get_output_handle(output_names[out_idx])
                .copy_to_cpu())
            np_boxes_list.append(
                self.predictor.get_output_handle(output_names[
                    out_idx + num_outs]).copy_to_cpu())
        # result = dict(boxes=np_score_list, boxes_num=np_boxes_list)
        postprocessor = PicoDetPostProcess(
            (800, 608), [[800., 608.]],
            np.array([[1.010101, 0.99346405]]),
            strides=[8, 16, 32, 64],
            nms_threshold=0.5)
        np_boxes, np_boxes_num = postprocessor(np_score_list, np_boxes_list)
        result = dict(boxes=np_boxes, boxes_num=np_boxes_num)
        # print(result)
        im_bboxes_num = result['boxes_num'][0]
        # print('im_bboxes_num:',im_bboxes_num)

        bboxs = result['boxes'][0:0 + im_bboxes_num, :]
        threshold = 0.5
        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        preds = []

        id2label = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}
        for dt in np_boxes:
            clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
            label = id2label[clsid + 1]
            result_di = {'bbox': bbox, 'label': label}
            preds.append(result_di)
            # print('result_di',result_di)
            # print('clsid, bbox, score:',clsid, bbox, score)

        elapse = time.time() - starttime
        return preds, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    layout_predictor = LayoutPredictor(args)
    count = 0
    total_time = 0

    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        layout_res, elapse = layout_predictor(img)

        logger.info("result: {}".format(layout_res))

        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    main(parse_args())
