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
import time
import sys
import paddle

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
import json
logger = get_logger()


class KIE(object):
    def __init__(self, args):
        self.args = args
        pre_process_list = [{
            'DecodeImage': {
                'img_mode': 'RGB',
                'channel_first': False,
            }
        }, {
            'NormalizeImage': {
                'std': [ 58.395, 57.12, 57.375 ],
                'mean': [ 123.675, 116.28, 103.53 ],
                'scale': '1',
                'order': 'hwc'
            }
        }, {
            'KieLabelEncode': {
                'character_dict_path': './train_data/wildreceipt/dict.txt'
                }
        }, {
            'KieResize': None
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'relations', 'texts', 'points', 'tag', 'shape']
            }
        }]

        self.preprocess_op = create_operators(pre_process_list)

        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
            args, 'kie', logger)

        # build post_process
    
    def bbox2roi(self, bbox_list):
        rois_list = []
        rois_num = []
        for img_id, bboxes in enumerate(bbox_list):
            rois_num.append(bboxes.shape[0])
            rois_list.append(bboxes)
        rois = np.concatenate(rois_list, 0)
        rois_num = np.array(rois_num, dtype='int32')
        return rois, rois_num

    def pre_process(self, img, relations, texts, gt_bboxes, tag, img_size):
        # img, relations, texts, gt_bboxes, tag, img_size = img.numpy(
        # ), relations.numpy(), texts.numpy(), gt_bboxes.numpy(), tag.numpy(
        # ).tolist(), img_size.numpy()

        temp_relations, temp_texts, temp_gt_bboxes = [], [], []
        h, w = int(np.max(img_size[:, 0])), int(np.max(img_size[:, 1]))
        img = np.array(img[:, :, :h, :w])
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            temp_relations.append(
                np.array(
                    relations[i, :num, :num, :], dtype='float32'))
            temp_texts.append(
                np.array(
                    texts[i, :num, :recoder_len], dtype='float32'))
            temp_gt_bboxes.append(
                np.array(
                    gt_bboxes[i, :num, ...], dtype='float32'))
        return img, temp_relations, temp_texts, temp_gt_bboxes


    def _preprocess(self, inputs):
        img = inputs[0]
        relations, texts, gt_bboxes, tag, img_size = inputs[1], inputs[
            2], inputs[3], inputs[4], inputs[-1]
        # padding 
        img = img[np.newaxis,:, :, :]
        relations = relations[np.newaxis]
        texts = texts[np.newaxis]
        gt_bboxes = gt_bboxes[np.newaxis]
        tag = [tag]
        img_size = np.array([img_size])

        img, relations, texts, gt_bboxes = self.pre_process(
            img, relations, texts, gt_bboxes, tag, img_size)
        
        boxes, rois_num = self.bbox2roi(gt_bboxes)
        return img, relations[0], texts[0], boxes, rois_num
    

    def forward(self, inputs):

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(
                input_names[i])
            input_tensor.copy_from_cpu(inputs[i])
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
    
        return outputs

def decode_from_file(data_dir, data_line):

    substr = data_line.strip("\n").split("\t")
    file_name = substr[0]
    label = substr[1]
    img_path = os.path.join(data_dir, file_name)
    data = {'img_path': img_path, 'label': label}
    if not os.path.exists(img_path):
        raise Exception("{} does not exist!".format(img_path))
    with open(data['img_path'], 'rb') as f:
        img = f.read()
        data['image'] = img
    return data


if __name__ == "__main__":

    args = utility.parse_args()
    kie_predict = KIE(args)
    preprocess_op = kie_predict.preprocess_op

    data_dir = "/paddle/KIE/train_data/wildreceipt/"
    anno_txt = "/paddle/KIE/train_data/wildreceipt/infer_debug.txt"
    data_lines = open(anno_txt, "r").readlines()
    data = decode_from_file(data_dir, data_lines[-1])

    inputs = transform(data, preprocess_op)

    outs = kie_predict._preprocess(inputs)

    kie_predict.forward(outs)




