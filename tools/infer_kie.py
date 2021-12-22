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
import paddle.nn.functional as F

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.utils.save_load import load_model
import tools.program as program
import time


def read_class_list(filepath):
    dict = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.split(" ")
            dict[key] = value.rstrip()
    return dict


def draw_kie_result(batch, node, idx_to_cls, count):
    img = batch[6].copy()
    boxes = batch[7]
    h, w = img.shape[:2]
    pred_img = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
    max_value, max_idx = paddle.max(node, -1), paddle.argmax(node, -1)
    node_pred_label = max_idx.numpy().tolist()
    node_pred_score = max_value.numpy().tolist()

    for i, box in enumerate(boxes):
        if i >= len(node_pred_label):
            break
        new_box = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]],
                   [box[0], box[3]]]
        Pts = np.array([new_box], np.int32)
        cv2.polylines(
            img, [Pts.reshape((-1, 1, 2))],
            True,
            color=(255, 255, 0),
            thickness=1)
        x_min = int(min([point[0] for point in new_box]))
        y_min = int(min([point[1] for point in new_box]))

        pred_label = str(node_pred_label[i])
        if pred_label in idx_to_cls:
            pred_label = idx_to_cls[pred_label]
        pred_score = '{:.2f}'.format(node_pred_score[i])
        text = pred_label + '(' + pred_score + ')'
        cv2.putText(pred_img, text, (x_min * 2, y_min),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    vis_img = np.ones((h, w * 3, 3), dtype=np.uint8) * 255
    vis_img[:, :w] = img
    vis_img[:, w:] = pred_img
    save_kie_path = os.path.dirname(config['Global'][
        'save_res_path']) + "/kie_results/"
    if not os.path.exists(save_kie_path):
        os.makedirs(save_kie_path)
    save_path = os.path.join(save_kie_path, str(count) + ".png")
    cv2.imwrite(save_path, vis_img)
    logger.info("The Kie Image saved in {}".format(save_path))


def main():
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])
    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        transforms.append(op)

    data_dir = config['Eval']['dataset']['data_dir']

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    class_path = config['Global']['class_path']
    idx_to_cls = read_class_list(class_path)
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()

    warmup_times = 0
    count_t = []
    with open(save_res_path, "wb") as fout:
        with open(config['Global']['infer_img'], "rb") as f:
            lines = f.readlines()
            for index, data_line in enumerate(lines):
                if index == 10:
                    warmup_t = time.time()
                data_line = data_line.decode('utf-8')
                substr = data_line.strip("\n").split("\t")
                img_path, label = data_dir + "/" + substr[0], substr[1]
                data = {'img_path': img_path, 'label': label}
                with open(data['img_path'], 'rb') as f:
                    img = f.read()
                    data['image'] = img
                st = time.time()
                batch = transform(data, ops)
                batch_pred = [0] * len(batch)
                for i in range(len(batch)):
                    batch_pred[i] = paddle.to_tensor(
                        np.expand_dims(
                            batch[i], axis=0))
                st = time.time()
                node, edge = model(batch_pred)
                node = F.softmax(node, -1)
                count_t.append(time.time() - st)
                draw_kie_result(batch, node, idx_to_cls, index)
    logger.info("success!")
    logger.info("It took {} s for predict {} images.".format(
        np.sum(count_t), len(count_t)))
    ips = len(count_t[warmup_times:]) / np.sum(count_t[warmup_times:])
    logger.info("The ips is {} images/s".format(ips))


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
