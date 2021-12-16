# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/whai362/PSENet/blob/python3/models/head/psenet_head.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import paddle
from paddle.nn import functional as F

from ppocr.postprocess.pse_postprocess.pse import pse


class PSEPostProcess(object):
    """
    The post process for PSE.
    """

    def __init__(self,
                 thresh=0.5,
                 box_thresh=0.85,
                 min_area=16,
                 box_type='box',
                 scale=4,
                 **kwargs):
        assert box_type in ['box', 'poly'], 'Only box and poly is supported'
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.min_area = min_area
        self.box_type = box_type
        self.scale = scale

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if not isinstance(pred, paddle.Tensor):
            pred = paddle.to_tensor(pred)
        pred = F.interpolate(
            pred, scale_factor=4 // self.scale, mode='bilinear')

        score = F.sigmoid(pred[:, 0, :, :])

        kernels = (pred > self.thresh).astype('float32')
        text_mask = kernels[:, 0, :, :]
        kernels[:, 0:, :, :] = kernels[:, 0:, :, :] * text_mask

        score = score.numpy()
        kernels = kernels.numpy().astype(np.uint8)

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            boxes, scores = self.boxes_from_bitmap(score[batch_index],
                                                   kernels[batch_index],
                                                   shape_list[batch_index])

            boxes_batch.append({'points': boxes, 'scores': scores})
        return boxes_batch

    def boxes_from_bitmap(self, score, kernels, shape):
        label = pse(kernels, self.min_area)
        return self.generate_box(score, label, shape)

    def generate_box(self, score, label, shape):
        src_h, src_w, ratio_h, ratio_w = shape
        label_num = np.max(label) + 1

        boxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < self.box_thresh:
                label[ind] = 0
                continue

            if self.box_type == 'box':
                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect)
            elif self.box_type == 'poly':
                box_height = np.max(points[:, 1]) + 10
                box_width = np.max(points[:, 0]) + 10

                mask = np.zeros((box_height, box_width), np.uint8)
                mask[points[:, 1], points[:, 0]] = 255

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                bbox = np.squeeze(contours[0], 1)
            else:
                raise NotImplementedError

            bbox[:, 0] = np.clip(np.round(bbox[:, 0] / ratio_w), 0, src_w)
            bbox[:, 1] = np.clip(np.round(bbox[:, 1] / ratio_h), 0, src_h)
            boxes.append(bbox)
            scores.append(score_i)
        return boxes, scores
