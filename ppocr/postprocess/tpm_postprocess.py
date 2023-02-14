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
"""
This code is refered from:
https://github.com/GXYM/TextPMs/blob/HEAD/util/detection.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import cv2
import paddle
import numpy as np
from skimage import segmentation


class TPMPostProcess(object):
    def __init__(self, cfg, box_type='poly', **kwargs):
        self.cfg = cfg
        # TODO: support quad output 
        if box_type != 'poly':
            raise ValueError('Only support polygon for now.')

    def fill_hole(self, input_mask):
        h, w = input_mask.shape
        canvas = np.zeros((h + 2, w + 2), np.uint8)
        canvas[1:h + 1, 1:w + 1] = input_mask.copy()

        mask = np.zeros((h + 4, w + 4), np.uint8)

        cv2.floodFill(canvas, mask, (0, 0), 1)
        canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

        return (~canvas | input_mask.astype(np.uint8))

    def sigmoid_alpha(self, x, k):
        betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
        dm = max(np.max(x), 0.0001)
        res = (2 / (1 + np.exp(-x * k / dm)) - 1) * betak
        return np.maximum(0, res)

    def watershed_segment(self, cfg, preds, scale=1.0):
        text_region = np.mean(preds[2:], axis=0)
        region = self.fill_hole(text_region >= cfg.threshold)

        text_kernal = np.mean(preds[0:2], axis=0)
        kernal = self.fill_hole(text_kernal >= cfg.threshold)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(kernal, cv2.MORPH_OPEN, kernel, iterations=1)
        kernal = cv2.erode(
            opening, kernel, iterations=1)  # sure foreground area
        ret, m = cv2.connectedComponents(kernal)

        distance = np.mean(preds[:], axis=0)
        distance = np.array(distance / np.max(distance) * 255, dtype=np.uint8)
        labels = segmentation.watershed(-distance, m, mask=region)
        boxes = []
        contours = []
        small_area = (300 if cfg.test_size[0] >= 256 else 150)
        for idx in range(1, np.max(labels) + 1):
            text_mask = labels == idx
            if np.sum(text_mask) < small_area / (cfg.scale * cfg.scale) \
                    or np.mean(preds[-1][text_mask]) < cfg.score_i:  # 150 / 300
                continue
            cont, _ = cv2.findContours(
                text_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            epsilon = 0.003 * cv2.arcLength(cont[0], True)
            approx = cv2.approxPolyDP(cont[0], epsilon, True)
            contours.append(approx.reshape((-1, 2)) * [scale, scale])

        return labels, boxes, contours

    def __call__(self, preds, sigmoid=False):

        boxes_batch = []
        scale = self.cfg.scale
        for batch_index in range(pred.shape[0]):
            if sigmoid is False:
                seg_maps = self.sigmoid_alpha(preds[batch_index, :, :h //
                                                    scale, :w // scale])
            else:
                seg_maps = preds[batch_index, :, :h // scale, :w // scale]

            labels, boxes, contours = watershed_segment(
                self.cfg, preds, scale=scale)
            boxes_batch.append({'points': poly})
