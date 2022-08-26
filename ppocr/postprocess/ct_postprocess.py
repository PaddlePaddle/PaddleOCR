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
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/post_processing/seg_detector_representer.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.nn.functional as F
import os
import os.path as osp
import numpy as np
import cv2
import paddle
from shapely.geometry import Polygon
import pyclipper


class CTPostProcess(object):
    """
    The post process for Centripetal Text (CT).
    """

    def __init__(self, min_score=0.88, min_area=16, box_type='poly', **kwargs):
        self.min_score = min_score
        self.min_area = min_area
        self.box_type = box_type

        self.coord = np.zeros((2, 300, 300), dtype=np.int32)
        for i in range(300):
            for j in range(300):
                self.coord[0, i, j] = j
                self.coord[1, i, j] = i

    # def _upsample(self, x, size, scale=1):
    #     H, W, _ = size
    #     print('====1===',x.shape, size)
    #     return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    # def _upsample(self, x, scale=1):
    #     return F.upsample(x, scale_factor=scale, mode='bilinear')

    def __call__(self, preds, batch):
        # print(batch.shape)
        # print(outs.shape)
        outs = preds['maps']
        out_scores = preds['score']

        if isinstance(outs, paddle.Tensor):
            outs = outs.numpy()
        if isinstance(out_scores, paddle.Tensor):
            out_scores = out_scores.numpy()

        batch_size = outs.shape[0]
        boxes_batch = []
        for idx in range(batch_size):
            bboxes = []
            scores = []

            img_shape = batch[idx]

            org_img_size = img_shape[:3]
            img_shape = img_shape[3:]
            img_size = img_shape[:2]

            out = np.expand_dims(outs[idx], axis=0)
            # print('===', out.shape)
            # # out = self._upsample(out, img_shape, 4)
            # print('===', out.shape)
            outputs = dict()

            # print(out.shape)
            score = np.expand_dims(
                out_scores[idx], axis=0)  #F.sigmoid(out[:, 0, :, :])

            kernel = out[:, 0, :, :] > 0.2
            # loc = paddle.cast(out[:, 1:, :, :], "float32")
            loc = out[:, 1:, :, :].astype("float32")

            score = score[0].astype(np.float32)
            kernel = kernel[0].astype(np.uint8)
            loc = loc[0].astype(np.float32)

            #print(score.shape, kernel.shape, loc.shape)
            label_num, label_kernel = cv2.connectedComponents(
                kernel, connectivity=4)

            #print(label_num, label_kernel.shape)

            for i in range(1, label_num):
                ind = (label_kernel == i)
                if ind.sum(
                ) < 10:  # pixel number less than 10, treated as background
                    label_kernel[ind] = 0

            label = np.zeros_like(label_kernel)
            h, w = label_kernel.shape
            pixels = self.coord[:, :h, :w].reshape(2, -1)
            points = pixels.transpose([1, 0]).astype(np.float32)

            off_points = (points + 10. / 4. * loc[:, pixels[1], pixels[0]].T
                          ).astype(np.int32)
            off_points[:, 0] = np.clip(off_points[:, 0], 0, label.shape[1] - 1)
            off_points[:, 1] = np.clip(off_points[:, 1], 0, label.shape[0] - 1)

            label[pixels[1], pixels[0]] = label_kernel[off_points[:, 1],
                                                       off_points[:, 0]]
            label[label_kernel > 0] = label_kernel[label_kernel > 0]

            score_pocket = [0.0]
            for i in range(1, label_num):
                ind = (label_kernel == i)
                if ind.sum() == 0:
                    score_pocket.append(0.0)
                    continue
                score_i = np.mean(score[ind])
                score_pocket.append(score_i)

            label_num = np.max(label) + 1
            label = cv2.resize(
                label, (img_size[1], img_size[0]),
                interpolation=cv2.INTER_NEAREST)

            scale = (float(org_img_size[1]) / float(img_size[1]),
                     float(org_img_size[0]) / float(img_size[0]))

            for i in range(1, label_num):
                ind = (label == i)
                points = np.array(np.where(ind)).transpose((1, 0))

                if points.shape[0] < self.min_area:
                    continue

                score_i = score_pocket[i]
                if score_i < self.min_score:
                    continue

                if self.box_type == 'rect':
                    rect = cv2.minAreaRect(points[:, ::-1])
                    bbox = cv2.boxPoints(rect) * scale
                    z = bbox.mean(0)
                    bbox = z + (bbox - z) * 0.85
                elif self.box_type == 'poly':
                    binary = np.zeros(label.shape, dtype='uint8')
                    binary[ind] = 1
                    try:
                        _, contours, _ = cv2.findContours(
                            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    except BaseException:
                        contours, _ = cv2.findContours(
                            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    bbox = contours[0] * scale

                bbox = bbox.astype('int32')
                bboxes.append(bbox.reshape(-1, 2))  #.reshape(-1))
                scores.append(score_i)

            boxes_batch.append({'points': bboxes})

            #print(len(bboxes), bboxes[0].shape, len(scores), scores[0])
            #image_name, _ = osp.splitext(osp.basename(img_path))

            #outputs.update(dict(bboxes=bboxes)) #, input_id=image_name))

        return boxes_batch
