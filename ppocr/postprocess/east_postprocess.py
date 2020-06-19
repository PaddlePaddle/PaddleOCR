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
from .locality_aware_nms import nms_locality
import cv2

import os
import sys
__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '..'))
import lanms


class EASTPostPocess(object):
    """
    The post process for EAST.
    """

    def __init__(self, params):
        self.score_thresh = params['score_thresh']
        self.cover_thresh = params['cover_thresh']
        self.nms_thresh = params['nms_thresh']

    def restore_rectangle_quad(self, origin, geometry):
        """
        Restore rectangle from quadrangle.
        """
        # quad
        origin_concat = np.concatenate(
            (origin, origin, origin, origin), axis=1)  # (n, 8)
        pred_quads = origin_concat - geometry
        pred_quads = pred_quads.reshape((-1, 4, 2))  # (n, 4, 2)
        return pred_quads

    def detect(self,
               score_map,
               geo_map,
               score_thresh=0.8,
               cover_thresh=0.1,
               nms_thresh=0.2):
        """
        restore text boxes from score map and geo map
        """
        score_map = score_map[0]
        geo_map = np.swapaxes(geo_map, 1, 0)
        geo_map = np.swapaxes(geo_map, 1, 2)
        # filter the score map
        xy_text = np.argwhere(score_map > score_thresh)
        if len(xy_text) == 0:
            return []
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        #restore quad proposals
        text_box_restored = self.restore_rectangle_quad(
            xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        # boxes = nms_locality(boxes.astype(np.float64), nms_thresh)
        boxes = lanms.merge_quadrangle_n9(boxes, nms_thresh)
        if boxes.shape[0] == 0:
            return []
        # Here we filter some low score boxes by the average score map, 
        #   this is different from the orginal paper.
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape(
                (-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > cover_thresh]
        return boxes

    def sort_poly(self, p):
        """
        Sort polygons.
        """
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4,\
            (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def __call__(self, outs_dict, ratio_list):
        score_list = outs_dict['f_score']
        geo_list = outs_dict['f_geo']
        img_num = len(ratio_list)
        dt_boxes_list = []
        for ino in range(img_num):
            score = score_list[ino]
            geo = geo_list[ino]
            boxes = self.detect(
                score_map=score,
                geo_map=geo,
                score_thresh=self.score_thresh,
                cover_thresh=self.cover_thresh,
                nms_thresh=self.nms_thresh)
            boxes_norm = []
            if len(boxes) > 0:
                ratio_h, ratio_w = ratio_list[ino]
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                for i_box, box in enumerate(boxes):
                    box = self.sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 \
                        or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    boxes_norm.append(box)
            dt_boxes_list.append(np.array(boxes_norm))
        return dt_boxes_list
