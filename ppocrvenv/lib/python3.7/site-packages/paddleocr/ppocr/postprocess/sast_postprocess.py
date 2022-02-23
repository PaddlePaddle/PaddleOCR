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

import os
import sys

__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '..'))

import numpy as np
from .locality_aware_nms import nms_locality
import paddle
import cv2
import time


class SASTPostProcess(object):
    """
    The post process for SAST.
    """

    def __init__(self,
                 score_thresh=0.5,
                 nms_thresh=0.2,
                 sample_pts_num=2,
                 shrink_ratio_of_width=0.3,
                 expand_scale=1.0,
                 tcl_map_thresh=0.5,
                 **kwargs):

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.sample_pts_num = sample_pts_num
        self.shrink_ratio_of_width = shrink_ratio_of_width
        self.expand_scale = expand_scale
        self.tcl_map_thresh = tcl_map_thresh

        # c++ la-nms is faster, but only support python 3.5
        self.is_python35 = False
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            self.is_python35 = True

    def point_pair2poly(self, point_pair_list):
        """
        Transfer vertical point_pairs into poly point in clockwise.
        """
        # constract poly
        point_num = len(point_pair_list) * 2
        point_list = [0] * point_num
        for idx, point_pair in enumerate(point_pair_list):
            point_list[idx] = point_pair[0]
            point_list[point_num - 1 - idx] = point_pair[1]
        return np.array(point_list).reshape(-1, 2)

    def shrink_quad_along_width(self,
                                quad,
                                begin_width_ratio=0.,
                                end_width_ratio=1.):
        """ 
        Generate shrink_quad_along_width.
        """
        ratio_pair = np.array(
            [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
        p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
        p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
        return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])

    def expand_poly_along_width(self, poly, shrink_ratio_of_width=0.3):
        """
        expand poly along width.
        """
        point_num = poly.shape[0]
        left_quad = np.array(
            [poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
        left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                     (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
        left_quad_expand = self.shrink_quad_along_width(left_quad, left_ratio,
                                                        1.0)
        right_quad = np.array(
            [
                poly[point_num // 2 - 2], poly[point_num // 2 - 1],
                poly[point_num // 2], poly[point_num // 2 + 1]
            ],
            dtype=np.float32)
        right_ratio = 1.0 + \
                      shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                      (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
        right_quad_expand = self.shrink_quad_along_width(right_quad, 0.0,
                                                         right_ratio)
        poly[0] = left_quad_expand[0]
        poly[-1] = left_quad_expand[-1]
        poly[point_num // 2 - 1] = right_quad_expand[1]
        poly[point_num // 2] = right_quad_expand[2]
        return poly

    def restore_quad(self, tcl_map, tcl_map_thresh, tvo_map):
        """Restore quad."""
        xy_text = np.argwhere(tcl_map[:, :, 0] > tcl_map_thresh)
        xy_text = xy_text[:, ::-1]  # (n, 2)

        # Sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 1])]

        scores = tcl_map[xy_text[:, 1], xy_text[:, 0], 0]
        scores = scores[:, np.newaxis]

        # Restore
        point_num = int(tvo_map.shape[-1] / 2)
        assert point_num == 4
        tvo_map = tvo_map[xy_text[:, 1], xy_text[:, 0], :]
        xy_text_tile = np.tile(xy_text, (1, point_num))  # (n, point_num * 2)
        quads = xy_text_tile - tvo_map

        return scores, quads, xy_text

    def quad_area(self, quad):
        """
        compute area of a quad.
        """
        edge = [(quad[1][0] - quad[0][0]) * (quad[1][1] + quad[0][1]),
                (quad[2][0] - quad[1][0]) * (quad[2][1] + quad[1][1]),
                (quad[3][0] - quad[2][0]) * (quad[3][1] + quad[2][1]),
                (quad[0][0] - quad[3][0]) * (quad[0][1] + quad[3][1])]
        return np.sum(edge) / 2.

    def nms(self, dets):
        if self.is_python35:
            import lanms
            dets = lanms.merge_quadrangle_n9(dets, self.nms_thresh)
        else:
            dets = nms_locality(dets, self.nms_thresh)
        return dets

    def cluster_by_quads_tco(self, tcl_map, tcl_map_thresh, quads, tco_map):
        """
        Cluster pixels in tcl_map based on quads.
        """
        instance_count = quads.shape[0] + 1  # contain background
        instance_label_map = np.zeros(tcl_map.shape[:2], dtype=np.int32)
        if instance_count == 1:
            return instance_count, instance_label_map

        # predict text center
        xy_text = np.argwhere(tcl_map[:, :, 0] > tcl_map_thresh)
        n = xy_text.shape[0]
        xy_text = xy_text[:, ::-1]  # (n, 2)
        tco = tco_map[xy_text[:, 1], xy_text[:, 0], :]  # (n, 2)
        pred_tc = xy_text - tco

        # get gt text center
        m = quads.shape[0]
        gt_tc = np.mean(quads, axis=1)  # (m, 2)

        pred_tc_tile = np.tile(pred_tc[:, np.newaxis, :],
                               (1, m, 1))  # (n, m, 2)
        gt_tc_tile = np.tile(gt_tc[np.newaxis, :, :], (n, 1, 1))  # (n, m, 2)
        dist_mat = np.linalg.norm(pred_tc_tile - gt_tc_tile, axis=2)  # (n, m)
        xy_text_assign = np.argmin(dist_mat, axis=1) + 1  # (n,)

        instance_label_map[xy_text[:, 1], xy_text[:, 0]] = xy_text_assign
        return instance_count, instance_label_map

    def estimate_sample_pts_num(self, quad, xy_text):
        """
        Estimate sample points number.
        """
        eh = (np.linalg.norm(quad[0] - quad[3]) +
              np.linalg.norm(quad[1] - quad[2])) / 2.0
        ew = (np.linalg.norm(quad[0] - quad[1]) +
              np.linalg.norm(quad[2] - quad[3])) / 2.0

        dense_sample_pts_num = max(2, int(ew))
        dense_xy_center_line = xy_text[np.linspace(
            0,
            xy_text.shape[0] - 1,
            dense_sample_pts_num,
            endpoint=True,
            dtype=np.float32).astype(np.int32)]

        dense_xy_center_line_diff = dense_xy_center_line[
            1:] - dense_xy_center_line[:-1]
        estimate_arc_len = np.sum(
            np.linalg.norm(
                dense_xy_center_line_diff, axis=1))

        sample_pts_num = max(2, int(estimate_arc_len / eh))
        return sample_pts_num

    def detect_sast(self,
                    tcl_map,
                    tvo_map,
                    tbo_map,
                    tco_map,
                    ratio_w,
                    ratio_h,
                    src_w,
                    src_h,
                    shrink_ratio_of_width=0.3,
                    tcl_map_thresh=0.5,
                    offset_expand=1.0,
                    out_strid=4.0):
        """
        first resize the tcl_map, tvo_map and tbo_map to the input_size, then restore the polys
        """
        # restore quad
        scores, quads, xy_text = self.restore_quad(tcl_map, tcl_map_thresh,
                                                   tvo_map)
        dets = np.hstack((quads, scores)).astype(np.float32, copy=False)
        dets = self.nms(dets)
        if dets.shape[0] == 0:
            return []
        quads = dets[:, :-1].reshape(-1, 4, 2)

        # Compute quad area
        quad_areas = []
        for quad in quads:
            quad_areas.append(-self.quad_area(quad))

        # instance segmentation
        # instance_count, instance_label_map = cv2.connectedComponents(tcl_map.astype(np.uint8), connectivity=8)
        instance_count, instance_label_map = self.cluster_by_quads_tco(
            tcl_map, tcl_map_thresh, quads, tco_map)

        # restore single poly with tcl instance.
        poly_list = []
        for instance_idx in range(1, instance_count):
            xy_text = np.argwhere(instance_label_map == instance_idx)[:, ::-1]
            quad = quads[instance_idx - 1]
            q_area = quad_areas[instance_idx - 1]
            if q_area < 5:
                continue

            #
            len1 = float(np.linalg.norm(quad[0] - quad[1]))
            len2 = float(np.linalg.norm(quad[1] - quad[2]))
            min_len = min(len1, len2)
            if min_len < 3:
                continue

            # filter small CC
            if xy_text.shape[0] <= 0:
                continue

            # filter low confidence instance
            xy_text_scores = tcl_map[xy_text[:, 1], xy_text[:, 0], 0]
            if np.sum(xy_text_scores) / quad_areas[instance_idx - 1] < 0.1:
                # if np.sum(xy_text_scores) / quad_areas[instance_idx - 1] < 0.05:
                continue

            # sort xy_text
            left_center_pt = np.array(
                [[(quad[0, 0] + quad[-1, 0]) / 2.0,
                  (quad[0, 1] + quad[-1, 1]) / 2.0]])  # (1, 2)
            right_center_pt = np.array(
                [[(quad[1, 0] + quad[2, 0]) / 2.0,
                  (quad[1, 1] + quad[2, 1]) / 2.0]])  # (1, 2)
            proj_unit_vec = (right_center_pt - left_center_pt) / \
                            (np.linalg.norm(right_center_pt - left_center_pt) + 1e-6)
            proj_value = np.sum(xy_text * proj_unit_vec, axis=1)
            xy_text = xy_text[np.argsort(proj_value)]

            # Sample pts in tcl map
            if self.sample_pts_num == 0:
                sample_pts_num = self.estimate_sample_pts_num(quad, xy_text)
            else:
                sample_pts_num = self.sample_pts_num
            xy_center_line = xy_text[np.linspace(
                0,
                xy_text.shape[0] - 1,
                sample_pts_num,
                endpoint=True,
                dtype=np.float32).astype(np.int32)]

            point_pair_list = []
            for x, y in xy_center_line:
                # get corresponding offset
                offset = tbo_map[y, x, :].reshape(2, 2)
                if offset_expand != 1.0:
                    offset_length = np.linalg.norm(
                        offset, axis=1, keepdims=True)
                    expand_length = np.clip(
                        offset_length * (offset_expand - 1),
                        a_min=0.5,
                        a_max=3.0)
                    offset_detal = offset / offset_length * expand_length
                    offset = offset + offset_detal
                    # original point
                ori_yx = np.array([y, x], dtype=np.float32)
                point_pair = (ori_yx + offset)[:, ::-1] * out_strid / np.array(
                    [ratio_w, ratio_h]).reshape(-1, 2)
                point_pair_list.append(point_pair)

            # ndarry: (x, 2), expand poly along width
            detected_poly = self.point_pair2poly(point_pair_list)
            detected_poly = self.expand_poly_along_width(detected_poly,
                                                         shrink_ratio_of_width)
            detected_poly[:, 0] = np.clip(
                detected_poly[:, 0], a_min=0, a_max=src_w)
            detected_poly[:, 1] = np.clip(
                detected_poly[:, 1], a_min=0, a_max=src_h)
            poly_list.append(detected_poly)

        return poly_list

    def __call__(self, outs_dict, shape_list):
        score_list = outs_dict['f_score']
        border_list = outs_dict['f_border']
        tvo_list = outs_dict['f_tvo']
        tco_list = outs_dict['f_tco']
        if isinstance(score_list, paddle.Tensor):
            score_list = score_list.numpy()
            border_list = border_list.numpy()
            tvo_list = tvo_list.numpy()
            tco_list = tco_list.numpy()

        img_num = len(shape_list)
        poly_lists = []
        for ino in range(img_num):
            p_score = score_list[ino].transpose((1, 2, 0))
            p_border = border_list[ino].transpose((1, 2, 0))
            p_tvo = tvo_list[ino].transpose((1, 2, 0))
            p_tco = tco_list[ino].transpose((1, 2, 0))
            src_h, src_w, ratio_h, ratio_w = shape_list[ino]

            poly_list = self.detect_sast(
                p_score,
                p_tvo,
                p_border,
                p_tco,
                ratio_w,
                ratio_h,
                src_w,
                src_h,
                shrink_ratio_of_width=self.shrink_ratio_of_width,
                tcl_map_thresh=self.tcl_map_thresh,
                offset_expand=self.expand_scale)
            poly_lists.append({'points': np.array(poly_list)})

        return poly_lists
