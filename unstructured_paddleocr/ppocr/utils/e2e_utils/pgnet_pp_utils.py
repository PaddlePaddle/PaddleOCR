# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import os
import sys

__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '..'))
from extract_textpoint_slow import *
from extract_textpoint_fast import generate_pivot_list_fast, restore_poly


class PGNet_PostProcess(object):
    # two different post-process
    def __init__(self,
                 character_dict_path,
                 valid_set,
                 score_thresh,
                 outs_dict,
                 shape_list,
                 point_gather_mode=None):
        self.Lexicon_Table = get_dict(character_dict_path)
        self.valid_set = valid_set
        self.score_thresh = score_thresh
        self.outs_dict = outs_dict
        self.shape_list = shape_list
        self.point_gather_mode = point_gather_mode

    def pg_postprocess_fast(self):
        p_score = self.outs_dict['f_score']
        p_border = self.outs_dict['f_border']
        p_char = self.outs_dict['f_char']
        p_direction = self.outs_dict['f_direction']
        if isinstance(p_score, paddle.Tensor):
            p_score = p_score[0].numpy()
            p_border = p_border[0].numpy()
            p_direction = p_direction[0].numpy()
            p_char = p_char[0].numpy()
        else:
            p_score = p_score[0]
            p_border = p_border[0]
            p_direction = p_direction[0]
            p_char = p_char[0]

        src_h, src_w, ratio_h, ratio_w = self.shape_list[0]
        instance_yxs_list, seq_strs = generate_pivot_list_fast(
            p_score,
            p_char,
            p_direction,
            self.Lexicon_Table,
            score_thresh=self.score_thresh,
            point_gather_mode=self.point_gather_mode)
        poly_list, keep_str_list = restore_poly(instance_yxs_list, seq_strs,
                                                p_border, ratio_w, ratio_h,
                                                src_w, src_h, self.valid_set)
        data = {
            'points': poly_list,
            'texts': keep_str_list,
        }
        return data

    def pg_postprocess_slow(self):
        p_score = self.outs_dict['f_score']
        p_border = self.outs_dict['f_border']
        p_char = self.outs_dict['f_char']
        p_direction = self.outs_dict['f_direction']
        if isinstance(p_score, paddle.Tensor):
            p_score = p_score[0].numpy()
            p_border = p_border[0].numpy()
            p_direction = p_direction[0].numpy()
            p_char = p_char[0].numpy()
        else:
            p_score = p_score[0]
            p_border = p_border[0]
            p_direction = p_direction[0]
            p_char = p_char[0]
        src_h, src_w, ratio_h, ratio_w = self.shape_list[0]
        is_curved = self.valid_set == "totaltext"
        char_seq_idx_set, instance_yxs_list = generate_pivot_list_slow(
            p_score,
            p_char,
            p_direction,
            score_thresh=self.score_thresh,
            is_backbone=True,
            is_curved=is_curved)
        seq_strs = []
        for char_idx_set in char_seq_idx_set:
            pr_str = ''.join([self.Lexicon_Table[pos] for pos in char_idx_set])
            seq_strs.append(pr_str)
        poly_list = []
        keep_str_list = []
        all_point_list = []
        all_point_pair_list = []
        for yx_center_line, keep_str in zip(instance_yxs_list, seq_strs):
            if len(yx_center_line) == 1:
                yx_center_line.append(yx_center_line[-1])

            offset_expand = 1.0
            if self.valid_set == 'totaltext':
                offset_expand = 1.2

            point_pair_list = []
            for batch_id, y, x in yx_center_line:
                offset = p_border[:, y, x].reshape(2, 2)
                if offset_expand != 1.0:
                    offset_length = np.linalg.norm(
                        offset, axis=1, keepdims=True)
                    expand_length = np.clip(
                        offset_length * (offset_expand - 1),
                        a_min=0.5,
                        a_max=3.0)
                    offset_detal = offset / offset_length * expand_length
                    offset = offset + offset_detal
                ori_yx = np.array([y, x], dtype=np.float32)
                point_pair = (ori_yx + offset)[:, ::-1] * 4.0 / np.array(
                    [ratio_w, ratio_h]).reshape(-1, 2)
                point_pair_list.append(point_pair)

                all_point_list.append([
                    int(round(x * 4.0 / ratio_w)),
                    int(round(y * 4.0 / ratio_h))
                ])
                all_point_pair_list.append(point_pair.round().astype(np.int32)
                                           .tolist())

            detected_poly, pair_length_info = point_pair2poly(point_pair_list)
            detected_poly = expand_poly_along_width(
                detected_poly, shrink_ratio_of_width=0.2)
            detected_poly[:, 0] = np.clip(
                detected_poly[:, 0], a_min=0, a_max=src_w)
            detected_poly[:, 1] = np.clip(
                detected_poly[:, 1], a_min=0, a_max=src_h)

            if len(keep_str) < 2:
                continue

            keep_str_list.append(keep_str)
            detected_poly = np.round(detected_poly).astype('int32')
            if self.valid_set == 'partvgg':
                middle_point = len(detected_poly) // 2
                detected_poly = detected_poly[
                    [0, middle_point - 1, middle_point, -1], :]
                poly_list.append(detected_poly)
            elif self.valid_set == 'totaltext':
                poly_list.append(detected_poly)
            else:
                print('--> Not supported format.')
                exit(-1)
        data = {
            'points': poly_list,
            'texts': keep_str_list,
        }
        return data
