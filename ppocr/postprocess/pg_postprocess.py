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

import os
import sys

__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '..'))

from ppocr.utils.e2e_utils.extract_textpoint import get_dict, generate_pivot_list, restore_poly
import paddle


class PGPostProcess(object):
    """
    The post process for PGNet.
    """

    def __init__(self, character_dict_path, valid_set, score_thresh, **kwargs):
        self.Lexicon_Table = get_dict(character_dict_path)
        self.valid_set = valid_set
        self.score_thresh = score_thresh

    def __call__(self, outs_dict, shape_list):
        p_score = outs_dict['f_score']
        p_border = outs_dict['f_border']
        p_char = outs_dict['f_char']
        p_direction = outs_dict['f_direction']
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

        src_h, src_w, ratio_h, ratio_w = shape_list[0]
        instance_yxs_list, seq_strs = generate_pivot_list(
            p_score,
            p_char,
            p_direction,
            self.Lexicon_Table,
            score_thresh=self.score_thresh)
        poly_list, keep_str_list = restore_poly(instance_yxs_list, seq_strs,
                                                p_border, ratio_w, ratio_h,
                                                src_w, src_h, self.valid_set)
        data = {
            'points': poly_list,
            'strs': keep_str_list,
        }
        return data
