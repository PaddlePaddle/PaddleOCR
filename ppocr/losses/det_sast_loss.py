# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
from .det_basic_loss import DiceLoss
import numpy as np


class SASTLoss(nn.Layer):
    """
    """

    def __init__(self, eps=1e-6, **kwargs):
        super(SASTLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        """
        tcl_pos: N x 128 x 3
        tcl_mask: N x 128 x 1
        tcl_label: N x X list or LoDTensor
        """

        f_score = predicts['f_score']
        f_border = predicts['f_border']
        f_tvo = predicts['f_tvo']
        f_tco = predicts['f_tco']

        l_score, l_border, l_mask, l_tvo, l_tco = labels[1:]

        #score_loss
        intersection = paddle.sum(f_score * l_score * l_mask)
        union = paddle.sum(f_score * l_mask) + paddle.sum(l_score * l_mask)
        score_loss = 1.0 - 2 * intersection / (union + 1e-5)

        #border loss
        l_border_split, l_border_norm = paddle.split(
            l_border, num_or_sections=[4, 1], axis=1)
        f_border_split = f_border
        border_ex_shape = l_border_norm.shape * np.array([1, 4, 1, 1])
        l_border_norm_split = paddle.expand(
            x=l_border_norm, shape=border_ex_shape)
        l_border_score = paddle.expand(x=l_score, shape=border_ex_shape)
        l_border_mask = paddle.expand(x=l_mask, shape=border_ex_shape)

        border_diff = l_border_split - f_border_split
        abs_border_diff = paddle.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = paddle.cast(border_sign, dtype='float32')
        border_sign.stop_gradient = True
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + \
                    (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = paddle.sum(border_out_loss * l_border_score * l_border_mask) / \
                    (paddle.sum(l_border_score * l_border_mask) + 1e-5)

        #tvo_loss
        l_tvo_split, l_tvo_norm = paddle.split(
            l_tvo, num_or_sections=[8, 1], axis=1)
        f_tvo_split = f_tvo
        tvo_ex_shape = l_tvo_norm.shape * np.array([1, 8, 1, 1])
        l_tvo_norm_split = paddle.expand(x=l_tvo_norm, shape=tvo_ex_shape)
        l_tvo_score = paddle.expand(x=l_score, shape=tvo_ex_shape)
        l_tvo_mask = paddle.expand(x=l_mask, shape=tvo_ex_shape)
        #
        tvo_geo_diff = l_tvo_split - f_tvo_split
        abs_tvo_geo_diff = paddle.abs(tvo_geo_diff)
        tvo_sign = abs_tvo_geo_diff < 1.0
        tvo_sign = paddle.cast(tvo_sign, dtype='float32')
        tvo_sign.stop_gradient = True
        tvo_in_loss = 0.5 * abs_tvo_geo_diff * abs_tvo_geo_diff * tvo_sign + \
                    (abs_tvo_geo_diff - 0.5) * (1.0 - tvo_sign)
        tvo_out_loss = l_tvo_norm_split * tvo_in_loss
        tvo_loss = paddle.sum(tvo_out_loss * l_tvo_score * l_tvo_mask) / \
                    (paddle.sum(l_tvo_score * l_tvo_mask) + 1e-5)

        #tco_loss
        l_tco_split, l_tco_norm = paddle.split(
            l_tco, num_or_sections=[2, 1], axis=1)
        f_tco_split = f_tco
        tco_ex_shape = l_tco_norm.shape * np.array([1, 2, 1, 1])
        l_tco_norm_split = paddle.expand(x=l_tco_norm, shape=tco_ex_shape)
        l_tco_score = paddle.expand(x=l_score, shape=tco_ex_shape)
        l_tco_mask = paddle.expand(x=l_mask, shape=tco_ex_shape)

        tco_geo_diff = l_tco_split - f_tco_split
        abs_tco_geo_diff = paddle.abs(tco_geo_diff)
        tco_sign = abs_tco_geo_diff < 1.0
        tco_sign = paddle.cast(tco_sign, dtype='float32')
        tco_sign.stop_gradient = True
        tco_in_loss = 0.5 * abs_tco_geo_diff * abs_tco_geo_diff * tco_sign + \
                    (abs_tco_geo_diff - 0.5) * (1.0 - tco_sign)
        tco_out_loss = l_tco_norm_split * tco_in_loss
        tco_loss = paddle.sum(tco_out_loss * l_tco_score * l_tco_mask) / \
                    (paddle.sum(l_tco_score * l_tco_mask) + 1e-5)

        # total loss
        tvo_lw, tco_lw = 1.5, 1.5
        score_lw, border_lw = 1.0, 1.0
        total_loss = score_loss * score_lw + border_loss * border_lw + \
                    tvo_loss * tvo_lw + tco_loss * tco_lw

        losses = {'loss':total_loss, "score_loss":score_loss,\
            "border_loss":border_loss, 'tvo_loss':tvo_loss, 'tco_loss':tco_loss}
        return losses
