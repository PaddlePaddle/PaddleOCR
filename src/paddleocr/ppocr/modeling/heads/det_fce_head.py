# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/dense_heads/fce_head.py
"""

from paddle import nn
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.nn.initializer import Normal
import paddle
from functools import partial


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class FCEHead(nn.Layer):
    """The class for implementing FCENet head.
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
    Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        in_channels (int): The number of input channels.
        scales (list[int]) : The scale of each layer.
        fourier_degree (int) : The maximum Fourier transform degree k.
    """

    def __init__(self, in_channels, fourier_degree=5):
        super().__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.fourier_degree = fourier_degree
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2

        self.out_conv_cls = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels_cls,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=ParamAttr(
                name="cls_weights", initializer=Normal(mean=0.0, std=0.01)
            ),
            bias_attr=True,
        )
        self.out_conv_reg = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=ParamAttr(
                name="reg_weights", initializer=Normal(mean=0.0, std=0.01)
            ),
            bias_attr=True,
        )

    def forward(self, feats, targets=None):
        cls_res, reg_res = multi_apply(self.forward_single, feats)
        level_num = len(cls_res)
        outs = {}
        if not self.training:
            for i in range(level_num):
                tr_pred = F.softmax(cls_res[i][:, 0:2, :, :], axis=1)
                tcl_pred = F.softmax(cls_res[i][:, 2:, :, :], axis=1)
                outs["level_{}".format(i)] = paddle.concat(
                    [tr_pred, tcl_pred, reg_res[i]], axis=1
                )
        else:
            preds = [[cls_res[i], reg_res[i]] for i in range(level_num)]
            outs["levels"] = preds
        return outs

    def forward_single(self, x):
        cls_predict = self.out_conv_cls(x)
        reg_predict = self.out_conv_reg(x)
        return cls_predict, reg_predict
