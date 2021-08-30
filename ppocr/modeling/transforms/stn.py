# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np


def conv3x3_block(in_channels, out_channels, stride=1):
    n = 3 * 3 * out_channels
    w = math.sqrt(2. / n)
    conv_layer = nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        weight_attr=nn.initializer.Normal(
            mean=0.0, std=w),
        bias_attr=nn.initializer.Constant(0))
    block = nn.Sequential(conv_layer, nn.BatchNorm2D(out_channels), nn.ReLU())
    return block


class STN(nn.Layer):
    def __init__(self, in_channels, num_ctrlpoints, activation='none'):
        super(STN, self).__init__()
        self.in_channels = in_channels
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.Sequential(
            conv3x3_block(in_channels, 32),  #32x64
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(32, 64),  #16x32
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(128, 256),  # 4*8
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(256, 256),  # 2*4,
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(256, 256))  # 1*2
        self.stn_fc1 = nn.Sequential(
            nn.Linear(
                2 * 256,
                512,
                weight_attr=nn.initializer.Normal(0, 0.001),
                bias_attr=nn.initializer.Constant(0)),
            nn.BatchNorm1D(512),
            nn.ReLU())
        fc2_bias = self.init_stn()
        self.stn_fc2 = nn.Linear(
            512,
            num_ctrlpoints * 2,
            weight_attr=nn.initializer.Constant(0.0),
            bias_attr=nn.initializer.Assign(fc2_bias))

    def init_stn(self):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate(
            [ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        if self.activation == 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        ctrl_points = paddle.to_tensor(ctrl_points)
        fc2_bias = paddle.reshape(
            ctrl_points, shape=[ctrl_points.shape[0] * ctrl_points.shape[1]])
        return fc2_bias

    def forward(self, x):
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.shape
        x = paddle.reshape(x, shape=(batch_size, -1))
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        x = paddle.reshape(x, shape=[-1, self.num_ctrlpoints, 2])
        return img_feat, x
