# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn as nn

from weight_init import weight_init_


class ResidualBlock(nn.Layer):
    """Residual Block with custom normalization."""

    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2D(in_planes, planes, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1)
        self.relu = nn.ReLU()

        if norm_fn == "group":
            num_groups = planes // 8
            self.norm1 = nn.GroupNorm(
                num_groups, planes, weight_attr=False, bias_attr=False)
            self.norm2 = nn.GroupNorm(
                num_groups, planes, weight_attr=False, bias_attr=False)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(
                    num_groups, planes, weight_attr=False, bias_attr=False)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2D(
                planes, weight_attr=False, bias_attr=False)
            self.norm2 = nn.BatchNorm2D(
                planes, weight_attr=False, bias_attr=False)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2D(
                    planes, weight_attr=False, bias_attr=False)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2D(
                planes, weight_attr=False, bias_attr=False)
            self.norm2 = nn.InstanceNorm2D(
                planes, weight_attr=False, bias_attr=False)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2D(
                    planes, weight_attr=False, bias_attr=False)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2D(
                    in_planes, planes, 1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Layer):
    """Basic Encoder with custom normalization."""

    def __init__(self, output_dim=128, norm_fn="batch"):
        super(BasicEncoder, self).__init__()

        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(8, 64, weight_attr=False, bias_attr=False)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2D(64, weight_attr=False, bias_attr=False)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2D(
                64, weight_attr=False, bias_attr=False)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2D(3, 64, 7, stride=2, padding=3)
        self.relu1 = nn.ReLU()

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(192, stride=2)

        self.conv2 = nn.Conv2D(192, output_dim, 1)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                weight_init_(
                    m.weight,
                    "kaiming_normal_",
                    mode="fan_out",
                    nonlinearity="relu")
            elif isinstance(m,
                            (nn.BatchNorm2D, nn.InstanceNorm2D, nn.GroupNorm)):
                weight_init_(m, "Constant", value=1, bias_value=0.0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = layer1, layer2

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        return x
