# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import paddle
import math
import paddle.nn.functional as F

class Conv_BN_ReLU(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias_attr=False)
        self.bn = nn.BatchNorm2D(out_planes, momentum=0.1)
        self.relu = nn.ReLU()

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Normal(0, math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FPN(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()

        # Top layer
        self.toplayer_ = Conv_BN_ReLU(in_channels[3], out_channels, kernel_size=1, stride=1, padding=0)
        # Lateral layers
        self.latlayer1_ = Conv_BN_ReLU(in_channels[2], out_channels, kernel_size=1, stride=1, padding=0)

        self.latlayer2_ = Conv_BN_ReLU(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)

        self.latlayer3_ = Conv_BN_ReLU(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_ = Conv_BN_ReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.smooth2_ = Conv_BN_ReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.smooth3_ = Conv_BN_ReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


        self.out_channels = out_channels * 4
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32',
                                                   default_initializer=paddle.nn.initializer.Normal(0,
                                                                                                    math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32',
                                                   default_initializer=paddle.nn.initializer.Constant(1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32',
                                                 default_initializer=paddle.nn.initializer.Constant(0.0))

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        f2, f3, f4, f5 = x
        p5 = self.toplayer_(f5)

        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4)
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3)
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2)
        p2 = self.smooth3_(p2)

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        fuse = paddle.concat([p2, p3, p4, p5], axis=1)
        return fuse