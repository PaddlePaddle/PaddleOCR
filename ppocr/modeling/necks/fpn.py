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
"""
This code is refer from:
https://github.com/whai362/PSENet/blob/python3/models/neck/fpn.py
"""

import paddle.nn as nn
import paddle
import math
import paddle.nn.functional as F


class Conv_BN_ReLU(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(out_planes, momentum=0.1)
        self.relu = nn.ReLU()

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Normal(
                        0, math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(1.0))
                m.bias = paddle.create_parameter(
                    shape=m.bias.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(0.0))

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()

        # Top layer
        self.toplayer_ = Conv_BN_ReLU(
            in_channels[3], out_channels, kernel_size=1, stride=1, padding=0)
        # Lateral layers
        self.latlayer1_ = Conv_BN_ReLU(
            in_channels[2], out_channels, kernel_size=1, stride=1, padding=0)

        self.latlayer2_ = Conv_BN_ReLU(
            in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)

        self.latlayer3_ = Conv_BN_ReLU(
            in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.smooth2_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.smooth3_ = Conv_BN_ReLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.out_channels = out_channels * 4
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Normal(
                        0, math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(1.0))
                m.bias = paddle.create_parameter(
                    shape=m.bias.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(0.0))

    def _upsample(self, x, scale=1):
        return F.upsample(x, scale_factor=scale, mode='bilinear')

    def _upsample_add(self, x, y, scale=1):
        return F.upsample(x, scale_factor=scale, mode='bilinear') + y

    def forward(self, x):
        f2, f3, f4, f5 = x
        p5 = self.toplayer_(f5)

        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4, 2)
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3, 2)
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2, 2)
        p2 = self.smooth3_(p2)

        p3 = self._upsample(p3, 2)
        p4 = self._upsample(p4, 4)
        p5 = self._upsample(p5, 8)

        fuse = paddle.concat([p2, p3, p4, p5], axis=1)
        return fuse


class UpBlok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class RRGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.FNUM = len(cfg.fuc_k)
        self.SepareConv0 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(5, 1),
                stride=1,
                padding=1),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, 5),
                stride=1,
                padding=1),
            nn.Conv2d(
                in_channels, 1, kernel_size=1, stride=1, padding=0), )
        channels2 = in_channels + 1
        self.SepareConv1 = nn.Sequential(
            nn.Conv2d(
                channels2, channels2, kernel_size=(5, 1), stride=1, padding=1),
            nn.Conv2d(
                channels2, channels2, kernel_size=(1, 5), stride=1, padding=1),
            nn.Conv2d(
                channels2, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        f_map = list()
        for i in range(self.FNUM):
            if i == 0:
                f = self.SepareConv0(x)
                f_map.append(f)
                continue
            b1 = torch.cat([x, f_map[i - 1]], dim=1)
            f = self.SepareConv1(b1)
            f_map.append(f)
        f_map = torch.cat(f_map, dim=1)
        return f_map


class UpSample(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2D(
                in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2DTranspose(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        return self.conv_block(x)


class TPMFPN(nn.Layer):
    def __init__(self, in_channels=2048):
        super(TPMFPN, self).__init__()

        self.up_conv5 = nn.Conv2DTranspose(
            in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.up_conv4 = UpBlok(1024 + 256, 128)
        self.up_conv3 = UpBlok(512 + 128, 64)
        self.up_conv2 = UpBlok(256 + 64, 32)
        self.up_conv1 = UpBlok(64 + 32, 16)

    def forward(self, x):
        C1, C2, C3, C4, C5 = x
        up5 = self.up_conv5(C5)
        up5 = F.relu(up5)

        up4 = self.up_conv4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.up_conv3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.up_conv2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.up_conv1(C1, up2)

        return up1
