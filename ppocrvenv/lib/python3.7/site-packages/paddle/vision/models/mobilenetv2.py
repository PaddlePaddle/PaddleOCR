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

import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F

from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'mobilenetv2_1.0':
    ('https://paddle-hapi.bj.bcebos.com/models/mobilenet_v2_x1.0.pdparams',
     '0340af0a901346c8d46f4529882fb63d')
}


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer=nn.BatchNorm2D):
        padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2D(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias_attr=False),
            norm_layer(out_planes),
            nn.ReLU6())


class InvertedResidual(nn.Layer):
    def __init__(self,
                 inp,
                 oup,
                 stride,
                 expand_ratio,
                 norm_layer=nn.BatchNorm2D):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(
                    inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer),
            nn.Conv2D(
                hidden_dim, oup, 1, 1, 0, bias_attr=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Layer):
    def __init__(self, scale=1.0, num_classes=1000, with_pool=True):
        """MobileNetV2 model from
        `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

        Args:
            scale (float): scale of channels in each layer. Default: 1.0.
            num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                                will not be defined. Default: 1000.
            with_pool (bool): use pool before the last fc layer or not. Default: True.

        Examples:
            .. code-block:: python

                from paddle.vision.models import MobileNetV2

                model = MobileNetV2()
        """
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.with_pool = with_pool
        input_channel = 32
        last_channel = 1280

        block = InvertedResidual
        round_nearest = 8
        norm_layer = nn.BatchNorm2D
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * scale, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, scale),
                                            round_nearest)
        features = [
            ConvBNReLU(
                3, input_channel, stride=2, norm_layer=norm_layer)
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer))
                input_channel = output_channel

        features.append(
            ConvBNReLU(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer))

        self.features = nn.Sequential(*features)

        if with_pool:
            self.pool2d_avg = nn.AdaptiveAvgPool2D(1)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)

        if self.with_pool:
            x = self.pool2d_avg(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.classifier(x)
        return x


def _mobilenet(arch, pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.load_dict(param)

    return model


def mobilenet_v2(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV2
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale: (float): scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            from paddle.vision.models import mobilenet_v2

            # build model
            model = mobilenet_v2()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v2(pretrained=True)

            # build mobilenet v2 with scale=0.5
            model = mobilenet_v2(scale=0.5)
    """
    model = _mobilenet(
        'mobilenetv2_' + str(scale), pretrained, scale=scale, **kwargs)
    return model
