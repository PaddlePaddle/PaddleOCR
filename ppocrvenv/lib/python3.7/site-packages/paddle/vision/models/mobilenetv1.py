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

import paddle
import paddle.nn as nn

from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'mobilenetv1_1.0':
    ('https://paddle-hapi.bj.bcebos.com/models/mobilenet_v1_x1.0.pdparams',
     '42a154c2f26f86e7457d6daded114e8c')
}


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)

        self._norm_layer = nn.BatchNorm2D(out_channels)
        self._act = nn.ReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._norm_layer(x)
        x = self._act(x)
        return x


class DepthwiseSeparable(nn.Layer):
    def __init__(self, in_channels, out_channels1, out_channels2, num_groups,
                 stride, scale):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvBNLayer(
            in_channels,
            int(out_channels1 * scale),
            kernel_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale))

        self._pointwise_conv = ConvBNLayer(
            int(out_channels1 * scale),
            int(out_channels2 * scale),
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, x):
        x = self._depthwise_conv(x)
        x = self._pointwise_conv(x)
        return x


class MobileNetV1(nn.Layer):
    """MobileNetV1 model from
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_.

    Args:
        scale (float): scale of channels in each layer. Default: 1.0.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            from paddle.vision.models import MobileNetV1

            model = MobileNetV1()
    """

    def __init__(self, scale=1.0, num_classes=1000, with_pool=True):
        super(MobileNetV1, self).__init__()
        self.scale = scale
        self.dwsl = []
        self.num_classes = num_classes
        self.with_pool = with_pool

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1)

        dws21 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(32 * scale),
                out_channels1=32,
                out_channels2=64,
                num_groups=32,
                stride=1,
                scale=scale),
            name="conv2_1")
        self.dwsl.append(dws21)

        dws22 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(64 * scale),
                out_channels1=64,
                out_channels2=128,
                num_groups=64,
                stride=2,
                scale=scale),
            name="conv2_2")
        self.dwsl.append(dws22)

        dws31 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=128,
                num_groups=128,
                stride=1,
                scale=scale),
            name="conv3_1")
        self.dwsl.append(dws31)

        dws32 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=256,
                num_groups=128,
                stride=2,
                scale=scale),
            name="conv3_2")
        self.dwsl.append(dws32)

        dws41 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=256,
                num_groups=256,
                stride=1,
                scale=scale),
            name="conv4_1")
        self.dwsl.append(dws41)

        dws42 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=512,
                num_groups=256,
                stride=2,
                scale=scale),
            name="conv4_2")
        self.dwsl.append(dws42)

        for i in range(5):
            tmp = self.add_sublayer(
                sublayer=DepthwiseSeparable(
                    in_channels=int(512 * scale),
                    out_channels1=512,
                    out_channels2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale),
                name="conv5_" + str(i + 1))
            self.dwsl.append(tmp)

        dws56 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(512 * scale),
                out_channels1=512,
                out_channels2=1024,
                num_groups=512,
                stride=2,
                scale=scale),
            name="conv5_6")
        self.dwsl.append(dws56)

        dws6 = self.add_sublayer(
            sublayer=DepthwiseSeparable(
                in_channels=int(1024 * scale),
                out_channels1=1024,
                out_channels2=1024,
                num_groups=1024,
                stride=1,
                scale=scale),
            name="conv6")
        self.dwsl.append(dws6)

        if with_pool:
            self.pool2d_avg = nn.AdaptiveAvgPool2D(1)

        if num_classes > 0:
            self.fc = nn.Linear(int(1024 * scale), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for dws in self.dwsl:
            x = dws(x)

        if self.with_pool:
            x = self.pool2d_avg(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        return x


def _mobilenet(arch, pretrained=False, **kwargs):
    model = MobileNetV1(**kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.load_dict(param)

    return model


def mobilenet_v1(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV1
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale: (float): scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            from paddle.vision.models import mobilenet_v1

            # build model
            model = mobilenet_v1()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v1(pretrained=True)

            # build mobilenet v1 with scale=0.5
            model = mobilenet_v1(scale=0.5)
    """
    model = _mobilenet(
        'mobilenetv1_' + str(scale), pretrained, scale=scale, **kwargs)
    return model
