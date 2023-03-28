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

from paddle import nn

from ppocr.modeling.backbones.det_mobilenet_v3 import ResidualUnit, ConvBNLayer, make_divisible

__all__ = ['MobileNetV3']


class MobileNetV3(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 model_name='small',
                 scale=0.5,
                 large_stride=None,
                 small_stride=None,
                 disable_se=False,
                 **kwargs):
        super(MobileNetV3, self).__init__()
        self.disable_se = disable_se
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), "large_stride type must " \
                                               "be list but got {}".format(type(large_stride))
        assert isinstance(small_stride, list), "small_stride type must " \
                                               "be list but got {}".format(type(small_stride))
        assert len(large_stride) == 4, "large_stride length must be " \
                                       "4 but got {}".format(len(large_stride))
        assert len(small_stride) == 4, "small_stride length must be " \
                                       "4 but got {}".format(len(small_stride))

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', large_stride[0]],
                [3, 64, 24, False, 'relu', (large_stride[1], 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (large_stride[2], 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 1],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', (large_stride[3], 1)],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', (small_stride[2], 1)],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', (small_stride[3], 1)],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scales are {} but input scale is {}".format(supported_scale, scale)

        inplanes = 16
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish')
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            se = se and not self.disable_se
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.conv2 = ConvBNLayer(
            in_channels=inplanes,
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish')

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
