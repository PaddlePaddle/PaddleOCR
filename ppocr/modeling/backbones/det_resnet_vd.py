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

from paddle import nn
from paddle.nn import functional as F
from paddle import ParamAttr

__all__ = ["ResNet"]


class ResNet(nn.Layer):
    def __init__(self, in_channels=3, layers=50, **kwargs):
        """
        the Resnet backbone network for detection module.
        Args:
            params(dict): the super parameters for network build
        """
        super(ResNet, self).__init__()
        supported_layers = {
            18: {
                'depth': [2, 2, 2, 2],
                'block_class': BasicBlock
            },
            34: {
                'depth': [3, 4, 6, 3],
                'block_class': BasicBlock
            },
            50: {
                'depth': [3, 4, 6, 3],
                'block_class': BottleneckBlock
            },
            101: {
                'depth': [3, 4, 23, 3],
                'block_class': BottleneckBlock
            },
            152: {
                'depth': [3, 8, 36, 3],
                'block_class': BottleneckBlock
            },
            200: {
                'depth': [3, 12, 48, 3],
                'block_class': BottleneckBlock
            }
        }
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers.keys(), layers)
        is_3x3 = True

        depth = supported_layers[layers]['depth']
        block_class = supported_layers[layers]['block_class']

        num_filters = [64, 128, 256, 512]

        conv = []
        if is_3x3 == False:
            conv.append(
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    act='relu'))
        else:
            conv.append(
                ConvBNLayer(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    act='relu',
                    name='conv1_1'))
            conv.append(
                ConvBNLayer(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    act='relu',
                    name='conv1_2'))
            conv.append(
                ConvBNLayer(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    act='relu',
                    name='conv1_3'))
        self.conv1 = nn.Sequential(*conv)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = []
        self.out_channels = []
        in_ch = 64
        for block_index in range(len(depth)):
            block_list = []
            for i in range(depth[block_index]):
                if layers >= 50:
                    if layers in [101, 152, 200] and block_index == 2:
                        if i == 0:
                            conv_name = "res" + str(block_index + 2) + "a"
                        else:
                            conv_name = "res" + str(block_index +
                                                    2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block_index + 2) + chr(97 + i)
                else:
                    conv_name = "res" + str(block_index + 2) + chr(97 + i)
                block_list.append(
                    block_class(
                        in_channels=in_ch,
                        out_channels=num_filters[block_index],
                        stride=2 if i == 0 and block_index != 0 else 1,
                        if_first=block_index == i == 0,
                        name=conv_name))
                in_ch = block_list[-1].out_channels
            self.out_channels.append(in_ch)
            self.stages.append(nn.Sequential(*block_list))
        for i, stage in enumerate(self.stages):
            self.add_sublayer(sublayer=stage, name="stage{}".format(i))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance")

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNLayerNew(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayerNew, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance")

    def __call__(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class ShortCut(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, name, if_first=False):
        super(ShortCut, self).__init__()
        self.use_conv = True
        if in_channels != out_channels or stride != 1:
            if if_first:
                self.conv = ConvBNLayer(
                    in_channels, out_channels, 1, stride, name=name)
            else:
                self.conv = ConvBNLayerNew(
                    in_channels, out_channels, 1, stride, name=name)
        elif if_first:
            self.conv = ConvBNLayer(
                in_channels, out_channels, 1, stride, name=name)
        else:
            self.use_conv = False

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        return x


class BottleneckBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, name, if_first):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels * 4,
            stride=stride,
            if_first=if_first,
            name=name + "_branch1")
        self.out_channels = out_channels * 4

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.short(x)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, name, if_first):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + "_branch2b")
        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            if_first=if_first,
            name=name + "_branch1")
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.short(x)
        return F.relu(y)


if __name__ == '__main__':
    import paddle

    paddle.disable_static()
    x = paddle.zeros([1, 3, 640, 640])
    x = paddle.to_variable(x)
    print(x.shape)
    net = ResNet(layers=18)
    y = net(x)

    for stage in y:
        print(stage.shape)
    # paddle.save(net.state_dict(),'1.pth')
