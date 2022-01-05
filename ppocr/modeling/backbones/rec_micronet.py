from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn

from ppocr.optimizer.dyshiftmax import DYShiftMax
from ppocr.modeling.backbones.det_mobilenet_v3 import make_divisible

M0_cfgs = [
    # s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
    [2, 1, 8, 3, 2, 2, 0, 4, 8, 2, 2, 2, 0, 1, 1],
    [2, 1, 12, 3, 2, 2, 0, 8, 12, 4, 4, 2, 2, 1, 1],
    [2, 1, 16, 5, 2, 2, 0, 12, 16, 4, 4, 2, 2, 1, 1],
    [1, 1, 32, 5, 1, 4, 4, 4, 32, 4, 4, 2, 2, 1, 1],
    [2, 1, 64, 5, 1, 4, 8, 8, 64, 8, 8, 2, 2, 1, 1],
    [1, 1, 96, 3, 1, 4, 8, 8, 96, 8, 8, 2, 2, 1, 2],
    [1, 1, 384, 3, 1, 4, 12, 12, 0, 0, 0, 2, 2, 1, 2],
]
M1_cfgs = [
    # s, n, c, ks, c1, c2, g1, g2, c3, g3, g4
    [2, 1, 8, 3, 2, 2, 0, 6, 8, 2, 2, 2, 0, 1, 1],
    [2, 1, 16, 3, 2, 2, 0, 8, 16, 4, 4, 2, 2, 1, 1],
    [2, 1, 16, 5, 2, 2, 0, 16, 16, 4, 4, 2, 2, 1, 1],
    [1, 1, 32, 5, 1, 6, 4, 4, 32, 4, 4, 2, 2, 1, 1],
    [2, 1, 64, 5, 1, 6, 8, 8, 64, 8, 8, 2, 2, 1, 1],
    [1, 1, 96, 3, 1, 6, 8, 8, 96, 8, 8, 2, 2, 1, 2],
    [1, 1, 576, 3, 1, 6, 12, 12, 0, 0, 0, 2, 2, 1, 2],
]
M2_cfgs = [
    # s, n, c, ks, c1, c2, g1, g2, c3, g3, g4
    [2, 1, 12, 3, 2, 2, 0, 8, 12, 4, 4, 2, 0, 1, 1],
    [2, 1, 16, 3, 2, 2, 0, 12, 16, 4, 4, 2, 2, 1, 1],
    [1, 1, 24, 3, 2, 2, 0, 16, 24, 4, 4, 2, 2, 1, 1],
    [2, 1, 32, 5, 1, 6, 6, 6, 32, 4, 4, 2, 2, 1, 1],
    [1, 1, 32, 5, 1, 6, 8, 8, 32, 4, 4, 2, 2, 1, 2],
    [1, 1, 64, 5, 1, 6, 8, 8, 64, 8, 8, 2, 2, 1, 2],
    [2, 1, 96, 5, 1, 6, 8, 8, 96, 8, 8, 2, 2, 1, 2],
    [1, 1, 128, 3, 1, 6, 12, 12, 128, 8, 8, 2, 2, 1, 2],
    [1, 1, 768, 3, 1, 6, 16, 16, 0, 0, 0, 2, 2, 1, 2],
]
M3_cfgs = [
    # s, n, c, ks, c1, c2, g1, g2, c3, g3, g4
    [2, 1, 16, 3, 2, 2, 0, 12, 16, 4, 4, 0, 2, 0, 1],
    [2, 1, 24, 3, 2, 2, 0, 16, 24, 4, 4, 0, 2, 0, 1],
    [1, 1, 24, 3, 2, 2, 0, 24, 24, 4, 4, 0, 2, 0, 1],
    [2, 1, 32, 5, 1, 6, 6, 6, 32, 4, 4, 0, 2, 0, 1],
    [1, 1, 32, 5, 1, 6, 8, 8, 32, 4, 4, 0, 2, 0, 2],
    [1, 1, 64, 5, 1, 6, 8, 8, 48, 8, 8, 0, 2, 0, 2],
    [1, 1, 80, 5, 1, 6, 8, 8, 80, 8, 8, 0, 2, 0, 2],
    [1, 1, 80, 5, 1, 6, 10, 10, 80, 8, 8, 0, 2, 0, 2],
    [1, 1, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0, 2],
    [1, 1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0, 2],
    [1, 1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0, 2],
    [1, 1, 432, 3, 1, 3, 12, 12, 0, 0, 0, 0, 2, 0, 2],
]


def get_micronet_config(mode):
    return eval(mode + '_cfgs')


class h_sigmoid(nn.Layer):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Layer):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class MaxGroupPooling(nn.Layer):
    def __init__(self, channel_per_group=2):
        super(MaxGroupPooling, self).__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x
        # max op
        b, c, h, w = x.shape

        # reshape
        y = paddle.reshape(x, [b, c // self.channel_per_group, -1, h, w])
        out = paddle.max(y, axis=2)
        return out


class SpatialSepConvSF(nn.Layer):
    def __init__(self, inp, oups, kernel_size, stride):
        super(SpatialSepConvSF, self).__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                oup1, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0),
                bias_attr=False,
                groups=1),
            nn.BatchNorm2D(oup1),
            nn.Conv2D(
                oup1,
                oup1 * oup2, (1, kernel_size), (1, stride),
                (0, kernel_size // 2),
                bias_attr=False,
                groups=oup1),
            nn.BatchNorm2D(oup1 * oup2),
            ChannelShuffle(oup1), )

    def forward(self, x):
        out = self.conv(x)
        return out


class ChannelShuffle(nn.Layer):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.shape

        channels_per_group = c // self.groups

        # reshape
        x = paddle.reshape(x, [b, self.groups, channels_per_group, h, w])

        x = paddle.transpose(x, (0, 2, 1, 3, 4))
        out = paddle.reshape(x, [b, -1, h, w])

        return out


class StemLayer(nn.Layer):
    def __init__(self, inp, oup, stride, groups=(4, 4)):
        super(StemLayer, self).__init__()

        g1, g2 = groups
        self.stem = nn.Sequential(
            SpatialSepConvSF(inp, groups, 3, stride),
            MaxGroupPooling(2) if g1 * g2 == 2 * oup else nn.ReLU6())

    def forward(self, x):
        out = self.stem(x)
        return out


class DepthSpatialSepConv(nn.Layer):
    def __init__(self, inp, expand, kernel_size, stride):
        super(DepthSpatialSepConv, self).__init__()

        exp1, exp2 = expand

        hidden_dim = inp * exp1
        oup = inp * exp1 * exp2

        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                inp * exp1, (kernel_size, 1), (stride, 1),
                (kernel_size // 2, 0),
                bias_attr=False,
                groups=inp),
            nn.BatchNorm2D(inp * exp1),
            nn.Conv2D(
                hidden_dim,
                oup, (1, kernel_size),
                1, (0, kernel_size // 2),
                bias_attr=False,
                groups=hidden_dim),
            nn.BatchNorm2D(oup))

    def forward(self, x):
        x = self.conv(x)
        return x


class GroupConv(nn.Layer):
    def __init__(self, inp, oup, groups=2):
        super(GroupConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv2D(
                inp, oup, 1, 1, 0, bias_attr=False, groups=self.groups[0]),
            nn.BatchNorm2D(oup))

    def forward(self, x):
        x = self.conv(x)
        return x


class DepthConv(nn.Layer):
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                oup,
                kernel_size,
                stride,
                kernel_size // 2,
                bias_attr=False,
                groups=inp),
            nn.BatchNorm2D(oup))

    def forward(self, x):
        out = self.conv(x)
        return out


class DYMicroBlock(nn.Layer):
    def __init__(self,
                 inp,
                 oup,
                 kernel_size=3,
                 stride=1,
                 ch_exp=(2, 2),
                 ch_per_group=4,
                 groups_1x1=(1, 1),
                 depthsep=True,
                 shuffle=False,
                 activation_cfg=None):
        super(DYMicroBlock, self).__init__()

        self.identity = stride == 1 and inp == oup

        y1, y2, y3 = activation_cfg['dy']
        act_reduction = 8 * activation_cfg['ratio']
        init_a = activation_cfg['init_a']
        init_b = activation_cfg['init_b']

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1
        hidden_dim2 = inp * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                DYShiftMax(
                    hidden_dim2,
                    hidden_dim2,
                    act_max=2.0,
                    act_relu=True if y2 == 2 else False,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=False) if y2 > 0 else nn.ReLU6(),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                ChannelShuffle(hidden_dim2 // 2)
                if shuffle and y2 != 0 else nn.Sequential(),
                GroupConv(hidden_dim2, oup, (g1, g2)),
                DYShiftMax(
                    oup,
                    oup,
                    act_max=2.0,
                    act_relu=False,
                    init_a=[1.0, 0.0],
                    reduction=act_reduction // 2,
                    init_b=[0.0, 0.0],
                    g=(g1, g2),
                    expansion=False) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle(oup // 2)
                if shuffle and oup % 2 == 0 and y3 != 0 else nn.Sequential(), )
        elif g2 == 0:
            self.layers = nn.Sequential(
                GroupConv(inp, hidden_dim2, gs1),
                DYShiftMax(
                    hidden_dim2,
                    hidden_dim2,
                    act_max=2.0,
                    act_relu=False,
                    init_a=[1.0, 0.0],
                    reduction=act_reduction,
                    init_b=[0.0, 0.0],
                    g=gs1,
                    expansion=False) if y3 > 0 else nn.Sequential(), )
        else:
            self.layers = nn.Sequential(
                GroupConv(inp, hidden_dim2, gs1),
                DYShiftMax(
                    hidden_dim2,
                    hidden_dim2,
                    act_max=2.0,
                    act_relu=True if y1 == 2 else False,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=False) if y1 > 0 else nn.ReLU6(),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride)
                if depthsep else
                DepthConv(hidden_dim2, hidden_dim2, kernel_size, stride),
                nn.Sequential(),
                DYShiftMax(
                    hidden_dim2,
                    hidden_dim2,
                    act_max=2.0,
                    act_relu=True if y2 == 2 else False,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=True) if y2 > 0 else nn.ReLU6(),
                ChannelShuffle(hidden_dim2 // 4)
                if shuffle and y1 != 0 and y2 != 0 else nn.Sequential()
                if y1 == 0 and y2 == 0 else ChannelShuffle(hidden_dim2 // 2),
                GroupConv(hidden_dim2, oup, (g1, g2)),
                DYShiftMax(
                    oup,
                    oup,
                    act_max=2.0,
                    act_relu=False,
                    init_a=[1.0, 0.0],
                    reduction=act_reduction // 2
                    if oup < hidden_dim2 else act_reduction,
                    init_b=[0.0, 0.0],
                    g=(g1, g2),
                    expansion=False) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle(oup // 2)
                if shuffle and y3 != 0 else nn.Sequential(), )

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.identity:
            out = out + identity

        return out


class MicroNet(nn.Layer):
    def __init__(self, mode='M0', **kwargs):
        super(MicroNet, self).__init__()

        self.cfgs = get_micronet_config(mode)

        activation_cfg = {}
        if mode == 'M0':
            input_channel = 4
            stem_groups = 2, 2
            out_ch = 384
            activation_cfg['init_a'] = 1.0, 1.0
            activation_cfg['init_b'] = 0.0, 0.0
        elif mode == 'M1':
            input_channel = 6
            stem_groups = 3, 2
            out_ch = 576
            activation_cfg['init_a'] = 1.0, 1.0
            activation_cfg['init_b'] = 0.0, 0.0
        elif mode == 'M2':
            input_channel = 8
            stem_groups = 4, 2
            out_ch = 768
            activation_cfg['init_a'] = 1.0, 1.0
            activation_cfg['init_b'] = 0.0, 0.0
        elif mode == 'M3':
            input_channel = 12
            stem_groups = 4, 3
            out_ch = 432
            activation_cfg['init_a'] = 1.0, 0.5
            activation_cfg['init_b'] = 0.0, 0.5
        else:
            raise NotImplementedError("mode[" + mode +
                                      "_model] is not implemented!")

        layers = [StemLayer(3, input_channel, stride=2, groups=stem_groups)]

        for idx, val in enumerate(self.cfgs):
            s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r = val

            t1 = (c1, c2)
            gs1 = (g1, g2)
            gs2 = (c3, g3, g4)
            activation_cfg['dy'] = [y1, y2, y3]
            activation_cfg['ratio'] = r

            output_channel = c
            layers.append(
                DYMicroBlock(
                    input_channel,
                    output_channel,
                    kernel_size=ks,
                    stride=s,
                    ch_exp=t1,
                    ch_per_group=gs1,
                    groups_1x1=gs2,
                    depthsep=True,
                    shuffle=True,
                    activation_cfg=activation_cfg, ))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(
                    DYMicroBlock(
                        input_channel,
                        output_channel,
                        kernel_size=ks,
                        stride=1,
                        ch_exp=t1,
                        ch_per_group=gs1,
                        groups_1x1=gs2,
                        depthsep=True,
                        shuffle=True,
                        activation_cfg=activation_cfg, ))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        self.out_channels = make_divisible(out_ch)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x
