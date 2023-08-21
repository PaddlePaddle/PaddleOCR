# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 14:54
# @Author  : zhoujun
import paddle
from paddle import nn, ParamAttr


class DBHead(nn.Layer):
    def __init__(self, in_channels, out_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2D(
                in_channels,
                in_channels // 4,
                3,
                padding=1,
                weight_attr=ParamAttr(
                    initializer=nn.initializer.KaimingNormal())),
            nn.BatchNorm2D(
                in_channels // 4,
                weight_attr=ParamAttr(initializer=nn.initializer.Constant(1)),
                bias_attr=ParamAttr(initializer=nn.initializer.Constant(1e-4))),
            nn.ReLU(),
            nn.Conv2DTranspose(
                in_channels // 4,
                in_channels // 4,
                2,
                2,
                weight_attr=ParamAttr(
                    initializer=nn.initializer.KaimingNormal())),
            nn.BatchNorm2D(
                in_channels // 4,
                weight_attr=ParamAttr(initializer=nn.initializer.Constant(1)),
                bias_attr=ParamAttr(initializer=nn.initializer.Constant(1e-4))),
            nn.ReLU(),
            nn.Conv2DTranspose(
                in_channels // 4,
                1,
                2,
                2,
                weight_attr=nn.initializer.KaimingNormal()),
            nn.Sigmoid())

        self.thresh = self._init_thresh(in_channels)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = paddle.concat(
                (shrink_maps, threshold_maps, binary_maps), axis=1)
        else:
            y = paddle.concat((shrink_maps, threshold_maps), axis=1)
        return y

    def _init_thresh(self,
                     inner_channels,
                     serial=False,
                     smooth=False,
                     bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2D(
                in_channels,
                inner_channels // 4,
                3,
                padding=1,
                bias_attr=bias,
                weight_attr=ParamAttr(
                    initializer=nn.initializer.KaimingNormal())),
            nn.BatchNorm2D(
                inner_channels // 4,
                weight_attr=ParamAttr(initializer=nn.initializer.Constant(1)),
                bias_attr=ParamAttr(initializer=nn.initializer.Constant(1e-4))),
            nn.ReLU(),
            self._init_upsample(
                inner_channels // 4,
                inner_channels // 4,
                smooth=smooth,
                bias=bias),
            nn.BatchNorm2D(
                inner_channels // 4,
                weight_attr=ParamAttr(initializer=nn.initializer.Constant(1)),
                bias_attr=ParamAttr(initializer=nn.initializer.Constant(1e-4))),
            nn.ReLU(),
            self._init_upsample(
                inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels,
                       out_channels,
                       smooth=False,
                       bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(
                    scale_factor=2, mode='nearest'), nn.Conv2D(
                        in_channels,
                        inter_out_channels,
                        3,
                        1,
                        1,
                        bias_attr=bias,
                        weight_attr=ParamAttr(
                            initializer=nn.initializer.KaimingNormal()))
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2D(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=1,
                        bias_attr=True,
                        weight_attr=ParamAttr(
                            initializer=nn.initializer.KaimingNormal())))
            return nn.Sequential(module_list)
        else:
            return nn.Conv2DTranspose(
                in_channels,
                out_channels,
                2,
                2,
                weight_attr=ParamAttr(
                    initializer=nn.initializer.KaimingNormal()))

    def step_function(self, x, y):
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))
