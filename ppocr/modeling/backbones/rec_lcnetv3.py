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

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Constant
import paddle.nn.functional as F
from paddle.nn import AdaptiveAvgPool2D, BatchNorm2D, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal

MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

NUM_CONV_BRANCHES = 4

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, (2, 1), False], [3, 128, 128, 1, False]],
    "blocks5":
    [[3, 128, 256, (1, 2), False], [5, 256, 256, 1, False],
     [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, (2, 1), True], [5, 512, 512, 1, True],
                [5, 512, 512, (2, 1), False], [5, 512, 512, 1, False]]
}


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class whswish_b(nn.Layer):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.act = nn.Hardswish()

        self.scale = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=scale_value),
            attr=ParamAttr(learning_rate=0.1))
        self.add_parameter("scale", self.scale)
        self.bias = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=bias_value),
            attr=ParamAttr(learning_rate=0.1))
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * self.act(x) + self.bias


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1,
                 lr_mult=1.0):
        super().__init__()
        self.stride = stride

        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), learning_rate=lr_mult),
            bias_attr=False)

        self.bn = BatchNorm2D(
            num_filters,
            weight_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult))
        self.hardswish = whswish_b()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.stride != 2:
            x = self.hardswish(x)
        return x


class MobileOneBlock(nn.Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int=1,
                 groups: int=1,
                 num_conv_branches: int=1) -> None:
        super(MobileOneBlock, self).__init__()
        self.is_repped = False
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        padding = (kernel_size - 1) // 2

        self.activation = whswish_b()

        # Re-parameterizable skip connection
        self.rbr_skip = nn.BatchNorm2D(num_features=in_channels) \
            if out_channels == in_channels and stride == 1 else None

        # Re-parameterizable conv branches
        rbr_conv = list()
        for _ in range(self.num_conv_branches):
            rbr_conv.append(
                self._conv_bn(
                    kernel_size=kernel_size, padding=padding))
        self.rbr_conv = nn.LayerList(rbr_conv)

        # Re-parameterizable scale branch
        self.rbr_scale = None
        if kernel_size > 1:
            self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)
        self.w = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=1.),
            attr=ParamAttr(learning_rate=0.1))
        self.b = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=0.),
            attr=ParamAttr(learning_rate=0.1))

    def forward(self, x):
        """ Apply forward pass. """
        if self.is_repped:
            out = self.reparam_conv(x)
            out = self.w * out + self.b
            if self.stride != 2:
                out = self.activation(out)
            return out
        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out

        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        out = self.w * out + self.b
        if self.stride != 2:
            out = self.activation(out)
        return out

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_sublayer(
            'conv',
            nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias_attr=False))
        mod_list.add_sublayer(
            'bn', nn.BatchNorm2D(num_features=self.out_channels))
        return mod_list

    def rep(self):
        if self.is_repped:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2D(
            in_channels=self.rbr_conv[0].conv._in_channels,
            out_channels=self.rbr_conv[0].conv._out_channels,
            kernel_size=self.rbr_conv[0].conv._kernel_size,
            stride=self.rbr_conv[0].conv._stride,
            padding=self.rbr_conv[0].conv._padding,
            dilation=self.rbr_conv[0].conv._dilation,
            groups=self.rbr_conv[0].conv._groups)
        self.reparam_conv.weight.set_value(kernel)
        self.reparam_conv.bias.set_value(bias)

        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.is_repped = True

        return kernel, bias

    def _get_kernel_bias(self):
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = paddle.nn.functional.pad(kernel_scale,
                                                    [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = paddle.zeros(
                    (self.in_channels, input_dim, self.kernel_size,
                     self.kernel_size),
                    dtype=branch.weight.dtype)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class DepthwiseSeparable(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False,
                 lr_mult=1.0):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = MobileOneBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=num_channels,
            num_conv_branches=NUM_CONV_BRANCHES)
        if use_se:
            self.se = SEModule(num_channels, lr_mult=lr_mult)
        self.pw_conv = MobileOneBlock(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            stride=1,
            num_conv_branches=NUM_CONV_BRANCHES)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class PPLCNet(nn.Layer):
    def __init__(self,
                 stages_pattern,
                 scale=1.0,
                 class_num=1000,
                 dropout_prob=0.2,
                 class_expand=1280,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 stride_list=[2, 2, 2, 2, 2],
                 use_last_conv=True,
                 return_patterns=None,
                 return_stages=None,
                 **kwargs):
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand
        self.lr_mult_list = lr_mult_list
        self.use_last_conv = use_last_conv
        self.stride_list = stride_list
        self.net_config = NET_CONFIG
        if isinstance(self.lr_mult_list, str):
            self.lr_mult_list = eval(self.lr_mult_list)

        assert isinstance(self.lr_mult_list, (
            list, tuple
        )), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list))
        assert len(self.lr_mult_list
                   ) == 6, "lr_mult_list length should be 6 but got {}".format(
                       len(self.lr_mult_list))

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2,
            lr_mult=self.lr_mult_list[0])

        self.blocks2 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[1]) for i, (k, in_c, out_c, s, se) in
            enumerate(self.net_config["blocks2"])
        ])

        self.blocks3 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[2]) for i, (k, in_c, out_c, s, se) in
            enumerate(self.net_config["blocks3"])
        ])

        self.blocks4 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[3]) for i, (k, in_c, out_c, s, se) in
            enumerate(self.net_config["blocks4"])
        ])

        self.blocks5 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[4]) for i, (k, in_c, out_c, s, se) in
            enumerate(self.net_config["blocks5"])
        ])

        self.blocks6 = nn.Sequential(* [
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                lr_mult=self.lr_mult_list[5]) for i, (k, in_c, out_c, s, se) in
            enumerate(self.net_config["blocks6"])
        ])

        self.avg_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        if self.use_last_conv:
            self.last_conv = Conv2D(
                in_channels=make_divisible(self.net_config["blocks6"][-1][2] *
                                           scale),
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.hardswish = whswish_b()
            self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        else:
            self.last_conv = None
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = Linear(
            self.class_expand if self.use_last_conv else
            make_divisible(self.net_config["blocks6"][-1][2] * scale),
            class_num)

        self.out_channels = make_divisible(512 * scale)

    def forward(self, x):
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        x = F.adaptive_avg_pool2d(x, [1, x.shape[-1] // 2])

        return x


def LCNetv3(**kwargs):
    """
    PPLCNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_0` model depends on args.
    """
    model = PPLCNet(
        scale=0.95, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    return model
