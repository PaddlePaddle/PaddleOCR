# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import collections.abc
from itertools import repeat
from collections import OrderedDict  # pylint: disable=g-importing-member

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant, Normal, KaimingUniform
from functools import partial
from typing import Union, Callable, Type, List, Tuple

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
normal_ = Normal(mean=0.0, std=0.01)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
kaiming_normal_ = KaimingUniform(nonlinearity="relu")


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class StdConv2dSame(nn.Conv2D):
    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding="SAME",
        dilation=1,
        groups=1,
        bias_attr=False,
        eps=1e-6,
        is_export=False,
    ):
        padding, is_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
        )
        self.same_pad = is_dynamic
        self.export = is_export
        self.eps = eps

    def forward(self, x):
        if not self.training:
            self.export = True
        if self.same_pad:
            if self.export:
                x = pad_same_export(x, self._kernel_size, self._stride, self._dilation)
            else:
                x = pad_same(x, self._kernel_size, self._stride, self._dilation)
        running_mean = paddle.to_tensor([0] * self._out_channels, dtype="float32")
        running_variance = paddle.to_tensor([1] * self._out_channels, dtype="float32")
        if self.export:
            weight = paddle.reshape(
                F.batch_norm(
                    self.weight.reshape([1, self._out_channels, -1]),
                    running_mean,
                    running_variance,
                    momentum=0.0,
                    epsilon=self.eps,
                    use_global_stats=False,
                ),
                self.weight.shape,
            )
        else:
            weight = paddle.reshape(
                F.batch_norm(
                    self.weight.reshape([1, self._out_channels, -1]),
                    running_mean,
                    running_variance,
                    training=True,
                    momentum=0.0,
                    epsilon=self.eps,
                ),
                self.weight.shape,
            )
        x = F.conv2d(
            x,
            weight,
            self.bias,
            self._stride,
            self._padding,
            self._dilation,
            self._groups,
        )
        return x


class StdConv2d(nn.Conv2D):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-6,
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
        )
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            training=True,
            momentum=0.0,
            epsilon=self.eps,
        ).reshape_as(self.weight)
        x = F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class MaxPool2dSame(nn.MaxPool2D):
    """Tensorflow like 'SAME' wrapper for 2D max pooling"""

    def __init__(
        self,
        kernel_size: int,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
        is_export=False,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        self.export = is_export
        super(MaxPool2dSame, self).__init__(
            kernel_size, stride, (0, 0), dilation, ceil_mode
        )

    def forward(self, x):
        if not self.training:
            self.export = True
        if self.export:
            x = pad_same_export(x, self.ksize, self.stride, value=-float("inf"))
        else:
            x = pad_same(x, self.ksize, self.stride, value=-float("inf"))
        return F.max_pool2d(x, self.ksize, self.stride, (0, 0), self.ceil_mode)


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_pool2d(pool_type, kernel_size, stride=None, is_export=False, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(
        padding, kernel_size, stride=stride, **kwargs
    )
    if is_dynamic:
        if pool_type == "avg":
            return AvgPool2dSame(
                kernel_size, stride=stride, is_export=is_export, **kwargs
            )
        elif pool_type == "max":
            return MaxPool2dSame(
                kernel_size, stride=stride, is_export=is_export, **kwargs
            )
        else:
            assert False, f"Unsupported pool type {pool_type}"


def get_same_padding(x, k, s, d):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def get_same_padding_export(x, k, s, d):
    x = paddle.to_tensor(x)
    k = paddle.to_tensor(k)
    s = paddle.to_tensor(s)
    d = paddle.to_tensor(d)
    return paddle.max((paddle.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same_export(x, k, s, d=(1, 1), value=0):
    ih, iw = x.shape[-2:]
    pad_h, pad_w = get_same_padding_export(
        ih, k[0], s[0], d[0]
    ), get_same_padding_export(iw, k[1], s[1], d[1])
    pad_h = pad_h.cast(paddle.int32)
    pad_w = pad_w.cast(paddle.int32)
    pad_list = paddle.to_tensor(
        [
            (pad_w // 2),
            (pad_w - pad_w // 2).cast(paddle.int32),
            (pad_h // 2).cast(paddle.int32),
            (pad_h - pad_h // 2).cast(paddle.int32),
        ]
    )

    if pad_h > 0 or pad_w > 0:
        if len(pad_list.shape) == 2:
            pad_list = pad_list.squeeze(1)
        x = F.pad(x, pad_list.cast(paddle.int32), value=value)
    return x


def pad_same(x, k, s, d=(1, 1), value=0, pad_h=None, pad_w=None):
    ih, iw = x.shape[-2:]

    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(
        iw, k[1], s[1], d[1]
    )
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            value=value,
        )
    return x


class AvgPool2dSame(nn.AvgPool2D):
    """Tensorflow like 'SAME' wrapper for 2D average pooling"""

    def __init__(
        self,
        kernel_size: int,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(
            kernel_size, stride, (0, 0), ceil_mode, count_include_pad
        )

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def adaptive_pool_feat_mult(pool_type="avg"):
    if pool_type == "catavgmax":
        return 2
    else:
        return 1


class SelectAdaptivePool2d(nn.Layer):
    """Selectable global pooling layer with dynamic input kernel size"""

    def __init__(self, output_size=1, pool_type="fast", flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = (
            pool_type or ""
        )  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == "":
            self.pool = nn.Identity()  # pass through

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + "pool_type="
            + self.pool_type
            + ", flatten="
            + str(self.flatten)
            + ")"
        )


def _create_pool(num_features, num_classes, pool_type="avg", use_conv=False):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert (
            num_classes == 0 or use_conv
        ), "Pooling can only be disabled if classifier is also removed or conv classifier is used"
        flatten_in_pool = (
            False  # disable flattening if pooling is pass-through (no pooling)
        )
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2D(num_features, num_classes, 1, bias_attr=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias_attr=True)
    return fc


class ClassifierHead(nn.Layer):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
        self, in_chs, num_classes, pool_type="avg", drop_rate=0.0, use_conv=False
    ):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(
            in_chs, num_classes, pool_type, use_conv=use_conv
        )
        self.fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def forward(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        x = self.flatten(x)
        return x


class EvoNormBatch2d(nn.Layer):
    def __init__(
        self, num_features, apply_act=True, momentum=0.1, eps=1e-5, drop_block=None
    ):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = paddle.create_parameter(
            paddle.ones(num_features), dtype="float32"
        )
        self.bias = paddle.create_parameter(paddle.zeros(num_features), dtype="float32")
        self.v = (
            paddle.create_parameter(paddle.ones(num_features), dtype="float32")
            if apply_act
            else None
        )
        self.register_buffer("running_var", paddle.ones([num_features]))
        self.reset_parameters()

    def reset_parameters(self):
        ones_(self.weight)
        zeros_(self.bias)
        if self.apply_act:
            ones_(self.v)

    def forward(self, x):
        x_type = x.dtype
        if self.v is not None:
            running_var = self.running_var.view(1, -1, 1, 1)
            if self.training:
                var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
                n = x.numel() / x.shape[1]
                running_var = var.detach() * self.momentum * (
                    n / (n - 1)
                ) + running_var * (1 - self.momentum)
                self.running_var.copy_(running_var.view(self.running_var.shape))
            else:
                var = running_var
            v = self.v.to(dtype=x_type).reshape(1, -1, 1, 1)
            d = x * v + (
                x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps
            ).sqrt().to(dtype=x_type)
            d = d.max((var + self.eps).sqrt().to(dtype=x_type))
            x = x / d
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class EvoNormSample2d(nn.Layer):
    def __init__(
        self, num_features, apply_act=True, groups=32, eps=1e-5, drop_block=None
    ):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act
        self.groups = groups
        self.eps = eps
        self.weight = paddle.create_parameter(
            paddle.ones(num_features), dtype="float32"
        )
        self.bias = paddle.create_parameter(paddle.zeros(num_features), dtype="float32")
        self.v = (
            paddle.create_parameter(paddle.ones(num_features), dtype="float32")
            if apply_act
            else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        ones_(self.weight)
        zeros_(self.bias)
        if self.apply_act:
            ones_(self.v)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.v is not None:
            n = x * (x * self.v.view(1, -1, 1, 1)).sigmoid()
            x = x.reshape(B, self.groups, -1)
            x = (
                n.reshape(B, self.groups, -1)
                / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            )
            x = x.reshape(B, C, H, W)
        return x * self.weight.reshape([1, -1, 1, 1]) + self.bias.reshape([1, -1, 1, 1])


class GroupNormAct(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(
        self,
        num_channels,
        num_groups=32,
        eps=1e-5,
        affine=True,
        apply_act=True,
        act_layer=nn.ReLU,
        drop_block=None,
    ):
        super(GroupNormAct, self).__init__(num_groups, num_channels, epsilon=eps)
        if affine:
            self.weight = paddle.create_parameter([num_channels], dtype="float32")
            self.bias = paddle.create_parameter([num_channels], dtype="float32")
            ones_(self.weight)
            zeros_(self.bias)
        if act_layer is not None and apply_act:
            act_args = {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = F.group_norm(
            x,
            num_groups=self._num_groups,
            epsilon=self._epsilon,
            weight=self.weight,
            bias=self.bias,
        )
        x = self.act(x)
        return x


class BatchNormAct2d(nn.BatchNorm2D):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_layer=nn.ReLU,
        drop_block=None,
    ):
        super(BatchNormAct2d, self).__init__(
            num_features,
            epsilon=eps,
            momentum=momentum,
            use_global_stats=track_running_stats,
        )
        if act_layer is not None and apply_act:
            act_args = dict()
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def _forward_python(self, x):
        return super(BatchNormAct2d, self).forward(x)

    def forward(self, x):
        x = self._forward_python(x)
        x = self.act(x)
        return x


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = (
        conv_weight.float()
    )  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError("Weight format not supported by conversion.")
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= 3 / float(in_chans)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def named_apply(
    fn: Callable, module: nn.Layer, name="", depth_first=True, include_root=False
) -> nn.Layer:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (7, 7),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "stem.conv",
        "classifier": "head.fc",
        **kwargs,
    }


def make_div(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PreActBottleneck(nn.Layer):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(
        self,
        in_chs,
        out_chs=None,
        bottle_ratio=0.25,
        stride=1,
        dilation=1,
        first_dilation=None,
        groups=1,
        act_layer=None,
        conv_layer=None,
        norm_layer=None,
        proj_layer=None,
        drop_path_rate=0.0,
        is_export=False,
    ):
        super().__init__()
        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                first_dilation=first_dilation,
                preact=True,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None

        self.norm1 = norm_layer(in_chs)
        self.conv1 = conv_layer(in_chs, mid_chs, 1, is_export=is_export)
        self.norm2 = norm_layer(mid_chs)
        self.conv2 = conv_layer(
            mid_chs,
            mid_chs,
            3,
            stride=stride,
            dilation=first_dilation,
            groups=groups,
            is_export=is_export,
        )
        self.norm3 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1, is_export=is_export)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )

    def zero_init_last(self):
        zeros_(self.conv3.weight)

    def forward(self, x):
        x_preact = self.norm1(x)

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact)

        # residual branch
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        x = self.drop_path(x)
        return x + shortcut


class Bottleneck(nn.Layer):
    """Non Pre-activation bottleneck block, equiv to V1.5/V1b Bottleneck. Used for ViT."""

    def __init__(
        self,
        in_chs,
        out_chs=None,
        bottle_ratio=0.25,
        stride=1,
        dilation=1,
        first_dilation=None,
        groups=1,
        act_layer=None,
        conv_layer=None,
        norm_layer=None,
        proj_layer=None,
        drop_path_rate=0.0,
        is_export=False,
    ):
        super().__init__()
        first_dilation = first_dilation or dilation
        act_layer = act_layer or nn.ReLU
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                preact=False,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
                is_export=is_export,
            )
        else:
            self.downsample = None

        self.conv1 = conv_layer(in_chs, mid_chs, 1, is_export=is_export)
        self.norm1 = norm_layer(mid_chs)
        self.conv2 = conv_layer(
            mid_chs,
            mid_chs,
            3,
            stride=stride,
            dilation=first_dilation,
            groups=groups,
            is_export=is_export,
        )
        self.norm2 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1, is_export=is_export)
        self.norm3 = norm_layer(out_chs, apply_act=False)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )
        self.act3 = act_layer()

    def zero_init_last(self):
        zeros_(self.norm3.weight)

    def forward(self, x):
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # residual
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop_path(x)
        x = self.act3(x + shortcut)
        return x


class DownsampleConv(nn.Layer):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        dilation=1,
        first_dilation=None,
        preact=True,
        conv_layer=None,
        norm_layer=None,
        is_export=False,
    ):
        super(DownsampleConv, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=stride, is_export=is_export)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(x))


class DownsampleAvg(nn.Layer):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        dilation=1,
        first_dilation=None,
        preact=True,
        conv_layer=None,
        norm_layer=None,
        is_export=False,
    ):
        """AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = (
                AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2D
            )
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, exclusive=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1, is_export=is_export)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))


class ResNetStage(nn.Layer):
    """ResNet Stage."""

    def __init__(
        self,
        in_chs,
        out_chs,
        stride,
        dilation,
        depth,
        bottle_ratio=0.25,
        groups=1,
        avg_down=False,
        block_dpr=None,
        block_fn=PreActBottleneck,
        is_export=False,
        act_layer=None,
        conv_layer=None,
        norm_layer=None,
        **block_kwargs,
    ):
        super(ResNetStage, self).__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(
            act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer
        )
        proj_layer = DownsampleAvg if avg_down else DownsampleConv
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.0
            stride = stride if block_idx == 0 else 1
            self.blocks.add_sublayer(
                str(block_idx),
                block_fn(
                    prev_chs,
                    out_chs,
                    stride=stride,
                    dilation=dilation,
                    bottle_ratio=bottle_ratio,
                    groups=groups,
                    first_dilation=first_dilation,
                    proj_layer=proj_layer,
                    drop_path_rate=drop_path_rate,
                    is_export=is_export,
                    **layer_kwargs,
                    **block_kwargs,
                ),
            )
            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None

    def forward(self, x):
        x = self.blocks(x)
        return x


def is_stem_deep(stem_type):
    return any([s in stem_type for s in ("deep", "tiered")])


def create_resnetv2_stem(
    in_chs,
    out_chs=64,
    stem_type="",
    preact=True,
    conv_layer=StdConv2d,
    norm_layer=partial(GroupNormAct, num_groups=32),
    is_export=False,
):
    stem = OrderedDict()
    assert stem_type in (
        "",
        "fixed",
        "same",
        "deep",
        "deep_fixed",
        "deep_same",
        "tiered",
    )

    # NOTE conv padding mode can be changed by overriding the conv_layer def
    if is_stem_deep(stem_type):
        # A 3 deep 3x3  conv stack as in ResNet V1D models
        if "tiered" in stem_type:
            stem_chs = (3 * out_chs // 8, out_chs // 2)  # 'T' resnets in resnet.py
        else:
            stem_chs = (out_chs // 2, out_chs // 2)  # 'D' ResNets
        stem["conv1"] = conv_layer(
            in_chs, stem_chs[0], kernel_size=3, stride=2, is_export=is_export
        )
        stem["norm1"] = norm_layer(stem_chs[0])
        stem["conv2"] = conv_layer(
            stem_chs[0], stem_chs[1], kernel_size=3, stride=1, is_export=is_export
        )
        stem["norm2"] = norm_layer(stem_chs[1])
        stem["conv3"] = conv_layer(
            stem_chs[1], out_chs, kernel_size=3, stride=1, is_export=is_export
        )
        if not preact:
            stem["norm3"] = norm_layer(out_chs)
    else:
        # The usual 7x7 stem conv
        stem["conv"] = conv_layer(
            in_chs, out_chs, kernel_size=7, stride=2, is_export=is_export
        )
        if not preact:
            stem["norm"] = norm_layer(out_chs)

    if "fixed" in stem_type:
        # 'fixed' SAME padding approximation that is used in BiT models
        stem["pad"] = paddle.nn.Pad2D(
            1, mode="constant", value=0.0, data_format="NCHW", name=None
        )
        stem["pool"] = nn.MaxPool2D(kernel_size=3, stride=2, padding=0)
    elif "same" in stem_type:
        # full, input size based 'SAME' padding, used in ViT Hybrid model
        stem["pool"] = create_pool2d(
            "max", kernel_size=3, stride=2, padding="same", is_export=is_export
        )
    else:
        # the usual Pypaddle symmetric padding
        stem["pool"] = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
    stem_seq = nn.Sequential()
    for key, value in stem.items():
        stem_seq.add_sublayer(key, value)

    return stem_seq


class ResNetV2(nn.Layer):
    """Implementation of Pre-activation (v2) ResNet mode.

    Args:
      x: input images with shape [N, 1, H, W]

    Returns:
      The extracted features [N, 1, H//16, W//16]
    """

    def __init__(
        self,
        layers,
        channels=(256, 512, 1024, 2048),
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        output_stride=32,
        width_factor=1,
        stem_chs=64,
        stem_type="",
        avg_down=False,
        preact=True,
        act_layer=nn.ReLU,
        conv_layer=StdConv2d,
        norm_layer=partial(GroupNormAct, num_groups=32),
        drop_rate=0.0,
        drop_path_rate=0.0,
        zero_init_last=False,
        is_export=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.is_export = is_export
        wf = width_factor
        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_resnetv2_stem(
            in_chans,
            stem_chs,
            stem_type,
            preact,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
            is_export=is_export,
        )
        stem_feat = (
            ("stem.conv3" if is_stem_deep(stem_type) else "stem.conv")
            if preact
            else "stem.norm"
        )
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))

        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [
            x.tolist()
            for x in paddle.linspace(0, drop_path_rate, sum(layers)).split(layers)
        ]
        block_fn = PreActBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetStage(
                prev_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                depth=d,
                avg_down=avg_down,
                act_layer=act_layer,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
                block_dpr=bdpr,
                block_fn=block_fn,
                is_export=is_export,
            )
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [
                dict(
                    num_chs=prev_chs,
                    reduction=curr_stride,
                    module=f"stages.{stage_idx}",
                )
            ]
            self.stages.add_sublayer(str(stage_idx), stage)

        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else nn.Identity()
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
            use_conv=True,
        )

        self.init_weights(zero_init_last=zero_init_last)

    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    def load_pretrained(self, checkpoint_path, prefix="resnet/"):
        _load_weights(self, checkpoint_path, prefix)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
            use_conv=True,
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_weights(module: nn.Layer, name: str = "", zero_init_last=True):
    if isinstance(module, nn.Linear) or (
        "head.fc" in name and isinstance(module, nn.Conv2D)
    ):
        normal_(module.weight)
        zeros_(module.bias)
    elif isinstance(module, nn.Conv2D):
        kaiming_normal_(module.weight)
        if module.bias is not None:
            zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2D, nn.LayerNorm, nn.GroupNorm)):
        ones_(module.weight)
        zeros_(module.bias)
    elif zero_init_last and hasattr(module, "zero_init_last"):
        module.zero_init_last()


@paddle.no_grad()
def _load_weights(model: nn.Layer, checkpoint_path: str, prefix: str = "resnet/"):
    import numpy as np

    def t2p(conv_weights):
        """Possibly convert HWIO to OIHW."""
        if conv_weights.ndim == 4:
            conv_weights = conv_weights.transpose([3, 2, 0, 1])
        return paddle.to_tensor(conv_weights)

    weights = np.load(checkpoint_path)
    stem_conv_w = adapt_input_conv(
        model.stem.conv.weight.shape[1],
        t2p(weights[f"{prefix}root_block/standardized_conv2d/kernel"]),
    )
    model.stem.conv.weight.copy_(stem_conv_w)
    model.norm.weight.copy_(t2p(weights[f"{prefix}group_norm/gamma"]))
    model.norm.bias.copy_(t2p(weights[f"{prefix}group_norm/beta"]))
    if (
        isinstance(getattr(model.head, "fc", None), nn.Conv2D)
        and model.head.fc.weight.shape[0]
        == weights[f"{prefix}head/conv2d/kernel"].shape[-1]
    ):
        model.head.fc.weight.copy_(t2p(weights[f"{prefix}head/conv2d/kernel"]))
        model.head.fc.bias.copy_(t2p(weights[f"{prefix}head/conv2d/bias"]))
    for i, (sname, stage) in enumerate(model.stages.named_children()):
        for j, (bname, block) in enumerate(stage.blocks.named_children()):
            cname = "standardized_conv2d"
            block_prefix = f"{prefix}block{i + 1}/unit{j + 1:02d}/"
            block.conv1.weight.copy_(t2p(weights[f"{block_prefix}a/{cname}/kernel"]))
            block.conv2.weight.copy_(t2p(weights[f"{block_prefix}b/{cname}/kernel"]))
            block.conv3.weight.copy_(t2p(weights[f"{block_prefix}c/{cname}/kernel"]))
            block.norm1.weight.copy_(t2p(weights[f"{block_prefix}a/group_norm/gamma"]))
            block.norm2.weight.copy_(t2p(weights[f"{block_prefix}b/group_norm/gamma"]))
            block.norm3.weight.copy_(t2p(weights[f"{block_prefix}c/group_norm/gamma"]))
            block.norm1.bias.copy_(t2p(weights[f"{block_prefix}a/group_norm/beta"]))
            block.norm2.bias.copy_(t2p(weights[f"{block_prefix}b/group_norm/beta"]))
            block.norm3.bias.copy_(t2p(weights[f"{block_prefix}c/group_norm/beta"]))
            if block.downsample is not None:
                w = weights[f"{block_prefix}a/proj/{cname}/kernel"]
                block.downsample.conv.weight.copy_(t2p(w))
