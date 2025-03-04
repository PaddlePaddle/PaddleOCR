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
https://github.com/PaddlePaddle/PaddleClas/blob/2f36cab604e439b59d1a854df34ece3b10d888e3/ppcls/arch/backbone/legendary_models/pp_hgnet_v2.py
"""

from __future__ import absolute_import, division, print_function

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Conv2D, BatchNorm, Linear, BatchNorm2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
from paddle.regularizer import L2Decay

from typing import Tuple, List, Dict, Union, Callable, Any
from ppocr.modeling.backbones.rec_donut_swin import DonutSwinModelOutput


class IdentityBasedConv1x1(nn.Conv2D):
    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias_attr=False,
        )

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = paddle.to_tensor(id_value)
        self.weight.set_value(paddle.zeros_like(self.weight))

    def forward(self, input):
        kernel = self.weight + self.id_tensor
        result = F.conv2d(
            input,
            kernel,
            None,
            stride=1,
            padding=0,
            dilation=self._dilation,
            groups=self._groups,
        )
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor


class BNAndPad(nn.Layer):
    def __init__(
        self,
        pad_pixels,
        num_features,
        epsilon=1e-5,
        momentum=0.1,
        last_conv_bias=None,
        bn=nn.BatchNorm2D,
    ):
        super().__init__()
        self.bn = bn(num_features, momentum=momentum, epsilon=epsilon)
        self.pad_pixels = pad_pixels
        self.last_conv_bias = last_conv_bias

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            bias = -self.bn._mean
            if self.last_conv_bias is not None:
                bias += self.last_conv_bias
            pad_values = self.bn.bias + self.bn.weight * (
                bias / paddle.sqrt(self.bn._variance + self.bn._epsilon)
            )
            """ pad """
            # TODO: n,h,w,c format is not supported yet
            n, c, h, w = output.shape
            values = pad_values.reshape([1, -1, 1, 1])
            w_values = values.expand([n, -1, self.pad_pixels, w])
            x = paddle.concat([w_values, output, w_values], axis=2)
            h = h + self.pad_pixels * 2
            h_values = values.expand([n, -1, h, self.pad_pixels])
            x = paddle.concat([h_values, x, h_values], axis=3)
            output = x
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def _mean(self):
        return self.bn._mean

    @property
    def _variance(self):
        return self.bn._variance

    @property
    def _epsilon(self):
        return self.bn._epsilon


def conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
):
    conv_layer = nn.Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_attr=False,
        padding_mode=padding_mode,
    )
    bn_layer = nn.BatchNorm2D(num_features=out_channels)
    se = nn.Sequential()
    se.add_sublayer("conv", conv_layer)
    se.add_sublayer("bn", bn_layer)
    return se


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn._variance + bn._epsilon).sqrt()
    return (
        kernel * ((gamma / std).reshape([-1, 1, 1, 1])),
        bn.bias - bn._mean * gamma / std,
    )


def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.transpose([1, 0, 2, 3]))
        b_hat = (k2 * b1.reshape([1, -1, 1, 1])).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.transpose([1, 0, 2, 3])
        k1_group_width = k1.shape[0] // groups
        k2_group_width = k2.shape[0] // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width : (g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width : (g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append(
                (
                    k2_slice
                    * b1[g * k1_group_width : (g + 1) * k1_group_width].reshape(
                        [1, -1, 1, 1]
                    )
                ).sum((1, 2, 3))
            )
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIV_depthconcat(kernels, biases):
    return paddle.cat(kernels, axis=0), paddle.cat(biases)


def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = paddle.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = (
        1.0 / kernel_size**2
    )
    return k


def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.shape[2]) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.shape[3]) // 2
    return F.pad(
        kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
    )


class DiverseBranchBlock(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
        is_repped=False,
        single_init=False,
        **kwargs,
    ):
        super().__init__()

        padding = (filter_size - 1) // 2
        dilation = 1

        in_channels = num_channels
        out_channels = num_filters
        kernel_size = filter_size
        internal_channels_1x1_3x3 = None
        nonlinear = act

        self.is_repped = is_repped

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nn.ReLU()

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if is_repped:
            self.dbb_reparam = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=True,
            )
        else:
            self.dbb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_sublayer(
                    "conv",
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=groups,
                        bias_attr=False,
                    ),
                )
                self.dbb_avg.add_sublayer(
                    "bn", BNAndPad(pad_pixels=padding, num_features=out_channels)
                )
                self.dbb_avg.add_sublayer(
                    "avg",
                    nn.AvgPool2D(kernel_size=kernel_size, stride=stride, padding=0),
                )
                self.dbb_1x1 = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                )
            else:
                self.dbb_avg.add_sublayer(
                    "avg",
                    nn.AvgPool2D(
                        kernel_size=kernel_size, stride=stride, padding=padding
                    ),
                )

            self.dbb_avg.add_sublayer("avgbn", nn.BatchNorm2D(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = (
                    in_channels if groups < out_channels else 2 * in_channels
                )  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_sublayer(
                    "idconv1", IdentityBasedConv1x1(channels=in_channels, groups=groups)
                )
            else:
                self.dbb_1x1_kxk.add_sublayer(
                    "conv1",
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=internal_channels_1x1_3x3,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=groups,
                        bias_attr=False,
                    ),
                )
            self.dbb_1x1_kxk.add_sublayer(
                "bn1",
                BNAndPad(pad_pixels=padding, num_features=internal_channels_1x1_3x3),
            )
            self.dbb_1x1_kxk.add_sublayer(
                "conv2",
                nn.Conv2D(
                    in_channels=internal_channels_1x1_3x3,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias_attr=False,
                ),
            )
            self.dbb_1x1_kxk.add_sublayer("bn2", nn.BatchNorm2D(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def forward(self, inputs):
        if self.is_repped:
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, "dbb_1x1"):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            paddle.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            paddle.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            paddle.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            paddle.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            paddle.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(
            self.dbb_origin.conv.weight, self.dbb_origin.bn
        )

        if hasattr(self, "dbb_1x1"):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, "idconv1"):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(
            k_1x1_kxk_first, self.dbb_1x1_kxk.bn1
        )
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(
            self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2
        )
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(
            k_1x1_kxk_first,
            b_1x1_kxk_first,
            k_1x1_kxk_second,
            b_1x1_kxk_second,
            groups=self.groups,
        )

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg, self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, "conv"):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(
                self.dbb_avg.conv.weight, self.dbb_avg.bn
            )
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(
                k_1x1_avg_first,
                b_1x1_avg_first,
                k_1x1_avg_second,
                b_1x1_avg_second,
                groups=self.groups,
            )
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch(
            (k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
            (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged),
        )

    def re_parameterize(self):
        if self.is_repped:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2D(
            in_channels=self.dbb_origin.conv._in_channels,
            out_channels=self.dbb_origin.conv._out_channels,
            kernel_size=self.dbb_origin.conv._kernel_size,
            stride=self.dbb_origin.conv._stride,
            padding=self.dbb_origin.conv._padding,
            dilation=self.dbb_origin.conv._dilation,
            groups=self.dbb_origin.conv._groups,
            bias_attr=True,
        )

        self.dbb_reparam.weight.set_value(kernel)
        self.dbb_reparam.bias.set_value(bias)

        self.__delattr__("dbb_origin")
        self.__delattr__("dbb_avg")
        if hasattr(self, "dbb_1x1"):
            self.__delattr__("dbb_1x1")
        self.__delattr__("dbb_1x1_kxk")
        self.is_repped = True


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.res_dict = {}
        self.res_name = self.full_name()
        self.pruner = None
        self.quanter = None

        self.init_net(*args, **kwargs)

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"logits": output}
        # 'list' is needed to avoid error raised by popping self.res_dict
        for res_key in list(self.res_dict):
            # clear the res_dict because the forward process may change according to input
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def init_net(
        self,
        stages_pattern=None,
        return_patterns=None,
        return_stages=None,
        freeze_befor=None,
        stop_after=None,
        *args,
        **kwargs,
    ):
        # init the output of net
        if return_patterns or return_stages:
            if return_patterns and return_stages:
                msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."

                return_stages = None

            if return_stages is True:
                return_patterns = stages_pattern

            # return_stages is int or bool
            if type(return_stages) is int:
                return_stages = [return_stages]
            if isinstance(return_stages, list):
                if max(return_stages) > len(stages_pattern) or min(return_stages) < 0:
                    msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."

                    return_stages = [
                        val
                        for val in return_stages
                        if val >= 0 and val < len(stages_pattern)
                    ]
                return_patterns = [stages_pattern[i] for i in return_stages]

            if return_patterns:
                # call update_res function after the __init__ of the object has completed execution, that is, the constructing of layer or model has been completed.
                def update_res_hook(layer, input):
                    self.update_res(return_patterns)

                self.register_forward_pre_hook(update_res_hook)

        # freeze subnet
        if freeze_befor is not None:
            self.freeze_befor(freeze_befor)

        # set subnet to Identity
        if stop_after is not None:
            self.stop_after(stop_after)

    def init_res(self, stages_pattern, return_patterns=None, return_stages=None):

        if return_patterns and return_stages:
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern
        # return_stages is int or bool
        if type(return_stages) is int:
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if max(return_stages) > len(stages_pattern) or min(return_stages) < 0:
                return_stages = [
                    val
                    for val in return_stages
                    if val >= 0 and val < len(stages_pattern)
                ]
            return_patterns = [stages_pattern[i] for i in return_stages]

        if return_patterns:
            self.update_res(return_patterns)

    def replace_sub(self, *args, **kwargs) -> None:
        msg = "The function 'replace_sub()' is deprecated, please use 'upgrade_sublayer()' instead."
        raise DeprecationWarning(msg)

    def upgrade_sublayer(
        self,
        layer_name_pattern: Union[str, List[str]],
        handle_func: Callable[[nn.Layer, str], nn.Layer],
    ) -> Dict[str, nn.Layer]:
        """use 'handle_func' to modify the sub-layer(s) specified by 'layer_name_pattern'.

        Args:
            layer_name_pattern (Union[str, List[str]]): The name of layer to be modified by 'handle_func'.
            handle_func (Callable[[nn.Layer, str], nn.Layer]): The function to modify target layer specified by 'layer_name_pattern'. The formal params are the layer(nn.Layer) and pattern(str) that is (a member of) layer_name_pattern (when layer_name_pattern is List type). And the return is the layer processed.

        Returns:
            Dict[str, nn.Layer]: The key is the pattern and corresponding value is the result returned by 'handle_func()'.

        Examples:

            from paddle import nn
            import paddleclas

            def rep_func(layer: nn.Layer, pattern: str):
                new_layer = nn.Conv2D(
                    in_channels=layer._in_channels,
                    out_channels=layer._out_channels,
                    kernel_size=5,
                    padding=2
                )
                return new_layer

            net = paddleclas.MobileNetV1()
            res = net.upgrade_sublayer(layer_name_pattern=["blocks[11].depthwise_conv.conv", "blocks[12].depthwise_conv.conv"], handle_func=rep_func)
            print(res)
            # {'blocks[11].depthwise_conv.conv': the corresponding new_layer, 'blocks[12].depthwise_conv.conv': the corresponding new_layer}
        """

        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        hit_layer_pattern_list = []
        for pattern in layer_name_pattern:
            # parse pattern to find target layer and its parent
            layer_list = parse_pattern_str(pattern=pattern, parent_layer=self)
            if not layer_list:
                continue

            sub_layer_parent = layer_list[-2]["layer"] if len(layer_list) > 1 else self
            sub_layer = layer_list[-1]["layer"]
            sub_layer_name = layer_list[-1]["name"]
            sub_layer_index_list = layer_list[-1]["index_list"]

            new_sub_layer = handle_func(sub_layer, pattern)

            if sub_layer_index_list:
                if len(sub_layer_index_list) > 1:
                    sub_layer_parent = getattr(sub_layer_parent, sub_layer_name)[
                        sub_layer_index_list[0]
                    ]
                    for sub_layer_index in sub_layer_index_list[1:-1]:
                        sub_layer_parent = sub_layer_parent[sub_layer_index]
                    sub_layer_parent[sub_layer_index_list[-1]] = new_sub_layer
                else:
                    getattr(sub_layer_parent, sub_layer_name)[
                        sub_layer_index_list[0]
                    ] = new_sub_layer
            else:
                setattr(sub_layer_parent, sub_layer_name, new_sub_layer)

            hit_layer_pattern_list.append(pattern)
        return hit_layer_pattern_list

    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.

        Args:
            stop_layer_name (str): The name of layer that stop forward and backward after this layer.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        layer_list = parse_pattern_str(stop_layer_name, self)
        if not layer_list:
            return False

        parent_layer = self
        for layer_dict in layer_list:
            name, index_list = layer_dict["name"], layer_dict["index_list"]
            if not set_identity(parent_layer, name, index_list):
                msg = f"Failed to set the layers that after stop_layer_name('{stop_layer_name}') to IdentityLayer. The error layer's name is '{name}'."
                return False
            parent_layer = layer_dict["layer"]

        return True

    def freeze_befor(self, layer_name: str) -> bool:
        """freeze the layer named layer_name and its previous layer.

        Args:
            layer_name (str): The name of layer that would be freezed.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        def stop_grad(layer, pattern):
            class StopGradLayer(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    x = self.layer(x)
                    x.stop_gradient = True
                    return x

            new_layer = StopGradLayer()
            return new_layer

        res = self.upgrade_sublayer(layer_name, stop_grad)
        if len(res) == 0:
            msg = "Failed to stop the gradient before the layer named '{layer_name}'"
            return False
        return True

    def update_res(self, return_patterns: Union[str, List[str]]) -> Dict[str, nn.Layer]:
        """update the result(s) to be returned.

        Args:
            return_patterns (Union[str, List[str]]): The name of layer to return output.

        Returns:
            Dict[str, nn.Layer]: The pattern(str) and corresponding layer(nn.Layer) that have been set successfully.
        """

        # clear res_dict that could have been set
        self.res_dict = {}

        class Handler(object):
            def __init__(self, res_dict):
                # res_dict is a reference
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                if hasattr(layer, "hook_remove_helper"):
                    layer.hook_remove_helper.remove()
                layer.hook_remove_helper = layer.register_forward_post_hook(
                    save_sub_res_hook
                )
                return layer

        handle_func = Handler(self.res_dict)

        hit_layer_pattern_list = self.upgrade_sublayer(
            return_patterns, handle_func=handle_func
        )

        if hasattr(self, "hook_remove_helper"):
            self.hook_remove_helper.remove()
        self.hook_remove_helper = self.register_forward_post_hook(
            self._return_dict_hook
        )

        return hit_layer_pattern_list


def save_sub_res_hook(layer, input, output):
    layer.res_dict[layer.res_name] = output


def set_identity(
    parent_layer: nn.Layer, layer_name: str, layer_index_list: str = None
) -> bool:
    """set the layer specified by layer_name and layer_index_list to Identity.

    Args:
        parent_layer (nn.Layer): The parent layer of target layer specified by layer_name and layer_index_list.
        layer_name (str): The name of target layer to be set to Identity.
        layer_index_list (str, optional): The index of target layer to be set to Identity in parent_layer. Defaults to None.

    Returns:
        bool: True if successfully, False otherwise.
    """

    stop_after = False
    for sub_layer_name in parent_layer._sub_layers:
        if stop_after:
            parent_layer._sub_layers[sub_layer_name] = Identity()
            continue
        if sub_layer_name == layer_name:
            stop_after = True

    if layer_index_list and stop_after:
        layer_container = parent_layer._sub_layers[layer_name]
        for num, layer_index in enumerate(layer_index_list):
            stop_after = False
            for i in range(num):
                layer_container = layer_container[layer_index_list[i]]
            for sub_layer_index in layer_container._sub_layers:
                if stop_after:
                    parent_layer._sub_layers[layer_name][sub_layer_index] = Identity()
                    continue
                if layer_index == sub_layer_index:
                    stop_after = True

    return stop_after


def parse_pattern_str(
    pattern: str, parent_layer: nn.Layer
) -> Union[None, List[Dict[str, Union[nn.Layer, str, None]]]]:
    """parse the string type pattern.

    Args:
        pattern (str): The pattern to describe layer.
        parent_layer (nn.Layer): The root layer relative to the pattern.

    Returns:
        Union[None, List[Dict[str, Union[nn.Layer, str, None]]]]: None if failed. If successfully, the members are layers parsed in order:
                                                                [
                                                                    {"layer": first layer, "name": first layer's name parsed, "index": first layer's index parsed if exist},
                                                                    {"layer": second layer, "name": second layer's name parsed, "index": second layer's index parsed if exist},
                                                                    ...
                                                                ]
    """

    pattern_list = pattern.split(".")
    if not pattern_list:
        msg = f"The pattern('{pattern}') is illegal. Please check and retry."
        return None

    layer_list = []
    while len(pattern_list) > 0:
        if "[" in pattern_list[0]:
            target_layer_name = pattern_list[0].split("[")[0]
            target_layer_index_list = list(
                index.split("]")[0] for index in pattern_list[0].split("[")[1:]
            )
        else:
            target_layer_name = pattern_list[0]
            target_layer_index_list = None

        target_layer = getattr(parent_layer, target_layer_name, None)

        if target_layer is None:
            msg = f"Not found layer named('{target_layer_name}') specified in pattern('{pattern}')."
            return None

        if target_layer_index_list:
            for target_layer_index in target_layer_index_list:
                if int(target_layer_index) < 0 or int(target_layer_index) >= len(
                    target_layer
                ):
                    msg = f"Not found layer by index('{target_layer_index}') specified in pattern('{pattern}'). The index should < {len(target_layer)} and > 0."
                    return None
                target_layer = target_layer[target_layer_index]

        layer_list.append(
            {
                "layer": target_layer,
                "name": target_layer_name,
                "index_list": target_layer_index_list,
            }
        )

        pattern_list = pattern_list[1:]
        parent_layer = target_layer

    return layer_list


class AdaptiveAvgPool2D(nn.AdaptiveAvgPool2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if paddle.device.get_device().startswith("npu"):
            self.device = "npu"
        else:
            self.device = None

        if isinstance(self._output_size, int) and self._output_size == 1:
            self._gap = True
        elif (
            isinstance(self._output_size, tuple)
            and self._output_size[0] == 1
            and self._output_size[1] == 1
        ):
            self._gap = True
        else:
            self._gap = False

    def forward(self, x):
        if self.device == "npu" and self._gap:
            # Global Average Pooling
            N, C, _, _ = x.shape
            x_mean = paddle.mean(x, axis=[2, 3])
            x_mean = paddle.reshape(x_mean, [N, C, 1, 1])
            return x_mean
        else:
            return F.adaptive_avg_pool2d(
                x,
                output_size=self._output_size,
                data_format=self._data_format,
                name=self._name,
            )


# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal, Constant
from paddle.nn import Conv2D, BatchNorm2D, ReLU, AdaptiveAvgPool2D, MaxPool2D
from paddle.regularizer import L2Decay
from paddle import ParamAttr

MODEL_URLS = {
    "PPHGNetV2_B0": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B0_ssld_pretrained.pdparams",
    "PPHGNetV2_B1": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B1_ssld_pretrained.pdparams",
    "PPHGNetV2_B2": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B2_ssld_pretrained.pdparams",
    "PPHGNetV2_B3": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B3_ssld_pretrained.pdparams",
    "PPHGNetV2_B4": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B4_ssld_pretrained.pdparams",
    "PPHGNetV2_B5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B5_ssld_pretrained.pdparams",
    "PPHGNetV2_B6": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B6_ssld_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

kaiming_normal_ = KaimingNormal()
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class LearnableAffineBlock(TheseusLayer):
    """
    Create a learnable affine block module. This module can significantly improve accuracy on smaller models.

    Args:
        scale_value (float): The initial value of the scale parameter, default is 1.0.
        bias_value (float): The initial value of the bias parameter, default is 0.0.
        lr_mult (float): The learning rate multiplier, default is 1.0.
        lab_lr (float): The learning rate, default is 0.01.
    """

    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0, lab_lr=0.01):
        super().__init__()
        self.scale = self.create_parameter(
            shape=[
                1,
            ],
            default_initializer=Constant(value=scale_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr),
        )
        self.add_parameter("scale", self.scale)
        self.bias = self.create_parameter(
            shape=[
                1,
            ],
            default_initializer=Constant(value=bias_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr),
        )
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(TheseusLayer):
    """
    ConvBNAct is a combination of convolution and batchnorm layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int/str): Padding or padding type for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        use_act: (bool): Whether to use activation function. Defaults to True.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        use_act=True,
        use_lab=False,
        lr_mult=1.0,
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding if isinstance(padding, str) else (kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=False,
        )
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
        )
        if self.use_act:
            self.act = ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock(lr_mult=lr_mult)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(TheseusLayer):
    """
    LightConvBNAct is a combination of pw and dw layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the depth-wise convolution kernel.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_lab=False,
        lr_mult=1.0,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(TheseusLayer):
    """
    StemBlock for PP-HGNetV2.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of middle channels.
        out_channels (int): Number of output channels.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.
    """

    def __init__(
        self, in_channels, mid_channels, out_channels, use_lab=False, lr_mult=1.0
    ):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.pool = nn.MaxPool2D(
            kernel_size=2, stride=1, ceil_mode=True, padding="SAME"
        )

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = paddle.concat([x1, x2], 1)
        x = self.stem3(x)
        x = self.stem4(x)

        return x


class HGV2_Block(TheseusLayer):
    """
    HGV2_Block, the basic unit that constitutes the HGV2_Stage.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of middle channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        layer_num (int): Number of layers in the HGV2 block. Defaults to 6.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int/str): Padding or padding type for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        use_act (bool): Whether to use activation function. Defaults to True.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        lr_mult (float): Learning rate multiplier for the layer. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size=3,
        layer_num=6,
        identity=False,
        light_block=True,
        use_lab=False,
        lr_mult=1.0,
    ):
        super().__init__()
        self.identity = identity

        self.layers = nn.LayerList()
        block_type = "LightConvBNAct" if light_block else "ConvBNAct"
        for i in range(layer_num):
            self.layers.append(
                eval(block_type)(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    stride=1,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                    lr_mult=lr_mult,
                )
            )
        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.aggregation_excitation_conv = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = paddle.concat(output, axis=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x += identity
        return x


class HGV2_Stage(TheseusLayer):
    """
    HGV2_Stage, the basic unit that constitutes the PPHGNetV2.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of middle channels.
        out_channels (int): Number of output channels.
        block_num (int): Number of blocks in the HGV2 stage.
        layer_num (int): Number of layers in the HGV2 block. Defaults to 6.
        is_downsample (bool): Whether to use downsampling operation. Defaults to False.
        light_block (bool): Whether to use light block. Defaults to True.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        use_lab (bool, optional): Whether to use the LAB operation. Defaults to False.
        lr_mult (float, optional): Learning rate multiplier for the layer. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        block_num,
        layer_num=6,
        is_downsample=True,
        light_block=True,
        kernel_size=3,
        use_lab=False,
        lr_mult=1.0,
    ):

        super().__init__()
        self.is_downsample = is_downsample
        if self.is_downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab,
                lr_mult=lr_mult,
            )

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HGV2_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=False if i == 0 else True,
                    light_block=light_block,
                    use_lab=use_lab,
                    lr_mult=lr_mult,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.is_downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPHGNetV2(TheseusLayer):
    """
    PPHGNetV2

    Args:
        stage_config (dict): Config for PPHGNetV2 stages. such as the number of channels, stride, etc.
        stem_channels: (list): Number of channels of the stem of the PPHGNetV2.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        use_last_conv (bool): Whether to use the last conv layer as the output channel. Defaults to True.
        class_expand (int): Number of channels for the last 1x1 convolutional layer.
        drop_prob (float): Dropout probability for the last 1x1 convolutional layer. Defaults to 0.0.
        class_num (int): The number of classes for the classification layer. Defaults to 1000.
        lr_mult_list (list): Learning rate multiplier for the stages. Defaults to [1.0, 1.0, 1.0, 1.0, 1.0].
    Returns:
        model: nn.Layer. Specific PPHGNetV2 model depends on args.
    """

    def __init__(
        self,
        stage_config,
        stem_channels=[3, 32, 64],
        use_lab=False,
        use_last_conv=True,
        class_expand=2048,
        dropout_prob=0.0,
        class_num=1000,
        lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        **kwargs,
    ):
        super().__init__()
        self.use_lab = use_lab
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        self.class_num = class_num

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
            use_lab=use_lab,
            lr_mult=lr_mult_list[0],
        )

        # stages
        self.stages = nn.LayerList()
        for i, k in enumerate(stage_config):
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                is_downsample,
                light_block,
                kernel_size,
                layer_num,
            ) = stage_config[k]
            self.stages.append(
                HGV2_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    is_downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    lr_mult=lr_mult_list[i + 1],
                )
            )

        self.avg_pool = AdaptiveAvgPool2D(1)

        if self.use_last_conv:
            self.last_conv = Conv2D(
                in_channels=out_channels,
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            )
            self.act = ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock()
            self.dropout = nn.Dropout(p=dropout_prob, mode="downscale_in_infer")

        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = nn.Linear(
            self.class_expand if self.use_last_conv else out_channels, self.class_num
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2D)):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


def PPHGNetV2_B0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B0
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B0` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [16, 16, 64, 1, False, False, 3, 3],
        "stage2": [64, 32, 256, 1, True, False, 3, 3],
        "stage3": [256, 64, 512, 2, True, True, 5, 3],
        "stage4": [512, 128, 1024, 1, True, True, 5, 3],
    }

    model = PPHGNetV2(
        stem_channels=[3, 16, 16], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B1(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B1
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B1` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 64, 1, False, False, 3, 3],
        "stage2": [64, 48, 256, 1, True, False, 3, 3],
        "stage3": [256, 96, 512, 2, True, True, 5, 3],
        "stage4": [512, 192, 1024, 1, True, True, 5, 3],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B2(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B2
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B2` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 96, 1, False, False, 3, 4],
        "stage2": [96, 64, 384, 1, True, False, 3, 4],
        "stage3": [384, 128, 768, 3, True, True, 5, 4],
        "stage4": [768, 256, 1536, 1, True, True, 5, 4],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B3(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B3
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B3` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [32, 32, 128, 1, False, False, 3, 5],
        "stage2": [128, 64, 512, 1, True, False, 3, 5],
        "stage3": [512, 128, 1024, 3, True, True, 5, 5],
        "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
    }

    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B5
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B5` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [64, 64, 128, 1, False, False, 3, 6],
        "stage2": [128, 128, 512, 2, True, False, 3, 6],
        "stage3": [512, 256, 1024, 5, True, True, 5, 6],
        "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
    }

    model = PPHGNetV2(
        stem_channels=[3, 32, 64], stage_config=stage_config, use_lab=False, **kwargs
    )
    return model


def PPHGNetV2_B6(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2_B6
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B6` model depends on args.
    """
    stage_config = {
        # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
        "stage1": [96, 96, 192, 2, False, False, 3, 6],
        "stage2": [192, 192, 512, 3, True, False, 3, 6],
        "stage3": [512, 384, 1024, 6, True, True, 5, 6],
        "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
    }

    model = PPHGNetV2(
        stem_channels=[3, 48, 96], stage_config=stage_config, use_lab=False, **kwargs
    )
    return model


class PPHGNetV2_B4(nn.Layer):
    """
    PPHGNetV2_B4
    Args:
        in_channels (int): Number of input channels. Default is 3 (for RGB images).
        class_num (int): Number of classes for classification. Default is 1000.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2_B4` model with defined architecture.
    """

    def __init__(self, in_channels=3, class_num=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2048
        stage_config = {
            # in_channels, mid_channels, out_channels, num_blocks, is_downsample, light_block, kernel_size, layer_num
            "stage1": [48, 48, 128, 1, False, False, 3, 6],
            "stage2": [128, 96, 512, 1, True, False, 3, 6],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
        }

        self.pphgnet_b4 = PPHGNetV2(
            stem_channels=[3, 32, 48],
            stage_config=stage_config,
            class_num=class_num,
            use_lab=False,
        )

    def forward(self, input_data):
        if self.training:
            pixel_values, label, attention_mask = input_data
        else:
            if isinstance(input_data, list):
                pixel_values = input_data[0]
            else:
                pixel_values = input_data
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = paddle.repeat_interleave(pixel_values, repeats=3, axis=1)
        pphgnet_b4_output = self.pphgnet_b4(pixel_values)
        b, c, h, w = pphgnet_b4_output.shape
        pphgnet_b4_output = pphgnet_b4_output.reshape([b, c, h * w]).transpose(
            [0, 2, 1]
        )
        pphgnet_b4_output = DonutSwinModelOutput(
            last_hidden_state=pphgnet_b4_output,
            pooler_output=None,
            hidden_states=None,
            attentions=False,
            reshaped_hidden_states=None,
        )
        if self.training:
            return pphgnet_b4_output, label, attention_mask
        else:
            return pphgnet_b4_output
