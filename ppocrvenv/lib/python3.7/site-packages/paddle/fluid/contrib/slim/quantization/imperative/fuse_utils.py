#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import paddle
import paddle.nn as nn
from . import utils


class Identity(nn.Layer):
    '''a layer to replace bn or relu layers'''

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def fuse_layers(model, layers_to_fuse, inplace=False):
    '''
       fuse layers in layers_to_fuse

       Args:
           model(paddle.nn.Layer): The model to be fused.
           layers_to_fuse(list): The layers' names to be fused. For
               example,"fuse_list = [["conv1", "bn1"], ["conv2", "bn2"]]".
               A TypeError would be raised if "fuse" was set as
               True but "fuse_list" was None.
                                 Default: None.
           inplace(bool): Whether apply fusing to the input model.
                          Default: False.

       Return
           fused_model(paddle.nn.Layer): The fused model.
    '''
    if inplace == False:
        model = copy.deepcopy(model)
    for layers in layers_to_fuse:
        _fuse_layers(model, layers)
    return model


def _fuse_layers(model, layers_list):
    '''fuse all the layers in layers_list'''
    layer_list = []
    for layer_name in layers_list:
        parent_layer, sub_name = utils.find_parent_layer_and_sub_name(
            model, layer_name)
        layer_list.append(getattr(parent_layer, sub_name))
    new_layers = _fuse_func(layer_list)
    for i, item in enumerate(layers_list):
        parent_layer, sub_name = utils.find_parent_layer_and_sub_name(model,
                                                                      item)
        setattr(parent_layer, sub_name, new_layers[i])


def _fuse_func(layer_list):
    '''choose the fuser method and fuse layers'''
    types = tuple(type(m) for m in layer_list)
    fusion_method = types_to_fusion_method.get(types, None)
    new_layers = [None] * len(layer_list)
    fused_layer = fusion_method(*layer_list)
    for handle_id, pre_hook_fn in layer_list[0]._forward_pre_hooks.items():
        fused_layer.register_forward_pre_hook(pre_hook_fn)
        del layer_list[0]._forward_pre_hooks[handle_id]
    for handle_id, hook_fn in layer_list[-1]._forward_post_hooks.items():
        fused_layer.register_forward_post_hook(hook_fn)
        del layer_list[-1]._forward_post_hooks[handle_id]
    new_layers[0] = fused_layer
    for i in range(1, len(layer_list)):
        identity = Identity()
        identity.training = layer_list[0].training
        new_layers[i] = identity
    return new_layers


def _fuse_conv_bn(conv, bn):
    '''fuse conv and bn for train or eval'''
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."
    if conv.training:
        assert bn._num_features == conv._out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        raise NotImplementedError
    else:
        return _fuse_conv_bn_eval(conv, bn)


def _fuse_conv_bn_eval(conv, bn):
    '''fuse conv and bn for eval'''
    assert (not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_weight, fused_bias = _fuse_conv_bn_weights(
        fused_conv.weight, fused_conv.bias, bn._mean, bn._variance, bn._epsilon,
        bn.weight, bn.bias)
    fused_conv.weight.set_value(fused_weight)
    if fused_conv.bias is None:
        fused_conv.bias = paddle.create_parameter(
            shape=[fused_conv._out_channels], is_bias=True, dtype=bn.bias.dtype)
    fused_conv.bias.set_value(fused_bias)
    return fused_conv


def _fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    '''fuse weights and bias of conv and bn'''
    if conv_b is None:
        conv_b = paddle.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = paddle.ones_like(bn_rm)
    if bn_b is None:
        bn_b = paddle.zeros_like(bn_rm)
    bn_var_rsqrt = paddle.rsqrt(bn_rv + bn_eps)
    conv_w = conv_w * \
        (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return conv_w, conv_b


def _fuse_linear_bn(linear, bn):
    '''fuse linear and bn'''
    assert (linear.training == bn.training),\
        "Linear and BN both must be in the same mode (train or eval)."
    if linear.training:
        assert bn._num_features == linear.weight.shape[
            1], 'Output channel of Linear must match num_features of BatchNorm'
        raise NotImplementedError
    else:
        return _fuse_linear_bn_eval(linear, bn)


def _fuse_linear_bn_eval(linear, bn):
    '''fuse linear and bn for eval'''
    assert (not (linear.training or bn.training)), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)

    fused_weight, fused_bias = _fuse_linear_bn_weights(
        fused_linear.weight, fused_linear.bias, bn._mean, bn._variance,
        bn._epsilon, bn.weight, bn.bias)
    fused_linear.weight.set_value(fused_weight)
    if fused_linear.bias is None:
        fused_linear.bias = paddle.create_parameter(
            shape=[fused_linear.weight.shape[1]],
            is_bias=True,
            dtype=bn.bias.dtype)
    fused_linear.bias.set_value(fused_bias)
    return fused_linear


def _fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w,
                            bn_b):
    '''fuse weights and bias of linear and bn'''
    if linear_b is None:
        linear_b = paddle.zeros_like(bn_rm)
    bn_scale = bn_w * paddle.rsqrt(bn_rv + bn_eps)
    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b
    return fused_w, fused_b


types_to_fusion_method = {
    (nn.Conv2D, nn.BatchNorm2D): _fuse_conv_bn,
    (nn.Linear, nn.BatchNorm1D): _fuse_linear_bn,
}
