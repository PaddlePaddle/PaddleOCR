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

import math
import numpy as np

import paddle
import paddle.nn.quant.quant_layers as quant_layers

from ..quantization_pass import _get_op_input_var_names
from ..quantization_pass import _get_op_output_var_names
from ..quantization_pass import _get_output_name_index
from ..quantization_pass import _get_input_name_index

layer_name_map = {
    'Conv2DTranspose': paddle.nn.Conv2DTranspose,
    'Conv2D': paddle.nn.Conv2D,
    'Linear': paddle.nn.Linear,
    'AdaptiveAvgPool2D': paddle.nn.AdaptiveAvgPool2D,
    'AdaptiveMaxPool2D': paddle.nn.AdaptiveMaxPool2D,
    'AvgPool2D': paddle.nn.AvgPool2D,
    'MaxPool2D': paddle.nn.MaxPool2D,
    'Hardswish': paddle.nn.Hardswish,
    'LeakyReLU': paddle.nn.LeakyReLU,
    'PReLU': paddle.nn.PReLU,
    'ReLU': paddle.nn.ReLU,
    'ReLU6': paddle.nn.ReLU6,
    'Sigmoid': paddle.nn.Sigmoid,
    'Softmax': paddle.nn.Softmax,
    'Swish': paddle.nn.Swish,
    'Tanh': paddle.nn.Tanh,
    'Hardswish': paddle.nn.Hardswish,
    'BatchNorm': paddle.nn.BatchNorm,
    'GroupNorm': paddle.nn.GroupNorm,
    'LayerNorm': paddle.nn.LayerNorm,
}

# Apply fake quant for the inputs of these layers
fake_quant_input_layers = [
    paddle.nn.Conv2D, paddle.nn.Linear, paddle.nn.Conv2DTranspose
]

# Apply fake quant for the output of these layers
# TODO(jc): fix the problem of adding duplicate fake_quant ops
# paddle.nn.AdaptiveAvgPool2D, paddle.nn.AvgPool2D, paddle.nn.ReLU,paddle.nn.LeakyReLU
fake_quant_output_layers = [
    paddle.nn.quant.add, paddle.nn.quant.subtract, paddle.nn.quant.multiply,
    paddle.nn.quant.divide
]

fake_quant_leaf_layers = [
    quant_layers.FakeQuantAbsMax,
    quant_layers.FakeQuantChannelWiseAbsMax,
    quant_layers.FakeQuantMovingAverageAbsMax,
    quant_layers.MovingAverageAbsMaxScale,
]

fake_quant_wrap_layers = [
    quant_layers.QuantizedConv2D, quant_layers.QuantizedLinear,
    quant_layers.QuantizedConv2DTranspose
]

# The weight format of these layers is Cin * Cout * H * W 
spec_channel_axis_layers = [paddle.nn.Conv2DTranspose, paddle.nn.Linear]

weight_op_types = [
    "conv2d", "depthwise_conv2d", "matmul", "conv2d_transpose",
    "depthwise_conv2d_transpose"
]

fake_quantize_dequantize_op_types = [
    "fake_quantize_dequantize_abs_max",
    "fake_channel_wise_quantize_dequantize_abs_max",
    "fake_quantize_dequantize_moving_average_abs_max"
]


def load_variable_data(scope, var_name):
    """
    Load variable value from scope
    """
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Can not find " + var_name + " in the scope."
    return np.array(var_node.get_tensor())


def find_previous_op(block, var_name):
    """
    Find the previous op for the input variable.
    """
    for op in block.ops:
        if var_name in op.output_arg_names:
            return op
    return None


def find_next_ops(block, var_name):
    """
    Find all followed ops for the input variable.
    """
    res_ops = []
    for op in block.ops:
        if var_name in op.input_arg_names:
            res_ops.append(op)
    return res_ops


def find_parent_layer_and_sub_name(model, name):
    """
    Given the model and the name of a layer, find the parent layer and
    the sub_name of the layer.
    For example, if name is 'block_1/convbn_1/conv_1', the parent layer is
    'block_1/convbn_1' and the sub_name is `conv_1`.
    Args:
        model(paddle.nn.Layer): the model to be quantized.
        name(string): the name of a layer

    Returns:
        parent_layer, subname
    """
    assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."
    assert len(name) > 0, "The input (name) should not be empty."

    last_idx = 0
    idx = 0
    parent_layer = model
    while idx < len(name):
        if name[idx] == '.':
            sub_name = name[last_idx:idx]
            if hasattr(parent_layer, sub_name):
                parent_layer = getattr(parent_layer, sub_name)
                last_idx = idx + 1
        idx += 1
    sub_name = name[last_idx:idx]
    return parent_layer, sub_name


def program_all_ops(program):
    """
    Return all ops for the input program.
    """
    all_ops = []
    for block in program.blocks:
        for op in block.ops:
            all_ops.append(op)
    return all_ops


def is_leaf_layer(layer):
    """
    Whether the layer is leaf layer.
    """
    return isinstance(layer, paddle.nn.Layer) \
        and len(layer.sublayers()) == 0


def fp_numpy_to_naive(x_np):
    """
    Convert numpy to float or list.
    """
    if x_np.size == 1:
        return float(x_np)
    else:
        return x_np.tolist()
