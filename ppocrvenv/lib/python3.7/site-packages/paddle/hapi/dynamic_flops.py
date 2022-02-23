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
import warnings
import paddle.nn as nn
import numpy as np
from .static_flops import static_flops, Table
from paddle.fluid.dygraph.dygraph_to_static.program_translator import unwrap_decorators

__all__ = []


def flops(net, input_size, custom_ops=None, print_detail=False):
    """Print a table about the FLOPs of network.

    Args:
        net (paddle.nn.Layer||paddle.static.Program): The network which could be a instance of paddle.nn.Layer in 
                    dygraph or paddle.static.Program in static graph.
        input_size (list): size of input tensor. Note that the batch_size in argument 'input_size' only support 1.
        custom_ops (A dict of function, optional): A dictionary which key is the class of specific operation such as 
                    paddle.nn.Conv2D and the value is the function used to count the FLOPs of this operation. This 
                    argument only work when argument 'net' is an instance of paddle.nn.Layer. The details could be found
                    in following example code. Default is None.
        print_detail (bool, optional): Whether to print the detail information, like FLOPs per layer, about the net FLOPs.
                    Default is False.

    Returns:
        Int: A number about the FLOPs of total network.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            class LeNet(nn.Layer):
                def __init__(self, num_classes=10):
                    super(LeNet, self).__init__()
                    self.num_classes = num_classes
                    self.features = nn.Sequential(
                        nn.Conv2D(
                            1, 6, 3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2D(2, 2),
                        nn.Conv2D(
                            6, 16, 5, stride=1, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2D(2, 2))

                    if num_classes > 0:
                        self.fc = nn.Sequential(
                            nn.Linear(400, 120),
                            nn.Linear(120, 84),
                            nn.Linear(
                                84, 10))

                def forward(self, inputs):
                    x = self.features(inputs)

                    if self.num_classes > 0:
                        x = paddle.flatten(x, 1)
                        x = self.fc(x)
                    return x

            lenet = LeNet()
            # m is the instance of nn.Layer, x is the intput of layer, y is the output of layer.
            def count_leaky_relu(m, x, y):
                x = x[0]
                nelements = x.numel()
                m.total_ops += int(nelements)

            FLOPs = paddle.flops(lenet, [1, 1, 28, 28], custom_ops= {nn.LeakyReLU: count_leaky_relu},
                                print_detail=True)
            print(FLOPs)

            #+--------------+-----------------+-----------------+--------+--------+
            #|  Layer Name  |   Input Shape   |   Output Shape  | Params | Flops  |
            #+--------------+-----------------+-----------------+--------+--------+
            #|   conv2d_2   |  [1, 1, 28, 28] |  [1, 6, 28, 28] |   60   | 47040  |
            #|   re_lu_2    |  [1, 6, 28, 28] |  [1, 6, 28, 28] |   0    |   0    |
            #| max_pool2d_2 |  [1, 6, 28, 28] |  [1, 6, 14, 14] |   0    |   0    |
            #|   conv2d_3   |  [1, 6, 14, 14] | [1, 16, 10, 10] |  2416  | 241600 |
            #|   re_lu_3    | [1, 16, 10, 10] | [1, 16, 10, 10] |   0    |   0    |
            #| max_pool2d_3 | [1, 16, 10, 10] |  [1, 16, 5, 5]  |   0    |   0    |
            #|   linear_0   |     [1, 400]    |     [1, 120]    | 48120  | 48000  |
            #|   linear_1   |     [1, 120]    |     [1, 84]     | 10164  | 10080  |
            #|   linear_2   |     [1, 84]     |     [1, 10]     |  850   |  840   |
            #+--------------+-----------------+-----------------+--------+--------+
            #Total Flops: 347560     Total Params: 61610
    """
    if isinstance(net, nn.Layer):
        # If net is a dy2stat model, net.forward is StaticFunction instance,
        # we set net.forward to original forward function.
        _, net.forward = unwrap_decorators(net.forward)

        inputs = paddle.randn(input_size)
        return dynamic_flops(
            net,
            inputs=inputs,
            custom_ops=custom_ops,
            print_detail=print_detail)
    elif isinstance(net, paddle.static.Program):
        return static_flops(net, print_detail=print_detail)
    else:
        warnings.warn(
            "Your model must be an instance of paddle.nn.Layer or paddle.static.Program."
        )
        return -1


def count_convNd(m, x, y):
    x = x[0]
    kernel_ops = np.product(m.weight.shape[2:])
    bias_ops = 1 if m.bias is not None else 0
    total_ops = int(y.numel()) * (
        x.shape[1] / m._groups * kernel_ops + bias_ops)
    m.total_ops += abs(int(total_ops))


def count_leaky_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    m.total_ops += int(nelements)


def count_bn(m, x, y):
    x = x[0]
    nelements = x.numel()
    if not m.training:
        total_ops = 2 * nelements
    m.total_ops += abs(int(total_ops))


def count_linear(m, x, y):
    total_mul = m.weight.shape[0]
    num_elements = y.numel()
    total_ops = total_mul * num_elements
    m.total_ops += abs(int(total_ops))


def count_avgpool(m, x, y):
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += int(total_ops)


def count_adap_avgpool(m, x, y):
    kernel = np.array(x[0].shape[2:]) // np.array(y.shape[2:])
    total_add = np.product(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops += abs(int(total_ops))


def count_zero_ops(m, x, y):
    m.total_ops += int(0)


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += p.numel()
    m.total_params[0] = abs(int(total_params))


def count_io_info(m, x, y):
    m.register_buffer('input_shape', paddle.to_tensor(x[0].shape))
    m.register_buffer('output_shape', paddle.to_tensor(y.shape))


register_hooks = {
    nn.Conv1D: count_convNd,
    nn.Conv2D: count_convNd,
    nn.Conv3D: count_convNd,
    nn.Conv1DTranspose: count_convNd,
    nn.Conv2DTranspose: count_convNd,
    nn.Conv3DTranspose: count_convNd,
    nn.layer.norm.BatchNorm2D: count_bn,
    nn.BatchNorm: count_bn,
    nn.ReLU: count_zero_ops,
    nn.ReLU6: count_zero_ops,
    nn.LeakyReLU: count_leaky_relu,
    nn.Linear: count_linear,
    nn.Dropout: count_zero_ops,
    nn.AvgPool1D: count_avgpool,
    nn.AvgPool2D: count_avgpool,
    nn.AvgPool3D: count_avgpool,
    nn.AdaptiveAvgPool1D: count_adap_avgpool,
    nn.AdaptiveAvgPool2D: count_adap_avgpool,
    nn.AdaptiveAvgPool3D: count_adap_avgpool
}


def dynamic_flops(model, inputs, custom_ops=None, print_detail=False):
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', paddle.zeros([1], dtype='int64'))
        m.register_buffer('total_params', paddle.zeros([1], dtype='int64'))
        m_type = type(m)

        flops_fn = None
        if m_type in custom_ops:
            flops_fn = custom_ops[m_type]
            if m_type not in types_collection:
                print("Customize Function has been applied to {}".format(
                    m_type))
        elif m_type in register_hooks:
            flops_fn = register_hooks[m_type]
            if m_type not in types_collection:
                print("{}'s flops has been counted".format(m_type))
        else:
            if m_type not in types_collection:
                print(
                    "Cannot find suitable count function for {}. Treat it as zero FLOPs.".
                    format(m_type))

        if flops_fn is not None:
            flops_handler = m.register_forward_post_hook(flops_fn)
            handler_collection.append(flops_handler)
        params_handler = m.register_forward_post_hook(count_parameters)
        io_handler = m.register_forward_post_hook(count_io_info)
        handler_collection.append(params_handler)
        handler_collection.append(io_handler)
        types_collection.add(m_type)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with paddle.framework.no_grad():
        model(inputs)

    total_ops = 0
    total_params = 0
    for m in model.sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['total_ops', 'total_params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            total_ops += m.total_ops
            total_params += m.total_params

    if training:
        model.train()
    for handler in handler_collection:
        handler.remove()

    table = Table(
        ["Layer Name", "Input Shape", "Output Shape", "Params", "Flops"])

    for n, m in model.named_sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['total_ops', 'total_params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            table.add_row([
                m.full_name(), list(m.input_shape.numpy()),
                list(m.output_shape.numpy()), int(m.total_params),
                int(m.total_ops)
            ])
            m._buffers.pop("total_ops")
            m._buffers.pop("total_params")
            m._buffers.pop('input_shape')
            m._buffers.pop('output_shape')
    if print_detail:
        table.print_table()
    print('Total Flops: {}     Total Params: {}'.format(
        int(total_ops), int(total_params)))
    return int(total_ops)
