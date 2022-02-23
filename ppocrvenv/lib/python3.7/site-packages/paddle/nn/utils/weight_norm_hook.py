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

import numpy as np
from ... import fluid
from ...fluid import dygraph
from ...fluid import layers as F
from ...fluid.layer_helper import LayerHelper
from ...fluid.data_feeder import check_variable_and_dtype

__all__ = []


def l2_norm(x, axis, epsilon=1e-12, name=None):
    if len(x.shape) == 1:
        axis = 0
    check_variable_and_dtype(x, "X", ("float32", "float64"), "norm")

    helper = LayerHelper("l2_normalize", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    norm = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="norm",
        inputs={"X": x},
        outputs={"Out": out,
                 "Norm": norm},
        attrs={
            "axis": 1 if axis is None else axis,
            "epsilon": epsilon,
        })
    return F.squeeze(norm, axes=[axis])


def norm_except_dim(p, dim):
    shape = p.shape
    ndims = len(shape)
    if dim == -1:
        return F.sqrt(F.reduce_sum(F.square(p)) + 1e-12)
    elif dim == 0:
        p_matrix = F.reshape(p, (shape[0], -1))
        return l2_norm(p_matrix, axis=1)
    elif dim == ndims - 1:
        p_matrix = F.reshape(p, (-1, shape[-1]))
        return l2_norm(p_matrix, axis=0)
    else:
        perm = list(range(ndims))
        perm[0] = dim
        perm[dim] = 0
        p_transposed = F.transpose(p, perm)
        return norm_except_dim(p_transposed, 0)


def _weight_norm(v, g, dim):
    shape = v.shape
    ndims = len(shape)

    if dim == -1:
        v_normalized = v / (F.sqrt(F.reduce_sum(F.square(v))) + 1e-12)
    elif dim == 0:
        p_matrix = F.reshape(v, (shape[0], -1))
        v_normalized = F.l2_normalize(p_matrix, axis=1)
        v_normalized = F.reshape(v_normalized, shape)
    elif dim == ndims - 1:
        p_matrix = F.reshape(v, (-1, shape[-1]))
        v_normalized = F.l2_normalize(p_matrix, axis=0)
        v_normalized = F.reshape(v_normalized, shape)
    else:
        perm = list(range(ndims))
        perm[0] = dim
        perm[dim] = 0
        p_transposed = F.transpose(v, perm)
        transposed_shape = p_transposed.shape
        p_matrix = F.reshape(p_transposed, (p_transposed.shape[0], -1))
        v_normalized = F.l2_normalize(p_matrix, axis=1)
        v_normalized = F.reshape(v_normalized, transposed_shape)
        v_normalized = F.transpose(v_normalized, perm)
    weight = F.elementwise_mul(
        v_normalized, g, axis=dim if dim is not None else -1)
    return weight


class WeightNorm(object):
    def __init__(self, name, dim):
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, layer):
        g = getattr(layer, self.name + '_g')
        v = getattr(layer, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(layer, name, dim):
        for k, hook in layer._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        # support dim is negative numeber, (dim = -1) == (dim = None)
        weight_dim = len(layer._parameters[name].shape)
        assert (
            dim < weight_dim and dim >= -1 * weight_dim
        ), "dim must set between [-R, R), R means the dimension of weight."
        if dim != -1:
            dim = (dim + weight_dim) % weight_dim

        fn = WeightNorm(name, dim)

        w = getattr(layer, name)
        del layer._parameters[name]

        g_var = norm_except_dim(w, dim)
        v = layer.create_parameter(w.shape, dtype=w.dtype)
        layer.add_parameter(name + "_v", v)
        g = layer.create_parameter(g_var.shape, dtype=g_var.dtype)
        layer.add_parameter(name + '_g', g)
        with dygraph.no_grad():
            F.assign(w, v)
            F.assign(g_var, g)
        setattr(layer, name, fn.compute_weight(layer))

        layer.register_forward_pre_hook(fn)
        return fn

    def remove(self, layer):
        w_var = self.compute_weight(layer)
        delattr(layer, self.name)
        del layer._parameters[self.name + '_g']
        del layer._parameters[self.name + '_v']
        w = layer.create_parameter(w_var.shape, dtype=w_var.dtype)
        layer.add_parameter(self.name, w)
        with dygraph.no_grad():
            F.assign(w_var, w)

    def __call__(self, layer, inputs):
        setattr(layer, self.name, self.compute_weight(layer))


def weight_norm(layer, name='weight', dim=0):
    r"""
    This weight_norm layer applies weight normalization to a parameter according to the 
    following formula:

    .. math::

        \mathbf{w} = g \dfrac{v}{\|v\|}

    Weight normalization is a reparameterization of the weight vectors in a neural network that 
    decouples the magnitude of those weight vectors from their direction. Weight normalization 
    replaces the parameter specified by `name`(eg: 'weight') with two parameters: one parameter 
    specifying the magnitude (eg: 'weight_g') and one parameter specifying the direction 
    (eg: 'weight_v'). Weight normalization has been implemented as discussed in this paper: 
    `Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    <https://arxiv.org/pdf/1602.07868.pdf>`_.

    Parameters:
        layer(Layer): Layer of paddle, which has weight.
        name(str, optional): Name of the weight parameter. Default: 'weight'.
        dim(int, optional): Dimension over which to compute the norm. Dim is a non-negative number 
              which is less than the rank of weight Tensor. For Example, dim can be chosen from 0, 
              1, 2, 3 for convolution whose weight shape is [cout, cin, kh, kw] and rank is 4. 
              If dim is set to None, meaning that all elements will be normalized. Default: 0.
    
    Returns:
        Origin layer with weight norm hook.

    Examples:
        .. code-block:: python

          import numpy as np
          from paddle.nn import Conv2D
          from paddle.nn.utils import weight_norm

          x = np.array([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')
          conv = Conv2D(3, 5, 3)
          wn = weight_norm(conv)
          print(conv.weight_g.shape)
          # [5]
          print(conv.weight_v.shape)
          # [5, 3, 3, 3]
    """
    WeightNorm.apply(layer, name, dim)
    return layer


def remove_weight_norm(layer, name='weight'):
    """
    remove weight normalization from layer.

    Parameters:
        layer(Layer): Layer of paddle, which has weight.
        name(str, optional): Name of the weight parameter. Default: 'weight'.

    Returns:
        Origin layer without weight norm

    Examples:
        .. code-block:: python
          
          import paddle
          from paddle.nn import Conv2D
          from paddle.nn.utils import weight_norm, remove_weight_norm

          conv = Conv2D(3, 5, 3)
          wn = weight_norm(conv)
          remove_weight_norm(conv)
          print(conv.weight_g)
          # AttributeError: 'Conv2D' object has no attribute 'weight_g'
    """
    for k, hook in layer._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(layer)
            del layer._forward_pre_hooks[k]
            return layer

    raise ValueError("weight_norm of '{}' not found in {}".format(name, layer))
