# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from ..layer.conv import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from ..layer.common import Linear
from .. import functional as F

__all__ = []


def normal_(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)
    x.set_value(temp_value)
    return x


class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(
                                 n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # transpose dim to front
            weight_mat = weight_mat.transpose([self.dim] + [
                d for d in range(weight_mat.dim()) if d != self.dim
            ])

        height = weight_mat.shape[0]

        return weight_mat.reshape([height, -1])

    def compute_weight(self, layer, do_power_iteration):
        weight = getattr(layer, self.name + '_orig')
        u = getattr(layer, self.name + '_u')
        v = getattr(layer, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with paddle.no_grad():
                for _ in range(self.n_power_iterations):
                    v.set_value(
                        F.normalize(
                            paddle.matmul(
                                weight_mat,
                                u,
                                transpose_x=True,
                                transpose_y=False),
                            axis=0,
                            epsilon=self.eps, ))

                    u.set_value(
                        F.normalize(
                            paddle.matmul(weight_mat, v),
                            axis=0,
                            epsilon=self.eps, ))
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = paddle.dot(u, paddle.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def __call__(self, layer, inputs):
        setattr(
            layer,
            self.name,
            self.compute_weight(
                layer, do_power_iteration=layer.training))

    @staticmethod
    def apply(layer, name, n_power_iterations, dim, eps):
        for k, hook in layer._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = layer._parameters[name]

        with paddle.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.shape

            # randomly initialize u and v
            u = layer.create_parameter([h])
            u = normal_(u, 0., 1.)
            v = layer.create_parameter([w])
            v = normal_(v, 0., 1.)
            u = F.normalize(u, axis=0, epsilon=fn.eps)
            v = F.normalize(v, axis=0, epsilon=fn.eps)

        # delete fn.name form parameters, otherwise you can not set attribute
        del layer._parameters[fn.name]
        layer.add_parameter(fn.name + "_orig", weight)
        # still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an Parameter and
        # gets added as a parameter. Instead, we register weight * 1.0 as a plain
        # attribute.
        setattr(layer, fn.name, weight * 1.0)
        layer.register_buffer(fn.name + "_u", u)
        layer.register_buffer(fn.name + "_v", v)
        layer.register_forward_pre_hook(fn)
        return fn


def spectral_norm(layer,
                  name='weight',
                  n_power_iterations=1,
                  eps=1e-12,
                  dim=None):
    r"""
    This spectral_norm layer applies spectral normalization to a parameter according to the 
    following Calculation:

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`n_power_iterations` should be a positive integer, do following
    calculations with U and V for :attr:`power_iters` rounds.

    .. math::

        \mathbf{v} := \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \frac{\mathbf{W} \mathbf{v}}{\|\mathbf{W} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}


    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Parameters:
        layer(Layer): Layer of paddle, which has weight.
        name(str, optional): Name of the weight parameter. Default: 'weight'.
        n_power_iterations(int, optional): The number of power iterations to calculate spectral norm. Default: 1.
        eps(float, optional): The epsilon for numerical stability in calculating norms. Default: 1e-12.
        dim(int, optional): The index of dimension which should be permuted to the first before reshaping Input(Weight) to matrix, it should be set as 0 if Input(Weight) is the weight of fc layer, and should be set as 1 if Input(Weight) is the weight of conv layer. Default: None.
        
    Returns:
        The original layer with the spectral norm hook

    Examples:
       .. code-block:: python

            from paddle.nn import Conv2D
            from paddle.nn.utils import Spectralnorm

            conv = Conv2D(3, 1, 3)
            sn_conv = spectral_norm(conv)
            print(sn_conv)
            # Conv2D(3, 1, kernel_size=[3, 3], data_format=NCHW)
            print(sn_conv.weight)
            # Tensor(shape=[1, 3, 3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #        [[[[-0.21090528,  0.18563725, -0.14127982],
            #           [-0.02310637,  0.03197737,  0.34353802],
            #           [-0.17117859,  0.33152047, -0.28408015]],
            # 
            #          [[-0.13336606, -0.01862637,  0.06959272],
            #           [-0.02236020, -0.27091628, -0.24532901],
            #           [ 0.27254242,  0.15516677,  0.09036587]],
            # 
            #          [[ 0.30169338, -0.28146112, -0.11768346],
            #           [-0.45765871, -0.12504843, -0.17482486],
            #           [-0.36866254, -0.19969313,  0.08783543]]]])

    """

    if dim is None:
        if isinstance(layer, (Conv1DTranspose, Conv2DTranspose, Conv3DTranspose,
                              Linear)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(layer, name, n_power_iterations, dim, eps)
    return layer
