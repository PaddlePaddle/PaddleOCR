# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn as nn
import paddle.nn.functional as F


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
            weight_mat = weight_mat.transpose([
                self.dim,
                * [d for d in range(weight_mat.dim()) if d != self.dim]
            ])

        height = weight_mat.shape[0]

        return weight_mat.reshape([height, -1])

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
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

    def remove(self, module):
        with paddle.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')

        module.add_parameter(self.name, weight.detach())

    def __call__(self, module, inputs):
        setattr(
            module,
            self.name,
            self.compute_weight(
                module, do_power_iteration=module.training))

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with paddle.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.shape

            # randomly initialize u and v
            u = module.create_parameter([h])
            u = normal_(u, 0., 1.)
            v = module.create_parameter([w])
            v = normal_(v, 0., 1.)
            u = F.normalize(u, axis=0, epsilon=fn.eps)
            v = F.normalize(v, axis=0, epsilon=fn.eps)

        # delete fn.name form parameters, otherwise you can not set attribute
        del module._parameters[fn.name]
        module.add_parameter(fn.name + "_orig", weight)
        # still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an Parameter and
        # gets added as a parameter. Instead, we register weight * 1.0 as a plain
        # attribute.
        setattr(module, fn.name, weight * 1.0)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module,
                  name='weight',
                  n_power_iterations=1,
                  eps=1e-12,
                  dim=None):

    if dim is None:
        if isinstance(module, (nn.Conv1DTranspose, nn.Conv2DTranspose,
                               nn.Conv3DTranspose, nn.Linear)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module
