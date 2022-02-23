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

import numpy as np

import paddle
from .. import Layer
from ...fluid.framework import core, in_dygraph_mode
from ...fluid.data_feeder import check_variable_and_dtype, check_type
from ...fluid.layer_helper import LayerHelper
from paddle import _C_ops

__all__ = []


class PairwiseDistance(Layer):
    r"""
    This operator computes the pairwise distance between two vectors. The
    distance is calculated by p-oreder norm:

    .. math::

        \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right) ^ {1/p}.

    Parameters:
        p (float): The order of norm. The default value is 2.
        epsilon (float, optional): Add small value to avoid division by zero,
            default value is 1e-6.
        keepdim (bool, optional): Whether to reserve the reduced dimension
            in the output Tensor. The result tensor is one dimension less than
            the result of ``'x-y'`` unless :attr:`keepdim` is True, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        x: :math:`[N, D]` where `D` is the dimension of vector, available dtype
            is float32, float64.
        y: :math:`[N, D]`, y have the same shape and dtype as x.
        out: :math:`[N]`. If :attr:`keepdim` is ``True``, the out shape is :math:`[N, 1]`.
            The same dtype as input tensor.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            paddle.disable_static()
            x_np = np.array([[1., 3.], [3., 5.]]).astype(np.float64)
            y_np = np.array([[5., 6.], [7., 8.]]).astype(np.float64)
            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            dist = paddle.nn.PairwiseDistance()
            distance = dist(x, y)
            print(distance.numpy()) # [5. 5.]

    """

    def __init__(self, p=2., epsilon=1e-6, keepdim=False, name=None):
        super(PairwiseDistance, self).__init__()
        self.p = p
        self.epsilon = epsilon
        self.keepdim = keepdim
        self.name = name
        check_type(self.p, 'porder', (float, int), 'PairwiseDistance')
        check_type(self.epsilon, 'epsilon', (float), 'PairwiseDistance')
        check_type(self.keepdim, 'keepdim', (bool), 'PairwiseDistance')

    def forward(self, x, y):
        if in_dygraph_mode():
            sub = _C_ops.elementwise_sub(x, y)
            return _C_ops.p_norm(sub, 'axis', 1, 'porder', self.p, 'keepdim',
                                 self.keepdim, 'epsilon', self.epsilon)

        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'PairwiseDistance')
        check_variable_and_dtype(y, 'y', ['float32', 'float64'],
                                 'PairwiseDistance')
        sub = paddle.subtract(x, y)

        helper = LayerHelper("PairwiseDistance", name=self.name)
        attrs = {
            'axis': 1,
            'porder': self.p,
            'keepdim': self.keepdim,
            'epsilon': self.epsilon,
        }
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='p_norm', inputs={'X': sub}, outputs={'Out': out}, attrs=attrs)

        return out

    def extra_repr(self):
        main_str = 'p={p}'
        if self.epsilon != 1e-6:
            main_str += ', epsilon={epsilon}'
        if self.keepdim != False:
            main_str += ', keepdim={keepdim}'
        if self.name != None:
            main_str += ', name={name}'
        return main_str.format(**self.__dict__)
