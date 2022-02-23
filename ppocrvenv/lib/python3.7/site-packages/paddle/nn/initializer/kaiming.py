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

# TODO: define the initializers of Kaiming functions in neural network
from ...fluid.initializer import MSRAInitializer

__all__ = []


class KaimingNormal(MSRAInitializer):
    r"""Implements the Kaiming Normal initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \sqrt{\frac{2.0}{fan\_in}}

    Args:
        fan_in (float32|None): fan_in for Kaiming normal Initializer. If None, it is\
        inferred from the variable. default is None.

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            linear = nn.Linear(2,
                               4,
                               weight_attr=nn.initializer.KaimingNormal())
            data = paddle.rand([30, 10, 2], dtype='float32')
            res = linear(data)

    """

    def __init__(self, fan_in=None):
        super(KaimingNormal, self).__init__(
            uniform=False, fan_in=fan_in, seed=0)


class KaimingUniform(MSRAInitializer):
    r"""Implements the Kaiming Uniform initializer

    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.
    
    In case of Uniform distribution, the range is [-x, x], where

    .. math::

        x = \sqrt{\frac{6.0}{fan\_in}}

    Args:
        fan_in (float32|None): fan_in for Kaiming uniform Initializer. If None, it is\
        inferred from the variable. default is None.

    Note:
        It is recommended to set fan_in to None for most cases.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            linear = nn.Linear(2,
                               4,
                               weight_attr=nn.initializer.KaimingUniform())
            data = paddle.rand([30, 10, 2], dtype='float32')
            res = linear(data)

    """

    def __init__(self, fan_in=None):
        super(KaimingUniform, self).__init__(
            uniform=True, fan_in=fan_in, seed=0)
