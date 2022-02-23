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

# TODO: define the initializers of Constant in neural network
from ...fluid.initializer import ConstantInitializer

__all__ = []


class Constant(ConstantInitializer):
    """Implement the constant initializer.

    Args:
        value (float32): constant value to initialize the parameter 

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            data = paddle.rand([30, 10, 2], dtype='float32')
            linear = nn.Linear(2,
                               4,
                               weight_attr=nn.initializer.Constant(value=2.0))
            res = linear(data)
            print(linear.weight.numpy())
            #result is [[2. 2. 2. 2.],[2. 2. 2. 2.]]

    """

    def __init__(self, value=0.0):
        if value is None:
            raise ValueError("value must not be none.")
        super(Constant, self).__init__(value=value, force_cpu=False)
