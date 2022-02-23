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

from ...fluid.initializer import UniformInitializer

__all__ = []


class Uniform(UniformInitializer):
    """The random uniform distribution initializer.

    Args:
        low (float, optional): lower boundary of the uniform distribution. The default value is -1.0.
        high (float, optional): upper boundary of the uniform distribution. The default value is 1.0.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A parameter initialized by random uniform distribution.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # linear.weight:  [[-0.46245047  0.05260676]
            #                  [ 0.38054508  0.29169726]]
            # linear.bias:  [-0.2734719   0.23939109]
            
            res = linear(data)
            # res:  [[[-0.3553773  0.5836951]]
            #        [[-0.3553773  0.5836951]]
            #        [[-0.3553773  0.5836951]]]
    """

    def __init__(self, low=-1.0, high=1.0, name=None):
        assert low is not None, 'low should not be None'
        assert high is not None, 'high should not be None'
        assert high >= low, 'high should greater or equal than low'
        super(Uniform, self).__init__(
            low=low, high=high, seed=0, diag_num=0, diag_step=0, diag_val=1.0)
