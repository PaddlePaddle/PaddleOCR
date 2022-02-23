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

from ...fluid.initializer import NormalInitializer
from ...fluid.initializer import TruncatedNormalInitializer

__all__ = []


class Normal(NormalInitializer):
    """The Random Normal (Gaussian) distribution initializer.

    Args:
        mean (float, optional): mean of the normal distribution. The default value is 0.0.
        std (float, optional): standard deviation of the normal distribution. The default value is 1.0.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A parameter initialized by Random Normal (Gaussian) distribution.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0))
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # linear.weight:  [[ 2.1973135 -2.2697184]
            #                  [-1.9104223 -1.0541488]]
            # linear.bias:  [ 0.7885926  -0.74719954]
            
            res = linear(data)
            # res:  [[[ 1.0754838 -4.071067 ]]
            #        [[ 1.0754838 -4.071067 ]]
            #        [[ 1.0754838 -4.071067 ]]]
    """

    def __init__(self, mean=0.0, std=1.0, name=None):
        assert mean is not None, 'mean should not be None'
        assert std is not None, 'std should not be None'
        super(Normal, self).__init__(loc=mean, scale=std, seed=0)


class TruncatedNormal(TruncatedNormalInitializer):
    """The Random TruncatedNormal (Gaussian) distribution initializer.

    Args:
        mean (float, optional): mean of the normal distribution. The default value is 0.0.
        std (float, optional): standard deviation of the normal distribution. The default value is 1.0.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A parameter initialized by Random TruncatedNormal (Gaussian) distribution.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # linear.weight:  [[-1.0981836  1.4140984]
            #                  [ 3.1390522 -2.8266568]]
            # linear.bias:  [-2.1546738 -1.6570673]

            res = linear(data)
            # res:  [[[-0.11380529 -3.0696259 ]]
            #        [[-0.11380529 -3.0696259 ]]
            #        [[-0.11380529 -3.0696259 ]]
    """

    def __init__(self, mean=0.0, std=1.0, name=None):
        assert mean is not None, 'mean should not be None'
        assert std is not None, 'std should not be None'
        super(TruncatedNormal, self).__init__(loc=mean, scale=std, seed=0)
