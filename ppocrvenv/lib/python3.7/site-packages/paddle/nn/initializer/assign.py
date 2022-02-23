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

from ...fluid import framework
from ...fluid import core
from ...fluid import unique_name
from ...fluid.core import VarDesc
from ...fluid.data_feeder import check_type
from ...fluid.initializer import NumpyArrayInitializer

__all__ = []


class Assign(NumpyArrayInitializer):
    """Init an parameter with a numpy array, list, or tensor.

    Args:
        value (Tensor|numpy.ndarray|list|tuple): numpy array, list, tuple, or tensor to initialize the parameter.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A parameter initialized by the input numpy array, list, or tensor.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            # numpy array
            data_1 = paddle.ones(shape=[1, 2], dtype='float32')
            weight_attr_1 = paddle.framework.ParamAttr(
                name="linear_weight_1", 
                initializer=paddle.nn.initializer.Assign(np.array([2, 2])))
            bias_attr_1 = paddle.framework.ParamAttr(
                name="linear_bias_1",
                initializer=paddle.nn.initializer.Assign(np.array([2])))
            linear_1 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
            # linear_1.weight:  [2. 2.]
            # linear_1.bias:  [2.]

            res_1 = linear_1(data_1)
            # res_1:  [6.]

            # python list
            data_2 = paddle.ones(shape=[1, 2], dtype='float32')
            weight_attr_2 = paddle.framework.ParamAttr(
                name="linear_weight_2",
                initializer=paddle.nn.initializer.Assign([2, 2]))
            bias_attr_2 = paddle.framework.ParamAttr(
                name="linear_bias_2",
                initializer=paddle.nn.initializer.Assign([2]))
            linear_2 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_2, bias_attr=bias_attr_2)
            # linear_2.weight:  [2. 2.]
            # linear_2.bias:  [2.]

            res_2 = linear_2(data_2)
            # res_2:  [6.]

            # tensor
            data_3 = paddle.ones(shape=[1, 2], dtype='float32')
            weight_attr_3 = paddle.framework.ParamAttr(
                name="linear_weight_3",
                initializer=paddle.nn.initializer.Assign(paddle.full([2], 2)))
            bias_attr_3 = paddle.framework.ParamAttr(
                name="linear_bias_3",
                initializer=paddle.nn.initializer.Assign(paddle.full([1], 2)))
            linear_3 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_3, bias_attr=bias_attr_3)
            # linear_3.weight:  [2. 2.]
            # linear_3.bias:  [2.]

            res_3 = linear_3(data_3)
            # res_3:  [6.]
    """

    def __init__(self, value, name=None):
        import numpy
        check_type(value, 'value',
                   (numpy.ndarray, list, tuple, framework.Variable), 'Assign')

        if (isinstance(value, (list, tuple))):
            value = numpy.array(value)

        # TODO: value is already is a tensor, accounting efficiency maybe it does not need to convert tensor to numpy data and then initialized.
        if (isinstance(value, framework.Variable)):
            value = value.numpy()

        super(Assign, self).__init__(value)
