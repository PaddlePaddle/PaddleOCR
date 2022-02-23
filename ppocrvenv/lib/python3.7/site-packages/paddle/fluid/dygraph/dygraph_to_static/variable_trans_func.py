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

from __future__ import print_function

import six
from paddle.utils import gast

from paddle.fluid import core
from paddle.fluid import unique_name
from paddle.fluid.framework import Variable
from paddle.fluid.layers import fill_constant
from paddle.fluid.layer_helper import LayerHelper

__all__ = [
    'create_bool_as_type', 'create_fill_constant_node',
    'create_static_variable_gast_node', 'data_layer_not_check',
    'to_static_variable', 'to_static_variable_gast_node'
]


def data_layer_not_check(name, shape, dtype='float32', lod_level=0):
    """
    This function creates a Tensor on the global block. The created Tensor
    doesn't check the dtype and the shape of feed data because dygraph input
    data can be various-length. This API is used in translating dygraph into
    static graph.

     Note: 
        The default :code:`stop_gradient` attribute of the Tensor created by
        this API is true, which means the gradient won't be passed backward
        through the data Tensor. Set :code:`var.stop_gradient = False` If
        user would like to pass backward gradient.

    Args:
       name (str): The name/alias of the Tensor, see :ref:`api_guide_Name`
           for more details.
       shape (list|tuple): List|Tuple of integers declaring the shape. You can
           set "None" at a dimension to indicate the dimension can be of any
           size. For example, it is useful to set changeable batch size as "None" 
       dtype (np.dtype|VarType|str, optional): The type of the data. Supported
           dtype: bool, float16, float32, float64, int8, int16, int32, int64,
           uint8. Default: float32
       lod_level (int, optional): The LoD level of the LoDTensor. Usually users
           don't have to set this value. For more details about when and how to
           use LoD level, see :ref:`user_guide_lod_tensor` . Default: 0

    Returns:
        Tensor: The global Tensor that gives access to the data.
    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in six.moves.range(len(shape)):
        if shape[i] is None:
            shape[i] = -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=core.VarDesc.VarType.LOD_TENSOR,
        stop_gradient=True,
        lod_level=lod_level,
        is_data=True,
        need_check_feed=False)


def to_static_variable_gast_node(name):
    func_code = "{} = paddle.jit.dy2static.to_static_variable({})".format(name,
                                                                          name)
    return gast.parse(func_code).body[0]


def create_static_variable_gast_node(name):
    func_code = "{} = paddle.jit.dy2static\
        .data_layer_not_check(name='{}', shape=[-1], dtype='float32')".format(
        name, unique_name.generate(name))
    return gast.parse(func_code).body[0]


def create_fill_constant_node(name, value):
    func_code = "{} = paddle.fluid.layers.fill_constant(shape=[1], ".format(
        name)
    if isinstance(value, bool):
        func_code += "dtype='bool', value={}, name='{}')".format(value, name)
        return gast.parse(func_code).body[0]
    if isinstance(value, float):
        func_code += "dtype='float64', value={}, name='{}')".format(value, name)
        return gast.parse(func_code).body[0]

    if isinstance(value, int):
        func_code += "dtype='int64', value={}, name='{}')".format(value, name)
        return gast.parse(func_code).body[0]


def to_static_variable(x):
    '''
    Translate a Python Tensor to PaddlePaddle static graph Tensor
    '''
    if isinstance(x, bool):
        return fill_constant(shape=[1], dtype='bool', value=x)
    if isinstance(x, float):
        return fill_constant(shape=[1], dtype='float64', value=x)

    if isinstance(x, six.integer_types):
        return fill_constant(shape=[1], dtype='int64', value=x)

    return x


def create_bool_as_type(x, value=True):
    '''
    Create a bool variable, which type is the same as x.
    '''
    if isinstance(x, Variable):
        return fill_constant(shape=[1], value=value, dtype="bool")
    else:
        return value
