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
"""
math functions
"""
from __future__ import print_function
import numpy as np

from paddle.common_ops_import import VarDesc
from paddle.common_ops_import import dygraph_only
from paddle.common_ops_import import OpProtoHolder
from paddle.common_ops_import import templatedoc
from paddle.common_ops_import import dygraph_utils

from paddle.tensor import cast
import paddle
from ..fluid import layers
from ..fluid.framework import core, _varbase_creator, in_dygraph_mode, Variable, convert_np_dtype_to_dtype_
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.layers.layer_function_generator import _generate_doc_string_, generate_activation_fn, generate_layer_fn
from ..fluid.dygraph.inplace_utils import inplace_apis_in_dygraph_only

# TODO: define math functions
# yapf: disable
from ..fluid.layers import abs    # noqa: F401
from ..fluid.layers import acos    # noqa: F401
from ..fluid.layers import asin    # noqa: F401
from ..fluid.layers import ceil    # noqa: F401
from ..fluid.layers import ceil_    # noqa: F401
from ..fluid.layers import cos    # noqa: F401
from ..fluid.layers import tan    # noqa: F401
from ..fluid.layers import sinh    # noqa: F401
from ..fluid.layers import cosh    # noqa: F401
from ..fluid.layers import exp    # noqa: F401
from ..fluid.layers import exp_    # noqa: F401
from ..fluid.layers import expm1    # noqa: F401
from ..fluid.layers import floor    # noqa: F401
from ..fluid.layers import floor_    # noqa: F401
from ..fluid.layers import log    # noqa: F401
from ..fluid.layers import reciprocal    # noqa: F401
from ..fluid.layers import reciprocal_    # noqa: F401
from ..fluid.layers import round    # noqa: F401
from ..fluid.layers import round_    # noqa: F401
from ..fluid.layers import rsqrt    # noqa: F401
from ..fluid.layers import rsqrt_    # noqa: F401
from ..fluid.layers import scale    # noqa: F401
from ..fluid.layers import square    # noqa: F401
from ..fluid.layers import stanh    # noqa: F401
from ..fluid.layers import atan    # noqa: F401
from ..fluid.layers import erf    # noqa: F401
from ..fluid.layers import sqrt    # noqa: F401
from ..fluid.layers import sqrt_    # noqa: F401
from ..fluid.layers import sin    # noqa: F401
from ..fluid.layers import lgamma    # noqa: F401

from ..fluid.layers import multiplex    # noqa: F401
from ..fluid import layers
from paddle import _C_ops

__all__ = []

_supported_int_dtype_ = [
    VarDesc.VarType.UINT8,
    VarDesc.VarType.INT8,
    VarDesc.VarType.INT16,
    VarDesc.VarType.INT32,
    VarDesc.VarType.INT64,
]

_supported_float_dtype_ = [
    VarDesc.VarType.FP32,
    VarDesc.VarType.FP64,
]


@inplace_apis_in_dygraph_only
def scale_(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Inplace version of ``scale`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_scale`.
    """
    _scale = scale.numpy().item(0) if isinstance(scale, Variable) else scale
    return _C_ops.scale_(x, 'scale',
                            float(_scale), 'bias',
                            float(bias), 'bias_after_scale', bias_after_scale)


def pow(x, y, name=None):
    """
    Compute the power of tensor elements. The equation is:

    .. math::
        out = x^{y} 

    **Note**:
    ``paddle.pow`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .


    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64, int32 or int64.
        y (float|int|Tensor): If it is an N-D Tensor, its data type should be the same as `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        N-D Tensor. A location into which the result is stored. Its dimension and data type are the same as `x`.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3], dtype='float32')

            # example 1: y is a float or int
            res = paddle.pow(x, 2)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1., 4., 9.])
            res = paddle.pow(x, 2.5)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1.         , 5.65685415 , 15.58845711])

            # example 2: y is a Tensor
            y = paddle.to_tensor([2], dtype='float32')
            res = paddle.pow(x, y)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1., 4., 9.])

    """
    # in dynamic graph mode
    if in_dygraph_mode():
        if isinstance(y, (int, float)):
            return _C_ops.pow(x, 'factor', y)
        elif isinstance(y, (paddle.Tensor, Variable)):
            return _elementwise_op_in_dygraph(
                x, y, axis=-1, act=None, op_name='elementwise_pow')
        else:
            raise TypeError('y must be scalar or tensor type, but received: %s '% (y.dtype))
    # in static graph mode
    else:
        if isinstance(y, (int, float)):
            helper = LayerHelper('pow', **locals())
            inputs = {'X': x}
            attrs = {'factor': y}
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs)
            return out
        elif isinstance(y, (paddle.Tensor, Variable)):
            # TODO A potential speed improvement is supporting different types in C++ and removing the cast ops here
            helper = LayerHelper('elementwise_pow', **locals())
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            return _elementwise_op(LayerHelper('elementwise_pow', **locals()))
        else:
            raise TypeError('y must be scalar or tensor type, but received: %s '% (type(y)))



@dygraph_only
def _elementwise_op_in_dygraph(x,
                               y,
                               axis=-1,
                               act=None,
                               use_mkldnn=False,
                               op_name=None):
    op = getattr(_C_ops, op_name)
    out = op(x, y, 'axis', axis, 'use_mkldnn', use_mkldnn)

    return dygraph_utils._append_activation_in_dygraph(
        out, act, use_mkldnn=use_mkldnn)


def _elementwise_op(helper):
    op_type = helper.layer_type
    original_op_type = helper.kwargs.get('original_op_type', op_type)
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    out = helper.kwargs.get('out', None)

    assert x is not None, 'x cannot be None in {}'.format(original_op_type)
    assert y is not None, 'y cannot be None in {}'.format(original_op_type)
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
        original_op_type)
    check_variable_and_dtype(
        y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
        original_op_type)

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)

    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type=op_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis,
               'use_mkldnn': use_mkldnn})
    return helper.append_activation(out)


def add(x, y, name=None):
    """
    Examples:

    ..  code-block:: python

        import paddle
        x = paddle.to_tensor([2, 3, 4], 'float64')
        y = paddle.to_tensor([1, 5, 2], 'float64')
        z = paddle.add(x, y)
        print(z)  # [3., 8., 6. ]

    """

    if in_dygraph_mode():
        return _C_ops.elementwise_add(x, y)

    return _elementwise_op(LayerHelper('elementwise_add', **locals()))


@inplace_apis_in_dygraph_only
def add_(x, y, name=None):
    """
    Inplace version of ``add`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_add`.
    """
    op_type = 'elementwise_add_'
    axis = -1

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError("The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(out_shape, x.shape))

    out = _elementwise_op_in_dygraph(
        x, y, axis=axis, op_name=op_type)
    return out


def subtract(x, y, name=None):
    """
    Substract two tensors element-wise. The equation is:

    .. math::
        out = x - y

    **Note**:
    ``paddle.subtract`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[5, 6], [3, 4]])
            res = paddle.subtract(x, y)
            print(res)
            #       [[-4, -4],
            #        [4, 4]]

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([1, 0, 4])
            res = paddle.subtract(x, y)
            print(res)
            #       [[[ 0,  2, -1],
            #         [ 0,  2, -1]]]

            x = paddle.to_tensor([2, np.nan, 5], dtype='float32')
            y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
            res = paddle.subtract(x, y)
            print(res)
            #       [ 1., nan, nan]

            x = paddle.to_tensor([5, np.inf, -np.inf], dtype='float64')
            y = paddle.to_tensor([1, 4, 5], dtype='float64')
            res = paddle.subtract(x, y)
            print(res)
            #       [   4.,  inf., -inf.]

    """
    op_type = 'elementwise_sub'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))


@inplace_apis_in_dygraph_only
def subtract_(x, y, name=None):
    """
    Inplace version of ``subtract`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_subtract`.
    """
    axis = -1
    act = None

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError("The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(out_shape, x.shape))

    out = _elementwise_op_in_dygraph(
        x, y, axis=axis, act=act, op_name='elementwise_sub_')
    return out


def divide(x, y, name=None):
    """
    Divide two tensors element-wise. The equation is:

    .. math::
        out = x / y

    **Note**:
    ``paddle.divide`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 4], dtype='float64')
            y = paddle.to_tensor([1, 5, 2], dtype='float64')
            z = paddle.divide(x, y)
            print(z)  # [2., 0.6, 2.]

    """
    op_type = 'elementwise_div'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))


def floor_divide(x, y, name=None):
    """
    Floor divide two tensors element-wise. The equation is:

    .. math::
        out = x // y

    **Note**:
    ``paddle.floor_divide`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be int32, int64.
        y (Tensor): the input tensor, it's data type should be int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with $x$.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 8, 7])
            y = paddle.to_tensor([1, 5, 3, 3])
            z = paddle.floor_divide(x, y)
            print(z)  # [2, 0, 2, 2]

    """
    op_type = 'elementwise_floordiv'
    axis = -1
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))


def remainder(x, y, name=None):
    r"""
    Mod two tensors element-wise. The equation is:

    .. math::

        out = x \% y

    **Note**:
    ``paddle.remainder`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 8, 7])
            y = paddle.to_tensor([1, 5, 3, 3])
            z = paddle.remainder(x, y)
            print(z)  # [0, 3, 2, 1]

    """
    op_type = 'elementwise_mod'
    axis = -1
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, op_name=op_type)

    return _elementwise_op(LayerHelper(op_type, **locals()))


mod = remainder  # noqa: F841
floor_mod = remainder  # noqa: F841


def multiply(x, y, name=None):
    """
    multiply two tensors element-wise. The equation is:

    .. math::
        out = x * y

    **Note**:
    ``paddle.multiply`` supports broadcasting. If you would like to know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        y (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        ..  code-block:: python

            import paddle

            x = paddle.to_tensor([[1, 2], [3, 4]])
            y = paddle.to_tensor([[5, 6], [7, 8]])
            res = paddle.multiply(x, y)
            print(res) # [[5, 12], [21, 32]]

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([2])
            res = paddle.multiply(x, y)
            print(res) # [[[2, 4, 6], [2, 4, 6]]]

    """
    op_type = 'elementwise_mul'
    act = None
    axis = -1

    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)

    if x.dtype != y.dtype:
        raise TypeError(
            'Input tensors must be same type, but received type of x: %s, type of y: %s '
            % (x.dtype, y.dtype))

    return _elementwise_op(LayerHelper(op_type, **locals()))

def maximum(x, y, name=None):
    """
    Compare two tensors and returns a new tensor containing the element-wise maxima. The equation is:

    .. math::
        out = max(x, y)

    **Note**:
    ``paddle.maximum`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[3, 4], [5, 6]])
            res = paddle.maximum(x, y)
            print(res)
            #    [[3, 4],
            #     [7, 8]]

            x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            y = paddle.to_tensor([3, 0, 4])
            res = paddle.maximum(x, y)
            print(res)
            #    [[3, 2, 4],
            #     [3, 2, 4]]

            x = paddle.to_tensor([2, 3, 5], dtype='float32')
            y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
            res = paddle.maximum(x, y)
            print(res)
            #    [ 2., nan, nan]

            x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
            y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
            res = paddle.maximum(x, y)
            print(res)
            #    [  5.,   3., inf.]
    """
    op_type = 'elementwise_max'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))

def minimum(x, y, name=None):
    """
    Compare two tensors and returns a new tensor containing the element-wise minima. The equation is:

    .. math::
        out = min(x, y)

    **Note**:
    ``paddle.minimum`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = paddle.to_tensor([[1, 2], [7, 8]])
            y = paddle.to_tensor([[3, 4], [5, 6]])
            res = paddle.minimum(x, y)
            print(res)
            #       [[1, 2],
            #        [5, 6]]

            x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            y = paddle.to_tensor([3, 0, 4])
            res = paddle.minimum(x, y)
            print(res)
            #       [[[1, 0, 3],
            #         [1, 0, 3]]]

            x = paddle.to_tensor([2, 3, 5], dtype='float32')
            y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
            res = paddle.minimum(x, y)
            print(res)
            #       [ 1., nan, nan]

            x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
            y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
            res = paddle.minimum(x, y)
            print(res)
            #       [   1., -inf.,    5.]
    """
    op_type = 'elementwise_min'
    axis = -1
    act = None
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))

for func in [
        add,
        multiply
]:
    proto_dict = {'add': 'elementwise_add', 'multiply': 'elementwise_mul'}
    op_proto = OpProtoHolder.instance().get_op_proto(proto_dict[func.__name__])

    additional_args_lines = [
        "name (string, optional): Name of the output. \
        Default is None. It's used to print debug info for developers. Details: \
        :ref:`api_guide_Name` "
    ]

    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=additional_args_lines,
        skip_attrs_set={"x_data_format", "y_data_format", "axis",
            "use_quantizer", "mkldnn_data_type", "Scale_x", "Scale_y", "Scale_out"
        }) + """\n""" + str(func.__doc__)


def sum(x, axis=None, dtype=None, keepdim=False, name=None):
    """
    Computes the sum of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the data type is bool, float16, float32, float64, int32 or int64.
        axis (int|list|tuple, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        dtype (str, optional): The dtype of output Tensor. The default value is None, the dtype
            of output is the same as input Tensor `x`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: Results of summation operation on the specified axis of input Tensor `x`,
        if `x.dtype='bool'`, `x.dtype='int32'`, it's data type is `'int64'`, 
        otherwise it's data type is the same as `x`.

    Raises:
        TypeError: The type of :attr:`axis` must be int, list or tuple.

    Examples:
        .. code-block:: python

            import paddle

            # x is a Tensor with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]])
            out1 = paddle.sum(x)  # [3.5]
            out2 = paddle.sum(x, axis=0)  # [0.3, 0.5, 1.1, 1.6]
            out3 = paddle.sum(x, axis=-1)  # [1.9, 1.6]
            out4 = paddle.sum(x, axis=1, keepdim=True)  # [[1.9], [1.6]]

            # y is a Tensor with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = paddle.to_tensor([[[1, 2], [3, 4]], 
                                  [[5, 6], [7, 8]]])
            out5 = paddle.sum(y, axis=[1, 2]) # [10, 26]
            out6 = paddle.sum(y, axis=[0, 1]) # [16, 20]
            
            # x is a Tensor with following elements:
            #    [[True, True, True, True]
            #     [False, False, False, False]]
            # Each example is followed by the corresponding output tensor.
            x = paddle.to_tensor([[True, True, True, True],
                                  [False, False, False, False]])
            out7 = paddle.sum(x)  # [4]
            out8 = paddle.sum(x, axis=0)  # [1, 1, 1, 1]
            out9 = paddle.sum(x, axis=1)  # [4, 0]
    """
    if axis is not None and not isinstance(axis, (list, tuple)):
        axis = [axis]

    if not axis:
        reduce_all_flag = True
    else:
        if len(axis) == len(x.shape):
            reduce_all_flag = True
        else:
            reduce_all_flag = False

    def get_dtype(x, dtype):
        if dtype is not None:
            return (True, dtype)
        src_type = convert_dtype(x.dtype)
        if src_type in ['bool','int32', 'int64']:
            return (True, 'int64')
        return (False, src_type)

    dtype_flag, dtype = get_dtype(x, dtype)
    if in_dygraph_mode():
        axis = axis if axis != None and axis != [] else [0]
        if dtype_flag:
            return _C_ops.reduce_sum(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag, 'in_dtype',
                                       x.dtype, 'out_dtype',
                                       convert_np_dtype_to_dtype_(dtype))
        else:
            return _C_ops.reduce_sum(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag)

    attrs = {
        'dim': axis if axis != None and axis != [] and axis != () else [0],
        'keep_dim': keepdim,
        'reduce_all': reduce_all_flag
    }

    if dtype_flag:
        attrs.update({
            'in_dtype': x.dtype,
            'out_dtype': convert_np_dtype_to_dtype_(dtype)
        })

    check_variable_and_dtype(
        x, 'x', ['bool', 'float16', 'float32', 'float64',
                'int32', 'int64', 'complex64', 'complex128',
                u'bool', u'float16', u'float32', u'float64',
                u'int32', u'int64', u'complex64', u'complex128'], 'sum')

    check_type(axis, 'axis', (int, list, tuple, type(None)), 'sum')

    helper = LayerHelper('sum', **locals())
    if dtype_flag:
        out = helper.create_variable_for_type_inference(
            dtype=convert_np_dtype_to_dtype_(dtype))
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reduce_sum',
        inputs={'X': x},
        outputs={'Out': out},
        attrs=attrs)
    return out


@templatedoc(op_type="sum")
def add_n(inputs, name=None):
    """
    This OP is used to sum one or more Tensor of the input.
    
    For example:

    .. code-block:: text
    
        Case 1:

            Input:
                input.shape = [2, 3]
                input = [[1, 2, 3],
                         [4, 5, 6]]

            Output:
                output.shape = [2, 3]
                output = [[1, 2, 3],
                          [4, 5, 6]]

        Case 2:
       
            Input:
                First input:
                    input1.shape = [2, 3]
                    Input1 = [[1, 2, 3],
                              [4, 5, 6]]

                The second input:
                    input2.shape = [2, 3]
                    input2 = [[7, 8, 9],
                              [10, 11, 12]]

                Output:
                    output.shape = [2, 3]
                    output = [[8, 10, 12],
                              [14, 16, 18]]

    Args:
        inputs (Tensor|list[Tensor]|tuple[Tensor]):  A Tensor or a list/tuple of Tensors. The shape and data type of the list/tuple elements should be consistent.
            Input can be multi-dimensional Tensor, and data types can be: float32, float64, int32, int64.
        name(str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor, the sum of input :math:`inputs` , its shape and data types are consistent with :math:`inputs`.

    Examples:
        .. code-block:: python

            import paddle

            input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
            input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
            output = paddle.add_n([input0, input1])
            # [[8., 10., 12.], 
            #  [14., 16., 18.]]
    """
    if in_dygraph_mode():
        if isinstance(inputs, Variable):
            inputs = [inputs]
        return _C_ops.sum(inputs, 'use_mkldnn', False)

    helper = LayerHelper('add_n', **locals())
    check_type(inputs, 'inputs', (Variable, tuple, list), 'add_n')
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) > 0:
            for input in inputs:
                check_variable_and_dtype(input, "inputs", \
                   ['float32', 'float64', 'int32', 'int64'], 'add_n')
    else:
        check_variable_and_dtype(inputs, "inputs", \
                ['float32', 'float64', 'int32', 'int64'], 'add_n')


    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('inputs'))
    helper.append_op(
        type='sum',
        inputs={'X': inputs},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})

    return out


def trunc(input, name=None):
    '''
    This API is used to returns a new tensor with the truncated integer values of input.
    
    Args:
        input (Tensor): The input tensor, it's data type should be int32, int64, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        Tensor: The output Tensor of trunc.
    
    Examples:
        .. code-block:: python

            import paddle

            input = paddle.rand([2,2],'float32')
            print(input)
            # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [[0.02331470, 0.42374918],
            #         [0.79647720, 0.74970269]])

            output = paddle.trunc(input)
            print(output)
            # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #         [[0., 0.],
            #         [0., 0.]]))
    '''
    if in_dygraph_mode():
        return _C_ops.trunc(input)
    else:
        inputs = {"X": input}
        attrs = {}

        helper = LayerHelper("trunc", **locals())
        check_variable_and_dtype(input, 'X', ['int32', 'int64', 'float32', 'float64'], 'trunc')
        out = helper.create_variable_for_type_inference(dtype=input.dtype)

        helper.append_op(
            type="trunc", inputs=inputs, attrs=attrs, outputs={"Out": out})
        return out



def mm(input, mat2, name=None):
    """

    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.


    Also note that if the raw tensor :math:`x` or :math:`mat2` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        input (Tensor): The input tensor which is a Tensor.
        mat2 (Tensor): The input tensor which is a Tensor.
        name(str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: The product Tensor.

    ::
        * example 1:

        input: [B, ..., M, K], mat2: [B, ..., K, N]
        out: [B, ..., M, N]

        * example 2:

        input: [B, M, K], mat2: [B, K, N]
        out: [B, M, N]

        * example 3:

        input: [B, M, K], mat2: [K, N]
        out: [B, M, N]

        * example 4:

        input: [M, K], mat2: [K, N]
        out: [M, N]

        * example 5:

        input: [B, M, K], mat2: [K]
        out: [B, M]

        * example 6:

        input: [K], mat2: [K]
        out: [1]

    Examples:
        .. code-block:: python

            import paddle
            input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
            mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
            out = paddle.mm(input, mat2)
            print(out)
            #        [[11., 14., 17., 20.],
            #         [23., 30., 37., 44.],
            #         [35., 46., 57., 68.]])


    """
    if in_dygraph_mode():
        return _C_ops.matmul_v2(input, mat2)

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(val, name,
                                     ['float16', 'float32', 'float64'], 'mm')
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if x_shape[-1] != y_shape[-2]:
            if not ((x_shape[-1] == -1) or (y_shape[-2] == -1)):
                raise ValueError(
                    "After performing an optional transpose, Input X's width should be "
                    "equal to Y's width for multiplication "
                    "prerequisites. But received X's shape: %s, Y's shape: %s\n"
                    % (x_shape, y_shape))

        if len(y_shape) > 2 and len(x_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                # don't check neg shape
                if dim_x < 0 or y_shape[i] < 0:
                    continue
                if dim_x != y_shape[i]:
                    raise ValueError(
                        "When the matrix is larger than 2 dimensions, the higher "
                        "dimensional values of the two matrices need to be equal. "
                        "But received x_shape[%d] != y_shape[%d]. X's shape: %s, "
                        "Y's shape: %s.\n" % (i, i, x_shape, y_shape))

    __check_input(input, mat2)

    helper = LayerHelper('mm', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='matmul_v2', inputs={'X': input,
                               'Y': mat2}, outputs={'Out': out})
    return out


def addmm(input, x, y, beta=1.0, alpha=1.0, name=None):
    """
    **addmm**

    This operator is used to perform matrix multiplication for input $x$ and $y$.
    $input$ is added to the final result.
    The equation is:

    ..  math::
        Out = alpha * x * y + beta * input

    $Input$, $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $input$.

    Args:
        input (Tensor): The input Tensor to be added to the final result.
        x (Tensor): The first input Tensor for matrix multiplication.
        y (Tensor): The second input Tensor for matrix multiplication.
        beta (float): Coefficient of $input$.
        alpha (float): Coefficient of $x*y$.
        name (str, optional): Name of the output. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Tensor: The output Tensor of addmm op.

    Examples:
        ..  code-block:: python
            
            import paddle

            x = paddle.ones([2,2])
            y = paddle.ones([2,2])
            input = paddle.ones([2,2])

            out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

            print(out)
            # [[10.5 10.5]
            # [10.5 10.5]]
    """
    input_shape = input.shape
    x_shape = x.shape
    y_shape = y.shape
    if not len(input_shape) == len(x_shape) == len(y_shape) == 2:
        raise ValueError("The dimention of input, x, y should be 2 but receive input's shape: {}, x's shape: {}, y's shape: {}".format(input_shape, x_shape, y_shape))
    if input_shape[0] != x_shape[0]:
        if input_shape[0] != 1:
            raise ValueError( "When x's dimension[0] is not equal with input's dimension[0], input's dimension[0] must be 1 but got {}".format(input_shape[0]))
        if input_shape[1] != y_shape[1] and input_shape[1] != 1:
            raise ValueError( "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(input_shape[1]))
    if input_shape[1] != y_shape[1]:
        if input_shape[1] != 1:
            raise ValueError( "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(input_shape[1]))
        if input_shape[0] != x_shape[0] and input_shape[0] != 1:
            raise ValueError( "When x's dimension[0] is not equal with input's dimension[0], input's dimension[0] must be 1 but got {}".format(input_shape[0]))
    if x_shape[1] != y_shape[0]:
        raise ValueError("The input Variable x's width must be equal with Variable y' height. But received x's shape = {}, y's shape = {}.".format(x_shape, y_shape))



    if in_dygraph_mode():
        out = _C_ops.addmm(input, x, y, "Alpha", alpha, "Beta", beta)
        return out

    inputs = {'Input': input, "X": x, "Y": y}
    attrs = {'Alpha': alpha, 'Beta': beta}

    helper = LayerHelper("addmm", **locals())
    check_variable_and_dtype(input, 'Input', ['float32', 'float64'], 'addmm')
    check_variable_and_dtype(x, 'X', ['float32', 'float64'], 'addmm')
    check_variable_and_dtype(y, 'Y', ['float32', 'float64'], 'addmm')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="addmm", inputs=inputs, attrs=attrs, outputs={"Out": out})
    return out


def logsumexp(x, axis=None, keepdim=False, name=None):
    r"""
    This OP calculates the log of the sum of exponentials of ``x`` along ``axis`` .

    .. math::
       logsumexp(x) = \\log\\sum exp(x)

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int|list|tuple, optional): The axis along which to perform
            logsumexp calculations. ``axis`` should be int, list(int) or
            tuple(int). If ``axis`` is a list/tuple of dimension(s), logsumexp
            is calculated along all element(s) of ``axis`` . ``axis`` or
            element(s) of ``axis`` should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` or element(s) of ``axis`` is
            less than 0, it works the same way as :math:`axis + D` . If
            ``axis`` is None, logsumexp is calculated along all elements of
            ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keep_dim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of logsumexp along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:

    .. code-block:: python

        import paddle

        x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
        out1 = paddle.logsumexp(x) # [3.4691226]
        out2 = paddle.logsumexp(x, 1) # [2.15317821, 3.15684602]

    """
    if isinstance(axis, int):
        axis = [axis]
    reduce_all = True if axis is None \
        or len(axis)==0 \
        or len(axis) == len(x.shape) else False
    if axis is None or len(axis) == 0:
        axis = [0]

    if in_dygraph_mode():
        return _C_ops.logsumexp(x, 'axis', axis, 'keepdim', keepdim, 'reduce_all', reduce_all)

    check_variable_and_dtype(x, 'x',
                             ['float32', 'float64'],
                             'logsumexp')

    helper = LayerHelper('logsumexp', **locals())
    attrs = {'axis': axis, 'keepdim': keepdim, 'reduce_all':reduce_all}
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='logsumexp', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    return out


def inverse(x, name=None):
    """
    Takes the inverse of the square matrix. A square matrix is a matrix with
    the same number of rows and columns. The input can be a square matrix
    (2-D Tensor) or batches of square matrices.

    Args:
        x (Tensor): The input tensor. The last two
            dimensions should be equal. When the number of dimensions is
            greater than 2, it is treated as batches of square matrix. The data
            type can be float32 and float64.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information,
            please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: A Tensor holds the inverse of x. The shape and data type
                        is the same as x.

    Examples:
        .. code-block:: python

            import paddle

            mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
            inv = paddle.inverse(mat)
            print(inv) # [[0.5, 0], [0, 0.5]]

    """
    if in_dygraph_mode():
        return _C_ops.inverse(x)

    def _check_input(x):
        check_variable_and_dtype(x, 'x',
                                 ['float32', 'float64'], 'inverse')
        if len(x.shape) < 2:
            raise ValueError(
                "The input of inverse is expected to be a Tensor whose number "
                "of dimensions is no less than 2. But reviced: %d, "
                "x's shape: %s." % (len(x.shape), x.shape))
    _check_input(x)
    helper = LayerHelper('inverse', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='inverse', inputs={'Input': [x] }, outputs={'Output': [out]})
    return out


def max(x, axis=None, keepdim=False, name=None):
    """

    Computes the maximum of tensor elements over the given axis.

    Args:
        x(Tensor): A tensor, the data type is float32,
            float64, int32, int64.
        axis(int|list|tuple, optional): The axis along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim(bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: python

            import paddle

            # data_x is a Tensor with shape [2, 4]
            # the axis is a int element

            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]])
            result1 = paddle.max(x)
            print(result1)
            #[0.9]
            result2 = paddle.max(x, axis=0)
            print(result2)
            #[0.2 0.3 0.6 0.9]
            result3 = paddle.max(x, axis=-1)
            print(result3)
            #[0.9 0.7]
            result4 = paddle.max(x, axis=1, keepdim=True)
            print(result4)
            #[[0.9]
            # [0.7]]

            # data_y is a Tensor with shape [2, 2, 2]
            # the axis is list 

            y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])
            result5 = paddle.max(y, axis=[1, 2])
            print(result5)
            #[4. 8.]
            result6 = paddle.max(y, axis=[0, 1])
            print(result6)
            #[7. 8.]
    """

    if axis is not None and not isinstance(axis, list):
        if isinstance(axis, tuple):
            axis = list(axis)
        elif isinstance(axis, int):
            axis= [axis]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".format(type(axis)))

    reduce_all = True if axis == None or axis == [] else False
    axis = axis if axis != None and axis != [] else [0]
    if in_dygraph_mode():
        return _C_ops.reduce_max(x, 'dim', axis, 'keep_dim', keepdim,
                                   'reduce_all', reduce_all)

    helper = LayerHelper('max', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'max')

    out = helper.create_variable_for_type_inference(
            dtype=x.dtype)
    helper.append_op(
        type='reduce_max',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all
        })
    return out

def min(x, axis=None, keepdim=False, name=None):
    """

    Computes the minimum of tensor elements over the given axis

    Args:
        x(Tensor): A tensor, the data type is float32, float64, int32, int64.
        axis(int|list|tuple, optional): The axis along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim(bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle

            # x is a tensor with shape [2, 4]
            # the axis is a int element
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]])
            result1 = paddle.min(x)
            print(result1)
            #[0.1]
            result2 = paddle.min(x, axis=0)
            print(result2)
            #[0.1 0.2 0.5 0.7]
            result3 = paddle.min(x, axis=-1)
            print(result3)
            #[0.2 0.1]
            result4 = paddle.min(x, axis=1, keepdim=True)
            print(result4)
            #[[0.2]
            # [0.1]]

            # y is a Tensor with shape [2, 2, 2]
            # the axis is list 
            y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])
            result5 = paddle.min(y, axis=[1, 2])
            print(result5)
            #[1. 5.]
            result6 = paddle.min(y, axis=[0, 1])
            print(result6)
            #[1. 2.]
    """

    if axis is not None and not isinstance(axis, list):
        if isinstance(axis, tuple):
            axis = list(axis)
        elif isinstance(axis, int):
            axis= [axis]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".format(type(axis)))
    reduce_all = True if axis == None or axis == [] else False
    axis = axis if axis != None and axis != [] else [0]
    if in_dygraph_mode():
        return _C_ops.reduce_min(x, 'dim', axis, 'keep_dim', keepdim,
                                   'reduce_all', reduce_all)

    helper = LayerHelper('min', **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'min')

    out = helper.create_variable_for_type_inference(
            dtype=x.dtype)
    helper.append_op(
        type='reduce_min',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all
        })
    return out


def log1p(x, name=None):
    r"""
    Calculates the natural log of the given input tensor, element-wise.

    .. math::
        Out = \\ln(x+1)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Tensor, the natural log of the input Tensor computed element-wise.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.to_tensor([[0], [1]], dtype='float32')
            res = paddle.log1p(data)
            # [[0.], [0.6931472]]
    """

    if in_dygraph_mode():
        return _C_ops.log1p(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], "log1p")
    inputs = {'X': [x]}
    helper = LayerHelper('log1p', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log1p", inputs={"X": x}, outputs={"Out": out})
    return out

def log2(x, name=None):
    r"""
    Calculates the log to the base 2 of the given input tensor, element-wise.

    .. math::

        Out = \\log_2x

    Args:
        x (Tensor): Input tensor must be one of the following types: float32, float64.
        name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`


    Returns:
        Tensor: The log to the base 2 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python
        
            import paddle

            # example 1: x is a float
            x_i = paddle.to_tensor([[1.0], [2.0]])
            res = paddle.log2(x_i) # [[0.], [1.0]]

            # example 2: x is float32
            x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
            paddle.to_tensor(x_i)
            res = paddle.log2(x_i)
            print(res) # [1.0]

            # example 3: x is float64
            x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
            paddle.to_tensor(x_i)
            res = paddle.log2(x_i)
            print(res) # [1.0]
    """
    if in_dygraph_mode():
        return _C_ops.log2(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], "log2")
    inputs = {'X': [x]}
    helper = LayerHelper('log2', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log2", inputs={"X": x}, outputs={"Out": out})
    return out


def log10(x, name=None):
    r"""
    Calculates the log to the base 10 of the given input tensor, element-wise.

    .. math::

        Out = \\log_10_x

    Args:
        x (Tensor): Input tensor must be one of the following types: float32, float64.
        name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`


    Returns:
        Tensor: The log to the base 10 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python
        
            import paddle

            # example 1: x is a float
            x_i = paddle.to_tensor([[1.0], [10.0]])
            res = paddle.log10(x_i) # [[0.], [1.0]]

            # example 2: x is float32
            x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
            paddle.to_tensor(x_i)
            res = paddle.log10(x_i)
            print(res) # [1.0]

            # example 3: x is float64
            x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
            paddle.to_tensor(x_i)
            res = paddle.log10(x_i)
            print(res) # [1.0]
    """
    if in_dygraph_mode():
        return _C_ops.log10(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], "log10")
    inputs = {'X': [x]}
    helper = LayerHelper('log10', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log10", inputs={"X": x}, outputs={"Out": out})
    return out


def clip(x, min=None, max=None, name=None):
    """
    This operator clip all elements in input into the range [ min, max ] and return
    a resulting tensor as the following equation:

    .. math::

        Out = MIN(MAX(x, min), max)

    Args:
        x (Tensor): An N-D Tensor with data type float32, float64, int32 or int64.
        min (float|int|Tensor): The lower bound with type ``float`` , ``int`` or a ``Tensor``
            with shape [1] and type ``int32``, ``float32``, ``float64``.
        max (float|int|Tensor): The upper bound with type ``float``, ``int`` or a ``Tensor``
            with shape [1] and type ``int32``, ``float32``, ``float64``.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor with the same data type and data shape as input.

    Examples:
        .. code-block:: python

            import paddle

            x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
            out1 = paddle.clip(x1, min=3.5, max=5.0)
            out2 = paddle.clip(x1, min=2.5)
            print(out1)
            # [[3.5, 3.5]
            # [4.5, 5.0]]
            print(out2)
            # [[2.5, 3.5]
            # [[4.5, 6.4]
    """

    x_dtype = str(x.dtype)
    if x_dtype == 'paddle.int32':
        min_ = np.iinfo(np.int32).min
        max_ = np.iinfo(np.int32).max - 2**7
    elif x_dtype == 'paddle.int64':
        min_ = np.iinfo(np.int64).min
        max_ = np.iinfo(np.int64).max - 2**39
    else:
        min_ = float(np.finfo(np.float32).min)
        max_ = float(np.finfo(np.float32).max)

    if in_dygraph_mode():
        if isinstance(min, Variable):
            min = min.numpy().item(0)
        if isinstance(max, Variable):
            max = max.numpy().item(0)
        min = min_ if min is None else min
        max = max_ if max is None else max
        return _C_ops.clip(x, "min", min, "max", max)

    if min is not None:
        check_type(min, 'min', (float, int, Variable), 'clip')
        if isinstance(min, Variable):
            check_dtype(min.dtype, 'min', ['float32', 'float64', 'int32'],
                        'clip', '(When the type of min in clip is Variable.)')
    if max is not None:
        check_type(max, 'max', (float, int, Variable), 'clip')
        if isinstance(max, Variable):
            check_dtype(max.dtype, 'max', ['float32', 'float64', 'int32'],
                        'clip', '(When the type of max in clip is Variable.)')

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'], 'clip')

    inputs = {'X': x}
    attrs = {'min': min_, 'max': max_}

    if isinstance(min, Variable):
        min.stop_gradient = True
        inputs['Min'] = min
    elif min is not None:
        attrs['min'] = min

    if isinstance(max, Variable):
        max.stop_gradient = True
        inputs['Max'] = max
    elif max is not None:
        attrs['max'] = max

    helper = LayerHelper('clip', **locals())
    output = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('x'))
    helper.append_op(
        type='clip', inputs=inputs, outputs={'Out': [output]}, attrs=attrs)

    return output


@inplace_apis_in_dygraph_only
def clip_(x, min=None, max=None, name=None):
    """
    Inplace version of ``clip`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_clip`.
    """
    fmin = float(np.finfo(np.float32).min)
    fmax = float(np.finfo(np.float32).max)
    if isinstance(min, Variable):
        min = min.numpy().item(0)
    if isinstance(max, Variable):
        max = max.numpy().item(0)
    min = fmin if min is None else min
    max = fmax if max is None else max
    return _C_ops.clip_(x, "min", min, "max", max)



def trace(x, offset=0, axis1=0, axis2=1, name=None):
    """
    **trace**

    This OP computes the sum along diagonals of the input tensor x.

    If ``x`` is 2D, returns the sum of diagonal.

    If ``x`` has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
    the 2D planes specified by axis1 and axis2. By default, the 2D planes formed by the first and second axes
    of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.
    - Note that if offset is out of input's shape indicated by axis1 and axis2, 0 will be returned.

    Args:
        x(Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be float32, float64, int32, int64.
        offset(int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1(int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2(int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            import paddle

            case1 = paddle.randn([2, 3])
            case2 = paddle.randn([3, 10, 10])
            case3 = paddle.randn([3, 10, 5, 10])
            data1 = paddle.trace(case1) # data1.shape = [1]
            data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) # data2.shape = [3]
            data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) # data2.shape = [3, 5]
    """
    def __check_input(input, offset, dim1, dim2):
        check_dtype(x.dtype, 'Input',
                    ['int32', 'int64', 'float16', 'float32', 'float64'],
                    'trace')

        input_shape = list(x.shape)
        assert len(input_shape) >= 2,                     \
                "The x must be at least 2-dimensional, "   \
                "But received Input x's dimensional: %s.\n" %  \
                len(input_shape)

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

        assert ((0 <= axis1_) and (axis1_ < len(input_shape))),     \
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape)), len(input_shape) - 1, axis1)

        assert ((0 <= axis2_) and (axis2_ < len(input_shape))),   \
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"   \
            % (-(len(input_shape)), len(input_shape) - 1, axis2)


        assert  axis1_ != axis2_,   \
               "axis1 and axis2 cannot be the same axis." \
                "But received axis1 = %d, axis2 = %d\n"%(axis1, axis2)

    __check_input(input, offset, axis1, axis2)
    if in_dygraph_mode():
        return _C_ops.trace(x, 'offset', offset, 'axis1', axis1, 'axis2', axis2)

    inputs = {'Input': [x]}
    attrs = {'offset': offset, 'axis1': axis1, 'axis2': axis2}
    helper = LayerHelper('trace', **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='trace',
        inputs={'Input': [x]},
        attrs={'offset': offset,
               'axis1': axis1,
               'axis2': axis2},
        outputs={'Out': [out]})
    return out

def diagonal(x, offset=0, axis1=0, axis2=1, name=None):
    """
    This OP computes the diagonals of the input tensor x.

    If ``x`` is 2D, returns the diagonal.
    If ``x`` has larger dimensions, diagonals be taken from the 2D planes specified by axis1 and axis2. 
    By default, the 2D planes formed by the first and second axis of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.
    
    Args:
        x(Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be bool, int32, int64, float16, float32, float64.
        offset(int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1(int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2(int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: a partial view of input tensor in specify two dimensions, the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.rand([2,2,3],'float32')
            print(x)
            # Tensor(shape=[2, 2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[[0.45661032, 0.03751532, 0.90191704],
            #          [0.43760979, 0.86177313, 0.65221709]],

            #         [[0.17020577, 0.00259554, 0.28954273],
            #          [0.51795638, 0.27325270, 0.18117726]]])

            out1 = paddle.diagonal(x)
            print(out1)
            #Tensor(shape=[3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[0.45661032, 0.51795638],
            #        [0.03751532, 0.27325270],
            #        [0.90191704, 0.18117726]])

            out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
            print(out2)
            #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[0.45661032, 0.86177313],
            #        [0.17020577, 0.27325270]])

            out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
            print(out3)
            #Tensor(shape=[3, 1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[0.43760979],
            #        [0.86177313],
            #        [0.65221709]])

            out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
            print(out4)
            #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[0.45661032, 0.86177313],
            #        [0.17020577, 0.27325270]])
            
    """
    if in_dygraph_mode():
        return _C_ops.diagonal(x, 'offset', offset, 'axis1', axis1, 'axis2', axis2)

    def __check_input(input, offset, dim1, dim2):
        check_dtype(x.dtype, 'Input',
                    ['bool', 'int32', 'int64', 'float16', 'float32', 'float64'],
                    'diagonal')

        input_shape = list(x.shape)
        assert len(input_shape) >= 2,                     \
                "The x must be at least 2-dimensional, "   \
                "But received Input x's dimensional: %s.\n" %  \
                len(input_shape)

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

        assert axis1_ < len(input_shape),     \
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape)), len(input_shape) - 1, axis1)

        assert axis2_ < len(input_shape),   \
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"   \
            % (-(len(input_shape)), len(input_shape) - 1, axis2)

        assert  axis1_ != axis2_,   \
               "axis1 and axis2 cannot be the same axis." \
                "But received axis1 = %d, axis2 = %d\n"%(axis1, axis2)

    __check_input(input, offset, axis1, axis2)
    helper = LayerHelper('diagonal', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='diagonal',
        inputs={'Input': [x]},
        attrs={'offset': offset,
               'axis1': axis1,
               'axis2': axis2},
               outputs={'Out': [out]})
    return out


@templatedoc(op_type="kron")
def kron(x, y, name=None):
    """

${comment}

    Args:
        x (Tensor): the fist operand of kron op, data type: float16, float32,
            float64, int32 or int64.
        y (Tensor): the second operand of kron op, data type: float16,
            float32, float64, int32 or int64. Its data type should be the same
            with x.
        name(str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output of kron op, data type: float16, float32, float64, int32 or int64. Its data is the same with x.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
            y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
            out = paddle.kron(x, y)
            print(out)
            #        [[1, 2, 3, 2, 4, 6],
            #         [ 4,  5,  6,  8, 10, 12],
            #         [ 7,  8,  9, 14, 16, 18],
            #         [ 3,  6,  9,  4,  8, 12],
            #         [12, 15, 18, 16, 20, 24],
            #         [21, 24, 27, 28, 32, 36]])
    """
    if in_dygraph_mode():
        return _C_ops.kron(x, y)

    helper = LayerHelper('kron', **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron')
    check_variable_and_dtype(y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron')

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="kron", inputs={"X": x, "Y": y}, outputs={"Out": out})
    return out


def cumsum(x, axis=None, dtype=None, name=None):
    """
    The cumulative sum of the elements along a given axis. 
    
    **Note**:
    The first element of the result is the same of the first element of the input. 

    Args:
        x (Tensor): The input tensor needed to be cumsumed.
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cumsum over the flattened array.
        dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumsum operator. 

    Examples:
        .. code-block:: python
            
            import paddle
            
            data = paddle.arange(12)
            data = paddle.reshape(data, (3, 4))

            y = paddle.cumsum(data)
            # [ 0  1  3  6 10 15 21 28 36 45 55 66]

            y = paddle.cumsum(data, axis=0)
            # [[ 0  1  2  3]
            #  [ 4  6  8 10]
            #  [12 15 18 21]]
            
            y = paddle.cumsum(data, axis=-1)
            # [[ 0  1  3  6]
            #  [ 4  9 15 22]
            #  [ 8 17 27 38]]

            y = paddle.cumsum(data, dtype='float64')
            print(y.dtype)
            # VarType.FP64
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = layers.cast(x, dtype)

    if in_dygraph_mode():
        if axis is None:
            return _C_ops.cumsum(x, 'flatten', flatten)
        else:
            return _C_ops.cumsum(x, 'axis', axis, 'flatten', flatten)

    check_type(x, 'x', (Variable), 'cumsum')
    locals_var = locals().copy()
    kwargs = dict()
    for name, val in locals_var.items():
        if val is not None:
            kwargs[name] = val
    _cum_sum_ = generate_layer_fn('cumsum')
    return _cum_sum_(**kwargs)

def cumprod(x, dim=None, dtype=None, name=None):
    """
    Compute the cumulative product of the input tensor x along a given dimension dim.

    **Note**:
    The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): the input tensor need to be cumproded.
        dim (int): the dimension along which the input tensor will be accumulated. It need to be in the range of [-x.rank, x.rank), where x.rank means the dimensions of the input tensor x and -1 means the last dimension.
        dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64, complex64, complex128. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumprod operator.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.arange(12)
            data = paddle.reshape(data, (3, 4))
            # [[ 0  1  2  3 ]
            #  [ 4  5  6  7 ]
            #  [ 8  9  10 11]]

            y = paddle.cumprod(data, dim=0)
            # [[ 0  1   2   3]
            #  [ 0  5  12  21]
            #  [ 0 45 120 231]]

            y = paddle.cumprod(data, dim=-1)
            # [[ 0   0   0    0]
            #  [ 4  20 120  840]
            #  [ 8  72 720 7920]]

            y = paddle.cumprod(data, dim=1, dtype='float64')
            # [[ 0.   0.   0.    0.]
            #  [ 4.  20. 120.  840.]
            #  [ 8.  72. 720. 7920.]]

            print(y.dtype)
            # paddle.float64

    """

    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = layers.cast(x, dtype)

    if in_dygraph_mode():
        return _C_ops.cumprod(x, 'dim', dim)

    check_variable_and_dtype(x, "x", ['complex64', 'complex128', 'float32', 'float64', 'int32', 'int64'], 'cumprod')
    check_type(dim, 'dim', int, 'cumprod')

    helper = LayerHelper('cumprod', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='cumprod', inputs={'X': x}, outputs={'Out': out}, attrs={'dim': dim})
    return out

def isfinite(x, name=None):
    """

    Return whether every element of input tensor is finite number or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is finite number or not.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            out = paddle.tensor.isfinite(x)
            print(out)  # [False  True  True False  True False False]
    """
    if in_dygraph_mode():
        return _C_ops.isfinite_v2(x)
    helper = LayerHelper("isfinite_v2", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'isfinite')
    out = helper.create_variable_for_type_inference('bool')
    helper.append_op(type="isfinite_v2", inputs={"X": x}, outputs={"Out": out})
    return out

def isinf(x, name=None):
    """

    Return whether every element of input tensor is `+/-INF` or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `+/-INF` or not.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            out = paddle.tensor.isinf(x)
            print(out)  # [ True False False  True False False False]
    """
    if in_dygraph_mode():
        return _C_ops.isinf_v2(x)
    helper = LayerHelper("isinf_v2", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'isinf')
    out = helper.create_variable_for_type_inference(dtype='bool')
    helper.append_op(type="isinf_v2", inputs={"X": x}, outputs={"Out": out})
    return out

def isnan(x, name=None):
    """

    Return whether every element of input tensor is `NaN` or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `NaN` or not.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            out = paddle.tensor.isnan(x)
            print(out)  # [False False False False False  True  True]
    """
    if in_dygraph_mode():
        return _C_ops.isnan_v2(x)
    helper = LayerHelper("isnan_v2", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'isnan')
    out = helper.create_variable_for_type_inference(dtype='bool')
    helper.append_op(type="isnan_v2", inputs={"X": x}, outputs={"Out": out})
    return out


def prod(x, axis=None, keepdim=False, dtype=None, name=None):
    """
    Compute the product of tensor elements over the given axis.

    Args:
        x(Tensor): The input tensor, its data type should be float32, float64, int32, int64.
        axis(int|list|tuple, optional): The axis along which the product is computed. If :attr:`None`, 
            multiply all elements of `x` and return a Tensor with a single element, 
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`. If :math:`axis[i]<0`, 
            the axis to reduce is :math:`x.ndim + axis[i]`. Default is None.
        dtype(str|np.dtype, optional): The desired date type of returned tensor, can be float32, float64, 
            int32, int64. If specified, the input tensor is casted to dtype before operator performed. 
            This is very useful for avoiding data type overflows. The default value is None, the dtype 
            of output is the same as input Tensor `x`.
        keepdim(bool, optional): Whether to reserve the reduced dimension in the output Tensor. The result 
            tensor will have one fewer dimension than the input unless `keepdim` is true. Default is False.
        name(string, optional): The default value is None. Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, result of product on the specified dim of input tensor.

    Raises:
        ValueError: The :attr:`dtype` must be float32, float64, int32 or int64.
        TypeError: The type of :attr:`axis` must be int, list or tuple.
    
    Examples:
        .. code-block:: python

            import paddle

            # the axis is a int element
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]])
            out1 = paddle.prod(x)
            # [0.0002268]

            out2 = paddle.prod(x, -1)
            # [0.027  0.0084]

            out3 = paddle.prod(x, 0)
            # [0.02 0.06 0.3  0.63]

            out4 = paddle.prod(x, 0, keepdim=True)
            # [[0.02 0.06 0.3  0.63]]

            out5 = paddle.prod(x, 0, dtype='int64')
            # [0 0 0 0]

            # the axis is list
            y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])
            out6 = paddle.prod(y, [0, 1])
            # [105. 384.]

            out7 = paddle.prod(y, (1, 2))
            # [  24. 1680.]

    """
    if dtype is not None:
        check_dtype(dtype, 'dtype', ['float32', 'float64', 'int32', 'int64'], 'prod')
        if x.dtype != convert_np_dtype_to_dtype_(dtype):
            x = layers.cast(x, dtype)

    return layers.reduce_prod(input=x, dim=axis, keep_dim=keepdim, name=name)


def sign(x, name=None):
    """
    This OP returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero.

    Args:
        x(Tensor): The input tensor. The data type can be float16, float32 or float64.
        name (str, optional): The default value is None. Normally there is no need for user to
            set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: The output sign tensor with identical shape and data type to the input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle

          x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
          out = paddle.sign(x=x)
          print(out)  # [1.0, 0.0, -1.0, 1.0]
    """
    if in_dygraph_mode():
        return _C_ops.sign(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'sign')
    helper = LayerHelper("sign", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(type='sign', inputs={'X': [x]}, outputs={'Out': [out]})

    return out


def tanh(x, name=None):
    r"""
    Tanh Activation Operator.

    .. math::
        out = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Args:
        x (Tensor): Input of Tanh operator, an N-D Tensor, with data type float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output of Tanh operator, a Tensor with same data type and shape as input.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            out = paddle.tanh(x)
            print(out)
            # [-0.37994896 -0.19737532  0.09966799  0.29131261]
    """
    if in_dygraph_mode():
        return _C_ops.tanh(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'tanh')
    check_type(x, 'x', (Variable), 'tanh')
    helper = LayerHelper('tanh', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='tanh', inputs={'X': x}, outputs={'Out': out})
    return out

@inplace_apis_in_dygraph_only
def tanh_(x, name=None):
    r"""
    Inplace version of ``tanh`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_tanh`.
    """
    return _C_ops.tanh_(x)


def increment(x, value=1.0, name=None):
    """
    The OP is usually used for control flow to increment the data of :attr:`x` by an amount :attr:`value`.
    Notice that the number of elements in :attr:`x` must be equal to 1.

    Args:
        x (Tensor): A tensor that must always contain only one element, its data type supports float32, float64, int32 and int64.
        value(float, optional): The amount to increment the data of :attr:`x`. Default: 1.0.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the elementwise-incremented tensor with the same shape and data type as :attr:`x`.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.zeros(shape=[1], dtype='float32')
            counter = paddle.increment(data)
            # [1.]

    """
    if in_dygraph_mode():
        return _C_ops.increment(x, 'step', value)

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'increment')
    helper = LayerHelper("increment", **locals())
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={'step': float(value)})
    return x


def all(x, axis=None, keepdim=False, name=None):
    """
    Computes the the ``logical and`` of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the input data type should be `bool`.
        axis (int|list|tuple, optional): The dimensions along which the ``logical and`` is compute. If
            :attr:`None`, and all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: Results the ``logical and`` on the specified axis of input Tensor `x`,  it's data type is bool.

    Raises:
        ValueError: If the data type of `x` is not bool.
        TypeError: The type of :attr:`axis` must be int, list or tuple.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            
            # x is a bool Tensor with following elements:
            #    [[True, False]
            #     [True, True]]
            x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
            print(x)
            x = paddle.cast(x, 'bool')
            
            # out1 should be [False]
            out1 = paddle.all(x)  # [False]
            print(out1)
            
            # out2 should be [True, False]
            out2 = paddle.all(x, axis=0)  # [True, False]
            print(out2)
            
            # keep_dim=False, out3 should be [False, True], out.shape should be (2,)
            out3 = paddle.all(x, axis=-1)  # [False, True]
            print(out3)
            
            # keep_dim=True, out4 should be [[False], [True]], out.shape should be (2,1)
            out4 = paddle.all(x, axis=1, keepdim=True)
            out4 = paddle.cast(out4, 'int32')  # [[False], [True]]
            print(out4)
            
    """
    if axis is not None and not isinstance(axis, (list, tuple)):
        axis = [axis]

    if not axis:
        reduce_all_flag = True
    else:
        if len(axis) == len(x.shape):
            reduce_all_flag = True
        else:
            reduce_all_flag = False

    if in_dygraph_mode():
        axis = axis if axis != None and axis != [] else [0]
        return _C_ops.reduce_all(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag)

    attrs = {
        'dim': axis if axis != None and axis != [] and axis != () else [0],
        'keep_dim': keepdim,
        'reduce_all': reduce_all_flag
    }
    check_variable_and_dtype(x, 'x', ['bool'], 'all')


    check_type(axis, 'axis', (int, list, tuple, type(None)), 'all')

    helper = LayerHelper('all', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reduce_all',
        inputs={'X': x},
        outputs={'Out': out},
        attrs=attrs)
    return out


def any(x, axis=None, keepdim=False, name=None):
    """
    Computes the the ``logical or`` of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the input data type should be `bool`.
        axis (int|list|tuple, optional): The dimensions along which the ``logical or`` is compute. If
            :attr:`None`, and all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: Results the ``logical or`` on the specified axis of input Tensor `x`,  it's data type is bool.

    Raises:
        ValueError: If the data type of `x` is not bool.
        TypeError: The type of :attr:`axis` must be int, list or tuple.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            
            # x is a bool Tensor with following elements:
            #    [[True, False]
            #     [False, False]]
            x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
            print(x)
            x = paddle.cast(x, 'bool')
            
            # out1 should be [True]
            out1 = paddle.any(x)  # [True]
            print(out1)
            
            # out2 should be [True, True]
            out2 = paddle.any(x, axis=0)  # [True, True]
            print(out2)
            
            # keep_dim=False, out3 should be [True, True], out.shape should be (2,)
            out3 = paddle.any(x, axis=-1)  # [True, True]
            print(out3)
            
            # keep_dim=True, result should be [[True], [True]], out.shape should be (2,1)
            out4 = paddle.any(x, axis=1, keepdim=True)
            out4 = paddle.cast(out4, 'int32')  # [[True], [True]]
            print(out4)
            
    """
    if axis is not None and not isinstance(axis, (list, tuple)):
        axis = [axis]

    if not axis:
        reduce_all_flag = True
    else:
        if len(axis) == len(x.shape):
            reduce_all_flag = True
        else:
            reduce_all_flag = False

    if in_dygraph_mode():
        axis = axis if axis != None and axis != [] else [0]
        return _C_ops.reduce_any(x, 'dim', axis, 'keep_dim', keepdim,
                                       'reduce_all', reduce_all_flag)

    attrs = {
        'dim': axis if axis != None and axis != [] and axis != () else [0],
        'keep_dim': keepdim,
        'reduce_all': reduce_all_flag
    }

    check_variable_and_dtype(x, 'x', ['bool'], 'any')


    check_type(axis, 'axis', (int, list, tuple, type(None)), 'any')

    helper = LayerHelper('any', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reduce_any',
        inputs={'X': x},
        outputs={'Out': out},
        attrs=attrs)
    return out

def broadcast_shape(x_shape, y_shape):
    """
    The function returns the shape of doing operation with broadcasting on tensors of x_shape and y_shape, please refer to :ref:`user_guide_broadcasting` for more details.

    Args:
        x_shape (list[int]|tuple[int]): A shape of tensor.
        y_shape (list[int]|tuple[int]): A shape of tensor.
        

    Returns:
        list[int], the result shape.

    Examples:
        .. code-block:: python

            import paddle

            shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
            # [2, 3, 3]
            
            # shape = paddle.broadcast_shape([2, 1, 3], [3, 3, 1])
            # ValueError (terminated with error message).

    """

    return core.broadcast_shape(x_shape, y_shape)

def conj(x, name=None):
    r"""
    This function computes the conjugate of the Tensor elementwisely.

    Args:
        x (Tensor): The input tensor which hold the complex numbers. 
            Optional data types are: complex64, complex128, float32, float64, int32 or int64.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        out (Tensor): The conjugate of input. The shape and data type is the same with input.
            If the elements of tensor is real type such as float32, float64, int32 or int64, the out is the same with input.

    Examples:
        .. code-block:: python

          import paddle
          data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
          #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
          #       [[(1+1j), (2+2j), (3+3j)],
          #        [(4+4j), (5+5j), (6+6j)]])

          conj_data=paddle.conj(data)
          #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
          #       [[(1-1j), (2-2j), (3-3j)],
          #        [(4-4j), (5-5j), (6-6j)]])

    """
    if in_dygraph_mode():
        return _C_ops.conj(x)

    check_variable_and_dtype(x, "x", ['complex64', 'complex128', 'float32', 'float64', 'int32', 'int64'], 'conj')

    helper = LayerHelper('conj', **locals())
    out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())

    helper.append_op(type='conj', inputs={'X': x}, outputs={'Out': [out]})
    return out

def digamma(x, name=None):
    r"""
    Calculates the digamma of the given input tensor, element-wise.

    .. math::
        Out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Tensor, the digamma of the input Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
            res = paddle.digamma(data)
            print(res)
            # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[-0.57721591,  0.03648996],
            #        [ nan       ,  5.32286835]])
    """

    if in_dygraph_mode():
        return _C_ops.digamma(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'digamma')
    helper = LayerHelper('digamma', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='digamma', inputs={'X': x}, outputs={'Out': out})
    return out

def neg(x, name=None):
    """
    This function computes the negative of the Tensor elementwisely.

    Args:
        x (Tensor): Input of neg operator, an N-D Tensor, with data type float32, float64, int8, int16, int32, or int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The negative of input Tensor. The shape and data type are the same with input Tensor.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            out = paddle.neg(x)
            print(out)
            # [0.4 0.2 -0.1 -0.3]
    """

    return layers.scale(x, scale=-1.0, bias=0.0, bias_after_scale=True, act=None, name=name)

def atan2(x, y, name=None):
    r"""
    Element-wise arctangent of x/y with consideration of the quadrant.

    Equation:
        .. math::

            atan2(x,y)=\left\{\begin{matrix}
            & tan^{-1}(\frac{x}{y}) & y > 0 \\
            & tan^{-1}(\frac{x}{y}) + \pi & x>=0, y < 0 \\
            & tan^{-1}(\frac{x}{y}) - \pi & x<0, y < 0 \\
            & +\frac{\pi}{2} & x>0, y = 0 \\
            & -\frac{\pi}{2} & x<0, y = 0 \\
            &\text{undefined} & x=0, y = 0
            \end{matrix}\right.

    Args:
        x (Tensor): An N-D Tensor, the data type is int32, int64, float16, float32, float64.
        y (Tensor): An N-D Tensor, must have the same type as `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float64 when the input data type is int).

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-1, +1, +1, -1]).astype('float32')
            #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [-1,  1,  1, -1])

            y = paddle.to_tensor([-1, -1, +1, +1]).astype('float32')
            #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [-1,  -1,  1, 1])

            out = paddle.atan2(x, y)
            #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [-2.35619450,  2.35619450,  0.78539819, -0.78539819])

    """

    if in_dygraph_mode():
        return _C_ops.atan2(x, y)
    else:
        check_variable_and_dtype(x, 'x', ['int32', 'int64', 'float16', 'float32', 'float64'], 'atan2')
        check_variable_and_dtype(y, 'y', ['int32', 'int64', 'float16', 'float32', 'float64'], 'atan2')

        helper = LayerHelper('atan2', **locals())
        inputs = {'X1' : x, 'X2' : y}
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
                type='atan2', inputs=inputs, outputs={'Out': out})
        return out
