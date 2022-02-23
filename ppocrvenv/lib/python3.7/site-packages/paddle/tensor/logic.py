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

from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_type, check_variable_and_dtype
from ..fluid.layers.layer_function_generator import templatedoc
from .. import fluid
from ..fluid.framework import in_dygraph_mode, Variable
from ..framework import VarBase as Tensor

# TODO: define logic functions of a tensor  
from ..fluid.layers import is_empty  # noqa: F401
from ..fluid.layers import logical_and  # noqa: F401
from ..fluid.layers import logical_not  # noqa: F401
from ..fluid.layers import logical_or  # noqa: F401
from ..fluid.layers import logical_xor  # noqa: F401

from paddle.common_ops_import import core
from paddle import _C_ops
from paddle.tensor.creation import full

__all__ = []


def equal_all(x, y, name=None):
    """
    This OP returns the truth value of :math:`x == y`. True if two inputs have the same elements, False otherwise.

    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
        y(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: output Tensor, data type is bool, value is [False] or [True].

    Examples:
        .. code-block:: python

          import paddle

          x = paddle.to_tensor([1, 2, 3])
          y = paddle.to_tensor([1, 2, 3])
          z = paddle.to_tensor([1, 4, 3])
          result1 = paddle.equal_all(x, y)
          print(result1) # result1 = [True ]
          result2 = paddle.equal_all(x, z)
          print(result2) # result2 = [False ]
    """
    if in_dygraph_mode():
        return _C_ops.equal_all(x, y)

    helper = LayerHelper("equal_all", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    helper.append_op(
        type='equal_all', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [out]})
    return out


@templatedoc()
def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    """
    ${comment}

    Args:
        x(Tensor): ${input_comment}.
        y(Tensor): ${other_comment}.
        rtol(rtoltype, optional): The relative tolerance. Default: :math:`1e-5` .
        atol(atoltype, optional): The absolute tolerance. Default: :math:`1e-8` .
        equal_nan(equalnantype, optional): ${equal_nan_comment}.
        name (str, optional): Name for the operation. For more information, please
            refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: ${out_comment}.

    Raises:
        TypeError: The data type of ``x`` must be one of float32, float64.
        TypeError: The data type of ``y`` must be one of float32, float64.
        TypeError: The type of ``rtol`` must be float.
        TypeError: The type of ``atol`` must be float.
        TypeError: The type of ``equal_nan`` must be bool.

    Examples:
        .. code-block:: python

          import paddle

          x = paddle.to_tensor([10000., 1e-07])
          y = paddle.to_tensor([10000.1, 1e-08])
          result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                  equal_nan=False, name="ignore_nan")
          np_result1 = result1.numpy()
          # [False]
          result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                      equal_nan=True, name="equal_nan")
          np_result2 = result2.numpy()
          # [False]

          x = paddle.to_tensor([1.0, float('nan')])
          y = paddle.to_tensor([1.0, float('nan')])
          result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                  equal_nan=False, name="ignore_nan")
          np_result1 = result1.numpy()
          # [False]
          result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                      equal_nan=True, name="equal_nan")
          np_result2 = result2.numpy()
          # [True]
    """

    if in_dygraph_mode():
        return _C_ops.allclose(x, y, 'rtol',
                               str(rtol), 'atol',
                               str(atol), 'equal_nan', equal_nan)

    check_variable_and_dtype(x, "input", ['float32', 'float64'], 'allclose')
    check_variable_and_dtype(y, "input", ['float32', 'float64'], 'allclose')
    check_type(rtol, 'rtol', float, 'allclose')
    check_type(atol, 'atol', float, 'allclose')
    check_type(equal_nan, 'equal_nan', bool, 'allclose')

    helper = LayerHelper("allclose", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')

    inputs = {'Input': x, 'Other': y}
    outputs = {'Out': out}
    attrs = {'rtol': str(rtol), 'atol': str(atol), 'equal_nan': equal_nan}
    helper.append_op(
        type='allclose', inputs=inputs, outputs=outputs, attrs=attrs)

    return out


@templatedoc()
def equal(x, y, name=None):
    """

    This layer returns the truth value of :math:`x == y` elementwise.

    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
        y(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: output Tensor, it's shape is the same as the input's Tensor,
        and the data type is bool. The result of this op is stop_gradient. 

    Examples:
        .. code-block:: python

          import paddle

          x = paddle.to_tensor([1, 2, 3])
          y = paddle.to_tensor([1, 3, 2])
          result1 = paddle.equal(x, y)
          print(result1)  # result1 = [True False False]
    """
    if not isinstance(y, (int, bool, float, Variable)):
        raise TypeError(
            "Type of input args must be float, bool, int or Tensor, but received type {}".
            format(type(y)))
    if not isinstance(y, Variable):
        y = full(shape=[1], dtype=x.dtype, fill_value=y)

    if in_dygraph_mode():
        return _C_ops.equal(x, y)

    check_variable_and_dtype(
        x, "x", ["bool", "float32", "float64", "int32", "int64"], "equal")
    check_variable_and_dtype(
        y, "y", ["bool", "float32", "float64", "int32", "int64"], "equal")
    helper = LayerHelper("equal", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    out.stop_gradient = True

    helper.append_op(
        type='equal', inputs={'X': [x],
                              'Y': [y]}, outputs={'Out': [out]})
    return out


@templatedoc()
def greater_equal(x, y, name=None):
    """
    This OP returns the truth value of :math:`x >= y` elementwise, which is equivalent function to the overloaded operator `>=`.

    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.greater_equal(x, y)
            print(result1)  # result1 = [True False True]
    """
    if in_dygraph_mode():
        return _C_ops.greater_equal(x, y)

    check_variable_and_dtype(x, "x",
                             ["bool", "float32", "float64", "int32", "int64"],
                             "greater_equal")
    check_variable_and_dtype(y, "y",
                             ["bool", "float32", "float64", "int32", "int64"],
                             "greater_equal")
    helper = LayerHelper("greater_equal", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    out.stop_gradient = True

    helper.append_op(
        type='greater_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [out]})
    return out


@templatedoc()
def greater_than(x, y, name=None):
    """
    This OP returns the truth value of :math:`x > y` elementwise, which is equivalent function to the overloaded operator `>`.

    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x` .

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.greater_than(x, y)
            print(result1)  # result1 = [False False True]
    """
    if in_dygraph_mode():
        return _C_ops.greater_than(x, y)

    check_variable_and_dtype(x, "x",
                             ["bool", "float32", "float64", "int32", "int64"],
                             "greater_than")
    check_variable_and_dtype(y, "y",
                             ["bool", "float32", "float64", "int32", "int64"],
                             "greater_than")
    helper = LayerHelper("greater_than", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    out.stop_gradient = True

    helper.append_op(
        type='greater_than',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [out]})
    return out


@templatedoc()
def less_equal(x, y, name=None):
    """
    This OP returns the truth value of :math:`x <= y` elementwise, which is equivalent function to the overloaded operator `<=`.

    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.less_equal(x, y)
            print(result1)  # result1 = [True True False]
    """
    if in_dygraph_mode():
        return _C_ops.less_equal(x, y)

    check_variable_and_dtype(
        x, "x", ["bool", "float32", "float64", "int32", "int64"], "less_equal")
    check_variable_and_dtype(
        y, "y", ["bool", "float32", "float64", "int32", "int64"], "less_equal")
    helper = LayerHelper("less_equal", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    out.stop_gradient = True

    helper.append_op(
        type='less_equal', inputs={'X': [x],
                                   'Y': [y]}, outputs={'Out': [out]})
    return out


@templatedoc()
def less_than(x, y, name=None):
    """
    This OP returns the truth value of :math:`x < y` elementwise, which is equivalent function to the overloaded operator `<`.

    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.less_than(x, y)
            print(result1)  # result1 = [False True False]
    """
    if in_dygraph_mode():
        return _C_ops.less_than(x, y)

    check_variable_and_dtype(
        x, "x", ["bool", "float32", "float64", "int32", "int64"], "less_than")
    check_variable_and_dtype(
        y, "y", ["bool", "float32", "float64", "int32", "int64"], "less_than")
    helper = LayerHelper("less_than", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    out.stop_gradient = True

    helper.append_op(
        type='less_than', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [out]})
    return out


@templatedoc()
def not_equal(x, y, name=None):
    """
    This OP returns the truth value of :math:`x != y` elementwise, which is equivalent function to the overloaded operator `!=`.
    
    **NOTICE**: The output of this OP has no gradient.

    Args:
        x(Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3])
            y = paddle.to_tensor([1, 3, 2])
            result1 = paddle.not_equal(x, y)
            print(result1)  # result1 = [False True True]
    """
    if in_dygraph_mode():
        return _C_ops.not_equal(x, y)

    check_variable_and_dtype(
        x, "x", ["bool", "float32", "float64", "int32", "int64"], "not_equal")
    check_variable_and_dtype(
        y, "y", ["bool", "float32", "float64", "int32", "int64"], "not_equal")
    helper = LayerHelper("not_equal", **locals())
    out = helper.create_variable_for_type_inference(dtype='bool')
    out.stop_gradient = True

    helper.append_op(
        type='not_equal', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [out]})
    return out


def is_tensor(x):
    """

    This function tests whether input object is a paddle.Tensor.

    Args:
        x (object): Object to test.

    Returns:
        A boolean value. True if 'x' is a paddle.Tensor, otherwise False.

    Examples:
        .. code-block:: python

            import paddle

            input1 = paddle.rand(shape=[2, 3, 5], dtype='float32')
            check = paddle.is_tensor(input1)
            print(check)  #True

            input3 = [1, 4]
            check = paddle.is_tensor(input3)
            print(check)  #False
            
    """
    return isinstance(x, Tensor)


def _bitwise_op(op_name, x, y, out=None, name=None, binary_op=True):
    if in_dygraph_mode():
        op = getattr(_C_ops, op_name)
        if binary_op:
            return op(x, y)
        else:
            return op(x)

    check_variable_and_dtype(
        x, "x", ["bool", "uint8", "int8", "int16", "int32", "int64"], op_name)
    if y is not None:
        check_variable_and_dtype(
            y, "y", ["bool", "uint8", "int8", "int16", "int32", "int64"],
            op_name)
    if out is not None:
        check_type(out, "out", Variable, op_name)

    helper = LayerHelper(op_name, **locals())
    if binary_op:
        assert x.dtype == y.dtype

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if binary_op:
        helper.append_op(
            type=op_name, inputs={"X": x,
                                  "Y": y}, outputs={"Out": out})
    else:
        helper.append_op(type=op_name, inputs={"X": x}, outputs={"Out": out})

    return out


@templatedoc()
def bitwise_and(x, y, out=None, name=None):
    """
    ${comment}
    
    Args:
        x (Tensor): ${x_comment}
        y (Tensor): ${y_comment}
        out(Tensor): ${out_comment}

    Returns:
        Tensor: ${out_comment}
        
    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([-5, -1, 1])
            y = paddle.to_tensor([4,  2, -3])
            res = paddle.bitwise_and(x, y)
            print(res)  # [0, 2, 1]
    """
    return _bitwise_op(
        op_name="bitwise_and", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def bitwise_or(x, y, out=None, name=None):
    """
    ${comment}
    
    Args:
        x (Tensor): ${x_comment}
        y (Tensor): ${y_comment}
        out(Tensor): ${out_comment}

    Returns:
        Tensor: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([-5, -1, 1])
            y = paddle.to_tensor([4,  2, -3])
            res = paddle.bitwise_or(x, y)
            print(res)  # [-1, -1, -3]
    """
    return _bitwise_op(
        op_name="bitwise_or", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def bitwise_xor(x, y, out=None, name=None):
    """
    ${comment}

    Args:
        x (Tensor): ${x_comment}
        y (Tensor): ${y_comment}
        out(Tensor): ${out_comment}

    Returns:
        Tensor: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([-5, -1, 1])
            y = paddle.to_tensor([4,  2, -3])
            res = paddle.bitwise_xor(x, y)
            print(res) # [-1, -3, -4]
    """
    return _bitwise_op(
        op_name="bitwise_xor", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def bitwise_not(x, out=None, name=None):
    """
    ${comment}

    Args:
        x(Tensor):  ${x_comment}
        out(Tensor): ${out_comment}
    
    Returns:
        Tensor: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([-5, -1, 1])
            res = paddle.bitwise_not(x)
            print(res) # [4, 0, -2]
    """

    return _bitwise_op(
        op_name="bitwise_not", x=x, y=None, name=name, out=out, binary_op=False)
