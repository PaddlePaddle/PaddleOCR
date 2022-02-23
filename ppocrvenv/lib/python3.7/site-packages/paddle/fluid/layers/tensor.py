#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unlessf required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import math
import numpy
import warnings

from ..layer_helper import LayerHelper
from ..param_attr import ParamAttr
from ..initializer import Initializer
from ..framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator, device_guard
from ..framework import Variable
from ..initializer import Constant
from ..core import VarDesc
from .. import core
from .layer_function_generator import templatedoc
from . import utils
from ..data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from paddle.utils import deprecated

from .utils import check_shape
from paddle import _C_ops

__all__ = [
    'create_tensor',
    'create_parameter',
    'create_global_var',
    'cast',
    'tensor_array_to_tensor',
    'concat',
    'sums',
    'assign',
    'fill_constant_batch_size_like',
    'fill_constant',
    'argmin',
    'argmax',
    'argsort',
    'ones',
    'zeros',
    'reverse',
    'has_inf',
    'has_nan',
    'isfinite',
    'range',
    'linspace',
    'zeros_like',
    'ones_like',
    'diag',
    'eye',
    'triu',
]


def create_tensor(dtype, name=None, persistable=False):
    """
    Create a variable, which will hold a Tensor with data type dtype.

    Args:
        dtype(string|numpy.dtype): the data type of Tensor to be created, the
            data type is bool, float16, float32, float64, int8, int16, int32 and int64.
        name(string, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        persistable(bool): Set the persistable flag of the create tensor.
            default value is False.

    Returns:
        Variable: The tensor to be created according to dtype.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          tensor = fluid.layers.create_tensor(dtype='float32')
    """
    check_dtype(dtype, 'dtype', [
        'bool', 'float16', 'float32', 'float64', 'int8', 'int32', 'int32',
        'int64'
    ], 'create_tensor')
    helper = LayerHelper("create_tensor", **locals())
    return helper.create_variable(
        name=helper.name, dtype=dtype, persistable=persistable)


def create_parameter(shape,
                     dtype,
                     name=None,
                     attr=None,
                     is_bias=False,
                     default_initializer=None):
    """
	:api_attr: Static Graph

    This function creates a parameter. The parameter is a learnable variable, which can have
    gradient, and can be optimized.

    NOTE: this is a very low-level API. This API is useful when you create
    operator by your self. instead of using layers.

    Parameters:
        shape (list of int): Shape of the parameter
        dtype (str): Data type of the parameter
        name (str, optional): For detailed information, please refer to
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        attr (ParamAttr, optional): Attributes of the parameter
        is_bias (bool, optional): This can affect which default initializer is chosen
                       when default_initializer is None. If is_bias,
                       initializer.Constant(0.0) will be used. Otherwise,
                       Xavier() will be used.
        default_initializer (Initializer, optional): Initializer for the parameter

    Returns:
        The created parameter.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            W = paddle.static.create_parameter(shape=[784, 200], dtype='float32')
    """
    check_type(shape, 'shape', (list, tuple, numpy.ndarray), 'create_parameter')
    for item in shape:
        check_type(item, 'item of shape',
                   (int, numpy.uint8, numpy.int8, numpy.int16, numpy.int32,
                    numpy.int64), 'create_parameter')

    check_dtype(dtype, 'dtype', [
        'bool', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32',
        'int64', 'uint8'
    ], 'create_parameter')
    check_type(attr, 'attr', (type(None), ParamAttr), 'create_parameter')
    check_type(default_initializer, 'default_initializer',
               (type(None), Initializer), 'create_parameter')

    helper = LayerHelper("create_parameter", **locals())
    if attr is None:
        attr = ParamAttr(name=name)
    return helper.create_parameter(attr, shape,
                                   convert_dtype(dtype), is_bias,
                                   default_initializer)


def create_global_var(shape,
                      value,
                      dtype,
                      persistable=False,
                      force_cpu=False,
                      name=None):
    """
    This function creates a new tensor variable with value in the global block(block 0).

    Parameters:
        shape (list[int]|tuple[int]): Shape of the variable
        value (float): The value of the variable. The new created
                      variable will be filled with it.
        dtype (str): Data type of the variable
        persistable (bool, optional): If this variable is persistable.
                           Default: False
        force_cpu (bool, optional): Force this variable to be on CPU.
                         Default: False
        name (str, optional): For detailed information, please refer to
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Variable: The created Variable

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            var = paddle.static.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                                           persistable=True, force_cpu=True, name='new_var')
    """
    check_type(shape, 'shape', (list, tuple, numpy.ndarray),
               'create_global_var')
    for item in shape:
        check_type(item, 'item of shape',
                   (int, numpy.uint8, numpy.int8, numpy.int16, numpy.int32,
                    numpy.int64), 'create_global_var')

    check_dtype(dtype, 'dtype', [
        'bool',
        'float16',
        'float32',
        'float64',
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint16',
    ], 'create_global_var')

    helper = LayerHelper("global_var", **locals())
    var = helper.create_global_variable(
        dtype=dtype,
        shape=shape,
        persistable=persistable,
        name=name,
        stop_gradient=True)
    helper.set_variable_initializer(
        var, initializer=Constant(
            value=float(value), force_cpu=force_cpu))

    return var


def cast(x, dtype):
    """

    This OP takes in the Tensor :attr:`x` with :attr:`x.dtype` and casts it
    to the output with :attr:`dtype`. It's meaningless if the output dtype
    equals the input dtype, but it's fine if you do so.

    Args:
        x(Tensor): An input N-D Tensor with data type bool, float16,
            float32, float64, int32, int64, uint8.
        dtype(np.dtype|core.VarDesc.VarType|str): Data type of the output:
            bool, float16, float32, float64, int8, int32, int64, uint8.

    Returns:
        Tensor: A Tensor with the same shape as input's.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([2, 3, 4], 'float64')
            y = paddle.cast(x, 'uint8')
    """
    if in_dygraph_mode():
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)
        out = _C_ops.cast(x, 'in_dtype', x.dtype, 'out_dtype', dtype)
        return out

    check_variable_and_dtype(x, 'x', [
        'bool', 'float16', 'float32', 'float64', 'int32', 'int64', 'uint8',
        'uint16'
    ], 'cast')
    check_dtype(dtype, 'dtype', [
        'bool', 'float16', 'float32', 'float64', 'int8', 'int32', 'int64',
        'uint8', 'uint16'
    ], 'cast')

    helper = LayerHelper('cast', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=x.stop_gradient)
    helper.append_op(
        type='cast',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'in_dtype': x.dtype,
               'out_dtype': out.dtype})
    return out


def concat(input, axis=0, name=None):
    """
    This OP concatenates the input along the axis.

    Args:
        input(list|tuple|Tensor): ``input`` can be Tensor, Tensor list or Tensor tuple which is with data type
            bool, float16, float32, float64, int32, int64. All the Tensors in ``input`` must have the same data type. 
        axis(int|Tensor, optional): Specify the axis to operate on the input Tensors.
            It's a scalar with data type int or a Tensor with shape [1] and data type int32 or int64.
            The effective range is [-R, R), where R is Rank(x). When ``axis < 0``, it works the same way
            as ``axis+R``. Default is 0.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor with the same data type as ``input``.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[1, 2, 3],
                            [4, 5, 6]])
            in2 = np.array([[11, 12, 13],
                            [14, 15, 16]])
            in3 = np.array([[21, 22],
                            [23, 24]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                x2 = fluid.dygraph.to_variable(in2)
                x3 = fluid.dygraph.to_variable(in3)
                # When the axis is negative, the real axis is (axis + Rank(x)).
                # As follows, axis is -1, Rank(x) is 2, the real axis is 1
                out1 = fluid.layers.concat(input=[x1, x2, x3], axis=-1)
                out2 = fluid.layers.concat(input=[x1, x2], axis=0)
                print(out1.numpy())
                # [[ 1  2  3 11 12 13 21 22]
                #  [ 4  5  6 14 15 16 23 24]]
                print(out2.numpy())
                # [[ 1  2  3]
                #  [ 4  5  6]
                #  [11 12 13]
                #  [14 15 16]]
    """

    if in_dygraph_mode():
        if isinstance(axis, Variable):
            axis = axis.numpy()
            axis = axis.item(0)
        if not isinstance(input, Variable):
            input = [t for t in input if t.shape.count(0) == 0]
        return _C_ops.concat(input, 'axis', axis)

    check_type(input, 'input', (list, tuple, Variable), 'concat')
    if not isinstance(input, Variable):
        for id, x in enumerate(input):
            check_variable_and_dtype(
                x, 'input[' + str(id) + ']',
                ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
                'concat')
            if x.dtype != input[0].dtype:
                raise TypeError(
                    "All the Tensors in the input must have the same data type.")
    else:
        input = [input]
    check_type(axis, 'axis', (int, Variable), 'concat')

    if isinstance(axis, Variable):
        check_dtype(
            axis.dtype, 'axis', ['int32', 'int64'], 'concat',
            "The data type of axis must be int32 or int64 when axis is a Tensor")

    helper = LayerHelper('concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())

    if input[0].desc.type() == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        # NOTE(liym27): Don't remove this if branch!
        # This feature is supported for Dynamic-to-Static, because after transformed, the type of inputs[0]
        # is LOD_TENSOR_ARRAY in some scenarios. And this feature can be used in static mode.

        assert len(input) == 1, "If the elements of 'input' in concat are Variable(LoDTensorArray), " \
                "number of the elements must be 1, but received %s." % len(input)
        out_index = helper.create_variable_for_type_inference(dtype="int32")
        helper.append_op(
            type='tensor_array_to_tensor',
            inputs={'X': input[0]},
            outputs={'Out': [out],
                     'OutIndex': [out_index]},
            attrs={'axis': axis,
                   'use_stack': False})
    else:
        inputs = {'X': input}
        attrs = {}
        if isinstance(axis, Variable):
            axis.stop_gradient = True
            inputs['AxisTensor'] = axis
        else:
            attrs['axis'] = axis

        helper.append_op(
            type='concat', inputs=inputs, outputs={'Out': [out]}, attrs=attrs)
    return out


def tensor_array_to_tensor(input, axis=1, name=None, use_stack=False):
    r"""
    This function concatenates or stacks all tensors in the input LoDTensorArray
    along the axis mentioned and returns that as the output.

    For Example:

    .. code-block:: text

        Case 1:

            Given:

                input.data = {[[0.6, 0.1, 0.3],
                               [0.5, 0.3, 0.2]],
                              [[1.3],
                               [1.8]],
                              [[2.3, 2.1],
                               [2.5, 2.4]]}

                axis = 1, use_stack = False

            Then:

                output.data = [[0.6, 0.1, 0.3, 1.3, 2.3, 2.1],
                               [0.5, 0.3, 0.2, 1.8, 2.5, 2.4]]

                output_index.data = [3, 1, 2]

        Case 2:

            Given:

                input.data = {[[0.6, 0.1],
                               [0.5, 0.3]],
                              [[0.3, 1.3],
                               [0.2, 1.8]],
                              [[2.3, 2.1],
                               [2.5, 2.4]]}

                axis = 1, use_stack = True

            Then:

                output.data = [[[0.6, 0.1]
                                [0.3, 1.3]
                                [2.3, 2.1],
                               [[0.5, 0.3]
                                [0.2, 1.8]
                                [2.5, 2.4]]]

                output_index.data = [2, 2, 2]

    Args:
        input(Variable): A LodTensorArray variable.
        axis(int): The axis along which the tensors in attr::`input` will be
            concatenated or stacked.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.
        use_stack(bool): Act as concat_op or stack_op. For stack mode, all
            tensors in the tensor array must have the same shape.

    Returns:
        Variable: The concatenated or stacked tensor variable.
        Variable: A 1-D tensor variable with int32 data type. The data in this \
            tensor contains all input including tensors' sizes along the axis.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            x0 = fluid.layers.assign(np.random.rand(2, 2).astype("float32"))
            x1 = fluid.layers.assign(np.random.rand(2, 2).astype("float32"))
            i = fluid.layers.fill_constant(shape=[1], dtype="int64", value=0)
            array = fluid.layers.create_array(dtype='float32')
            fluid.layers.array_write(x0, i, array)
            fluid.layers.array_write(x1, i + 1, array)
            output, output_index = fluid.layers.tensor_array_to_tensor(input=array)
    """
    if in_dygraph_mode():
        assert isinstance(
            input, list), "The 'input' in tensor_array_to_tensor must be list"
        from .nn import stack, concat
        from ..dygraph import to_variable
        op = stack if use_stack else concat
        res = op(input, axis=axis)
        sizes = to_variable(
            numpy.array(list(map(lambda x: int(x.shape[axis]), input))))
        return res, sizes

    check_type(input, 'input', (list, Variable), 'tensor_array_to_tensor')
    if isinstance(input, list):
        for i, input_x in enumerate(input):
            check_type(input_x, 'input[' + str(i) + ']', Variable,
                       'tensor_array_to_tensor')
    helper = LayerHelper('tensor_array_to_tensor', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    out_index = helper.create_variable_for_type_inference(dtype="int32")
    helper.append_op(
        type='tensor_array_to_tensor',
        inputs={'X': input},
        outputs={'Out': [out],
                 'OutIndex': [out_index]},
        attrs={'axis': axis,
               'use_stack': use_stack})
    return out, out_index


def sums(input, out=None):
    r"""
    This function computes the sum of multiple input Tensors elementwisely.

    - Case 1, sum of 3 Tensors

    .. code-block:: text

        # Input Tensors
        x0.shape = [2, 3]
        x0.data = [[1., 2., 3.],
                   [4., 5., 6.]]
        x1.shape = [2, 3]
        x1.data = [[10., 20., 30.],
                   [40., 50., 60.]]
        x2.shape = [2, 3]
        x2.data = [[100., 200., 300.],
                   [400., 500., 600.]]

        # Output Tensor
        out.shape = [2, 3]
        out.data = [[111., 222., 333.],
                    [444., 555., 666.]]

    Args:
        input (list): A list of Variables which hold input Tensors with the same
            data type and shape. Optional data types are: float32, float64, int32, int64.
        out (Variable, optional): Output Tensor. It can be any existing Variable.
            The default value is None, then a new Variable will be created and returned.

    Returns:
        Variable: The sum of inputs. The shape and data type is the same with input. \
            If :code:`out` is not None, the returned value is :code:`out` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x0 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=1)
            x1 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=2)
            x2 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=3)
            x3 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=0)

            # Sum of multiple Tensors, the result is stored to a new Variable sum0 (sum0=x0+x1+x2, the value is [[6, ..., 6], ..., [6, ..., 6]])
            sum0 = fluid.layers.sums(input=[x0, x1, x2])

            # Sum of multiple Tensors, sum1 and x3 represents the same Variable (x3=x0+x1+x2, the value is [[6, ..., 6], ..., [6, ..., 6]])
            sum1 = fluid.layers.sums(input=[x0, x1, x2], out=x3)
    """
    check_type(input, 'input', (Variable, tuple, list), 'sums')
    if isinstance(input, list) or isinstance(input, tuple):
        for input_section in input:
            check_variable_and_dtype(input_section, "input", \
                    ['float16', 'float32', 'float64', 'int32', 'int64'], 'sums')
    else:
        check_variable_and_dtype(input, "input", \
                ['float16', 'float32', 'float64', 'int32', 'int64'], 'sums')

    helper = LayerHelper('sum', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    else:
        check_variable_and_dtype(
            out, "out", ['float32', 'float64', 'int32', 'int64'], 'sums')

    helper.append_op(
        type='sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})
    return out


def assign(input, output=None):
    """

    The OP copies the :attr:`input` to the :attr:`output`.

    Parameters:
        input (Tensor|numpy.ndarray|list|tuple|scalar): A tensor, numpy ndarray, tuple/list of scalar,
            or scalar. Its data type supports float16, float32, float64, int32, int64, and bool.
            Note: the float64 data will be converted to float32 because of current platform protobuf
            data limitation.
        output (Tensor, optional): A tensor. If :attr:`output` is None, a new tensor will
            be created as :attr:`output`. Default: None.

    Returns:
        Tensor: A tensor with the same shape, data type and value as :attr:`input`.

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np
          data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          array = np.array([[1, 1],
                            [3, 4],
                            [1, 3]]).astype(np.int64)
          result1 = paddle.zeros(shape=[3, 3], dtype='float32')
          paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
          result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
          result3 = paddle.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    """
    helper = LayerHelper('assign', **locals())
    check_type(input, 'input', (Variable, numpy.ndarray, list, tuple, float,
                                int, bool), 'assign')
    is_inplace = True if output is not None else False

    if numpy.isscalar(input) and not isinstance(input, str):
        input = numpy.array([input])
    elif isinstance(input, (list, tuple)):
        input = numpy.array(input)
    # NOTE(Aurelius84): Why we judge core.VarBase?
    # In case of @to_static, a VarBase can be as input of `assign`,
    # but in_dygraph_mode()==False under @to_static, which means
    # isinstance(VarBase, Variable) == False. It will cause return None
    # after this api.
    if isinstance(input, (Variable, core.VarBase)):
        check_dtype(input.dtype, 'input', [
            'float16', 'uint16', 'float32', 'float64', 'int32', 'int64',
            'uint8', 'bool'
        ], 'assign', '(When the type of input in assign is Variable.)')
        if output is None:
            output = helper.create_variable_for_type_inference(
                dtype=input.dtype)
        helper.append_op(
            type='assign', inputs={'X': [input]}, outputs={'Out': [output]})
    elif isinstance(input, numpy.ndarray):
        dtype = convert_np_dtype_to_dtype_(input.dtype)
        if dtype == VarDesc.VarType.FP64:
            # Setting FP64 numpy data is not supported in Paddle, so we
            # use FP32 here
            warnings.warn(
                "paddle.assign doesn't support float64 input now due "
                "to current platform protobuf data limitation, we convert "
                "it to float32")
            dtype = VarDesc.VarType.FP32
        if dtype == VarDesc.VarType.BOOL:
            value_name = "bool_values"
            values = [bool(v) for v in input.flat]
        elif dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in input.flat]
        elif dtype == VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in input.flat]
        elif dtype == VarDesc.VarType.INT64:
            value_name = "int64_values"
            values = [int(v) for v in input.flat]
        else:
            raise TypeError(
                "When the type of 'input' in assign is numpy.ndarray, "
                "the data type of 'input' must be bool, float32, int32 or int64, but "
                "received %s." % convert_dtype(dtype))
        if input.size > 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")
        if output is None:
            output = helper.create_variable_for_type_inference(
                dtype=input.dtype)
        helper.append_op(
            type='assign_value',
            outputs={'Out': [output]},
            attrs={
                'dtype': dtype,
                'shape': list(input.shape),
                value_name: values
            })

    if is_inplace and in_dygraph_mode():
        output._bump_inplace_version()

    return output


def fill_constant(shape, dtype, value, force_cpu=False, out=None, name=None):
    """

    This OP creates a Tensor with specified `shape` and `dtype`, and
    initializes it with a constant specified by `value`.

    The attribute `stop_gradient` of the created Tensor is set to True.

    Args:
        shape(list|tuple|Tensor): Shape of the output Tensor, the data type of ``shape`` is int32 or int64.
            If ``shape`` is a list or tuple, the elements of it should be integers or Tensors with shape [1].
            If ``shape`` is an Tensor, it should be an 1-D Tensor with date type int32 or int64.
        dtype(np.dtype|str): Data type of the output Tensor which can
            be float16, float32, float64, uint8, int16, int32, int64.
        value(bool|float|int|Tensor): The constant value used to initialize 
            the Tensor to be created. If ``value`` is an Tensor, it should be an 1-D Tensor.
        force_cpu(bool, optional): data should be on CPU if it's true, default value is False.
        out(Tensor, optional): Optional output which can be any created 
            Tensor that meets the requirements to store the result of operation.
            if ``out`` is None, a new Tensor will be create to store the result.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Tensor which is created according to shape and dtype.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          # attr shape is a list which doesn't contain  Tensor.
          data1 = fluid.layers.fill_constant(shape=[2,1], value=0, dtype='int64') # data1=[[0],[0]]
          data2 = fluid.layers.fill_constant(shape=[2,1], value=5, dtype='int64', out=data1)
          # data1=[[5], [5]] data2=[[5], [5]]

          # attr shape is a list which contains Tensor.
          positive_2 = fluid.layers.fill_constant([1], "int32", 2)
          data3 = fluid.layers.fill_constant(shape=[1, positive_2], dtype='float32', value=1.5) # data3=[[1.5, 1.5]]

          # attr shape is a Tensor.
          shape = fluid.layers.fill_constant([2], "int32", 2) # shape=[2,2]
          data4 = fluid.layers.fill_constant(shape=shape, dtype='bool', value=True) # data4=[[True,True],[True,True]]
          
          # attr value is a Tensor.
          val = fluid.layers.fill_constant([1], "float32", 2.0) # val=[2.0]
          data5 = fluid.layers.fill_constant(shape=[2,1], value=val, dtype='float32') #data5=[[2.0],[2.0]]
    """

    attrs = {'force_cpu': force_cpu}
    dtype = convert_dtype(dtype)
    if not isinstance(value, Variable):
        if dtype in ['uint8', 'int16', 'int32', 'int64']:
            attrs['str_value'] = str(int(value))
            attrs['value'] = int(value)
        else:
            attrs['str_value'] = str(float(value))
            attrs['value'] = float(value)

    if in_dygraph_mode():
        shape = utils.convert_shape_to_list(shape)
        if out is None:
            out = _varbase_creator(dtype=dtype)

        if isinstance(value, Variable):
            if dtype in ['uint8', 'int16', 'int32', 'int64']:
                attrs['str_value'] = str(int(value.numpy().item(0)))
            else:
                attrs['str_value'] = str(float(value.numpy().item(0)))

        _C_ops.fill_constant(out, 'value',
                             float(value), 'force_cpu', force_cpu, 'dtype',
                             out.dtype, 'str_value', attrs['str_value'],
                             'shape', shape)
        out.stop_gradient = True
        return out

    helper = LayerHelper("fill_constant", **locals())
    inputs = {}
    if isinstance(value, Variable):
        if convert_dtype(value.dtype) != dtype:
            value = cast(value, dtype)
        inputs['ValueTensor'] = value

    check_shape(shape)
    check_dtype(dtype, 'dtype', [
        'bool', 'float16', 'float32', 'float64', 'uint8', 'int16', 'int32',
        'int64'
    ], 'fill_constant')
    check_type(shape, 'shape', (Variable, list, tuple), 'fill_constant')

    if out is not None:
        check_variable_and_dtype(out, 'out', [convert_dtype(dtype)],
                                 'fill_constant')

    helper = LayerHelper("fill_constant", **locals())
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='fill_constant')

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    attrs['dtype'] = out.dtype
    helper.append_op(
        type='fill_constant',
        inputs=inputs,
        outputs={'Out': [out]},
        attrs=attrs,
        stop_gradient=True)
    out.stop_gradient = True
    return out


@deprecated(since='1.8.0', update_to="paddle.fluid.layers.fill_constant")
@templatedoc()
def fill_constant_batch_size_like(input,
                                  shape,
                                  dtype,
                                  value,
                                  input_dim_idx=0,
                                  output_dim_idx=0,
                                  force_cpu=False):
    """
    This OP creates a Tesnor according the shape and dtype, and initializes the
    Tensor with the constants provided in ``value``. When the input is LoDTensor
    and the input_dim_idx is 0, the output_dim_idx dimension is set to the value
    of the batch_size input by the input, the Stop_gradient attribute of the created
    Tensor is False by default.

    Args:
        input(Variable): Tensor which data type is float32, float64, int32 and int64.
        shape(list): The shape of Tensor to be created, Tensor's shape may be changed
            according the input.
        dtype(np.dtype|core.VarDesc.VarType|str): The data type of created Tensor which
            can be float32, float64, int32, int64.
        value(float|int): The constant value used to initialize the Tensor to be created. 
        input_dim_idx(int): When the value is 0 and the input is LoDTensor, the output_dim_idx
            dimension of the created Tensor is set to the batch_size value of input.
            The default value is 0.
        output_dim_idx(int): Used to specify which dimension of Tensor is created to be set
            the value of batch_size of input Tensor. The default value is 0.
        force_cpu(bool): data should be on CPU if it's true, default value is False.

    Returns:
        Variable: Tensor which will be created according to dtype.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             like = fluid.layers.fill_constant(shape=[1,2], value=10, dtype='int64') #like=[[10, 10]]
             data = fluid.layers.fill_constant_batch_size_like(
                    input=like, shape=[1], value=0, dtype='int64') #like=[[10, 10]] data=[0]

    """
    helper = LayerHelper("fill_constant_batch_size_like", **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)
    attrs = {
        'shape': shape,
        'dtype': out.dtype,
        'value': float(value),
        'input_dim_idx': input_dim_idx,
        'output_dim_idx': output_dim_idx,
        'force_cpu': force_cpu
    }
    if convert_dtype(dtype) in ['int64', 'int32']:
        attrs['str_value'] = str(int(value))
    else:
        attrs['str_value'] = str(float(value))
    helper.append_op(
        type='fill_constant_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': [out]},
        attrs=attrs)
    out.stop_gradient = True
    return out


def argmin(x, axis=0):
    """
	:alias_main: paddle.argmin
	:alias: paddle.argmin,paddle.tensor.argmin,paddle.tensor.search.argmin
	:old_api: paddle.fluid.layers.argmin

    **argmin**

    This OP computes the indices of the min elements of the input tensor's
    element along the provided axis.

    Args:
        x(Variable): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.

    Returns:
        Variable: A Tensor with data type int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[[5,8,9,5],
                            [0,0,1,7],
                            [6,9,2,4]],
                            [[5,2,4,2],
                            [4,7,7,9],
                            [1,7,0,6]]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.argmin(x=x, axis=-1)
                out2 = fluid.layers.argmin(x=x, axis=0)
                out3 = fluid.layers.argmin(x=x, axis=1)
                out4 = fluid.layers.argmin(x=x, axis=2)
                print(out1.numpy())
                # [[0 0 2]
                #  [1 0 2]]
                print(out2.numpy())
                # [[0 1 1 1]
                #  [0 0 0 0]
                #  [1 1 1 0]]
                print(out3.numpy())
                # [[1 1 1 2]
                #  [2 0 2 0]]
                print(out4.numpy())
                # [[0 0 2]
                #  [1 0 2]]
    """
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'uint8', 'int16', 'int32', 'int64'],
        'argmin')
    helper = LayerHelper("arg_min", **locals())
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_min',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    out.stop_gradient = True
    return out


def argmax(x, axis=0):
    """
    **argmax**

    This OP computes the indices of the max elements of the input tensor's
    element along the provided axis.

    Args:
        x(Variable): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.

    Returns:
        Variable: A Tensor with data type int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[[5,8,9,5],
                            [0,0,1,7],
                            [6,9,2,4]],
                            [[5,2,4,2],
                            [4,7,7,9],
                            [1,7,0,6]]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.argmax(x=x, axis=-1)
                out2 = fluid.layers.argmax(x=x, axis=0)
                out3 = fluid.layers.argmax(x=x, axis=1)
                out4 = fluid.layers.argmax(x=x, axis=2)
                print(out1.numpy())
                # [[2 3 1]
                #  [0 3 1]]
                print(out2.numpy())
                # [[0 0 0 0]
                #  [1 1 1 1]
                #  [0 0 0 1]]
                print(out3.numpy())
                # [[2 2 0 1]
                #  [0 1 1 1]]
                print(out4.numpy())
                # [[2 3 1]
                #  [0 3 1]]
    """
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'uint8', 'int16', 'int32', 'int64'],
        'argmax')
    helper = LayerHelper("arg_max", **locals())
    out = helper.create_variable_for_type_inference(VarDesc.VarType.INT64)
    helper.append_op(
        type='arg_max',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    out.stop_gradient = True
    return out


def argsort(input, axis=-1, descending=False, name=None):
    """
	:alias_main: paddle.argsort
	:alias: paddle.argsort,paddle.tensor.argsort,paddle.tensor.search.argsort
	:old_api: paddle.fluid.layers.argsort

    This OP sorts the input along the given axis, and returns sorted output
    data Varibale and its corresponding index Variable with the same shape as
    :attr:`input`.

    Args:
        input(Variable): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is 0.
        descending(bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        tuple: A tuple of sorted data Variable(with the same shape and data
        type as input) and the sorted indices(with the same shape as input's
        and with data type int64).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            in1 = np.array([[[5,8,9,5],
                            [0,0,1,7],
                            [6,9,2,4]],
                            [[5,2,4,2],
                            [4,7,7,9],
                            [1,7,0,6]]]).astype(np.float32)
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.argsort(input=x, axis=-1)
                out2 = fluid.layers.argsort(input=x, axis=0)
                out3 = fluid.layers.argsort(input=x, axis=1)
                print(out1[0].numpy())
                # [[[5. 5. 8. 9.]
                #   [0. 0. 1. 7.]
                #   [2. 4. 6. 9.]]
                #  [[2. 2. 4. 5.]
                #   [4. 7. 7. 9.]
                #   [0. 1. 6. 7.]]]
                print(out1[1].numpy())
                # [[[0 3 1 2]
                #   [0 1 2 3]
                #   [2 3 0 1]]
                #  [[1 3 2 0]
                #   [0 1 2 3]
                #   [2 0 3 1]]]
                print(out2[0].numpy())
                # [[[5. 2. 4. 2.]
                #   [0. 0. 1. 7.]
                #   [1. 7. 0. 4.]]
                #  [[5. 8. 9. 5.]
                #   [4. 7. 7. 9.]
                #   [6. 9. 2. 6.]]]
                print(out3[0].numpy())
                # [[[0. 0. 1. 4.]
                #   [5. 8. 2. 5.]
                #   [6. 9. 9. 7.]]
                #  [[1. 2. 0. 2.]
                #   [4. 7. 4. 6.]
                #   [5. 7. 7. 9.]]]
    """
    check_variable_and_dtype(
        input, 'input',
        ['float32', 'float64', 'int16', 'int32', 'int64', 'uint8'], 'argsort')
    helper = LayerHelper("argsort", **locals())
    out = helper.create_variable_for_type_inference(
        dtype=input.dtype, stop_gradient=True)
    ids = helper.create_variable_for_type_inference(
        VarDesc.VarType.INT64, stop_gradient=True)
    helper.append_op(
        type='argsort',
        inputs={'X': input},
        outputs={'Out': out,
                 'Indices': ids},
        attrs={'axis': axis,
               'descending': descending})
    return out, ids


def ones(shape, dtype, force_cpu=False):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 1.
    Its :attr:`stop_gradient` will be set to True to stop gradient computation.

    Parameters:
        shape(tuple|list|Tensor): Shape of output Tensor, the data type of shape is int32 or int64.
        dtype (np.dtype|str): Data type of output Tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        force_cpu (bool, optional): Whether force to store the output Tensor in CPU memory.
            If :attr:`force_cpu` is False, the output Tensor will be stored in running device memory.
            Default: False.

    Returns:
        Tensor: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 1.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data0 = fluid.layers.ones(shape=[2, 4], dtype='float32') # [[1., 1., 1., 1.], [1., 1., 1., 1.]]
          
          # shape is a Tensor
          shape = fluid.layers.fill_constant(shape=[2], dtype='int32', value=2)
          data1 = fluid.layers.ones(shape=shape, dtype='int32') #[[1, 1], [1, 1]]
    """
    return fill_constant(value=1.0, **locals())


def zeros(shape, dtype, force_cpu=False, name=None):
    """
    The OP creates a tensor of specified :attr:`shape` and :attr:`dtype`, and fills it with 0.
    Its :attr:`stop_gradient` will be set to True to stop gradient computation.

    Parameters:
        shape(tuple|list|Tensor): Shape of output Tensor, the data type of ``shape`` is int32 or int64.
        dtype (np.dtype|str): Data type of output Tensor, it supports
            bool, float16, float32, float64, int32 and int64.
        force_cpu (bool, optional): Whether force to store the output Tensor in CPU memory.
            If :attr:`force_cpu` is False, the output Tensor will be stored in running device memory.
            Default: False.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A tensor of data type :attr:`dtype` with shape :attr:`shape` and all elements set to 0.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]
          
          # shape is a Tensor
          shape = fluid.layers.fill_constant(shape=[2], dtype='int32', value=2)
          data1 = fluid.layers.zeros(shape=shape, dtype='int32') #[[0, 0], [0, 0]]
    """
    return fill_constant(value=0.0, **locals())


def reverse(x, axis):
    """
	:alias_main: paddle.reverse
	:alias: paddle.reverse,paddle.tensor.reverse,paddle.tensor.manipulation.reverse
	:old_api: paddle.fluid.layers.reverse

    The OP reverses the tensor :attr:`x` along the given :attr:`axis`.

    .. code-block:: text

        Case 1:

            Given a LoDTensor:
                x = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
                axis = [0, 1]

            Then:
                output = [[8, 7, 6], [5, 4, 3], [2, 1, 0]]

        Case 2:

            Given a LoDTensorArray:
                x = {[[0, 1], [2, 3]],
                     [[4, 5, 6]],
                     [[7],[8], [9]]}
                axis = 0

            Then:
                output = {[[7],[8], [9]],
                          [[4, 5, 6]],
                          [[0, 1], [2, 3]]}

    Parameters:
        x (Variable): A tensor or LoDTensorArray to be reversed, its data type supports bool, float32, float64, int32, int64 and uint8.
                      If input is a LoDTensorArray, returns a new reversed LoDTensorArray without changing the internal order of each inner tensor.
        axis (int|tuple|list): A dimension or a set of dimensions of :attr:`x` to reverse. Must be
            in the range [-rank( :attr:`x` ), rank( :attr:`x` )). If it is a tuple or a list, reversing
            will be apply on each axis in the tuple or list. If input is a LoDTensorArray, the value of axis shall be 0, or a
            list [0] or tuple (0, ) with shape [1].

    Returns:
        Variable: The reversed tensor with the same shape and data type as :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          data = fluid.layers.assign(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype='float32')) # [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]
          result1 = fluid.layers.reverse(data, 0) # [[6., 7., 8.], [3., 4., 5.], [0., 1., 2.]]
          result2 = fluid.layers.reverse(data, [0, 1]) # [[8., 7., 6.], [5., 4., 3.], [2., 1., 0.]]

          # example of LoDTensorArray
          data1 = fluid.layers.assign(np.array([[0, 1, 2]], dtype='float32'))
          data2 = fluid.layers.assign(np.array([[3, 4, 5]], dtype='float32'))
          tensor_array = fluid.layers.create_array(dtype='float32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
          fluid.layers.array_write(data1, i, tensor_array)
          fluid.layers.array_write(data2, i+1, tensor_array)

          reversed_tensor_array = fluid.layers.reverse(tensor_array, 0) # {[[3, 4, 5]], [[0, 1, 2]]}
    """
    check_variable_and_dtype(
        x, 'x', ('float32', 'float64', 'int32', 'int64', 'uint8'), 'reverse')
    check_type(axis, 'axis', (int, tuple, list), 'reverse')
    if isinstance(axis, int):
        axis = [axis]
    helper = LayerHelper("reverse", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reverse',
        inputs={'X': x},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def save(x, file_path, overwrite=True):
    """
    Saves a variable as a file.

    Args:
        x(variable): The Tensor/LoDTensor to be saved.
        file_path(str): The file path where the variable will be saved.
        overwrite(bool): Whether or not cover the given file when it has already
            existed. If it's set 'False' and the file is existed, a runtime
            error will be thrown.
    """
    helper = LayerHelper("save", **locals())
    helper.append_op(
        type="save",
        inputs={"input": x},
        outputs={},
        args={"file_path": file_path,
              "overwrite": overwrite})


def save_combine(x, file_path, overwrite=True):
    """
    Saves a list of variables into a single file.

    Args:
        x(list): A list of Tensor/LoDTensor variables to be saved together in
                 a single file.
        file_path(str): The file path where variables will be saved.
        overwrite(bool): Whether or not cover the given file when it has already
            existed. If it's set 'False' and the file is existed, a runtime
            error will be thrown.

    Returns:
        There is no return value.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            v1 = fluid.layers.data(name="data",
                                   shape=(4, 6),
                                   dtype="float32")
            v2 = fluid.layers.data(name="data",
                                   shape=(6, 8, 4),
                                   dtype="float32")
            normed = fluid.layers.save_combine([v1, v2], file_path="output")
    """
    helper = LayerHelper("save_combine", **locals())
    helper.append_op(
        type="save_combine",
        inputs={"input": x},
        outputs={},
        args={"file_path": file_path,
              "overwrite": overwrite})


def load_combine(out, file_path):
    """
    Loads a list of variable from a single file.

    Args:
        out(list): The list of variables to be read from the disk file.
        file_path(str): The path of the disk file.
    """
    helper = LayerHelper("load_combine", **locals())
    helper.append_op(
        type="load_combine",
        inputs={},
        output={"Out": out},
        args={"file_path": file_path})


def has_inf(x):
    """
    Test if any of x contains an infinity number

    Args:
       x (Tensor): The Tensor to be checked.

    Returns:
       Tensor: The tensor storing the output, only a bool value, indicating that whether there is infinity number in x or not.
    
    Examples:
        .. code-block:: python
          
          import paddle
          data = paddle.randn(shape=[4, 32, 32], dtype="float32")
          res = paddle.fluid.layers.has_inf(data)
          # [False]

    """
    if in_dygraph_mode():
        return _C_ops.isinf(x)

    check_type(x, 'x', (Variable), 'has_inf')
    helper = LayerHelper("isinf", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="isinf", inputs={"X": x}, outputs={"Out": out})
    return out


def has_nan(x):
    """
    Test if any of x contains a NAN

    Args:
       x (Tensor): The Tensor to be checked.

    Returns:
       Tensor: The tensor variable storing the output, only a bool value, indicating that whether there is NAN in x or not.
    
    Examples:
        .. code-block:: python
    
          import paddle
          data = paddle.randn(shape=[2,3], dtype="float32")
          res = paddle.fluid.layers.has_nan(data)
          # [False]

    """
    if in_dygraph_mode():
        return _C_ops.isnan(x)

    check_type(x, 'x', (Variable), 'has_nan')
    helper = LayerHelper("isnan", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="isnan", inputs={"X": x}, outputs={"Out": out})
    return out


def isfinite(x):
    """

    Test if any of x contains an infinity/NAN number. If all the elements are finite,
    returns true, else false.

    Args:
        x(Tensor): The Tensor to be checked.

    Returns:
        Tensor: The tensor storing the output, contains a bool value.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.rand(shape=[4, 6], dtype='float32')
            y = paddle.fluid.layers.isfinite(x)
            print(y)

    """
    check_variable_and_dtype(x, "x", ["float32", "float64", "int32", "int64"],
                             "isfinite")
    helper = LayerHelper("isfinite", **locals())

    out = helper.create_variable_for_type_inference(dtype='bool')
    helper.append_op(type="isfinite", inputs={"X": x}, outputs={"Out": out})
    return out


def range(start, end, step, dtype, name=None):
    """
    This OP returns a 1-D Tensor with spaced values within a given interval.

    Values are generated into the half-open interval [``start``, ``end``) with
    the ``step``. (the interval including ``start`` but excluding ``end``).

    If ``dtype`` is float32 or float64, we advise adding a small epsilon to
    ``end`` to avoid floating point rounding errors when comparing against ``end``.

    Parameters:
        start(float|int|Tensor): Start of interval. The interval includes this
            value. If ``start`` is a Tensor, it is a 1-D Tensor with shape [1],
            with data type int32, int64, float32, float64.
        end(float|int|Tensor): End of interval. The interval does not include
            this value. If ``end`` is a Tensor, it is a 1-D Tensor with shape
            [1], with data type int32, int64, float32, float64.
        step(float|int|Tensor): Spacing between values. For any out, it is
            the istance between two adjacent values, out[i+1] - out[i]. If
            ``step`` is a Tensor, it is a 1-D Tensor with shape [1], with data
            type int32, int64, float32, float64.
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of the
            output tensor. Supported data types: int32, int64, float32, float64.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns: 
        Tensor: A 1-D Tensor with values from the interval [``start``, ``end``)
            taken with common difference ``step`` beginning from ``start``. Its
            data type is set by ``dtype``.

    Raises:
        TypeError: If ``dtype`` is not int32, int64, float32, float64.

    examples:

        .. code-block:: python

            import paddle.fluid as fluid

            out1 = fluid.layers.range(0, 10, 2, 'int32')
            # [0, 2, 4, 6, 8]

            start_var = fluid.layers.fill_constant([1], 'int64', 3)
            out2 = fluid.layers.range(start_var, 7, 1, 'int64')
            # [3, 4, 5, 6]

    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if not isinstance(start, Variable):
        with device_guard("cpu"):
            start = fill_constant([1], dtype, start, force_cpu=True)
    elif start.dtype != dtype:
        start = cast(start, dtype)

    if not isinstance(end, Variable):
        with device_guard("cpu"):
            end = fill_constant([1], dtype, end, force_cpu=True)
    elif end.dtype != dtype:
        end = cast(end, dtype)

    if not isinstance(step, Variable):
        with device_guard("cpu"):
            step = fill_constant([1], dtype, step, force_cpu=True)
    elif step.dtype != dtype:
        step = cast(step, dtype)

    if in_dygraph_mode():
        out = _C_ops.range(start, end, step)
        out.stop_gradient = True
        return out

    out_shape = None
    if not isinstance(start, Variable) and not isinstance(
            end, Variable) and not isinstance(step, Variable):
        out_shape = [int(math.ceil((end - start) / step))]

    check_dtype(dtype, 'dtype', ['float32', 'float64', 'int32', 'int64'],
                'range/arange')
    helper = LayerHelper('range', **locals())
    out = helper.create_variable_for_type_inference(dtype, shape=out_shape)
    helper.append_op(
        type='range',
        inputs={'Start': start,
                'End': end,
                'Step': step},
        outputs={'Out': out})
    out.stop_gradient = True
    return out


def linspace(start, stop, num, dtype=None, name=None):
    r"""
    This OP return fixed number of evenly spaced values within a given interval.

    Args:
        start(int|float|Tensor): The input :attr:`start` is start variable of range. It is a scalar, \
            or a Tensor of shape [1] with input data type int32, int64, float32 or float64.
        stop(int|float|Tensor): The input :attr:`stop` is start variable of range. It is a scalar, \
            or a Tensor of shape [1] with input data type int32, int64, float32 or float64.
        num(int|Tensor): The input :attr:`num` is given num of the sequence. It is an int scalar, \
            or a Tensor of shape [1] with data type int32.
        dtype(np.dtype|str, optional): The data type of output tensor, it could be
            int32, int64, float32 and float64. Default: if None, the data type is float32.
        name(str, optional): Normally there is no need for user to set this property. 
            For more information, please refer to :ref:`api_guide_Name`.Default: None.

    Returns:
        Tensor: the output data type will be float32, float64. The 1-D tensor with fixed number of evenly spaced values, \
        the data shape of this tensor is :math:`[num]` . If the :attr:`num` is set 1, the output tensor just has \
        the value with input :attr:`start`. 

    Examples:
        .. code-block:: python

             import paddle
             data = paddle.linspace(0, 10, 5, 'float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
             data = paddle.linspace(0, 10, 1, 'float32') # [0.0]

    """
    if dtype is None:
        dtype = 'float32'
    tensor_num = num
    tensor_start = start
    tensor_stop = stop
    if not isinstance(num, Variable):
        check_type(num, 'num', (int), 'linspace')
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)
    if not isinstance(start, Variable):
        with device_guard("cpu"):
            tensor_start = fill_constant([1], dtype, start)
    if not isinstance(stop, Variable):
        with device_guard("cpu"):
            tensor_stop = fill_constant([1], dtype, stop)
    if not isinstance(num, Variable):
        with device_guard("cpu"):
            tensor_num = fill_constant([1], 'int32', num)
    if in_dygraph_mode():
        return _C_ops.linspace(tensor_start, tensor_stop, tensor_num, 'dtype',
                               dtype)

    helper = LayerHelper("linspace", **locals())

    start_dtype = convert_dtype(tensor_start.dtype)
    stop_dtype = convert_dtype(tensor_stop.dtype)
    out_dtype = convert_dtype(dtype)
    if isinstance(start, Variable):
        check_dtype(start.dtype, 'start',
                    ['float32', 'float64', 'int32', 'int64'], 'linspace')
    else:
        check_type(start, 'start', (int, float), 'linspace')

    if isinstance(stop, Variable):
        check_dtype(stop.dtype, 'stop',
                    ['float32', 'float64', 'int32', 'int64'], 'linspace')
    else:
        check_type(stop, 'stop', (int, float), 'linspace')
    if isinstance(num, Variable):
        check_dtype(num.dtype, 'num', ['int32'], 'linspace')
    check_dtype(dtype, 'dtype', ['int32', 'int64', 'float32', 'float64'],
                'linspace')
    if ((stop_dtype == "float64" or start_dtype == "float64") and
            out_dtype in ["float32", "int32"]) or ((stop_dtype == "int64" or
                                                    start_dtype == "int64") and
                                                   out_dtype == "int32"):
        raise ValueError(
            "The dtype of start/stop is {}/{} but the attr(dtype) of linspace is {}, "
            "which may cause data type overflows. Please reset attr(dtype) of linspace."
            .format(start_dtype, stop_dtype, dtype))

    out = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type='linspace',
        inputs={'Start': tensor_start,
                'Stop': tensor_stop,
                'Num': tensor_num},
        attrs={'dtype': dtype},
        outputs={'Out': [out]})
    if isinstance(num, int):
        out.desc.set_shape((num, ))
    return out


def zeros_like(x, out=None):
    """
    This OP creates a zeros tensor which has identical shape and dtype 
    with `x`.

    Args:
        x(Variable): The input tensor which specifies shape and dtype, the
            input data dtype could be bool, float32, float64, int32, int64.
        out(Variable, optional): If is :attr:`None` , the op will create the
            variable as output, the data type and shape of this variable will
            be same as input :attr:`x`. If is a tensor, the data type and shape
            need to be same as input :attr:`x`. The default value is :attr:`None` .

    Returns:
        Variable: The N-D tensor, the element in tensor is related to input
            data type, if the input data type is bool, the output value is
            False, otherwise is zero. The output shape is the same as the input.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = fluid.data(name='x', dtype='float32', shape=[3])
          data = fluid.layers.zeros_like(x) # [0.0, 0.0, 0.0]

    """

    check_variable_and_dtype(
        x, "x", ['bool', 'float32', 'float64', 'int32', 'int64'], 'ones_like')
    helper = LayerHelper("zeros_like", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        check_variable_and_dtype(
            out, "out", ['bool', 'float32', 'float64', 'int32', 'int64'],
            'zeros_like')

    helper.append_op(
        type='fill_zeros_like', inputs={'X': [x]}, outputs={'Out': [out]})
    out.stop_gradient = True
    return out


@deprecated(since="2.0.0", update_to="paddle.diag")
def diag(diagonal):
    r"""
	:alias_main: paddle.diag
	:alias: paddle.diag,paddle.tensor.diag,paddle.tensor.creation.diag
	:old_api: paddle.fluid.layers.diag

    This OP creates a square matrix which has diagonal values specified by input :attr:`diagonal`.

    Args:
        diagonal(Variable|numpy.ndarray): The input tensor should be 1D tensor, the input shape is :math:`[ N]` , \
            specifying diagonal values by this input tensor. The input data type should be float32, float64, int32, int64.

    Returns:
        Variable, the output data type is the same as input data type.: The tensor variable storing the square matrix, \
            the diagonal values specified by input :attr:`diagonal`. the output shape is :math:`[N, N]` with two dims.

    Examples:
        .. code-block:: python

          # [[3, 0, 0]
          #  [0, 4, 0]
          #  [0, 0, 5] 

          import paddle.fluid as fluid
          import numpy as np
          diagonal = np.arange(3, 6, dtype='int32')
          data = fluid.layers.diag(diagonal)
          # diagonal.shape=(3,) data.shape=(3, 3)

    """
    check_type(diagonal, 'diagonal', (Variable, numpy.ndarray), 'diag')
    check_dtype(diagonal.dtype, 'diagonal',
                ['float32', 'float64', 'int32', 'int64'], 'diag')
    helper = LayerHelper("diag", **locals())

    if not isinstance(diagonal, Variable):
        diagonal = assign(diagonal)

    out = helper.create_variable_for_type_inference(dtype=diagonal.dtype)

    helper.append_op(
        type='diag', inputs={'Diagonal': [diagonal]}, outputs={'Out': [out]})

    out.stop_gradient = True
    return out


def eye(num_rows,
        num_columns=None,
        batch_shape=None,
        dtype='float32',
        name=None):
    """
    This function constructs a or a batch of 2-D tensor with ones on the diagonal and zeros elsewhere. 

    Args:
        num_rows(int): the number of rows in each batch tensor.
        num_columns(int, optional): the number of columns in each batch tensor.
            If None, default: num_rows.
        batch_shape(list, optional): If provided, the returned tensor will have a leading
            batch size of this shape, the data type of ``batch_shape`` is int. Default is None.
        dtype(np.dtype|str, optional): The data type of the returned tensor.
            It should be int32, int64, float16, float32, float64, default is 'float32'.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: An identity Tensor or LoDTensor of shape batch_shape + [num_rows, num_columns].

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.eye(3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]
          #  [0, 0, 1]]

          data = fluid.layers.eye(2, 3, dtype='int32')
          # [[1, 0, 0]
          #  [0, 1, 0]]

          data = fluid.layers.eye(2, batch_shape=[3])
          # Construct a batch of 3 identity tensors, each 2 x 2.
          # data[i, :, :] is a 2 x 2 identity tensor, i = 0, 1, 2.

    """

    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)
    if num_columns is not None:
        if not isinstance(num_columns, int) or num_columns < 0:
            raise TypeError("num_columns should be a non-negative int")
    else:
        num_columns = num_rows

    if in_dygraph_mode():
        out = _C_ops.eye('dtype', dtype, 'num_rows', num_rows, 'num_columns',
                         num_columns)

    else:
        helper = LayerHelper("eye", **locals())
        check_dtype(dtype, 'dtype',
                    ['float16', 'float32', 'float64', 'int32', 'int64'], 'eye')
        if not isinstance(num_rows, int) or num_rows < 0:
            raise TypeError("num_rows should be a non-negative int")
        out = helper.create_variable_for_type_inference(dtype=dtype)
        helper.append_op(
            type='eye',
            inputs={},
            outputs={'Out': [out]},
            attrs={
                'num_rows': num_rows,
                'num_columns': num_columns,
                'dtype': dtype
            },
            stop_gradient=True)

    if batch_shape is not None:
        re_shape = [1] * len(batch_shape)
        re_shape = re_shape + [num_rows, num_columns]
        expand_times = batch_shape + [1, 1]
        if in_dygraph_mode():
            out = _C_ops.reshape(out, 'shape', re_shape)
            return _C_ops.expand(out, None, 'expand_times', expand_times)

        if not isinstance(batch_shape, list):
            raise TypeError("batch_shape should be a list")
        for batch_val in (batch_shape):
            if batch_val <= 0:
                raise TypeError("batch_shape should be a positive int list")

        from .nn import reshape, expand
        out = reshape(x=out, shape=re_shape)
        out = expand(x=out, expand_times=expand_times)

    out.stop_gradient = True
    return out


def ones_like(x, out=None):
    """
    **ones_like**

    This function creates a ones tensor which has identical shape and dtype 
    with `x`.

    Args:
        x(Variable): The input tensor which specifies shape and dtype.
        out(Variable): The output tensor.

    Returns:
        out(Variable): The tensor variable storing the output.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          x = fluid.layers.data(name='x', dtype='float32', shape=[3], append_batch_size=False)
          data = fluid.layers.ones_like(x) # [1.0, 1.0, 1.0]

    """
    check_variable_and_dtype(
        x, "x", ['bool', 'float32', 'float64', 'int32', 'int64'], 'ones_like')

    helper = LayerHelper("ones_like", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        check_variable_and_dtype(
            out, "out", ['bool', 'float32', 'float64', 'int32', 'int64'],
            'ones_like')
    helper.append_op(
        type='fill_any_like',
        inputs={'X': [x]},
        attrs={'value': 1.0},
        outputs={'Out': [out]})
    return out


@deprecated(since="2.0.0", update_to="paddle.triu")
def triu(input, diagonal=0, name=None):
    import paddle
    return paddle.tensor.triu(x=input, diagonal=diagonal, name=name)
