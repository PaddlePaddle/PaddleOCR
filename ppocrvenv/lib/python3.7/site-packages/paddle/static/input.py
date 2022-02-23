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

import six

import paddle
from paddle.fluid import core, Variable
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_type
from paddle.fluid.framework import convert_np_dtype_to_dtype_
from paddle.fluid.framework import static_only

__all__ = []


@static_only
def data(name, shape, dtype=None, lod_level=0):
    """
    **Data Layer**

    This function creates a variable on the global block. The global variable
    can be accessed by all the following operators in the graph. The variable
    is a placeholder that could be fed with input, such as Executor can feed
    input into the variable. When `dtype` is None, the dtype
    will get from the global dtype by `paddle.get_default_dtype()`.

    Args:
       name (str): The name/alias of the variable, see :ref:`api_guide_Name`
           for more details.
       shape (list|tuple): List|Tuple of integers declaring the shape. You can
           set "None" or -1 at a dimension to indicate the dimension can be of any
           size. For example, it is useful to set changeable batch size as "None" or -1.
       dtype (np.dtype|str, optional): The type of the data. Supported
           dtype: bool, float16, float32, float64, int8, int16, int32, int64,
           uint8. Default: None. When `dtype` is not set, the dtype will get
           from the global dtype by `paddle.get_default_dtype()`.
       lod_level (int, optional): The LoD level of the LoDTensor. Usually users
           don't have to set this value. For more details about when and how to
           use LoD level, see :ref:`user_guide_lod_tensor` . Default: 0.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          import numpy as np
          import paddle
          paddle.enable_static()

          # Creates a variable with fixed size [3, 2, 1]
          # User can only feed data of the same shape to x
          # the dtype is not set, so it will set "float32" by
          # paddle.get_default_dtype(). You can use paddle.get_default_dtype() to
          # change the global dtype
          x = paddle.static.data(name='x', shape=[3, 2, 1])

          # Creates a variable with changeable batch size -1.
          # Users can feed data of any batch size into y,
          # but size of each data sample has to be [2, 1]
          y = paddle.static.data(name='y', shape=[-1, 2, 1], dtype='float32')

          z = x + y

          # In this example, we will feed x and y with np-ndarray "1"
          # and fetch z, like implementing "1 + 1 = 2" in PaddlePaddle
          feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

          exe = paddle.static.Executor(paddle.framework.CPUPlace())
          out = exe.run(paddle.static.default_main_program(),
                        feed={
                            'x': feed_data,
                            'y': feed_data
                        },
                        fetch_list=[z.name])

          # np-ndarray of shape=[3, 2, 1], dtype=float32, whose elements are 2
          print(out)

    """
    helper = LayerHelper('data', **locals())
    check_type(name, 'name', (six.binary_type, six.text_type), 'data')
    check_type(shape, 'shape', (list, tuple), 'data')

    shape = list(shape)
    for i in six.moves.range(len(shape)):
        if shape[i] is None:
            shape[i] = -1

    if dtype:
        return helper.create_global_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            stop_gradient=True,
            lod_level=lod_level,
            is_data=True,
            need_check_feed=True)
    else:
        return helper.create_global_variable(
            name=name,
            shape=shape,
            dtype=paddle.get_default_dtype(),
            type=core.VarDesc.VarType.LOD_TENSOR,
            stop_gradient=True,
            lod_level=lod_level,
            is_data=True,
            need_check_feed=True)


class InputSpec(object):
    """
    InputSpec describes the signature information of the model input, such as ``shape`` , ``dtype`` , ``name`` .

    This interface is often used to specify input tensor information of models in high-level API.
    It's also used to specify the tensor information for each input parameter of the forward function
    decorated by `@paddle.jit.to_static`.

    Args:
        shape (tuple(integers)|list[integers]): List|Tuple of integers
            declaring the shape. You can set "None" or -1 at a dimension
            to indicate the dimension can be of any size. For example,
            it is useful to set changeable batch size as "None" or -1.
        dtype (np.dtype|str, optional): The type of the data. Supported
            dtype: bool, float16, float32, float64, int8, int16, int32, int64,
            uint8. Default: float32.
        name (str): The name/alias of the variable, see :ref:`api_guide_Name`
            for more details.

    Examples:
        .. code-block:: python

            from paddle.static import InputSpec

            input = InputSpec([None, 784], 'float32', 'x')
            label = InputSpec([None, 1], 'int64', 'label')

            print(input)  # InputSpec(shape=(-1, 784), dtype=VarType.FP32, name=x)
            print(label)  # InputSpec(shape=(-1, 1), dtype=VarType.INT64, name=label)
    """

    def __init__(self, shape, dtype='float32', name=None):
        # replace `None` in shape  with -1
        self.shape = self._verify(shape)
        # convert dtype into united represention
        if dtype is not None:
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)
        self.dtype = dtype
        self.name = name

    def _create_feed_layer(self):
        return data(self.name, shape=self.shape, dtype=self.dtype)

    def __repr__(self):
        return '{}(shape={}, dtype={}, name={})'.format(
            type(self).__name__, self.shape, self.dtype, self.name)

    @classmethod
    def from_tensor(cls, tensor, name=None):
        """
        Generates a InputSpec based on the description of input tensor.

        Args:
            tensor(Tensor): the source tensor to generate a InputSpec instance

        Returns:
            A InputSpec instance generated from Tensor.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.static import InputSpec

                paddle.disable_static()

                x = paddle.to_tensor(np.ones([2, 2], np.float32))
                x_spec = InputSpec.from_tensor(x, name='x')
                print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)

        """
        if isinstance(tensor, (Variable, core.VarBase)):
            return cls(tensor.shape, tensor.dtype, name or tensor.name)
        else:
            raise ValueError(
                "Input `tensor` should be a Tensor, but received {}.".format(
                    type(tensor).__name__))

    @classmethod
    def from_numpy(cls, ndarray, name=None):
        """
        Generates a InputSpec based on the description of input np.ndarray.

        Args:
            tensor(Tensor): the source numpy ndarray to generate a InputSpec instance

        Returns:
            A InputSpec instance generated from Tensor.

        Examples:
            .. code-block:: python

                import numpy as np
                from paddle.static import InputSpec

                x = np.ones([2, 2], np.float32)
                x_spec = InputSpec.from_numpy(x, name='x')
                print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)

        """
        return cls(ndarray.shape, ndarray.dtype, name)

    def batch(self, batch_size):
        """
        Inserts `batch_size` in front of the `shape`.

        Args:
            batch_size(int): the inserted integer value of batch size.

        Returns:
            The original InputSpec instance by inserting `batch_size` in front of `shape`.

        Examples:
            .. code-block:: python

                from paddle.static import InputSpec

                x_spec = InputSpec(shape=[64], dtype='float32', name='x')
                x_spec.batch(4)
                print(x_spec) # InputSpec(shape=(4, 64), dtype=VarType.FP32, name=x)

        """
        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) != 1:
                raise ValueError(
                    "Length of batch_size: {} shall be 1, but received {}.".
                    format(batch_size, len(batch_size)))
            batch_size = batch_size[1]
        elif not isinstance(batch_size, six.integer_types):
            raise TypeError("type(batch_size) shall be `int`, but received {}.".
                            format(type(batch_size).__name__))

        new_shape = [batch_size] + list(self.shape)
        self.shape = tuple(new_shape)

        return self

    def unbatch(self):
        """
        Removes the first element of `shape`.

        Returns:
            The original InputSpec instance by removing the first element of `shape` .

        Examples:
            .. code-block:: python

                from paddle.static import InputSpec

                x_spec = InputSpec(shape=[4, 64], dtype='float32', name='x')
                x_spec.unbatch()
                print(x_spec) # InputSpec(shape=(64,), dtype=VarType.FP32, name=x)

        """
        if len(self.shape) == 0:
            raise ValueError(
                "Not support to unbatch a InputSpec when len(shape) == 0.")

        self.shape = self._verify(self.shape[1:])
        return self

    def _verify(self, shape):
        """
        Verifies the input shape and modifies `None` into `-1`.
        """
        if not isinstance(shape, (list, tuple)):
            raise TypeError(
                "Type of `shape` in InputSpec should be one of (tuple, list), but received {}.".
                format(type(shape).__name__))
        if len(shape) == 0:
            raise ValueError(
                "`shape` in InputSpec should contain at least 1 element, but received {}.".
                format(shape))

        for i, ele in enumerate(shape):
            if ele is not None:
                if not isinstance(ele, six.integer_types):
                    raise ValueError(
                        "shape[{}] should be an `int`, but received `{}`:{}.".
                        format(i, type(ele).__name__, ele))
            if ele is None or ele < -1:
                shape[i] = -1

        return tuple(shape)

    def __hash__(self):
        # Note(Aurelius84): `name` is not considered as a field to compute hashkey.
        # Because it's no need to generate a new program in following cases while using
        # @paddle.jit.to_static.
        #
        # Case 1:
        #      foo(x_var)
        #      foo(y_var)
        #  x_var and y_var hold same shape and dtype, they should share a same program.
        #
        #
        # Case 2:
        #      foo(x_var)
        #      foo(x_np)  # x_np is a numpy.ndarray.
        #  x_var and x_np hold same shape and dtype, they should also share a same program.
        return hash((tuple(self.shape), self.dtype))

    def __eq__(self, other):
        slots = ['shape', 'dtype', 'name']
        return (type(self) is type(other) and all(
            getattr(self, attr) == getattr(other, attr) for attr in slots))

    def __ne__(self, other):
        return not self == other
