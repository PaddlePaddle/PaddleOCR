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

# Define functions about array.

from ..fluid import layers

__all__ = []


def array_length(array):
    """
    This OP is used to get the length of the input array.

    Args:
        array (list|Tensor): The input array that will be used to compute the length. In dynamic mode, ``array`` is a Python list. But in static mode, array is a Tensor whose VarType is LOD_TENSOR_ARRAY.

    Returns:
        Tensor: 1-D Tensor with shape [1], which is the length of array.

    Examples:
        .. code-block:: python

            import paddle

            arr = paddle.tensor.create_array(dtype='float32')
            x = paddle.full(shape=[3, 3], fill_value=5, dtype="float32")
            i = paddle.zeros(shape=[1], dtype="int32")

            arr = paddle.tensor.array_write(x, i, array=arr)

            arr_len = paddle.tensor.array_length(arr)
            print(arr_len)  # 1
    """
    return layers.array_length(array)


def array_read(array, i):
    """
    This OP is used to read data at the specified position from the input array.

    Case:

    .. code-block:: text

        Input:
            The shape of first three tensors are [1], and that of the last one is [1,2]:
                array = ([0.6], [0.1], [0.3], [0.4, 0.2])
            And:
                i = [3]

        Output:
            output = [0.4, 0.2]

    Args:
        array (list|Tensor): The input array. In dynamic mode, ``array`` is a Python list. But in static mode, array is a Tensor whose ``VarType`` is ``LOD_TENSOR_ARRAY``.
        i (Tensor): 1-D Tensor, whose shape is [1] and dtype is int64. It represents the
            specified read position of ``array``.

    Returns:
        Tensor: A Tensor that is read at the specified position of ``array``.

    Examples:
        .. code-block:: python

            import paddle

            arr = paddle.tensor.create_array(dtype="float32")
            x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
            i = paddle.zeros(shape=[1], dtype="int32")

            arr = paddle.tensor.array_write(x, i, array=arr)

            item = paddle.tensor.array_read(arr, i)
            print(item)     # [[5., 5., 5.]]
    """
    return layers.array_read(array, i)


def array_write(x, i, array=None):
    """
    This OP writes the input ``x`` into the i-th position of the ``array`` returns the modified array.
    If ``array`` is none, a new array will be created and returned.

    Args:
        x (Tensor): The input data to be written into array. It's multi-dimensional
            Tensor or LoDTensor. Data type: float32, float64, int32, int64 and bool.
        i (Tensor): 1-D Tensor with shape [1], which represents the position into which
            ``x`` is written.
        array (list|Tensor, optional): The array into which ``x`` is written. The default value is None,
            when a new array will be created and returned as a result. In dynamic mode, ``array`` is a Python list.
            But in static mode, array is a Tensor whose ``VarType`` is ``LOD_TENSOR_ARRAY``.

    Returns:
        list|Tensor: The input ``array`` after ``x`` is written into.

    Examples:
        .. code-block:: python

            import paddle

            arr = paddle.tensor.create_array(dtype="float32")
            x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
            i = paddle.zeros(shape=[1], dtype="int32")

            arr = paddle.tensor.array_write(x, i, array=arr)

            item = paddle.tensor.array_read(arr, i)
            print(item)     # [[5., 5., 5.]]
    """
    return layers.array_write(x, i, array)


def create_array(dtype, initialized_list=None):
    """
    This OP creates an array. It is used as the input of :ref:`api_paddle_tensor_array_array_read` and
    :ref:`api_paddle_tensor_array_array_write`.

    Args:
        dtype (str): The data type of the elements in the array. Support data type: float32, float64, int32, int64 and bool.
        initialized_list(list): Used to initialize as default value for created array.
                    All values in initialized list should be a Tensor.

    Returns:
        list|Tensor: An empty array. In dynamic mode, ``array`` is a Python list. But in static mode, array is a Tensor
        whose ``VarType`` is ``LOD_TENSOR_ARRAY``.

    Examples:
        .. code-block:: python

            import paddle

            arr = paddle.tensor.create_array(dtype="float32")
            x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
            i = paddle.zeros(shape=[1], dtype="int32")

            arr = paddle.tensor.array_write(x, i, array=arr)

            item = paddle.tensor.array_read(arr, i)
            print(item)     # [[5., 5., 5.]]

    """
    return layers.create_array(dtype, initialized_list)
