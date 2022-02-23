#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from . import core
from .data_feeder import DataToLoDTensorConverter
import numpy as np

__all__ = ['create_lod_tensor', 'create_random_int_lodtensor']


def create_lod_tensor(data, recursive_seq_lens, place):
    """
    Create a LoDTensor from a numpy array, list or existing LoDTensor.

    The implementation is as follows:

    1. Check whether the length-based LoD, i.e., :code:`recursive_seq_lens`
       is valid.

    2. Convert :code:`recursive_seq_lens` to a offset-based LoD.

    3. Based on :code:`place` , copy the :code:`data` from a numpy array, list
       or existing LoDTensor to CPU or GPU device.

    4. Set offset-based LoD to the output LoDTensor.

    Suppose we want to create a LoDTensor to hold data for word sequences,
    where each word is represented by an integer. If we want to create
    a LoDTensor to represent two sentences, one of 2 words, and one of 3 words.

    Then :code:`data` would be a numpy array of integers with shape (5, 1).
    :code:`recursive_seq_lens` would be [[2, 3]], indicating the word number
    in each sentence. This length-based :code:`recursive_seq_lens` [[2, 3]]
    would be converted to offset-based LoD [[0, 2, 5]] inside the function
    call.

    Please reference :ref:`user_guide_lod_tensor` for more details regarding LoD.

    Args:
        data (numpy.ndarray|list|LoDTensor): a numpy array, a list or ad LoDTensor
                holding the data to be copied.
        recursive_seq_lens (list[list[int]]): a list of lists indicating the
                length-based LoD info.
        place (CPUPlace|CUDAPlace): CPU or GPU place indicating where the data
                in the created LoDTensor will be stored.

    Returns:
         A LoDTensor with tensor data and recursive_seq_lens info.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            t = fluid.create_lod_tensor(np.ndarray([5, 30]), [[2, 3]], fluid.CPUPlace())
    """
    if isinstance(data, core.LoDTensor):
        return create_lod_tensor(np.array(data), recursive_seq_lens, place)
    elif isinstance(data, list):
        # dtype and shape are not important here,
        # we only want to reuse code of DataToLoDTensorConverter
        converter = DataToLoDTensorConverter(
            place=place,
            lod_level=len(recursive_seq_lens),
            shape=[],
            dtype=core.VarDesc.VarType.FP32)

        new_recursive_seq_lens = []
        for seq in data:
            new_recursive_seq_lens.append(len(seq))
            converter.feed(seq)

        assert [
            new_recursive_seq_lens
        ] == recursive_seq_lens, "data and recursive_seq_lens do not match"

        arr = np.array(converter.data)

        # FIXME(zjl): the original logic of create_lod_tensor would append
        # 1 to the shape. Maybe it is not a right way? Currently, we only
        # follow the previous logic
        arr = arr.reshape(arr.shape + (1, ))
        tensor = core.LoDTensor()
        tensor.set(arr, place)
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        return tensor
    elif isinstance(data, np.ndarray):
        tensor = core.LoDTensor()
        tensor.set(data, place)
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        assert tensor.has_valid_recursive_sequence_lengths(
        ), "the provided lod info is invalid"
        return tensor
    else:
        raise TypeError(
            "data should be either a LoDTensor, a Numpy array or a list")


def create_random_int_lodtensor(recursive_seq_lens, base_shape, place, low,
                                high):
    """
	:api_attr: Static Graph

    Create a LoDTensor containing random integers.

    The implementation is as follows:

    1. Obtain the shape of output LoDTensor based on :code:`recursive_seq_lens`
       and :code:`base_shape` . The first dimension of the shape is the total
       length of sequences, while the other dimensions are the same as
       :code:`base_shape` .

    2. Create a numpy array of random integers, and parse the created numpy
       array as parameter :code:`data` of :ref:`api_fluid_create_lod_tensor` to
       create the output LoDTensor.

    Suppose we want to create a LoDTensor to hold data for 2 sequences, where
    the dimension of the sequences are [2, 30] and [3, 30] respectively.
    The :code:`recursive_seq_lens` would be [[2, 3]], and :code:`base_shape`
    would be [30] (the other dimensions excluding the sequence length).
    Therefore, the shape of the output LoDTensor would be [5, 30], where
    the first dimension 5 is the total lengths of the sequences, and the
    other dimensions are :code:`base_shape`.

    Args:
        recursive_seq_lens (list[list[int]]): a list of lists indicating the
                length-based LoD info.
        base_shape (list[int]): the shape of the output LoDTensor excluding
                the first dimension.
        place (CPUPlace|CUDAPlace): CPU or GPU place indicating where
                the data in the created LoDTensor will be stored.
        low (int): the lower bound of the random integers.
        high (int): the upper bound of the random integers.

    Returns:
        A LoDTensor with tensor data and recursive_seq_lens info, whose data
        is inside [low, high].

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          t = fluid.create_random_int_lodtensor(recursive_seq_lens=[[2, 3]],
                    base_shape=[30], place=fluid.CPUPlace(), low=0, high=10)
          print(t.shape()) # [5, 30]
    """
    assert isinstance(base_shape, list), "base_shape should be a list"
    # append the total number of basic elements to the front of its shape
    overall_shape = [sum(recursive_seq_lens[-1])] + base_shape
    # the range of integer data elements is [low, high]
    data = np.random.random_integers(low, high, overall_shape).astype("int64")
    return create_lod_tensor(data, recursive_seq_lens, place)
