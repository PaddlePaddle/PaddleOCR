# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from ..fluid.core import LoDTensor
from ..fluid.framework import in_dygraph_mode
from ..fluid.data_feeder import check_type, check_dtype, convert_dtype

__all__ = [
    'to_dlpack',
    'from_dlpack',
]


def to_dlpack(x):
    """
    Encodes a tensor to DLPack.

    Args:
        x (Tensor): The input tensor, and the data type can be `bool`, `float16`, `float32`,
                    `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`,
                    `complex128`.

    Returns:
        dltensor, and the data type is PyCapsule.
    
    Examples:
        .. code-block:: python

            import paddle
            # x is a tensor with shape [2, 4]
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]])
            dlpack = paddle.utils.dlpack.to_dlpack(x)
            print(dlpack)
            # <capsule object "dltensor" at 0x7f6103c681b0>
    """

    if in_dygraph_mode():
        if not isinstance(x, paddle.Tensor):
            raise TypeError(
                "The type of 'x' in to_dlpack must be paddle.Tensor,"
                " but received {}.".format(type(x)))

        return x.value().get_tensor()._to_dlpack()

    check_type(x, 'x', (LoDTensor), 'to_dlpack')
    return x._to_dlpack()


def from_dlpack(dlpack):
    """
    Decodes a DLPack to a tensor.
    
    Args:
        dlpack (PyCapsule): a PyCapsule object with the dltensor.

    Returns:
        out (Tensor): a tensor decoded from DLPack. One thing to be noted, if we get 
                      an input dltensor with data type as `bool`, we return the decoded
                      tensor as `uint8`.

    Examples:
        .. code-block:: python

            import paddle
            # x is a tensor with shape [2, 4]
            x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                  [0.1, 0.2, 0.6, 0.7]])
            dlpack = paddle.utils.dlpack.to_dlpack(x)
            x = paddle.utils.dlpack.from_dlpack(dlpack)
            print(x)
            # Tensor(shape=[2, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #  [[0.20000000, 0.30000001, 0.50000000, 0.89999998],
            #  [0.10000000, 0.20000000, 0.60000002, 0.69999999]]) 
    """

    t = type(dlpack)
    dlpack_flag = (t.__module__ == 'builtins' and t.__name__ == 'PyCapsule')
    if not dlpack_flag:
        raise TypeError(
            "The type of 'dlpack' in from_dlpack must be PyCapsule object,"
            " but received {}.".format(type(dlpack)))

    if in_dygraph_mode():
        out = paddle.fluid.core.from_dlpack(dlpack)
        out = paddle.to_tensor(out)
        return out

    out = paddle.fluid.core.from_dlpack(dlpack)
    return out
