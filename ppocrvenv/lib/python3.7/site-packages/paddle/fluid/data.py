#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import six

from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_dtype, check_type
from ..utils import deprecated
from paddle.fluid.framework import static_only

__all__ = ['data']


@static_only
@deprecated(since="2.0.0", update_to="paddle.static.data")
def data(name, shape, dtype='float32', lod_level=0):
    """
    **Data Layer**

    This function creates a variable on the global block. The global variable
    can be accessed by all the following operators in the graph. The variable
    is a placeholder that could be fed with input, such as Executor can feed
    input into the variable.

    Note: 
        `paddle.fluid.layers.data` is deprecated. It will be removed in a
        future version. Please use this `paddle.fluid.data`. 
       
        The `paddle.fluid.layers.data` set shape and dtype at compile time but
        does NOT check the shape or the dtype of fed data, this
        `paddle.fluid.data` checks the shape and the dtype of data fed by
        Executor or ParallelExecutor during run time.

        To feed variable size inputs, users can set None or -1 on the variable
        dimension when using :code:`paddle.fluid.data`, or feed variable size
        inputs directly to :code:`paddle.fluid.layers.data` and PaddlePaddle
        will fit the size accordingly.

        The default :code:`stop_gradient` attribute of the Variable created by
        this API is true, which means the gradient won't be passed backward
        through the data Variable. Set :code:`var.stop_gradient = False` If
        user would like to pass backward gradient.

    Args:
       name (str): The name/alias of the variable, see :ref:`api_guide_Name`
           for more details.
       shape (list|tuple): List|Tuple of integers declaring the shape. You can
           set "None" or -1 at a dimension to indicate the dimension can be of any
           size. For example, it is useful to set changeable batch size as "None" or -1.
       dtype (np.dtype|VarType|str, optional): The type of the data. Supported
           dtype: bool, float16, float32, float64, int8, int16, int32, int64,
           uint8. Default: float32.
       lod_level (int, optional): The LoD level of the LoDTensor. Usually users
           don't have to set this value. For more details about when and how to
           use LoD level, see :ref:`user_guide_lod_tensor` . Default: 0.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.fluid as fluid
          import numpy as np
          paddle.enable_static()

          # Creates a variable with fixed size [3, 2, 1]
          # User can only feed data of the same shape to x
          x = fluid.data(name='x', shape=[3, 2, 1], dtype='float32')

          # Creates a variable with changeable batch size -1.
          # Users can feed data of any batch size into y,
          # but size of each data sample has to be [2, 1]
          y = fluid.data(name='y', shape=[-1, 2, 1], dtype='float32')

          z = x + y

          # In this example, we will feed x and y with np-ndarray "1"
          # and fetch z, like implementing "1 + 1 = 2" in PaddlePaddle
          feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

          exe = fluid.Executor(fluid.CPUPlace())
          out = exe.run(fluid.default_main_program(),
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

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=core.VarDesc.VarType.LOD_TENSOR,
        stop_gradient=True,
        lod_level=lod_level,
        is_data=True,
        need_check_feed=True)
