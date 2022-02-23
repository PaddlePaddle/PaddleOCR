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

# TODO: define the extention functions

import numpy as np
from ...fluid.data_feeder import check_dtype
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode
from ...static import Variable
from ...tensor.creation import assign
from ...fluid import core, dygraph_utils
from ...fluid.layers.layer_function_generator import templatedoc
from ...fluid.layers.sequence_lod import sequence_mask

__all__ = []


def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    """
    This OP creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) 
    are filled by ``input``. By default, a 2D plane formed by the last two dimensions 
    of the returned tensor will be selected.

    The argument ``offset`` determines which diagonal is generated:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.

    Args:
        input(Tensor|numpy.ndarray): The input tensor. Must be at least 1-dimensional. The input data type should be float32, float64, int32, int64.
        offset(int, optional): Which diagonal to consider. Default: 0 (main diagonal).
        dim1(int, optional): The first dimension with respect to which to take diagonal. Default: -2.
        dim2(int, optional): The second dimension with respect to which to take diagonal. Default: -1.
    
    Returns:
        Tensor, the output data type is the same as input data type.
    
    Examples:
        .. code-block:: python

            import paddle.nn.functional as F
            import numpy as np
            
            diag_embed = np.random.randn(2, 3).astype('float32')
            # [[ 0.7545889 , -0.25074545,  0.5929117 ],
            #  [-0.6097662 , -0.01753256,  0.619769  ]]

            data1 = F.diag_embed(diag_embed)
            data1.numpy()
            # [[[ 0.7545889 ,  0.        ,  0.        ],
            #  [ 0.        , -0.25074545,  0.        ],
            #   [ 0.        ,  0.        ,  0.5929117 ]],

            # [[-0.6097662 ,  0.        ,  0.        ],
            #  [ 0.        , -0.01753256,  0.        ],
            #  [ 0.        ,  0.        ,  0.619769  ]]]

            data2 = F.diag_embed(diag_embed, offset=-1, dim1=0, dim2=2)
            data2.numpy()
            # [[[ 0.        ,  0.        ,  0.        ,  0.        ],
            #   [ 0.7545889 ,  0.        ,  0.        ,  0.        ],
            #   [ 0.        , -0.25074545,  0.        ,  0.        ],
            #   [ 0.        ,  0.        ,  0.5929117 ,  0.        ]],
            #
            #  [[ 0.        ,  0.        ,  0.        ,  0.        ],
            #   [-0.6097662 ,  0.        ,  0.        ,  0.        ],
            #   [ 0.        , -0.01753256,  0.        ,  0.        ],
            #   [ 0.        ,  0.        ,  0.619769  ,  0.        ]]]

            data3 = F.diag_embed(diag_embed, offset=1, dim1=0, dim2=2)
            data3.numpy()
            # [[[ 0.        ,  0.7545889 ,  0.        ,  0.        ],
            #   [ 0.        , -0.6097662 ,  0.        ,  0.        ]],
            #
            #  [[ 0.        ,  0.        , -0.25074545,  0.        ],
            #   [ 0.        ,  0.        , -0.01753256,  0.        ]],
            #
            #  [[ 0.        ,  0.        ,  0.        ,  0.5929117 ],
            #   [ 0.        ,  0.        ,  0.        ,  0.619769  ]],
            #
            #  [[ 0.        ,  0.        ,  0.        ,  0.        ],
            #   [ 0.        ,  0.        ,  0.        ,  0.        ]]]
    """
    inputs = {'Input': [input]}
    attrs = {'offset': offset, 'dim1': dim1, 'dim2': dim2}

    if not isinstance(input, Variable):
        input = assign(input)

    def __check_input(input, offset, dim1, dim2):
        check_dtype(input.dtype, 'Input',
                    ['int32', 'int64', 'float16', 'float32', 'float64'],
                    'diag_embed')

        input_shape = list(input.shape)
        assert len(input_shape) >= 1,                     \
                "Input must be at least 1-dimensional, "   \
                "But received Input's dimensional: %s.\n" %  \
                len(input_shape)

        assert np.abs(dim1) <= len(input_shape),    \
            "Dim1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape) + 1), len(input_shape), dim1)

        assert np.abs(dim2) <= len(input_shape),      \
            "Dim2 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape) + 1), len(input_shape), dim2)

        dim1_ = dim1 if dim1 >= 0 else len(input_shape) + dim1 + 1
        dim2_ = dim2 if dim2 >= 0 else len(input_shape) + dim2 + 1
        assert dim1_ != dim2_,       \
               "dim1 and dim2 cannot be the same dimension." \
                "But received dim1 = %d, dim2 = %d\n"%(dim1, dim2)

    if not in_dygraph_mode():
        __check_input(input, offset, dim1, dim2)
    helper = LayerHelper("diag_embed", **locals())

    out = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='diag_embed',
        inputs={'Input': [input]},
        attrs={'offset': offset,
               'dim1': dim1,
               'dim2': dim2},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out
