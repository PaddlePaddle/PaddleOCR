#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
import paddle
from ...fluid.framework import in_dygraph_mode, default_main_program
from paddle.fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode
from paddle import _C_ops


def sparse_attention(query,
                     key,
                     value,
                     sparse_csr_offset,
                     sparse_csr_columns,
                     name=None):
    r"""
    This operator sparsify the Attention matrix in Transformer module
    to achieve the effect of reducing memory consumption and computation. 
    The sparse layout is expressed in CSR format and contains two parameters, 
    ``offset`` and ``columns``. The equation is: 

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module. 
    The dimensions of the three parameters are the same. 
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:    
        This API is only used in ``CUDA 11.3`` and above versions.

    Args:
        query(Tensor): The query tensor in the Attention module. 
                        4-D tensor with shape: 
                        [batch_size, num_heads, seq_len, head_dim]. 
                        The dtype can be float32 and float64.
        key(Tensor): The key tensor in the Attention module. 
                        4-D tensor with shape: 
                        [batch_size, num_heads, seq_len, head_dim]. 
                        The dtype can be float32 and float64.
        value(Tensor): The value tensor in the Attention module. 
                        4-D tensor with shape:  
                        [batch_size, num_heads, seq_len, head_dim]. 
                        The dtype can be float32 and float64.
        sparse_csr_offset(Tensor): The sparsity feature in the Attention module 
                        is expressed in the CSR format, and the offset represents 
                        the number of non-zero elements in each row of the matrix.
                        3-D tensor with shape:   
                        [batch_size, num_heads, seq_len + 1]. 
                        The dtype should be int32.
        sparse_csr_columns(Tensor): The sparsity feature in the Attention module 
                        is expressed in the CSR format, and the columns represent 
                        the column index values of non-zero elements in the matrix.
                        3-D tensor with shape:  
                        [batch_size, num_heads, sparse_nnz]. 
                        The dtype should be int32.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        4-D tensor with shape:
        [batch_size, num_heads, seq_len, head_dim]. 
        The dtype can be float32 or float64.

    Examples:
        .. code-block:: python

            # required: skiptest
            import paddle
            import numpy as np
            
            query_data = np.array([[[[0, 1,], [2, 3],
                    [ 0, 1], [2, 3]]]]).astype("float32")
            key_data = np.array([[[[0, 1,], [2, 3],
                            [ 0, 1], [2, 3]]]]).astype("float32")
            value_data = np.array([[[[0, 1,], [2, 3],
                            [ 0, 1], [2, 3]]]]).astype("float32")
            sparse_csr_offset_data = np.array([[[0, 2,
                            4, 6, 8]]]).astype("int32")
            sparse_csr_columns_data = np.array([[[0, 1,
                            0, 1, 2, 3, 2, 3]]]).astype("int32")
            print(query_data.shape)
            # (1, 1, 4, 2)
            print(sparse_csr_offset_data.shape)
            # (1, 1, 5)
            print(sparse_csr_columns_data.shape)
            # (1, 1, 8)
            paddle.disable_static()
            query = paddle.to_tensor(query_data, stop_gradient=False, 
                            place=paddle.CUDAPlace(0))
            key = paddle.to_tensor(key_data, stop_gradient=False, 
                            place=paddle.CUDAPlace(0))
            value = paddle.to_tensor(value_data, stop_gradient=False, 
                            place=paddle.CUDAPlace(0))
            offset = paddle.to_tensor(sparse_csr_offset_data, stop_gradient=False, 
                            place=paddle.CUDAPlace(0))
            columns = paddle.to_tensor(sparse_csr_columns_data, stop_gradient=False, 
                            place=paddle.CUDAPlace(0))
            output = paddle.nn.functional.sparse_attention(query, key, 
                            value, offset, columns)
            print(output)
            
            # [[[[1.60885942, 2.60885954],
            #       [1.99830270, 2.99830270],
            #       [1.60885942, 2.60885954],
            #       [1.99830270, 2.99830270]]]]
    """
    if in_dygraph_mode():
        result_attention, result_sdd, result_softmax = _C_ops.sparse_attention(
            query, key, value, sparse_csr_offset, sparse_csr_columns)
        return result_attention

    helper = LayerHelper('sparse_attention', **locals())
    dtype = helper.input_dtype(input_param_name='Q')
    out = helper.create_variable_for_type_inference(dtype)
    result_sdd = helper.create_variable_for_type_inference(dtype)
    result_softmax = helper.create_variable_for_type_inference(dtype)
    inputs = {
        'Q': query,
        'K': key,
        'V': value,
        'Offset': sparse_csr_offset,
        'Columns': sparse_csr_columns
    }
    outputs = {
        'Out': out,
        'SparseDotSdd': result_sdd,
        'Softmax': result_softmax
    }
    helper.append_op(type='sparse_attention', inputs=inputs, outputs=outputs)
    return out
