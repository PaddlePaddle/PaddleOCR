# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
All layers just related to the neural network.
"""
from __future__ import print_function

import os
import inspect
import warnings

import numpy as np
import six

import paddle
from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant, NumpyArrayInitializer
from ..framework import Variable, OpProtoHolder, in_dygraph_mode, dygraph_only, _dygraph_tracer, default_main_program, _varbase_creator, static_only, _global_flags
from .. import dygraph_utils
from ..param_attr import ParamAttr
from .layer_function_generator import autodoc, templatedoc, _generate_doc_string_
from .tensor import concat, assign, fill_constant, zeros, tensor_array_to_tensor
from . import utils
from .. import unique_name
from functools import reduce
from .. import core
from ...utils import deprecated
from ..data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
import paddle
from paddle.utils import deprecated
from paddle import _C_ops

__all__ = [
    'fc',
    'embedding',
    'linear_chain_crf',
    'crf_decoding',
    'cos_sim',
    'chunk_eval',
    'conv2d',
    'conv3d',
    'softmax',
    'pool2d',
    'pool3d',
    'adaptive_pool2d',
    'adaptive_pool3d',
    'batch_norm',
    'inplace_abn',
    'instance_norm',
    'data_norm',
    'conv2d_transpose',
    'conv3d_transpose',
    'reduce_sum',
    'reduce_mean',
    'reduce_max',
    'reduce_min',
    'reduce_prod',
    'reduce_all',
    'reduce_any',
    'dropout',
    'split',
    'ctc_greedy_decoder',
    'l2_normalize',
    'matmul',
    'topk',
    'transpose',
    'im2sequence',
    'row_conv',
    'multiplex',
    'layer_norm',
    'group_norm',
    'spectral_norm',
    'smooth_l1',
    'one_hot',
    'autoincreased_step_counter',
    'reshape',
    'squeeze',
    'unsqueeze',
    'lod_reset',
    'lod_append',
    'lrn',
    'pad',
    'pad_constant_like',
    'label_smooth',
    'roi_pool',
    'roi_align',
    'dice_loss',
    'image_resize',
    'image_resize_short',
    'resize_linear',
    'resize_bilinear',
    'resize_trilinear',
    'resize_nearest',
    'gather',
    'gather_nd',
    'scatter',
    'scatter_nd_add',
    'scatter_nd',
    'random_crop',
    'mean_iou',
    'relu',
    'selu',
    'log',
    'crop',
    'crop_tensor',
    'elu',
    'relu6',
    'pow',
    'stanh',
    'hard_sigmoid',
    'swish',
    'prelu',
    'brelu',
    'leaky_relu',
    'soft_relu',
    'flatten',
    'stack',
    'pad2d',
    'unstack',
    'unique',
    'unique_with_counts',
    'expand',
    'expand_as',
    'scale',
    'elementwise_add',
    'elementwise_div',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'elementwise_mod',
    'elementwise_floordiv',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'sampling_id',
    'gaussian_random_batch_size_like',
    'sum',
    'slice',
    'strided_slice',
    'shape',
    'rank',
    'size',
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
    'clip',
    'clip_by_norm',
    'mean',
    'mul',
    'maxout',
    'space_to_depth',
    'affine_grid',
    'affine_channel',
    'similarity_focus',
    'hash',
    'grid_sampler',
    'log_loss',
    'add_position_encoding',
    'bilinear_tensor_product',
    'merge_selected_rows',
    'get_tensor_from_selected_rows',
    'shuffle_channel',
    'temporal_shift',
    'py_func',
    'psroi_pool',
    'prroi_pool',
    'pixel_shuffle',
    'fsp_matrix',
    'continuous_value_model',
    'where',
    'sign',
    'deformable_conv',
    'unfold',
    'deformable_roi_pooling',
    'filter_by_instag',
    'shard_index',
    'hard_swish',
    'mish',
    'gather_tree',
    'uniform_random',
    'unbind',
]


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


def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       name=None):
    r"""
    :api_attr: Static Graph

    **Fully Connected Layer**

    This operator creates a fully connected layer in the network. It can take
    a Tensor(or LoDTensor) or a list of Tensor(or LoDTensor) as its inputs(see
    Args in detail). It creates a variable called weight for each input Tensor,
    which represents a fully connected weight matrix from each input unit to
    each output unit. The fully connected layer multiplies each input Tensor
    with its corresponding weight to produce an output Tensor with shape :math:`[M, size]` ,
    where M is batch size. If a list of Tensor is given, the results of
    multiple output Tensors with shape :math:`[M, size]` will be summed up. If :attr:`bias_attr`
    is not None, a bias variable will be created and added to the output.
    Finally, if :attr:`act` is not None, it will be applied to the output as well.

    When the input is a single Tensor(or LoDTensor):

    .. math::

        Out = Act({XW + b})

    When the input is a list of Tensor(or LoDTensor):

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    In the above equation:

    * :math:`N`: Number of the input. N equals to len(input) if input is list of Variable.
    * :math:`X_i`: The i-th input tensor.
    * :math:`W_i`: The i-th weights matrix corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output Tensor.

    .. code-block:: text

        Case 1:
        Given a single Tensor data_1, and num_flatten_dims = 2:
            data_1.data = [[[0.1, 0.2],
                            [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size

            out = fluid.layers.fc(input=data_1, size=1, num_flatten_dims=2)

        Then output is:
            out.data = [[0.83234344], [0.34936576]]
            out.shape = (1, 2, 1)

        Case 2:
        Given a list of Tensor:
            data_1.data = [[[0.1, 0.2],
                           [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size

            data_2 = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3)

            out = fluid.layers.fc(input=[data_1, data_2], size=2)

        Then:
            out.data = [[0.18669507, 0.1893476]]
            out.shape = (1, 2)

    Args:
        input (Variable|list of Variable): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` or
            a list of Tensor(or LoDTensor). The dimensions of the input Tensor is at least 2 and the data
            type should be float32 or float64.
        size(int): The number of output units in this layer, which also means the feature size of output
            Tensor(or LoDTensor).
        num_flatten_dims (int): The fc layer can accept an input Tensor with more than
            two dimensions. If this happens, the multidimensional tensor will first be flattened
            into a 2-D matrix. The parameter :attr:`num_flatten_dims` determines how the input
            Tensor is flattened: the first :attr:`num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest :math:`rank(X) - num\_flatten\_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, assuming that
            X is a 5-dimensional Tensor with a shape [2, 3, 4, 5, 6], and :attr:`num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default: 1.
        param_attr (ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr): To specify the bias parameter property. Default: None, which means the
            default bias parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        act (str): Activation to be applied to the output of this layer, such as tanh, softmax,
            sigmoid, relu. For more information, please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: Tensor or LoDTensor calculated by fc layer. The data type is same with input.

    Raises:
        ValueError: If dimensions of the input Tensor is less than 2.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle
          paddle.enable_static()
          # when input is single tensor
          data = fluid.data(name="data", shape=[-1, 32], dtype="float32")
          fc = fluid.layers.fc(input=data, size=1000, act="tanh")

          # when input are multiple tensors
          data_1 = fluid.data(name="data_1", shape=[-1, 32], dtype="float32")
          data_2 = fluid.data(name="data_2", shape=[-1, 36], dtype="float32")
          fc = fluid.layers.fc(input=[data_1, data_2], size=1000, act="tanh")
    """
    helper = LayerHelper("fc", **locals())
    check_type(input, 'input', (list, tuple, Variable), 'fc')
    if isinstance(input, (list, tuple)):
        for i, input_x in enumerate(input):
            check_type(input_x, 'input[' + str(i) + ']', Variable, 'fc')
    dtype = helper.input_dtype()
    check_dtype(dtype, 'input', ['float16', 'uint16', 'float32', 'float64'],
                'fc')
    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        if num_flatten_dims == -1:
            num_flatten_dims = len(input_shape) - 1
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False)
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="mul",
            inputs={"X": input_var,
                    "Y": w},
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": num_flatten_dims,
                   "y_num_col_dims": 1})
        mul_results.append(tmp)

    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)


@deprecated(since="2.0.0", update_to="paddle.nn.functional.embedding")
def embedding(input,
              size,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              param_attr=None,
              dtype='float32'):
    r"""
    :api_attr: Static Graph

    **WARING:** This OP will be deprecated in a future release. This OP requires the
    last dimension of Tensor shape must be equal to 1. It is recommended to use
    fluid. :ref:`api_fluid_embedding` .

    The operator is used to lookup embeddings vector of ids provided by :attr:`input` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`size` (vocab_size, emb_size) and :attr:`dtype` .

    This OP requires the last dimension of Tensor shape must be equal to 1. The shape
    of output Tensor is generated by replacing the last dimension of the input Tensor shape
    with emb_size.

    **Note:** The id in :attr:`input` must satisfy :math:`0 =< id < size[0]` ,
    otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        input is a Tensor. padding_idx = -1
            input.data = [[[1], [3]], [[2], [4]], [[4], [127]]]
            input.shape = [3, 2, 1]
        Given size = [128, 16]
        output is a Tensor:
            out.shape = [3, 2, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654]],

                        [[0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365]],

                        [[0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]]  # padding data
        The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
        It will pad all-zero data when ids is 127.

        Case 2:

        input is a LoDTensor with 1-level LoD. padding_idx = 0
            input.lod = [[2, 3]]
            input.data = [[1], [3], [2], [4], [0]]
            input.shape = [5, 1]
        Given size = [128, 16]
        output is a LoDTensor:
            out.lod = [[2, 3]]
            out.shape = [5, 16]
            out.data = [[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654],
                        [0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]  # padding data
        It will pad all-zero data when ids is 0.

    Args:
        input(Variable): A Tensor or LoDTensor with type int64, which contains the id information.
            The last dimension of Tensor shape must be equal to 1. The value of the input id should
            satisfy :math:`0<= id < size[0]` .
        size(tuple|list): The shape of lookup table parameter. It should have two elements which
            indicates the size of the dictionary of embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_fluid_optimizer_AdadeltaOptimizer` , :ref:`api_fluid_optimizer_AdamaxOptimizer` ,
            :ref:`api_fluid_optimizer_DecayedAdagradOptimizer` , :ref:`api_fluid_optimizer_FtrlOptimizer` ,
            :ref:`api_fluid_optimizer_LambOptimizer` and :ref:`api_fluid_optimizer_LarsMomentumOptimizer` .
            In these case, is_sparse must be False. Default: False.
        is_distributed(bool): Whether to store the embedding matrix in a distributed manner. Only used
            in multi-machine distributed CPU training. Default: False.
        padding_idx(int|long|None): padding_idx needs to be in the interval [-vocab_size, vocab_size).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        param_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`size` . Then :ref:`api_fluid_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example 2 for details.
        dtype(str|core.VarDesc.VarType): It refers to the data type of output Tensor.
            It must be float32 or float64. Default: float32.

    Returns:
        Variable: Embedding Tensor or LoDTensor mapped by input. The data type is the same as :attr:`dtype` .

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          import paddle
          paddle.enable_static()
          
          data = fluid.data(name='x', shape=[None, 1], dtype='int64')

          # example 1
          emb_1 = fluid.embedding(input=data, size=[128, 64])

          # example 2: load custom or pre-trained word vectors
          weight_data = np.random.random(size=(128, 100))  # word vectors with numpy format
          w_param_attrs = fluid.ParamAttr(
              name="emb_weight",
              learning_rate=0.5,
              initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
              trainable=True)
          emb_2 = fluid.layers.embedding(input=data, size=(128, 100), param_attr=w_param_attrs, dtype='float32')
    """

    helper = LayerHelper('embedding', **locals())
    check_variable_and_dtype(input, 'input', ['int64'],
                             'fluid.layers.embedding')
    check_dtype(dtype, 'dtype', ['uint16', 'float16', 'float32', 'float64'],
                'fluid.layers.embedding')

    if is_distributed:
        is_distributed = False
        warnings.warn(
            "is_distributed is go out of use, `fluid.contrib.layers.sparse_embedding` is your needed"
        )

    remote_prefetch = True if is_sparse else False

    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_variable_for_type_inference(dtype)
    padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
        size[0] + padding_idx)
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': tmp},
        attrs={
            'is_sparse': is_sparse,
            'is_distributed': is_distributed,
            'remote_prefetch': remote_prefetch,
            'padding_idx': padding_idx
        })
    return tmp


def _pull_sparse(input,
                 size,
                 table_id,
                 accessor_class,
                 name="embedding",
                 ctr_label_name="",
                 padding_id=0,
                 dtype='float32',
                 scale_sparse_grad=True):
    r"""
    **Pull Fleet Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    Fleet lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which
            contains the IDs information.
        size(int): The embedding size parameter, which indicates the size of
            each embedding vector respectively.
        table_id(int): the fleet table id of this embedding.
        accessor_class(str): the pslib accessor of the table, default is DownpourCtrAccessor.
        ctr_label_name(str): the layer name of click.
        padding_id(int): the padding id during lookup, default is 0.
        dtype(str): The dtype refers to the data type of output tensor. Only supports
            float32 now.
        scale_sparse_grad(bool): whether to scale sparse gradient with batch size. default
            is True.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.nn._pull_sparse(
              input=data, size=11, table_id=0, accessor_class="DownpourCtrAccessor")
    """
    helper = LayerHelper(name, **locals())
    inputs = helper.multiple_input()
    outs = [helper.create_variable_for_type_inference(dtype)]
    input_names = [i.name for i in inputs]
    attrs = {
        'EmbeddingDim': size,
        'TableId': table_id,
        'AccessorClass': accessor_class,
        'CtrLabelName': ctr_label_name,
        'PaddingId': padding_id,
        'ScaleSparseGrad': scale_sparse_grad,
        'InputNames': input_names,
        # this is only for compatible with embedding op
        'is_distributed': True
    }
    # this is only for compatible with embedding op
    w, _ = helper.create_or_get_global_variable(
        name=name, shape=[size], dtype=dtype, is_bias=False, persistable=True)
    helper.append_op(
        type='pull_sparse',
        inputs={'Ids': inputs,
                'W': w},
        outputs={'Out': outs},
        attrs=attrs)
    if len(outs) == 1:
        return outs[0]
    return outs


def _pull_sparse_v2(input,
                    size,
                    table_id,
                    accessor_class,
                    name="embedding",
                    ctr_label_name="",
                    padding_id=0,
                    dtype='float32',
                    scale_sparse_grad=True):
    r"""
    **Pull Fleet Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    Fleet lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which
            contains the IDs information.
        size(int): The embedding size parameter, which indicates the size of
            each embedding vector respectively.
        table_id(int): the pslib table id of this embedding.
        accessor_class(str): the fleet accessor of the table, default is DownpourCtrAccessor.
        ctr_label_name(str): the layer name of click.
        padding_id(int): the padding id during lookup, default is 0.
        dtype(str): The dtype refers to the data type of output tensor. Only supports
            float32 now.
        scale_sparse_grad(bool): whether to scale sparse gradient with batch size. default
            is True.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.nn._pull_sparse_v2(
              input=data, size=11, table_id=0, accessor_class="DownpourCtrAccessor")
    """
    helper = LayerHelper(name, **locals())
    inputs = helper.multiple_input()
    outs = [helper.create_variable_for_type_inference(dtype)]
    input_names = [i.name for i in inputs]
    attrs = {
        'EmbeddingDim': size,
        'TableId': table_id,
        'AccessorClass': accessor_class,
        'CtrLabelName': ctr_label_name,
        'PaddingId': padding_id,
        'ScaleSparseGrad': scale_sparse_grad,
        'InputNames': input_names,
        # this is only for compatible with embedding op
        'is_distributed': True
    }
    # this is only for compatible with embedding op
    w, _ = helper.create_or_get_global_variable(
        name=name, shape=[size], dtype=dtype, is_bias=False, persistable=True)
    helper.append_op(
        type='pull_sparse_v2',
        inputs={'Ids': inputs,
                'W': w},
        outputs={'Out': outs},
        attrs=attrs)
    if len(outs) == 1:
        return outs[0]
    return outs


def _pull_box_sparse(input,
                     size,
                     dtype='float32',
                     is_distributed=False,
                     is_sparse=False):
    r"""
    **Pull Box Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    BoxPS lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which
            contains the IDs information.
        size(int): The embedding size parameter, which indicates the size of
            each embedding vector respectively.
        dtype(str): The dtype refers to the data type of output tensor. Only supports
	    float32 now.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.pull_box_sparse(input=data, size=[11])
    """
    helper = LayerHelper('pull_box_sparse', **locals())
    if dtype != 'float32':
        raise ValueError(
            "BoxPS only support float type embedding now, and your type is: " +
            dtype)
    helper.input_dtype()
    inputs = helper.multiple_input()
    outs = [
        helper.create_variable_for_type_inference(dtype)
        for i in range(len(inputs))
    ]
    w = helper.create_parameter(
        attr=helper.param_attr, shape=[size], dtype=dtype, is_bias=False)
    helper.append_op(
        type='pull_box_sparse',
        inputs={'Ids': inputs,
                'W': w},
        outputs={'Out': outs},
        attrs={
            'size': size,
            'is_distributed': is_distributed,
            'is_sparse': is_sparse
        })
    if len(outs) == 1:
        return outs[0]
    return outs


@templatedoc()
def linear_chain_crf(input, label, param_attr=None, length=None):
    """
    :api_attr: Static Graph

    Linear Chain CRF.

    ${comment}

    Args:
        input(${emission_type}): ${emission_comment}
        label(${label_type}): ${label_comment}
        Length(${length_type}): ${length_comment}
        param_attr(ParamAttr): The attribute of the learnable parameter for transition parameter.

    Returns:
        output(${emission_exps_type}): ${emission_exps_comment} \n
        output(${transition_exps_type}): ${transition_exps_comment} \n
        output(${log_likelihood_type}): ${log_likelihood_comment} \n

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            paddle.enable_static()

            #define net structure, using LodTensor
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                input_data = fluid.data(name='input_data', shape=[-1,10], dtype='float32')
                label = fluid.data(name='label', shape=[-1,1], dtype='int')
                emission= fluid.layers.fc(input=input_data, size=10, act="tanh")
                crf_cost = fluid.layers.linear_chain_crf(
                    input=emission,
                    label=label,
                    param_attr=fluid.ParamAttr(
                    name='crfw',
                    learning_rate=0.01))
            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            #define data, using LoDTensor
            a = fluid.create_lod_tensor(np.random.rand(12,10).astype('float32'), [[3,3,4,2]], place)
            b = fluid.create_lod_tensor(np.array([[1],[1],[2],[3],[1],[1],[1],[3],[1],[1],[1],[1]]),[[3,3,4,2]] , place)
            feed1 = {'input_data':a,'label':b}
            loss= exe.run(train_program,feed=feed1, fetch_list=[crf_cost])
            print(loss)

            #define net structure, using padding
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                input_data2 = fluid.data(name='input_data2', shape=[-1,10,10], dtype='float32')
                label2 = fluid.data(name='label2', shape=[-1,10,1], dtype='int')
                label_length = fluid.data(name='length', shape=[-1,1], dtype='int')
                emission2= fluid.layers.fc(input=input_data2, size=10, act="tanh", num_flatten_dims=2)
                crf_cost2 = fluid.layers.linear_chain_crf(
                    input=emission2,
                    label=label2,
                    length=label_length,
                    param_attr=fluid.ParamAttr(
                     name='crfw',
                     learning_rate=0.01))

            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)

            #define data, using padding
            cc=np.random.rand(4,10,10).astype('float32')
            dd=np.random.rand(4,10,1).astype('int64')
            ll=np.array([[3],[3],[4],[2]])
            feed2 = {'input_data2':cc,'label2':dd,'length':ll}
            loss2= exe.run(train_program,feed=feed2, fetch_list=[crf_cost2])
            print(loss2)
            #[array([[ 7.8902354],
            #        [ 7.3602567],
            #        [ 10.004011],
            #        [ 5.86721  ]], dtype=float32)]

            #you can use find_var to get transition parameter.
            transition=np.array(fluid.global_scope().find_var('crfw').get_tensor())
            print(transition)

    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'linear_chain_crf')
    check_variable_and_dtype(label, 'label', ['int64'], 'linear_chain_crf')
    helper = LayerHelper('linear_chain_crf', **locals())
    size = input.shape[2] if length else input.shape[1]
    transition = helper.create_parameter(
        attr=helper.param_attr,
        shape=[size + 2, size],
        dtype=helper.input_dtype())
    alpha = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    emission_exps = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    transition_exps = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    log_likelihood = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    this_inputs = {
        "Emission": [input],
        "Transition": transition,
        "Label": [label]
    }
    if length:
        this_inputs['Length'] = [length]
    helper.append_op(
        type='linear_chain_crf',
        inputs=this_inputs,
        outputs={
            "Alpha": [alpha],
            "EmissionExps": [emission_exps],
            "TransitionExps": transition_exps,
            "LogLikelihood": log_likelihood
        })

    return log_likelihood


@templatedoc()
def crf_decoding(input, param_attr, label=None, length=None):
    """
    :api_attr: Static Graph

    ${comment}

    Args:
        input(Tensor): ${emission_comment}

        param_attr (ParamAttr|None): To specify the weight parameter attribute.
            Default: None, which means the default weight parameter property is
            used. See usage for details in :ref:`api_paddle_fluid_param_attr_ParamAttr` .

        label(${label_type}, optional): ${label_comment}

        length(${length_type}, optional): ${length_comment}

    Returns:
        Tensor: ${viterbi_path_comment}

    Examples:
        .. code-block:: python

           import paddle
           paddle.enable_static()

           # LoDTensor-based example
           num_labels = 10
           feature = paddle.static.data(name='word_emb', shape=[-1, 784], dtype='float32', lod_level=1)
           label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64', lod_level=1)
           emission = paddle.static.nn.fc(feature, size=num_labels)

           crf_cost = paddle.fluid.layers.linear_chain_crf(input=emission, label=label,
                     param_attr=paddle.ParamAttr(name="crfw"))
           crf_decode = paddle.static.nn.crf_decoding(input=emission,
                     param_attr=paddle.ParamAttr(name="crfw"))

           # Common tensor example
           num_labels, max_len = 10, 20
           feature = paddle.static.data(name='word_emb_pad', shape=[-1, max_len, 784], dtype='float32')
           label = paddle.static.data(name='label_pad', shape=[-1, max_len, 1], dtype='int64')
           length = paddle.static.data(name='length', shape=[-1, 1], dtype='int64')
           emission = paddle.static.nn.fc(feature, size=num_labels,
                                      num_flatten_dims=2)

           crf_cost = paddle.fluid.layers.linear_chain_crf(input=emission, label=label, length=length,
                     param_attr=paddle.ParamAttr(name="crfw_pad"))
           crf_decode = paddle.static.nn.crf_decoding(input=emission, length=length,
                     param_attr=paddle.ParamAttr(name="crfw_pad"))
    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'crf_decoding')
    helper = LayerHelper('crf_decoding', **locals())
    transition = helper.get_parameter(param_attr.name)
    viterbi_path = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.INT64)
    inputs = {"Emission": [input], "Transition": transition, "Label": label}
    if length:
        inputs['Length'] = length
    helper.append_op(
        type='crf_decoding',
        inputs=inputs,
        outputs={"ViterbiPath": [viterbi_path]})

    return viterbi_path


@templatedoc()
def cos_sim(X, Y):
    """
    ${comment}

    Args:
        X (Tensor): ${x_comment}.
        Y (Tensor): ${y_comment}.

    Returns:
        A Tensor representing the output of cosine(X, Y).

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.rand(shape=[3, 7], dtype='float32')
            y = paddle.rand(shape=[1, 7], dtype='float32')
            out = paddle.fluid.layers.cos_sim(x, y)
            print(out)

    """
    check_variable_and_dtype(X, 'X', ['float32'], 'cos_sim')
    check_variable_and_dtype(Y, 'Y', ['float32'], 'cos_sim')
    helper = LayerHelper('cos_sim', **locals())
    out = helper.create_variable_for_type_inference(dtype=X.dtype)
    xnorm = helper.create_variable_for_type_inference(dtype=X.dtype)
    ynorm = helper.create_variable_for_type_inference(dtype=X.dtype)
    helper.append_op(
        type='cos_sim',
        inputs={'X': [X],
                'Y': [Y]},
        outputs={'Out': [out],
                 'XNorm': [xnorm],
                 'YNorm': [ynorm]})
    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.dropout")
def dropout(x,
            dropout_prob,
            is_test=None,
            seed=None,
            name=None,
            dropout_implementation="downgrade_in_infer"):
    """

    Computes dropout.

    Drop or keep each element of `x` independently. Dropout is a regularization
    technique for reducing overfitting by preventing neuron co-adaption during
    training. The dropout operator randomly sets (according to the given dropout
    probability) the outputs of some units to zero, while others are remain
    unchanged.

    dropout op can be removed from the program to make the program more efficient.

    Args:
        x (Variable): The input tensor variable. The data type is float16 or float32 or float64.
        dropout_prob (float): Probability of setting units to zero.
        is_test (bool): A flag indicating whether it is in test phrase or not. 
                        Default None, in dynamic graph, it use global tracer mode; in static graph, it means False.
        seed (int): A Python integer used to create random seeds. If this
                    parameter is set to None, a random seed is used.
                    NOTE: If an integer seed is given, always the same output
                    units will be dropped. DO NOT use a fixed seed in training.Default: None.
        name (str|None): A name for this layer(optional). If set None, the layer
                         will be named automatically.
        dropout_implementation(string): ['downgrade_in_infer'(default)|'upscale_in_train']

                                        1. downgrade_in_infer(default), downgrade the outcome at inference

                                           - train: out = input * mask
                                           - inference: out = input * (1.0 - dropout_prob)

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)
                                        2. upscale_in_train, upscale the outcome at training time

                                           - train: out = input * mask / ( 1.0 - dropout_prob )
                                           - inference: out = input

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)


    Returns:
        A Variable holding Tensor representing the dropout, has same shape and data type with `x`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            
            paddle.enable_static()
            x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
            dropped = fluid.layers.dropout(x, dropout_prob=0.5)
    """
    # fast return for p == 0
    if dropout_prob == 0:
        return x

    if in_dygraph_mode():
        if (seed is None or
                seed == 0) and default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        if is_test is None:
            is_test = not _dygraph_tracer()._train_mode
        out, mask = _C_ops.dropout(
            x, 'dropout_prob', dropout_prob, 'is_test', is_test, 'fix_seed',
            seed is not None, 'seed', seed if seed is not None else 0,
            'dropout_implementation', dropout_implementation)
        return out

    def get_attrs(prog, dropout_prob, is_test, seed):
        if (seed is None or seed == 0) and prog.random_seed != 0:
            seed = prog.random_seed
        attrs = {
            'dropout_prob': dropout_prob,
            'is_test': is_test,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0,
            'dropout_implementation': dropout_implementation,
        }
        return attrs

    helper = LayerHelper('dropout', **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'dropout')

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    mask = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)

    attrs = get_attrs(helper.main_program, dropout_prob, is_test, seed)

    helper.append_op(
        type='dropout',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'Mask': [mask]},
        attrs=attrs)
    return out


@templatedoc()
def chunk_eval(input,
               label,
               chunk_scheme,
               num_chunk_types,
               excluded_chunk_types=None,
               seq_length=None):
    r"""
    This operator computes the precision, recall and F1-score for chunk detection.
    It is often used in sequence tagging tasks, such as Named Entity Recognition(NER).

    For some basics of chunking, please refer to
    `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_ .

    This operator supports IOB, IOE, IOBES and IO (also known as plain) tagging schemes.
    Here is a NER example for the usage of these tagging schemes:

    .. code-block:: python

       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
              Li     Ming    works  at  Agricultural   Bank   of    China  in  Beijing.
       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
       IO     I-PER  I-PER   O      O   I-ORG          I-ORG  I-ORG I-ORG  O   I-LOC
       IOB    B-PER  I-PER   O      O   B-ORG          I-ORG  I-ORG I-ORG  O   B-LOC
       IOE    I-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   E-LOC
       IOBES  B-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   S-LOC
       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========

    There are three chunk types(named entity types) including PER(person), ORG(organization)
    and LOC(location), and we can see that the labels have the form `<tag type>-<chunk type>` .

    Since the implementation of this operator actually uses label ids rather than
    label strings, to make it work, there should be a way to map label ids to
    tag types and chunk types. This operator uses the following way to do mapping:

    .. code-block:: python

       tag_type = label % num_tag_type
       chunk_type = label / num_tag_type

    where `num_tag_type` is the num of tag types in the tagging scheme, `num_chunk_type`
    is the num of chunk types, and `tag_type` get its value from the following table.

    .. code-block:: python

       Scheme Begin Inside End   Single
        plain   0     -      -     -
        IOB     0     1      -     -
        IOE     -     0      1     -
        IOBES   0     1      2     3

    Accordingly, in the above NER example, if the tagging scheme is IOB and chunk
    types are ORG, PER and LOC, then the label ids would be as follows:

    .. code-block:: python

       B-ORG  0
       I-ORG  1
       B-PER  2
       I-PER  3
       B-LOC  4
       I-LOC  5
       O      6

    With which we can map each label id to the corresponding tag type and chunk
    type correctly.

    Args:
        input (Tensor): A Tensor representing the predicted labels
            from the network. Its shape would be `[N, M, 1]`,
            where `N` stands for batch size, `M` for sequence length. 
            The data type should be int64.
        label (Tensor): A Tensor representing the ground-truth labels.
            It should have the same shape, lod and data type as ``input`` .
        chunk_scheme (str): Indicate the tagging schemes used here. The value must
            be IOB, IOE, IOBES or plain.
        num_chunk_types (int): The number of chunk types.
        excluded_chunk_types (list, optional): Indicate the chunk types shouldn't
            be taken into account. It should be a list of chunk type ids(integer).
            Default None.
        seq_length(Tensor, optional): A 1D Tensor containing the length of each
            sequence when ``input`` and ``label`` are Tensor. Default None.

    Returns:
        tuple: A tuple including precision, recall, F1-score, chunk number detected, \
            chunk number in ground-truth, chunk number correctly detected. Each \
            is a Tensor with shape `[1]`. The data type of precision, recall and \
            F1-score all is float32, and the others' data type all is int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            dict_size = 10000
            label_dict_len = 7
            sequence = fluid.data(
                name='id', shape=[None, 1], lod_level=1, dtype='int64')
            embedding = fluid.embedding(
                input=sequence, size=[dict_size, 512])
            hidden = fluid.layers.fc(input=embedding, size=512)
            label = fluid.data(
                name='label', shape=[None, 1], lod_level=1, dtype='int64')
            crf = fluid.layers.linear_chain_crf(
                input=hidden, label=label, param_attr=fluid.ParamAttr(name="crfw"))
            crf_decode = fluid.layers.crf_decoding(
                input=hidden, param_attr=fluid.ParamAttr(name="crfw"))
            fluid.layers.chunk_eval(
                input=crf_decode,
                label=label,
                chunk_scheme="IOB",
                num_chunk_types=int((label_dict_len - 1) / 2))
    """
    helper = LayerHelper("chunk_eval", **locals())

    check_variable_and_dtype(input, 'input', ['int64'], 'chunk_eval')
    check_variable_and_dtype(label, 'label', ['int64'], 'chunk_eval')

    # prepare output
    precision = helper.create_variable_for_type_inference(dtype="float32")
    recall = helper.create_variable_for_type_inference(dtype="float32")
    f1_score = helper.create_variable_for_type_inference(dtype="float32")
    num_infer_chunks = helper.create_variable_for_type_inference(dtype="int64")
    num_label_chunks = helper.create_variable_for_type_inference(dtype="int64")
    num_correct_chunks = helper.create_variable_for_type_inference(
        dtype="int64")

    this_input = {"Inference": [input], "Label": [label]}

    if seq_length is not None:
        this_input["SeqLength"] = [seq_length]

    helper.append_op(
        type="chunk_eval",
        inputs=this_input,
        outputs={
            "Precision": [precision],
            "Recall": [recall],
            "F1-Score": [f1_score],
            "NumInferChunks": [num_infer_chunks],
            "NumLabelChunks": [num_label_chunks],
            "NumCorrectChunks": [num_correct_chunks]
        },
        attrs={
            "num_chunk_types": num_chunk_types,
            "chunk_scheme": chunk_scheme,
            "excluded_chunk_types": excluded_chunk_types or []
        })
    return (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
            num_correct_chunks)


@deprecated(since="2.0.0", update_to="paddle.nn.functional.softmax")
def softmax(input, use_cudnn=True, name=None, axis=-1):
    r"""
    This operator implements the softmax layer. The calculation process is as follows:

    1. The dimension :attr:`axis` of the ``input`` will be permuted to the last.

    2. Then the input tensor will be logically flattened to a 2-D matrix. The matrix's
    second dimension(row length) is the same as the dimension :attr:`axis` of the input
    tensor, and the first dimension(column length) is the product of all other
    dimensions of the input tensor. For each row of the matrix, the softmax operator
    squashes the K-dimensional(K is the width of the matrix, which is also the size
    of the input tensor's dimension :attr:`axis`) vector of arbitrary real values to a
    K-dimensional vector of real values in the range [0, 1] that add up to 1.

    3. After the softmax operation is completed, the inverse operations of steps 1 and 2
    are performed to restore the two-dimensional matrix to the same dimension as the ``input``.

    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.

    For each row :math:`i` and each column :math:`j` in the matrix, we have:

    .. math::

        Out[i, j] = \\frac{\\exp(X[i, j])}{\\sum_j(exp(X[i, j])}

    Example:

    .. code-block:: text

        Case 1:
          Input:
            X.shape = [2, 3, 4]
            X.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]

          Attrs:
            axis = -1

          Output:
            Out.shape = [2, 3, 4]
            Out.data = [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                        [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]

        Case 2:
          Input:
            X.shape = [2, 3, 4]
            X.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]
          Attrs:
            axis = 1

          Output:
            Out.shape = [2, 3, 4]
            Out.data = [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
                         [0.01786798, 0.01786798, 0.04661262, 0.04661262],
                         [0.97555875, 0.97555875, 0.93623955, 0.93623955]],
                        [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
                         [0.26762315, 0.26762315, 0.26762315, 0.26762315],
                         [0.72747516, 0.72747516, 0.72747516, 0.72747516]]]

    Args:
        input (Tensor): The input tensor. A multi-dimension ``Tensor`` with type float32 or float64.
        use_cudnn (bool, optional): Use cudnn kernel or not, it is valid only when the cudnn \
            library is installed. To improve performance, set use_cudnn to True by default.
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` . Default: None.
            will be named automatically. Default: None.
        axis (int, optional): The index of dimension to perform softmax calculations, it should
            be in range :math:`[-1, rank - 1]`, while :math:`rank` is the rank of
            input tensor. Default: -1. -1 means the last dimension.

    Returns:
        Tensor: ``Tensor`` indicates the output of softmax. The data type and shape are the same as ``input`` .

    Examples:

        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            x = paddle.to_tensor([[[2.0, 3.0, 4.0, 5.0],
                                [3.0, 4.0, 5.0, 6.0],
                                [7.0, 8.0, 8.0, 9.0]],
                                [[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0],
                                [6.0, 7.0, 8.0, 9.0]]], dtype='float32')
            y = F.softmax(x, axis=1)
            print(y)
            # [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
            #   [0.01786798, 0.01786798, 0.04661262, 0.04661262],
            #   [0.97555870, 0.97555870, 0.93623954, 0.93623954]],
            #  [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
            #   [0.26762316, 0.26762316, 0.26762316, 0.26762316],
            #   [0.72747517, 0.72747517, 0.72747517, 0.72747517]]]

    """

    if in_dygraph_mode():
        return _C_ops.softmax(input, 'axis', axis, 'use_cudnn', use_cudnn)

    inputs = {"X": [input]}
    attrs = {"axis": axis, "use_cudnn": use_cudnn}

    helper = LayerHelper('softmax', **locals())
    check_variable_and_dtype(input, 'input/x',
                             ['float16', 'float32', 'float64'], 'softmax')

    dtype = helper.input_dtype()
    softmax_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs=attrs)
    return softmax_out


def conv2d(input,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           name=None,
           data_format="NCHW"):
    r"""
    :api_attr: Static Graph

    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW or NHWC format, where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    Filter is in MCHW format, where M is the number of output image channels,
    C is the number of input image channels, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input image channels divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Tensor): The input is 4-D Tensor with shape [N, C, H, W], the data type
            of input is float16 or float32 or float64.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size
            is a tuple, it must contain two integers, (filter_size_height,
            filter_size_width). Otherwise, filter_size_height = filter_size_width =\
            filter_size.
        stride (int|tuple): The stride size. It means the stride in convolution.
            If stride is a tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings
            on both sides for each dimension.If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when
            `data_format` is `"NCHW"`, `padding` can be in the form `[[0,0], [0,0],
            [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|tuple): The dilation size. It means the spacing between the kernel
            points. If dilation is a tuple, it must contain two integers, (dilation_height,
            dilation_width). Otherwise, dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        name(str|None): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        A Tensor representing the conv2d, whose data type is the
        same with input. If act is None, the tensor storing the convolution
        result, and if act is not None, the tensor storing convolution
        and non-linearity activation result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If the channel dimmention of the input is less than or equal to zero.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0
            or the element corresponding to the input's channel is not 0.
        ShapeError: If the input is not 4-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels * groups.
        ShapeError: If the number of output channels is not be divided by groups.

    Examples:
        .. code-block:: python

          import paddle
          paddle.enable_static()
          
          data = paddle.static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          conv2d = paddle.static.nn.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
          print(conv2d.shape) # [-1, 2, 30, 30]
    """

    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'conv2d')
    if len(input.shape) != 4:
        raise ValueError("Input size should be 4, "
                         "but received {}".format(len(input.shape)))
    num_channels = input.shape[1]
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. Received "
                         "Attr(use_cudnn): %s. " % str(use_cudnn))

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    channel_last = (data_format == "NHWC")
    num_channels = input.shape[3] if channel_last else input.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input(%s) should be defined. "
            "Received: %s." % (str(input.shape), str(num_channels)))
    assert param_attr is not False, "param_attr should not be False here."

    if groups is None:
        num_filter_channels = num_channels
    elif groups <= 0:
        raise ValueError("the groups of input must be greater than 0, "
                         "but received the groups of input is {}".format(
                             groups))
    else:
        if num_channels % groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(num_channels, input.shape, groups))
        num_filter_channels = num_channels // groups

    l_type = 'conv2d'
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'

    if (num_channels == groups and num_filters % num_channels == 0 and
            core.is_compiled_with_rocm()):
        l_type = 'depthwise_conv2d'

    # NPU only supports depthwise_conv2d when  "input_channel = output_channel = groups"
    if core.is_compiled_with_npu():
        if (num_channels == groups and num_channels == num_filters):
            l_type = 'depthwise_conv2d'
        else:
            l_type = 'conv2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    # padding
    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')
            if utils._is_symmetric_padding(padding, 2):
                padding = [padding[0], padding[2]]

        else:
            padding = utils.convert_to_list(padding, 2, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'." %
                str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0]

    padding = _update_padding(padding, data_format)

    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        if filter_elem_num <= 0:
            raise ValueError(
                "Invalid filter number, excepted number is larger than 0, but"
                " received {}, please check the input shape and "
                "filter size.".format(filter_elem_num))
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if (core.is_compiled_with_cuda() and paddle.fluid.get_flags(
            "FLAGS_conv2d_disable_cudnn")["FLAGS_conv2d_disable_cudnn"]):
        use_cudnn = False

    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter_param,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format,
        })

    if data_format == 'NCHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=3, dim_end=4)

    return helper.append_activation(pre_act)


def conv3d(input,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           name=None,
           data_format="NCDHW"):
    r"""
    :api_attr: Static Graph

    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCDHW or NDHWC format. Where N is batch size C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. Convlution3D is similar with Convlution2D
    but adds one dimension(depth). If bias attribution and activation type are
    provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`

        - Output:
          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

            D_{out}&= \\frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\\\
            H_{out}&= \\frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

    Args:
        input (Tensor): The input is 5-D Tensor with shape [N, C, D, H, W], the data
            type of input is float16 or float32 or float64.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_depth, filter_size_height,
            filter_size_width). Otherwise, filter_size_depth = filter_size_height = \
            filter_size_width = filter_size.
        stride (int|tuple): The stride size. It means the stride in convolution. If stride is a
            tuple, it must contain three integers, (stride_depth, stride_height, stride_width).
            Otherwise, stride_depth = stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings
            on both sides for each dimension. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `pool_padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|tuple): The dilation size. It means the spacing between the kernel points.
            If dilation is a tuple, it must contain three integers, (dilation_depth, dilation_height,
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int): The groups number of the Conv3d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str|None): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        A Variable holding Tensor representing the conv3d, whose data type is
        the same with input. If act is None, the tensor variable storing the
        convolution result, and if act is not None, the tensor variable storing
        convolution and non-linearity activation result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".
        ValueError: If the channel dimmention of the input is less than or equal to zero.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0
            or the element corresponding to the input's channel is not 0.
        ShapeError: If the input is not 5-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels * groups.
        ShapeError: If the number of output channels is not be divided by groups.

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np
	  
          paddle.enable_static()
          data = paddle.static.data(name='data', shape=[None, 3, 12, 32, 32], dtype='float32')
          param_attr = paddle.framework.ParamAttr(name='conv3d.weight', initializer=paddle.nn.initializer.XavierNormal(), learning_rate=0.001)
          res = paddle.static.nn.conv3d(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
          place = paddle.CPUPlace()
          exe = paddle.static.Executor(place)
          exe.run(paddle.static.default_startup_program())
          x = np.random.rand(1, 3, 12, 32, 32).astype("float32")
          output = exe.run(feed={"data": x}, fetch_list=[res])
          print(output)
    """

    l_type = 'conv3d'
    assert param_attr is not False, "param_attr should not be False here."
    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. Received "
                         "Attr(use_cudnn): %s. " % str(use_cudnn))

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    channel_last = (data_format == "NDHWC")
    if len(input.shape) != 5:
        raise ValueError(
            "Input should be 5D tensor, but received input with the shape of {}".
            format(input.shape))
    num_channels = input.shape[4] if channel_last else input.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input(%s) should be defined. "
            "Received: %s." % (str(input.shape), str(num_channels)))

    if groups is None:
        num_filter_channels = num_channels
    elif groups <= 0:
        raise ValueError(
            "the groups of conv3d should be greater than 0. Received groups: {}".
            format(groups))
    else:
        if num_channels % groups != 0:
            raise ValueError(
                "The number of input channels must be divisible by Attr(groups). "
                "Received: number of channels(%s), groups(%s)." %
                (str(num_channels), str(groups)))
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 3, 'filter_size')
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 5:
            if is_list_or_tuple(padding[0]) and (data_format == "NCDHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:5]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NDHWC"):
                if not (padding[0] == [0, 0] and padding[4] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:4]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 6, 'padding')
            if utils._is_symmetric_padding(padding, 3):
                padding = [padding[0], padding[2], padding[4]]
        elif is_list_or_tuple(padding) and len(padding) == 6:
            padding = utils.convert_to_list(padding, 6, 'padding')
            if utils._is_symmetric_padding(padding, 3):
                padding = [padding[0], padding[2], padding[4]]
        else:
            padding = utils.convert_to_list(padding, 3, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'." %
                str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0]

    padding = _update_padding(padding, data_format)

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * filter_size[
            2] * num_channels
        if filter_elem_num <= 0:
            raise ValueError(
                "Invalid filter number, excepted number is larger than 0, but"
                " received {}, please check the input shape and "
                "filter size.".format(filter_elem_num))

        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter_param,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format,
        })

    if data_format == 'NCDHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=4, dim_end=5)

    return helper.append_activation(pre_act)


@templatedoc()
def pool2d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           name=None,
           exclusive=True,
           data_format="NCHW"):
    """

    ${comment}

    Args:
        input (Variable): The input tensor of pooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` or
                          `"NHWC"`, where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        pool_type: ${pooling_type_comment}
        pool_stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
        pool_padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when `data_format` is `"NCHW"`,
            `pool_padding` can be in the form `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Otherwise, the pool padding size will be a square of an int.
        global_pooling (bool): ${global_pooling_comment}
        use_cudnn (bool): ${use_cudnn_comment}
        ceil_mode (bool): ${ceil_mode_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NHWC"`.
                The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `pool_type` is not "max" nor "avg".
        ValueError: If `global_pooling` is False and `pool_size` is -1.
        TypeError: If `use_cudnn` is not a bool value.
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If `pool_padding` is a string, but not "SAME" or "VALID".
        ValueError: If `pool_padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `pool_padding` is a list or tuple, but the elements in the batch or channel dimensions are non-zero.
        ShapeError: If the input is not a 4-D or 5-D Tensor.
        ShapeError: If the dimension of input minus the size of `pool_stride` is not 2.
        ShapeError: If the size of `pool_size` and `pool_stride` is not equal.
        ShapeError: If the output's shape calculated is not greater than 0.


    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import paddle

          paddle.enable_static()

          data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')

          # max pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "max",
            pool_stride = 1,
            global_pooling=False)

          # average pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=False)

          # global average pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=True)

          # Attr(pool_padding) is a list with 4 elements, Attr(data_format) is "NCHW".
          out_1 = fluid.layers.pool2d(
            input = data,
            pool_size = 3,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = [1, 2, 1, 0],
            data_format = "NCHW")

          # Attr(pool_padding) is a string, Attr(data_format) is "NCHW".
          out_2 = fluid.layers.pool2d(
            input = data,
            pool_size = 3,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = "VALID",
            data_format = "NCHW")
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown Attr(pool_type): '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When Attr(global_pooling) is False, Attr(pool_size) must be passed "
            "and be a valid value. Received pool_size: %s." % str(pool_size))

    if not isinstance(use_cudnn, bool):
        raise TypeError("Attr(use_cudnn) should be True or False. Received "
                        "Attr(use_cudnn): %s." % str(use_cudnn))

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
    pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')

    def update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')

            if utils._is_symmetric_padding(padding, 2):
                padding = [padding[0], padding[2]]
        else:
            padding = utils.convert_to_list(padding, 2, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(pool_padding, str):
        pool_padding = pool_padding.upper()
        if pool_padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(pool_padding))
        if pool_padding == "VALID":
            padding_algorithm = "VALID"
            pool_padding = [0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif pool_padding == "SAME":
            padding_algorithm = "SAME"
            pool_padding = [0, 0]

    pool_padding = update_padding(pool_padding, data_format)

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": use_cudnn,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": data_format,
        })

    return pool_out


@templatedoc()
def pool3d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           name=None,
           exclusive=True,
           data_format="NCDHW"):
    """

    ${comment}

    Args:
        input (Variable): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W]. The format of
                          input tensor is `"NCDHW"` or `"NDHWC"`, where `N` is batch size, `C` is
                          the number of channels, `D` is the depth of the feature,
                          `H` is the height of the feature, and `W` is the width
                          of the feature.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size
            is a tuple or list, it must contain three integers,
            (pool_size_Depth, pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        pool_type (string): ${pooling_type_comment}
        pool_stride (string|int|list|tuple)): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool stride size is a tuple or list,
            it must contain three integers, `[stride_Depth, stride_Height, stride_Width]`.
            Otherwise, the pool stride size will be a cube of an int.
        pool_padding (int|list|tuple): The pool padding size. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `pool_padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
        global_pooling (bool): ${global_pooling_comment}
        use_cudnn (bool): ${use_cudnn_comment}
        ceil_mode (bool): ${ceil_mode_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is true.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                `[batch_size, input_channels, input_depth, input_height, input_width]`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `pool_type` is not "max" nor "avg".
        ValueError: If `global_pooling` is False and `pool_size` is -1.
        TypeError: If `use_cudnn` is not a bool value.
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".
        ValueError: If `pool_padding` is a string, but not "SAME" or "VALID".
        ValueError: If `pool_padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `pool_padding` is a list or tuple, but the elements in the batch or channel dimensions are non-zero.
        ShapeError: If the input is not a 4-D or 5-D Tensor.
        ShapeError: If the dimension of input minus the size of `pool_stride` is not 2.
        ShapeError: If the size of `pool_size` and `pool_stride` is not equal.
        ShapeError: If the output's shape calculated is not greater than 0.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import paddle

          paddle.enable_static()

          data = fluid.data(name='data', shape=[None, 3, 32, 32, 32], dtype='float32')

          # max pool3d
          pool3d = fluid.layers.pool3d(
            input = data,
            pool_size = 2,
            pool_type = "max",
            pool_stride = 1,
            global_pooling=False)

          # average pool3d
          pool3d = fluid.layers.pool3d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=False)

          # global average pool3d
          pool3d = fluid.layers.pool3d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=True)

          # example 1:
          # Attr(pool_padding) is a list with 6 elements, Attr(data_format) is "NCDHW".
          out_1 = fluid.layers.pool3d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = [1, 2, 1, 0, 1, 2],
            global_pooling = False,
            data_format = "NCDHW")

          # example 2:
          # Attr(pool_padding) is a string, Attr(data_format) is "NCDHW".
          out_2 = fluid.layers.pool3d(
            input = data,
            pool_size = 3,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = "VALID",
            global_pooling = False,
            data_format = "NCDHW")

    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown Attr(pool_type): '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When Attr(global_pooling) is False, Attr(pool_size) must be passed "
            "and be a valid value. Received Attr(pool_size): %s." %
            str(pool_size))

    if not isinstance(use_cudnn, bool):
        raise TypeError("Attr(use_cudnn) should be True or False. Received "
                        "Attr(use_cudnn): %s. " % str(use_cudnn))

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s" % str(data_format))

    pool_size = utils.convert_to_list(pool_size, 3, 'pool_size')
    pool_stride = utils.convert_to_list(pool_stride, 3, 'pool_stride')

    def update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, (list, tuple)):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 5:
            if is_list_or_tuple(padding[0]) and (data_format == "NCDHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:5]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NDHWC"):
                if not (padding[0] == [0, 0] and padding[4] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:4]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 6, 'padding')
            if utils._is_symmetric_padding(padding, 3):
                padding = [padding[0], padding[2], padding[4]]

        elif is_list_or_tuple(padding) and len(padding) == 6:
            padding = utils.convert_to_list(padding, 6, 'padding')
            if utils._is_symmetric_padding(padding, 3):
                padding = [padding[0], padding[2], padding[4]]
        else:
            padding = utils.convert_to_list(padding, 3, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(pool_padding, str):
        pool_padding = pool_padding.upper()
        if pool_padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(pool_padding))
        if pool_padding == "VALID":
            padding_algorithm = "VALID"
            pool_padding = [0, 0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", ceil_mode must be False. "
                    "Received ceil_mode: True.")
        elif pool_padding == "SAME":
            padding_algorithm = "SAME"
            pool_padding = [0, 0, 0]

    pool_padding = update_padding(pool_padding, data_format)

    op_type = "pool3d"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": use_cudnn,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": data_format,
        })

    return pool_out


@deprecated(since="2.0.0")
@templatedoc(op_type="pool2d")
def adaptive_pool2d(input,
                    pool_size,
                    pool_type="max",
                    require_index=False,
                    name=None):
    r"""

    This operation calculates the output based on the input, pool_size,
    pool_type parameters. Input(X) and output(Out) are in NCHW format, where N is batch
    size, C is the number of channels, H is the height of the feature, and W is
    the width of the feature. Parameters(pool_size) should contain two elements which
    represent height and width, respectively. Also the H and W dimensions of output(Out)
    is same as Parameter(pool_size). The output tensor shape will be [N, C, pool_size[0], pool_size[1]]

    For average adaptive pool2d:

    ..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

    Args:
        input (Tensor): The input tensor of pooling operator, which is a 4-D tensor
                          with shape [N, C, H, W].  The format of input tensor is NCHW,
                          where N is batch size, C is the number of channels, H is the
                          height of the feature, and W is the width of the feature.
                          The data type is float32 or float64.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
        pool_type: ${pooling_type_comment}
        require_index (bool): If true, the index of max pooling point will be returned along
            with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of adaptive pooling result. The data type is same
                  as input tensor.

    Raises:
        ValueError: 'pool_type' is not 'max' nor 'avg'.
        ValueError: invalid setting 'require_index' true when 'pool_type' is 'avg'.
        ValueError: 'pool_size' should be a list or tuple with length as 2.

    Examples:
        .. code-block:: python

          # average adaptive pool2d
          # suppose input data in shape of [N, C, H, W], `pool_size` is [m, n],
          # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
          # of input data into m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(m):
          #         for j in range(n):
          #             hstart = floor(i * H / m)
          #             hend = ceil((i + 1) * H / m)
          #             wstart = floor(i * W / n)
          #             wend = ceil((i + 1) * W / n)
          #             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
          #
          import paddle
          paddle.enable_static()
          data = paddle.rand(shape=[1,3,32,32])
          pool_out = paddle.fluid.layers.adaptive_pool2d(
                            input=data,
                            pool_size=[3, 3],
                            pool_type='avg')

          # max adaptive pool2d
          # suppose input data in shape of [N, C, H, W], `pool_size` is [m, n],
          # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
          # of input data into m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(m):
          #         for j in range(n):
          #             hstart = floor(i * H / m)
          #             hend = ceil((i + 1) * H / m)
          #             wstart = floor(i * W / n)
          #             wend = ceil((i + 1) * W / n)
          #             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
          #
          import paddle
          data = paddle.rand(shape=[1,3,32,32])
          pool_out = paddle.fluid.layers.adaptive_pool2d(
                            input=data,
                            pool_size=[3, 3],
                            pool_type='max')
    """
    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'adaptive_pool2d')
    check_type(pool_type, 'pool_type', str, 'adaptive_pool2d')
    check_type(pool_size, 'pool_size', (int, list, tuple), 'adaptive_pool2d')
    check_type(require_index, 'require_index', bool, 'adaptive_pool2d')
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if pool_type == "avg" and require_index:
        raise ValueError(
            "invalid setting 'require_index' true when 'pool_type' is 'avg'.")

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')

    if pool_type == "max":
        l_type = 'max_pool2d_with_index'
    else:
        l_type = "pool2d"

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    if pool_type == "max":
        mask = helper.create_variable_for_type_inference(dtype)
        outputs["Mask"] = mask

    helper.append_op(
        type=l_type,
        inputs={"X": input},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (pool_out, mask) if require_index else pool_out


@deprecated(since="2.0.0")
@templatedoc(op_type="pool3d")
def adaptive_pool3d(input,
                    pool_size,
                    pool_type="max",
                    require_index=False,
                    name=None):
    r"""

    This operation calculates the output based on the input, pool_size,
    pool_type parameters. Input(X) and output(Out) are in NCDHW format, where N is batch
    size, C is the number of channels, D is the depth of the feature, H is the height of
    the feature, and W is the width of the feature. Parameters(pool_size) should contain
    three elements which represent height and width, respectively. Also the D, H and W
    dimensions of output(Out) is same as Parameter(pool_size). The output tensor shape
    will be [N, C, pool_size[0], pool_size[1], pool_size[2]]

    For average adaptive pool3d:

    ..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}

    Args:
        input (Tensor): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W]. The format of input tensor is NCDHW, where
                          N is batch size, C is the number of channels, D is the depth of the feature,
                          H is the height of the feature, and W is the width of the feature.
                          The data type is float32 or float64.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three integers, (Depth, Height, Width).
        pool_type: ${pooling_type_comment}
        require_index (bool): If true, the index of max pooling point will be returned along
            with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of adaptive pooling result. The data type is same as input tensor.

    Raises:
        ValueError: 'pool_type' is not 'max' nor 'avg'.
        ValueError: invalid setting 'require_index' true when 'pool_type' is 'avg'.
        ValueError: 'pool_size' should be a list or tuple with length as 2.

    Examples:
        .. code-block:: python

          # average adaptive pool3d
          # suppose input data in shape of [N, C, D, H, W], `pool_size` is [l, m, n],
          # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
          # of input data into l * m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(l):
          #         for j in range(m):
          #             for k in range(n):
          #                 dstart = floor(i * D / l)
          #                 dend = ceil((i + 1) * D / l)
          #                 hstart = floor(j * H / m)
          #                 hend = ceil((j + 1) * H / m)
          #                 wstart = floor(k * W / n)
          #                 wend = ceil((k + 1) * W / n)
          #                 output[:, :, i, j, k] =
          #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
          #

          import paddle
          paddle.enable_static()
          data = paddle.rand(shape=[1,3,32,32,32])
          pool_out = paddle.fluid.layers.adaptive_pool3d(
                            input=data,
                            pool_size=[3, 3, 3],
                            pool_type='avg')

          # max adaptive pool3d
          # suppose input data in shape of [N, C, D, H, W], `pool_size` is [l, m, n],
          # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
          # of input data into l * m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(l):
          #         for j in range(m):
          #             for k in range(n):
          #                 dstart = floor(i * D / l)
          #                 dend = ceil((i + 1) * D / l)
          #                 hstart = floor(j * H / m)
          #                 hend = ceil((j + 1) * H / m)
          #                 wstart = floor(k * W / n)
          #                 wend = ceil((k + 1) * W / n)
          #                 output[:, :, i, j, k] =
          #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
          #

          import paddle
          data = paddle.rand(shape=[1,3,32,32,32])
          pool_out = paddle.fluid.layers.adaptive_pool3d(
                            input=data,
                            pool_size=[3, 3, 3],
                            pool_type='max')
    """
    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'adaptive_pool3d')
    check_type(pool_type, 'pool_type', str, 'adaptive_pool3d')
    check_type(pool_size, 'pool_size', (int, list, tuple), 'adaptive_pool3d')
    check_type(require_index, 'require_index', bool, 'adaptive_pool3d')
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if pool_type == "avg" and require_index:
        raise ValueError(
            "invalid setting 'require_index' true when 'pool_type' is 'avg'.")

    pool_size = utils.convert_to_list(pool_size, 3, 'pool_size')

    if pool_type == "max":
        l_type = 'max_pool3d_with_index'
    else:
        l_type = "pool3d"

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    if pool_type == "max":
        mask = helper.create_variable_for_type_inference(dtype)
        outputs["Mask"] = mask

    helper.append_op(
        type=l_type,
        inputs={"X": input},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (pool_out, mask) if require_index else pool_out


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               in_place=False,
               name=None,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=True,
               use_global_stats=False):
    r"""
    :api_attr: Static Graph

    **Batch Normalization Layer**

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

        moving\_mean = moving\_mean * momentum + mini-batch\_mean * (1. - momentum) \\\\
        moving\_var = moving\_var * momentum + mini-batch\_var * (1. - momentum)


    moving_mean is global mean and moving_var is global variance.

    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global (or running) statistics. (It usually got from the
    pre-trained model.)
    The training and testing (or inference) have the same behavior:

    ..  math::

        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}}  \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta

    Note:
        if build_strategy.sync_batch_norm=True, the batch_norm in network will use
        sync_batch_norm automatically.
        `is_test = True` can only be used in test program and inference program, `is_test` CANNOT be set to True in train program, if you want to use global status from pre_train model in train program, please set `use_global_stats = True`.

    Args:
        input(Tensor): The rank of input Tensor can be 2, 3, 4, 5. The data type
            is float16 or float32 or float64.
        act(string, Default None): Activation type, linear|relu|prelu|...
        is_test (bool, Default False): A flag indicating whether it is in
            test phrase or not.
        momentum(float|Tensor, Default 0.9): The value used for the moving_mean and
            moving_var computation. This should be a float number or a Tensor with
            shape [1] and data type as float32. The updated formula is:
            :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)`
            :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)`
            Default is 0.9.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized
	     with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
	     If the Initializer of the bias_attr is not set, the bias is initialized zero.
	     Default: None.
        data_layout (str, optional): Specify the data format of the input, and the data format of the output
             will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
             The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
             `[batch_size, input_channels, input_height, input_width]`.
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(str|None): For detailed information, please refer to :ref:`api_guide_Name`.
            Usually name is no need to set and None by default.
        moving_mean_name(str, Default None): The name of moving_mean which store the global Mean. If it
            is set to None, batch_norm will save global mean with a random name, otherwise, batch_norm
            will save global mean with the string.
        moving_variance_name(str, Default None): The name of the moving_variance which store the global Variance.
            If it is set to None, batch_norm will save global variance with a random name, otherwise, batch_norm
            will save global variance with the string.
        do_model_average_for_mean_and_var(bool, Default True): Whether parameter mean and variance should do model
            average when model average is enabled.
        use_global_stats(bool, Default False): Whether to use global mean and
            variance. In inference or test mode, set use_global_stats to true
            or is_test to true, and the behavior is equivalent.
            In train mode, when setting use_global_stats True, the global mean
            and variance are also used during train period.
    Returns:
        A Tensor which is the result after applying batch normalization on the input,
        has same shape and data type with input.

    Examples:

        .. code-block:: python

            import paddle
            
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
            hidden1 = paddle.static.nn.fc(x=x, size=200)
            print(hidden1.shape)
            # [3, 200]
            hidden2 = paddle.static.nn.batch_norm(input=hidden1)
            print(hidden2.shape)
            # [3, 200]
    """
    assert bias_attr is not False, "bias_attr should not be False in batch_norm."
    helper = LayerHelper('batch_norm', **locals())

    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'batch_norm')
    dtype = helper.input_dtype()

    # use fp32 for bn parameter
    if dtype == core.VarDesc.VarType.FP16:
        dtype = core.VarDesc.VarType.FP32

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0))
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)

    mean = helper.create_parameter(
        attr=ParamAttr(
            name=moving_mean_name,
            initializer=Constant(0.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=Constant(1.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance_out share the same memory
    variance_out = variance
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    reserve_space = None
    if not is_test:
        reserve_space = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype(), stop_gradient=True)

    batch_norm_out = input if in_place else \
            helper.create_variable_for_type_inference(dtype)

    inputs = {
        "X": input,
        "Scale": scale,
        "Bias": bias,
        "Mean": mean,
        "Variance": variance
    }
    attrs = {
        "epsilon": epsilon,
        "is_test": is_test,
        "data_layout": data_layout,
        "use_mkldnn": False,
        "fuse_with_relu": False,
        "use_global_stats": use_global_stats
    }
    if isinstance(momentum, Variable):
        inputs['MomemtumTensor'] = momentum
    else:
        attrs['momentum'] = momentum

    outputs = {
        "Y": batch_norm_out,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
        "SavedMean": saved_mean,
        "SavedVariance": saved_variance
    }
    if reserve_space is not None:
        outputs["ReserveSpace"] = reserve_space

    helper.append_op(
        type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs)

    return helper.append_activation(batch_norm_out)


def inplace_abn(input,
                act=None,
                is_test=False,
                momentum=0.9,
                epsilon=1e-05,
                param_attr=None,
                bias_attr=None,
                data_layout='NCHW',
                name=None,
                moving_mean_name=None,
                moving_variance_name=None,
                do_model_average_for_mean_and_var=True,
                use_global_stats=False,
                act_alpha=1.0):
    r"""
    **In-place Activation Batch Normalization Layer**

    This layer calculates batch normalization and activation with in-place memory.
    For batch normalization calculations, see `fluid.layers.batch_norm`.
    For in-place activation batch normalization, see `In-Place Activated BatchNorm for
    Memory-Optimized Training of DNNs <https://arxiv.org/abs/1712.02616>`_

    `inplace_abn` only support activation type as `None`, `identity`, `leaky_relu`,
    `elu` currently.
    `inplace_abn` only support data type as `float32`, `float64` currently.

    Note:
        if build_strategy.sync_batch_norm=True, the batch_norm in network will use
        sync_batch_norm automatically.
        `is_test = True` can only be used in test program and inference program, `is_test` CANNOT be set to True in train program, if you want to use global status from pre_train model in train program, please set `use_global_stats = True`.

    Args:
        input(Variable): The rank of input variable can be 2, 3, 4, 5. The data type
            is float16 or float32 or float64.
        act(string, Default None): Activation type, linear|relu|prelu|...
        is_test (bool, Default False): A flag indicating whether it is in
            test phrase or not.
        momentum(float|Variable, Default 0.9): The value used for the moving_mean and
            moving_var computation. This should be a float number or a Variable with
            shape [1] and data type as float32. The updated formula is:
            :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)`
            :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)`
            Default is 0.9.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of inplace_abn. If it is set to None or one attribute of ParamAttr, inplace_abn
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized
	     with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of inplace_abn.
             If it is set to None or one attribute of ParamAttr, inplace_abn
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
	     If the Initializer of the bias_attr is not set, the bias is initialized zero.
	     Default: None.
        data_layout (str, optional): Specify the data format of the input, and the data format of the output
             will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
             The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
             `[batch_size, input_channels, input_height, input_width]`.
        name(str|None): For detailed information, please refer to :ref:`api_guide_Name`.
            Usually name is no need to set and None by default.
        moving_mean_name(str, Default None): The name of moving_mean which store the global Mean. If it
            is set to None, inplace_abn will save global mean with a random name, otherwise, inplace_abn
            will save global mean with the string.
        moving_variance_name(str, Default None): The name of the moving_variance which store the global Variance.
            If it is set to None, inplace_abn, will save global variance with a random name, otherwise, inplace_abn
            will save global variance with the string.
        do_model_average_for_mean_and_var(bool, Default True): Whether parameter mean and variance should do model
            average when model average is enabled.
        use_global_stats(bool, Default False): Whether to use global mean and
            variance. In inference or test mode, set use_global_stats to true
            or is_test to true, and the behavior is equivalent.
            In train mode, when setting use_global_stats True, the global mean
            and variance are also used during train period.
        act_alpha(float, Default 1.0): when activation is in ['elu', 'identity', 'leaky_relu'],
            inplace activative batch normalization will be used, and alpha parameter for activation
            can be given by this parameter.
    Returns:
        A Variable holding Tensor which is the result after applying batch normalization and activation on the input,
        has same shape and data type with input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
            hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
            hidden2 = fluid.layers.inplace_abn(input=hidden1)
            hidden3 = fluid.layers.inplace_abn(input=hidden2, act='leaky_relu', act_alpha=0.2)

    """
    assert act in [None, 'identity', 'leaky_relu', 'elu'], \
        "inplace_abn only support act as None, 'identity', " \
        "'leaky_relu', 'elu' currently"
    assert bias_attr is not False, "bias_attr should not be False in inplace_abn."
    helper = LayerHelper('inplace_abn', **locals())

    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'inplace_abn')
    dtype = helper.input_dtype()

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0))
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)

    mean = helper.create_parameter(
        attr=ParamAttr(
            name=moving_mean_name,
            initializer=Constant(0.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=Constant(1.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    reserve_space = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    batch_norm_out = input

    inputs = {
        "X": input,
        "Scale": scale,
        "Bias": bias,
        "Mean": mean,
        "Variance": variance
    }
    attrs = {
        "epsilon": epsilon,
        "is_test": is_test,
        "data_layout": data_layout,
        "use_mkldnn": False,
        "fuse_with_relu": False,
        "use_global_stats": use_global_stats,
        "activation": act,
        "alpha": act_alpha,
    }
    if isinstance(momentum, Variable):
        inputs['MomemtumTensor'] = momentum
    else:
        attrs['momentum'] = momentum
    outputs = {
        "Y": batch_norm_out,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
        "SavedMean": saved_mean,
        "SavedVariance": saved_variance
    }
    if reserve_space is not None:
        outputs["ReserveSpace"] = reserve_space

    helper.append_op(
        type="inplace_abn", inputs=inputs, outputs=outputs, attrs=attrs)

    return batch_norm_out


def instance_norm(input,
                  epsilon=1e-05,
                  param_attr=None,
                  bias_attr=None,
                  name=None):
    r"""
    :api_attr: Static Graph

    **Instance Normalization Layer**

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW} x_i \\qquad &//\\
        \\ mean\ of\ one\  feature\ map\ in\ mini-batch \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ variance\ of\ one\ feature\ map\ in\ mini-batch \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Note:
        `H` means height of feature map, `W` means width of feature map.

    Args:
        input(Tensor): The rank of input tensor can be 2, 3, 4, 5.
            The data type is float32 or float64.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None|bool, optional): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized
	     with Xavier. If the param_attr is set to False, instance_norm will not create param_attr.
             Default: None.
        bias_attr(ParamAttr|None|bool, optional): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
	     If the Initializer of the bias_attr is not set, the bias is initialized zero.
             If the bias_attr is set to False, instance_norm will not create bias_attr.
	     Default: None.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        A Tensor which is the result after applying instance normalization on the input,
        has same shape and data type with input.

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
            hidden1 = paddle.static.nn.fc(x, size=200)
            hidden2 = paddle.static.nn.instance_norm(hidden1)
    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'instance_norm')
    if param_attr is False:
        assert bias_attr is False, "param_attr and bias_attr must be set to Fasle at the same time in instance_norm"

    helper = LayerHelper('instance_norm', **locals())
    dtype = helper.input_dtype()

    # use fp32 for in parameter
    if dtype == core.VarDesc.VarType.FP16:
        dtype = core.VarDesc.VarType.FP32

    input_shape = input.shape
    if len(input.shape) < 2 or len(input.shape) > 5:
        raise ValueError(
            'expected 2D or 3D or 4D or 5D input (got {}D input, input shape is: {})'.
            format(len(input.shape), input_shape))
    channel_num = input_shape[1]

    param_shape = [channel_num]

    if param_attr != False and bias_attr != False:
        # create parameter
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0))
        bias = helper.create_parameter(
            attr=helper.bias_attr,
            shape=param_shape,
            dtype=dtype,
            is_bias=True,
            default_initializer=Constant(0.0))

    # create output
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)

    instance_norm_out = helper.create_variable_for_type_inference(dtype)

    inputs = {"X": input}
    if param_attr != False and bias_attr != False:
        inputs["Scale"] = scale
        inputs["Bias"] = bias

    helper.append_op(
        type="instance_norm",
        inputs=inputs,
        outputs={
            "Y": instance_norm_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        },
        attrs={"epsilon": epsilon, })

    return instance_norm_out


@static_only
def data_norm(input,
              act=None,
              epsilon=1e-05,
              param_attr=None,
              data_layout='NCHW',
              in_place=False,
              name=None,
              moving_mean_name=None,
              moving_variance_name=None,
              do_model_average_for_mean_and_var=True,
              slot_dim=-1,
              sync_stats=False,
              summary_decay_rate=0.9999999,
              enable_scale_and_shift=False):
    r"""
    :api_attr: Static Graph

    **Data Normalization Layer**

    This op can be used as a normalizer function for conv2d and fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Args:
        input(Tensor): The input Tensor.
        act(string, Default None): Activation type, linear|relu|prelu|...
        epsilon(float, Default 1e-05):
        param_attr(ParamAttr): The parameter attribute for Parameter `scale`.
        data_layout (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.
        moving_mean_name(string, Default None): The name of moving_mean which store the global Mean.
        moving_variance_name(string, Default None): The name of the moving_variance which store the global Variance.
        do_model_average_for_mean_and_var(bool, Default True): Whether parameter mean and variance
            should do model average when model average is enabled.
        slot_dim(int): The embedding dimension of one slot. Slot is a set of one specific feature. In pslib mode, we
            distinguish feature ids by slot and pull their embeddings from parameter server (pslib). The first
            place of the embedding is the historical show number (occurence time of this feature id with a label 0).
            If the input of this op is concated by slot-wise embeddings, and the show number is zero when this slot
            is new or empty, the normalization result may be impractical. To avoid this, we add slot_dim to locate
            the show number and judge if the show number is zero. If so, we choose to skip normalization on this
            embedding.
        sync_stats(bool, Default False): When running with multiple GPU cards, using allreduce to sync the
            summary messages.
        summary_decay_rate(float, Default 0.9999999): The decay rate when updating summary.
        enable_scale_and_shift(bool, Default False): do scale&shift after normalization.

    Returns:
        Tensor: A tensor which is the result after applying data normalization on the input.

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.randn(shape=[32,100])
            hidden2 = paddle.static.nn.data_norm(input=x)
    """
    helper = LayerHelper('data_norm', **locals())
    dtype = helper.input_dtype()

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    batch_size_default = 1e4
    batch_sum_default = 0.0
    batch_square_sum_default = 1e4
    scale_w_default = 1.0
    bias_default = 0.0

    if param_attr and isinstance(param_attr, dict):
        batch_size_default = param_attr.get("batch_size", 1e4)
        batch_sum_default = param_attr.get("batch_sum", 0.0)
        batch_square_sum_default = param_attr.get("batch_square", 1e4)
    if enable_scale_and_shift:
        scale_w_default = param_attr.get("scale_w", 1.0)
        bias_default = param_attr.get("bias", 0.0)

    # create scale and shift(bias) when enable_scale_and_shift is True
    if name == None:
        name = "dn"
    if enable_scale_and_shift:
        scale_w = helper.create_parameter(
            attr=ParamAttr(
                name=name + '.scale_w',
                initializer=Constant(value=float(scale_w_default)),
                trainable=True),
            shape=param_shape,
            dtype=input.dtype)
        bias = helper.create_parameter(
            attr=ParamAttr(
                name=name + '.bias',
                initializer=Constant(value=float(bias_default)),
                trainable=True),
            shape=param_shape,
            dtype=input.dtype)
    # create parameter
    batch_size = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_size',
            initializer=Constant(value=float(batch_size_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    batch_sum = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_sum',
            initializer=Constant(value=float(batch_sum_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    batch_square_sum = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_square_sum',
            initializer=Constant(value=float(batch_square_sum_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    means = helper.create_variable(dtype=dtype, stop_gradient=True)
    scales = helper.create_variable(dtype=dtype, stop_gradient=True)

    data_norm_out = input if in_place else helper.create_variable(dtype=dtype)

    inputs = {
        "X": input,
        "BatchSize": batch_size,
        "BatchSum": batch_sum,
        "BatchSquareSum": batch_square_sum
    }
    attrs = {
        "epsilon": epsilon,
        "data_layout": data_layout,
        "sync_stats": sync_stats,
        "summary_decay_rate": summary_decay_rate,
    }
    if slot_dim > 0:
        attrs["slot_dim"] = slot_dim
    if enable_scale_and_shift:
        attrs["enable_scale_and_shift"] = enable_scale_and_shift
    if enable_scale_and_shift:
        inputs["scale_w"] = scale_w
        inputs["bias"] = bias
    helper.append_op(
        type="data_norm",
        inputs=inputs,
        outputs={
            "Y": data_norm_out,
            "Means": means,
            "Scales": scales,
            "BatchSize": batch_size,
            "BatchSum": batch_sum,
            "BatchSquareSum": batch_square_sum
        },
        attrs=attrs)

    return helper.append_activation(data_norm_out)


@templatedoc()
def layer_norm(input,
               scale=True,
               shift=True,
               begin_norm_axis=1,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               act=None,
               name=None):
    r"""
    :api_attr: Static Graph

    **Layer Normalization Layer**

    The API implements the function of the Layer Normalization Layer and can be applied to mini-batch input data.
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

    The formula is as follows:

    ..  math::

        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} x_i

        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}{(x_i - \\mu)^2} + \\epsilon}

        y & = f(\\frac{g}{\\sigma}(x - \\mu) + b)

    - :math:`x`: the vector representation of the summed inputs to the neurons in that layer.
    - :math:`H`: the number of hidden units in a layers
    - :math:`\\epsilon`: the small value added to the variance to prevent division by zero.
    - :math:`g`: the trainable scale parameter.
    - :math:`b`: the trainable bias parameter.

    Args:
        input(Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64.
        scale(bool, optional): Whether to learn the adaptive gain :math:`g` after
            normalization. Default: True.
        shift(bool, optional): Whether to learn the adaptive bias :math:`b` after
            normalization. Default: True.
        begin_norm_axis(int, optional): The normalization will be performed along
            dimensions from :attr:`begin_norm_axis` to :attr:`rank(input)`.
            Default: 1.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            gain :math:`g`. If :attr:`scale` is False, :attr:`param_attr` is
            omitted. If :attr:`scale` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the learnable
            bias :math:`b`. If :attr:`shift` is False, :attr:`bias_attr` is
            omitted. If :attr:`shift` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        act(str, optional): Activation to be applied to the output of layer normalization.
                  Default: None.
        name(str): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: ``Tensor``  indicating the normalized result, the data type is the same as  ``input`` , and the return dimension is the same as  ``input`` .

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[8, 32, 32], dtype='float32')
            output = paddle.static.nn.layer_norm(input=x, begin_norm_axis=1)
            print(output.shape)  # [8, 32, 32]
    """
    assert in_dygraph_mode(
    ) is not True, "please use LayerNorm instead of layer_norm in dygraph mode!"
    helper = LayerHelper('layer_norm', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'layer_norm')
    dtype = helper.input_dtype()

    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    param_shape = [reduce(lambda x, y: x * y, input_shape[begin_norm_axis:])]
    if scale:
        assert param_attr is not False, "param_attr should not be False when using scale."
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0))
        inputs['Scale'] = scale
    else:
        if param_attr:
            warnings.warn("param_attr is only available with scale is True.")
    if shift:
        assert bias_attr is not False, "bias_attr should not be False when using shift."
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)
        inputs['Bias'] = bias
    else:
        if bias_attr:
            warnings.warn("bias_attr is only available with shift is True.")

    # create output
    mean_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    variance_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    layer_norm_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="layer_norm",
        inputs=inputs,
        outputs={
            "Y": layer_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={"epsilon": epsilon,
               "begin_norm_axis": begin_norm_axis})

    return helper.append_activation(layer_norm_out)


@templatedoc()
def group_norm(input,
               groups,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               act=None,
               data_layout='NCHW',
               name=None):
    """
    :api_attr: Static Graph

    **Group Normalization Layer**

    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

    Parameters:
        input(Tensor): Tensor with dimension greater than 1, the data type is float32 or float64.
        groups(int): The number of groups that divided from channels, the data type
            is int32.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero, the data type is float32. Default: 1e-05.
        param_attr(ParamAttr|bool, optional): ParamAttr object that specifies weight parameter
            attribute. If a bool type, only False is supported, which means there is no weight parameter.
            Default: None, the default weight parameter attribute is used. For more information, please
            refer to :ref:`api_guide_ParamAttr` .
        bias_attr(ParamAttr|bool, optional): ParamAttr object that specifies bias parameter
            attribute. If a bool type, only False is supported, which means there is no bias parameter.
            Default: None, the default bias parameter attribute is used. For more information, please
            refer to :ref:`api_guide_ParamAttr` .
        act(str, optional): Activation to be applied to the output of group normalization.
        data_layout(str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, *]`.
        name (str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A Tensor has same data type and data format with `input`.

    Examples:
       .. code-block:: python

            import paddle
            paddle.enable_static()
            
            data = paddle.static.data(name='data', shape=[2, 8, 32, 32], dtype='float32')
            x = paddle.static.nn.group_norm(input=data, groups=4)
            print(x.shape) # [2, 8, 32, 32]
    """
    helper = LayerHelper('group_norm', **locals())
    dtype = helper.input_dtype()
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'group_norm')
    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    if len(input_shape) < 2:
        raise ValueError(
            f"The dimensions of Op(fluid.layers.group_norm)'s input should be more than 1. But received {len(input_shape)}"
        )
    if data_layout != 'NCHW' and data_layout != 'NHWC':
        raise ValueError(
            "Param(data_layout) of Op(fluid.layers.group_norm) got wrong value: received "
            + data_layout + " but only NCHW or NHWC supported.")
    channel_num = input_shape[1] if data_layout == 'NCHW' else input_shape[-1]
    param_shape = [channel_num]
    if param_attr:
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0))
        inputs['Scale'] = scale
    if bias_attr:
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)
        inputs['Bias'] = bias

    # create output
    mean_out = helper.create_variable(dtype=dtype, stop_gradient=True)
    variance_out = helper.create_variable(dtype=dtype, stop_gradient=True)
    group_norm_out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="group_norm",
        inputs=inputs,
        outputs={
            "Y": group_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={
            "epsilon": epsilon,
            "groups": groups,
            "data_layout": data_layout
        })

    return helper.append_activation(group_norm_out)


@templatedoc()
def spectral_norm(weight, dim=0, power_iters=1, eps=1e-12, name=None):
    r"""
    :api_attr: Static Graph

    **Spectral Normalization Layer**

    This operation calculates the spectral normalization value of weight parameters of
    fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
    Parameters. Output tensor will be in same shape with input tensor.
    Calculations are showed as follows.

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`power_iters` should be a positive integer, do following
    calculations with U and V for :attr:`power_iters` rounds. Calculations
    as follows:

    .. math::

        \mathbf{v} := \\frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \\frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \\frac{\mathbf{W}}{\sigma(\mathbf{W})}


    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Args:
        weight(Tensor): ${weight_comment}
        dim(int): ${dim_comment}
        power_iters(int): ${power_iters_comment}
        eps(float): ${eps_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: A tensor of weight parameters after spectral normalization.
                  The data type and shape is same as input tensor.

    Examples:
       .. code-block:: python

            import paddle

            paddle.enable_static()
            weight = paddle.static.data(name='weight', shape=[2, 8, 32, 32], dtype='float32')
            x = paddle.static.nn.spectral_norm(weight=weight, dim=1, power_iters=2)
            print(x.shape) # [2, 8, 32, 32]
    """
    helper = LayerHelper('spectral_norm', **locals())
    check_variable_and_dtype(weight, 'weight', ['float32', 'float64'],
                             'spectral_norm')
    check_type(dim, 'dim', int, 'spectral_norm')
    check_type(power_iters, 'power_iters', int, 'spectral_norm')
    check_type(eps, 'eps', float, 'spectral_norm')
    dtype = weight.dtype

    # create intput and parameters
    inputs = {'Weight': weight}
    input_shape = weight.shape
    assert weight.numel() > 0, "Any dimension of input cannot be equal to 0."
    assert dim < len(input_shape), ("The input `dim` should be less than the "
                                    "rank of `weight`, but received dim="
                                    "{}".format(dim))
    h = input_shape[dim]
    w = np.prod(input_shape) // h

    u = helper.create_parameter(
        attr=ParamAttr(),
        shape=[h],
        dtype=dtype,
        default_initializer=Normal(0., 1.))
    u.stop_gradient = True
    inputs['U'] = u
    v = helper.create_parameter(
        attr=ParamAttr(),
        shape=[w],
        dtype=dtype,
        default_initializer=Normal(0., 1.))
    inputs['V'] = v
    v.stop_gradient = True

    # create output
    out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="spectral_norm",
        inputs=inputs,
        outputs={"Out": out, },
        attrs={
            "dim": dim,
            "power_iters": power_iters,
            "eps": eps,
        })

    return out


def conv2d_transpose(input,
                     num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=None,
                     param_attr=None,
                     bias_attr=None,
                     use_cudnn=True,
                     act=None,
                     name=None,
                     data_format='NCHW'):
    r"""
    :api_attr: Static Graph

    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCHW or NHWC format. Where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Parameters(dilations, strides, paddings) are two elements. These two elements
    represent height and width, respectively. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a 4-D Tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a 4-D Tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, a 4-D Tensor with data format 'NCHW' or 'NHWC', the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - pad_height_top - pad_height_bottom + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - pad_width_left - pad_width_right + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] ]

    Note:
          The conv2d_transpose can be seen as the backward of the conv2d. For conv2d,
          when stride > 1, conv2d maps multiple input shape to the same output shape,
          so for conv2d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, W_{out} = W^\prime_{out}`;
          else, the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}`
          and :math:`H^\prime_{out} + strides[0]`, and the :math:`W_{out}` of the output size must
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[1]`,
          conv2d_transpose can compute the kernel size automatically.

    Args:
        input(Tensor): 4-D Tensor with [N, C, H, W] or [N, H, W, C] format,
                         its data type is float32 or float64.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple, optional): The output image size. If output size is a
            tuple, it must contain two integers, (image_height, image_width). None if use
            filter_size, padding, and stride to calculate output_size.
            If output_size and filter_size are specified at the same time, They
            should follow the formula above. Default: None. output_size and filter_size
            should not be None at the same time.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_height, filter_size_width).
            Otherwise, filter_size_height = filter_size_width = filter_size. None if
            use output size to calculate filter_size. Default: None. filter_size and
            output_size should not be None at the same time.
        stride(int|tuple, optional): The stride size. It means the stride in transposed convolution.
            If stride is a tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding(str|int|list|tuple, optional): The padding size. It means the number of zero-paddings 
            on both sides for each dimension. If `padding` is a string, either 'VALID' or 
            'SAME' which is the padding algorithm. If `padding` is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or 
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCHW"`, `padding` can be in the form 
            `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `padding` can be in the form 
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation(int|tuple, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a tuple, it must contain two integers, (dilation_height, dilation_width).
            Otherwise, dilation_height = dilation_width = dilation. Default: dilation = 1.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_height, filter_size_width).
            Otherwise, filter_size_height = filter_size_width = filter_size. None if
            use output size to calculate filter_size. Default: None.
        groups(int, optional): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        A Tensor representing the conv2d_transpose, whose
        data type is the same with input and shape is (num_batches, channels, out_h,
        out_w) or (num_batches, out_h, out_w, channels). If act is None, the tensor 
        storing the transposed convolution result, and if act is not None, the
        tensor storing transposed convolution and non-linearity activation
        result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0
            or the element corresponding to the input's channel is not 0.
        ValueError: If `output_size` and filter_size are None at the same time.
        ShapeError: If the input is not 4-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels.
        ShapeError: If the size of `output_size` is not equal to that of `stride`.

    Examples:
       .. code-block:: python

          import paddle
          paddle.enable_static()

          data = paddle.static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          conv2d_transpose = paddle.static.nn.conv2d_transpose(input=data, num_filters=2, filter_size=3)
          print(conv2d_transpose.shape) # [-1, 2, 34, 34]
    """
    assert param_attr is not False, "param_attr should not be False in conv2d_transpose."
    if len(input.shape) != 4:
        raise ValueError("Input size should be 4, "
                         "but received {}".format(len(input.shape)))

    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of Op(fluid.layers.conv2d_transpose) got wrong value: received "
            + data_format + " but only NCHW or NHWC supported.")

    input_channel = input.shape[1] if data_format == 'NCHW' else input.shape[-1]
    op_type = 'conv2d_transpose'
    if (input_channel == groups and num_filters == input_channel and
            not use_cudnn):
        op_type = 'depthwise_conv2d_transpose'

    helper = LayerHelper(op_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv2d_transpose must be Variable")

    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')
        else:
            padding = utils.convert_to_list(padding, 2, 'padding')
            padding = [padding[0], padding[0], padding[1], padding[1]]
        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'." %
                str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0, 0]

    padding = _update_padding(padding, data_format)

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        h_in = input.shape[2] if data_format == 'NCHW' else input.shape[1]
        w_in = input.shape[3] if data_format == 'NCHW' else input.shape[2]

        filter_size_h = (output_size[0] - (h_in - 1) * stride[0] + padding[0] +
                         padding[1] - 1) // dilation[0] + 1
        filter_size_w = (output_size[1] - (w_in - 1) * stride[1] + padding[2] +
                         padding[3] - 1) // dilation[1] + 1
        filter_size = [filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 2,
                                            'conv2d_transpose.filter_size')

    if len(padding) == 4 and utils._is_symmetric_padding(padding, 2):
        padding = [padding[0], padding[2]]

    if output_size is None:
        output_size = []
    elif isinstance(output_size, (list, tuple, int)):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        raise ValueError("output_size should be int, list[int] or tuple[int]")

    if groups is None:
        groups = 1
    elif groups <= 0:
        raise ValueError("the groups of input must be greater than 0, "
                         "but received the groups of input is {}".format(
                             groups))

    filter_shape = [input_channel, num_filters // groups] + filter_size

    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format
        })

    if data_format == 'NCHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=3, dim_end=4)
    out = helper.append_activation(pre_act)
    return out


def conv3d_transpose(input,
                     num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=None,
                     param_attr=None,
                     bias_attr=None,
                     use_cudnn=True,
                     act=None,
                     name=None,
                     data_format='NCDHW'):
    r"""
    :api_attr: Static Graph

    The convolution3D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCDHW or NDHWC format. Where N is batch size, C is the number of channels,
    D is the depth of the feature, H is the height of the feature, and W
    is the width of the feature. Parameters(dilations, strides, paddings) are
    two elements. These two elements represent height and width, respectively.
    The details of convolution transpose layer, please refer to the following
    explanation and references `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \ast X + b)

    In the above equation:

    * :math:`X`: Input value, a Tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a Tensor with MCDHW format.
    * :math:`\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, D_f, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

           D^\prime_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\\\
           H^\prime_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1 \\\\
           D_{out} &\in [ D^\prime_{out}, D^\prime_{out} + strides[0] ] \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[1] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[2] ]

    Note:
          The conv3d_transpose can be seen as the backward of the conv3d. For conv3d,
          when stride > 1, conv3d maps multiple input shape to the same output shape,
          so for conv3d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, :math:`H_{out} = \
          H^\prime_{out}, W_{out} = W^\prime_{out}`; else, the :math:`D_{out}` of the output
          size must between :math:`D^\prime_{out}` and :math:`D^\prime_{out} + strides[0]`,
          the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}`
          and :math:`H^\prime_{out} + strides[1]`, and the :math:`W_{out}` of the output size must
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[2]`,
          conv3d_transpose can compute the kernel size automatically.

    Args:
        input(Tensor): The input is 5-D Tensor with shape [N, C, D, H, W] or [N, D, H, W, C], the data type
            of input is float32 or float64.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple, optional): The output image size. If output size is a
            tuple, it must contain three integers, (image_depth, image_height, image_width). This
            parameter only works when filter_size is None. If output_size and filter_size are
            specified at the same time, They should follow the formula above. Default: None.
            Output_size and filter_size should not be None at the same time.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_depth, filter_size_height,
            filter_size_width). Otherwise, filter_size_depth = filter_size_height = \
            filter_size_width = filter_size. None if use output size to
            calculate filter_size. Default: None. filter_size and output_size should not be
            None at the same time.
        padding(int|list|str|tuple, optional): The padding size. The padding argument effectively
            adds `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a string,
            either 'VALID' or 'SAME' supported, which is the padding algorithm. If `padding`
            is a tuple or list, it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `'NCDHW'`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `'NDHWC'`, `padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        stride(int|tuple, optional): The stride size. It means the stride in transposed convolution.
            If stride is a tuple, it must contain three integers, (stride_depth, stride_height,
            stride_width). Otherwise, stride_depth = stride_height = stride_width = stride.
            Default: stride = 1.
        dilation(int|tuple, optional): The dilation size. It means the spacing between the kernel points.
            If dilation is a tuple, it must contain three integers, (dilation_depth, dilation_height,
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups(int, optional): The groups number of the Conv3d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups=1
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d_transpose. If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv3d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        A Variable holding Tensor representing the conv3d_transpose, whose data
        type is the same with input and shape is (num_batches, channels, out_d, out_h,
        out_w) or (num_batches, out_d, out_h, out_w, channels). If act is None, the tensor
        variable storing the transposed convolution result, and if act is not None, the tensor
        variable storing transposed convolution and non-linearity activation result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0
            or the element corresponding to the input's channel is not 0.
        ValueError: If `output_size` and filter_size are None at the same time.
        ShapeError: If the input is not 5-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels.
        ShapeError: If the size of `output_size` is not equal to that of `stride`.

    Examples:
       .. code-block:: python

          import paddle
          import numpy as np
	    
          paddle.enable_static()
          data = paddle.static.data(name='data', shape=[None, 3, 12, 32, 32], dtype='float32')
          param_attr = paddle.framework.ParamAttr(name='conv3d.weight', initializer=paddle.nn.initializer.XavierNormal(), learning_rate=0.001)
          res = paddle.static.nn.conv3d_transpose(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)
          place = paddle.CPUPlace()
          exe = paddle.static.Executor(place)
          exe.run(paddle.static.default_startup_program())
          x = np.random.rand(1, 3, 12, 32, 32).astype("float32")
          output = exe.run(feed={"data": x}, fetch_list=[res])
          print(output)
    """
    assert param_attr is not False, "param_attr should not be False in conv3d_transpose."
    if data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Param(data_format) of Op(fluid.layers.conv3d_transpose) got wrong value: received "
            + data_format + " but only NCDHW or NDHWC supported.")

    l_type = "conv3d_transpose"
    helper = LayerHelper(l_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv3d_transpose must be Variable")
    if len(input.shape) != 5:
        raise ValueError(
            "Input should be 5D tensor, but received input with the shape of {}".
            format(input.shape))
    input_channel = input.shape[1] if data_format == 'NCDHW' else input.shape[
        -1]

    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 5:
            if is_list_or_tuple(padding[0]) and (data_format == "NCDHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:5]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NDHWC"):
                if not (padding[0] == [0, 0] and padding[4] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:4]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 6, 'padding')

        elif is_list_or_tuple(padding) and len(padding) == 6:
            padding = utils.convert_to_list(padding, 6, 'padding')

        else:
            padding = utils.convert_to_list(padding, 3, 'padding')
            padding = [
                padding[0], padding[0], padding[1], padding[1], padding[2],
                padding[2]
            ]
        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'." %
                str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0, 0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0, 0, 0, 0]

    padding = _update_padding(padding, data_format)

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size, output_size]

        d_in = input.shape[2] if data_format == 'NCDHW' else input.shape[1]
        h_in = input.shape[3] if data_format == 'NCDHW' else input.shape[2]
        w_in = input.shape[4] if data_format == 'NCDHW' else input.shape[3]

        filter_size_d = (output_size[0] - (d_in - 1) * stride[0] + padding[0] +
                         padding[1] - 1) // dilation[0] + 1
        filter_size_h = (output_size[1] - (h_in - 1) * stride[1] + padding[2] +
                         padding[3] - 1) // dilation[1] + 1
        filter_size_w = (output_size[2] - (w_in - 1) * stride[2] + padding[4] +
                         padding[5] - 1) // dilation[2] + 1
        filter_size = [filter_size_d, filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 3,
                                            'conv3d_transpose.filter_size')

    if len(padding) == 6 and utils._is_symmetric_padding(padding, 3):
        padding = [padding[0], padding[2], padding[4]]

    if output_size is None:
        output_size = []
    elif isinstance(output_size, (list, tuple, int)):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        raise ValueError("output_size should be int, list[int] or tuple[int]")

    groups = 1 if groups is None else groups
    if groups <= 0:
        raise ValueError(
            "the groups of conv3d_transpose should be greater than 0. Received groups: {}".
            format(groups))
    if num_filters % groups != 0:
        raise ValueError("Attr(num_filters) must be divisible by groups,"
                         "Received: Attr(num_filters) is {}, the groups is {}".
                         format(num_filters, groups))

    filter_shape = [input_channel, num_filters // groups] + filter_size
    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    if data_format == 'NCDHW':
        data_format = 'NCHW'
    if data_format == 'NDHWC':
        data_format = 'NHWC'

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=l_type,
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format
        })

    if data_format == 'NCHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=4, dim_end=5)
    out = helper.append_activation(pre_act)
    return out


def reduce_sum(input, dim=None, keep_dim=False, name=None):
    """

    Computes the sum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of summation operation on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Raises:
        TypeError, if out data type is different with the input data type.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_sum(x)  # [3.5]
            fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
            fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_sum(y, dim=[1, 2]) # [10, 26]
            fluid.layers.reduce_sum(y, dim=[0, 1]) # [16, 20]

    """
    if dim is not None and not isinstance(dim, list):
        dim = [dim]

    if in_dygraph_mode():
        reduce_all = True if dim == None or dim == [] or len(dim) == len(
            input.shape) else False
        dim = dim if dim != None and dim != [] else [0]
        return _C_ops.reduce_sum(input, 'dim', dim, 'keep_dim', keep_dim,
                                 'reduce_all', reduce_all)
    attrs = {
        'dim': dim if dim != None and dim != [] else [0],
        'keep_dim': keep_dim,
        'reduce_all': True
        if dim == None or dim == [] or len(dim) == len(input.shape) else False
    }
    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'reduce_sum')
    helper = LayerHelper('reduce_sum', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs=attrs)
    return out


@deprecated(since="2.0.0", update_to="paddle.mean")
def reduce_mean(input, dim=None, keep_dim=False, name=None):
    """
    Computes the mean of the input tensor's elements along the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimension along which the mean is computed. If
            `None`, compute the mean over all elements of :attr:`input`
            and return a variable with a single element, otherwise it
            must be in the range :math:`[-rank(input), rank(input))`. If
            :math:`dim[i] < 0`, the dimension to reduce is
            :math:`rank(input) + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of average on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Raises:
        TypeError, if out data type is different with the input data type.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_mean(x)  # [0.4375]
            fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
            fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
            fluid.layers.reduce_mean(x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_mean(y, dim=[1, 2]) # [2.5, 6.5]
            fluid.layers.reduce_mean(y, dim=[0, 1]) # [4.0, 5.0]
    """

    return paddle.mean(x=input, axis=dim, keepdim=keep_dim, name=name)


def reduce_max(input, dim=None, keep_dim=False, name=None):
    """

    Computes the maximum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimension along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of maximum on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_max(x)  # [0.9]
            fluid.layers.reduce_max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
            fluid.layers.reduce_max(x, dim=-1)  # [0.9, 0.7]
            fluid.layers.reduce_max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_max(y, dim=[1, 2]) # [4.0, 8.0]
            fluid.layers.reduce_max(y, dim=[0, 1]) # [7.0, 8.0]
    """
    helper = LayerHelper('reduce_max', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_max',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None and dim != [] else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None or dim == [] or
            len(dim) == len(input.shape) else False
        })
    return out


def reduce_min(input, dim=None, keep_dim=False, name=None):
    """

    Computes the minimum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, result of minimum on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_min(x)  # [0.1]
            fluid.layers.reduce_min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
            fluid.layers.reduce_min(x, dim=-1)  # [0.2, 0.1]
            fluid.layers.reduce_min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_min(y, dim=[1, 2]) # [1.0, 5.0]
            fluid.layers.reduce_min(y, dim=[0, 1]) # [1.0, 2.0]
    """
    helper = LayerHelper('reduce_min', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_min',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None and dim != [] else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None or dim == [] or
            len(dim) == len(input.shape) else False
        })
    return out


def reduce_prod(input, dim=None, keep_dim=False, name=None):
    """

    Computes the product of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (int|list|tuple, optional): The dimensions along which the product is performed. If
            :attr:`None`, multiply all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, result of product on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_prod(x)  # [0.0002268]
            fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
            fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
            fluid.layers.reduce_prod(x, dim=1,
                                     keep_dim=True)  # [[0.027], [0.0084]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_prod(y, dim=[1, 2]) # [24.0, 1680.0]
            fluid.layers.reduce_prod(y, dim=[0, 1]) # [105.0, 384.0]
    """
    helper = LayerHelper('reduce_prod', **locals())
    if dim is not None and not isinstance(dim, list):
        if isinstance(dim, tuple):
            dim = list(dim)
        elif isinstance(dim, int):
            dim = [dim]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".
                format(type(dim)))
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'reduce_prod')
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_prod',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None and dim != [] else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None or dim == [] or
            len(dim) == len(input.shape) else False
        })
    return out


def reduce_all(input, dim=None, keep_dim=False, name=None):
    """

    This OP computes the ``logical and`` of tensor elements over the given dimension, and output the result.

    Args:
        input (Tensor): the input tensor, it's data type should be `bool`.
        dim (list|int|optional): The dimension along which the logical and is computed.
            If :attr:`None`, compute the logical and over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`. The default value is None.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true. The default value is False.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically. The default value is None.

    Returns:
        Tensor, the output data type is bool. : The reduced tensor variable with ``logical and`` in given dims.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import numpy as np

            # x is a bool Tensor variable with following elements:
            #    [[True, False]
            #     [True, True]]
            x = fluid.layers.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
            x = fluid.layers.cast(x, 'bool')

            out = fluid.layers.reduce_all(x)  # False
            out = fluid.layers.reduce_all(x, dim=0)  # [True, False]
            out = fluid.layers.reduce_all(x, dim=-1)  # [False, True]
            # keep_dim=False, x.shape=(2,2), out.shape=(2,)

            out = fluid.layers.reduce_all(x, dim=1, keep_dim=True)  # [[False], [True]]
            # keep_dim=True, x.shape=(2,2), out.shape=(2,1)

    """
    check_variable_and_dtype(input, 'input', ('bool'), 'reduce_all')
    helper = LayerHelper('reduce_all', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_all',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None and dim != [] else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None or dim == [] or
            len(dim) == len(input.shape) else False
        })
    return out


def reduce_any(input, dim=None, keep_dim=False, name=None):
    """
    This OP computes the ``logical or`` of tensor elements over the given dimension, and output the result.

    Args:
        input (Tensor): the input tensor, it's data type should be `bool`.
        dim (list|int|optional): The dimension along which the logical and is computed.
            If :attr:`None`, compute the logical and over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`. The default value is None.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true. The default value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output data type is bool. : The reduced tensor variable with ``logical or`` in given dims.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import numpy as np

            # x is a bool Tensor variable with following elements:
            #    [[True, False]
            #     [False, False]]
            x = fluid.layers.assign(np.array([[1, 0], [0, 0]], dtype='int32'))
            x = fluid.layers.cast(x, 'bool')

            out = fluid.layers.reduce_any(x)  # True
            out = fluid.layers.reduce_any(x, dim=0)  # [True, False]
            out = fluid.layers.reduce_any(x, dim=-1)  # [True, False]
            # keep_dim=False, x.shape=(2,2), out.shape=(2,)

            out = fluid.layers.reduce_any(x, dim=1,
                                     keep_dim=True)  # [[True], [False]]
            # keep_dim=True, x.shape=(2,2), out.shape=(2,1)

    """
    check_variable_and_dtype(input, 'input', ('bool'), 'reduce_any')
    helper = LayerHelper('reduce_any', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_any',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None and dim != [] else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None or dim == [] or
            len(dim) == len(input.shape) else False
        })
    return out


def split(input, num_or_sections, dim=-1, name=None):
    """
    Split the input tensor into multiple sub-Tensors.

    Args:
        input (Tensor): A N-D Tensor. The data type is bool, float16, float32, float64, int32 or int64.
        num_or_sections (int|list|tuple): If ``num_or_sections`` is int, then the ``num_or_sections`` 
            indicates the number of equal sized sub-Tensors that the ``input``
            will be divided into. If ``num_or_sections`` is a list or tuple, the length of it 
            indicates the number of sub-Tensors and the elements in it indicate the sizes of sub-Tensors'
            dimension orderly. The length of the list mustn't be larger than the ``input`` 's size of specified dim.
        dim (int|Tensor, optional): The dimension along which to split, it can be a scalar with type ``int`` or
            a ``Tensor`` with shape [1] and data type ``int32`` or ``int64``. If :math:`dim < 0`,
            the dimension to split along is :math:`rank(input) + dim`. Default is -1.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property. 
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        list(Tensor): The list of segmented Tensors.

    Example:
        .. code-block:: python

            import paddle.fluid as fluid

            # input is a Tensor which shape is [3, 9, 5]
            input = fluid.data(
                 name="input", shape=[3, 9, 5], dtype="float32")

            out0, out1, out2 = fluid.layers.split(input, num_or_sections=3, dim=1)
            # out0.shape [3, 3, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 3, 5]

            out0, out1, out2 = fluid.layers.split(input, num_or_sections=[2, 3, 4], dim=1)
            # out0.shape [3, 2, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 4, 5]

            out0, out1, out2 = fluid.layers.split(input, num_or_sections=[2, 3, -1], dim=1)
            # out0.shape [3, 2, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 4, 5]
            
            # dim is negative, the real dim is (rank(input) + axis) which real
            # value is 1.
            out0, out1, out2 = fluid.layers.split(input, num_or_sections=3, dim=-2)
            # out0.shape [3, 3, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 3, 5]

    """
    if in_dygraph_mode():
        num = None
        attrs = ()

        if isinstance(dim, Variable):
            dim = dim.numpy()
            dim = dim.item(0)
        assert len(input.shape) + dim >= 0, "(rank(x) + axis) must >= 0"
        dim = (len(input.shape) + dim) if dim < 0 else dim
        attrs += ('axis', dim)

        if isinstance(num_or_sections, int):
            num = num_or_sections
            attrs += ('num', num_or_sections)
        elif isinstance(num_or_sections, (list, tuple)):
            num = len(num_or_sections)
            if utils._contain_var(num_or_sections):
                for index, item in enumerate(num_or_sections):
                    if isinstance(item, Variable):
                        num_or_sections[index] = num_or_sections[index].numpy()[
                            0]
                attrs += ('sections', list(num_or_sections))
            else:
                attrs += ('sections', list(num_or_sections))
        else:
            raise TypeError(
                "The type of 'num_or_sections' in split must be int, list or tuple in imperative mode, but "
                "received %s." % (type(num_or_sections)))
        return _C_ops.split(input, num, *attrs)

    check_variable_and_dtype(
        input, 'input',
        ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'], 'split')
    check_type(num_or_sections, 'num_or_sections', (list, int, tuple), 'split')
    check_type(dim, 'dim', (int, Variable), 'split')
    if isinstance(dim, Variable):
        check_dtype(dim.dtype, 'dim', ['int32', 'int64'], 'split')

    helper = LayerHelper('split', **locals())

    input_shape = input.shape
    inputs = {'X': input}
    attrs = {'num': num_or_sections if isinstance(num_or_sections, int) else 0}

    def _get_SectionsTensorList(one_list):
        tensor_list = []
        unk_dim_idx = -1
        for idx, dim_size in enumerate(one_list):
            if isinstance(dim_size, Variable):
                dim_size.stop_gradient = True
                tensor_list.append(dim_size)
            else:
                assert (isinstance(dim_size, int))
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one value of 'num_or_section' in split can "
                        "be -1. But received num_or_section[%d] is also -1." %
                        idx)
                    unk_dim_idx = idx
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant(
                    [1], 'int32', dim_size, force_cpu=True, out=temp_out)
                tensor_list.append(temp_out)
        return tensor_list

    if isinstance(dim, Variable):
        dim.stop_gradient = True
        inputs['AxisTensor'] = dim
    else:
        assert len(input.shape) + dim >= 0, "(rank(x) + axis) must >= 0"
        dim = (len(input_shape) + dim) if dim < 0 else dim
        attrs['axis'] = dim

    if isinstance(num_or_sections, int):
        assert num_or_sections > 1, 'num_or_sections must be more than 1.'
        if isinstance(dim, int) and input_shape[dim] > 0:
            assert input_shape[dim] % num_or_sections ==0, \
                "The input's size along the split dimension " \
                "must be evenly divisible by Attr(num_or_sections). " \
                "But %d is not evenly divisible by %d. " % (num_or_sections,input_shape[dim])
        num = num_or_sections
    else:
        if isinstance(dim, int) and input_shape[dim] > 0:
            assert len(num_or_sections) <= input_shape[
                dim], 'len(num_or_sections) must not be more than input.shape[dim].'
        num = len(num_or_sections)
        attrs['sections'] = list(
            map(lambda ele: -1 if isinstance(ele, Variable) else ele,
                num_or_sections))
        if utils._contain_var(num_or_sections):
            inputs['SectionsTensorList'] = _get_SectionsTensorList(
                num_or_sections)

    outs = [
        helper.create_variable_for_type_inference(dtype=helper.input_dtype())
        for i in range(num)
    ]
    helper.append_op(
        type='split', inputs=inputs, outputs={'Out': outs}, attrs=attrs)
    return outs


def l2_normalize(x, axis, epsilon=1e-12, name=None):
    r"""

    This op normalizes `x` along dimension `axis` using an L2
    norm. For a 1-D tensor (`dim` is fixed to 0), this layer computes

    .. math::

        y = \\frac{x}{ \sqrt{\sum {x^2} + epsion }}

    For `x` with more dimensions, this layer independently normalizes each 1-D
    slice along dimension `axis`.

    Args:
        x(Variable|list): The input tensor could be N-D tensor, and the input data type could be float16, float32 or float64.
        axis(int): The axis on which to apply normalization. If `axis < 0`, \
            the dimension to normalization is rank(X) + axis. -1 is the
            last dimension.
        epsilon(float): The epsilon value is used to avoid division by zero, \
            the default value is 1e-12.
    name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: The output has the same shape and data type with `x`.

    Examples:

    .. code-block:: python
        :name: code-example1
        
        import paddle
        
        X = paddle.randn(shape=[3, 5], dtype='float64')
        out = paddle.fluid.layers.l2_normalize(X, axis=-1)
        print(out)

        # [[ 0.21558504  0.56360189  0.47466096  0.46269539 -0.44326736]
        #  [-0.70602414 -0.52745777  0.37771788 -0.2804768  -0.04449922]
        #  [-0.33972208 -0.43014923  0.31772556  0.76617881 -0.10761525]]

    """

    if len(x.shape) == 1:
        axis = 0
    if in_dygraph_mode():
        _, out = _C_ops.norm(x, 'axis', 1
                             if axis is None else axis, 'epsilon', epsilon)
        return out

    check_variable_and_dtype(x, "X", ("float16", "float32", "float64"), "norm")

    helper = LayerHelper("l2_normalize", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    norm = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="norm",
        inputs={"X": x},
        outputs={"Out": out,
                 "Norm": norm},
        attrs={
            "axis": 1 if axis is None else axis,
            "epsilon": epsilon,
        })
    return out


@deprecated(since="2.0.0", update_to="paddle.matmul")
def matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None):
    """
    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is rank-1 of shape :math:`[D]`, then for
      :math:`x` it is treated as :math:`[1, D]` in nontransposed form and as
      :math:`[D, 1]` in transposed form, whereas for :math:`y` it is the
      opposite: It is treated as :math:`[D, 1]` in nontransposed form and as
      :math:`[1, D]` in transposed form.

    - After transpose, the two tensors are 2-D or n-D and matrix multiplication
      performs in the following way.

      - If both are 2-D, they are multiplied like conventional matrices.
      - If either is n-D, it is treated as a stack of matrices residing in the
        last two dimensions and a batched matrix multiply supporting broadcast
        applies on the two tensors.

    Also note that if the raw tensor :math:`x` or :math:`y` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a Tensor or LoDTensor.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        alpha (float): The scale of output. Default 1.0.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The product Tensor (or LoDTensor) variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], y: [B, ..., K, N]
            # fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

            # x: [B, M, K], y: [B, K, N]
            # fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [B, M, K], y: [K, N]
            # fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [M, K], y: [K, N]
            # fluid.layers.matmul(x, y)  # out: [M, N]

            # x: [B, M, K], y: [K]
            # fluid.layers.matmul(x, y)  # out: [B, M]

            # x: [K], y: [K]
            # fluid.layers.matmul(x, y)  # out: [1]

            # x: [M], y: [N]
            # fluid.layers.matmul(x, y, True, True)  # out: [M, N]

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            x = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')
            y = fluid.layers.data(name='y', shape=[3, 2], dtype='float32')
            out = fluid.layers.matmul(x, y, True, True)
    """
    if in_dygraph_mode():
        out = _varbase_creator(dtype=x.dtype)
        _C_ops.matmul(x, y, out, 'transpose_X', transpose_x, 'transpose_Y',
                      transpose_y, 'alpha', float(alpha))
        return out

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(
                val, name, ['float16', 'float32', 'float64'], 'matmul')
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if transpose_x:
            x_shape[-2], x_shape[-1] = x_shape[-1], x_shape[-2]
        if transpose_y:
            y_shape[-2], y_shape[-1] = y_shape[-1], y_shape[-2]
        if x_shape[-1] != y_shape[-2]:
            assert (x_shape[-1] == -1) or (y_shape[-2] == -1),                         \
                "After performing an optional transpose, Input X's width should be "   \
                "equal to Y's width for multiplication "                               \
                "prerequisites. But received X's shape: %s, Y's shape: %s\n" %         \
                (x_shape, y_shape)

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

    attrs = {
        'transpose_X': transpose_x,
        'transpose_Y': transpose_y,
        'alpha': float(alpha),
    }

    __check_input(x, y)

    helper = LayerHelper('matmul', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='matmul',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs=attrs)
    return out


def topk(input, k, name=None):
    """
    :alias_main: paddle.topk
	:alias: paddle.topk,paddle.tensor.topk,paddle.tensor.search.topk
	:old_api: paddle.fluid.layers.topk

    This OP is used to find values and indices of the k largest entries
    for the last dimension.

    If the input is a 1-D Tensor, finds the k largest entries and outputs
    their values and indices.

    If the input is a Tensor with higher rank, this operator computes the top k
    entries along the last dimension.

    .. code-block:: text

        Case 1:

          Input:
            input.shape = [3, 4]
            input.data = [[5, 4, 2, 3],
                     [9, 7, 10, 25],
                     [6, 2, 10, 1]]
            k = 2

          Output:
            The first output:
            values.shape = [3, 2]
            values.data = [[5, 4],
                      [10, 25],
                      [6, 10]]

            The second output:
            indices.shape = [3, 2]
            indices.data = [[0, 1],
                       [2, 3],
                       [0, 2]]

    Args:
        input(Variable): The input tensor. Support data types: float32, float64.
        k(int | Variable): The number of top elements to look for along the last dimension
                           of input tensor.
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Values (Variable): Input tensor's k largest elements along each last dimensional slice. The dimension is: :math:`input.shape[:-1]+[k]`.
        Indices (Variable): Indices of k largest elements alone the last dimension of input. The dimension is same as values.

    Raises:
        ValueError: If :math:`k < 1` or :math:`k > last dimension of input`.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            # set batch size=None
            input = fluid.data(name="input", shape=[None, 13, 11], dtype='float32')
            top5_values, top5_indices = layers.topk(input, k=5) # top5_values.shape[None, 13, 5], top5_indices.shape=[None, 13, 5]

            # 1D Tensor
            input1 = fluid.data(name="input1", shape=[None, 13], dtype='float32')
            top5_values, top5_indices = layers.topk(input1, k=5) #top5_values.shape=[None, 5], top5_indices.shape=[None, 5]

            # k=Variable
            input2 = fluid.data(name="input2", shape=[None, 13, 11], dtype='float32')
            vk = fluid.data(name="vk", shape=[None, 1], dtype='int32') # save k in vk.data[0]
            vk_values, vk_indices = layers.topk(input2, k=vk) #vk_values.shape=[None, 13, k], vk_indices.shape=[None, 13, k]

    """
    if in_dygraph_mode():
        _k = k.numpy().item(0) if isinstance(k, Variable) else k
        out, indices = _C_ops.top_k(input, 'k', _k)
        out.stop_gradient = True
        indices.stop_gradient = True
        return out, indices

    inputs = {"X": [input]}
    attrs = {}
    if isinstance(k, Variable):
        inputs['K'] = [k]
    else:
        attrs = {'k': k}

    helper = LayerHelper("top_k", **locals())
    values = helper.create_variable_for_type_inference(dtype=input.dtype)
    indices = helper.create_variable_for_type_inference(dtype="int64")

    helper.append_op(
        type="top_k",
        inputs=inputs,
        outputs={"Out": [values],
                 "Indices": [indices]},
        attrs=attrs)
    values.stop_gradient = True
    indices.stop_gradient = True
    return values, indices


def ctc_greedy_decoder(input,
                       blank,
                       input_length=None,
                       padding_value=0,
                       name=None):
    r"""
    This op is used to decode sequences by greedy policy by the following steps:

    1. Get the indexes of maximum value for each row in input. a.k.a.
       numpy.argmax(input, axis=0).
    2. For each sequence in result of step1, merge repeated tokens between two
       blanks and delete all blanks.

    This op is implemented in two modes: lod and padding, either of them can be used.
    The input can be either LoDTensor or Tensor, corresponding to lod and padding
    mode respectively.

    A simple example as below:

    .. code-block:: text

        Given:
        (1) for lod mode:

        input.data = [[0.6, 0.1, 0.3, 0.1],
                      [0.3, 0.2, 0.4, 0.1],
                      [0.1, 0.5, 0.1, 0.3],
                      [0.5, 0.1, 0.3, 0.1],

                      [0.5, 0.1, 0.3, 0.1],
                      [0.2, 0.2, 0.2, 0.4],
                      [0.2, 0.2, 0.1, 0.5],
                      [0.5, 0.1, 0.3, 0.1]]

        input.lod = [[4, 4]]

        Computation:

        step1: Apply argmax to first input sequence which is input.data[0:4]. Then we get:
               [[0], [2], [1], [0]]
        step2: merge repeated tokens and remove blank which is 0. Then we get first output sequence:
               [[2], [1]]

        Finally:

        output.data = [[2],
                       [1],
                       [3]]

        output.lod = [[2, 1]]

        (2) for padding mode:

         input.data = [[[0.6, 0.1, 0.3, 0.1],
                        [0.3, 0.2, 0.4, 0.1],
                        [0.1, 0.5, 0.1, 0.3],
                        [0.5, 0.1, 0.3, 0.1]],

                       [[0.5, 0.1, 0.3, 0.1],
                        [0.2, 0.2, 0.2, 0.4],
                        [0.2, 0.2, 0.1, 0.5],
                        [0.5, 0.1, 0.3, 0.1]]]

        input_length.data = [[4], [4]]
        input.shape = [2, 4, 4]

        step1: Apply argmax to first input sequence which is input.data[0:4]. Then we get:
               [[0], [2], [1], [0]], for input.data[4:8] is [[0], [3], [3], [0]], shape is [2,4,1]
        step2: Change the argmax result to use padding mode, then argmax result is
                [[0, 2, 1, 0], [0, 3, 3, 0]], shape is [2, 4], lod is [], input_length is [[4], [4]]
        step3: Apply ctc_align to padding argmax result, padding_value is 0

        Finally:
        output.data = [[2, 1, 0, 0],
                       [3, 0, 0, 0]]
        output_length.data = [[2], [1]]


    Parameters:

        input(Variable): the probabilities of variable-length sequences. When in lod mode,
                         it is a 2-D LoDTensor with LoD information. It's shape is [Lp, num_classes + 1]
                         where Lp is the sum of all input sequences' length and
                         num_classes is the true number of classes. When in padding mode,
                         it is a 3-D Tensor with padding, It's shape is [batch_size, N, num_classes + 1].
                         (not including the blank label). The data type can be float32 or float64.
        blank(int): the blank label index of Connectionist Temporal
                    Classification (CTC) loss, which is in the half-opened
                    interval [0, num_classes + 1).
        input_length(Variable, optional): 2-D LoDTensor, shape is [batch_size, 1], data type is int64.
                                 It is used for padding mode. In lod mode, input_length is None.
        padding_value(int): padding value.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        For lod mode, returns the result of CTC greedy decoder, 2-D LoDTensor, shape is [Lp, 1], \
        data type is int64. 'Lp' is the sum of all output sequences' length. If all the sequences \
        in result were empty, the result LoDTensor will be [-1] with  empty \
        LoD [[]].

        For padding mode, returns a tuple of (output, output_length), which was described as below:

        output, 2-D Tensor, shape is [batch_size, N], data type is int64.

        output_length, 2-D Tensor, shape is [batch_size, 1], data type is int64. It is the length of \
                           each sequence of output for padding mode.

    Return type:
        For lod mode: Variable

        For padding mode: tuple of two Variables (output, output_length).


    Examples:
        .. code-block:: python

            # for lod mode
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 8], dtype='float32', lod_level=1)
            cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)

            # for padding mode
            x_pad = fluid.data(name='x_pad', shape=[10, 4, 8], dtype='float32')
            x_pad_len = fluid.data(name='x_pad_len', shape=[10, 1], dtype='int64')
            out, out_len = fluid.layers.ctc_greedy_decoder(input=x_pad, blank=0,
                            input_length=x_pad_len)

    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'ctc_greedy_decoder')

    helper = LayerHelper("ctc_greedy_decoder", **locals())
    _, topk_indices = topk(input, k=1)

    # ctc align op
    ctc_out = helper.create_variable_for_type_inference(dtype="int64")

    if input_length is None:
        helper.append_op(
            type="ctc_align",
            inputs={"Input": [topk_indices]},
            outputs={"Output": [ctc_out]},
            attrs={"merge_repeated": True,
                   "blank": blank})
        return ctc_out
    else:
        ctc_out_len = helper.create_variable_for_type_inference(dtype="int64")
        ctc_input = squeeze(topk_indices, [2])

        helper.append_op(
            type="ctc_align",
            inputs={"Input": [ctc_input],
                    "InputLength": [input_length]},
            outputs={"Output": [ctc_out],
                     "OutputLength": [ctc_out_len]},
            attrs={
                "merge_repeated": True,
                "blank": blank,
                "padding_value": padding_value
            })
        return ctc_out, ctc_out_len


def transpose(x, perm, name=None):
    """
    Permute the data dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    Args:
        x (Tensor): The input Tensor. It is a N-D Tensor of data types bool, float32, float64, int32.
        perm (list|tuple): Permute the input according to the data of perm.
        name (str): The name of this layer. It is optional.

    Returns:
        Tensor: A transposed n-D Tensor, with data type being bool, float32, float64, int32, int64.

    For Example:

        .. code-block:: text

         x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
             [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
         shape(x) =  [2,3,4]

         # Example 1
         perm0 = [1,0,2]
         y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
                   [[ 5  6  7  8]  [17 18 19 20]]
                   [[ 9 10 11 12]  [21 22 23 24]]]
         shape(y_perm0) = [3,2,4]

         # Example 2
         perm1 = [2,1,0]
         y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
                   [[ 2 14] [ 6 18] [10 22]]
                   [[ 3 15]  [ 7 19]  [11 23]]
                   [[ 4 16]  [ 8 20]  [12 24]]]
         shape(y_perm1) = [4,3,2]

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.randn([2, 3, 4])
            x_transposed = paddle.transpose(x, perm=[1, 0, 2])
            print(x_transposed.shape)
            # [3L, 2L, 4L]

    """
    if in_dygraph_mode():
        out, _ = _C_ops.transpose2(x, 'axis', perm)
        return out

    check_variable_and_dtype(
        x, 'x', ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
        'transpose')
    check_type(perm, 'perm', (list, tuple), 'transpose')
    if isinstance(perm, tuple):
        perm = list(perm)
    if len(perm) != len(x.shape):
        raise ValueError(
            "Input(perm) is the permutation of dimensions of Input(x), "
            "its length should be equal to dimensions of Input(x), "
            "but received dimension of Input(x) is %s, "
            "the length of Input(perm) is %s." % (len(x.shape), len(perm)))
    for idx, dim in enumerate(perm):
        if dim >= len(x.shape):
            raise ValueError(
                "Each element in Input(perm) should be less than Input(x)'s dimension, "
                "but %d-th element in Input(perm) is %d which exceeds Input(x)'s "
                "dimension %d." % (idx, perm[idx], len(x.shape)))

    helper = LayerHelper('transpose', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='transpose2',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'XShape': [x_shape]},
        attrs={'axis': perm})
    return out


def im2sequence(input,
                filter_size=1,
                stride=1,
                padding=0,
                input_image_size=None,
                out_stride=1,
                name=None):
    r"""
    :api_attr: Static Graph

    Extracts image patches from the input tensor to form a tensor of shape
    {input.batch_size * output_height * output_width, filter_size_height *
    filter_size_width * input.channels}. This op use filter to scan images
    and convert these images to sequences. After expanding, the number of time step are
    output_height * output_width for an image, in which output_height and
    output_width are calculated by below equation:

    .. math::

        output\_height  = 1 + \
            (padding\_up + padding\_down + input\_height  - filter\_size\_height  + stride\_height - 1) / stride\_height \\\\
        output\_width  = 1 + \
            (padding\_left + padding\_right + input\_width  - filter\_size\_width  + stride\_width - 1) / stride\_width

    And the dimension of each time step is filter_size_height * filter_size_width * input.channels.

    Parameters:
        input (Variable): The input should be a 4-D Tensor in :math:`NCHW` format. The data type is float32.

        filter_size(int32 | List[int32]): The filter size. If filter_size is a List,
            it must contain two integers, :math:`[filter\_size\_height, filter\_size\_width]` .
            Otherwise, the filter size will be a square :math:`[filter\_size, filter\_size]` . Default is 1.

        stride(int32 | List[int32]): The stride size. If stride is a List, it must
            contain two integers, :math:`[stride\_height, stride\_width]` . Otherwise, the stride size will be a square :math:`[stride\_size, stride\_size]` . Default is 1.

        padding(int32 | List[int32]): The padding size. If padding is a List, it can
            contain four integers like :math:`[padding\_up, padding\_left, padding\_down, padding\_right]` to indicate
            paddings of four direction.  Or it can contain two integers :math:`[padding\_height, padding\_width]` which means
            padding_up = padding_down = padding_height and
            padding_left = padding_right = padding_width. Otherwise, a scalar padding means
            padding_up = padding_down = padding_left = padding_right = padding.
            Default is 0.

        input_image_size(Variable, optional): the input contains image real size.It's dim
            is :math:`[batchsize, 2]` . It is just for batch inference when not None. Default is None.

        out_stride(int32 | List[int32]): The scaling of image through CNN. It is valid only when input_image_size is not None.
            If out_stride is List,  it must contain two integers,
            :math:`[out\_stride\_height, out\_stride\_W]` . Otherwise,
            the out_stride_height = out_stride_width = out_stride. Default is 1.

        name (str, optional): The default value is None.  Normally there is no need for
                    user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
            The output is a 2-D LoDTensor with shape {input.batch\_size * output\_height * output\_width, \
            filter\_size\_height * filter\_size\_width * input.channels}. The data type is float32.

    Return Type: Variable

    Examples:

        .. code-block:: text

            Given:

            x = [[[[ 6.  2.  1.]
                   [ 8.  3.  5.]
                   [ 0.  2.  6.]]

                  [[ 2.  4.  4.]
                   [ 6.  3.  0.]
                   [ 6.  4.  7.]]]

                 [[[ 6.  7.  1.]
                   [ 5.  7.  9.]
                   [ 2.  4.  8.]]

                  [[ 1.  2.  1.]
                   [ 1.  3.  5.]
                   [ 9.  0.  8.]]]]

            x.dims = {2, 2, 3, 3}

            And:

            filter = [2, 2]
            stride = [1, 1]
            padding = [0, 0]

            Then:

            output.data = [[ 6.  2.  8.  3.  2.  4.  6.  3.]
                           [ 2.  1.  3.  5.  4.  4.  3.  0.]
                           [ 8.  3.  0.  2.  6.  3.  6.  4.]
                           [ 3.  5.  2.  6.  3.  0.  4.  7.]
                           [ 6.  7.  5.  7.  1.  2.  1.  3.]
                           [ 7.  1.  7.  9.  2.  1.  3.  5.]
                           [ 5.  7.  2.  4.  1.  3.  9.  0.]
                           [ 7.  9.  4.  8.  3.  5.  0.  8.]]

            output.dims = {8, 8}

            output.lod = [[4, 4]]

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            data = fluid.data(name='data', shape=[None, 3, 32, 32],
                                     dtype='float32')
            output = fluid.layers.im2sequence(
                input=data, stride=[1, 1], filter_size=[2, 2])


    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")

    check_variable_and_dtype(input, 'input', ['float32'], 'im2sequence')

    if isinstance(filter_size, int):
        filter_size = [filter_size, filter_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding]
    if len(padding) == 2:
        padding.append(padding[0])
        padding.append(padding[1])
    inputs = {"X": input}
    attrs = {"kernels": filter_size, "strides": stride, "paddings": padding}
    if input_image_size:
        if isinstance(out_stride, int):
            out_stride = [out_stride, out_stride]
        inputs["Y"] = input_image_size
        attrs["out_stride"] = out_stride
    helper = LayerHelper('im2sequence', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='im2sequence', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


@templatedoc()
def row_conv(input, future_context_size, param_attr=None, act=None):
    """
    :api_attr: Static Graph

    ${comment}

    Args:
        input (${x_type}): ${x_comment}.
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc.
        act (str): Non-linear activation to be applied to output variable.

    Returns:
        ${out_comment}.

    Examples:

      .. code-block:: python

        # for LodTensor inputs
        import paddle
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[9, 16],
                               dtype='float32', lod_level=1)
        out = paddle.static.nn.row_conv(input=x, future_context_size=2)
        # for Tensor inputs
        x = paddle.static.data(name='x', shape=[9, 4, 16], dtype='float32')
        out = paddle.static.nn.row_conv(input=x, future_context_size=2)
    """
    helper = LayerHelper('row_conv', **locals())
    check_variable_and_dtype(input, 'input', ['float32'], 'row_conv')
    dtype = helper.input_dtype()
    filter_shape = [future_context_size + 1, input.shape[-1]]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='row_conv',
        inputs={'X': [input],
                'Filter': [filter_param]},
        outputs={'Out': [out]})
    return helper.append_activation(out)


@templatedoc()
def multiplex(inputs, index, name=None):
    """

    Based on the given index parameter, the OP selects a specific row from each input Tensor to construct the output Tensor.

    If the input of this OP contains :math:`m` Tensors, where :math:`I_{i}` means the i-th input Tensor, :math:`i` between :math:`[0,m)` .

    And :math:`O` means the output, where :math:`O[i]` means the i-th row of the output, then the output satisfies that :math:`O[i] = I_{index[i]}[i]` .

    For Example:

            .. code-block:: text

                Given:

                inputs = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
                          [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]],
                          [[2,0,3,4], [2,1,7,8], [2,2,4,2], [2,3,3,4]],
                          [[3,0,3,4], [3,1,7,8], [3,2,4,2], [3,3,3,4]]]

                index = [[3],[0],[1],[2]]

                out = [[3,0,3,4],    # out[0] = inputs[index[0]][0] = inputs[3][0] = [3,0,3,4]
                       [0,1,3,4],    # out[1] = inputs[index[1]][1] = inputs[0][1] = [0,1,3,4]
                       [1,2,4,2],    # out[2] = inputs[index[2]][2] = inputs[1][2] = [1,2,4,2]
                       [2,3,3,4]]    # out[3] = inputs[index[3]][3] = inputs[2][3] = [2,3,3,4]


    Args:
        inputs (list): The input Tensor list. The list elements are N-D Tensors of data types float32, float64, int32, int64. All input Tensor shapes should be the same and rank must be at least 2.
        index (Tensor): Used to select some rows in the input Tensor to construct an index of the output Tensor. It is a 2-D Tensor with data type int32 or int64 and shape [M, 1], where M is the number of input Tensors.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: Output of multiplex OP, with data type being float32, float64, int32, int64.

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np
            img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
            img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
            inputs = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
            index = paddle.to_tensor(np.array([[1], [0]]).astype(np.int32))
            res = paddle.multiplex(inputs, index)
            print(res) # [array([[5., 6.], [3., 4.]], dtype=float32)]

    """
    if in_dygraph_mode():
        return _C_ops.multiplex(index, inputs)
    helper = LayerHelper('multiplex', **locals())

    check_type(inputs, 'inputs', (list), 'multiplex')
    if len(inputs) < 2:
        raise ValueError(
            "inputs should be a list object with at least 2 elements.")
    for id, x in enumerate(inputs):
        check_variable_and_dtype(x, 'input[' + str(id) + ']',
                                 ['float32', 'float64', 'int32', 'int64'],
                                 'multiplex')
    check_variable_and_dtype(index, "index", ['int32', 'int64'], 'multiplex')

    out = helper.create_variable_for_type_inference(inputs[0].dtype)
    helper.append_op(
        type='multiplex',
        inputs={'X': inputs,
                'Ids': index},
        outputs={'Out': [out]})
    return out


def smooth_l1(x, y, inside_weight=None, outside_weight=None, sigma=None):
    """

    This layer computes the smooth L1 loss for Variable :attr:`x` and :attr:`y`.
    It takes the first dimension of :attr:`x` and :attr:`y` as batch size.
    For each instance, it computes the smooth L1 loss element by element first
    and then sums all the losses. So the shape of output Variable is
    [batch_size, 1].

    Args:
        x (Variable): A tensor with rank at least 2. The input value of smooth
            L1 loss op with shape [batch_size, dim1, ..., dimN].
            A LoDTensor or Tensor with type float32.
        y (Variable): A tensor with rank at least 2. The target value of smooth
            L1 loss op with same shape as :attr:`x`.
            A LoDTensor or Tensor with type float32.
        inside_weight (Variable|None):  A tensor with rank at least 2. This
            input is optional and should have same shape with :attr:`x`. If
            provided, the result of (:attr:`x` - :attr:`y`) will be multiplied
            by this tensor element by element.
            A Tensor with type float32.
        outside_weight (Variable|None): A tensor with rank at least 2. This
            input is optional and should have same shape with :attr:`x`. If
            provided, the out smooth L1 loss will be multiplied by this tensor
            element by element.
            A Tensor with type float32.
        sigma (float|None): Hyper parameter of smooth L1 loss layer. A float
           scalar with default value 1.0.

    Returns:
        Variable: The output smooth L1 loss with shape [batch_size, 1].  A Tensor with type float32.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            paddle.enable_static()
            data = fluid.data(name="x", shape=[-1, 3], dtype="float32")
            label = fluid.data(name="y", shape=[-1, 3], dtype="float32")
            result = fluid.layers.smooth_l1(data,label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            x = np.random.rand(3,3).astype("float32")
            y = np.random.rand(3,3).astype("float32")
            output= exe.run(feed={"x":x, "y":y},
                             fetch_list=[result])
            print(output)

            #[array([[0.08220536],
            #       [0.36652038],
            #      [0.20541131]], dtype=float32)]

    """
    check_variable_and_dtype(x, 'X', ['float32', 'float64'], 'smooth_l1_loss')
    check_variable_and_dtype(y, 'Y', ['float32', 'float64'], 'smooth_l1_loss')

    helper = LayerHelper('smooth_l1_loss', **locals())

    diff = helper.create_variable_for_type_inference(dtype=x.dtype)
    loss = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='smooth_l1_loss',
        inputs={
            'X': x,
            'Y': y,
            'InsideWeight': inside_weight,
            'OutsideWeight': outside_weight
        },
        outputs={'Diff': diff,
                 'Out': loss},
        attrs={'sigma': sigma if sigma is not None else 1.0})
    return loss


@deprecated(since='2.0.0', update_to='paddle.nn.functional.one_hot')
def one_hot(input, depth, allow_out_of_range=False):
    """

    **WARING:** This OP requires the last dimension of Tensor shape must be equal to 1.
    This OP will be deprecated in a future release. It is recommended to use fluid. :ref:`api_fluid_one_hot` .

    The operator converts each id in the input to an one-hot vector with a
    :attr:`depth` length. The value in the vector dimension corresponding to the id
    is 1, and the value in the remaining dimension is 0.

    The shape of output Tensor or LoDTensor is generated by adding :attr:`depth` dimension
    behind the last dimension of the input shape.

    .. code-block:: text

        Example 1 (allow_out_of_range=False):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [3], [0]]
            depth = 4

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.],
                        [1., 0., 0., 0.]]

        Example 2 (allow_out_of_range=True):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [5], [0]]
            depth = 4
            allow_out_of_range = True

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 0.], # This id is 5, which goes beyond depth, so set it all-zeros data.
                        [1., 0., 0., 0.]]

        Example 3 (allow_out_of_range=False):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [5], [0]]
            depth = 4
            allow_out_of_range = False

        output: Throw an exception for Illegal value
            The second dimension in X is 5, which is greater than depth.
            Allow_out_of_range =False means that does not allow the word id to exceed depth,
            so it throws an exception.

    Args:
        input(Variable): Tensor or LoDTensor with shape :math:`[N_1, N_2, ..., N_k, 1]` ,
            which contains at least one dimension and the last dimension must be 1.
            The data type is int32 or int64.
        depth(scalar): An integer defining the :attr:`depth` of the one hot dimension. If input
            is word id, depth is generally the dictionary size.
        allow_out_of_range(bool): A bool value indicating whether the input
            indices could be out of range :math:`[0, depth)` . When input indices are
            out of range, exceptions :code:`Illegal value` is raised if :attr:`allow_out_of_range`
            is False, or zero-filling representations is created if it is set True.
            Default: False.

    Returns:
        Variable: The one-hot representations of input. A Tensor or LoDTensor with type float32.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            # Correspond to the first example above, where label.shape is [4, 1] and one_hot_label.shape is [4, 4].
            label = fluid.data(name="label", shape=[4, 1], dtype="int64")
            one_hot_label = fluid.layers.one_hot(input=label, depth=4)
    """
    if in_dygraph_mode():
        if isinstance(depth, Variable):
            depth = depth.numpy()
            assert depth.shape == (
                1, ), "depth of type Variable should have shape [1]"
            depth = depth.item(0)
        out = _C_ops.one_hot(input, 'depth', depth, 'allow_out_of_range',
                             allow_out_of_range)
        out.stop_gradient = True
        return out

    helper = LayerHelper("one_hot", **locals())
    check_variable_and_dtype(input, 'input', ['int32', 'int64'], 'one_hot')
    check_type(depth, 'depth', (six.integer_types, Variable), 'one_hot')
    one_hot_out = helper.create_variable_for_type_inference(dtype='float32')

    if not isinstance(depth, Variable):
        # user attribute
        inputs = {'X': input}
        attrs = {'depth': depth, 'allow_out_of_range': allow_out_of_range}
    else:
        depth.stop_gradient = True
        inputs = {'X': input, 'depth_tensor': depth}
        attrs = {'allow_out_of_range': allow_out_of_range}
    helper.append_op(
        type="one_hot",
        inputs=inputs,
        attrs=attrs,
        outputs={'Out': one_hot_out})
    one_hot_out.stop_gradient = True
    return one_hot_out


def autoincreased_step_counter(counter_name=None, begin=1, step=1):
    """
    :api_attr: Static Graph

    Create an auto-increase variable. which will be automatically increased
    by 1 in every iteration. By default, the first return of this counter is 1,
    and the step size is 1.

    Args:
        counter_name(str, optional): The counter name. Default '@STEP_COUNTER@'.
        begin(int, optional): The first return value of this counter. Default 1.
        step(int, optional): The step size. Default 1.

    Returns:
        Variable: The auto-increased Variable with data type int64.

    Examples:
        .. code-block:: python

           import paddle.fluid as fluid
           import paddle
           paddle.enable_static()
           global_step = fluid.layers.autoincreased_step_counter(
               counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)
    """
    helper = LayerHelper('global_step_counter')
    if counter_name is None:
        counter_name = '@STEP_COUNTER@'
    counter, is_new_var = helper.create_or_get_global_variable(
        name=counter_name,
        dtype='int64',
        shape=[1],
        persistable=True,
        belong_to_optimizer=True)
    if is_new_var:
        helper.set_variable_initializer(
            counter, initializer=Constant(
                value=begin - 1, force_cpu=True))
        helper.main_program.global_block()._prepend_op(
            type='increment',
            inputs={'X': [counter]},
            outputs={'Out': [counter]},
            attrs={'step': float(step)})
        counter.stop_gradient = True

    return counter


def reshape(x, shape, actual_shape=None, act=None, inplace=False, name=None):
    r"""
    :alias_main: paddle.reshape
	:alias: paddle.reshape,paddle.tensor.reshape,paddle.tensor.manipulation.reshape

    This operator changes the shape of ``x`` without changing its data.

    The target shape can be given by ``shape`` or ``actual_shape``.
    When ``shape`` and ``actual_shape`` are set at the same time,
    ``actual_shape`` has a higher priority than ``shape``
    but at this time ``shape`` can only be an integer list or tuple, and ``shape`` still should be set correctly to
    guarantee shape inference in compile-time.

    Some tricks exist when specifying the target shape.

    1. -1 means the value of this dimension is inferred from the total element
    number of x and remaining dimensions. Thus one and only one dimension can
    be set -1.

    2. 0 means the actual dimension value is going to be copied from the
    corresponding dimension of x. The index of 0s in shape can not exceed
    the dimension of x.

    Here are some examples to explain it.

    1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [6, 8], the reshape operator will transform x into a 2-D tensor with
    shape [6, 8] and leaving x's data unchanged.

    2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    specified is [2, 3, -1, 2], the reshape operator will transform x into a
    4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this
    case, one dimension of the target shape is set to -1, the value of this
    dimension is inferred from the total element number of x and remaining
    dimensions.

    3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor
    with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case,
    besides -1, 0 means the actual dimension value is going to be copied from
    the corresponding dimension of x.

    **Note**:
        The parameter ``actual_shape`` will be deprecated in the future and only use ``shape`` instead to represent the target shape.

    Args:
        x(Tensor): An N-D Tensor. The data type is ``float32``, ``float64``, ``int32`` or ``int64``.
        shape(list|tuple|Tensor): Define the target shape. At most one dimension of the target shape can be -1.
                        The data type is ``int32`` . If ``shape`` is a list or tuple, the elements of it should be integers or Tensors with shape [1].
                        If ``shape`` is an Tensor, it should be an 1-D Tensor .
        actual_shape(variable, optional): An 1-D ``Tensor`` or ``LoDTensor`` . The data type is ``int32`` . If provided, reshape
                                according to this given shape rather than ``shape`` specifying shape.
                                That is to say ``actual_shape`` has a higher priority
                                than ``shape(list|tuple)`` but not ``shape(Tensor)``. \
                                This argument ``actual_shape`` will be removed in a future version. \
                                Instructions for updating: ``actual_shape`` will be removed in future versions and replaced by ``shape``.
        act (str, optional): The non-linear activation to be applied to the reshaped input. Default None.
        inplace(bool, optional): If ``inplace`` is True, the input and output of ``layers.reshape``
                       are the same variable. Otherwise, the input and output of
                       ``layers.reshape`` are different variable. Default False. Note that if ``x``
                       is more than one OPs' input, ``inplace`` must be False.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.
                            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A reshaped Tensor with the same data type as ``x``. It is a new tensor variable if ``inplace`` is ``False``, otherwise it is ``x``. If ``act`` is None, return the reshaped tensor variable, otherwise return the activated tensor variable.


    Examples:
        .. code-block:: python
            
            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()
            
            # example 1:
            # attr shape is a list which doesn't contain Tensors.
            data_1 = fluid.data(
              name='data_1', shape=[2, 4, 6], dtype='float32')
            reshaped_1 = fluid.layers.reshape(
              x=data_1, shape=[-1, 0, 3, 2])
            # the shape of reshaped_1 is [2,4,3,2].

            # example 2:
            # attr shape is a list which contains Tensors.
            data_2 = fluid.layers.fill_constant([2,25], "int32", 3)
            dim = fluid.layers.fill_constant([1], "int32", 5)
            reshaped_2 = fluid.layers.reshape(data_2, shape=[dim, 10])
            # the shape of reshaped_2 is [5,10].

            # example 3:
            data_3 = fluid.data(
              name="data_3", shape=[2,4,6], dtype='float32')
            reshaped_3 = fluid.layers.reshape(x=data_3, shape=[6,8])
            # the shape of reshaped_3 is [6,8].
    """
    if in_dygraph_mode():
        #TODO(zhiqiu): enable inplace in dygraph mode.
        if inplace:
            warnings.warn(
                "Inplace on reshape is not allowed and will be discarded in dygraph mode currently."
            )
        if isinstance(shape, (list, tuple)):
            shape = [
                item.numpy().item(0) if isinstance(item, Variable) else item
                for item in shape
            ]
            out, _ = _C_ops.reshape2(x, None, 'shape', shape)
        elif isinstance(shape, Variable):
            shape.stop_gradient = True
            out, _ = _C_ops.reshape2(x, shape)
        else:
            raise ValueError(
                "shape must be an instance of `list`, `tuple` or `Variable`,"
                " got '{}.'".format(type(shape)))

        return dygraph_utils._append_activation_in_dygraph(out, act)

    check_variable_and_dtype(x, 'x', [
        'float16', 'float32', 'float64', 'int32', 'int64', 'bool', 'uint16'
    ], 'reshape')
    check_type(shape, 'shape', (list, tuple, Variable), 'reshape')
    check_type(actual_shape, 'actual_shape', (Variable, type(None)), 'reshape')

    helper = LayerHelper("reshape2", **locals())

    def get_attr_shape(list_shape):
        unk_dim_idx = -1
        attrs_shape = []
        for dim_idx, dim_size in enumerate(list_shape):
            if isinstance(dim_size, Variable):
                attrs_shape.append(-1)
            else:
                attrs_shape.append(dim_size)
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one dimension value of 'shape' in reshape can "
                        "be -1. But received shape[%d] is also -1." % dim_idx)
                    unk_dim_idx = dim_idx
                elif dim_size == 0:
                    assert dim_idx < len(x.shape), (
                        "The index of 0 in `shape` must be less than "
                        "the input tensor X's dimensions. "
                        "But received shape[%d] = 0, X's dimensions = %d." %
                        (dim_idx, len(x.shape)))
                else:
                    assert dim_size > 0, (
                        "Each dimension value of 'shape' in reshape must not "
                        "be negative except one unknown dimension. "
                        "But received shape[%d] = %s." %
                        (dim_idx, str(dim_size)))
        return attrs_shape

    inputs = {"X": x}
    attrs = {}
    if isinstance(shape, Variable):
        shape.stop_gradient = True
        inputs["Shape"] = shape
    elif isinstance(shape, (list, tuple)):
        assert len(shape) > 0, ("The size of 'shape' in reshape can't be zero, "
                                "but received %s." % len(shape))
        attrs["shape"] = get_attr_shape(shape)
        if utils._contain_var(shape):
            inputs['ShapeTensor'] = utils._convert_to_tensor_list(shape)
        elif isinstance(actual_shape, Variable):
            actual_shape.stop_gradient = True
            inputs["Shape"] = actual_shape

    out = x if inplace else helper.create_variable_for_type_inference(
        dtype=x.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="reshape2",
        inputs=inputs,
        attrs=attrs,
        outputs={"Out": out,
                 "XShape": x_shape})

    return helper.append_activation(out)


def squeeze(input, axes, name=None):
    """
    This OP will squeeze single-dimensional entries of input tensor's shape. If axes is provided, will
    remove the dims by axes, the dims selected by axes should be one. If not provide axes, all dims equal
    to one will be deleted.


    .. code-block:: text

        Case1:

          Input:
            X.shape = (1, 3, 1, 5)
            axes = [0]
          Output:
            Out.shape = (3, 1, 5)

        Case2:

          Input:
            X.shape = (1, 3, 1, 5)
            axes = []
          Output:
            Out.shape = (3, 5)

        Case3:

          Input:
            X.shape = [1,3,1,5]
            axes = [-2]
          Output:
            Out.shape = [1,3,5]

    Args:
        input (Variable): The input Tensor. Supported data type: float32, float64, bool, int8, int32, int64.
                          axes (list): One integer or List of integers, indicating the dimensions to be squeezed.
                          Axes range is :math:`[-rank(input), rank(input))`.
                          If axes is negative, :math:`axes=axes+rank(input)`.
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Variable: Output squeezed Tensor. Data type is same as input Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            # set batch size=None
            x = fluid.data(name='x', shape=[None, 5, 1, 10])
            y = layers.squeeze(input=x, axes=[2]) # y.shape=[None, 5, 10]

    """
    if in_dygraph_mode():
        out, _ = _C_ops.squeeze2(input, 'axes', axes)
        return out

    helper = LayerHelper("squeeze", **locals())
    check_variable_and_dtype(
        input, 'input',
        ['float16', 'float32', 'float64', 'bool', 'int8', 'int32', 'int64'],
        'squeeze')
    check_type(axes, 'axis/axes', (list, tuple), 'squeeze')
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="squeeze2",
        inputs={"X": input},
        attrs={"axes": axes},
        outputs={"Out": out,
                 "XShape": x_shape})

    return out


def unsqueeze(input, axes, name=None):
    """
    Insert single-dimensional entries to the shape of a Tensor. Takes one
    required argument axes, a list of dimensions that will be inserted.
    Dimension indices in axes are as seen in the output tensor.

    For example:

    .. code-block:: text

      Given a tensor such that tensor with shape [3, 4, 5],
      then Unsqueezed tensor with axes=[0, 4] has shape [1, 3, 4, 5, 1].

    Args:
        input (Variable): The input Tensor to be unsqueezed. Supported data type: float32, float64, bool, int8, int32, int64.
        axes (int|list|tuple|Variable): Indicates the dimensions to be inserted. The data type is ``int32`` . If ``axes`` is a list or tuple, the elements of it should be integers or Tensors with shape [1]. If ``axes`` is an Variable, it should be an 1-D Tensor .
        name (str|None): Name for this layer.

    Returns:
        Variable: Unsqueezed Tensor, with the same data type as input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[5, 10])
            y = fluid.layers.unsqueeze(input=x, axes=[1])

    """
    if in_dygraph_mode():
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, Variable):
            axes = axes.numpy().tolist()
        elif isinstance(axes, (list, tuple)):
            axes = [
                item.numpy().item(0) if isinstance(item, Variable) else item
                for item in axes
            ]
        out, _ = _C_ops.unsqueeze2(input, 'axes', axes)
        return out

    check_type(axes, 'axis/axes', (int, list, tuple, Variable), 'unsqueeze')
    check_variable_and_dtype(
        input, 'input',
        ['float16', 'float32', 'float64', 'bool', 'int8', 'int32', 'int64'],
        'unsqueeze')
    helper = LayerHelper("unsqueeze2", **locals())
    inputs = {"X": input}
    attrs = {}

    if isinstance(axes, int):
        axes = [axes]
    if isinstance(axes, Variable):
        axes.stop_gradient = True
        inputs["AxesTensor"] = axes
    elif isinstance(axes, (list, tuple)):
        if utils._contain_var(axes):
            inputs["AxesTensorList"] = utils._convert_to_tensor_list(axes)
        else:
            attrs["axes"] = axes

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="unsqueeze2",
        inputs=inputs,
        attrs=attrs,
        outputs={"Out": out,
                 "XShape": x_shape})

    return out


def lod_reset(x, y=None, target_lod=None):
    """
    Set LoD of :attr:`x` to a new one specified by :attr:`y` or
    :attr:`target_lod`. When :attr:`y` provided, :attr:`y.lod` would be
    considered as target LoD first, otherwise :attr:`y.data` would be
    considered as target LoD. If :attr:`y` is not provided, target LoD should
    be specified by :attr:`target_lod`. If target LoD is specified by
    :attr:`y.data` or :attr:`target_lod`, only one level LoD is supported.

    .. code-block:: text

        * Example 1:

            Given a 1-level LoDTensor x:
                x.lod =  [[ 2,           3,                   1 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            target_lod: [4, 2]

            then we get a 1-level LoDTensor:
                out.lod =  [[4,                          2]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 2:

            Given a 1-level LoDTensor x:
                x.lod =  [[2,            3,                   1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a Tensor:
                y.data = [[2, 4]]
                y.dims = [1, 3]

            then we get a 1-level LoDTensor:
                out.lod =  [[2,            4]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 3:

            Given a 1-level LoDTensor x:
                x.lod =  [[2,            3,                   1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a 2-level LoDTensor:
                y.lod =  [[2, 2], [2, 2, 1, 1]]
                y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
                y.dims = [6, 1]

            then we get a 2-level LoDTensor:
                out.lod =  [[2, 2], [2, 2, 1, 1]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a Tensor or LoDTensor. 
                      The data type should be int32, int64, float32 or float64.
        y (Variable, optional): If provided, output's LoD would be derived from :attr:`y`. 
                                If y's lod level>0, the data type can be any type. 
                                If y's lod level=0, the data type should be int32.
        target_lod (list|tuple, optional): One level LoD which should be considered
                                      as target LoD when :attr:`y` not provided.

    Returns:
        Variable: Output variable with LoD specified by this layer.

    Raises:
        ValueError: If :attr:`y` and :attr:`target_lod` are both None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[10])
            y = fluid.layers.data(name='y', shape=[10, 20], lod_level=2)
            out = fluid.layers.lod_reset(x=x, y=y)
    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'lod_reset')
    helper = LayerHelper("lod_reset", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    if y is not None:
        check_type(y, 'y', (Variable), 'lod_reset')
        #TODO: check y.lod_level = 0 dtype
        helper.append_op(
            type="lod_reset", inputs={'X': x,
                                      'Y': y}, outputs={'Out': out})
    elif target_lod is not None:
        helper.append_op(
            type="lod_reset",
            inputs={'X': x},
            attrs={'target_lod': target_lod},
            outputs={'Out': out})
    else:
        raise ValueError("y and target_lod should not be both none.")
    return out


def lod_append(x, level):
    """
    Append level to LoD of :attr:`x`.

    .. code-block:: text

        * Example 1:

            given a 1-level LoDTensor x:
                x.lod =  [[ 2,           3,                   1 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            level: [1, 1, 1, 1, 1, 1, 1]

            then we get a 2-level LoDTensor:
                x.lod =  [[ 2, 3, 1 ], [1, 1, 1, 1, 1, 1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a tensor or LoDTensor. 
                      The data type should be int32, int64, float32 or float64.
        level (list|tuple|Variable, optional): The LoD level to be appended into LoD of x. 
                                               If level is variable and its lod level>0, the data type can be any type.
                                               If level is variable and its lod level=0, the data type should be int32.
    Returns:
        Variable: Output variable with new LoD level.

    Raises:
        ValueError: If :attr:`y` is None or and :attr:`level` is not Iterator.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[6, 10], lod_level=1)
            out = fluid.layers.lod_append(x, [1,1,1,1,1,1])
    """
    from collections import Iterable
    if x is None:
        raise ValueError("Input(x) can't be None.")
    if (not isinstance(level, Iterable)) and (not isinstance(level, Variable)):
        raise ValueError("Input(level) must be list, tuple or Variable.")

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'lod_append')

    helper = LayerHelper("lod_append", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {'X': x}
    attrs = {'append': True}

    if isinstance(level, Variable):
        inputs['Y'] = level
        #TODO: check y.lod_level = 0 dtype
    else:
        attrs['target_lod'] = level
    helper.append_op(
        type="lod_reset", inputs=inputs, attrs=attrs, outputs={'Out': out})
    return out


def lrn(input, n=5, k=1.0, alpha=1e-4, beta=0.75, name=None,
        data_format='NCHW'):
    r"""
    :alias_main: paddle.nn.functional.lrn
	:alias: paddle.nn.functional.lrn,paddle.nn.functional.norm.lrn
	:old_api: paddle.fluid.layers.lrn

    This operator implements the Local Response Normalization Layer.
    This layer performs a type of "lateral inhibition" by normalizing over local input regions.
    For more information, please refer to `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    The formula is as follows:

    .. math::

        Output(i, x, y) = Input(i, x, y) / \\left(k + \\alpha \\sum\\limits^{\\min(C-1, i + n/2)}_{j = \\max(0, i - n/2)}(Input(j, x, y))^2\\right)^{\\beta}

    In the above equation:

    - :math:`n` : The number of channels to sum over.
    - :math:`k` : The offset (avoid being divided by 0).
    - :math:`\\alpha` : The scaling parameter.
    - :math:`\\beta` : The exponent parameter.


    Args:
        input (Variable): Input feature, 4D-Tensor with the shape of [N,C,H,W] or [N, H, W, C],
            where N is the batch size, C is the input channel, H is Height, W is weight. The data
            type is float32. The rank of this tensor must be 4, otherwise it will raise ValueError.
        n (int, optional): The number of channels to sum over. Default: 5
        k (float, optional): An offset, positive. Default: 1.0
        alpha (float, optional): The scaling parameter, positive. Default:1e-4
        beta (float, optional): The exponent, positive. Default:0.75
        name (str, optional): The default value is None. Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        Variable: A tensor variable storing the transformation result with the same shape and data type as input.


    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        data = fluid.data(
            name="data", shape=[None, 3, 112, 112], dtype="float32")
        lrn = fluid.layers.lrn(input=data)
        print(lrn.shape)  # [-1, 3, 112, 112]
        print(lrn.dtype)  # float32
    """
    helper = LayerHelper('lrn', **locals())
    check_variable_and_dtype(input, 'input', ['float32'], 'lrn')
    dtype = helper.input_dtype()
    input_shape = input.shape
    dims = len(input_shape)

    if dims != 4:
        raise ValueError(
            "Input's dimension size of Op(lrn) must be 4, but received %d." %
            (dims))
    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of Op(lrn) got wrong value: received " +
            data_format + " but only NCHW or NHWC supported.")

    mid_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    lrn_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="lrn",
        inputs={"X": input},
        outputs={
            "Out": lrn_out,
            "MidOut": mid_out,
        },
        attrs={
            "n": n,
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "data_format": data_format
        })

    return lrn_out


def pad(x, paddings, pad_value=0., name=None):
    r"""
    :alias_main: paddle.nn.functional.pad
	:alias: paddle.nn.functional.pad,paddle.nn.functional.common.pad
	:old_api: paddle.fluid.layers.pad

    This op will pad a tensor with a constant value given by :attr:`pad_value`, and the
    padded shape is specified by :attr:`paddings`.

    Specifically, the number of values padded before the elements of :attr:`x`
    in dimension :attr:`i` is indicated by :attr:`paddings[2*i]`, and the number
    of values padded after the elements of :attr:`x` in dimension :attr:`i` is
    indicated by :attr:`paddings[2*i+1]`.

    See below for an example.

    .. code-block:: text

        Given:
            x = [[1, 2], [3, 4]]

            paddings = [0, 1, 1, 2]

            pad_value = 0

        Return:
            out = [[0, 1, 2, 0, 0]
                   [0, 3, 4, 0, 0]
                   [0, 0, 0, 0, 0]]

    Args:
        x (Variable): Tensor, data type is float32.
        paddings (list): A list of integers. Its elements specify the padded
                         width before and after each dimension in turn.
                         The length of :attr:`paddings` must be equal to
                         :math:`rank(x) \\times 2`.
        pad_value (float): The constant value used to pad.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        The padded tensor, with the same data type and rank as :attr:`x`

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            # x is a rank 2 tensor variable
            import paddle.fluid as fluid
            x = fluid.data(name='data', shape=[300, 300], dtype='float32')
            out = fluid.layers.pad(x=x, paddings=[0, 1, 1, 2], pad_value=0.)
    """
    check_variable_and_dtype(x, 'x', [
        'float16', 'float32', 'float64', 'int32', 'int64', 'complex64',
        'complex128'
    ], "pad")

    helper = LayerHelper('pad', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='pad',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'paddings': paddings,
               'pad_value': float(pad_value)})
    return out


def pad_constant_like(x, y, pad_value=0., name=None):
    r"""
    Pad :attr:`y` with :attr:`pad_value`, the number of values padded to
    the edges of each axis is specified by the difference of the shape
    of :attr:`x` and :attr:`y` . ((0, shape_x_0 - shape_y_0), ... (0, shape_x_n - shape_y_n))
    specify padding widths for each axis. The input should be a k-D tensor(k > 0 and k < 7).

    See below for an example.

    .. code-block:: text

        Given:
            X = [[[[ 0,  1,  2],
                   [ 3,  4,  5]],
                  [[ 6,  7,  8],
                   [ 9, 10, 11]],
                  [[12, 13, 14],
                   [15, 16, 17]]],
                 [[[18, 19, 20],
                   [21, 22, 23]],
                  [[24, 25, 26],
                   [27, 28, 29]],
                  [[30, 31, 32],
                   [33, 34, 35]]]]

            X.shape = (2, 3, 2, 3)

            Y = [[[[35, 36, 37]],
                  [[38, 39, 40]],
                  [[41, 42, 43]]]]

            Y.shape = (1, 3, 1, 3)

        And
            pad_value = 0.

        Return:
            Out = [[[[35, 36, 37],
                     [ 0,  0,  0]],
                    [[38, 39, 40],
                     [ 0,  0,  0]],
                    [[41, 42, 43],
                     [ 0,  0,  0]]],
                   [[[ 0,  0,  0],
                     [ 0,  0,  0]],
                    [[ 0,  0,  0],
                     [ 0,  0,  0]],
                    [[ 0,  0,  0],
                     [ 0,  0,  0]]]]

            Out.shape = [2, 3, 2, 3]


    Args:
        x (Variable): Tensor, its shape specifies the shape of output.
        y (Variable): Tensor, its rank is the same with :attr:`x`, and for each dimension :math:`i` ,
                      :math:`y\_shape[i] <= x\_shape[i]` . The data type can be float32 or float64.
        pad_value (float): The constant value used to pad.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        The padded tensor, with the same shape as :attr:`x` and the same data type as :attr:`y`

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            # x is a rank 4 tensor variable, x.shape = (2, 3, 2, 3)
            # y is a rank 4 tensor variable, y.shape = (1, 3, 1, 3)
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2,3,2,3], dtype='float32')
            y = fluid.data(name='y', shape=[1,3,1,3], dtype='float32')
            out = fluid.layers.pad_constant_like(x=x, y=y, pad_value=0.)
            # out is a rank 4 tensor variable, and out.shape = [2, 3 ,2 , 3]
    """
    check_type(x, 'x', (Variable), 'pad_constant_like')
    check_variable_and_dtype(y, 'y', ['float32', 'float64', 'int32', 'int64'],
                             "pad_constant_like")

    helper = LayerHelper('pad_constant_like', **locals())
    dtype = helper.input_dtype(input_param_name='y')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='pad_constant_like',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'pad_value': float(pad_value)})
    return out


def label_smooth(label,
                 prior_dist=None,
                 epsilon=0.1,
                 dtype="float32",
                 name=None):
    r"""
    :alias_main: paddle.nn.functional.label_smooth
	:alias: paddle.nn.functional.label_smooth,paddle.nn.functional.common.label_smooth
	:old_api: paddle.fluid.layers.label_smooth

    Label smoothing is a mechanism to regularize the classifier layer and is called
    label-smoothing regularization (LSR).

    Label smoothing is proposed to encourage the model to be less confident,
    since optimizing the log-likelihood of the correct label directly may
    cause overfitting and reduce the ability of the model to adapt. Label
    smoothing replaces the ground-truth label :math:`y` with the weighted sum
    of itself and some fixed distribution :math:`\mu`. For class :math:`k`,
    i.e.

    .. math::

        \\tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k,

    where :math:`1 - \epsilon` and :math:`\epsilon` are the weights
    respectively, and :math:`\\tilde{y}_k` is the smoothed label. Usually
    uniform distribution is used for :math:`\mu`.

    See more details about label smoothing in https://arxiv.org/abs/1512.00567.

    Parameters:
        label(Variable): The input variable containing the label data. The
                        label data should use one-hot representation. It's
                        a multidimensional tensor with a shape of
                        :math:`[N_1, ..., Depth]`, where Depth is class number. The dtype can be "float32" and "float64".
        prior_dist(Variable, optional): The prior distribution to be used to smooth
                        labels. If not provided, an uniform distribution
                        is used. It's a multidimensional tensor with a shape of
                        :math:`[1, class\_num]` . The default value is None.
        epsilon(float, optional): The weight used to mix up the original ground-truth
                        distribution and the fixed distribution. The default value is
                        0.1.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type can be set
                        as 'float32', 'float64'. The default value is 'float32'.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        Variable: The tensor variable containing the smoothed labels.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            label = layers.data(name="label", shape=[1], dtype="int32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = layers.label_smooth(
                label=one_hot_label, epsilon=0.1, dtype="float32")
    """
    if epsilon > 1. or epsilon < 0.:
        raise ValueError("The value of epsilon must be between 0 and 1.")

    if in_dygraph_mode():
        return _C_ops.label_smooth(label, prior_dist, 'epsilon', float(epsilon))

    check_variable_and_dtype(label, 'label', ['float32', 'float64'],
                             'label_smooth')

    helper = LayerHelper("label_smooth", **locals())
    label.stop_gradient = True
    smooth_label = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="label_smooth",
        inputs={"X": label,
                "PriorDist": prior_dist} if prior_dist else {"X": label},
        outputs={"Out": smooth_label},
        attrs={"epsilon": float(epsilon)})
    return smooth_label


@templatedoc()
def roi_pool(input,
             rois,
             pooled_height=1,
             pooled_width=1,
             spatial_scale=1.0,
             rois_num=None,
             name=None):
    """

    This operator implements the roi_pooling layer.
    Region of interest pooling (also known as RoI pooling) is to perform max pooling on inputs of nonuniform sizes to obtain fixed-size feature maps (e.g. 7*7).

    The operator has three steps:

        1. Dividing each region proposal into equal-sized sections with the pooled_width and pooled_height;
        2. Finding the largest value in each section;
        3. Copying these max values to the output buffer.

    For more information, please refer to https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn

    Args:
        input (Variable): Input feature, 4D-Tensor with the shape of [N,C,H,W], where N is the batch size, C is the input channel, H is Height, W is weight. The data type is float32 or float64.
        rois (Variable): ROIs (Regions of Interest) to pool over. 2D-LoDTensor with the shape of [num_rois,4], the lod level is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is the top left coordinates, and (x2, y2) is the bottom right coordinates.
        pooled_height (int, optional): The pooled output height, data type is int32. Default: 1
        pooled_width (int, optional): The pooled output height, data type is int32. Default: 1
        spatial_scale (float, optional): Multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling. Default: 1.0
        rois_num (Tensor): The number of RoIs in each image. Default: None
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.


    Returns:
        Variable: The pooled feature, 4D-Tensor with the shape of [num_rois, C, pooled_height, pooled_width].


    Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle
        paddle.enable_static()

        DATATYPE='float32'

        place = fluid.CPUPlace()
        #place = fluid.CUDAPlace(0)

        input_data = np.array([i for i in range(1,17)]).reshape(1,1,4,4).astype(DATATYPE)
        roi_data =fluid.create_lod_tensor(np.array([[1., 1., 2., 2.], [1.5, 1.5, 3., 3.]]).astype(DATATYPE),[[2]], place)
        rois_num_data = np.array([2]).astype('int32')

        x = fluid.data(name='input', shape=[None,1,4,4], dtype=DATATYPE)
        rois = fluid.data(name='roi', shape=[None,4], dtype=DATATYPE)
        rois_num = fluid.data(name='rois_num', shape=[None], dtype='int32')

        pool_out = fluid.layers.roi_pool(
                input=x,
                rois=rois,
                pooled_height=1,
                pooled_width=1,
                spatial_scale=1.0,
                rois_num=rois_num)

        exe = fluid.Executor(place)
        out, = exe.run(feed={'input':input_data ,'roi':roi_data, 'rois_num': rois_num_data}, fetch_list=[pool_out.name])
        print(out)   #array([[[[11.]]], [[[16.]]]], dtype=float32)
        print(np.array(out).shape)  # (2, 1, 1, 1)
    """
    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        pool_out, argmaxes = _C_ops.roi_pool(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale)
        return pool_out, argmaxes

    check_variable_and_dtype(input, 'input', ['float32'], 'roi_pool')
    check_variable_and_dtype(rois, 'rois', ['float32'], 'roi_pool')
    helper = LayerHelper('roi_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    argmaxes = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {
        "X": input,
        "ROIs": rois,
    }
    if rois_num is not None:
        inputs['RoisNum'] = rois_num
    helper.append_op(
        type="roi_pool",
        inputs=inputs,
        outputs={"Out": pool_out,
                 "Argmax": argmaxes},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale
        })
    return pool_out


@templatedoc()
def roi_align(input,
              rois,
              pooled_height=1,
              pooled_width=1,
              spatial_scale=1.0,
              sampling_ratio=-1,
              rois_num=None,
              name=None):
    """

    ${comment}

    Args:
        input (Variable): ${x_comment}
        rois (Variable): ROIs (Regions of Interest) to pool over.It should be
            a 2-D LoDTensor of shape (num_rois, 4), the lod level is 1. The
            data type is float32 or float64. Given as [[x1, y1, x2, y2], ...],
            (x1, y1) is the top left coordinates, and (x2, y2) is the bottom
            right coordinates.
        pooled_height (int32, optional): ${pooled_height_comment} Default: 1
        pooled_width (int32, optional): ${pooled_width_comment} Default: 1
        spatial_scale (float32, optional): ${spatial_scale_comment} Default: 1.0
        sampling_ratio(int32, optional): ${sampling_ratio_comment} Default: -1
        rois_num (Tensor): The number of RoIs in each image. Default: None
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Variable:

        Output: ${out_comment}.


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()

            x = fluid.data(
                name='data', shape=[None, 256, 32, 32], dtype='float32')
            rois = fluid.data(
                name='rois', shape=[None, 4], dtype='float32')
            rois_num = fluid.data(name='rois_num', shape=[None], dtype='int32')
            align_out = fluid.layers.roi_align(input=x,
                                               rois=rois,
                                               pooled_height=7,
                                               pooled_width=7,
                                               spatial_scale=0.5,
                                               sampling_ratio=-1,
                                               rois_num=rois_num)
    """
    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        align_out = _C_ops.roi_align(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale,
            "sampling_ratio", sampling_ratio)
        return align_out

    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'roi_align')
    check_variable_and_dtype(rois, 'rois', ['float32', 'float64'], 'roi_align')
    helper = LayerHelper('roi_align', **locals())
    dtype = helper.input_dtype()
    align_out = helper.create_variable_for_type_inference(dtype)
    inputs = {
        "X": input,
        "ROIs": rois,
    }
    if rois_num is not None:
        inputs['RoisNum'] = rois_num
    helper.append_op(
        type="roi_align",
        inputs=inputs,
        outputs={"Out": align_out},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale,
            "sampling_ratio": sampling_ratio
        })
    return align_out


def dice_loss(input, label, epsilon=0.00001, name=None):
    r"""

    Dice loss for comparing the similarity between the input predictions and the label.
    This implementation is for binary classification, where the input is sigmoid
    predictions of each pixel, usually used for segmentation task. The dice loss can
    be defined as the following equation:

    .. math::

        dice\_loss &= 1 - \frac{2 * intersection\_area}{total\_area} \\
                  &= \frac{(total\_area - intersection\_area) - intersection\_area}{total\_area} \\
                  &= \frac{(union\_area - intersection\_area)}{total\_area}


    Parameters:
        input (Tensor): Tensor, rank>=2, shape is :math:`[N_1, N_2, ..., N_k, D]`, where :math:`N_1` is
                          the batch_size, :math:`D` is the number of categories. It is usually the output
                          predictions of sigmoid activation. The data type can be float32 or float64.
        label (Tensor): Tensor, the groud truth with the same rank as input, shape is :math:`[N_1, N_2, ..., N_k, 1]`.
                          where :math:`N_1` is the batch_size. The data type can be int32 or int64.
        epsilon (float): The epsilon will be added to the numerator and denominator.
                         If both input and label are empty, it makes sure dice is 1.
                         Default: 0.00001
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor, which shape is [1], data type is the same as `input` .

    Example:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            x = paddle.randn((3,224,224,2))
            label = paddle.randint(high=2, shape=(3,224,224,1))
            predictions = F.softmax(x)
            loss = F.dice_loss(input=predictions, label=label)
    """
    assert input.dtype in (paddle.float32, paddle.float64)
    assert label.dtype in (paddle.int32, paddle.int64)
    assert len(input.shape) >= 2, \
        "The rank of input should be greater than or equal to 2."
    assert len(input.shape) == len(label.shape), (
        "The rank of input and label should be equal, "
        "but received input: %d, label: %d." %
        (len(input.shape), len(label.shape)))
    assert label.shape[-1] == 1, ("The last dimension of label should be 1, "
                                  "but received %d." % label.shape[-1])
    assert input.shape[:-1] == label.shape[:-1], (
        "All dimensions should be equal except the last one.")
    assert input.numel() > 0 and label.numel() > 0, \
        "Any dimension of input and label cannot be equal to 0."

    label = squeeze(label, [-1])
    label = paddle.nn.functional.one_hot(label, input.shape[-1])
    reduce_dim = list(range(1, len(input.shape)))
    inse = reduce_sum(input * label, dim=reduce_dim)
    dice_denominator = reduce_sum(
        input, dim=reduce_dim) + reduce_sum(
            label, dim=reduce_dim)
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    return reduce_mean(dice_score)


def image_resize(input,
                 out_shape=None,
                 scale=None,
                 name=None,
                 resample='BILINEAR',
                 actual_shape=None,
                 align_corners=True,
                 align_mode=1,
                 data_format='NCHW'):
    """

    This op resizes a batch of images.

    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or a 4-D Tensor of the shape (num_batches, channels, in_h, in_w)
    or (num_batches, in_h, in_w, channels), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    and the resizing only applies on the three dimensions(depth, height and width).

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the
    future and only use :attr:`out_shape` instead.

    Supporting resample methods:
        'LINEAR' : Linear interpolation 

        'BILINEAR' : Bilinear interpolation

        'TRILINEAR' : Trilinear interpolation

        'NEAREST' : Nearest neighbor interpolation
        
        'BICUBIC' : Bicubic interpolation
    
    Linear interpolation is the method of using a line connecting two known quantities 
    to determine the value of an unknown quantity between the two known quantities.
    
    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.
    The linear interpolation is performed on three directions.
    
    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:

            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)

            else:

              scale_factor = float(in_size/out_size)


        Nearest neighbor interpolation:

          if:
              align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

        linear interpolation:

          if:
              align_corners = False , align_mode = 0

              input : (N,C,W_in)
              output: (N,C,W_out) where:

              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,W_in)
              output: (N,C,H_out,W_out) where:

              W_out = W_{in} * scale_{factor}

        Bilinear interpolation:

          if:
              align_corners = False , align_mode = 0

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Trilinear interpolation:

          if:
              align_corners = False , align_mode = 0

              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5


          else:

              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = D_{in} * scale_{factor}
       
        Trilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}
        

    For details of linear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Linear_interpolation.
    
    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.
    
    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.
    
    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.
    
    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation

    Parameters:
        input (Variable): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape (list|tuple|Variable|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w) 
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor. 
             Default: None. If a list, each element can be an integer or a Tensor Variable of shape: [1].
             If a Tensor Variable, its dimensions size should be a 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set.
             And :attr:`out_shape` has a higher priority than :attr:`scale`.
             Default: None.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
        resample(str): The resample method. It supports 'LINEAR', 'BICUBIC', 'BILINEAR', 'TRILINEAR'
                       and 'NEAREST' currently. Default: 'BILINEAR'
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output
                                shape dynamically, because :attr:`actual_shape`
                                will be deprecated. When using actual_shape to
                                specify output shape, one of :attr:`out_shape`
                                and :attr:`scale` should also be set, otherwise
                                errors would be occurred in graph constructing stage.
                                Default: None
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.
                               Default: True
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the fomula in the 
                            the example code above, it can be \'0\' for src_idx = scale*(dst_indx+0.5)-0.5 , 
                            can be \'1\' for src_idx = scale*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.

    Returns:
        A 3-D Tensor of the shape (num_batches, channels, out_w) or (num_batches, out_w, channels),
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).

    Raises:
        TypeError: out_shape should be a list or tuple or Variable.
        TypeError: actual_shape should either be Variable or None.
        ValueError: The 'resample' of image_resize can only be 'LINEAR', 'BILINEAR',
                    'TRILINEAR', 'BICUBIC' or 'NEAREST' currently.
        ValueError: 'LINEAR' only support 3-D tensor.
        ValueError: 'BICUBIC', 'BILINEAR' and 'NEAREST' only support 4-D tensor.
        ValueError: 'TRILINEAR' only support 5-D tensor.
        ValueError: One of out_shape and scale must not be None.
        ValueError: out_shape length should be 1 for input 3-D tensor.
        ValueError: out_shape length should be 2 for input 4-D tensor.
        ValueError: out_shape length should be 3 for input 5-D tensor.
        ValueError: scale should be greater than zero.
        TypeError: align_corners should be a bool value
        ValueError: align_mode can only be '0' or '1'
        ValueError: data_format can only be 'NCW', 'NWC', 'NCHW', 'NHWC', 'NCDHW' or 'NDHWC'.

    Examples:
        .. code-block:: python

	    #declarative mode
	    import paddle
	    import paddle.fluid as fluid
	    import numpy as np
	    paddle.enable_static()
	    input = fluid.data(name="input", shape=[None,3,6,10])

	    #1
	    output = fluid.layers.image_resize(input=input,out_shape=[12,12])

	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = fluid.layers.image_resize(input=input,out_shape=[12,dim1])

	    #3
	    #x = np.array([3,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = fluid.layers.image_resize(input=input,out_shape=shape_tensor)

	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = fluid.layers.image_resize(input=input,scale=scale_tensor)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    input_data = np.random.rand(2,3,6,10).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)

	    print(output_data[0].shape)

	    #1
	    # (2, 3, 12, 12)
	    #2
	    # (2, 3, 12, 2)
	    #3
	    # (2, 3, 3, 12)
	    #4
	    # (2, 3, 3, 5)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.image_resize(input=input, out_shape=[12,12])
    		print(output.shape)

		# [2L, 3L, 12L, 12L]

    """
    resample_methods = {
        'LINEAR': 'linear',
        'BILINEAR': 'bilinear',
        'TRILINEAR': 'trilinear',
        'NEAREST': 'nearest',
        'LINEAR': 'linear',
    }
    resample = resample.upper()
    if resample not in resample_methods:
        raise ValueError(
            "The 'resample' of image_resize can only be 'LINEAR', 'BILINEAR', 'TRILINEAR' "
            "or 'NEAREST' currently.")
    resample_type = resample_methods[resample]

    if resample == 'LINEAR' and len(input.shape) != 3:
        raise ValueError("'LINER only support 3-D tensor.")
    elif resample in ['BILINEAR', 'NEAREST'] and len(input.shape) != 4:
        raise ValueError("'BILINEAR' and 'NEAREST' only support 4-D tensor.")
    elif resample == 'TRILINEAR' and len(input.shape) != 5:
        raise ValueError("'TRILINEAR'only support 5-D tensor.")

    if not isinstance(align_corners, bool):
        raise TypeError("Attr align_corners should be a bool value")
    if align_mode != 0 and align_mode != 1:
        raise ValueError("align_mode can only be 0 or 1")

    if out_shape is None and scale is None:
        raise ValueError("One of out_shape and scale must not be None.")
    helper = LayerHelper('{}_interp'.format(resample_type), **locals())
    dtype = helper.input_dtype()

    if len(input.shape) == 3 and data_format not in ['NCW', 'NWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCW` or `NWC` supported for 3-D input.")
    elif len(input.shape) == 4 and data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCHW` or `NHWC` supported for 4-D input.")
    elif len(input.shape) == 5 and data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCDHW` or `NDHWC` supported for 5-D input.")

    def _is_list_or_turple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if data_format == 'NCHW' or data_format == 'NCDHW' or data_format == 'NCW':
        data_layout = 'NCHW'
    if data_format == 'NHWC' or data_format == 'NDHWC' or data_format == 'NWC':
        data_layout = 'NHWC'

    inputs = {"X": input}
    attrs = {
        "out_d": -1,
        "out_h": -1,
        "out_w": -1,
        "interp_method": resample_type,
        "align_corners": align_corners,
        "align_mode": align_mode,
        "data_layout": data_layout
    }

    if out_shape is not None:
        if isinstance(out_shape, Variable):
            out_shape.stop_gradient = True
            inputs['OutSize'] = out_shape
        else:
            if not (_is_list_or_turple_(out_shape)):
                raise TypeError(
                    "out_shape should be a list or tuple or Variable.")
            # Validate the shape
            contain_var = False
            for dim_idx, dim_size in enumerate(out_shape):
                if isinstance(dim_size, Variable):
                    contain_var = True
                    continue
                assert dim_size > 0, (
                    "Each dimension size given in out_shape must be greater than 0."
                )

            if contain_var:
                new_size_tensor = []
                size_list = []
                for dim in out_shape:
                    if isinstance(dim, Variable):
                        dim.stop_gradient = True
                        new_size_tensor.append(dim)
                        size_list.append(-1)
                    else:
                        assert (isinstance(dim, int))
                        temp_out = helper.create_variable_for_type_inference(
                            'int32')
                        fill_constant(
                            [1], 'int32', dim, force_cpu=True, out=temp_out)
                        new_size_tensor.append(temp_out)
                        size_list.append(dim)
                inputs['SizeTensor'] = new_size_tensor

            if len(input.shape) == 3:
                if len(out_shape) != 1:
                    raise ValueError("out_shape length should be 1 for "
                                     "input 3-D tensor.")
                if contain_var:
                    attrs['out_w'] = size_list[0]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_w'] = out_shape[0]
            elif len(input.shape) == 4:
                if len(out_shape) != 2:
                    raise ValueError("out_shape length should be 2 for "
                                     "input 4-D tensor.")
                if contain_var:
                    attrs['out_h'] = size_list[0]
                    attrs['out_w'] = size_list[1]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_h'] = out_shape[0]
                    attrs['out_w'] = out_shape[1]
            if len(input.shape) == 5:
                if len(out_shape) != 3:
                    raise ValueError("out_shape length should be 3 for "
                                     "input 5-D tensor.")
                if contain_var:
                    attrs['out_d'] = size_list[0]
                    attrs['out_h'] = size_list[1]
                    attrs['out_w'] = size_list[2]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_d'] = out_shape[0]
                    attrs['out_h'] = out_shape[1]
                    attrs['out_w'] = out_shape[2]

    else:
        if isinstance(scale, Variable):
            scale.stop_gradient = True
            inputs["Scale"] = scale
        elif isinstance(scale, float) or isinstance(scale, int):
            if scale <= 0:
                raise ValueError("Attr(scale) should be greater than zero.")
            attrs['scale'] = float(scale)
        else:
            raise TypeError(
                "Attr(scale)'s type should be float, int or Variable.")

    if isinstance(actual_shape, Variable):
        warnings.warn(
            "actual_shape will be deprecated, it is recommended to use "
            "out_shape instead of actual_shape to specify output shape dynamically."
        )
        actual_shape.stop_gradient = True
        inputs["OutSize"] = actual_shape
    elif actual_shape is not None:
        raise TypeError("actual_shape should either be Variable or None.")
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='{}_interp'.format(resample_type),
        inputs=inputs,
        outputs={"Out": out},
        attrs=attrs)
    return out


@templatedoc(op_type="linear_interp")
def resize_linear(input,
                  out_shape=None,
                  scale=None,
                  name=None,
                  actual_shape=None,
                  align_corners=True,
                  align_mode=1,
                  data_format='NCW'):
    """
    This op resizes the input by performing linear interpolation based on given
    output shape which specified by actual_shape, out_shape and scale
    in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in 
    the future and only use :attr:`out_shape` instead.

    Align_corners and align_mode are optional parameters,the calculation 
    method of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)

        Linear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,W_in)
              output: (N,C,W_out) where:
              
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,W_in)
              output: (N,C,W_out) where:
              W_out = W_{in} * scale_{factor}

    Parameters:
        input(Variable): 3-D Tensor(NCW), its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): Output shape of resize linear
            layer, the shape is (out_w,). Default: None. If a list, each 
            element can be an integer or a Tensor Variable with shape: [1]. If a 
            Tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set. 
             And :attr:`out_shape` has a higher priority than :attr:`scale`. 
             Default: None.
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occurred in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        align_mode(bool): ${align_mode_comment}
        data_format (str, optional): Specify the data format of the input, and the data format of the output 
            will be consistent with that of the input. An optional string from: `"NCW"`, `"NWC"`.
            The default is `"NCW"`. When it is `"NCW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_width]`.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  
            For more information, please refer to :ref:`api_guide_Name`

    Returns:
	Variable: 3-D tensor(NCW or NWC).
    
    Examples:
        .. code-block:: python
	
	    #declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[None,3,100])

	    output = fluid.layers.resize_linear(input=input,out_shape=[50,])

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.random.rand(1,3,100).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)
 
	    print(output_data[0].shape)

	    # (1, 3, 50)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.resize_linear(input=input, out_shape=[50,])
    		print(output.shape)

		# [1L, 3L, 50L]

    """

    return image_resize(input, out_shape, scale, name, 'LINEAR', actual_shape,
                        align_corners, align_mode, data_format)


@templatedoc(op_type="bilinear_interp")
def resize_bilinear(input,
                    out_shape=None,
                    scale=None,
                    name=None,
                    actual_shape=None,
                    align_corners=True,
                    align_mode=1,
                    data_format='NCHW'):
    """

    This op resizes the input by performing bilinear interpolation based on given
    output shape which specified by actual_shape, out_shape and scale
    in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in
    the future and only use :attr:`out_shape` instead.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation

    Align_corners and align_mode are optional parameters,the calculation
    method of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:

            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)

            else:

              scale_factor = float(in_size/out_size)

        Bilinear interpolation:

          if:
              align_corners = False , align_mode = 0

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    Parameters:
        input(Variable): 4-D Tensor(NCHW), its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): Output shape of resize bilinear
            layer, the shape is (out_h, out_w).Default: None. If a list, each
            element can be an integer or a Tensor Variable with shape: [1]. If a
            Tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set.
             And :attr:`out_shape` has a higher priority than :attr:`scale`.
             Default: None.
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output
                                shape dynamically, because :attr:`actual_shape`
                                will be deprecated. When using actual_shape to
                                specify output shape, one of :attr:`out_shape`
                                and :attr:`scale` should also be set, otherwise
                                errors would be occurred in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        align_mode(bool): ${align_mode_comment}
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
	Variable: 4-D tensor(NCHW or NHWC).

    Examples:
        .. code-block:: python

	    #declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    import paddle
	    paddle.enable_static()
	    input = fluid.data(name="input", shape=[None,3,6,10])

	    #1
	    output = fluid.layers.resize_bilinear(input=input,out_shape=[12,12])

	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = fluid.layers.resize_bilinear(input=input,out_shape=[12,dim1])

	    #3
	    #x = np.array([3,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = fluid.layers.resize_bilinear(input=input,out_shape=shape_tensor)

	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = fluid.layers.resize_bilinear(input=input,scale=scale_tensor)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    input_data = np.random.rand(2,3,6,10).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)

	    print(output_data[0].shape)

	    #1
	    # (2, 3, 12, 12)
	    #2
	    # (2, 3, 12, 2)
	    #3
	    # (2, 3, 3, 12)
	    #4
	    # (2, 3, 3, 5)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.resize_bilinear(input=input, out_shape=[12,12])
    		print(output.shape)

		# [2L, 3L, 12L, 12L]

    """

    return image_resize(input, out_shape, scale, name, 'BILINEAR', actual_shape,
                        align_corners, align_mode, data_format)


@templatedoc(op_type="trilinear_interp")
def resize_trilinear(input,
                     out_shape=None,
                     scale=None,
                     name=None,
                     actual_shape=None,
                     align_corners=True,
                     align_mode=1,
                     data_format='NCDHW'):
    """

    This op resizes the input by performing trilinear interpolation based on given
    output shape which specified by actual_shape, out_shape and scale
    in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated
    in the future and only use :attr:`out_shape` instead.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.
    The linear interpolation is performed on three directions.

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation

    Align_corners and align_mode are optional parameters,the calculation
    method of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:

            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)

            else:

              scale_factor = float(in_size/out_size)

        Bilinear interpolation:

          if:

              align_corners = False , align_mode = 0

              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    Parameters:
        input(${x_type}): 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): The output shape of resized tensor, the shape is (out_d, out_h, out_w). Default: None. Every element should be an integer or a Tensor Variable with shape: [1] if it is a list. If it is a Tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input depth, height or width.
             At least one of :attr:`out_shape` or :attr:`scale` must be set.
             And :attr:`out_shape` has a higher priority than :attr:`scale`.
             Default: None.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output
                                shape dynamically, because :attr:`actual_shape`
                                will be deprecated. When using actual_shape to
                                specify output shape, one of :attr:`out_shape`
                                and :attr:`scale` should also be set, otherwise
                                errors would be occurred in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        align_mode(bool): ${align_mode_comment}
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCDHW"`, `"NDHWC"`.
            The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_depth, input_height, input_width]`.

    Returns:
        Variable: A 5-D Tensor(NCDHW or NDHWC)

    Examples:
        .. code-block:: python

	    #declarative mode
	    import paddle.fluid as fluid
	    import paddle
	    import numpy as np
	    paddle.enable_static()
	    input = fluid.data(name="input", shape=[None,3,6,8,10])

	    #1
	    output = fluid.layers.resize_trilinear(input=input,out_shape=[12,12,12])

	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = fluid.layers.resize_trilinear(input=input,out_shape=[12,dim1,4])

	    #3
	    #x = np.array([3,12,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[3], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = fluid.layers.resize_trilinear(input=input,out_shape=shape_tensor)

	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = fluid.layers.resize_trilinear(input=input,scale=scale_tensor)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    input_data = np.random.rand(2,3,6,8,10).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)

	    print(output_data[0].shape)

	    #1
	    # (2, 3, 12, 12, 12)
	    #2
	    # (2, 3, 12, 2, 4)
	    #3
	    # (2, 3, 3, 12, 12)
	    #4
	    # (2, 3, 3, 4, 5)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.resize_trilinear(input=input, out_shape=[12,12,12])
    		print(output.shape)

		# [2L, 3L, 12L, 12L, 12L]



    """

    return image_resize(input, out_shape, scale, name, 'TRILINEAR',
                        actual_shape, align_corners, align_mode, data_format)


@templatedoc(op_type="nearest_interp")
def resize_nearest(input,
                   out_shape=None,
                   scale=None,
                   name=None,
                   actual_shape=None,
                   align_corners=True,
                   data_format='NCHW'):
    """

    This op resizes the input by performing nearest neighbor interpolation in both the
    height direction and the width direction based on given output shape
    which is specified by actual_shape, out_shape and scale in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the
    future and only use :attr:`out_shape` instead.

    Example:

    .. code-block:: text

        For scale:

            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)

            else:

              scale_factor = float(in_size/out_size)

        Nearest neighbor interpolation:

          if:
              align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = floor(H_{in} * scale_{factor})
              W_out = floor(W_{in} * scale_{factor})

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})


    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation

    Parameters:
        input(${x_type}): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): The output shape of resized tensor, the shape is (out_h, out_w). Default: None. Every element should be an integer or a tensor Variable with shape: [1] if it is a list. If it is a tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set.
             And :attr:`out_shape` has a higher priority than :attr:`scale`.
             Default: None.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
	actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output
                                shape dynamically, because :attr:`actual_shape`
                                will be deprecated. When using actual_shape to
                                specify output shape, one of :attr:`out_shape`
                                and :attr:`scale` should also be set, otherwise
                                errors would be occurred in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
	Variable: 4-D tensor(NCHW or NHWC).

    Examples:
        .. code-block:: python

	    #declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    import paddle
	    paddle.enable_static()

	    input = fluid.data(name="input", shape=[None,3,6,10])

	    #1
	    output = fluid.layers.resize_nearest(input=input,out_shape=[12,12])

	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = fluid.layers.resize_nearest(input=input,out_shape=[12,dim1])

	    #3
	    #x = np.array([3,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = fluid.layers.resize_nearest(input=input,out_shape=shape_tensor)

	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = fluid.layers.resize_nearest(input=input,scale=scale_tensor)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    input_data = np.random.rand(2,3,6,10).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)

	    print(output_data[0].shape)

	    #1
	    # (2, 3, 12, 12)
	    #2
	    # (2, 3, 12, 2)
	    #3
	    # (2, 3, 3, 12)
	    #4
	    # (2, 3, 3, 5)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.resize_nearest(input=input, out_shape=[12,12])
    		print(output.shape)

		# [2L, 3L, 12L, 12L]



    """

    return image_resize(
        input,
        out_shape,
        scale,
        name,
        'NEAREST',
        actual_shape,
        align_corners,
        align_mode=1,
        data_format=data_format)


def image_resize_short(input, out_short_len, resample='BILINEAR'):
    """
    This op resizes a batch of images. The short edge of input images will be
    resized to the given 'out_short_len'. The long edge of input images
    will be resized proportionately to make images' length-width ratio
    constant.

    Parameters:
        input (Variable): 4-D tensor(NCHW), The input tensor of image resize layer.
        out_short_len(int): The length of output images' short edge.
        resample (str): resample method, default: BILINEAR.

    Returns:
        Variable: 4-D tensor(NCHW).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(name="input", shape=[None,3,6,9], dtype="float32")
            out = fluid.layers.image_resize_short(input, out_short_len=3)
    """
    in_shape = input.shape
    if len(in_shape) != 4:
        raise ValueError(
            "The rank of input must be 4 (num_batches, channels, in_h, in_w).")
    hw = in_shape[2:4]
    short_idx = hw.index(min(hw))
    long_idx = 1 - short_idx
    out_shape = list(hw)
    out_shape[short_idx] = out_short_len
    out_shape[long_idx] = int(
        float(out_shape[long_idx]) * (float(out_short_len) / float(hw[
            short_idx])) + 0.5)
    return image_resize(input=input, out_shape=out_shape, resample=resample)


@deprecated(since="2.0.0", update_to="paddle.gather")
def gather(input, index, overwrite=True):
    """

    Output is obtained by gathering entries of the outer-most dimension
    of X indexed by `index` and concatenate them together.

    .. math::

        Out = X[Index]


    .. code-block:: text


                Given:

                X = [[1, 2],
                     [3, 4],
                     [5, 6]]

                Index = [1, 2]

                Then:

                Out = [[3, 4],
                       [5, 6]]

    Args:
        input (Tensor): The source input tensor with rank>=1. Supported data type is
            int32, int64, float32, float64 and uint8 (only for CPU),
            float16 (only for GPU).
        index (Tensor): The index input tensor with rank=1. Data type is int32 or int64.
        overwrite (bool, optional): The mode that updating the grad when has same index.
            If True, use the overwrite mode to update the grad of the same index,
	    if False, use the accumulate mode to update the grad of the same index.
	    Default value is True.

    Returns:
        output (Tensor): The output is a tensor with the same rank as input.
    
    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            x = fluid.data(name='x', shape=[-1, 5], dtype='float32')
            index = fluid.data(name='index', shape=[-1, 1], dtype='int32')
            output = fluid.layers.gather(x, index)
    """
    if in_dygraph_mode():
        return _C_ops.gather(input, index, None, 'overwrite', overwrite)

    check_variable_and_dtype(
        input, 'x',
        ['float16', 'float32', 'float64', 'int32', 'int64', 'uint8'], 'gather')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'], 'gather')
    helper = LayerHelper('gather', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="gather",
        inputs={"X": input,
                "Index": index},
        outputs={"Out": out},
        attrs={'overwrite': overwrite})
    return out


@deprecated(since="2.0.0", update_to="paddle.gather_nd")
def gather_nd(input, index, name=None):
    """
    **Gather Nd Layer**

    This function is actually a high-dimensional extension of :code:`gather`
    and supports for simultaneous indexing by multiple axes. :attr:`index` is a
    K-dimensional integer tensor, which is regarded as a (K-1)-dimensional
    tensor of :attr:`index` into :attr:`input`, where each element defines
    a slice of params:

    .. math::

        output[(i_0, ..., i_{K-2})] = input[index[(i_0, ..., i_{K-2})]]

    Obviously, :code:`index.shape[-1] <= input.rank` . And, the output tensor has
    shape :code:`index.shape[:-1] + input.shape[index.shape[-1]:]` .

    .. code-block:: text

            Given:
                input = [[[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]],
                         [[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]]]
                input.shape = (2, 3, 4)

            * Case 1:
                index = [[1]]

                gather_nd(input, index)
                         = [input[1, :, :]]
                         = [[12, 13, 14, 15],
                            [16, 17, 18, 19],
                            [20, 21, 22, 23]]

            * Case 2:
                index = [[0,2]]

                gather_nd(input, index)
                         = [input[0, 2, :]]
                         = [8, 9, 10, 11]

            * Case 3:
                index = [[1, 2, 3]]

                gather_nd(input, index)
                         = [input[1, 2, 3]]
                         = [23]

    Args:
        input (Tensor): The input Tensor which it's data type should be bool, float32, float64, int32, int64.
        index (Tensor): The index input with rank > 1, index.shape[-1] <= input.rank.
                        Its dtype should be int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
                        For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        output (Tensor): A tensor with the shape index.shape[:-1] + input.shape[index.shape[-1]:]

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            x = fluid.data(name='x', shape=[3, 4, 5], dtype='float32')
            index = fluid.data(name='index', shape=[2, 2], dtype='int32')
            output = fluid.layers.gather_nd(x, index)

    """
    if in_dygraph_mode():
        return _C_ops.gather_nd(input, index)
    check_variable_and_dtype(input, 'input',
                             ['bool', 'float32', 'float64', 'int32', 'int64'],
                             'gather_np')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'], 'gather_np')
    helper = LayerHelper('gather_nd', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="gather_nd",
        inputs={"X": input,
                "Index": index},
        outputs={"Out": output})
    return output


@deprecated(since="2.0.0", update_to="paddle.scatter")
def scatter(input, index, updates, name=None, overwrite=True):
    """
    :alias_main: paddle.scatter
	:alias: paddle.scatter,paddle.tensor.scatter,paddle.tensor.manipulation.scatter
	:old_api: paddle.fluid.layers.scatter

    **Scatter Layer**

    Output is obtained by updating the input on selected indices based on updates.

    .. code-block:: python

        import numpy as np

        #input:
        input = np.array([[1, 1], [2, 2], [3, 3]])
        index = np.array([2, 1, 0, 1])
        # shape of updates should be the same as input
        # shape of updates with dim > 1 should be the same as input
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        overwrite = False

        # calculation:
        if not overwrite:
            for i in range(len(index)):
                input[index[i]] = np.zeros((2))

        for i in range(len(index)):
            if (overwrite):
                input[index[i]] = updates[i]
            else:
                input[index[i]] += updates[i]
        # output:
        out = np.array([[3, 3], [6, 6], [1, 1]])
        out.shape # [3, 2]

    Args:
        input (Variable): The input N-D Tensor with rank>=1. Data type can be float32.
        index (Variable): The index 1-D Tensor. Data type can be int32, int64. The length of index cannot exceed updates's length, and the value in index cannot exceed input's length.
        updates (Variable): update input with updates parameter based on index. shape should be the same as input, and dim value with dim > 1 should be the same as input.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .
        overwrite (bool): The mode that updating the output when there are same indices.
            If True, use the overwrite mode to update the output of the same index,
	    if False, use the accumulate mode to update the output of the same index.
	    Default value is True.

    Returns:
        Variable(Tensor|LoDTensor): The output is a Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np
            import paddle.fluid as fluid
            paddle.enable_static()

            input = fluid.layers.data(name='data', shape=[3, 2], dtype='float32', append_batch_size=False)
            index = fluid.layers.data(name='index', shape=[4], dtype='int64', append_batch_size=False)
            updates = fluid.layers.data(name='update', shape=[4, 2], dtype='float32', append_batch_size=False)

            output = fluid.layers.scatter(input, index, updates, overwrite=False)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            in_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
            index_data = np.array([2, 1, 0, 1]).astype(np.int64)
            update_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'data':in_data, "index":index_data, "update":update_data}, fetch_list=[output])
            print(res)
            # [array([[3., 3.],
            #   [6., 6.],
            #   [1., 1.]], dtype=float32)]
    """
    helper = LayerHelper('scatter', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="scatter",
        inputs={"X": input,
                "Ids": index,
                "Updates": updates},
        attrs={'overwrite': overwrite},
        outputs={"Out": out})
    return out


def scatter_nd_add(ref, index, updates, name=None):
    r"""
    **Scatter_nd_add Layer**

    Output is obtained by applying sparse addition to a single value
    or slice in a Variable.

    :attr:`ref` is a Tensor with rank :math:`R`
    and :attr:`index` is a Tensor with rank :math:`K` . Thus, :attr:`index`
    has shape :math:`[i_0, i_1, ..., i_{K-2}, Q]` where :math:`Q \leq R` . :attr:`updates`
    is a Tensor with rank :math:`K - 1 + R - Q` and its
    shape is :math:`index.shape[:-1] + ref.shape[index.shape[-1]:]` .

    According to the :math:`[i_0, i_1, ..., i_{K-2}]` of :attr:`index` ,
    add the corresponding :attr:`updates` slice to the :attr:`ref` slice
    which is obtained by the last one dimension of :attr:`index` .

    .. code-block:: text

        Given:

        * Case 1:
            ref = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          we get:

            output = [0, 22, 12, 14, 4, 5]

        * Case 2:
            ref = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            ref.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          we get:

            output = [[67, 19], [-16, -27]]

    Args:
        ref (Variable): The ref input. Its dtype should be int32, int64, float32, float64.
        index (Variable): The index input with rank > 1 and index.shape[-1] <= ref.rank.
                          Its dtype should be int32 or int64 as it is used as indexes.
        updates (Variable): The updated value of scatter_nd_add op, and it must have the same dtype
                            as ref. It must have the shape index.shape[:-1] + ref.shape[index.shape[-1]:].
        name (str|None): The output variable name. If set None, the layer will be named automatically.

    Returns:
        output (Variable): The output is a tensor with the same shape and dtype as ref.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            ref = fluid.data(name='ref', shape=[3, 5, 9, 10], dtype='float32')
            index = fluid.data(name='index', shape=[3, 2], dtype='int32')
            updates = fluid.data(name='update', shape=[3, 9, 10], dtype='float32')

            output = fluid.layers.scatter_nd_add(ref, index, updates)
    """

    if in_dygraph_mode():
        op = getattr(_C_ops, 'scatter_nd_add')
        return op(ref, index, updates)

    if ref.dtype != updates.dtype:
        raise ValueError("ref and updates must have same data type.")

    helper = LayerHelper('scatter_nd_add', **locals())
    dtype = helper.input_dtype(input_param_name='ref')
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="scatter_nd_add",
        inputs={"X": ref,
                "Index": index,
                "Updates": updates},
        outputs={"Out": output})
    return output


def scatter_nd(index, updates, shape, name=None):
    """
    **Scatter_nd Layer**

    Output is obtained by scattering the :attr:`updates` in a new tensor according
    to :attr:`index` . This op is similar to :code:`scatter_nd_add`, except the
    tensor of :attr:`shape` is zero-initialized. Correspondingly, :code:`scatter_nd(index, updates, shape)`
    is equal to :code:`scatter_nd_add(paddle.zeros(shape, updates.dtype), index, updates)` .
    If :attr:`index` has repeated elements, then the corresponding updates are accumulated.
    Because of the numerical approximation issues, the different order of repeated elements
    in :attr:`index` may cause different results. The specific calculation method can be
    seen :code:`scatter_nd_add` . This op is the inverse of the :code:`gather_nd` op.

    Args:
        index (Tensor): The index input with ndim > 1 and index.shape[-1] <= len(shape).
                          Its dtype should be int32 or int64 as it is used as indexes.
        updates (Tensor): The updated value of scatter_nd op. Its dtype should be float32, float64.
                            It must have the shape index.shape[:-1] + shape[index.shape[-1]:]
        shape(tuple|list): Shape of output tensor.
        name (str|None): The output Tensor name. If set None, the layer will be named automatically.

    Returns:
        output (Tensor): The output is a tensor with the same type as :attr:`updates` .

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np

            index_data = np.array([[1, 1],
                                   [0, 1],
                                   [1, 3]]).astype(np.int64)
            index = paddle.to_tensor(index_data)
            updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
            shape = [3, 5, 9, 10]

            output = paddle.scatter_nd(index, updates, shape)
    """
    return scatter_nd_add(zeros(shape, updates.dtype), index, updates, name)


@templatedoc()
def random_crop(x, shape, seed=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        shape(${shape_type}): ${shape_comment}
        seed(int|${seed_type}|None): ${seed_comment} By default, the seed will
            get from `random.randint(-65536, 65535)`.

    Returns:
        ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            img = fluid.data("img", [None, 3, 256, 256])
            # cropped_img is [-1, 3, 224, 224]
            cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])

            # cropped_img2 shape: [-1, 2, 224, 224]
            # cropped_img2 = fluid.layers.random_crop(img, shape=[2, 224, 224])

            # cropped_img3 shape: [-1, 3, 128, 224]
            # cropped_img3 = fluid.layers.random_crop(img, shape=[128, 224])

    """
    helper = LayerHelper("random_crop", **locals())
    check_variable_and_dtype(x, 'x',
                             ['float32', 'float64', 'uint8', 'int16', 'int32'],
                             'random_crop')
    check_type(shape, 'shape', (list, Variable), 'random_crop')
    dtype = x.dtype
    out = helper.create_variable_for_type_inference(dtype)
    if seed is None:
        seed = np.random.randint(-65536, 65536)
    op_attrs = {"shape": shape}
    if isinstance(seed, int):
        op_attrs["startup_seed"] = seed
        seed = helper.create_variable(
            name=unique_name.generate("random_crop_seed"),
            dtype="int64",
            persistable=True)
    elif not isinstance(seed, Variable):
        raise ValueError("'seed' must be a Variable or an int.")
    helper.append_op(
        type="random_crop",
        inputs={"X": x,
                "Seed": seed},
        outputs={"Out": out,
                 "SeedOut": seed},
        attrs=op_attrs)
    return out


def log(x, name=None):
    r"""
    Calculates the natural log of the given input tensor, element-wise.

    .. math::

        Out = \\ln(x)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`


    Returns:
        Tensor: The natural log of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python

            import paddle

            x = [[2,3,4], [7,8,9]]
            x = paddle.to_tensor(x, dtype='float32')
            res = paddle.log(x)
            # [[0.693147, 1.09861, 1.38629], [1.94591, 2.07944, 2.19722]]
    """
    if in_dygraph_mode():
        return _C_ops.log(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], "log")
    inputs = {'X': [x]}
    helper = LayerHelper('log', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log", inputs={"X": x}, outputs={"Out": out})
    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.relu")
def relu(x, name=None):
    """
    ${comment}

    Args:
        x(Variable): ${x_comment}
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Variable: ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            in1 = np.array([[-1,0],[1,2.6]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.relu(x1)
                print(out1.numpy())
                # [[0.  0. ]
                #  [1.  2.6]]
"""
    if in_dygraph_mode():
        return _C_ops.relu(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'relu')

    inputs = {'X': [x]}
    helper = LayerHelper('relu', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="relu", inputs={"X": helper.input('x')}, outputs={"Out": out})
    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.selu")
def selu(x, scale=None, alpha=None, name=None):
    r"""

    Selu Operator.

    The equation is:

    .. math::
        selu= \\lambda*
        \\begin{cases}
            x                      &\\quad \\text{ if } x>0 \n
            \\alpha * e^x - \\alpha  &\\quad \\text{ if } x<=0
        \\end{cases}


    The input `X` can carry the LoD (Level of Details) information,
    or not. And the output shares the LoD information with input `X`.

    Args:
        x (Variable): The input N-D Tensor.
        scale(float, optional): lambda in selu activation function,
            the default value is 1.0507009873554804934193349852946.
            For more information about this value, please refer
            to: https://arxiv.org/abs/1706.02515.
        alpha(float, optional): alpha in selu activation function,
            the default value is 1.6732632423543772848170429916717.
            For more information about this value, please refer
            to: https://arxiv.org/abs/1706.02515.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .


    Returns:
        Variable(Tensor|LoDTensor): The output Tensor or LoDTensor with the same shape and LoD information as input.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            paddle.enable_static()

            inputs = fluid.layers.data(name="x", shape=[2, 2], dtype="float32")
            output = fluid.layers.selu(inputs)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            img = np.array([[0, 1],[2, 3]]).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
            print(res) # [array([[0.      , 1.050701],[2.101402, 3.152103]], dtype=float32)]
    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'selu')

    helper = LayerHelper('selu', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    attrs = {}
    if scale is not None:
        attrs["scale"] = scale
    if alpha is not None:
        attrs["alpha"] = alpha

    helper.append_op(
        type="selu", inputs={"X": x}, outputs={"Out": out}, attrs=attrs)
    return out


def mean_iou(input, label, num_classes):
    r"""
    Mean Intersection-Over-Union is a common evaluation metric for
    semantic image segmentation, which first computes the IOU for each
    semantic class and then computes the average over classes.
    IOU is defined as follows:

    .. math::

        IOU = \\frac{true\_positive}{(true\_positive + false\_positive + false\_negative)}.

    The predictions are accumulated in a confusion matrix and mean-IOU
    is then calculated from it.


    Parameters:
        input (Tensor): A n-D Tensor of prediction results for semantic labels with type int32 or int64.
        label (Tensor): A Tensor of ground truth labels with type int32 or int64.
                           Its shape should be the same as input.
        num_classes (int32): The possible number of labels.

    Returns:
	Three Tensors.

        - mean_iou(Tensor) : A 1-D Tensor representing the mean intersection-over-union with shape [1]. \
			    Data type is float32.
        - out_wrong(Tensor) : A 1-D Tensor with shape [num_classes]. Data type is int32. \
			     The wrong numbers of each class.
        - out_correct(Tensor): A 1-D  Tensor with shape [num_classes]. Data type is int32. The correct numbers of each class.


    Examples:

        .. code-block:: python

            import paddle

            iou_shape = [64, 32, 32]
            num_classes = 5
            predict = paddle.randint(low=0, high=255, shape=iou_shape, dtype='int64')
            label = paddle.randint(low=0, high=255, shape=iou_shape, dtype='int64')
            mean_iou, out_wrong, out_correct = paddle.metric.mean_iou(predict, label, num_classes)
    """
    if in_dygraph_mode():
        return _C_ops.mean_iou(input, label, 'num_classes', num_classes)

    helper = LayerHelper('mean_iou', **locals())
    check_variable_and_dtype(input, 'Predictions', ['int32', 'int64'],
                             'mean_iou')
    check_variable_and_dtype(label, 'Labels', ['int32', 'int64'], 'mean_iou')
    out_mean_iou = helper.create_variable_for_type_inference(dtype='float32')
    out_wrong = helper.create_variable_for_type_inference(dtype='int32')
    out_correct = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="mean_iou",
        inputs={"Predictions": input,
                "Labels": label},
        outputs={
            "OutMeanIou": out_mean_iou,
            "OutWrong": out_wrong,
            "OutCorrect": out_correct
        },
        attrs={"num_classes": num_classes})
    return out_mean_iou, out_wrong, out_correct


def crop(x, shape=None, offsets=None, name=None):
    """
    Crop input into output, as specified by offsets and shape.

    **Warning:** THIS OP IS DEPRECATED. It will be removed in the future version.
    Instructions for updating: Use :ref:`api_fluid_layers_crop_tensor` instead.

    .. code-block:: text

        * Case 1:
            Given
                X = [[0, 1, 2, 0, 0]
                     [0, 3, 4, 0, 0]
                     [0, 0, 0, 0, 0]],
            and
                shape = [2, 2],
                offsets = [0, 1],
            output is:
                Out = [[1, 2],
                       [3, 4]].
        * Case 2:
            Given
                X = [[0, 1, 2, 5, 0]
                     [0, 3, 4, 6, 0]
                     [0, 0, 0, 0, 0]],
            and shape is tensor
                shape = [[0, 0, 0]
                         [0, 0, 0]]
            and
                offsets = [0, 1],

            output is:
                Out = [[1, 2, 5],
                       [3, 4, 6]].

    Parameters:
        x (Variable): Tensor, data type can be float32 or float64.
        shape (Variable|list/tuple of integers): The output shape is specified
            by `shape`, which can be a Tensor or a list/tuple of integers.
            If it is a Tensor, it's rank must be the same as `x` , only
            it's shape will be used, and the value of it will be ignored. This way
            is suitable for the case that the output shape may be changed each
            iteration. If it is a list/tuple of integers, it's length must be the same
            as the rank of `x`
        offsets (Variable|list/tuple of integers|None): Specifies the cropping
            offsets at each dimension. It can be a Tensor or a list/tuple
            of integers. If it is a Tensor, it's rank must be the same as `x`.
            This way is suitable for the case that the offsets may be changed
            each iteration. If it is a list/tuple of integers, it's length must be the
            same as the rank of `x`. If None, the offsets are 0 at each dimension.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name` . Usually name is no need to set and
            None by default.

    Returns:
        The cropped Tensor, which has the same rank and data type with `x`

    Return Type:
        Variable

    Raises:
        ValueError: If shape is not a list, tuple or Variable.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            x = fluid.data(name="x", shape=[3, 3, 5], dtype="float32")
            y = fluid.data(name="y", shape=[2, 2, 3], dtype="float32")
            crop = fluid.layers.crop(x, shape=y)

            # or
            z = fluid.data(name="z", shape=[3, 3, 5], dtype="float32")
            crop = fluid.layers.crop(z, shape=[2, 2, 3])

    """
    check_variable_and_dtype(x, 'x', ['float32'], 'crop')
    check_type(shape, 'shape', (list, tuple, Variable), 'crop')
    helper = LayerHelper('crop', **locals())

    if offsets is None:
        offsets = [0] * len(x.shape)

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x}
    attrs = {}
    if isinstance(shape, Variable):
        ipts['Y'] = shape
    else:
        attrs['shape'] = shape
    if isinstance(offsets, Variable):
        ipts['Offsets'] = offsets
    else:
        attrs['offsets'] = offsets

    helper.append_op(
        type='crop',
        inputs=ipts,
        outputs={'Out': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def crop_tensor(x, shape=None, offsets=None, name=None):
    """
    Crop input into output, as specified by offsets and shape.

    .. code-block:: text

        * Case 1 (input is a 2-D Tensor):
            Input:
                X.shape = [3, 5]
                X.data = [[0, 1, 2, 0, 0],
                          [0, 3, 4, 0, 0],
                          [0, 0, 0, 0, 0]]
            Parameters:
                shape = [2, 2]
                offsets = [0, 1]
            Output:
                Out.shape = [2, 2]
                Out.data = [[1, 2],
                            [3, 4]]
        * Case 2 (input is a 3-D Tensor):
            Input:
                X.shape = [2, 3, 4]
                X.data =  [[[0, 1, 2, 3],
                            [0, 5, 6, 7],
                            [0, 0, 0, 0]],
                           [[0, 3, 4, 5],
                            [0, 6, 7, 8],
                            [0, 0, 0, 0]]]
            Parameters:
                shape = [2, 2, -1]
                offsets = [0, 0, 1]
            Output:
                Out.shape = [2, 2, 3]
                Out.data  = [[[1, 2, 3],
                              [5, 6, 7]],
                             [[3, 4, 5],
                              [6, 7, 8]]]

    Parameters:
        x (Tensor): 1-D to 6-D Tensor, the data type is float32, float64, int32 or int64.
        shape (list|tuple|Tensor): The output shape is specified
            by `shape`. Its data type is int32. If a list/tuple, it's length must be
            the same as the dimension size of `x`. If a Tensor, it should be a 1-D Tensor.
            When it is a list, each element can be an integer or a Tensor of shape: [1].
            If Variable contained, it is suitable for the case that the shape may
            be changed each iteration.
        offsets (list|tuple|Variable, optional): Specifies the cropping
            offsets at each dimension. Its data type is int32. If a list/tuple, it's length
            must be the same as the dimension size of `x`. If a Tensor, it should be a 1-D
            Tensor. When it is a list, each element can be an integer or a Tensor of shape: [1].
            If Variable contained, it is suitable for the case that the offsets may be changed
            each iteration. Default: None, the offsets are 0 at each dimension.
        name(str, optional): The default value is None. Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: The cropped Tensor has same data type with `x`.

    Examples:

        .. code-block:: python
          :name: code-example1

            import paddle
            x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            # x.shape = [3, 3]
            # x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

            # shape can be a 1-D Tensor or list or tuple.
            shape = paddle.to_tensor([2, 2], dtype='int32')
            # shape = [2, 2]
            # shape = (2, 2)
            out = paddle.crop(x, shape)
            # out.shape = [2, 2]
            # out = [[1,2], [4,5]]

            # offsets can be a 1-D Tensor or list or tuple.
            offsets = paddle.to_tensor([0, 1], dtype='int32')
            # offsets = [1, 0]
            # offsets = (1, 1)
            out = paddle.crop(x, shape, offsets)
            # out.shape = [2, 2]
            # if offsets = [0, 0], out = [[1,2], [4,5]]
            # if offsets = [0, 1], out = [[2,3], [5,6]]
            # if offsets = [1, 0], out = [[4,5], [7,8]]
            # if offsets = [1, 1], out = [[5,6], [8,9]]

    """
    helper = LayerHelper('crop_tensor', **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'crop_tensor')
    check_type(shape, 'shape', (list, tuple, Variable), 'crop_tensor')
    check_type(offsets, 'offsets', (list, tuple, Variable, type(None)),
               'crop_tensor')

    if offsets is None:
        offsets = [0] * len(x.shape)

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x}
    attrs = {}

    def _attr_shape_check(shape_val):
        if not isinstance(shape_val, int):
            raise TypeError(
                "Attr(shape)'s dtype of Op(crop_tensor) should be int32, but received: %s."
                % type(shape_val))
        if shape_val == 0:
            raise ValueError(
                "Attr(shape) of Op(crop_tensor) should not be zero, but received: %s."
                % str(shape_val))
        if shape_val < -1:
            raise ValueError(
                "When the element in Attr(shape) of Op(crop_tensor) is negative, only -1 is supported, but received: %s."
                % str(shape_val))

    def _attr_offsets_check(offset_val):
        if not isinstance(offset_val, int):
            raise TypeError(
                "Attr(offsets)'s dtype of Op(crop_tensor) should be int32, but received: %s."
                % type(offset_val))
        if offset_val < 0:
            raise ValueError(
                "Attr(offsets) of Op(crop_tensor) should be greater or equal to zero, but received: %s."
                % str(offset_val))

    if isinstance(offsets, Variable):
        offsets.stop_gradient = True
        ipts['Offsets'] = offsets
        attrs['offsets'] = [-1] * len(x.shape)
    elif utils._contain_var(offsets):
        new_offsets_tensor = []
        offsets_attr = []
        for dim in offsets:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_offsets_tensor.append(dim)
                offsets_attr.append(-1)
            else:
                _attr_offsets_check(dim)
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_offsets_tensor.append(temp_out)
                offsets_attr.append(dim)
        ipts['OffsetsTensor'] = new_offsets_tensor
        attrs['offsets'] = offsets_attr
    else:
        for offset in offsets:
            _attr_offsets_check(offset)
        attrs['offsets'] = offsets

    if isinstance(shape, Variable):
        shape.stop_gradient = True
        ipts['Shape'] = shape
    elif utils._contain_var(shape):
        new_shape_tensor = []
        shape_attr = []
        for dim_size in shape:
            if isinstance(dim_size, Variable):
                dim_size.stop_gradient = True
                new_shape_tensor.append(dim_size)
                shape_attr.append(0)
            else:
                _attr_shape_check(dim_size)
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant(
                    [1], 'int32', dim_size, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
                shape_attr.append(dim_size)
        ipts['ShapeTensor'] = new_shape_tensor
        attrs['shape'] = shape_attr
    else:
        for dim_size in shape:
            _attr_shape_check(dim_size)
        attrs['shape'] = shape

    helper.append_op(
        type='crop_tensor',
        inputs=ipts,
        outputs={'Out': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def affine_grid(theta, out_shape, name=None):
    """
    :alias_main: paddle.nn.functional.affine_grid
	:alias: paddle.nn.functional.affine_grid,paddle.nn.functional.vision.affine_grid
	:old_api: paddle.fluid.layers.affine_grid

    It generates a grid of (x,y) coordinates using the parameters of
    the affine transformation that correspond to a set of points where
    the input feature map should be sampled to produce the transformed
    output feature map.

    Args:
        theta (Variable) - A Tensor with shape [N, 2, 3]. It contains a batch of affine transform parameters.
                           The data type can be float32 or float64.
        out_shape (Variable | list | tuple): The shape of target output with format [batch_size, channel, height, width].
                                             ``out_shape`` can be a Tensor or a list or tuple. The data
                                             type must be int32.
        name(str|None): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: A Tensor with shape [batch_size, H, W, 2] while 'H' and 'W' are the height and width of feature map in affine transformation. The data type is the same as `theta`.

    Raises:
        ValueError: If the type of arguments is not supported.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            place = fluid.CPUPlace()
            theta = fluid.data(name="x", shape=[None, 2, 3], dtype="float32")
            out_shape = fluid.data(name="y", shape=[4], dtype="int32")
            grid_0 = fluid.layers.affine_grid(theta, out_shape)
            grid_1 = fluid.layers.affine_grid(theta, [5, 3, 28, 28])
            batch_size=2
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            output= exe.run(feed={"x": np.random.rand(batch_size,2,3).astype("float32"),
                                  "y": np.array([5, 3, 28, 28]).astype("int32")},
                                  fetch_list=[grid_0.name, grid_1.name])
            print(output[0])
            print(output[1])
    """
    helper = LayerHelper('affine_grid')

    check_variable_and_dtype(theta, 'theta', ['float32', 'float64'],
                             'affine_grid')

    if not (isinstance(out_shape, list) or isinstance(out_shape, tuple) or \
            isinstance(out_shape, Variable)):
        raise ValueError("The out_shape should be a list, tuple or Variable.")

    if not isinstance(theta, Variable):
        raise ValueError("The theta should be a Variable.")

    out = helper.create_variable_for_type_inference(theta.dtype)
    ipts = {'Theta': theta}
    attrs = {}
    if isinstance(out_shape, Variable):
        ipts['OutputShape'] = out_shape
        check_variable_and_dtype(out_shape, 'out_shape', ['int32'],
                                 'affine_grid')
    else:
        attrs['output_shape'] = out_shape
    if core.is_compiled_with_rocm():
        # ROCM platform do not have MIOPEN kernel for affine_grid
        attrs['use_cudnn'] = False

    helper.append_op(
        type='affine_grid',
        inputs=ipts,
        outputs={'Output': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def pad2d(input,
          paddings=[0, 0, 0, 0],
          mode='constant',
          pad_value=0.0,
          data_format="NCHW",
          name=None):
    """

    Pad 2-d images according to 'paddings' and 'mode'.
    If mode is 'reflect', paddings[0] and paddings[1] must be no greater
    than height-1. And the width dimension has the same condition.

    Parameters:
        input (Tensor): The input image with [N, C, H, W] format or [N, H, W, C] format, which is a 4-D Tensor with data type float32.
        paddings (Tensor | List[int32]): The padding size. If padding is a List, it must
            contain four integers, (padding_top, padding_bottom, padding_left, padding_right).
            Otherwise, it is a 1-D Tensor with shape [4]. Data type is int32.
            Default is [0, 0, 0, 0].
        mode (str): Three modes: 'constant' (default), 'reflect', 'edge' .
        	When in 'constant' mode, this op uses a constant value to pad the input tensor.
        	When in 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
        	When in 'edge' mode, uses input boundaries to pad the input tensor.
        	Default is 'constant'
        pad_value (float32): The value to fill the padded areas in 'constant' mode . Default is 0.0
        data_format (str): An string from: "NHWC", "NCHW". Specify the data format of
                           the input data.
                           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
                    user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns: Tensor, a 4-D Tensor padded according to paddings and mode and data type is same as input.

    Examples:
        .. code-block:: text

            Input = [[[[1., 2., 3.],
                       [4., 5., 6.]]]]

            Case 0:
                paddings = [0, 1, 2, 3],
                mode = 'constant'
                pad_value = 0
                Out = [[[[0., 0., 1., 2., 3., 0., 0., 0.],
                         [0., 0., 4., 5., 6., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0.]]]]

            Case 1:
                paddings = [0, 1, 2, 1],
                mode = 'reflect'
                Out = [[[[3., 2., 1., 2., 3., 2.],
                         [6., 5., 4., 5., 6., 5.],
                         [3., 2., 1., 2., 3., 2.]]]]

            Case 2:
                paddings = [0, 1, 2, 1],
                mode = 'edge'
                Out = [[[[1., 1., 1., 2., 3., 3.],
                         [4., 4., 4., 5., 6., 6.],
                         [4., 4., 4., 5., 6., 6.]]]]

    Code Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F

            # example 1
            x_shape = (1, 1, 3, 4)
            x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
            tensor_x = paddle.to_tensor(x)
            y = paddle.fluid.layers.pad2d(tensor_x, paddings=[1, 2, 2, 1], pad_value=1, mode='constant')
            print(y.numpy())
            # [[[[ 1.  1.  1.  1.  1.  1.  1.]
            #    [ 1.  1.  1.  2.  3.  4.  1.]
            #    [ 1.  1.  5.  6.  7.  8.  1.]
            #    [ 1.  1.  9. 10. 11. 12.  1.]
            #    [ 1.  1.  1.  1.  1.  1.  1.]
            #    [ 1.  1.  1.  1.  1.  1.  1.]]]]

            # example 2
            x_shape = (1, 1, 2, 3)
            x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
            tensor_x = paddle.to_tensor(x)
            y = paddle.fluid.layers.pad2d(tensor_x, paddings=[1, 1, 1, 1], mode='reflect')
            print(y.numpy())
            # [[[[5. 4. 5. 6. 5.]
            #    [2. 1. 2. 3. 2.]
            #    [5. 4. 5. 6. 5.]
            #    [2. 1. 2. 3. 2.]]]]
    """
    if in_dygraph_mode():
        _paddings = paddings.numpy().tolist() if isinstance(
            paddings, Variable) else paddings
        return _C_ops.pad2d(input, 'mode', mode, 'pad_value', pad_value,
                            'data_format', data_format, 'paddings', _paddings)

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64', 'int32', 'int64'],
        "pad2d")

    attrs = {'mode': mode, 'pad_value': pad_value, 'data_format': data_format}
    inputs = {'X': [input]}
    if isinstance(paddings, Variable):
        inputs['Paddings'] = [paddings]
        attrs['paddings'] = []
    else:
        attrs['paddings'] = paddings

    helper = LayerHelper('pad2d', **locals())

    assert mode in ['reflect', 'edge', 'constant'
                    ], "mode should be one of constant, reflect, edge."

    dtype = helper.input_dtype(input_param_name='input')
    out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='pad2d', inputs=inputs, outputs={"Out": out}, attrs=attrs)

    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.elu")
def elu(x, alpha=1.0, name=None):
    """
    :alias_main: paddle.nn.functional.elu
	:alias: paddle.nn.functional.elu,paddle.nn.functional.activation.elu
	:old_api: paddle.fluid.layers.elu

    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        alpha(${alpha_type}|1.0): ${alpha_comment}
        name(str|None): The default value is None. Normally there is no need for user to set this property.
                        For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        ${out_type}: ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            input_elu = np.array([[-1,6],[1,15.6]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(input_elu)
                y = fluid.layers.elu(x, alpha=0.2)
                print(y.numpy())
                # [[-0.12642411  6.        ]
                # [ 1.          15.6       ]]
    """
    helper = LayerHelper('elu', **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'elu')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='elu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'alpha': alpha})
    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.relu6")
def relu6(x, threshold=6.0, name=None):
    """

    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        threshold(float, optional): ${threshold_comment}
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            in1 = np.array([[-1,0],[2.5,7.8]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.relu6(x=x1, threshold=6.0)
                print(out1.numpy())
                # [[0.  0. ]
                #  [2.5 6. ]]
    """
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'relu6')

    helper = LayerHelper('relu6', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='relu6',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'threshold': threshold,
            'use_mkldnn': _global_flags()["FLAGS_use_mkldnn"]
        })
    return out


@templatedoc()
def pow(x, factor=1.0, name=None):
    """
    This is Pow Activation Operator.

    :math:`out = x^{factor}`

    Args:
        x(Variable): A ``Tensor`` or ``LoDTensor`` . The data type is ``float32`` or ``float64``.
        factor(float32|Variable, optional): A scalar with type ``float32`` or a ``Tensor`` with shape [1] and type ``float32``.  The exponential factor of Pow. Default 1.0.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``x``.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.data(name="x", shape=[32,32], dtype="float32")

            # example 1: argument factor is float
            y_1 = fluid.layers.pow(x, factor=2.0)
            # y_1 is x^{2.0}

            # example 2: argument factor is Variable
            factor_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
            y_2 = fluid.layers.pow(x, factor=factor_tensor)
            # y_2 is x^{3.0}
    """
    check_variable_and_dtype(
        x, 'x', ['int32', 'int64', 'float16', 'float32', 'float64'], 'pow')

    helper = LayerHelper('pow', **locals())
    inputs = {'X': x}
    attrs = {}
    if isinstance(factor, Variable):
        check_variable_and_dtype(factor, 'factor', ['float32'], 'pow')
        factor.stop_gradient = True
        inputs['FactorTensor'] = factor
    else:
        attrs['factor'] = factor

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


@templatedoc()
def stanh(x, scale_a=0.67, scale_b=1.7159, name=None):
    """
    stanh activation.

    .. math::

        out = b * \\frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        scale_a (float, optional): The scale factor a of the input. Default is 0.67.
        scale_b (float, optional): The scale factor b of the output. Default is 1.7159.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) # [1.00616539, 1.49927628, 1.65933108, 1.70390463]

    """

    if in_dygraph_mode():
        return _C_ops.stanh(x, 'scale_a', scale_a, 'scale_b', scale_b)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'stanh')

    helper = LayerHelper('stanh', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='stanh',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'scale_a': scale_a,
               'scale_b': scale_b})
    return out


@templatedoc()
def hard_sigmoid(x, slope=0.2, offset=0.5, name=None):
    """
    ${comment}
    Parameters:
        x (${x_type}): ${x_comment}
        slope (float, optional): ${slope_comment}
        offset (float, optional): ${offset_comment}
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`

    Returns:
        ${out_type}: ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()

            data = fluid.layers.fill_constant(shape=[3, 2], value=0.5, dtype='float32') # [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
            result = fluid.layers.hard_sigmoid(data) # [[0.6, 0.6], [0.6, 0.6], [0.6, 0.6]]
    """
    if in_dygraph_mode():
        return _C_ops.hard_sigmoid(x, 'slope', slope, 'offset', offset)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'hard_sigmoid')

    helper = LayerHelper('hard_sigmoid', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='hard_sigmoid',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'slope': slope,
               'offset': offset})
    return out


@templatedoc()
def swish(x, beta=1.0, name=None):
    r"""
    :alias_main: paddle.nn.functional.swish
	:alias: paddle.nn.functional.swish,paddle.nn.functional.activation.swish
	:old_api: paddle.fluid.layers.swish

    Elementwise swish activation function. See `Searching for Activation Functions <https://arxiv.org/abs/1710.05941>`_ for more details.

    Equation:

    .. math::
        out = \\frac{x}{1 + e^{- beta * x}}

    Args:
        x(Variable): Tensor or LoDTensor, dtype: float32 or float64, the input of swish activation.

        beta(float): Constant beta of swish operator, default 1.0.

        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:

        Variable: Output of the swish activation, Tensor or LoDTensor, with the same dtype and shape with the input x.

    Examples:

        .. code-block:: python

            # declarative mode
            import numpy as np
            from paddle import fluid

            x = fluid.data(name="x", shape=(-1, 3), dtype="float32")
            y = fluid.layers.swish(x, beta=2.0)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            start = fluid.default_startup_program()
            main = fluid.default_main_program()

            data = np.random.randn(2, 3).astype("float32")
            exe.run(start)
            y_np, = exe.run(main, feed={"x": data}, fetch_list=[y])

            data
            # array([[-1.1239197 ,  1.3391294 ,  0.03921051],
            #        [ 1.1970421 ,  0.02440812,  1.2055548 ]], dtype=float32)
            y_np
            # array([[-0.2756806 ,  1.0610548 ,  0.01998957],
            #        [ 0.9193261 ,  0.01235299,  0.9276883 ]], dtype=float32)


        .. code-block:: python

            # imperative mode
            import numpy as np
            from paddle import fluid
            import paddle.fluid.dygraph as dg

            data = np.random.randn(2, 3).astype("float32")
            place = fluid.CPUPlace()
            with dg.guard(place) as g:
                x = dg.to_variable(data)
                y = fluid.layers.swish(x)
                y_np = y.numpy()
            data
            # array([[-0.0816701 ,  1.1603649 , -0.88325626],
            #        [ 0.7522361 ,  1.0978601 ,  0.12987892]], dtype=float32)
            y_np
            # array([[-0.03916847,  0.8835007 , -0.25835553],
            #        [ 0.51126915,  0.82324016,  0.06915068]], dtype=float32)
    """
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'swish')

    helper = LayerHelper('swish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='swish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'slope': beta})
    return out


@deprecated(since="2.0.0", update_to="paddle.static.nn.prelu")
def prelu(x, mode, param_attr=None, data_format="NCHW", name=None):
    r"""
    prelu activation.

    .. math::
        prelu(x) = max(0, x) + \alpha * min(0, x)

    There are three modes for the activation:

    .. code-block:: text

        all: All elements share same alpha.
        channel: Elements in same channel share same alpha.
        element: All elements do not share alpha. Each element has its own alpha.

    Parameters:
    
        x (Tensor): The input Tensor or LoDTensor with data type float32.

        mode (str): The mode for weight sharing.

        param_attr (ParamAttr|None, optional): The parameter attribute for the learnable \
        weight (alpha), it can be create by ParamAttr. None by default. \
        For detailed information, please refer to :ref:`api_fluid_ParamAttr`.

        name (str, optional): Name for the operation (optional, default is None). \
        For more information, please refer to :ref:`api_guide_Name`.
        
        data_format(str, optional): Data format that specifies the layout of input.
            It may be "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". Default: "NCHW".

    Returns:
        Tensor: A tensor with the same shape and data type as x.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-1., 2., 3.])
            param = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.2))
            out = paddle.static.nn.prelu(x, 'all', param)
            # [-0.2, 2., 3.]

    """
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'prelu')

    helper = LayerHelper('prelu', **locals())
    if mode not in ['all', 'channel', 'element']:
        raise ValueError('mode should be one of all, channel, element.')

    alpha_shape = [1]
    if mode == 'channel':

        true_data_format = [
            'NC', 'NCL', 'NCHW', 'NCDHW', 'NLC', 'NHWC', 'NDHWC'
        ]
        if data_format not in true_data_format:
            raise ValueError(
                "data_format must be one of 'NC', 'NCL', 'NCHW', 'NCDHW', "
                "'NLC', 'NHWC', 'NDHWC' but receive {}".format(data_format))

        data_format = 'NCHW' if data_format[1] == 'C' else 'NHWC'

        assert len(
            x.shape
        ) >= 2, "The size of input shape should be equal or larger than 2 in prelu() when mode is 'channel'"
        #NOTE(zhiqiu): The alpha_shape should be [1, channel] + [1] * len(x.shape[2:]).
        # To be consistent with Prelu, it is simplified.
        #NOTE(zhiqiu): Revert shape to [1, channel, 1, 1] for compatibility with saved model of old version.
        #NOTE(GuoxiaWang): support NHWC data format
        if data_format == 'NHWC':
            alpha_shape = [1, 1, 1, x.shape[-1]]
        else:
            alpha_shape = [1, x.shape[1], 1, 1]

    elif mode == 'element':
        assert len(
            x.shape
        ) >= 1, "The size of input shape should be equal or larger than 1 in prelu() when mode is 'element'"
        alpha_shape = [1] + list(x.shape)[1:]
    dtype = helper.input_dtype(input_param_name='x')
    alpha = helper.create_parameter(
        attr=helper.param_attr,
        shape=alpha_shape,
        dtype=dtype,
        is_bias=False,
        default_initializer=Constant(0.25))
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x,
                'Alpha': alpha},
        attrs={"mode": mode,
               "data_format": data_format},
        outputs={"Out": out})
    return out


@templatedoc()
def brelu(x, t_min=0.0, t_max=24.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        t_min(${t_min_type}|0.0): ${t_min_comment}
        t_max(${t_max_type}|24.0): ${t_max_comment}
        name(str|None): The default value is None. Normally there is no need for user to set this property.
                        For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        ${out_type}: ${out_comment}

    Examples:

    .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            import numpy as np
            paddle.enable_static()

            input_brelu = np.array([[-1,6],[1,15.6]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(input_brelu)
                y = fluid.layers.brelu(x, t_min=1.0, t_max=10.0)
                print(y.numpy())
                #[[ 1.  6.]
                #[ 1. 10.]]
    """
    if in_dygraph_mode():
        return _C_ops.brelu(x, 't_min', t_min, 't_max', t_max)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'brelu')

    helper = LayerHelper('brelu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='brelu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'t_min': t_min,
               't_max': t_max})
    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.leaky_relu")
@templatedoc()
def leaky_relu(x, alpha=0.02, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        alpha(${alpha_type}|0.02): ${alpha_comment}
        name(str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[-1, 2], [3, -4]], dtype='float32')
            y = paddle.fluid.layers.leaky_relu(x, alpha=0.1)
            print(y) # [[-0.1, 2], [3, -0.4]]

    """
    return paddle.nn.functional.leaky_relu(x, alpha, name)


def soft_relu(x, threshold=40.0, name=None):
    r"""

    SoftRelu Activation Operator.

    $out = \ln(1 + \exp(\max(\min(x, threshold), -threshold)))$

    Args:
        x(Variable): Input of soft_relu operator. Data type can be float32, float64.
        threshold(float, optional): The threshold value of soft_relu, default value being 40.0.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable(Tensor|LoDTensor)): Output of soft_relu operator, shape and LoD same as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import numpy as np
            import paddle

            paddle.enable_static()
            inputs = fluid.layers.data(name="x", shape=[2, 2], dtype="float32")
            output = fluid.layers.soft_relu(inputs, threshold=20.0)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            img = np.array([[0, 1],[2, 3]]).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
            print(res) # [array([[0.6931472, 1.3132616], [2.126928 , 3.0485873]], dtype=float32)]
    """
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'soft_relu')

    helper = LayerHelper('soft_relu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='soft_relu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold})
    return out


def flatten(x, axis=1, name=None):
    r"""
    **Flatten op**

    Flatten the input tensor into a 2D matrix.

    For Example:

    .. code-block:: text

        Case 1:

          Given
            X.shape = (3, 100, 100, 4)

          and
            axis = 2

          We get:
            Out.shape = (3 * 100, 4 * 100)

        Case 2:

          Given
            X.shape = (3, 100, 100, 4)

          and
            axis = 0

          We get:
            Out.shape = (1, 3 * 100 * 100 * 4)

    Args:
        x (Variable): A tensor of rank >= axis. A tensor with type float32,
                      float64, int8, int32, int64, uint8.
        axis (int): Indicate up to which input dimensions (exclusive) should
                    be flattened to the outer dimension of the output.
                    The value for axis must be in the range [0, R], where R
                    is the rank of the input tensor. Default: 1.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.

    Returns:
        Variable: A 2D tensor with the contents of the input tensor, with input \
                  dimensions up to axis flattened to the outer dimension of \
                  the output and remaining input dimensions flattened into the \
                  inner dimension of the output. A Tensor with type same as input x.

    Raises:
        ValueError: If x is not a variable.
        ValueError: If axis is not in range [0, rank(x)].

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()
            x = fluid.data(name="x", shape=[4, 4, 3], dtype="float32")
            # x shape is [4, 4, 3]
            out = fluid.layers.flatten(x=x, axis=2)
            # out shape is [16, 3]
    """
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int8', 'int32', 'int64', 'uint8'],
        'flatten')
    helper = LayerHelper('flatten', **locals())

    if not (isinstance(x, Variable)):
        raise ValueError("The input x should be a Variable")

    if not (isinstance(axis, int)) or axis > len(x.shape) or axis < 0:
        raise ValueError("The axis should be a int, and in range [0, rank(x)]")

    out = helper.create_variable_for_type_inference(x.dtype)
    x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='flatten2',
        inputs={"X": x},
        outputs={'Out': out,
                 'XShape': x_shape},
        attrs={"axis": axis})
    return out


def stack(x, axis=0, name=None):
    """

    This OP stacks all the inputs :code:`x` along axis.

    .. code-block:: text

        Case 1:

          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]

          Attrs:
            axis = 0

          Output:
            Out.dims = [3, 1, 2]
            Out.data =[ [ [1.0, 2.0] ],
                        [ [3.0, 4.0] ],
                        [ [5.0, 6.0] ] ]


        Case 2:


          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]


          Attrs:
            axis = 1 or axis = -2

          Output:
            Out.shape = [1, 3, 2]
            Out.data =[ [ [1.0, 2.0]
                          [3.0, 4.0]
                          [5.0, 6.0] ] ]


    Args:
        x (list(Variable)|tuple(Variable)): Input :code:`x` can be a :code:`list` or :code:`tuple` of Tensors, the shapes of all these Tensors
                                     must be the same. Supposing input is N dims
                                     Tensors :math:`[d_0, d_1, ..., d_{n-1}]`, the output is N+1 dims
                                     Tensor :math:`[d_0, d_1, d_{axis-1}, len(x), d_{axis}, ..., d_{n-1}]`.
                                     Supported data types: float32, float64, int32, int64.
        axis (int, optional): The axis along which all inputs are stacked. ``axis`` range is ``[-(R+1), R+1)``,
                              where ``R`` is the number of dimensions of the first input tensor ``x[0]``. 
                              If ``axis < 0``, ``axis = axis+R+1``. The default value of axis is 0.
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.
    

    Returns:
        Variable: The stacked Tensor, has same data type with input Tensors. Output dim is :math:`rank(x[0])+1`.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            # set batch size=None
            x1 = fluid.data(name='x1', shape=[None, 1, 2], dtype='int32')
            x2 = fluid.data(name='x2', shape=[None, 1, 2], dtype='int32')
            # stack Tensor list
            data = layers.stack([x1,x2]) # stack according to axis 0, data.shape=[2, None, 1, 2]

            data = layers.stack([x1,x2], axis=1) # stack according to axis 1, data.shape=[None, 2, 1, 2]


    """
    axis = 0 if axis is None else axis

    if in_dygraph_mode():
        return _C_ops.stack(x, 'axis', axis)

    if not isinstance(x, list) and not isinstance(x, tuple):
        # NOTE:(zhiqiu) Only support Variable as input if the Variable is a LOD_TENSOR_ARRAY create by create_array, array_write, array_read, etc.
        # In that case, Variable is array of tensors indeed.
        if isinstance(x, Variable) and x.desc.type(
        ) == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
            x = [x]
        else:
            raise TypeError("The type of '%s' in %s must be %s, but received %s"
                            % ('x', 'stack',
                               'list[Tensor], tuple[Tensor] or TensorArray',
                               type(x)))

    helper = LayerHelper('stack', **locals())

    out = helper.create_variable_for_type_inference(x[0].dtype)
    if x[0].desc.type() == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        assert len(x) == 1, "If the elements of 'x' in stack are Variable(LoDTensorArray), " \
                            "number of the elements must be 1, but received %s." % len(x)
        out_index = helper.create_variable_for_type_inference(dtype="int32")

        for i in x:
            check_variable_and_dtype(i, 'x', \
                ['float16', 'float32', 'float64', 'int32', 'int64'], 'stack')

        helper.append_op(
            type='tensor_array_to_tensor',
            inputs={'X': x[0]},
            outputs={'Out': [out],
                     'OutIndex': [out_index]},
            attrs={'axis': axis,
                   'use_stack': True})
    else:
        helper.append_op(
            type='stack',
            inputs={'X': x},
            outputs={'Y': out},
            attrs={'axis': axis})

    return out


@templatedoc(op_type="filter_by_instag")
def filter_by_instag(ins, ins_tag, filter_tag, is_lod, out_val_if_empty=0):
    """
    **Filter By Instag Layer**

    This function filter a batch of ins by instag,
    There are multiple ins, and every ins belongs to some tags.
    We can specify some tags we want. So the ins which belongs to that tags
    remains in the output, and others removed.

    For example, one batch has 4 ins. Every ins has its tag list.

       | Ins   |   Ins_Tag |
       |:-----:|:------:|
       |  0    |   0, 1 |
       |  1    |   1, 3 |
       |  2    |   0, 3 |
       |  3    |   2, 6 |

    And Lod is [1,1,1,1]

    And the filter tags [1]

    From the definition above, ins which has tag 1 can pass the filter
    So Ins 0 and Ins 1 can pass and be seen in the output,
    Ins 2 and 3 cannot pass because they do not has tag 1.

    Actually, if is_lod is false, it is normal tensor that equals to
    lod_tensor with all 1, similar to the example above.

    Args:
        ins (Variable): Input Variable (LoDTensor), usually it is 2D tensor
                        And first dimension can have lod info or not.
        ins_tag (Variable): Input Variable (LoDTensor), usually it is 1D list
                        And split them by lod info
        filter_tag (Variable): Input Variable (1D Tensor/List), usually it is
                        list that holds the tags.
        is_lod (Bool): Boolean value to indicate ins is lod tensor or not.
        out_val_if_empty(Int64): If the output after filter is empty, this value
                        will be set to Output tensor.

    Returns:
        Variable: filtered ins (LoDTensor) and loss weight (Tensor)

    Examples:
        .. code-block:: python

          import paddle.fluid.layers as layers
          ins = layers.data(name='Ins', shape=[-1,32], lod_level=0, dtype='float64')
          ins_tag = layers.data(name='Ins_tag', shape=[-1,16], lod_level=0, dtype='int64')
          filter_tag = layers.data(name='Filter_tag', shape=[-1,16], dtype='int64')
          out, loss_weight = layers.filter_by_instag(ins,  ins_tag,  filter_tag, True)

    """
    helper = LayerHelper('filter_by_instag', **locals())

    out = helper.create_variable_for_type_inference(dtype=ins.dtype)
    loss_weight = helper.create_variable_for_type_inference(dtype=np.float64)
    mmap = helper.create_variable_for_type_inference(dtype=ins_tag.dtype)
    helper.append_op(
        type='filter_by_instag',
        inputs={'Ins': ins,
                'Ins_tag': ins_tag,
                'Filter_tag': filter_tag},
        outputs={'Out': out,
                 'LossWeight': loss_weight,
                 'IndexMap': mmap},
        attrs={'is_lod': is_lod,
               'out_val_if_empty': out_val_if_empty})

    return [out, loss_weight]


def unstack(x, axis=0, num=None):
    """
    :alias_main: paddle.unstack
	:alias: paddle.unstack,paddle.tensor.unstack,paddle.tensor.manipulation.unstack
	:old_api: paddle.fluid.layers.unstack

    **UnStack Layer**

    This layer unstacks input Tensor :code:`x` into several Tensors along :code:`axis`.

    If :code:`axis` < 0, it would be replaced with :code:`axis+rank(x)`.
    If :code:`num` is None, it would be inferred from :code:`x.shape[axis]`,
    and if :code:`x.shape[axis]` <= 0 or is unknown, :code:`ValueError` is
    raised.

    Args:
        x (Tensor): Input Tensor. It is a N-D Tensors of data types float32, float64, int32, int64.
        axis (int): The axis along which the input is unstacked.
        num (int|None): The number of output variables.

    Returns:
        list(Tensor): The unstacked Tensors list. The list elements are N-D Tensors of data types float32, float64, int32, int64.

    Raises:
        ValueError: If x.shape[axis] <= 0 or axis is not in range [-D, D).

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.ones(name='x', shape=[2, 3, 5], dtype='float32')  # create a tensor with shape=[2, 3, 5]
            y = paddle.unstack(x, axis=1)  # unstack with second axis, which results 3 tensors with shape=[2, 5]

    """
    if in_dygraph_mode():
        if num == None:
            num = x.shape[axis]
        if num == 0:
            return []
        return _C_ops.unstack(x, num, 'axis', int(axis), 'num', num)

    helper = LayerHelper('unstack', **locals())
    if num is None:
        if axis is None or x.shape[axis] <= 0:
            raise ValueError('unknown unstack number')
        else:
            num = x.shape[axis]

    outs = []
    for _ in range(num):
        outs.append(helper.create_variable_for_type_inference(x.dtype))

    helper.append_op(
        type='unstack',
        inputs={'X': [x]},
        outputs={'Y': outs},
        attrs={'axis': axis,
               'num': num})
    return outs


@deprecated(since='2.0.0', update_to="paddle.expand")
def expand(x, expand_times, name=None):
    """
    :alias_main: paddle.expand
	:alias: paddle.expand,paddle.tensor.expand,paddle.tensor.manipulation.expand
	:old_api: paddle.fluid.layers.expand

    This operation tiles ``x`` multiple times according to the parameter ``expand_times``.
    The times number for each dimension of ``x`` is set by the parameter ``expand_times``.
    The rank of ``x`` should be less than or equal to 6. Please note that size of ``expand_times`` must be the same
    with X's rank. Following is a using case:


    .. code-block:: text

        Input(X) is a 3-D tensor with shape [2, 3, 1]:

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        Attr(expand_times):  [1, 2, 2]

        Output(Out) is a 3-D tensor with shape [2, 6, 2]:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]

    Args:
        x (Variable): A ``Tensor`` or ``LoDTensor`` with dimension in [1, 6]. The data type is ``bool``, ``float32``, ``float64`` or ``int32`` .
        expand_times (list|tuple|Variable): The data type is ``int32`` . If ``expand_times`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``expand_times`` is an Variable, it should be an 1-D Tensor.
                Expand times number for each dimension of ``x`` .
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``x``. After expanding, size of each dimension of output is equal to the size of the corresponding dimension of ``x`` multiplying the corresponding value given by ``expand_times`` .

    Raises:
        TypeError: The type of ``expand_times`` must be list, tuple or Variable.
        ValueError: The elements of ``expand_times`` cannot be negative.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # example 1:
            data_1 = fluid.layers.fill_constant(shape=[2, 3, 1], dtype='int32', value=0)
            expanded_1 = fluid.layers.expand(data_1, expand_times=[1, 2, 2])
            # the shape of expanded_1 is [2, 6, 2].

            # example 2:
            data_2 = fluid.layers.fill_constant(shape=[12, 14], dtype="int32", value=3)
            expand_times = fluid.layers.fill_constant(shape=[2], dtype="int32", value=4)
            expanded_2 = fluid.layers.expand(data_2, expand_times=expand_times)
            # the shape of expanded_2 is [48, 56].
    """
    if in_dygraph_mode():
        attrs = ()
        expand_times_tensor = None
        if isinstance(expand_times, (list, tuple)):
            expand_times = [
                item.numpy().item(0) if isinstance(item, Variable) else item
                for item in expand_times
            ]
            attrs += ('expand_times', expand_times)
        elif isinstance(expand_times, Variable):
            expand_times_tensor = expand_times
            expand_times_tensor.stop_gradient = True

        return _C_ops.expand(x, expand_times_tensor, *attrs)

    inputs = {"X": [x]}
    attrs = {}
    check_variable_and_dtype(
        x, 'x', ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
        'expand')
    check_type(expand_times, 'expand_times', (list, tuple, Variable), 'expand')
    if convert_dtype(x.dtype) == 'bool' and x.stop_gradient == True:
        raise ValueError(
            "expand op bool date type must set the stop_gradient to be False")

    helper = LayerHelper('expand', input=x, **locals())

    def get_attr_expand_times(list_expand_times):
        attrs_expand_times = []
        for idx, times in enumerate(list_expand_times):
            if isinstance(times, Variable):
                attrs_expand_times.append(-1)
            else:
                attrs_expand_times.append(times)
                assert times > 0, (
                    "Each element given in expand_times must not be negative.")
        return attrs_expand_times

    if isinstance(expand_times, Variable):
        expand_times.stop_gradient = True
        inputs['ExpandTimes'] = expand_times
    elif isinstance(expand_times, (list, tuple)):
        attrs['expand_times'] = get_attr_expand_times(expand_times)
        if utils._contain_var(expand_times):
            inputs['expand_times_tensor'] = utils._convert_to_tensor_list(
                expand_times)

    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='expand', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


@deprecated(since='2.0.0', update_to="paddle.expand_as")
def expand_as(x, target_tensor, name=None):
    """
    :alias_main: paddle.expand_as
	:alias: paddle.expand_as,paddle.tensor.expand_as,paddle.tensor.manipulation.expand_as
	:old_api: paddle.fluid.layers.expand_as
    
    expand_as operator tiles to the input by given expand tensor. You should set expand tensor
    for each dimension by providing tensor 'target_tensor'. The rank of X
    should be in [1, 6]. Please note that size of 'target_tensor' must be the same
    with X's rank. Following is a using case:


    .. code-block:: text

        Input(X) is a 3-D tensor with shape [2, 3, 1]:

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        target_tensor's shape:  [2, 6, 2]

        Output(Out) is a 3-D tensor with shape [2, 6, 2]:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]


    Args:
        x (Variable): A Tensor with dtype float64, float32, int32.
        A tensor with rank in [1, 6].
        target_tensor (Variable): A Tensor with dtype float64, float32, int32.
        target_tensor for expanding to Input(X). Only use target_tensor'shape.

    Returns:
        Variable: A Tensor with dtype float64, float32, int32.
        After expanding, size of each dimension of Output(Out) is equal to the size
        of the corresponding dimension of target_tensor multiplying the corresponding
        value given by target_tensor.


    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            paddle.enable_static()

            data = fluid.layers.data(name="data", shape=[-1,10], dtype='float64')
            target_tensor = fluid.layers.data(
              name="target_tensor", shape=[-1,20], dtype='float64')
            result = fluid.layers.expand_as(x=data, target_tensor=target_tensor)
            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            x = np.random.rand(3,10)
            y = np.random.rand(3,20)
            output= exe.run(feed={"data":x,"target_tensor":y},fetch_list=[result.name])
            print(output[0].shape)
            #(3,20)

    """
    if in_dygraph_mode():
        return _C_ops.expand_as(x, target_tensor)

    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64', 'bool'], 'expand_as')
    check_variable_and_dtype(target_tensor, 'target_tensor',
                             ['float32', 'float64', 'int32', 'int64', 'bool'],
                             'expand_as')
    helper = LayerHelper('expand_as', input=x, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    inputs = {'X': x, 'target_tensor': target_tensor}
    helper.append_op(type='expand_as', inputs=inputs, outputs={'Out': out})
    return out


from paddle.fluid.framework import convert_np_dtype_to_dtype_


@deprecated(since='1.8.0', update_to="paddle.uniform")
@templatedoc()
def uniform_random_batch_size_like(input,
                                   shape,
                                   dtype='float32',
                                   input_dim_idx=0,
                                   output_dim_idx=0,
                                   min=-1.0,
                                   max=1.0,
                                   seed=0):
    """
    This OP initializes a variable with random values sampled from a
    uniform distribution in the range [min, max). The input_dim_idx used to get the input dimension value which will be used to resize the output dimension.

    .. code-block:: text

        *Case 1:

            Given:
                input =[[0.946741  , 0.1357001 , 0.38086128]]    # input.shape=[1,3]
                shape=[2,4]

            result.shape[output_dim_idx] = input.shape[input_dim_idx],
            output_dim_idx = 0,
            input_dim_idx = 0,
            result.shape[0] = input.shape[0],
            then:
                result=[[ 0.3443427 , -0.23056602,  0.3477049 ,  0.06139076]]    # result.shape=[1,4]

       *Case 2:

           Given:
               input =[[0.946741  , 0.1357001 , 0.38086128]]     # input.shape=[1,3]
               shape=[2,4]
               input_dim_idx=1
               output_dim_idx=1

           result.shape[output_dim_idx] = input.shape[input_dim_idx],
           output_dim_idx = 1,
           input_dim_idx = 1,
           result.shape[1] = input.shape[1],
           then:
               result=[[-0.23133647, -0.84195036,  0.21441269],
                       [-0.08774924,  0.25605237, -0.09403259]]    # result.shape=[2,3]
    Args:
        input (Variable): A Tensor. Supported data types: float32, float64.
        shape (tuple|list): A python list or python tuple. The shape of the output Tensor, the data type is int.
        input_dim_idx (int, optional): An index used to get the input dimension value which will be used to resize the output dimension. Default  0.
        output_dim_idx (int, optional): An index used to indicate the specific dimension that will be replaced by corresponding input dimension value. Default 0.
        min (float, optional): The lower bound on the range of random values to generate, the min is included in the range. Default -1.0.
        max (float, optional): The upper bound on the range of random values to generate, the max is excluded in the range. Default 1.0.
        seed (int, optional):  Random seed used for generating samples. 0 means use a seed generated by the system.Note that if seed is not 0, this operator will always generate the same random numbers every time.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type of output Tensor. Supported data types: float32, float64. Default float32.
    Returns:
        Variable: A Tensor of the specified shape filled with uniform_random values. The shape of the Tensor is determined by the shape parameter and the specified dimension of the input Tensor.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            # example 1:
            input = fluid.data(name="input", shape=[1, 3], dtype='float32')
            out_1 = fluid.layers.uniform_random_batch_size_like(input, [2, 4]) # out_1.shape=[1, 4]

            # example 2:
            out_2 = fluid.layers.uniform_random_batch_size_like(input, [2, 4], input_dim_idx=1, output_dim_idx=1) # out_2.shape=[2, 3]


    """
    check_variable_and_dtype(input, 'Input', ("float32", 'float64', "uint16"),
                             'uniform_random_batch_size_like')
    check_type(shape, 'shape', (list, tuple), 'uniform_random_batch_size_like')
    check_dtype(dtype, 'dtype', ('float32', 'float64', "uint16"),
                'uniform_random_batch_size_like')

    helper = LayerHelper('uniform_random_batch_size_like', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='uniform_random_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx,
            'min': min,
            'max': max,
            'seed': seed,
            'dtype': c_dtype
        })

    return out


@deprecated(since="2.0.0", update_to="paddle.normal")
@templatedoc()
def gaussian_random(shape,
                    mean=0.0,
                    std=1.0,
                    seed=0,
                    dtype='float32',
                    name=None):
    """
    This OP returns a Tensor filled with random values sampled from a Gaussian
    distribution, with ``shape`` and ``dtype``.

    Args:
        shape(list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        mean(float|int, optional): Mean of the output tensor, default is 0.0.
        std(float|int, optional): Standard deviation of the output tensor, default
            is 1.0.
        seed(int, optional): ${seed_comment}
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of
            the output Tensor. Supported data types: float32, float64.
            Default is float32.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a Gaussian
        distribution, with ``shape`` and ``dtype``.

    Examples:
       .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            # example 1:
            # attr shape is a list which doesn't contain Tensor.
            result_1 = fluid.layers.gaussian_random(shape=[3, 4])
            # [[-0.31261674,  1.8736548,  -0.6274357,   0.96988016],
            #  [-0.12294637,  0.9554768,   1.5690808,  -1.2894802 ],
            #  [-0.60082096, -0.61138713,  1.5345167,  -0.21834975]]

            # example 2:
            # attr shape is a list which contains Tensor.
            dim_1 = fluid.layers.fill_constant([1], "int64", 2)
            dim_2 = fluid.layers.fill_constant([1], "int32", 3)
            result_2 = fluid.layers.gaussian_random(shape=[dim_1, dim_2])
            # [[ 0.51398206, -0.3389769,   0.23597084],
            #  [ 1.0388143,  -1.2015356,  -1.0499583 ]]

            # example 3:
            # attr shape is a Tensor, the data type must be int64 or int32.
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = fluid.layers.gaussian_random(var_shape)
            # if var_shape's value is [2, 3]
            # result_3 is:
            # [[-0.12310527,  0.8187662,   1.923219  ]
            #  [ 0.70721835,  0.5210541,  -0.03214082]]
       
       .. code-block:: python
       
           # declarative mode
           # required: skiptest
           import numpy as np
           from paddle import fluid
   
           x = fluid.layers.gaussian_random((2, 3), std=2., seed=10)
   
           place = fluid.CPUPlace()
           exe = fluid.Executor(place)
           start = fluid.default_startup_program()
           main = fluid.default_main_program()
   
           exe.run(start)
           x_np, = exe.run(main, feed={}, fetch_list=[x])

           x_np
           # array([[2.3060477, 2.676496 , 3.9911983],
           #        [0.9990833, 2.8675377, 2.2279181]], dtype=float32)

       .. code-block:: python

           # imperative mode
           import numpy as np
           from paddle import fluid
           import paddle.fluid.dygraph as dg
    
           place = fluid.CPUPlace()
           with dg.guard(place) as g:
               x = fluid.layers.gaussian_random((2, 4), mean=2., dtype="float32", seed=10)
               x_np = x.numpy()       
           x_np
           # array([[2.3060477 , 2.676496  , 3.9911983 , 0.9990833 ],
           #        [2.8675377 , 2.2279181 , 0.79029655, 2.8447366 ]], dtype=float32)
    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils.convert_shape_to_list(shape)
        return _C_ops.gaussian_random('shape', shape, 'mean',
                                      float(mean), 'std',
                                      float(std), 'seed', seed, 'dtype', dtype)

    check_type(shape, 'shape', (list, tuple, Variable), 'gaussian_random/randn')
    check_dtype(dtype, 'dtype', ['float32', 'float64'], 'gaussian_random/randn')

    inputs = {}
    attrs = {
        'mean': mean,
        'std': std,
        'seed': seed,
        'dtype': dtype,
        'use_mkldnn': False
    }
    utils.get_shape_tensor_inputs(
        inputs=inputs,
        attrs=attrs,
        shape=shape,
        op_type='gaussian_random/randn')

    helper = LayerHelper('gaussian_random', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='gaussian_random',
        inputs=inputs,
        outputs={'Out': out},
        attrs=attrs)

    return out


@templatedoc()
def sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32'):
    """
    This op is used for sampling id from multinomial distribution from the input, sampling one id for one sample.

    Parameters:
        x (Variable): 2-D tensor, [batch_size, input_feature_dimensions]
        min (Float): minimum , default 0.0.
        max (Float): maximum, default 1.0.
        seed (Float): Random seed, default 0. if seed is not 0, will generate same number every time.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of output data : float32, float_16, int etc

    Returns:
        Variable: sampling tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(
                name="X",
                shape=[13, 11],
                dtype='float32')

            out = fluid.layers.sampling_id(x)
    """

    helper = LayerHelper('sampling_id', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sampling_id',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'min': min,
               'max': max,
               'seed': seed})

    return out


@deprecated(since='1.8.0', update_to="paddle.normal")
@templatedoc()
def gaussian_random_batch_size_like(input,
                                    shape,
                                    input_dim_idx=0,
                                    output_dim_idx=0,
                                    mean=0.0,
                                    std=1.0,
                                    seed=0,
                                    dtype='float32'):
    """
    ${comment}

    Args:
        input (Variable): ${input_comment}
        shape (tuple|list): ${shape_comment}
        input_dim_idx (int): ${input_dim_idx_comment}
        output_dim_idx (int): ${output_dim_idx_comment}
        mean (float): ${mean_comment}
        std (float): ${std_comment}
        seed (int): ${seed_comment}
        dtype(np.dtype|core.VarDesc.VarType|str): The type of output data, float32 or float_64.

    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            input = fluid.data(name="input", shape=[13, 11], dtype='float32')

            out = fluid.layers.gaussian_random_batch_size_like(
                input, shape=[-1, 11], mean=1.0, std=2.0)
    """

    helper = LayerHelper('gaussian_random_batch_size_like', **locals())
    check_type(input, 'input', (Variable),
               'fluid.layers.gaussian_random_batch_size_like')
    check_type(shape, 'shape', (list, tuple),
               'fluid.layers.gaussian_random_batch_size_like')
    check_dtype(dtype, 'dtype', ['float16', 'float32', 'int'],
                'fluid.layers.gaussian_random_batch_size_like')
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='gaussian_random_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx,
            'mean': mean,
            'std': std,
            'seed': seed,
            'dtype': c_dtype
        })

    return out


@templatedoc()
def sum(x):
    """
    ${comment}

    Case 1:
    ::
        Input:
            Input. Shape = [2, 3]
            Input = [[1, 2, 3],
                     [4, 5, 6]]

        Output:
            The output. Shape = [2, 3]
            Output = [[1, 2, 3],
                      [4, 5, 6]]

    Case 2:
    ::
        Input:
            First input:
            Input1. Shape = [2, 3]
            Input1 = [[1, 2, 3],
                      [4, 5, 6]]

        The second input:
            Input2. Shape = [2, 3]
            Input2 = [[7, 8, 9],
                      [10, 11, 12]]

        Output:
            The output. Shape = [2, 3]
            Output = [[8, 10, 12],
                      [14, 16, 18]]

    Args:
        x (Variable|list(Variable)): ${x_comment}

    Returns:
        Variable: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input0 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=5)
            input1 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=3)
            sum = fluid.layers.sum([input0, input1])

            # You can print out 'sum' via executor.
            out = fluid.layers.Print(sum, message="the sum of input0 and input1: ")
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_main_program())

            # The printed result is:
            # 1570701754	the sum of input0 and input1: 	The place is:CPUPlace
            # Tensor[sum_0.tmp_0]
            #    shape: [2,3,]
            #    dtype: l
            #    data: 8,8,8,8,8,8,

            # the sum of input0 and input1 is 2-D Tensor with shape [2,3].
            # dtype is the corresponding C++ data type, which may vary in different environments.
            # Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t,
            #       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux,
            #       and '__int64' on Windows. They both represent 64-bit integer variables.
    """

    return paddle.add_n(x)


@templatedoc()
def slice(input, axes, starts, ends):
    """
    This operator produces a slice of ``input`` along multiple axes. Similar to numpy:
    https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slice uses ``axes``, ``starts`` and ``ends`` attributes to specify the start and
    end dimension for each axis in the list of axes and Slice uses this information
    to slice the input data tensor. If a negative value is passed to
    ``starts`` or ``ends`` such as :math:`-i`,  it represents the reverse position of the
    axis :math:`i-1` (here 0 is the initial position).
    If the value passed to ``starts`` or ``ends`` is greater than n
    (the number of elements in this dimension), it represents n.
    For slicing to the end of a dimension with unknown size, it is recommended
    to pass in INT_MAX. The size of ``axes`` must be equal to ``starts`` and ``ends``.
    Following examples will explain how slice works:

    .. code-block:: text

        Case1:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [1, 0]
                ends = [2, 3]
            Then:
                result = [ [5, 6, 7], ]

        Case2:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [-1, 1000]       # -1 denotes the reverse 0th position of dimension 0.
            Then:
                result = [ [2, 3, 4], ] # result = data[0:1, 1:4]
    
    Args:
        input (Tensor): A ``Tensor`` . The data type is ``float16``, ``float32``, ``float64``, ``int32`` or ``int64``.
        axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to .
        starts (list|tuple|Tensor): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``starts`` is an Tensor, it should be an 1-D Tensor.
                It represents starting indices of corresponding axis in ``axes``.
        ends (list|tuple|Tensor): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``ends`` is an Tensor, it should be an 1-D Tensor .
                It represents ending indices of corresponding axis in ``axes``.

    Returns:
        Tensor:  A ``Tensor``. The data type is same as ``input``.

    Raises:
        TypeError: The type of ``starts`` must be list, tuple or Tensor.
        TypeError: The type of ``ends`` must be list, tuple or Tensor.

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.rand(shape=[4, 5, 6], dtype='float32')
            # example 1:
            # attr starts is a list which doesn't contain tensor.
            axes = [0, 1, 2]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)
            # sliced_1 is input[0:3, 0:2, 2:4].

            # example 2:
            # attr starts is a list which contain tensor.
            minus_3 = paddle.full([1], -3, "int32")
            sliced_2 = paddle.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)
            # sliced_2 is input[0:3, 0:2, 2:4].
    """
    if in_dygraph_mode():
        attrs = ()
        starts_tensor = None
        ends_tensor = None

        if isinstance(axes, (list, tuple)):
            axes = list(axes)
            if len(axes) == 0:
                raise ValueError(
                    "Input axes should not be an empty list/tuple.")
            for i in range(len(axes)):
                if axes[i] < 0:
                    axes[i] = max(0, axes[i] + len(input.shape))
                else:
                    axes[i] = min(len(input.shape) - 1, axes[i])

        else:
            raise ValueError(
                "Input axes must be a python list or tuple, but reveived {}".
                format(type(axes)))

        infer_flags = list(1 for i in range(len(axes)))

        if isinstance(starts, (list, tuple)):
            starts = [
                item.numpy().item(0) if isinstance(item, Variable) else item
                for item in starts
            ]
            attrs += ('starts', starts)
        elif isinstance(starts, Variable):
            starts_tensor = starts
            starts.stop_gradient = True
            infer_flags = list(-1 for i in range(len(axes)))

        if isinstance(ends, (list, tuple)):
            ends = [
                item.numpy().item(0) if isinstance(item, Variable) else item
                for item in ends
            ]
            attrs += ('ends', ends)
        elif isinstance(ends, Variable):
            ends_tensor = ends
            ends_tensor.stop_gradient = True
            infer_flags = list(-1 for i in range(len(axes)))

        return _C_ops.slice(input, starts_tensor, ends_tensor, 'axes', axes,
                            'infer_flags', infer_flags, *attrs)

    if not isinstance(starts, (list, tuple, Variable)):
        raise ValueError(
            "Input starts must be an Variable, python list or tuple.")
    if not isinstance(ends, (list, tuple, Variable)):
        raise ValueError(
            "Input ends must be an Variable, python list or tuple.")

    helper = LayerHelper('slice', **locals())

    inputs = {'Input': input}
    attrs = {'axes': axes}
    infer_flags = list(1 for i in range(len(axes)))

    # starts
    if isinstance(starts, Variable):
        starts.stop_gradient = True
        inputs['StartsTensor'] = starts
        infer_flags = list(-1 for i in range(len(axes)))
    elif isinstance(starts, (list, tuple)):
        attrs['starts'] = []
        if utils._contain_var(starts):
            inputs['StartsTensorList'] = utils._convert_to_tensor_list(starts)
            for i, dim in enumerate(starts):
                if isinstance(dim, Variable):
                    attrs['starts'].append(-1)
                    infer_flags[i] = -1
                else:
                    attrs['starts'].append(dim)
        else:
            attrs['starts'] = starts

    # ends
    if isinstance(ends, Variable):
        ends.stop_gradient = True
        inputs['EndsTensor'] = ends
        infer_flags = list(-1 for i in range(len(axes)))
    elif isinstance(ends, (list, tuple)):
        attrs['ends'] = []
        if utils._contain_var(ends):
            inputs['EndsTensorList'] = utils._convert_to_tensor_list(ends)
            for i, dim in enumerate(ends):
                if isinstance(dim, Variable):
                    attrs['ends'].append(-1)
                    infer_flags[i] = -1
                else:
                    attrs['ends'].append(dim)
        else:
            attrs['ends'] = ends

    # infer_flags
    attrs['infer_flags'] = infer_flags
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('input'))
    helper.append_op(
        type='slice', inputs=inputs, attrs=attrs, outputs={'Out': out})

    return out


@deprecated(since='2.0.0', update_to="paddle.strided_slice")
def strided_slice(input, axes, starts, ends, strides):
    """
    :alias_main: paddle.strided_slice
	:alias: paddle.strided_slice,paddle.tensor.strided_slice,paddle.tensor.manipulation.strided_slice
	:old_api: paddle.fluid.layers.strided_slice

    This operator produces a slice of ``input`` along multiple axes. Similar to numpy:
    https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slice uses ``axes``, ``starts`` and ``ends`` attributes to specify the start and
    end dimension for each axis in the list of axes and Slice uses this information
    to slice the input data tensor. If a negative value is passed to
    ``starts`` or ``ends`` such as :math:`-i`,  it represents the reverse position of the
    axis :math:`i-1` th(here 0 is the initial position). The ``strides`` represents steps of
    slicing and if the ``strides`` is negative, slice operation is in the opposite direction.
    If the value passed to ``starts`` or ``ends`` is greater than n
    (the number of elements in this dimension), it represents n.
    For slicing to the end of a dimension with unknown size, it is recommended
    to pass in INT_MAX. The size of ``axes`` must be equal to ``starts`` , ``ends`` and ``strides``.
    Following examples will explain how strided_slice works:

    .. code-block:: text

        Case1:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [1, 0]
                ends = [2, 3]
                strides = [1, 1]
            Then:
                result = [ [5, 6, 7], ]

        Case2:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [2, 0]
                strides = [1, -1]
            Then:
                result = [ [8, 7, 6], ]

        Case3:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [-1, 1000]
                strides = [1, 3]
            Then:
                result = [ [2], ]
    Args:
        input (Variable): An N-D ``Tensor`` or ``LoDTensor`` . The data type is ``bool``, ``float32``, ``float64``, ``int32`` or ``int64``.
        axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to.
                            It's optional. If it is not provides, it will be treated as :math:`[0,1,...,len(starts)-1]`.
        starts (list|tuple|Variable): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``starts`` is an Variable, it should be an 1-D Tensor.
                It represents starting indices of corresponding axis in ``axes``.
        ends (list|tuple|Variable): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``ends`` is an Variable, it should be an 1-D Tensor .
                It represents ending indices of corresponding axis in ``axes``.
        strides (list|tuple|Variable): The data type is ``int32`` . If ``strides`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``strides`` is an Variable, it should be an 1-D Tensor .
                It represents slice step of corresponding axis in ``axes``.

    Returns:
        Variable:  A ``Tensor`` or ``LoDTensor`` with the same dimension as ``input``. The data type is same as ``input``.

    Raises:
        TypeError: The type of ``starts`` must be list, tuple or Variable.
        TypeError: The type of ``ends`` must be list, tuple or Variable.
        TypeError: The type of ``strides`` must be list, tuple or Variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle

            paddle.enable_static()
            input = fluid.data(
                name="input", shape=[3, 4, 5, 6], dtype='float32')

            # example 1:
            # attr starts is a list which doesn't contain tensor Variable.
            axes = [0, 1, 2]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            strides_1 = [1, 1, 1]
            strides_2 = [1, 1, 2]
            sliced_1 = fluid.layers.strided_slice(input, axes=axes, starts=starts, ends=ends, strides=strides_1)
            # sliced_1 is input[:, 0:3:1, 0:2:1, 2:4:1].


            # example 2:
            # attr starts is a list which contain tensor Variable.
            minus_3 = fluid.layers.fill_constant([1], "int32", -3)
            sliced_2 = fluid.layers.strided_slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)
            # sliced_2 is input[:, 0:3:1, 0:2:1, 2:4:2].
    """
    helper = LayerHelper('strided_slice', **locals())

    check_variable_and_dtype(input, 'input',
                             ['bool', 'float32', 'float64', 'int32', 'int64'],
                             'strided_slice')
    check_type(axes, 'axes', (list, tuple), 'strided_slice')
    check_type(starts, 'starts', (list, tuple, Variable), 'strided_slice')
    check_type(ends, 'ends', (list, tuple, Variable), 'strided_slice')
    check_type(strides, 'strides', (list, tuple, Variable), 'strided_slice')

    def check_list_elements_dtype(list_input, input_name):
        if isinstance(list_input, Variable):
            check_dtype(list_input.dtype, input_name, ['int32'],
                        'strided_slice')
        else:
            for i, var in enumerate(list_input):
                var_name = input_name + '[' + str(i) + ']'
                if isinstance(var, Variable):
                    check_dtype(var.dtype, var_name, ['int32'], 'strided_slice')

    check_list_elements_dtype(axes, 'axes')
    check_list_elements_dtype(starts, 'starts')
    check_list_elements_dtype(ends, 'ends')
    check_list_elements_dtype(strides, 'strides')

    def get_new_list_tensor(old_list):
        new_list_tensor = []
        for dim in old_list:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_list_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_list_tensor.append(temp_out)
        return new_list_tensor

    inputs = {'Input': input}
    attrs = {'axes': axes}
    infer_flags = list(1 for i in range(len(axes)))

    if in_dygraph_mode():
        inputs = {'Input': input}
        attrs = {
            'axes': axes,
            'starts': starts,
            'ends': ends,
            'strides': strides,
            'infer_flags': infer_flags
        }
    else:
        # starts
        if isinstance(starts, Variable):
            starts.stop_gradient = True
            inputs['StartsTensor'] = starts
        elif isinstance(starts, (list, tuple)):
            attrs['starts'] = []
            if utils._contain_var(starts):
                inputs['StartsTensorList'] = get_new_list_tensor(starts)
                for i, dim in enumerate(starts):
                    if isinstance(dim, Variable):
                        attrs['starts'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['starts'].append(dim)
            else:
                attrs['starts'] = starts

        # ends
        if isinstance(ends, Variable):
            ends.stop_gradient = True
            inputs['EndsTensor'] = ends
        elif isinstance(ends, (list, tuple)):
            attrs['ends'] = []
            if utils._contain_var(ends):
                inputs['EndsTensorList'] = get_new_list_tensor(ends)
                for i, dim in enumerate(ends):
                    if isinstance(dim, Variable):
                        attrs['ends'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['ends'].append(dim)
            else:
                attrs['ends'] = ends

        # strides
        if isinstance(strides, Variable):
            strides.stop_gradient = True
            inputs['StridesTensor'] = strides
        elif isinstance(strides, (list, tuple)):
            attrs['strides'] = []
            if utils._contain_var(strides):
                inputs['StridesTensorList'] = get_new_list_tensor(strides)
                for i, dim in enumerate(strides):
                    if isinstance(dim, Variable):
                        attrs['strides'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['strides'].append(dim)
            else:
                attrs['strides'] = strides
        attrs['infer_flags'] = infer_flags
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('input'))
    helper.append_op(
        type='strided_slice', inputs=inputs, attrs=attrs, outputs={'Out': out})

    return out


def shape(input):
    """
    :alias_main: paddle.shape
	:alias: paddle.shape,paddle.tensor.shape,paddle.tensor.attribute.shape
	:old_api: paddle.fluid.layers.shape

    **Shape Layer**

    Get the shape of the input.

    .. code-block:: text

        Case1:
            Given N-D Tensor:
                input = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

            Then:
                input.shape = [2, 4]

        Case2:
            Given SelectedRows:
                input.rows = [0, 4, 19]
                input.height = 20
                input.value = [ [1, 2], [3, 4], [5, 6] ]  # inner tensor
            Then:
                input.shape = [3, 2]

    Args:
        input (Variable): The input can be N-D Tensor or SelectedRows with data type bool, float16, float32, float64, int32, int64.
                          If input variable is type of SelectedRows, returns the shape of it's inner tensor.

    Returns:
        Variable (Tensor): The shape of the input variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            paddle.enable_static()

            inputs = fluid.data(name="x", shape=[3, 100, 100], dtype="float32")
            output = fluid.layers.shape(inputs)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            img = np.ones((3, 100, 100)).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
            print(res) # [array([  3, 100, 100], dtype=int32)]
    """
    if in_dygraph_mode():
        out = _C_ops.shape(input)
        out.stop_gradient = True
        return out

    check_variable_and_dtype(input, 'input', [
        'bool', 'float16', 'float32', 'float64', 'int32', 'int64', 'complex64',
        'complex128'
    ], 'shape')
    helper = LayerHelper('shape', **locals())
    out = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type='shape',
        inputs={'Input': input},
        outputs={'Out': out},
        stop_gradient=True)

    return out


def rank(input):
    """

    The OP returns the number of dimensions for a tensor, which is a 0-D int32 Tensor.

    Args:
        input (Tensor): The input N-D tensor with shape of :math:`[N_1, N_2, ..., N_k]`, the data type is arbitrary.

    Returns:
        Tensor, the output data type is int32.: The 0-D tensor with the dimensions of the input Tensor.

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.rand((3, 100, 100))
            rank = paddle.rank(input)
            print(rank)
            # 3
    """
    check_type(input, 'input', (Variable), 'input')
    ndims = len(input.shape)
    out = assign(np.array(ndims, 'int32'))

    return out


@deprecated(since="2.0.0", update_to="paddle.numel")
def size(input):
    """
    **Size Layer**

    Returns the number of elements for a tensor, which is a int64 Tensor with shape [1].

    Args:
        input (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, int32, int64.

    Returns:
        Tensor: The number of elements for the input Tensor.

    Raises:
        TypeError: ``input`` must be a Tensor and the data type of ``input`` must be one of bool, float16, float32, float64, int32, int64.
    
    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid.layers as layers
            paddle.enable_static()

            input = layers.data(
                name="input", shape=[3, 100], dtype="float32", append_batch_size=False)
            rank = layers.size(input) # 300
    """

    if in_dygraph_mode():
        return _C_ops.size(input)
    check_variable_and_dtype(
        input, 'input',
        ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'], "size")
    helper = LayerHelper('size', **locals())
    out = helper.create_variable_for_type_inference(dtype='int64')
    helper.append_op(type='size', inputs={'Input': input}, outputs={'Out': out})

    return out


def _elementwise_op(helper):
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)
    check_variable_and_dtype(
        x, 'x', ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
        op_type)
    check_variable_and_dtype(
        y, 'y', ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
        op_type)

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type=op_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis,
               'use_mkldnn': use_mkldnn})
    return helper.append_activation(out)


def scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Scale operator.

    Putting scale and bias to the input Tensor as following:

    ``bias_after_scale`` is True:

    .. math::
                            Out=scale*X+bias

    ``bias_after_scale`` is False:

    .. math::
                            Out=scale*(X+bias)

    Args:
        x(Tensor): Input N-D Tensor of scale operator. Data type can be float32, float64, int8, int16, int32, int64, uint8.
        scale(float|Tensor): The scale factor of the input, it should be a float number or a Tensor with shape [1] and data type as float32.
        bias(float): The bias to be put on the input.
        bias_after_scale(bool): Apply bias addition after or before scaling. It is useful for numeric stability in some circumstances.
        act(str, optional): Activation applied to the output such as tanh, softmax, sigmoid, relu.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: Output tensor of scale operator, with shape and data type same as input.

    Examples:
        .. code-block:: python
            
            # scale as a float32 number
            import paddle

            data = paddle.randn(shape=[2,3], dtype='float32')
            res = paddle.scale(data, scale=2.0, bias=1.0)

        .. code-block:: python

            # scale with parameter scale as a Tensor
            import paddle

            data = paddle.randn(shape=[2, 3], dtype='float32')
            factor = paddle.to_tensor([2], dtype='float32')
            res = paddle.scale(data, scale=factor, bias=1.0)

    """

    if in_dygraph_mode():
        _scale = scale.numpy().item(0) if isinstance(scale, Variable) else scale
        out = _C_ops.scale(x, 'scale',
                           float(_scale), 'bias',
                           float(bias), 'bias_after_scale', bias_after_scale)
        return dygraph_utils._append_activation_in_dygraph(out)

    check_variable_and_dtype(x, "x", [
        'float16', 'uint16', 'float32', 'float64', 'int8', 'int16', 'int32',
        'int64', 'uint8'
    ], "scale")
    inputs = {'X': [x]}
    attrs = {
        'bias': float(bias),
        'bias_after_scale': bias_after_scale,
    }
    if isinstance(scale, Variable):
        inputs['ScaleTensor'] = [scale]
    else:
        attrs['scale'] = float(scale)
    helper = LayerHelper('scale', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='scale', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return helper.append_activation(out)


def elementwise_add(x, y, axis=-1, act=None, name=None):
    """

Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle
        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_add(x, y)
        # z = x + y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # [3., 8., 6.]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_add(x, y, axis=1)
        # z = x + y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = fluid.layers.elementwise_add(x, y, axis=3)
        # z = x + y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) # z.shape=[2,3,4,5]

    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x,
            y,
            axis=axis,
            act=act,
            op_name='elementwise_add',
            use_mkldnn=_global_flags()["FLAGS_use_mkldnn"])

    return _elementwise_op(LayerHelper('elementwise_add', **locals()))


@deprecated(since="2.0.0", update_to="paddle.divide")
def elementwise_div(x, y, axis=-1, act=None, name=None):
    """

Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_div(x, y)
        # z = x / y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # [2., 0.6, 2.]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_div(x, y, axis=1)
        # z = x / y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = fluid.layers.elementwise_div(x, y, axis=3)
        # z = x / y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) # z.shape=[2,3,4,5]

    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_div')

    return _elementwise_op(LayerHelper('elementwise_div', **locals()))


def elementwise_sub(x, y, axis=-1, act=None, name=None):
    """

Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_sub(x, y)
        # z = x - y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # [1., -2., 2.]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_sub(x, y, axis=1)
        # z = x - y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = fluid.layers.elementwise_sub(x, y, axis=3)
        # z = x - y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) # z.shape=[2,3,4,5]

    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_sub')

    return _elementwise_op(LayerHelper('elementwise_sub', **locals()))


@deprecated(since="2.0.0", update_to="paddle.multiply")
def elementwise_mul(x, y, axis=-1, act=None, name=None):
    """

Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_mul(x, y)
        # z = x * y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # [2., 15., 8.]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_mul(x, y, axis=1)
        # z = x * y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = fluid.layers.elementwise_mul(x, y, axis=3)
        # z = x * y

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) # z.shape=[2,3,4,5]

    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_mul')

    return _elementwise_op(LayerHelper('elementwise_mul', **locals()))


def elementwise_max(x, y, axis=-1, act=None, name=None):
    """
    :alias_main: paddle.elementwise_max
	:alias: paddle.elementwise_max,paddle.tensor.elementwise_max,paddle.tensor.math.elementwise_max
	:old_api: paddle.fluid.layers.elementwise_max

Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_max(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[2, 5, 4]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_max(x, y, axis=1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value)#[[[[1., 1., 1., 1., 1.] .... [1., 1., 1., 1., 1.]]]]

    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_max')

    return _elementwise_op(LayerHelper('elementwise_max', **locals()))


def elementwise_min(x, y, axis=-1, act=None, name=None):
    """
    :alias_main: paddle.elementwise_min
	:alias: paddle.elementwise_min,paddle.tensor.elementwise_min,paddle.tensor.math.elementwise_min
	:old_api: paddle.fluid.layers.elementwise_min

Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_min(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[1, 3, 2]

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_min(x, y, axis=1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value)#[[[[0., 0., 0., 0., 0.] .... [0., 0., 0., 0., 0.]]]]
    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_min')

    return _elementwise_op(LayerHelper('elementwise_min', **locals()))


def elementwise_pow(x, y, axis=-1, act=None, name=None):
    """

Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_pow(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[2, 243, 16]
    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_pow')
    return _elementwise_op(LayerHelper('elementwise_pow', **locals()))


@deprecated(since="2.0.0", update_to="paddle.remainder")
def elementwise_mod(x, y, axis=-1, act=None, name=None):
    """

Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.array([10, 15, 8]).astype('int32'),
                "y": np.array([3, 6, 5]).astype('int32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='int32')
        y = fluid.data(name="y", shape=[3], dtype='int32')
        z = fluid.layers.elementwise_mod(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[1, 3, 3]
    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_mod')

    return _elementwise_op(LayerHelper('elementwise_mod', **locals()))


@deprecated(since="2.0.0", update_to="paddle.floor_divide")
def elementwise_floordiv(x, y, axis=-1, act=None, name=None):
    """

Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        import paddle

        def gen_data():
            return {
                "x": np.array([10, 15, 8]).astype('int32'),
                "y": np.array([3, 7, 5]).astype('int32')
            }
        paddle.enable_static()
        x = fluid.data(name="x", shape=[3], dtype='int32')
        y = fluid.data(name="y", shape=[3], dtype='int32')
        z = fluid.layers.elementwise_floordiv(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[3, 2, 1]
    """
    if in_dygraph_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_floordiv')

    return _elementwise_op(LayerHelper('elementwise_floordiv', **locals()))


for func in [
        elementwise_add,
        elementwise_div,
        elementwise_sub,
        elementwise_mul,
        elementwise_max,
        elementwise_pow,
        elementwise_min,
        elementwise_mod,
        elementwise_floordiv,
]:
    op_proto = OpProtoHolder.instance().get_op_proto(func.__name__)

    # insert the c++ doc string on top of python doc string
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "axis (int32, optional): If X.dimension != Y.dimension, \
            Y.dimension must be a subsequence of x.dimension. \
            And axis is the start dimension index for broadcasting Y onto X. ",
            "act (string, optional): Activation applied to the output. \
            Default is None. Details: :ref:`api_guide_activations_en` ",
            "name (string, optional): Name of the output. \
            Default is None. It's used to print debug info for developers. Details: \
            :ref:`api_guide_Name` "
        ],
        skip_attrs_set={
            "x_data_format", "y_data_format", "axis", "use_quantizer",
            "mkldnn_data_type", "Scale_x", "Scale_y", "Scale_out"
        }) + """\n""" + str(func.__doc__)

    doc_list = func.__doc__.splitlines()

    for idx, val in enumerate(doc_list):
        if val.startswith("Warning: ") and val.endswith(
                " instead."
        ) and "and will be removed in future versions." in val:
            doc_list.insert(0, doc_list.pop(idx))
            func.__doc__ = "\n" + "\n".join(i for i in doc_list)
            break

for func in []:
    op_proto = OpProtoHolder.instance().get_op_proto(func.__name__)
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "act (basestring|None): Activation applied to the output.",
            "name (basestring|None): Name of the output."
        ])
    func.__doc__ = func.__doc__ + """

Examples:
  .. code-block:: python

    import paddle.fluid as fluid
    # example 1: shape(x) = (2, 3, 4, 5), shape(y) = (2, 3, 4, 5)
    x0 = fluid.layers.data(name="x0", shape=[2, 3, 4, 5], dtype='float32')
    y0 = fluid.layers.data(name="y0", shape=[2, 3, 4, 5], dtype='float32')
    z0 = fluid.layers.%s(x0, y0)

    # example 2: shape(X) = (2, 3, 4, 5), shape(Y) = (5)
    x1 = fluid.layers.data(name="x1", shape=[2, 3, 4, 5], dtype='float32')
    y1 = fluid.layers.data(name="y1", shape=[5], dtype='float32')
    z1 = fluid.layers.%s(x1, y1)

    # example 3: shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    x2 = fluid.layers.data(name="x2", shape=[2, 3, 4, 5], dtype='float32')
    y2 = fluid.layers.data(name="y2", shape=[4, 5], dtype='float32')
    z2 = fluid.layers.%s(x2, y2, axis=2)

    # example 4: shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    x3 = fluid.layers.data(name="x3", shape=[2, 3, 4, 5], dtype='float32')
    y3 = fluid.layers.data(name="y3", shape=[3, 4], dtype='float32')
    z3 = fluid.layers.%s(x3, y3, axis=1)

    # example 5: shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    x4 = fluid.layers.data(name="x4", shape=[2, 3, 4, 5], dtype='float32')
    y4 = fluid.layers.data(name="y4", shape=[2], dtype='float32')
    z4 = fluid.layers.%s(x4, y4, axis=0)

    # example 6: shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
    x5 = fluid.layers.data(name="x5", shape=[2, 3, 4, 5], dtype='float32')
    y5 = fluid.layers.data(name="y5", shape=[2], dtype='float32')
    z5 = fluid.layers.%s(x5, y5, axis=0)
    """ % (func.__name__, func.__name__, func.__name__, func.__name__,
           func.__name__, func.__name__)


def _logical_op(op_name, x, y, out=None, name=None, binary_op=True):
    if in_dygraph_mode():
        op = getattr(_C_ops, op_name)
        if binary_op:
            return op(x, y)
        else:
            return op(x)
    check_variable_and_dtype(x, "x", [
        "bool", "int8", "int16", "int32", "int64", "float32", "float64"
    ], op_name)
    if y is not None:
        check_variable_and_dtype(y, "y", [
            "bool", "int8", "int16", "int32", "int64", "float32", "float64"
        ], op_name)
    if out is not None:
        check_type(out, "out", Variable, op_name)

    helper = LayerHelper(op_name, **locals())

    if binary_op and x.dtype != y.dtype:
        raise ValueError(
            "(InvalidArgument) The DataType of %s Op's Variable must be consistent, but received %s and %s."
            % (op_name, x.dtype, y.dtype))

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if binary_op:
        helper.append_op(
            type=op_name, inputs={"X": x,
                                  "Y": y}, outputs={"Out": out})
    else:
        helper.append_op(type=op_name, inputs={"X": x}, outputs={"Out": out})

    return out


def logical_and(x, y, out=None, name=None):
    r"""

    ``logical_and`` operator computes element-wise logical AND on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x \&\& y

    .. note::
        ``paddle.logical_and`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
        out(Tensor): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([True])
            y = paddle.to_tensor([True, False, True, False])
            res = paddle.logical_and(x, y)
            print(res) # [True False True False]
    """
    return _logical_op(
        op_name="logical_and", x=x, y=y, name=name, out=out, binary_op=True)


def logical_or(x, y, out=None, name=None):
    """

    ``logical_or`` operator computes element-wise logical OR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = x || y

    .. note::
        ``paddle.logical_or`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.
    
    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
        out(Tensor): The ``Variable`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x_data = np.array([True, False], dtype=np.bool).reshape(2, 1)
            y_data = np.array([True, False, True, False], dtype=np.bool).reshape(2, 2)
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            res = paddle.logical_or(x, y)
            print(res) # [[ True  True] [ True False]]
    """
    return _logical_op(
        op_name="logical_or", x=x, y=y, name=name, out=out, binary_op=True)


def logical_xor(x, y, out=None, name=None):
    r"""

    ``logical_xor`` operator computes element-wise logical XOR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
    Each element of ``out`` is calculated by

    .. math::

        out = (x || y) \&\& !(x \&\& y)

    .. note::
        ``paddle.logical_xor`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.

    Args:
        x (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
        y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
        out(Tensor): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            x_data = np.array([True, False], dtype=np.bool).reshape([2, 1])
            y_data = np.array([True, False, True, False], dtype=np.bool).reshape([2, 2])
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            res = paddle.logical_xor(x, y)
            print(res) # [[False,  True], [ True, False]]
    """
    return _logical_op(
        op_name="logical_xor", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def logical_not(x, out=None, name=None):
    """

    ``logical_not`` operator computes element-wise logical NOT on ``x``, and returns ``out``. ``out`` is N-dim boolean ``Variable``.
    Each element of ``out`` is calculated by

    .. math::

        out = !x

    Args:
        x(Tensor):  Operand of logical_not operator. Must be a Tensor of type bool, int8, int16, in32, in64, float32, or float64.
        out(Tensor): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor` will be created to save the output.
        name(str|None): The default value is None. Normally there is no need for users to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([True, False, True, False])
            res = paddle.logical_not(x)
            print(res) # [False  True False  True]
    """

    return _logical_op(
        op_name="logical_not", x=x, y=None, name=name, out=out, binary_op=False)


@templatedoc()
def clip(x, min, max, name=None):
    """
	:old_api: paddle.fluid.layers.clip

    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        min(float): ${min_comment}
        max(float): ${max_comment}
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_comment}

    Return Type:
        ${out_type}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(
                name='data', shape=[1], dtype='float32')
            reward = fluid.layers.clip(x=input, min=-1.0, max=1.0)
    """

    helper = LayerHelper("clip", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'clip')

    if name is None:
        name = unique_name.generate_with_ignorable_key(".".join(
            [helper.name, 'tmp']))

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="clip",
        inputs={"X": x},
        attrs={"min": min,
               "max": max},
        outputs={"Out": out})

    return out


@templatedoc()
def clip_by_norm(x, max_norm, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        max_norm(${max_norm_type}): ${max_norm_comment}
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor:

        out(${out_type}): ${out_comment}


    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            input = paddle.to_tensor([[2.0, 2.0], [2.0, 2.0]], dtype='float32')
            reward = fluid.layers.clip_by_norm(x=input, max_norm=1.0)
            # [[0.5, 0.5], [0.5, 0.5]]
    """

    if in_dygraph_mode():
        return _C_ops.clip_by_norm(x, 'max_norm', max_norm)

    helper = LayerHelper("clip_by_norm", **locals())
    check_variable_and_dtype(x, 'X', ['float32'], 'clip_by_norm')
    check_type(max_norm, 'max_norm', (float), 'clip_by_norm')

    if name is None:
        name = unique_name.generate_with_ignorable_key(".".join(
            [helper.name, 'tmp']))

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="clip_by_norm",
        inputs={"X": x},
        attrs={"max_norm": max_norm},
        outputs={"Out": out})

    return out


@deprecated(since="2.0.0", update_to="paddle.mean")
@templatedoc()
def mean(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            input = fluid.layers.data(
                name='data', shape=[2, 3], dtype='float32')
            mean = fluid.layers.mean(input)
    """

    if in_dygraph_mode():
        return _C_ops.mean(x)

    helper = LayerHelper("mean", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'mean')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="mean", inputs={"X": x}, attrs={}, outputs={"Out": out})

    return out


@templatedoc()
def merge_selected_rows(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            b = fluid.default_main_program().global_block()
            var = b.create_var(
                name="X", dtype="float32", persistable=True,
                type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
            y = fluid.layers.merge_selected_rows(var)
    """

    helper = LayerHelper("merge_selected_rows", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="merge_selected_rows",
        inputs={"X": x},
        attrs={},
        outputs={"Out": out})
    return out


def mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None):
    """
    Mul Operator.
    This operator is used to perform matrix multiplication for input $x$ and $y$.
    The equation is:

    ..  math::
        Out = x * y

    Both the input $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $x$.

    Args:
        x (Variable): The first input Tensor/LoDTensor of mul_op.
        y (Variable): The second input Tensor/LoDTensor of mul_op.
        x_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs. If the input $x$ is a tensor with more than two dimensions, $x$ will be flattened into a two-dimensional matrix first. The flattening rule is: the first `num_col_dims` will be flattened to form the first dimension of the final matrix (the height of the matrix), and the rest `rank(x) - num_col_dims` dimensions are flattened to form the second dimension of the final matrix (the width of the matrix). As a result, height of the flattened matrix is equal to the product of $x$'s first `x_num_col_dims` dimensions' sizes, and width of the flattened matrix is equal to the product of $x$'s last `rank(x) - num_col_dims` dimensions' size. For example, suppose $x$ is a 6-dimensional tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3. Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default is 1.
        y_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs. If the input $y$ is a tensor with more than two dimensions, $y$ will be flattened into a two-dimensional matrix first. The attribute `y_num_col_dims` determines how $y$ is flattened. See comments of `x_num_col_dims` for more details. Default is 1.
        name (str, optional): Name of the output. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Variable(Tensor/LoDTensor): The output Tensor/LoDTensor of mul op.

    Examples:
        ..  code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            dataX = fluid.layers.data(name="dataX", append_batch_size = False, shape=[2, 5], dtype="float32")
            dataY = fluid.layers.data(name="dataY", append_batch_size = False, shape=[5, 3], dtype="float32")
            output = fluid.layers.mul(dataX, dataY,
                                      x_num_col_dims = 1,
                                      y_num_col_dims = 1)


    """
    if in_dygraph_mode():
        return _C_ops.mul(x, y, 'x_num_col_dims', x_num_col_dims,
                          'y_num_col_dims', y_num_col_dims)

    inputs = {"X": [x], "Y": [y]}
    attrs = {"x_num_col_dims": x_num_col_dims, "y_num_col_dims": y_num_col_dims}
    helper = LayerHelper("mul", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'mul')
    check_variable_and_dtype(y, 'y', ['float16', 'float32', 'float64'], 'mul')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="mul", inputs={"X": x,
                            "Y": y}, attrs=attrs, outputs={"Out": out})
    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.maxout")
@templatedoc()
def maxout(x, groups, name=None, axis=1):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        groups(int): ${groups_comment}
        axis(int, optional): ${axis_comment}
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Variable: ${out_comment}

    Raises:
        ValueError: If `axis` is not 1, -1 or 3.
        ValueError: If the number of input channels can not be divisible by `groups`.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()

            input = fluid.data(
                name='data',
                shape=[None, 256, 32, 32],
                dtype='float32')
            out = fluid.layers.maxout(input, groups=2)
    """
    return paddle.nn.functional.maxout(**locals())


def space_to_depth(x, blocksize, name=None):
    r"""

    Gives a blocksize to space_to_depth the input LoDtensor with Layout: [batch, channel, height, width]

    This op rearranges blocks of spatial data, into depth. More specifically, this op outputs a copy of \
        theinput LoDtensor where values from the height and width dimensions are moved to the channel \
        dimension.
    The attr blocksize indicates the input block size.

    space_to_depth will reorganize the elements of input with shape[batch, channel, height, width] \
        according to blocksize to construct output with shape \
        [batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]:

    - Non-overlapping blocks of size block_size x block size are rearranged into depth at each location.
    - The Y, X coordinates within each block of the input become the high order component of the output channel index
    - channel should be divisible by square of blocksize
    - height, width should be divsible by blocksize

    This OP is useful for resizing the activations between convolutions \
        (but keeping all data)

    .. code-block:: text

        Given the input x with the shape [1, 1, 4, 4]:
        x.data = [[[[1,   2,  5,  6],
                    [3,   4,  7,  8],
                    [9,  10, 13, 14],
                    [11, 12, 15, 16]]]]
        blocksize = 2

        then get the output with the shape [1, 4, 2, 2]:
        out.data = [[[[1,   2],  [3,  4]],
                     [[5,   6],  [7,  8]],
                     [[9,  10], [11, 12]],
                     [[13, 14], [15, 16]]]]

    Args:
        x (Variable): The input, which should be 4 dims Tensor or LodTensor, with the shape \
            [batch, channel, height, width]
        blocksize (int): The blocksize to select the element on each feature map should be > 2
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: The output, which should be 4 dims Tensor or LodTensor, with the shape \
            [batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]

    Return Type: Variable

    Raises:
        TypeError: blocksize type must be int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import numpy as np
            import paddle

            paddle.enable_static()
            data = fluid.data(
                name='data', shape=[1, 4, 2, 2], dtype='float32')
            space_to_depthed = fluid.layers.space_to_depth(
                x=data, blocksize=2)

            exe = fluid.Executor(fluid.CPUPlace())
            data_np = np.arange(0,16).reshape((1,4,2,2)).astype('float32')

            print(data_np)
            #array([[[[ 0.,  1.], [ 2.,  3.]],
            #        [[ 4.,  5.], [ 6.,  7.]],
            #        [[ 8.,  9.], [10., 11.]],
            #        [[12., 13.], [14., 15.]]]], dtype=float32)

            out_main = exe.run(fluid.default_main_program(),
                        feed={'data': data_np},
                        fetch_list=[space_to_depthed])

            print(out_main)
            #[array([[[[ 0.]], [[ 4.]], [[ 1.]], [[ 5.]],
            #         [[ 8.]], [[12.]], [[ 9.]], [[13.]],
            #         [[ 2.]], [[ 6.]], [[ 3.]], [[ 7.]],
            #         [[10.]], [[14.]], [[11.]], [[15.]]]], dtype=float32)]

    """

    helper = LayerHelper("space_to_depth", **locals())

    if not (isinstance(blocksize, int)):
        raise ValueError("blocksize must be a python Int")

    check_variable_and_dtype(x, 'x', \
        ['float16', 'float32', 'float64', 'int32', 'int64'], 'space_to_depth')

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="space_to_depth",
        inputs={"X": x},
        attrs={"blocksize": blocksize},
        outputs={"Out": out})
    return out


def affine_channel(x,
                   scale=None,
                   bias=None,
                   data_layout='NCHW',
                   name=None,
                   act=None):
    """

    Applies a separate affine transformation to each channel of the input.
    Useful for replacing spatial batch norm with its equivalent fixed
    transformation. The input also can be 2D tensor and applies a affine
    transformation in second dimension.

    Args:
        x (Variable): Feature map input can be a 4D tensor with order NCHW
            or NHWC. It also can be a 2D tensor and the affine transformation
            is applied in the second dimension.The data type is float32 or float64.
        scale (Variable): 1D input of shape (C), the c-th element is the scale
            factor of the affine transformation for the c-th channel of
            the input.The data type is float32 or float64.
        bias (Variable): 1D input of shape (C), the c-th element is the bias
            of the affine transformation for the c-th channel of the input.
            The data type is float32 or float64.
        data_layout (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. If input is 2D Tensor, you can ignore
            data_layout.
        name (str, default None): The name of this layer. For more information,
            please refer to :ref:`api_guide_Name` .
        act (str, default None): Activation to be applied to the output of this layer.

    Returns:
        Variable: A tensor which has the same shape, data layout and data type with x.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            import paddle.fluid as fluid
            import paddle

            paddle.enable_static()
            use_gpu = False
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)

            data = fluid.data(name='data', shape=[None, 1, 2, 2], dtype='float32')
            input_scale = fluid.layers.create_parameter(shape=[1], dtype="float32",
                                    default_initializer=fluid.initializer.Constant(2.0))
            input_bias = fluid.layers.create_parameter(shape=[1],dtype="float32",
                                    default_initializer=fluid.initializer.Constant(0.5))
            out = fluid.layers.affine_channel(data,scale=input_scale,
                                    bias=input_bias)

            exe.run(fluid.default_startup_program())
            test_program = fluid.default_main_program().clone(for_test=True)

            [out_array] = exe.run(test_program,
                                  fetch_list=out,
                                  feed={'data': np.ones([1,1,2,2]).astype('float32')})
            # out_array is [[[[2.5, 2.5],
            #                [2.5, 2.5]]]] with shape: [1, 1, 2, 2]

    """
    helper = LayerHelper("affine_channel", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'affine_channel')
    check_type(scale, 'scale', (Variable, type(None)), 'affine_channel')
    check_type(bias, 'bias', (Variable, type(None)), 'affine_channel')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="affine_channel",
        inputs={"X": x,
                'Scale': scale,
                'Bias': bias},
        attrs={"data_layout": data_layout},
        outputs={"Out": out})
    return helper.append_activation(out)


def similarity_focus(input, axis, indexes, name=None):
    r"""
    SimilarityFocus Operator

    Generate a similarity focus mask with the same shape of input using the following method:

    1. Extract the 3-D tensor(here the first dimension is BatchSize) corresponding
       to the axis according to the indexes. For example, if axis=1 and indexes=[a],
       it will get the matrix T=X[:, a, :, :]. In this case, if the shape of input X
       is (BatchSize, A, B, C), the shape of tensor T is (BatchSize, B, C).
    2. For each index, find the largest numbers in the tensor T, so that the same
       row and same column has at most one number(what it means is that if the
       largest number has been found in the i-th row and the j-th column, then
       the numbers in the i-th row or j-th column will be skipped. And then the
       next largest number will be selected from the remaining numbers. Obviously
       there will be min(B, C) numbers), and mark the corresponding position of the
       3-D similarity focus mask as 1, otherwise as 0. Do elementwise-or for
       each index.
    3. Broadcast the 3-D similarity focus mask to the same shape of input X.

    Refer to `Similarity Focus Layer <http://www.aclweb.org/anthology/N16-1108>`_

    .. code-block:: text

        * Example :

            Given a 4-D tensor x with the shape (BatchSize, C, A, B), where C is
            the number of channels and the shape of feature map is (A, B):
                x.shape = (2, 3, 2, 2)
                x.data = [[[[0.8, 0.1],
                            [0.4, 0.5]],

                           [[0.9, 0.7],
                            [0.9, 0.9]],

                           [[0.8, 0.9],
                            [0.1, 0.2]]],


                          [[[0.2, 0.5],
                            [0.3, 0.4]],

                           [[0.9, 0.7],
                            [0.8, 0.4]],

                           [[0.0, 0.2],
                            [0.4, 0.7]]]]

            Given axis: 1 (the axis of the channel)
            Given indexes: [0]

            then we get a 4-D tensor out with the same shape of input x:
                out.shape = (2, 3, 2, 2)
                out.data = [[[[1.0, 0.0],
                              [0.0, 1.0]],

                             [[1.0, 0.0],
                              [0.0, 1.0]],

                             [[1.0, 0.0],
                              [0.0, 1.0]]],

                            [[[0.0, 1.0],
                              [1.0, 0.0]],

                             [[0.0, 1.0],
                              [1.0, 0.0]],

                             [[0.0, 1.0],
                              [1.0, 0.0]]]]

    Args:
        input(Variable): The input tensor variable(default float). It should
            be a 4-D tensor with shape [BatchSize, A, B, C]. Data type is
            float32 or float64.
        axis(int): Indicating the dimension to be selected. It can only be
            1, 2 or 3.
        indexes(list): Indicating the indexes of the selected dimension.

    Returns:
        Variable: A tensor variable with the same shape and same type \
                  as the input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            data = fluid.data(
                name='data', shape=[-1, 3, 2, 2], dtype='float32')
            fluid.layers.similarity_focus(input=data, axis=1, indexes=[0])
    """
    helper = LayerHelper('similarity_focus', **locals())
    # check attrs
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             "similarity_focus")
    check_type(axis, 'axis', int, "similarity_focus")
    check_type(indexes, 'indexes', list, "similarity_focus")
    if axis != 1 and axis != 2 and axis != 3:
        raise ValueError("axis must be 1, 2 or 3.")
    if len(indexes) == 0:
        raise ValueError("indexes can not be empty.")

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='similarity_focus',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={"axis": axis,
               "indexes": indexes})
    return out


def hash(input, hash_size, num_hash=1, name=None):
    """

    This OP hash the input to an integer less than the hash_size.
    The hash algorithm we used was xxHash - Extremely fast hash algorithm
    (https://github.com/Cyan4973/xxHash/tree/v0.6.5)

    Args:
        input(Variable): A **Two-Dimensional** LoDTensor with type int32, int64.
             **Only support LoDTensor**.
        num_hash(int, optional): The times of hash, default is 1.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
       Variable: A LoDTensor with the same data type as input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            paddle.enable_static()

            place = fluid.core.CPUPlace()

            x = fluid.data(name="x", shape=[2,2], dtype="int32", lod_level=1)
            res = fluid.layers.hash(name="res", input=x, hash_size=1000, num_hash=4)

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            in1 = np.array([[1,2],[3,4]]).astype("int32")
            print(in1)
            x_i = fluid.create_lod_tensor(in1, [[0, 2]], place)
            res = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res], return_numpy=False)
            print(np.array(res[0]))
            # [[[722]
            #   [407]
            #   [337]
            #   [395]]
            #  [[603]
            #   [590]
            #   [386]
            #   [901]]]
    """
    check_variable_and_dtype(input, 'input', ['int32', 'int64'], 'hash')
    check_type(hash_size, 'hash_size', int, 'hash')
    check_type(num_hash, 'num_hash', int, 'hash')
    helper = LayerHelper('hash', **locals())
    out = helper.create_variable_for_type_inference(
        helper.input_dtype(), stop_gradient=True)
    helper.append_op(
        type='hash',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'num_hash': num_hash,
               'mod_by': hash_size})
    return out


@templatedoc()
def grid_sampler(x, grid, name=None):
    """

    This operation samples input X by using bilinear interpolation based on
    flow field grid, which is usually generated by :code:`affine_grid` . The grid of
    shape [N, H, W, 2] is the concatenation of (x, y) coordinates
    with shape [N, H, W] each, where x is indexing the 4th dimension
    (in width dimension) of input data x and y is indexing the 3rd
    dimension (in height dimension), finally results is the bilinear
    interpolation value of 4 nearest corner points. The output tensor
    shape will be [N, C, H, W].

    .. code-block:: text

        Step 1:
        Get (x, y) grid coordinates and scale to [0, H-1/W-1].

        .. code-block:: text

            grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1)
            grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

        Step 2:
        Indices input data X with grid (x, y) in each [H, W] area, and bilinear
        interpolate point value by 4 nearest points.

          wn ------- y_n ------- en
          |           |           |
          |          d_n          |
          |           |           |
         x_w --d_w-- grid--d_e-- x_e
          |           |           |
          |          d_s          |
          |           |           |
          ws ------- y_s ------- wn

        x_w = floor(x)              // west side x coord
        x_e = x_w + 1               // east side x coord
        y_n = floor(y)              // north side y coord
        y_s = y_s + 1               // south side y coord

        d_w = grid_x - x_w          // distance to west side
        d_e = x_e - grid_x          // distance to east side
        d_n = grid_y - y_n          // distance to north side
        d_s = y_s - grid_y          // distance to south side

        wn = X[:, :, y_n, x_w]      // north-west point value
        en = X[:, :, y_n, x_e]      // north-east point value
        ws = X[:, :, y_s, x_w]      // south-east point value
        es = X[:, :, y_s, x_w]      // north-east point value

        output = wn * d_e * d_s + en * d_w * d_s
               + ws * d_e * d_n + es * d_w * d_n

    Args:
        x(Variable): The input tensor, which is a 4-D tensor with shape
                     [N, C, H, W], N is the batch size, C is the channel
                     number, H and W is the feature height and width.
                     The data type is float32 or float64.
        grid(Variable): Input grid tensor of shape [N, H, W, 2]. The
                        data type is float32 or float64.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Variable: Output of shape [N, C, H, W] data samples input X
                  using bilnear interpolation based on input grid.
                  The data type is same as input tensor.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid as fluid
            import paddle

            paddle.enable_static()
            # use with affine_grid
            x = fluid.data(name='x', shape=[None, 10, 32, 32], dtype='float32')
            theta = fluid.layers.data(name='theta', shape=[2, 3], dtype='float32')
            grid = fluid.layers.affine_grid(theta=theta, out_shape=[3, 10, 32, 32])
            out = fluid.layers.grid_sampler(x=x, grid=grid)

    """
    helper = LayerHelper("grid_sampler", **locals())

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'grid_sampler')
    check_variable_and_dtype(grid, 'grid', ['float32', 'float64'],
                             'grid_sampler')
    if not isinstance(x, Variable):
        return ValueError("The x should be a Variable")

    if not isinstance(grid, Variable):
        return ValueError("The grid should be a Variable")

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x, 'Grid': grid}

    attrs = {'use_cudnn': False} if core.is_compiled_with_rocm() else {}

    helper.append_op(
        type='grid_sampler', inputs=ipts, outputs={'Output': out}, attrs=attrs)
    return out


def log_loss(input, label, epsilon=1e-4, name=None):
    r"""

    **Negative Log Loss Layer**

    This layer accepts input predictions and target label and returns the
    negative log loss.

    .. math::

        Out = -label * \log{(input + \epsilon)}
              - (1 - label) * \log{(1 - input + \epsilon)}

    Args:
        input (Tensor|list):  A 2-D tensor with shape [N x 1], where N is the
                                batch size. This input is a probability computed
                                by the previous operator. Data type float32.
        label (Tensor|list):  The ground truth which is a 2-D tensor with
                                shape [N x 1], where N is the batch size.
                                Data type float32.
        epsilon (float, optional): A small number for numerical stability. Default 1e-4.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Tensor, which shape is [N x 1], data type is float32.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn.functional as F

          label = paddle.randn((10,1))
          prob = paddle.randn((10,1))
          cost = F.log_loss(input=prob, label=label)
    """
    helper = LayerHelper('log_loss', **locals())
    check_variable_and_dtype(input, 'input', ['float32'], 'log_loss')
    check_variable_and_dtype(label, 'label', ['float32'], 'log_loss')

    loss = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='log_loss',
        inputs={'Predicted': [input],
                'Labels': [label]},
        outputs={'Loss': [loss]},
        attrs={'epsilon': epsilon})
    return loss


def add_position_encoding(input, alpha, beta, name=None):
    r"""

    This operator performs weighted sum of input feature at each position
    (position in the sequence) and the corresponding position encoding.

    For more details of position encoding, please refer to `Attention Is All You
    Need <http://arxiv.org/pdf/1706.03762.pdf>`_ .

    The formula is as follows:

    .. math::
        PE(pos, 2i) &= \\sin{(pos / 10000^{2i / P})}   \\\\
        PE(pos, 2i + 1) &= \\cos{(pos / 10000^{2i / P})}  \\\\
        Out(:, pos, i) &= \\alpha * input(:, pos, i) + \\beta * PE(pos, i)

    Where:
      - :math:`PE(pos, 2i)` : the value at even index `2i` for encoding of position `pos`.
      - :math:`PE(pos, 2i + 1)` : the value at odd index `2i+1` for encoding of position `pos`

    Args:
        input(Variable): A Tensor or LoDTensor (lod level is 1). If it is a
            Tensor, the shape should be `[N, M, P]`, where `N` stands for
            batch size, `M` for sequence length, `P` for the size of feature
            dimension. If it is a LoDTensor, the shape should be `[N, P]`,
            where `N` stands for the total sequence lengths in this mini-batch,
            `P` for the size of feature. The data type should be float32 or float64.
        alpha(float): Indicate the weight coefficient for `input` when performing
            weighted sum.
        beta(float): Indicate the weight coefficient for position encoding when
            performing weighted sum.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Variable: A Tensor or LoDTensor. It has the same shape, data type and lod as `input`.

    Examples:
        .. code-block:: python

          import paddle

          tensor = paddle.randn([16, 32, 64])
          position_tensor = paddle.fluid.layers.add_position_encoding(
                input=tensor, alpha=1.0, beta=1.0)

    """
    if in_dygraph_mode():
        return _C_ops.add_position_encoding(input, "alpha", alpha, "beta", beta)

    helper = LayerHelper('add_position_encoding', **locals())
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             "add_position_encoding")
    dtype = helper.input_dtype()

    out = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type="add_position_encoding",
        inputs={"X": input},
        outputs={"Out": out},
        attrs={"alpha": alpha,
               "beta": beta})
    return out


def bilinear_tensor_product(x,
                            y,
                            size,
                            act=None,
                            name=None,
                            param_attr=None,
                            bias_attr=None):
    r"""
    :api_attr: Static Graph

    **Bilinear Tensor Product Layer**

    This layer performs bilinear tensor product on two inputs.
    For example:

    .. math::
       out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

    In this formula:
      - :math:`x`: the first input contains M elements, shape is [batch_size, M].
      - :math:`y`: the second input contains N elements, shape is [batch_size, N].
      - :math:`W_{i}`: the i-th learned weight, shape is [M, N].
      - :math:`out_{i}`: the i-th element of out, shape is [batch_size, size].
      - :math:`y^\mathrm{T}`: the transpose of :math:`y_{2}`.

    Args:
        x (Variable): 2-D input tensor with shape [batch_size, M]. Data type
            is float32 or float64.
        y (Variable): 2-D input tensor with shape [batch_size, N]. Data type
            should be same as **x**.
        size (int): The dimension of this layer.
        act (str|None): Activation to be applied to the output of this layer. Default None.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        param_attr (ParamAttr|None): To specify the weight parameter attribute.
            Default: None, which means the default weight parameter property is
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr|None): To specify the bias parameter attribute.
            Default: None, which means the default bias parameter property is
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
    Returns:
        Variable: A 2-D Tensor of shape [batch_size, size]. Data type is the same as input **x**.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            layer1 = paddle.static.data("t1", shape=[-1, 5], dtype="float32")
            layer2 = paddle.static.data("t2", shape=[-1, 4], dtype="float32")
            tensor = paddle.static.nn.bilinear_tensor_product(x=layer1, y=layer2, size=1000)
    """
    helper = LayerHelper('bilinear_tensor_product', **locals())
    dtype = helper.input_dtype('x')

    param_shape = [size, x.shape[1], y.shape[1]]

    w = helper.create_parameter(
        attr=helper.param_attr, shape=param_shape, dtype=dtype, is_bias=False)
    out = helper.create_variable_for_type_inference(dtype=dtype)

    inputs = {"X": x, "Y": y, "Weight": w}
    if helper.bias_attr:
        bias_size = [1, size]
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)
        inputs["Bias"] = bias
    helper.append_op(
        type="bilinear_tensor_product", inputs=inputs, outputs={"Out": out})

    # add activation
    return helper.append_activation(out)


@templatedoc()
def get_tensor_from_selected_rows(x, name=None):
    """
    This operator gets tensor data from input with SelectedRows type, and outputs a LoDTensor.

    .. code-block:: text

        input x is SelectedRows:
           x.rows = [0, 5, 5, 4, 19]
           x.height = 20
           x.value = [[1, 1] [2, 2] [2, 2] [3, 3] [6, 6]]

        Ouput is LoDTensor:
           out.shape = [5, 2]
           out.data = [[1, 1],
                       [2, 2],
                       [2, 2],
                       [3, 3],
                       [6, 6]]

    Args:
        x(SelectedRows): Input with SelectedRows type. The data type is float32, float64, int32 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: LoDTensor transformed from SelectedRows. The data type is same with input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            b = fluid.default_main_program().global_block()
            input = b.create_var(name="X", dtype="float32", persistable=True, type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
            out = fluid.layers.get_tensor_from_selected_rows(input)
    """

    check_type(x, 'x', Variable, 'get_tensor_from_selected_rows')
    if x.type != core.VarDesc.VarType.SELECTED_ROWS:
        raise TypeError(
            "The type of 'x' in get_tensor_from_selected_rows must be SELECTED_ROWS."
        )
    helper = LayerHelper('get_tensor_from_selected_rows', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='get_tensor_from_selected_rows',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={})
    return out


def shuffle_channel(x, group, name=None):
    """
    This operator shuffles the channels of input x.
    It divide the input channels in each group into :attr:`group` subgroups,
    and obtain a new order by selecting element from every subgroup one by one.

    Please refer to the paper
    https://arxiv.org/pdf/1707.01083.pdf

    .. code-block:: text

        Given a 4-D tensor input with the shape (N, C, H, W):
            input.shape = (1, 4, 2, 2)
            input.data =[[[[0.1, 0.2],
                           [0.2, 0.3]],

                          [[0.3, 0.4],
                           [0.4, 0.5]],

                          [[0.5, 0.6],
                           [0.6, 0.7]],

                          [[0.7, 0.8],
                           [0.8, 0.9]]]]
            Given group: 2
            then we get a 4-D tensor out with the same shape of input:
            out.shape = (1, 4, 2, 2)
            out.data = [[[[0.1, 0.2],
                          [0.2, 0.3]],

                         [[0.5, 0.6],
                          [0.6, 0.7]],

                         [[0.3, 0.4],
                          [0.4, 0.5]],

                         [[0.7, 0.8],
                          [0.8, 0.9]]]]

    Args:
        x(Variable): The input tensor variable. It should be a 4-D tensor with shape [N, C, H, W]
        group(int): Indicating the counts of subgroups, It should divide the number of channels.

    Returns:
        out(Variable): the channels shuffling result is a tensor variable with the
        same shape and same type as the input.

    Raises:
        ValueError: If group is not an int type variable.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()
            input = fluid.data(name='input', shape=[None,4,2,2], dtype='float32')
            out = fluid.layers.shuffle_channel(x=input, group=2)
    """
    helper = LayerHelper("shuffle_channel", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(group, int):
        raise TypeError("group must be int type")

    helper.append_op(
        type="shuffle_channel",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"group": group})
    return out


@templatedoc()
def temporal_shift(x, seg_num, shift_ratio=0.25, name=None, data_format="NCHW"):
    """

    **Temporal Shift Operator**

    ${comment}

    Args:
        x(Tensor): ${x_comment}
        seg_num(int): ${seg_num_comment}
        shift_ratio(float): ${shift_ratio_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCHW" or "NHWC". Default: "NCHW".

    Returns:
        out(Tensor): The temporal shifting result is a tensor with the
        same shape and same data type as the input.

    Raises:
        TypeError: seg_num must be int type.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            input = paddle.randn([6, 4, 2, 2])
            out = F.temporal_shift(x=input, seg_num=2, shift_ratio=0.2)
    """
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Attr(data_format) should be 'NCHW' or 'NHWC'. "
                         "Received Attr(data_format): {}.".format(data_format))
    if in_dygraph_mode():
        return _C_ops.temporal_shift(x, 'seg_num', seg_num, 'shift_ratio',
                                     shift_ratio, 'data_format', data_format)

    helper = LayerHelper("temporal_shift", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'temporal_shift')
    check_type(seg_num, 'seg_num', int, 'temporal_shift')
    check_type(shift_ratio, 'shift_ratio', float, 'temporal_shift')

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(seg_num, int):
        raise TypeError("seg_num must be int type.")

    helper.append_op(
        type="temporal_shift",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={
            "seg_num": seg_num,
            "shift_ratio": shift_ratio,
            "data_format": data_format
        })
    return out


class PyFuncRegistry(object):
    _register_funcs = []

    def __init__(self, func):
        if func is None or not callable(func):
            raise TypeError('func must be a Python function')

        self._func = func
        # find named args using reflection
        args = inspect.getargspec(self._func)
        if len(args[0]) == 0 and args[1] is None and args[2] is None:
            # Function with no inputs
            self._named_args = None
        else:
            self._named_args = args[0]
        self._id = core._append_python_callable_object_and_return_id(self)
        '''
        Why record self here?

        1. For debug usage. Users can call
           :code:`py_func.registered_func(idx)` method
           to find the registered function corresponding
           to :code:`idx`.

        2. For increasing reference count of self.
           It seems that to release Python object
           whose reference count is 1 would cause
           segmentation fault error in C++ side.
           May be lack of Python GC in C++ side?
        '''
        PyFuncRegistry._register_funcs.append(self)

    @classmethod
    def registered_func(cls, idx):
        return cls._register_funcs[idx]._func

    @classmethod
    def registered_func_num(cls):
        return len(cls._register_funcs)

    @property
    def id(self):
        return self._id

    def __call__(self, *args):
        if self._named_args is None:
            func_ret = self._func()
        else:
            kwargs = dict()
            idx = 0
            for arg in self._named_args:
                kwargs[arg] = args[idx]
                idx += 1
            func_ret = self._func(*args[idx:], **kwargs)

        if not isinstance(func_ret, (list, tuple)):
            func_ret = (func_ret, )

        ret = []
        for each_ret in func_ret:
            if each_ret is None or isinstance(each_ret, core.LoDTensor):
                ret.append(each_ret)
                continue

            if not isinstance(each_ret, np.ndarray):
                each_ret = np.array(each_ret)

            tensor = core.LoDTensor()
            tensor.set(each_ret, core.CPUPlace())
            ret.append(tensor)

        return tuple(ret)


@static_only
@templatedoc()
def py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None):
    """
    :api_attr: Static Graph

    This OP is used to register customized Python OP to Paddle. The design
    principe of py_func is that Tensor and numpy array can be converted to each
    other easily. So you can use Python and numpy API to register a python OP.

    The forward function of the registered OP is ``func`` and the backward function
    of that is ``backward_func``. Paddle will call ``func`` at forward runtime and
    call ``backward_func`` at backward runtime(if ``backward_func`` is not  None).
    ``x`` is the input of ``func``, whose type must be Tensor; ``out`` is
    the output of ``func``, whose type can be either Tensor or numpy array.

    The input of the backward function ``backward_func`` is ``x``, ``out`` and
    the gradient of ``out``. If ``out`` have no gradient, the relevant input of
    ``backward_func`` is None. If ``x`` do not have a gradient, the user should
    return None in ``backward_func``.

    The data type and shape of ``out`` should also be set correctly before this
    API is called, and the data type and shape of the gradient of ``out`` and
    ``x`` will be inferred automatically.

    This API can also be used to debug the neural network by setting the ``func``
    as a function that only print variables.

    Args:
        func (callable): The forward function of the registered OP. When the network
            is running, the forward output ``out`` will be calculated according to this
            function and the forward input ``x``. In ``func`` , it's suggested that we
            actively convert Tensor into a numpy array, so that we can use Python and
            numpy API arbitrarily. If not, some operations of numpy may not be compatible.
        x (Tensor|tuple(Tensor)|list[Tensor]): The input of the forward function ``func``.
            It can be Tensor|tuple(Tensor)|list[Tensor]. In addition, Multiple Tensor
            should be passed in the form of tuple(Tensor) or list[Tensor].
        out (T|tuple(T)|list[T]): The output of the forward function ``func``, it can be
            T|tuple(T)|list[T], where T can be either Tensor or numpy array. Since Paddle
            cannot automatically infer the shape and type of ``out``, you must create
            ``out`` in advance.
        backward_func (callable, optional): The backward function of the registered OP.
            Its default value is None, which means there is no reverse calculation. If
            it is not None, ``backward_func`` is called to calculate the gradient of
            ``x`` when the network is at backward runtime.
        skip_vars_in_backward_input (Tensor, optional): It's used to limit the input
            list of ``backward_func``, and it can be Tensor|tuple(Tensor)|list[Tensor].
            It must belong to either ``x`` or ``out``. The default  value is None, which means
            that no tensors need to be removed from ``x`` and ``out``. If it is not None,
            these tensors will not be the input of ``backward_func``. This parameter is only
            useful when ``backward_func`` is not None.

    Returns:
        Tensor|tuple(Tensor)|list[Tensor]: The output ``out`` of the forward function ``func``.

    Examples:
        .. code-block:: python

            # example 1:
            import paddle
            import six
            import numpy as np

            paddle.enable_static()

            # Creates a forward function, Tensor can be input directly without
            # being converted into numpy array.
            def tanh(x):
                return np.tanh(x)

            # Skip x in backward function and return the gradient of x
            # Tensor must be actively converted to numpy array, otherwise,
            # operations such as +/- can't be used.
            def tanh_grad(y, dy):
                return np.array(dy) * (1 - np.square(np.array(y)))

            # Creates a forward function for debugging running networks(print value)
            def debug_func(x):
                print(x)

            def create_tmp_var(name, dtype, shape):
                return paddle.static.default_main_program().current_block().create_var(
                    name=name, dtype=dtype, shape=shape)

            def simple_net(img, label):
                hidden = img
                for idx in six.moves.range(4):
                    hidden = paddle.static.nn.fc(hidden, size=200)
                    new_hidden = create_tmp_var(name='hidden_{}'.format(idx),
                        dtype=hidden.dtype, shape=hidden.shape)

                    # User-defined forward and backward
                    hidden = paddle.static.py_func(func=tanh, x=hidden,
                        out=new_hidden, backward_func=tanh_grad,
                        skip_vars_in_backward_input=hidden)

                    # User-defined debug functions that print out the input Tensor
                    paddle.static.py_func(func=debug_func, x=hidden, out=None)

                prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
                ce_loss = paddle.nn.loss.CrossEntropyLoss()
                return ce_loss(prediction, label)

            x = paddle.static.data(name='x', shape=[1,4], dtype='float32')
            y = paddle.static.data(name='y', shape=[1,10], dtype='int64')
            res = simple_net(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            input1 = np.random.random(size=[1,4]).astype('float32')
            input2 = np.random.randint(1, 10, size=[1,10], dtype='int64')
            out = exe.run(paddle.static.default_main_program(),
                          feed={'x':input1, 'y':input2},
                          fetch_list=[res.name])
            print(out)

        .. code-block:: python

            # example 2:
            # This example shows how to turn Tensor into numpy array and
            # use numpy API to register an Python OP
            import paddle
            import numpy as np

            paddle.enable_static()

            def element_wise_add(x, y):
                # Tensor must be actively converted to numpy array, otherwise,
                # numpy.shape can't be used.
                x = np.array(x)
                y = np.array(y)

                if x.shape != y.shape:
                    raise AssertionError("the shape of inputs must be the same!")

                result = np.zeros(x.shape, dtype='int32')
                for i in range(len(x)):
                    for j in range(len(x[0])):
                        result[i][j] = x[i][j] + y[i][j]

                return result

            def create_tmp_var(name, dtype, shape):
                return paddle.static.default_main_program().current_block().create_var(
                            name=name, dtype=dtype, shape=shape)

            def py_func_demo():
                start_program = paddle.static.default_startup_program()
                main_program = paddle.static.default_main_program()

                # Input of the forward function
                x = paddle.static.data(name='x', shape=[2,3], dtype='int32')
                y = paddle.static.data(name='y', shape=[2,3], dtype='int32')

                # Output of the forward function, name/dtype/shape must be specified
                output = create_tmp_var('output','int32', [3,1])

                # Multiple Variable should be passed in the form of tuple(Variale) or list[Variale]
                paddle.static.py_func(func=element_wise_add, x=[x,y], out=output)

                exe=paddle.static.Executor(paddle.CPUPlace())
                exe.run(start_program)

                # Feed numpy array to main_program
                input1 = np.random.randint(1, 10, size=[2,3], dtype='int32')
                input2 = np.random.randint(1, 10, size=[2,3], dtype='int32')
                out = exe.run(main_program,
                            feed={'x':input1, 'y':input2},
                            fetch_list=[output.name])
                print("{0} + {1} = {2}".format(input1, input2, out))

            py_func_demo()

            # Reference output:
            # [[5, 9, 9]   + [[7, 8, 4]  =  [array([[12, 17, 13]
            #  [7, 5, 2]]     [1, 3, 3]]            [8, 8, 5]], dtype=int32)]
    """
    helper = LayerHelper('py_func', **locals())
    check_type(x, 'X', (list, tuple, Variable, type(None)), 'py_func')
    if x is None:
        x = []
    elif isinstance(x, Variable):
        x = [x]
    elif isinstance(x, tuple):
        x = list(x)
    elif not isinstance(x, (list, tuple, Variable)):
        raise TypeError('Input must be Variable/list(Variable)/tuple(Variable)')
    check_type(out, 'Out', (list, tuple, Variable, type(None)), 'py_func')
    if out is None:
        out_list = []
    elif isinstance(out, Variable):
        out_list = [out]
    elif isinstance(out, tuple):
        out_list = list(out)
    elif isinstance(out, list):
        out_list = out
    else:
        raise TypeError(
            'Output must be Variable/list(Variable)/tuple(Variable)')

    fwd_func_id = PyFuncRegistry(func).id
    bwd_func_id = PyFuncRegistry(
        backward_func).id if backward_func is not None else -1

    for each_out in out_list:
        if len(each_out.shape) == 0:
            raise ValueError(
                'Output shapes of py_func op should be provided by users manually'
            )

    backward_skip_vars = set()
    if backward_func is not None and skip_vars_in_backward_input is not None:
        if isinstance(skip_vars_in_backward_input, Variable):
            skip_vars_in_backward_input = [skip_vars_in_backward_input]

        fwd_in_out = [v.name for v in x]
        fwd_in_out.extend([v.name for v in out_list])
        fwd_in_out = set(fwd_in_out)
        backward_skip_vars = set()
        for v in skip_vars_in_backward_input:
            if not v.name in fwd_in_out:
                raise ValueError(
                    'Variable {} is not found in forward inputs and outputs'
                    .format(v.name))
            backward_skip_vars.add(v.name)

    helper.append_op(
        type='py_func',
        inputs={'X': x},
        outputs={'Out': out_list},
        attrs={
            'forward_callable_id': fwd_func_id,
            'backward_callable_id': bwd_func_id,
            'backward_skip_vars': list(backward_skip_vars)
        })
    return out


# For debug usage
py_func.registered_func = PyFuncRegistry.registered_func
py_func.registered_func_num = PyFuncRegistry.registered_func_num


@templatedoc()
def psroi_pool(input,
               rois,
               output_channels,
               spatial_scale,
               pooled_height,
               pooled_width,
               name=None):
    """

    ${comment}

    Parameters:
        input (Variable): ${x_comment}
        rois (Variable): LoDTensor, ROIs (Regions of Interest) to pool over.It should be
                         a 2-D LoDTensor of shape (num_rois, 4), the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates. The data type is the same as `input`
        output_channels (int): ${output_channels_comment}
        spatial_scale (float): ${spatial_scale_comment} Default: 1.0
        pooled_height (int): ${pooled_height_comment} Default: 1
        pooled_width (int): ${pooled_width_comment} Default: 1
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_comment}.

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            x = fluid.data(name='x', shape=[100, 490, 28, 28], dtype='float32')
            rois = fluid.data(name='rois', shape=[None, 4], lod_level=1, dtype='float32')
            pool_out = fluid.layers.psroi_pool(x, rois, 10, 1.0, 7, 7)
    """
    helper = LayerHelper('psroi_pool', **locals())
    # check attrs
    if not isinstance(output_channels, int):
        raise TypeError("output_channels must be int type")
    if not isinstance(spatial_scale, float):
        raise TypeError("spatial_scale must be float type")
    if not isinstance(pooled_height, int):
        raise TypeError("pooled_height must be int type")
    if not isinstance(pooled_width, int):
        raise TypeError("pooled_width must be int type")
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='psroi_pool',
        inputs={'X': input,
                'ROIs': rois},
        outputs={'Out': out},
        attrs={
            'output_channels': output_channels,
            'spatial_scale': spatial_scale,
            'pooled_height': pooled_height,
            'pooled_width': pooled_width
        })
    return out


@templatedoc()
def prroi_pool(input,
               rois,
               spatial_scale=1.0,
               pooled_height=1,
               pooled_width=1,
               batch_roi_nums=None,
               name=None):
    """

    The precise roi pooling implementation for paddle. Reference: https://arxiv.org/pdf/1807.11590.pdf

    Args:
        input (Variable):The input of precise roi pooliing.The shape of input tensor is
                        [N,C,H,W]. Where N is batch size,C is number of input channels,H
                        is height of the feature, and W is the width of the feature.
        rois (Variable): ROIs (Regions of Interest) to pool over.It should be
                        a 2-D LoDTensor or Tensor of shape (num_rois, 4), the lod level
                        is 1 when it is LoDTensor. The LoD include the rois's batch index
                        information. If rois is Tensor, its batch index information should
                        be provided by batch_index.
                        Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                        the top left coordinates, and (x2, y2) is the bottom
                        right coordinates.
        spatial_scale (float): Ratio of input feature map height (or width) to raw image height (or width).
                             Equals the reciprocal of total stride in convolutional layers, Default: 1.0.
        pooled_height (integer): The pooled output height. Default: 1.
        pooled_width (integer): The pooled output width. Default: 1.
        batch_roi_nums (Variable): The number of roi for each image in batch. It
                         should be 1-D Tensor, with shape [N] and dtype int64,
                         where N is the batch size. Default: None. Be note: The lod of input should be
                         empty when batch_roi_nums has values;
        name (str, default None): The name of this operation.

    Returns:
        Variable(Tensor):The shape of the returned Tensor is (N, C, pooled_height, pooled_width), with value type float32,float16. N, C denote batch_size and channels of input respectively.

    Examples:
        .. code-block:: python

            ## prroi_pool without batch_roi_num
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 490, 28, 28], dtype='float32')
            rois = fluid.data(name='rois', shape=[None, 4], lod_level=1, dtype='float32')
            pool_out = fluid.layers.prroi_pool(x, rois, 1.0, 7, 7)

            ## prroi_pool with batch_roi_num
            batchsize=4
            x2 = fluid.data(name='x2', shape=[batchsize, 490, 28, 28], dtype='float32')
            rois2 = fluid.data(name='rois2', shape=[batchsize, 4], dtype='float32')
            batch_rois_num = fluid.data(name='rois_nums', shape=[batchsize], dtype='int64')
            pool_out2 = fluid.layers.prroi_pool(x2, rois2, 1.0, 7, 7, batch_roi_nums=batch_rois_num)


    """
    check_variable_and_dtype(input, 'input', ['float32'], 'prroi_pool')
    check_variable_and_dtype(rois, 'rois', ['float32'], 'prroi_pool')
    helper = LayerHelper('prroi_pool', **locals())
    # check attrs
    if not isinstance(spatial_scale, float):
        raise TypeError("spatial_scale must be float type")
    if not isinstance(pooled_height, int):
        raise TypeError("pooled_height must be int type")
    if not isinstance(pooled_width, int):
        raise TypeError("pooled_width must be int type")
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    inputs_op = {'X': input, 'ROIs': rois}
    if batch_roi_nums is not None:
        inputs_op['BatchRoINums'] = batch_roi_nums
    helper.append_op(
        type='prroi_pool',
        inputs=inputs_op,
        outputs={'Out': out},
        attrs={
            'spatial_scale': spatial_scale,
            'pooled_height': pooled_height,
            'pooled_width': pooled_width
        })
    return out


def pixel_shuffle(x, upscale_factor):
    """

    This op rearranges elements in a tensor of shape [N, C, H, W]
    to a tensor of shape [N, C/r**2, H*r, W*r].
    This is useful for implementing efficient sub-pixel convolution
    with a stride of 1/r.
    Please refer to the paper: `Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ .
    by Shi et. al (2016) for more details.

    Parameters:

        x(Variable): 4-D tensor, the data type should be float32 or float64.
        upscale_factor(int): factor to increase spatial resolution.

    Returns:
        Out(Variable): Reshaped tensor according to the new dimension.

    Raises:
        ValueError: If the square of upscale_factor cannot divide the channels of input.

    Examples:
        .. code-block:: python

	    # declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[2,9,4,4])
	    output = fluid.layers.pixel_shuffle(x=input, upscale_factor=3)
	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    input_data = np.random.rand(2,9,4,4).astype("float32")
	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)

 	    # print(output.shape)
	    # (2L, 1L, 12L, 12L)

    """

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'pixel_shuffle')
    helper = LayerHelper("pixel_shuffle", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(upscale_factor, int):
        raise TypeError("upscale factor must be int type")

    helper.append_op(
        type="pixel_shuffle",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"upscale_factor": upscale_factor})
    return out


def fsp_matrix(x, y):
    """

    **FSP matrix op**

    This op is used to calculate the flow of solution procedure (FSP) matrix of two 4-D Tensor feature maps.
    Given feature map x with shape [x_channel, h, w] and feature map y with shape
    [y_channel, h, w], we can get the fsp matrix of x and y in two steps:

    1. reshape x into matrix with shape [x_channel, h * w] and reshape and
       transpose y into matrix with shape [h * w, y_channel].
    2. multiply x and y to get fsp matrix with shape [x_channel, y_channel].

    The output is a batch of fsp matrices.

    Args:

        x (Variable): A 4-D Tensor feature map with shape [batch_size, x_channel, height, width].
                      A Tensor with type float32, float64.
        y (Variable): A 4-D Tensor feature map with shape [batch_size, y_channel, height, width].
                      The y_channel can be different with the x_channel of Input(X)
                      while the other dimensions must be the same with Input(X)'s. A Tensor with
                      type float32, float64.

    Returns:

        fsp matrix (Variable): The output of FSP op with shape [batch_size, x_channel, y_channel].
        The x_channel is the channel of x and the y_channel is the channel of y. A Tensor with
        type float32, float64.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.data(name='data', shape=[None, 3, 32, 32])
            feature_map_0 = fluid.layers.conv2d(data, num_filters=2,
                                                filter_size=3)
            feature_map_1 = fluid.layers.conv2d(feature_map_0, num_filters=2,
                                                filter_size=1)
            loss = fluid.layers.fsp_matrix(feature_map_0, feature_map_1)

    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'fsp_matrix')
    check_variable_and_dtype(y, 'y', ['float32', 'float64'], 'fsp_matrix')
    helper = LayerHelper('fsp_matrix', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype(
        input_param_name='x'))
    helper.append_op(type='fsp', inputs={'X': x, 'Y': y}, outputs={'Out': out})
    return out


def continuous_value_model(input, cvm, use_cvm=True):
    r"""

    **continuous_value_model layers**

    Now, this OP is used in CTR project to remove or dispose show and click value in :attr:`input`.

    :attr:`input` is an embedding vector including show and click value, whose shape is :math:`[N, D]` (N is batch size. D is `2 + embedding dim` ).
    Show and click at first two dims of embedding vector D.
    If :attr:`use_cvm` is True, it will calculate :math:`log(show)` and :math:`log(click)` , and output shape is :math:`[N, D]` .
    If :attr:`use_cvm` is False, it will remove show and click from :attr:`input` , and output shape is :math:`[N, D - 2]` .
    :attr:`cvm` is show_click info, whose shape is :math:`[N, 2]` .

    Args:
        input (Variable): The input variable. A 2-D LoDTensor with shape :math:`[N, D]` , where N is the batch size, D is `2 + the embedding dim` . `lod level = 1` .
        A Tensor with type float32, float64.
        cvm (Variable): Show and click variable. A 2-D Tensor with shape :math:`[N, 2]` , where N is the batch size, 2 is show and click.
        A Tensor with type float32, float64.
        use_cvm  (bool):  Use show_click or not. if use, the output dim is the same as input.
                          if not use, the output dim is `input dim - 2` (remove show and click)

    Returns:

        Variable: A 2-D LodTensor with shape :math:`[N, M]` . if :attr:`use_cvm` = True, M is equal to input dim D. if False, M is equal to `D - 2`. \
        A Tensor with same type as input.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          input = fluid.data(name="input", shape=[64, 1], dtype="int64")
          label = fluid.data(name="label", shape=[64, 1], dtype="int64")
          embed = fluid.layers.embedding(
                            input=input,
                            size=[100, 11],
                            dtype='float32')
          ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=[-1, 1], dtype="int64", value=1)
          show_clk = fluid.layers.cast(fluid.layers.concat([ones, label], axis=1), dtype='float32')
          show_clk.stop_gradient = True
          input_with_cvm = fluid.layers.continuous_value_model(embed, show_clk, True)

    """
    helper = LayerHelper('cvm', **locals())
    out = helper.create_variable(dtype=input.dtype)
    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'cvm')
    helper.append_op(
        type='cvm',
        inputs={'X': [input],
                'CVM': [cvm]},
        outputs={'Y': [out]},
        attrs={"use_cvm": use_cvm})
    return out


def where(condition):
    """
    Return an int64 tensor with rank 2, specifying the coordinate of true element in `condition`.

    Args:
        condition(Variable): A bool tensor with rank at least 1, the data type is bool.

    Returns:
        Variable, the output data type is int64. : The tensor variable storing a 2-D tensor, which involves all coordinate.

    Examples:
        .. code-block:: python

             import paddle.fluid as fluid
             import paddle.fluid.layers as layers
             import numpy as np

             # condition is a tensor [True, False, True]
             condition = layers.assign(np.array([1, 0, 1], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[0], [2]]

             # condition is a tensor [[True, False], [False, True]]
             condition = layers.assign(np.array([[1, 0], [0, 1]], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[0, 0], [1, 1]]

             # condition is a tensor [False, False, False]
             condition = layers.assign(np.array([0, 0, 0], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[]]

    """
    if in_dygraph_mode():
        return _C_ops.where_index(condition)

    helper = LayerHelper("where_index", **locals())

    out = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.INT64)

    helper.append_op(
        type='where_index',
        inputs={'Condition': condition},
        outputs={'Out': [out]})
    return out


@deprecated(since="2.0.0", update_to="paddle.sign")
def sign(x):
    r"""
    This OP returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero.

    Args:
        x(Variable|numpy.ndarray): The input variable could be N-D tensor or N-D numpy array, \
            the input data type is float32 or float64.

    Returns:
        Variable, the output data type is the same as input data type. : The output sign tensor with identical shape to input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          # [1.0, 0.0, -1.0]
          data = fluid.layers.sign(np.array([3.0, 0.0, -2.0], dtype='float32'))
    """

    helper = LayerHelper("sign", **locals())
    check_type(x, 'x', (Variable, np.ndarray), 'sign')
    if isinstance(x, np.ndarray):
        x = assign(x)
    check_dtype(x.dtype, 'x', ['float16', 'float32', 'float64'], 'sign')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(type='sign', inputs={'X': [x]}, outputs={'Out': [out]})

    return out


def unique(x, dtype='int32'):
    r"""
    Return a unique tensor for `x` and an index tensor pointing to this unique tensor.

    Args:
        x(Tensor): A 1-D input tensor, it's data type should be float32, float64, int32, int64.
        dtype(np.dtype|str, optional): The type of index tensor: int32, int64. Default: int32.

    Returns:
        tuple: (out, index). `out` is the unique tensor for `x`, with identical dtype to `x`, and \
            `index` is an index tensor pointing to `out`, by which user can recover the original `x` tensor.

    Examples:
        .. code-block:: python

             import numpy as np
             import paddle.fluid as fluid
             x = fluid.layers.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
             out, index = fluid.layers.unique(x) # out is [2, 3, 1, 5]; index is [0, 1, 1, 2, 3, 1]
    """

    check_variable_and_dtype(x, "x", ['float32', 'float64', 'int32', 'int64'],
                             "unique")
    helper = LayerHelper("unique", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    index = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='unique',
        inputs={'X': x},
        attrs={'dtype': convert_np_dtype_to_dtype_(dtype)},
        outputs={'Out': [out],
                 'Index': [index]})

    return out, index


def unique_with_counts(x, dtype='int32'):
    r"""
    This OP return a unique tensor for `x` , and count tensor that the count of unique result in raw input, \
    and an index tensor pointing to this unique tensor.

    **NOTICE**: This op support the variable type of Tensor only.

    Args:
        x(Variable): A 1-D input tensor with input shape of :math:`[N]` , the input data type is float32, float64, int32, int64.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of count and index tensor, it could be int32, int64. Defalut value is int32.

    Returns:
        tuple, the variable type in tuple is Tensor, the output :attr:`out` data type is the same as input :attr:`x`, \
        and data type of output :attr:`index` and :attr:`count` will be int32 or int64.: The :attr:`out` is unique tensor for input :attr:`x`,\
        the data shape is :math:`[K]`, the `K` may be different to the `N` in shape of :attr:`x`. :attr:`index` is an index tensor pointing\
        to :attr:`out`, the data shape is :math:`[N]` , the data shape is the same as input :attr:`x`. :attr:`count` is count of unique element in\
        the :attr:`x`, the data shape is :math:`[K]`, the data shape is the same as output :attr:`out`.

    Examples:
        .. code-block:: python

             import numpy as np
             import paddle.fluid as fluid
             x = fluid.layers.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
             out, index, count = fluid.layers.unique_with_counts(x) # out is [2, 3, 1, 5]; index is [0, 1, 1, 2, 3, 1]
                                                        # count is [1, 3, 1, 1]
            # x.shape=(6,) out.shape=(4,), index.shape=(6,), count.shape=(4,)
    """
    check_variable_and_dtype(x, "x", ['float32', 'float64', 'int32', 'int64'],
                             "unique_with_counts")
    if not (dtype == 'int32' or dtype == 'int64'):
        raise TypeError(
            "Op unique_with_counts, index dtype must be int32 or int64")

    if x is None or len(x.shape) != 1:
        raise ValueError(
            "Op unique_with_counts, x must not be null and size of dim must be 1"
        )

    helper = LayerHelper("unique_with_counts", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    index = helper.create_variable_for_type_inference(dtype)

    count = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='unique_with_counts',
        inputs={'X': x},
        attrs={'dtype': convert_np_dtype_to_dtype_(dtype)},
        outputs={'Out': [out],
                 'Index': [index],
                 'Count': [count]})

    return out, index, count


def deformable_conv(input,
                    offset,
                    mask,
                    num_filters,
                    filter_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=None,
                    deformable_groups=None,
                    im2col_step=None,
                    param_attr=None,
                    bias_attr=None,
                    modulated=True,
                    name=None):
    r"""
    :api_attr: Static Graph

    **Deformable Convolution op**

    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          Offset shape: :math:`(N, 2 * deformable\_groups * H_f * H_w, H_{in}, W_{in})`

          Mask shape: :math:`(N, deformable\_groups * H_f * H_w, H_{in}, W_{in})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Variable): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Variable): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        Mask (Variable, Optional): The input mask of deformable convolution layer.
            A Tensor with type float32, float64. It should be None when you use
            deformable convolution v1.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups (int): The groups number of the deformable conv layer. According to
            grouped convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        deformable_groups (int): The number of deformable group partitions.
            Default: deformable_groups = 1.
        im2col_step (int): Maximum number of images per im2col computation;
            The total batch size should be devisable by this value or smaller
            than this value; if you face out of memory problem, you can try
            to use a smaller value here.
            Default: im2col_step = 64.
        param_attr (ParamAttr, Optional): The parameter attribute for learnable parameters/weights
            of deformable conv. If it is set to None or one attribute of ParamAttr,
            deformable conv will create ParamAttr as param_attr.
            If the Initializer of the param_attr is not set, the parameter is
            initialized with :math:`Normal(0.0, std)`, and the
            :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool, Optional): The parameter attribute for the bias of
            deformable conv layer. If it is set to False, no bias will be added
            to the output units. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        modulated (bool): Make sure which version should be used between v1 and v2, where v2 is \
            used while True. Default: True.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.
    Returns:
        Variable: The tensor variable storing the deformable convolution \
                  result. A Tensor with type float32, float64.
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:
        .. code-block:: python

          #deformable conv v2:

          import paddle.fluid as fluid
          import paddle
          paddle.enable_static()
          
          C_in, H_in, W_in = 3, 32, 32
          filter_size, deformable_groups = 3, 1
          data = fluid.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
          offset = fluid.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          mask = fluid.data(name='mask', shape=[None, deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          out = fluid.layers.deformable_conv(input=data, offset=offset, mask=mask,
                                             num_filters=2, filter_size=filter_size, padding=1, modulated=True)

          #deformable conv v1:

          import paddle.fluid as fluid
          C_in, H_in, W_in = 3, 32, 32
          filter_size, deformable_groups = 3, 1
          data = fluid.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
          offset = fluid.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          out = fluid.layers.deformable_conv(input=data, offset=offset, mask=None,
                                             num_filters=2, filter_size=filter_size, padding=1, modulated=False)
    """

    check_variable_and_dtype(input, "input", ['float32', 'float64'],
                             'deformable_conv')
    check_variable_and_dtype(offset, "offset", ['float32', 'float64'],
                             'deformable_conv')
    check_type(mask, 'mask', (Variable, type(None)), 'deformable_conv')

    num_channels = input.shape[1]
    assert param_attr is not False, "param_attr should not be False here."

    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    if not isinstance(input, Variable):
        raise TypeError("Input of deformable_conv must be Variable")
    if not isinstance(offset, Variable):
        raise TypeError("Input Offset of deformable_conv must be Variable")

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    input_shape = input.shape
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        if filter_elem_num <= 0:
            raise ValueError(
                "Invalid filter number, excepted number is larger than 0, but"
                " received {}, please check the input shape and "
                "filter size.".format(filter_elem_num))
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if modulated:
        helper.append_op(
            type='deformable_conv',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
                'Mask': mask,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    else:
        helper.append_op(
            type='deformable_conv_v1',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output


def unfold(x, kernel_sizes, strides=1, paddings=0, dilations=1, name=None):
    r"""

    This op returns a col buffer of sliding local blocks of input x, also known
    as im2col for batched 2D image tensors. For each block under the convolution filter,
    all element will be rearranged as a column. While the convolution filter sliding over
    the input feature map, a series of such columns will be formed.

    For each input :math:`x` with shape [N, C, H, W], the output shape [N, Cout, Lout]
    can be calculated as following.

    .. math::

        dkernel[0] &= dilations[0] \times (kernel\_sizes[0] - 1) + 1

        dkernel[1] &= dilations[1] \times (kernel\_sizes[1] - 1) + 1

        hout &= \frac{H + paddings[0] + paddings[2] - dkernel[0]}{strides[0]} + 1

        wout &= \frac{W + paddings[1] + paddings[3] - dkernel[1]}{strides[1]} + 1

        Cout &= C \times kernel\_sizes[0] \times kernel\_sizes[1]

        Lout &= hout \times wout


    Parameters:
        x(Tensor):              4-D Tensor, input tensor of format [N, C, H, W],
                                  data type can be float32 or float64
        kernel_sizes(int|list):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list):        The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [sride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list):       The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list):      the dilations of convolution kernel, should be
                                  [dilation_h, dilation_w], or an integer dilation treated as
                                  [dilation, dilation]. For default, it will be [1, 1].
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`


    Returns:
        The tensor corresponding to the sliding local blocks.
        The output shape is [N, Cout, Lout] as decriabled above.
        Cout is the  total number of values within each block,
        and Lout is the total number of such blocks.
        The data type of output is the same as the input :math:`x`

    Return Type:
        Tensor

    Examples:

        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            x = paddle.randn((100,3,224,224))
            y = F.unfold(x, [3, 3], 1, 1, 1)
    """

    helper = LayerHelper("unfold", **locals())

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'unfold')

    assert len(x.shape) == 4, \
            "input should be the format of [N, C, H, W]"

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes, kernel_sizes]
    else:
        assert isinstance(kernel_sizes, list) and (len(kernel_sizes) == 2), \
            "kernel_sizes should either be an integer or a list of two integers"

    if isinstance(strides, int):
        strides = [strides, strides]
    else:
        assert isinstance(strides, list) and (len(strides) == 2), \
            "strides should either be an integer or a list of two integers"

    if isinstance(dilations, int):
        dilations = [dilations, dilations]
    else:
        assert isinstance(dilations, list) and (len(dilations) == 2), \
            "dilations should either be an integer or a list of two integers"

    if isinstance(paddings, int):
        paddings = [paddings] * 4
    elif isinstance(paddings, list):
        if len(paddings) == 2:
            paddings = paddings * 2
        elif len(paddings) == 4:
            pass
        else:
            raise ValueError(
                "paddings should either be an integer or a list of 2 or 4 integers"
            )
    else:
        raise ValueError(
            "Unexpected type of paddings, it should be either an integer or a list"
            "of 2 or 4 integers")

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="unfold",
        inputs={"X": x},
        outputs={"Y": out},
        attrs={
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "paddings": paddings,
            "dilations": dilations
        })
    return out


def deformable_roi_pooling(input,
                           rois,
                           trans,
                           no_trans=False,
                           spatial_scale=1.0,
                           group_size=[1, 1],
                           pooled_height=1,
                           pooled_width=1,
                           part_size=None,
                           sample_per_part=1,
                           trans_std=0.1,
                           position_sensitive=False,
                           name=None):
    r"""

    Deformable ROI Pooling Layer

    Performs deformable region-of-interest pooling on inputs. As described
    in `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_, it will get offset for each bin after
    roi pooling so that pooling at correct region. Batch_size will change to the number of region bounding boxes after deformable_roi_pooling.

    The operation has three steps:

    1. Dividing each region proposal into equal-sized sections with the pooled_width and pooled_height.

    2. Add offset to pixel in ROI to get new location and the new value which are computed directly through
       bilinear interpolation with four nearest pixel.

    3. Sample several points in each bin to get average values as output.


    Args:
        input (Variable):The input of deformable roi pooling and it is tensor which value type is float32. The shape of input is
                         [N, C, H, W]. Where N is batch size, C is number of input channels,
                         H is height of the feature, and W is the width of the feature.
        rois (Variable): ROIs (Regions of Interest) with type float32 to pool over. It should be
                         a 2-D LoDTensor of shape (num_rois, 4), and the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates, which value type is float32.
        trans (Variable): Offset of features on ROIs while pooling which value type is float32. The format is [N, C, H, W], where
                          N is number of ROIs, C is number of channels, which indicate the offset distance
                          in the x and y directions, H is pooled height, and W is pooled width.
        no_trans (bool): Whether to add offset to get new value or not while roi pooling, which value with type bool is True or False.
                         If value is True, no offset will be added in operation. Default: False.
        spatial_scale (float): Ratio of input feature map height (or width) to raw image height (or width), which value type is float32.
                         Equals the reciprocal of total stride in convolutional layers, Default: 1.0.
        group_size (list|tuple): The number of groups which input channels are divided and the input is list or tuple, which value type is int32. (eg.number of input channels
                          is k1 * k2 * (C + 1), which k1 and k2 are group width and height and C+1 is number of output
                          channels.) eg.(4, 6), which 4 is height of group and 6 is width of group. Default: [1, 1].
        pooled_height (int): The pooled output height which value type is int32. Default: 1.
        pooled_width (int): The pooled output width which value type is int32. Default: 1.
        part_size (list|tuple): The height and width of offset which values in list or tuple is int32, eg.(4, 6), which height is 4 and width is 6, and values always equal to pooled_height \
                         and pooled_width. Default: if None, default value is [pooled_height, pooled_width].
        sample_per_part (int): The number of samples in each bin which value type is int32. If value is bigger, it will consume more performance. Default: 1.
        trans_std (float): Coefficient of offset which value type is float32. It controls weight of offset. Default: 0.1.
        position_sensitive (bool): Whether to choose deformable psroi pooling mode or not, and value type is bool(True or False). If value is False, input dimension equals to output dimension. \
                                   If value is True, input dimension should be output dimension * pooled_height * pooled_width. Default: False.
        name (str|None): Name of layer. Default: None.
    Returns:
        Variable: Output of deformable roi pooling is that, if position sensitive is False, input dimension equals to output dimension. If position sensitive is True,\
                  input dimension should be the result of output dimension divided by pooled height and pooled width.

    Examples:
      .. code-block:: python

        # position_sensitive=True
        import paddle.fluid as fluid
        input = fluid.data(name="input",
                           shape=[2, 192, 64, 64],
                           dtype='float32')
        rois = fluid.data(name="rois",
                          shape=[-1, 4],
                          dtype='float32',
                          lod_level=1)
        trans = fluid.data(name="trans",
                           shape=[2, 384, 64, 64],
                           dtype='float32')
        x = fluid.layers.deformable_roi_pooling(input=input,
                                                rois=rois,
                                                trans=trans,
                                                no_trans=False,
                                                spatial_scale=1.0,
                                                group_size=(1, 1),
                                                pooled_height=8,
                                                pooled_width=8,
                                                part_size=(8, 8),
                                                sample_per_part=4,
                                                trans_std=0.1,
                                                position_sensitive=True)

        # position_sensitive=False
        import paddle.fluid as fluid
        input = fluid.data(name="input",
                           shape=[2, 192, 64, 64],
                           dtype='float32')
        rois = fluid.data(name="rois",
                          shape=[-1, 4],
                          dtype='float32',
                          lod_level=1)
        trans = fluid.data(name="trans",
                           shape=[2, 384, 64, 64],
                           dtype='float32')
        x = fluid.layers.deformable_roi_pooling(input=input,
                                                rois=rois,
                                                trans=trans,
                                                no_trans=False,
                                                spatial_scale=1.0,
                                                group_size=(1, 1),
                                                pooled_height=8,
                                                pooled_width=8,
                                                part_size=(8, 8),
                                                sample_per_part=4,
                                                trans_std=0.1,
                                                position_sensitive=False)
    """

    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'deformable_roi_pooling')
    check_variable_and_dtype(rois, 'rois', ['float32', 'float64'],
                             'deformable_roi_pooling')
    check_variable_and_dtype(trans, 'trans', ['float32', 'float64'],
                             'deformable_roi_pooling')
    check_type(group_size, 'group_size', (list, tuple),
               'deformable_roi_pooling')
    if part_size is not None:
        check_type(part_size, 'part_size', (list, tuple),
                   'deformable_roi_pooling')

    input_channels = input.shape[1]
    if position_sensitive == False:
        output_channels = input_channels
    else:
        output_channels = input_channels / pooled_height / pooled_width

    if part_size is None:
        part_height = pooled_height
        part_width = pooled_width
        part_size = [part_height, part_width]
    part_size = utils.convert_to_list(part_size, 2, 'part_size')
    group_size = utils.convert_to_list(group_size, 2, 'group_size')
    helper = LayerHelper('deformable_psroi_pooling', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    top_count = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="deformable_psroi_pooling",
        inputs={"Input": input,
                "ROIs": rois,
                "Trans": trans},
        outputs={"Output": output,
                 "TopCount": top_count},
        attrs={
            "no_trans": no_trans,
            "spatial_scale": spatial_scale,
            "output_dim": output_channels,
            "group_size": group_size,
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "part_size": part_size,
            "sample_per_part": sample_per_part,
            "trans_std": trans_std
        })
    return output


@deprecated(since="2.0.0", update_to="paddle.shard_index")
def shard_index(input, index_num, nshards, shard_id, ignore_value=-1):
    """
    Reset the values of `input` according to the shard it beloning to.
    Every value in `input` must be a non-negative integer, and
    the parameter `index_num` represents the integer above the maximum
    value of `input`. Thus, all values in `input` must be in the range
    [0, index_num) and each value can be regarded as the offset to the beginning
    of the range. The range is further split into multiple shards. Specifically,
    we first compute the `shard_size` according to the following formula,
    which represents the number of integers each shard can hold. So for the
    i'th shard, it can hold values in the range [i*shard_size, (i+1)*shard_size).
    ::

        shard_size = (index_num + nshards - 1) // nshards

    For each value `v` in `input`, we reset it to a new value according to the
    following formula:
    ::
   
        v = v - shard_id * shard_size if shard_id * shard_size <= v < (shard_id+1) * shard_size else ignore_value

    That is, the value `v` is set to the new offset within the range represented by the shard `shard_id`
    if it in the range. Otherwise, we reset it to be `ignore_value`.

    Args:
        input (Tensor): Input tensor with data type int64 or int32. It's last dimension must be 1.
        index_num (int): An integer represents the integer above the maximum value of `input`.
        nshards (int): The number of shards.
        shard_id (int): The index of the current shard.
        ignore_value (int): An integer value out of sharded index range.

    Returns:
        Tensor.

    Examples:
        .. code-block:: python

            import paddle
            label = paddle.to_tensor([[16], [1]], "int64")
            shard_label = paddle.shard_index(input=label,
                                             index_num=20,
                                             nshards=2,
                                             shard_id=0)
            print(shard_label)
            # [[-1], [1]]
    """
    check_variable_and_dtype(input, 'input', ['int64', 'int32'], 'shard_index')
    op_type = 'shard_index'
    helper = LayerHelper(op_type, **locals())
    if shard_id < 0 or shard_id >= nshards:
        raise ValueError('The shard_id(%d) should be in [0, %d)' %
                         (shard_id, nshards))

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'X': [input]},
        outputs={'Out': out},
        attrs={
            'index_num': index_num,
            'nshards': nshards,
            'shard_id': shard_id,
            'ignore_value': ignore_value
        },
        stop_gradient=True)
    return out


@templatedoc()
def hard_swish(x, threshold=6.0, scale=6.0, offset=3.0, name=None):
    r"""
    This operator implements the hard_swish activation function.
    Hard_swish is proposed in MobileNetV3, and performs better in computational stability and efficiency compared to swish function.
    For more details please refer to: https://arxiv.org/pdf/1905.02244.pdf

    The formula is as follows:

    .. math::

        out = \\frac{x * (min(max(0, x+offset), threshold))}{scale}

    In the above equation:

    ``threshold`` and ``scale`` should be positive, ``offset`` can be positive or negative. It is recommended to use default parameters.

    Args:
        x (Variable): Input feature, multi-dimensional Tensor. The data type should be float32 or float64.
        threshold (float, optional): The threshold in Relu function. Default: 6.0
        scale (float, optional): The scale factor. Default: 6.0
        offset (float, optional): The offset factor. Default: 3.0
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: The output tensor with the same shape and data type as input.


    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import paddle
        import numpy as np
        paddle.enable_static()

        DATATYPE='float32'

        x_data = np.array([i for i in range(1,5)]).reshape([1,1,4]).astype(DATATYPE)

        x = fluid.data(name="x", shape=[None,1,4], dtype=DATATYPE)
        y = fluid.layers.hard_swish(x)

        place = fluid.CPUPlace()
        #place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        out, = exe.run(feed={'x':x_data}, fetch_list=[y.name])
        print(out)  # [[0.66666667, 1.66666667,3., 4.]]
    """
    if in_dygraph_mode():
        return _C_ops.hard_swish(x, 'threshold', threshold, 'scale', scale,
                                 'offset', offset)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'hard_swish')

    helper = LayerHelper('hard_swish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='hard_swish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold,
               'scale': scale,
               'offset': offset})
    return out


@templatedoc()
def mish(x, threshold=20, name=None):
    r"""
    This operator implements the mish activation function.
    Refer to `Mish: A Self Regularized Non-Monotonic Neural
    Activation Function <https://arxiv.org/abs/1908.08681>`_


    The formula is as follows if :attr:`threshold` is :code:`None` or negative:

    .. math::

        out = x * \\tanh(\\ln(1 + e^{x}))

    The formula is as follows if :attr:`threshold` is set as positive value:

    .. math::

	out = \\begin{cases}
		x \\ast \\tanh(x), \\text{if } x > \\text{threshold} \\\\
		x \\ast \\tanh(e^{x}), \\text{if } x < -\\text{threshold} \\\\
		x \\ast \\tanh(\\ln(1 + e^{x})),  \\text{otherwise}
	      \\end{cases}

    Args:
        x (Variable): Input feature, multi-dimensional Tensor. The data type
                      should be float16, float32 or float64.
        threshold (float|None): threshold for softplus in Mish operator.
                Approximate value of softplus will be used if absolute value
                of input is greater than :attr:threshold and :attr:threshold
                is set as positive value. For none or negative threshold,
                approximate value is not used. Default 20.
        name (str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`

    Returns:
        Variable: The output tensor with the same shape and data type as input.


    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        DATATYPE='float32'

        x_data = np.array([i for i in range(1,5)]).reshape([1,1,4]).astype(DATATYPE)

        x = fluid.data(name="x", shape=[None,1,4], dtype=DATATYPE)
        y = fluid.layers.mish(x)

        place = fluid.CPUPlace()
        # place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        out, = exe.run(feed={'x':x_data}, fetch_list=[y.name])
        print(out)  # [[0.66666667, 1.66666667, 3., 4.]]
    """
    if in_dygraph_mode():
        return _C_ops.mish(x, 'threshold', threshold)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'mish')
    check_type(threshold, 'threshold', (float, int), 'mish')
    assert threshold > 0, "threshold of mish should be greater than 0, " \
                          "but got {}".format(threshold)

    helper = LayerHelper('mish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='mish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold})
    return out


def gather_tree(ids, parents):
    r"""
    To be used after beam search. After beam search, we get selected ids at
    each time step and the corresponding parents in the search tree. Both ids
    and parents have the layout :attr:`[max_time, batch_size, beam_size]`. Then
    :attr:`gather_tree` is used to backtrace from the last time step and
    generate the full sequences by collecting selected ids.

    Here is an example:

    .. code-block:: text

            Given:
                ids = [[[2 2]
                        [6 1]]
                       [[3 9]
                        [6 1]]
                       [[0 1]
                        [9 0]]]
                parents = [[[0 0]
                            [1 1]]
                           [[1 0]
                            [1 0]]
                           [[0 0]
                            [0 1]]]

            Then:
                gather_tree(ids, parents)
                         = [[[2 2]
                             [1 6]]
                            [[3 3]
                             [6 1]]
                            [[0 1]
                             [9 0]]]

    Args:
        ids(Tensor): A Tensor with shape :attr:`[length, batch_size, beam_size]`
            and data type :attr:`int32` or :attr:`int64`. It contains the selected
            ids of all time steps.
        parents(Tensor): A Tensor with the same shape and data type as :attr:`ids`,
            It contains the parents corresponding to selected ids when searching
            among beams.

    Returns:
            A Tensor with the same shape and data type as :attr:`ids`. \
            It contains the full sequences. The sequences are collected from \
            :attr:`ids` by backtracing according to :attr:`parents`.

    Examples:
        .. code-block:: python

            import paddle

            ids = paddle.to_tensor([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]])

            parents = paddle.to_tensor([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]])

            final_sequences = paddle.nn.functional.gather_tree(ids, parents)
            # [[[2, 2], [1, 6]], [[3, 3], [6, 1]], [[0, 1], [9, 0]]]

    """
    if in_dygraph_mode():
        return _C_ops.gather_tree(ids, parents)
    else:
        helper = LayerHelper('gather_tree', **locals())
        check_variable_and_dtype(ids, 'ids', ['int32', 'int64'], 'gather_tree')
        check_variable_and_dtype(parents, 'parents', ['int32', 'int64'],
                                 'gather_tree')
        out = helper.create_variable_for_type_inference(dtype=ids.dtype)

        helper.append_op(
            type="gather_tree",
            inputs={"Ids": ids,
                    "Parents": parents},
            outputs={"Out": out})

        return out


@deprecated(since="2.0.0", update_to="paddle.uniform")
@templatedoc()
def uniform_random(shape, dtype='float32', min=-1.0, max=1.0, seed=0,
                   name=None):
    """
    This OP returns a Tensor filled with random values sampled from a uniform
    distribution in the range [``min``, ``max``), with ``shape`` and ``dtype``.

    Examples:
    ::

        Input:
          shape = [1, 2]

        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape(list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of
            the output Tensor. Supported data types: float32, float64.
            Default is float32.
        min(float|int, optional): The lower bound on the range of random values
            to generate, ``min`` is included in the range. Default is -1.0.
        max(float|int, optional): The upper bound on the range of random values
            to generate, ``max`` is excluded in the range. Default is 1.0.
        seed(int, optional): Random seed used for generating samples. 0 means
            use a seed generated by the system. Note that if seed is not 0,
            this operator will always generate the same random numbers every
            time. Default is 0.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [``min``, ``max``), with ``shape`` and ``dtype``.

    Raises:
        TypeError: If ``shape`` is not list, tuple, Tensor.
        TypeError: If ``dtype`` is not float32, float64.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            # example 1:
            # attr shape is a list which doesn't contain Tensor.
            result_1 = fluid.layers.uniform_random(shape=[3, 4])
            # [[ 0.84524226,  0.6921872,   0.56528175,  0.71690357],
            #  [-0.34646994, -0.45116323, -0.09902662, -0.11397249],
            #  [ 0.433519,    0.39483607, -0.8660099,   0.83664286]]

            # example 2:
            # attr shape is a list which contains Tensor.
            dim_1 = fluid.layers.fill_constant([1], "int64", 2)
            dim_2 = fluid.layers.fill_constant([1], "int32", 3)
            result_2 = fluid.layers.uniform_random(shape=[dim_1, dim_2])
            # [[-0.9951253,   0.30757582, 0.9899647 ],
            #  [ 0.5864527,   0.6607096,  -0.8886161 ]]

            # example 3:
            # attr shape is a Tensor, the data type must be int64 or int32.
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = fluid.layers.uniform_random(var_shape)
            # if var_shape's value is [2, 3]
            # result_3 is:
            # [[-0.8517412,  -0.4006908,   0.2551912 ],
            #  [ 0.3364414,   0.36278176, -0.16085452]]

    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils.convert_shape_to_list(shape)
        return _C_ops.uniform_random('shape', shape, 'min',
                                     float(min), 'max',
                                     float(max), 'seed', seed, 'dtype', dtype)

    check_type(shape, 'shape', (list, tuple, Variable), 'uniform_random/rand')
    check_dtype(dtype, 'dtype', ('float32', 'float64', 'uint16'),
                'uniform_random/rand')

    inputs = dict()
    attrs = {'seed': seed, 'min': min, 'max': max, 'dtype': dtype}
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='uniform_random/rand')

    helper = LayerHelper("uniform_random", **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="uniform_random", inputs=inputs, attrs=attrs,
        outputs={"Out": out})
    utils.try_set_static_shape_tensor(out, shape)
    return out


def unbind(input, axis=0):
    """
    Removes a tensor dimension, then split the input tensor into multiple sub-Tensors.
    Args:
        input (Variable): The input variable which is an N-D Tensor, data type being float32, float64, int32 or int64.
       
        axis (int32|int64, optional): A scalar with type ``int32|int64`` shape [1]. The dimension along which to unbind. If :math:`axis < 0`, the
            dimension to unbind along is :math:`rank(input) + axis`. Default is 0.
    Returns:
        list(Variable): The list of segmented Tensor variables.

    Example:
        .. code-block:: python
            import paddle
            # input is a variable which shape is [3, 4, 5]
            input = paddle.fluid.data(
                 name="input", shape=[3, 4, 5], dtype="float32")
            [x0, x1, x2] = paddle.tensor.unbind(input, axis=0)
            # x0.shape [4, 5]
            # x1.shape [4, 5]
            # x2.shape [4, 5]
            [x0, x1, x2, x3] = paddle.tensor.unbind(input, axis=1)
            # x0.shape [3, 5]
            # x1.shape [3, 5]
            # x2.shape [3, 5]
            # x3.shape [3, 5]

    """
    helper = LayerHelper("unbind", **locals())
    check_type(input, 'input', (Variable), 'unbind')
    dtype = helper.input_dtype()
    check_dtype(dtype, 'unbind', ['float32', 'float64', 'int32', 'int64'],
                'unbind')
    if not isinstance(axis, (int)):
        raise TypeError("The type of 'axis'  must be int, but received %s." %
                        (type(axis)))
    if isinstance(axis, np.generic):
        axis = np.asscalar(axis)
    input_shape = input.shape
    axis_ = axis if axis >= 0 else len(input_shape) + axis
    num = input_shape[axis_]
    outs = [
        helper.create_variable_for_type_inference(dtype=helper.input_dtype())
        for i in range(num)
    ]

    helper.append_op(
        type="unbind",
        inputs={"X": input},
        outputs={"Out": outs},
        attrs={"axis": axis})
    return outs
