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
import six
from . import layers
from .data_feeder import check_variable_and_dtype, convert_dtype
from ..utils import deprecated

__all__ = [
    "simple_img_conv_pool",
    "sequence_conv_pool",
    "glu",
    "scaled_dot_product_attention",
    "img_conv_group",
]


def simple_img_conv_pool(input,
                         num_filters,
                         filter_size,
                         pool_size,
                         pool_stride,
                         pool_padding=0,
                         pool_type='max',
                         global_pooling=False,
                         conv_stride=1,
                         conv_padding=0,
                         conv_dilation=1,
                         conv_groups=1,
                         param_attr=None,
                         bias_attr=None,
                         act=None,
                         use_cudnn=True):
    r"""
	:api_attr: Static Graph

    The simple_img_conv_pool api is composed of :ref:`api_fluid_layers_conv2d` and :ref:`api_fluid_layers_pool2d` .

    Args:
        input (Variable): 4-D Tensor, shape is [N, C, H, W], data type can be float32 or float64.
        num_filters(int): The number of filters. It is the same as the output channels.
        filter_size (int|list|tuple): The filter size. If filter_size is a list or
            tuple, it must contain two integers, (filter_size_H, filter_size_W). Otherwise,
            the filter_size_H = filter_size_W = filter_size.
        pool_size (int|list|tuple): The pooling size of pool2d layer. If pool_size
            is a list or tuple, it must contain two integers, (pool_size_H, pool_size_W).
            Otherwise, the pool_size_H = pool_size_W = pool_size.
        pool_stride (int|list|tuple): The pooling stride of pool2d layer. If pool_stride
            is a list or tuple, it must contain two integers, (pooling_stride_H, pooling_stride_W).
            Otherwise, the pooling_stride_H = pooling_stride_W = pool_stride.
        pool_padding (int|list|tuple): The padding of pool2d layer. If pool_padding is a list or
            tuple, it must contain two integers, (pool_padding_H, pool_padding_W).
            Otherwise, the pool_padding_H = pool_padding_W = pool_padding. Default 0.
        pool_type (str): Pooling type can be :math:`max` for max-pooling or :math:`avg` for
            average-pooling. Default :math:`max`.
        global_pooling (bool): Whether to use the global pooling. If global_pooling = true,
            pool_size and pool_padding while be ignored. Default False
        conv_stride (int|list|tuple): The stride size of the conv2d Layer. If stride is a
            list or tuple, it must contain two integers, (conv_stride_H, conv_stride_W). Otherwise,
            the conv_stride_H = conv_stride_W = conv_stride. Default: conv_stride = 1.
        conv_padding (int|list|tuple): The padding size of the conv2d Layer. If padding is
            a list or  tuple, it must contain two integers, (conv_padding_H, conv_padding_W).
            Otherwise, the conv_padding_H = conv_padding_W = conv_padding. Default: conv_padding = 0.
        conv_dilation (int|list|tuple): The dilation size of the conv2d Layer. If dilation is
            a list or tuple, it must contain two integers, (conv_dilation_H, conv_dilation_W).
            Otherwise, the conv_dilation_H = conv_dilation_W = conv_dilation. Default: conv_dilation = 1.
        conv_groups (int): The groups number of the conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`.
            Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        act (str): Activation type for conv2d, if it is set to None, activation is not
            appended. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True

    Return:
        4-D Tensor, the result of input after conv2d and pool2d, with the same data type as :attr:`input`

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            img = fluid.data(name='img', shape=[100, 1, 28, 28], dtype='float32')
            conv_pool = fluid.nets.simple_img_conv_pool(input=img,
                                                        filter_size=5,
                                                        num_filters=20,
                                                        pool_size=2,
                                                        pool_stride=2,
                                                        act="relu")
    """
    conv_out = layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=conv_stride,
        padding=conv_padding,
        dilation=conv_dilation,
        groups=conv_groups,
        param_attr=param_attr,
        bias_attr=bias_attr,
        act=act,
        use_cudnn=use_cudnn)

    pool_out = layers.pool2d(
        input=conv_out,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        pool_padding=pool_padding,
        global_pooling=global_pooling,
        use_cudnn=use_cudnn)
    return pool_out


def img_conv_group(input,
                   conv_num_filter,
                   pool_size,
                   conv_padding=1,
                   conv_filter_size=3,
                   conv_act=None,
                   param_attr=None,
                   conv_with_batchnorm=False,
                   conv_batchnorm_drop_rate=0.0,
                   pool_stride=1,
                   pool_type="max",
                   use_cudnn=True):
    """
	:api_attr: Static Graph

    The Image Convolution Group is composed of Convolution2d, BatchNorm, DropOut,
    and Pool2D. According to the input arguments, img_conv_group will do serials of
    computation for Input using Convolution2d, BatchNorm, DropOut, and pass the last
    result to Pool2D.

    Args:
        input (Variable): The input is 4-D Tensor with shape [N, C, H, W], the data type of input is float32 or float64.
        conv_num_filter(list|tuple): Indicates the numbers of filter of this group.
        pool_size (int|list|tuple): The pooling size of Pool2D Layer. If pool_size
            is a list or tuple, it must contain two integers, (pool_size_height, pool_size_width).
            Otherwise, the pool_size_height = pool_size_width = pool_size.
        conv_padding (int|list|tuple): The padding size of the Conv2D Layer. If padding is
            a list or tuple, its length must be equal to the length of conv_num_filter.
            Otherwise the conv_padding of all Conv2D Layers are the same. Default 1.
        conv_filter_size (int|list|tuple): The filter size. If filter_size is a list or
            tuple, its length must be equal to the length of conv_num_filter.
            Otherwise the conv_filter_size of all Conv2D Layers are the same. Default 3.
        conv_act (str): Activation type for Conv2D Layer that is not followed by BatchNorm.
            Default: None.
        param_attr (ParamAttr): The parameters to the Conv2D Layer. Default: None
        conv_with_batchnorm (bool|list): Indicates whether to use BatchNorm after Conv2D Layer.
            If conv_with_batchnorm is a list, its length must be equal to the length of
            conv_num_filter. Otherwise, conv_with_batchnorm indicates whether all the
            Conv2D Layer follows a BatchNorm. Default False.
        conv_batchnorm_drop_rate (float|list): Indicates the drop_rate of Dropout Layer
            after BatchNorm. If conv_batchnorm_drop_rate is a list, its length must be
            equal to the length of conv_num_filter. Otherwise, drop_rate of all Dropout
            Layers is conv_batchnorm_drop_rate. Default 0.0.
        pool_stride (int|list|tuple): The pooling stride of Pool2D layer. If pool_stride
            is a list or tuple, it must contain two integers, (pooling_stride_H,
            pooling_stride_W). Otherwise, the pooling_stride_H = pooling_stride_W = pool_stride.
            Default 1.
        pool_type (str): Pooling type can be :math:`max` for max-pooling and :math:`avg` for
            average-pooling. Default :math:`max`.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True

    Return:
        A Variable holding Tensor representing the final result after serial computation using Convolution2d,
        BatchNorm, DropOut, and Pool2D, whose data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            
            img = fluid.data(name='img', shape=[None, 1, 28, 28], dtype='float32')
            conv_pool = fluid.nets.img_conv_group(input=img,
                                                  conv_padding=1,
                                                  conv_num_filter=[3, 3],
                                                  conv_filter_size=3,
                                                  conv_act="relu",
                                                  pool_size=2,
                                                  pool_stride=2)
    """
    tmp = input
    assert isinstance(conv_num_filter, list) or \
        isinstance(conv_num_filter, tuple)

    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            assert len(obj) == len(conv_num_filter)
            return obj

    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    param_attr = __extend_list__(param_attr)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)

    for i in six.moves.range(len(conv_num_filter)):
        local_conv_act = conv_act
        if conv_with_batchnorm[i]:
            local_conv_act = None

        tmp = layers.conv2d(
            input=tmp,
            num_filters=conv_num_filter[i],
            filter_size=conv_filter_size[i],
            padding=conv_padding[i],
            param_attr=param_attr[i],
            act=local_conv_act,
            use_cudnn=use_cudnn)

        if conv_with_batchnorm[i]:
            tmp = layers.batch_norm(input=tmp, act=conv_act)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = layers.dropout(x=tmp, dropout_prob=drop_rate)

    pool_out = layers.pool2d(
        input=tmp,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        use_cudnn=use_cudnn)
    return pool_out


def sequence_conv_pool(input,
                       num_filters,
                       filter_size,
                       param_attr=None,
                       act="sigmoid",
                       pool_type="max",
                       bias_attr=None):
    """
	:api_attr: Static Graph

    **This api takes input as an LoDTensor. If input is a Tensor, please use** 
    :ref:`api_fluid_nets_simple_img_conv_pool` **instead**

    The sequence_conv_pool is composed of :ref:`api_fluid_layers_sequence_conv` 
    and :ref:`api_fluid_layers_sequence_pool` .

    Args:
        input (Variable): 2-D LoDTensor, the input of sequence_conv, 
            which supports variable-time length input sequence. 
            The underlying of input is a matrix with shape
            (T, N), where T is the total time steps in this mini-batch and N is
            the input_hidden_size. The data type is float32 or float64.
        num_filters(int): The number of filter.
        filter_size (int): The filter size.
        param_attr (ParamAttr): The parameters of the sequence_conv Layer. Default: None.
        act (str|None): Activation type for Sequence_conv Layer. 
                        If set to None, no activation will be applied. Default: "sigmoid".
        pool_type (str): Pooling type can be :math:`max` for max-pooling, :math:`average` for
            average-pooling, :math:`sum` for sum-pooling, :math:`sqrt` for sqrt-pooling.
            Default :math:`max`.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of sequence_conv.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, sequence_conv
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.

    Returns:
        The final result after sequence_conv and sequence_pool. 
        It is a 2-D Tensor, with the same data type as :attr:`input`

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            input_dim = 100 #len(word_dict)
            emb_dim = 128
            hid_dim = 512
            data = fluid.data(name="words", shape=[None, 1], dtype="int64", lod_level=1)
            emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
            seq_conv = fluid.nets.sequence_conv_pool(input=emb,
                                                     num_filters=hid_dim,
                                                     filter_size=3,
                                                     act="tanh",
                                                     pool_type="sqrt")
    """

    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'input')
    conv_out = layers.sequence_conv(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=param_attr,
        bias_attr=bias_attr,
        act=act)

    pool_out = layers.sequence_pool(input=conv_out, pool_type=pool_type)
    return pool_out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.glu")
def glu(input, dim=-1):
    r"""
	:api_attr: Static Graph

    The Gated Linear Units(GLU) composed by :ref:`api_fluid_layers_split` , 
    :ref:`api_fluid_layers_sigmoid`  and :ref:`api_fluid_layers_elementwise_mul` . 
    Specifically, GLU will plit the input into two equal-sized parts,
    :math:`a` and :math:`b`, along the given dimension and then compute as
    following:

        .. math::

            {GLU}(a, b)= a \otimes \sigma(b)

    Refer to `Language Modeling with Gated Convolutional Networks
    <https://arxiv.org/pdf/1612.08083.pdf>`_.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor. 
                          The supported data types include float32, float64 
                          and float16 (only for GPU).
        dim (int, optional): The dimension along which to split. If :math:`dim < 0`, the
            dimension to split along is :math:`rank(input) + dim`. Default -1.

    Returns:
        Variable: Variable with half the size and same data type of input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            
            data = fluid.data(
                name="words", shape=[-1, 6, 3, 9], dtype="float32")
            # shape of output: [-1, 3, 3, 9]
            output = fluid.nets.glu(input=data, dim=1)
    """
    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             "glu")
    a, b = layers.split(input, num_or_sections=2, dim=dim)
    act_b = layers.sigmoid(x=b)
    out = layers.elementwise_mul(x=a, y=act_b)
    return out


def scaled_dot_product_attention(queries,
                                 keys,
                                 values,
                                 num_heads=1,
                                 dropout_rate=0.):
    r"""
	:api_attr: Static Graph

    This interface Multi-Head Attention using scaled dot product.
    Attention mechanism can be seen as mapping a query and a set of key-value
    pairs to an output. Multi-Head Attention performs attention using multi-head
    parallel, and the inputs of attention would be transformed by linear projection.
    The formula is as follows:

    .. math::

        MultiHead(Q, K, V ) & = Concat(head_1, ..., head_h)

        where \  head_i & = Attention(QW_i^Q , KW_i^K , VW_i^V )

        Attention(Q, K, V) & = softmax (\\frac{QK^\mathrm{T}}{\sqrt{d_k}}) V

    For more details, please refer to `Attention Is All You Need
    <https://arxiv.org/pdf/1706.03762.pdf>`_ .

    Note that the implementation is adapted to batch, and all matrix multiplication
    in :math:`Attention(Q, K, V)` is batched matrix multiplication. Refer to
    :ref:`api_fluid_layers_matmul` .

    Args:
        queries (Variable): A 3-D Tensor with shape :math:`[N, L_q, d_k \\times h]` ,
            where :math:`N` stands for batch size, :math:`L_q` for the sequence length
            of query, :math:`d_k \\times h` for the feature size of query, :math:`h` for
            head number. The data type should be float32 or float64.
        keys (Variable): A 3-D Tensor with shape :math:`[N, L_k, d_k \\times h]` ,
            where :math:`N` stands for batch size, :math:`L_k` for the sequence length
            of key, :math:`d_k \\times h` for the feature size of key, :math:`h` for head
            number. The data type should be the same as ``queries`` .
        values (Variable): A 3-D Tensor with shape :math:`[N, L_k, d_v \\times h]` ,
            where :math:`N` stands for batch size, :math:`L_k` for the sequence length
            of key, :math:`d_v \\times h` for the feature size of value, :math:`h` for head
            number. The data type should be the same as ``queries`` .
        num_heads (int, optional): Indicate the number of head. If the number
            is 1, linear projection would not be performed on inputs. Default: 1.
        dropout_rate (float, optional): The rate to drop the attention weight.
            Default: 0.0, which means no dropout.

    Returns:
        Variable: A 3-D Tensor with shape :math:`[N, L_q, d_v \\times h]` , \
            where :math:`N` stands for batch size, :math:`L_q` for the sequence \
            length of query, :math:`d_v \\times h` for the feature size of value. \
            It has the same data type with inputs, representing the output of \
            Multi-Head Attention.

    Raises:
        TypeError: The dtype of inputs keys, values and queries should be the same.
        ValueError: Inputs queries, keys and values should all be 3-D tensors.
        ValueError: The hidden size of queries and keys should be the same.
        ValueError: The max sequence length in value batch and in key batch should be the same.
        ValueError: he hidden size of keys must be divisible by the number of attention heads.
        ValueError: he hidden size of values must be divisible by the number of attention heads.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            
            queries = fluid.data(name="queries", shape=[3, 5, 9], dtype="float32")
            keys = fluid.data(name="keys", shape=[3, 6, 9], dtype="float32")
            values = fluid.data(name="values", shape=[3, 6, 10], dtype="float32")
            contexts = fluid.nets.scaled_dot_product_attention(queries, keys, values)
            contexts.shape  # [3, 5, 10]
    """
    check_variable_and_dtype(queries, 'queries', ['float32', 'float64'],
                             "scaled_dot_product_attention")
    check_variable_and_dtype(keys, 'keys', ['float32', 'float64'],
                             "scaled_dot_product_attention")
    check_variable_and_dtype(values, 'values', ['float32', 'float64'],
                             "scaled_dot_product_attention")

    if not (queries.dtype == keys.dtype == values.dtype):
        raise TypeError(
            "The dtype of keys, values and queries should be the same."
            "But received queries.dtype = %s, "
            " keys.dtype = %s, values.dtype) = %s." %
            (convert_dtype(queries.dtype), convert_dtype(keys.dtype),
             convert_dtype(values.dtype)))

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs queries, keys and values should all be 3-D tensors."
            "But received len(queries.shape) = %d, "
            "len(keys.shape) = %d, len(values.shape) = %d." %
            (len(queries.shape), len(keys.shape), len(values.shape)))

    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            "The hidden size of queries and keys should be the same."
            "But received queries' hidden size = %d and keys' hidden size = %d."
            % (queries.shape[-1], keys.shape[-1]))
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(
            "The max sequence length in value batch and in key batch "
            "should be the same. But received max sequence length in value batch "
            "= %d, in key batch = %d." % (values.shape[-2], keys.shape[-2]))
    if keys.shape[-1] % num_heads != 0:
        raise ValueError("The hidden size of keys (%d) must be divisible "
                         "by the number of attention heads (%d)." %
                         (keys.shape[-1], num_heads))
    if values.shape[-1] % num_heads != 0:
        raise ValueError("The hidden size of values (%d) must be divisible "
                         "by the number of attention heads (%d)." %
                         (values.shape[-1], num_heads))

    def __compute_qkv(queries, keys, values, num_heads):
        """
        Add linear projection to queries, keys, and values.

        Args:
            queries(Tensor): a 3-D input Tensor.
            keys(Tensor): a 3-D input Tensor.
            values(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads. Linearly project the inputs
                            ONLY when num_heads > 1.

        Returns:
            Tensor: linearly projected output Tensors: queries', keys' and
                    values'. They have the same shapes with queries, keys and
                    values.
        """

        if num_heads == 1:
            return queries, keys, values

        q = layers.fc(input=queries, size=queries.shape[-1], num_flatten_dims=2)
        k = layers.fc(input=keys, size=keys.shape[-1], num_flatten_dims=2)
        v = layers.fc(input=values, size=values.shape[-1], num_flatten_dims=2)
        return q, k, v

    def __split_heads(x, num_heads):
        """
        Reshape the last dimension of input tensor x so that it becomes two
        dimensions.

        Args:
            x(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads.

        Returns:
            Tensor: a Tensor with shape [..., n, m/num_heads], where m is size
                    of the last dimension of x.
        """
        if num_heads == 1:
            return x

        hidden_size = x.shape[-1]
        # reshape the 3-D input: [batch_size, max_sequence_length, hidden_dim]
        # into a 4-D output:
        # [batch_size, max_sequence_length, num_heads, hidden_size_per_head].
        reshaped = layers.reshape(
            x=x,
            shape=list(x.shape[:-1]) + [num_heads, hidden_size // num_heads])

        # permute the dimensions into:
        # [batch_size, num_heads, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Reshape the last two dimensions of input tensor x so that it becomes
        one dimension.

        Args:
            x(Tensor): a 4-D input Tensor with shape
                       [bs, num_heads, max_sequence_length, hidden_dim].

        Returns:
            Tensor: a Tensor with shape
                    [bs, max_sequence_length, num_heads * hidden_dim].
        """

        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        return layers.reshape(
            x=trans_x,
            shape=list(
                map(int, [
                    trans_x.shape[0], trans_x.shape[1], trans_x.shape[2] *
                    trans_x.shape[3]
                ])))

    q, k, v = __compute_qkv(queries, keys, values, num_heads)

    q = __split_heads(q, num_heads)
    k = __split_heads(k, num_heads)
    v = __split_heads(v, num_heads)

    key_dim_per_head = keys.shape[-1] // num_heads
    scaled_q = layers.scale(x=q, scale=key_dim_per_head**-0.5)
    product = layers.matmul(x=scaled_q, y=k, transpose_y=True)

    weights = layers.reshape(
        x=layers.reshape(
            x=product, shape=[-1, product.shape[-1]], act="softmax"),
        shape=product.shape)
    if dropout_rate:
        weights = layers.dropout(
            weights, dropout_prob=dropout_rate, is_test=False)
    ctx_multiheads = layers.matmul(weights, v)
    return __combine_heads(ctx_multiheads)
