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
from paddle.nn import functional as F
from paddle.incubate.nn import functional as incubate_f
from paddle.nn import Layer
from paddle.framework import ParamAttr
import paddle
from paddle.nn.layer.transformer import _convert_attention_mask, _convert_param_attr_to_list
from paddle.nn.initializer import Constant

import collections


class FusedMultiHeadAttention(Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        attn_dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout in attention.
            0 for no dropout. Default 0.5.
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        normalize_before (bool, optional): Indicate  whether it is pre_layer_norm
            (True) or post_layer_norm architecture (False). Default False.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Now, only False is supported. Default False.
        weight_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr`.
        bias_attr (ParamAttr|bool, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            # input: [batch_size, sequence_length, embed_dim]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.incubate.nn.FusedMultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout_rate=0.5,
                 attn_dropout_rate=0.5,
                 kdim=None,
                 vdim=None,
                 normalize_before=False,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None,
                 epsilon=1e-5,
                 name=None):
        super(FusedMultiHeadAttention, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but recieved {}".format(embed_dim))
        assert num_heads > 0, ("Expected nhead to be greater than 0, "
                               "but recieved {}".format(num_heads))

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._epsilon = epsilon

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.need_weights = need_weights
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        assert need_weights == False, "Only support need_weight is False now."

        self.qkv_weight = self.create_parameter(
            shape=[3, num_heads, self.head_dim, embed_dim],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.qkv_bias = self.create_parameter(
            shape=[3, num_heads, self.head_dim],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.linear_weight = self.create_parameter(
            shape=[embed_dim, embed_dim],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.linear_bias = self.create_parameter(
            shape=[embed_dim],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)

        self.pre_ln_scale = self.create_parameter(
            attr=self._weight_attr,
            shape=[embed_dim],
            default_initializer=Constant(value=1.0))
        self.pre_ln_bias = self.create_parameter(
            attr=self._bias_attr, shape=[embed_dim], is_bias=True)
        self.ln_scale = self.create_parameter(
            attr=self._weight_attr,
            shape=[embed_dim],
            default_initializer=Constant(value=1.0))
        self.ln_bias = self.create_parameter(
            attr=self._bias_attr, shape=[embed_dim], is_bias=True)

        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.name = name

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        """
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                Now, only None is supported. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output.
        """
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, query.dtype)

        assert cache == None, "Only support cache is None now."

        out = incubate_f.fused_multi_head_attention(
            x=query,
            qkv_weight=self.qkv_weight,
            linear_weight=self.linear_weight,
            pre_layer_norm=self.normalize_before,
            pre_ln_scale=self.pre_ln_scale,
            pre_ln_bias=self.pre_ln_bias,
            ln_scale=self.ln_scale,
            ln_bias=self.ln_bias,
            pre_ln_epsilon=self._epsilon,
            qkv_bias=self.qkv_bias,
            linear_bias=self.linear_bias,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            ln_epsilon=self._epsilon,
            training=self.training,
            name=self.name)
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'embed_dim={}, num_heads={}, dropout_rate={}, attn_dropout_rate={}, epsilon={}, kdim={}, vdim={}, normalize_before={}, need_weights={}, dtype={}{}'.format(
            self.embed_dim, self.num_heads, self.dropout_rate,
            self.attn_dropout_rate, self._epsilon, self.kdim, self.vdim,
            self.normalize_before, self.need_weights, self._dtype, name_str)


class FusedFeedForward(Layer):
    """
    Parameters:
        d_model (int): The expected feature size in the input and output.
        dim_feedforward (int): The hidden layer size.
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess. Default 0.1
        epsilon (float, optional): he small value added to the variance to prevent
            division by zero. Default: 1e-05.
        activation (str, optional): The activation function. Default relu.
        act_dropout_rate (float, optional): The dropout probability after activition.
            If None, use the value of `dropout_rate`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into, preprocessing or postprocessing. Default False
        weight_attr (ParamAttr, optional): The attribute for the learnable weight of this layer.
            The default value is None and the weight will be initialized to zero. For detailed
            information, please refer to paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias of thi layer.
            If it is set to False, no bias will be added to the output. If it is set to None or one
            kind of ParamAttr, a bias parameter will be created according to ParamAttr. For detailed
            information, please refer to paddle.ParamAttr. The default value is None and the bias
            will be initialized to zero.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedFeedForward

            fused_feedforward_layer = FusedFeedForward(8, 8)
            x = paddle.rand((1, 8, 8))
            out = fused_feedforward_layer(x)
            print(out.numpy().shape)
            # (1, 8, 8)
    """

    def __init__(self,
                 d_model,
                 dim_feedforward,
                 dropout_rate=0.1,
                 epsilon=1e-05,
                 activation="relu",
                 act_dropout_rate=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):

        super(FusedFeedForward, self).__init__()
        assert d_model > 0, (
            "Expected d_model to be greater than 0, but recieved {}".format(
                d_model))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, but recieved {}".
            format(dim_feedforward))

        self._dtype = self._helper.get_default_dtype()
        self._d_model = d_model
        self._dim_feedforward = dim_feedforward
        self._dropout_rate = dropout_rate
        self._act_dropout_rate = dropout_rate if act_dropout_rate is None else act_dropout_rate
        self._act_method = activation
        self._normalize_before = normalize_before
        self._epsilon = epsilon

        self._linear1_weight = self.create_parameter(
            shape=[d_model, dim_feedforward],
            attr=weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self._linear1_bias = self.create_parameter(
            shape=[dim_feedforward],
            attr=bias_attr,
            dtype=self._dtype,
            is_bias=True)

        self._linear2_weight = self.create_parameter(
            shape=[dim_feedforward, d_model],
            attr=weight_attr,
            dtype=self._dtype,
            is_bias=False)

        self._linear2_bias = self.create_parameter(
            shape=[d_model], attr=bias_attr, dtype=self._dtype, is_bias=True)

        self._ln1_scale = self.create_parameter(
            shape=[d_model],
            attr=None,
            is_bias=False,
            default_initializer=Constant(1.0))
        self._ln1_bias = self.create_parameter(
            shape=[d_model], attr=None, is_bias=True)

        self._ln2_scale = self.create_parameter(
            shape=[d_model],
            attr=None,
            is_bias=False,
            default_initializer=Constant(1.0))
        self._ln2_bias = self.create_parameter(
            shape=[d_model], attr=None, is_bias=True)
        self.name = name

    def forward(self, src, cache=None):
        out = incubate_f.fused_feedforward(
            src,
            self._linear1_weight,
            self._linear2_weight,
            self._linear1_bias,
            self._linear2_bias,
            self._ln1_scale,
            self._ln1_bias,
            self._ln2_scale,
            self._ln2_bias,
            dropout1_rate=self._act_dropout_rate,
            dropout2_rate=self._dropout_rate,
            activation=self._act_method,
            ln1_epsilon=self._epsilon,
            ln2_epsilon=self._epsilon,
            pre_layer_norm=self._normalize_before,
            training=self.training,
            name=self.name)
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'd_model={}, dim_feedforward={}, dropout_rate={}, epsilon={}, activation={}, act_dropout_rate={}, normalize_before={}, dtype={}{}'.format(
            self._d_model, self._dim_feedforward, self._dropout_rate,
            self._epsilon, self._act_method, self._act_dropout_rate,
            self._normalize_before, self._dtype, name_str)


class FusedTransformerEncoderLayer(Layer):
    """
    FusedTransformerEncoderLayer is composed of two sub-layers which are self (multi-head)
    attention and feedforward network. Before and after each sub-layer, pre-process
    and post-precess would be applied on the input and output accordingly. If
    `normalize_before` is True, pre-process is layer normalization and post-precess
    includes dropout, residual connection. Otherwise, no pre-process and post-precess
    includes dropout, residual connection, layer normalization.

    Parameters:
        d_model (int): The expected feature size in the input and output.
        nhead (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout_rate (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout_rate (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, `weight_attr[0]` would be used as `weight_attr` for
            MHA, and `weight_attr[1]` would be used as `weight_attr` for linear in FFN.
            Otherwise, MHA and FFN both use it as `weight_attr` to create parameters.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, `bias_attr[0]` would be used as `bias_attr` for
            MHA, and `bias_attr[1]` would be used as `bias_attr` for linear in FFN.
            Otherwise, MHA and FFN both use it as `bias_attr` to create parameters.
            The `False` value means the corresponding layer would not have trainable
            bias parameter. See usage for details in :code:`ParamAttr` . Default: None,
            which means the default bias parameter property is used.


    Examples:

        .. code-block:: python

	    # required: gpu
            import paddle
            from paddle.incubate.nn import FusedTransformerEncoderLayer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = FusedTransformerEncoderLayer(128, 2, 512)
            enc_output = encoder_layer(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout_rate=0.1,
                 activation="relu",
                 attn_dropout_rate=None,
                 act_dropout_rate=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(FusedTransformerEncoderLayer, self).__init__()
        assert d_model > 0, ("Expected d_model to be greater than 0, "
                             "but recieved {}".format(d_model))
        assert nhead > 0, ("Expected nhead to be greater than 0, "
                           "but recieved {}".format(nhead))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, "
            "but recieved {}".format(dim_feedforward))
        attn_dropout_rate = dropout_rate if attn_dropout_rate is None else attn_dropout_rate
        act_dropout_rate = dropout_rate if act_dropout_rate is None else act_dropout_rate
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.fused_attn = FusedMultiHeadAttention(
            d_model,
            nhead,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            normalize_before=self.normalize_before,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])

        self.ffn = FusedFeedForward(
            d_model,
            dim_feedforward,
            dropout_rate=dropout_rate,
            activation=activation,
            act_dropout_rate=act_dropout_rate,
            normalize_before=self.normalize_before,
            weight_attr=weight_attrs[1],
            bias_attr=bias_attrs[1])

    def forward(self, src, src_mask=None, cache=None):
        """
        Applies a Transformer encoder layer on the input.
        Parameters:
            src (Tensor): The input of Transformer encoder layer. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float32 or float64.
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
            cache (Tensor, optional): It is an instance of `MultiHeadAttention.Cache`.
                See `TransformerEncoderLayer.gen_cache` for more details. It is
                only used for inference and should be None for training. Default
                None.
        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `enc_input`, representing the output of Transformer encoder \
                layer. Or a tuple if `cache` is not None, except for encoder \
                layer output, the tuple includes the new cache which is same \
                as input `cache` argument but `incremental_cache` has an \
                incremental length. See `MultiHeadAttention.gen_cache` and \
                `MultiHeadAttention.forward` for more details.
        """
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        if cache is None:
            attn_out = self.fused_attn(src, attn_mask=src_mask)
        else:
            attn_out, incremental_cache = self.fused_attn(
                src, attn_mask=src_mask, cache=cache)

        ffn_out = self.ffn(attn_out)

        return ffn_out if cache is None else (ffn_out, incremental_cache)


class FusedTransformer(Layer):
    """
    A Transformer model composed of an instance of `TransformerEncoder` and an
    instance of `TransformerDecoder`. While the embedding layer and output layer
    are not included.

    Please refer to `Attention is all you need <http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`_ ,
    and see `TransformerEncoder` and `TransformerDecoder` for more details.

    Users can configurate the model architecture with corresponding parameters.
    Note the usage of `normalize_before` representing where to apply layer
    normalization (in pre-process or post-precess of multi-head attention or FFN),
    and some transformer like models are different on this, such as
    `BERT <https://arxiv.org/abs/1810.04805>`_ and `GPT2 <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_ .
    The default architecture here places layer normalization in post-process and
    applies another layer normalization on the output of last encoder/decoder layer.

    Parameters:
        d_model (int, optional): The expected feature size in the encoder/decoder input
            and output. Default 512
        nhead (int, optional): The number of heads in multi-head attention(MHA). Default 8
        num_encoder_layers (int, optional): The number of layers in encoder. Default 6
        num_decoder_layers (int, optional): The number of layers in decoder. Default 6
        dim_feedforward (int, optional): The hidden layer size in the feedforward network(FFN). Default 2048
        dropout (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, the length of `weight_attr` could be 1, 2 or 3. If it is 3,
            `weight_attr[0]` would be used as `weight_attr` for self attention, `weight_attr[1]`
            would be used as `weight_attr` for cross attention of `TransformerDecoder`,
            and `weight_attr[2]` would be used as `weight_attr` for linear in FFN.
            If it is 2, `weight_attr[0]` would be used as `weight_attr` both for self attention
            and cross attntion and `weight_attr[1]` would be used as `weight_attr` for
            linear in FFN. If it is 1, `weight_attr[0]` would be used as `weight_attr`
            for self attention, cross attention and linear in FFN. Otherwise,
            the three sub-layers all uses it as `weight_attr` to create parameters.
            Default: None, which means the default weight parameter property is used.
            See usage for details
            in :code:`ParamAttr` .
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, the length of `bias_attr` could be 1, 2 or 3. If it is 3,
            `bias_attr[0]` would be used as `bias_attr` for self attention, `bias_attr[1]`
            would be used as `bias_attr` for cross attention of `TransformerDecoder`,
            and `bias_attr[2]` would be used as `bias_attr` for linear in FFN.
            If it is 2, `bias_attr[0]` would be used as `bias_attr` both for self attention
            and cross attntion and `bias_attr[1]` would be used as `bias_attr` for
            linear in FFN. If it is 1, `bias_attr[0]` would be used as `bias_attr`
            for self attention, cross attention and linear in FFN. Otherwise,
            the three sub-layers all uses it as `bias_attr` to create parameters.
            The `False` value means the corresponding layer would not have trainable
            bias parameter. See usage for details in :code:`ParamAttr` .
            Default: None,which means the default bias parameter property is used.
        custom_encoder (Layer, optional): If custom encoder is provided, use it as the encoder.
            Default None
        custom_decoder (Layer, optional): If custom decoder is provided, use it as the decoder.
            Default None

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import Transformer

            # src: [batch_size, tgt_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # tgt: [batch_size, src_len, d_model]
            dec_input = paddle.rand((2, 6, 128))
            # src_mask: [batch_size, n_head, src_len, src_len]
            enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
            # tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
            dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
            # memory_mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 6, 4))
            transformer = Transformer(128, 2, 4, 4, 512)
            output = transformer(enc_input,
                                 dec_input,
                                 enc_self_attn_mask,
                                 dec_self_attn_mask,
                                 cross_attn_mask)  # [2, 6, 128]
    """

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 custom_encoder=None,
                 custom_decoder=None):
        super(fusedTransformer, self).__init__()
        raise NotImplementedError()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        raise NotImplementedError()
