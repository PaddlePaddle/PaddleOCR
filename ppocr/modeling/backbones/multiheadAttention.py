import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import Linear
from paddle.nn.initializer import XavierUniform as xavier_uniform_
from paddle.nn.initializer import Constant as constant_
from paddle.nn.initializer import XavierNormal as xavier_normal_

zeros_ = constant_(value=0.)
ones_ = constant_(value=1.)

class MultiheadAttention(nn.Layer):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.out_proj = Linear(embed_dim, embed_dim, bias_attr=bias)

        if add_bias_kv:
            self.bias_k = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
            self.add_parameter("bias_k", self.bias_k)
            self.bias_v = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
            self.add_parameter("bias_v", self.bias_v)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

        self.conv1 = paddle.nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv2 = paddle.nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=(1, 1))
        self.conv3 = paddle.nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim * 3, kernel_size=(1, 1))

    def _reset_parameters(self):


        xavier_uniform_(self.out_proj.weight)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, qkv_ = [False,False,False]):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        qkv_same = qkv_[0]
        kv_same = qkv_[1]

        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]
        assert key.shape == value.shape

        if qkv_same:
            # self-attention
            q, k, v = self._in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self._in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self._in_proj_kv(key)
        else:
            q = self._in_proj_q(query)
            k = self._in_proj_k(key)
            v = self._in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            self.bias_k = paddle.concat([self.bias_k for i in range(bsz)],axis=1)
            self.bias_v = paddle.concat([self.bias_v for i in range(bsz)],axis=1)
            k = paddle.concat([k, self.bias_k])
            v = paddle.concat([v, self.bias_v])
            if attn_mask is not None:
                attn_mask = paddle.concat([attn_mask, paddle.zeros([attn_mask.shape[0], 1],dtype=attn_mask.dtype)], axis=1)
            if key_padding_mask is not None:
                key_padding_mask = paddle.concat(
                    [key_padding_mask,paddle.zeros([key_padding_mask.shape[0], 1],dtype=key_padding_mask.dtype)], axis=1)

        q = q.reshape([tgt_len, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])
        if k is not None:
            k = k.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])
        if v is not None:
            v = v.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])

 

        src_len = k.shape[1]

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len

        if self.add_zero_attn:
            src_len += 1
            k = paddle.concat([k, paddle.zeros((k.shape[0], 1) + k.shape[2:],dtype=k.dtype)], axis=1)
            v = paddle.concat([v, paddle.zeros((v.shape[0], 1) + v.shape[2:],dtype=v.dtype)], axis=1)
            if attn_mask is not None:
                attn_mask = paddle.concat([attn_mask, paddle.zeros([attn_mask.shape[0], 1],dtype=attn_mask.dtype)], axis=1)
            if key_padding_mask is not None:
                key_padding_mask = paddle.concat(
                    [key_padding_mask, paddle.zeros([key_padding_mask.shape[0], 1],dtype=key_padding_mask.dtype)], axis=1)
        attn_output_weights = paddle.bmm(q, k.transpose([0,2,1]))
        assert list(attn_output_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            key = key_padding_mask.unsqueeze(1).unsqueeze(2).astype('float32')
            y = paddle.full(shape=key.shape, dtype='float32', fill_value='-inf')
            y = paddle.where(key==0.,key, y)
            attn_output_weights += y
            attn_output_weights = attn_output_weights.reshape([bsz*self.num_heads, tgt_len, src_len])

        attn_output_weights = F.softmax(
            attn_output_weights.astype('float32'), axis=-1,
            dtype=paddle.float32 if attn_output_weights.dtype == paddle.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = paddle.bmm(attn_output_weights, v)
        assert list(attn_output.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose([1, 0,2]).reshape([tgt_len, bsz, embed_dim])
        attn_output = self.out_proj(attn_output)
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            attn_output_weights = attn_output_weights.sum(axis=1) / self.num_heads
        else:
            attn_output_weights = None

        return attn_output, attn_output_weights

    def _in_proj_qkv(self, query):
        query = query.transpose([1, 2, 0])
        query = paddle.unsqueeze(query, axis=2)
        res = self.conv3(query)
        res = paddle.squeeze(res, axis=2)
        res = res.transpose([2, 0, 1])
        return res.chunk(3, axis=-1)

    def _in_proj_kv(self, key):
        key = key.transpose([1, 2, 0])
        key = paddle.unsqueeze(key, axis=2)
        res = self.conv2(key)
        res = paddle.squeeze(res, axis=2)
        res = res.transpose([2, 0, 1])
        return res.chunk(2, axis=-1)

    def _in_proj_q(self, query):
        query = query.transpose([1, 2, 0])
        query = paddle.unsqueeze(query, axis=2)
        res = self.conv1(query)
        res = paddle.squeeze(res, axis=2)
        res = res.transpose([2, 0, 1])
        return res

    def _in_proj_k(self, key):
        
        key = key.transpose([1, 2, 0])
        key = paddle.unsqueeze(key, axis=2)
        res = self.conv1(key)
        res = paddle.squeeze(res, axis=2)
        res = res.transpose([2, 0, 1])
        return res

    def _in_proj_v(self, value):
        
        value = value.transpose([1,2,0])#(1, 2, 0)
        value = paddle.unsqueeze(value, axis=2)
        res = self.conv1(value)
        res = paddle.squeeze(res, axis=2)
        res = res.transpose([2, 0, 1])
        return res



class MultiheadAttentionOptim(nn.Layer):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super(MultiheadAttentionOptim, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.out_proj = Linear(embed_dim, embed_dim, bias_attr=bias)

        self._reset_parameters()

        self.conv1 = paddle.nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv2 = paddle.nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))
        self.conv3 = paddle.nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, 1))

    def _reset_parameters(self):


        xavier_uniform_(self.out_proj.weight)


    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """


        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]
        assert key.shape == value.shape

        q = self._in_proj_q(query)
        k = self._in_proj_k(key)
        v = self._in_proj_v(value)
        q *= self.scaling


        q = q.reshape([tgt_len, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])
        k = k.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])
        v = v.reshape([-1, bsz * self.num_heads, self.head_dim]).transpose([1, 0, 2])


        src_len = k.shape[1]

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len

        
        attn_output_weights = paddle.bmm(q, k.transpose([0,2,1]))
        assert list(attn_output_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            key = key_padding_mask.unsqueeze(1).unsqueeze(2).astype('float32')
            
            y = paddle.full(shape=key.shape, dtype='float32', fill_value='-inf')
           
            y = paddle.where(key==0.,key, y)

            attn_output_weights += y
            attn_output_weights = attn_output_weights.reshape([bsz*self.num_heads, tgt_len, src_len])

        attn_output_weights = F.softmax(
            attn_output_weights.astype('float32'), axis=-1,
            dtype=paddle.float32 if attn_output_weights.dtype == paddle.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = paddle.bmm(attn_output_weights, v)
        assert list(attn_output.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose([1, 0,2]).reshape([tgt_len, bsz, embed_dim])
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            attn_output_weights = attn_output_weights.sum(axis=1) / self.num_heads
        else:
            attn_output_weights = None

        return attn_output, attn_output_weights


    def _in_proj_q(self, query):
        query = query.transpose([1, 2, 0])
        query = paddle.unsqueeze(query, axis=2)
        res = self.conv1(query)
        res = paddle.squeeze(res, axis=2)
        res = res.transpose([2, 0, 1])
        return res

    def _in_proj_k(self, key):
        
        key = key.transpose([1, 2, 0])
        key = paddle.unsqueeze(key, axis=2)
        res = self.conv2(key)
        res = paddle.squeeze(res, axis=2)
        res = res.transpose([2, 0, 1])
        return res

    def _in_proj_v(self, value):
        
        value = value.transpose([1,2,0])#(1, 2, 0)
        value = paddle.unsqueeze(value, axis=2)
        res = self.conv3(value)
        res = paddle.squeeze(res, axis=2)
        res = res.transpose([2, 0, 1])
        return res