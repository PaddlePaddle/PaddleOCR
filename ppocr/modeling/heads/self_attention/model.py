from functools import partial
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers

encoder_data_input_fields = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias", )


def wrap_layer_with_block(layer, block_idx):
    """
    Make layer define support indicating block, by which we can add layers
    to other blocks within current block. This will make it easy to define
    cache among while loop.
    """

    class BlockGuard(object):
        """
        BlockGuard class.

        BlockGuard class is used to switch to the given block in a program by
        using the Python `with` keyword.
        """

        def __init__(self, block_idx=None, main_program=None):
            self.main_program = fluid.default_main_program(
            ) if main_program is None else main_program
            self.old_block_idx = self.main_program.current_block().idx
            self.new_block_idx = block_idx

        def __enter__(self):
            self.main_program.current_block_idx = self.new_block_idx

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.main_program.current_block_idx = self.old_block_idx
            if exc_type is not None:
                return False  # re-raise exception
            return True

    def layer_wrapper(*args, **kwargs):
        with BlockGuard(block_idx):
            return layer(*args, **kwargs)

    return layer_wrapper


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         gather_idx=None,
                         static_kv=False):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      bias_attr=False,
                      num_flatten_dims=2)
        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        fc_layer = wrap_layer_with_block(
            layers.fc, fluid.default_main_program().current_block()
            .parent_idx) if cache is not None and static_kv else layers.fc
        k = fc_layer(
            input=keys,
            size=d_key * n_head,
            bias_attr=False,
            num_flatten_dims=2)
        v = fc_layer(
            input=values,
            size=d_value * n_head,
            bias_attr=False,
            num_flatten_dims=2)
        return q, k, v

    def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Reshape input tensors at the last dimension to split multi-heads
        and then transpose. Specifically, transform the input tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped_q = layers.reshape(
            x=queries, shape=[0, 0, n_head, d_key], inplace=True)
        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        q = layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
        # For encoder-decoder attention in inference, insert the ops and vars
        # into global block to use as cache among beam search.
        reshape_layer = wrap_layer_with_block(
            layers.reshape,
            fluid.default_main_program().current_block()
            .parent_idx) if cache is not None and static_kv else layers.reshape
        transpose_layer = wrap_layer_with_block(
            layers.transpose,
            fluid.default_main_program().current_block().
            parent_idx) if cache is not None and static_kv else layers.transpose
        reshaped_k = reshape_layer(
            x=keys, shape=[0, 0, n_head, d_key], inplace=True)
        k = transpose_layer(x=reshaped_k, perm=[0, 2, 1, 3])
        reshaped_v = reshape_layer(
            x=values, shape=[0, 0, n_head, d_value], inplace=True)
        v = transpose_layer(x=reshaped_v, perm=[0, 2, 1, 3])

        if cache is not None:  # only for faster inference
            if static_kv:  # For encoder-decoder attention in inference
                cache_k, cache_v = cache["static_k"], cache["static_v"]
                # To init the static_k and static_v in cache.
                # Maybe we can use condition_op(if_else) to do these at the first
                # step in while loop to replace these, however it might be less
                # efficient.
                static_cache_init = wrap_layer_with_block(
                    layers.assign,
                    fluid.default_main_program().current_block().parent_idx)
                static_cache_init(k, cache_k)
                static_cache_init(v, cache_v)
            else:  # For decoder self-attention in inference
                cache_k, cache_v = cache["k"], cache["v"]
            # gather cell states corresponding to selected parent
            select_k = layers.gather(cache_k, index=gather_idx)
            select_v = layers.gather(cache_v, index=gather_idx)
            if not static_kv:
                # For self attention in inference, use cache and concat time steps.
                select_k = layers.concat([select_k, k], axis=2)
                select_v = layers.concat([select_v, v], axis=2)
            # update cell states(caches) cached in global block
            layers.assign(select_k, cache_k)
            layers.assign(select_v, cache_v)
            return q, select_k, select_v
        return q, k, v

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        # print(q)
        # print(k)

        product = layers.matmul(x=q, y=k, transpose_y=True, alpha=d_key**-0.5)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights, dropout_prob=dropout_rate, seed=None, is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
    q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_model,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         bias_attr=False,
                         num_flatten_dims=2)
    return proj_out


def positionwise_feed_forward(x, d_inner_hid, d_hid, dropout_rate):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act="relu")
    if dropout_rate:
        hidden = layers.dropout(
            hidden, dropout_prob=dropout_rate, seed=None, is_test=False)
    out = layers.fc(input=hidden, size=d_hid, num_flatten_dims=2)
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.initializer.Constant(1.),
                bias_attr=fluid.initializer.Constant(0.))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out, dropout_prob=dropout_rate, seed=None, is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def prepare_encoder(
        src_word,  # [b,t,c]
        src_pos,
        src_vocab_size,
        src_emb_dim,
        src_max_len,
        dropout_rate=0.,
        bos_idx=0,
        word_emb_param_name=None,
        pos_enc_param_name=None):
    """Add word embeddings and position encodings.
    The output tensor has a shape of:
    [batch_size, max_src_length_in_batch, d_model].
    This module is used at the bottom of the encoder stacks.
    """

    src_word_emb = src_word
    src_word_emb = layers.cast(src_word_emb, 'float32')

    src_word_emb = layers.scale(x=src_word_emb, scale=src_emb_dim**0.5)
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            name=pos_enc_param_name, trainable=False))
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc
    return layers.dropout(
        enc_input, dropout_prob=dropout_rate, seed=None,
        is_test=False) if dropout_rate else enc_input


def prepare_decoder(src_word,
                    src_pos,
                    src_vocab_size,
                    src_emb_dim,
                    src_max_len,
                    dropout_rate=0.,
                    bos_idx=0,
                    word_emb_param_name=None,
                    pos_enc_param_name=None):
    """Add word embeddings and position encodings.
        The output tensor has a shape of:
        [batch_size, max_src_length_in_batch, d_model].
        This module is used at the bottom of the encoder stacks.
        """
    src_word_emb = layers.embedding(
        src_word,
        size=[src_vocab_size, src_emb_dim],
        padding_idx=bos_idx,  # set embedding of bos to 0
        param_attr=fluid.ParamAttr(
            name=word_emb_param_name,
            initializer=fluid.initializer.Normal(0., src_emb_dim**-0.5)))

    src_word_emb = layers.scale(x=src_word_emb, scale=src_emb_dim**0.5)
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            name=pos_enc_param_name, trainable=False))
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc
    return layers.dropout(
        enc_input, dropout_prob=dropout_rate, seed=None,
        is_test=False) if dropout_rate else enc_input


def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  preprocess_cmd="n",
                  postprocess_cmd="da"):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        pre_process_layer(enc_input, preprocess_cmd,
                          prepostprocess_dropout), None, None, attn_bias, d_key,
        d_value, d_model, n_head, attention_dropout)
    attn_output = post_process_layer(enc_input, attn_output, postprocess_cmd,
                                     prepostprocess_dropout)
    ffd_output = positionwise_feed_forward(
        pre_process_layer(attn_output, preprocess_cmd, prepostprocess_dropout),
        d_inner_hid, d_model, relu_dropout)
    return post_process_layer(attn_output, ffd_output, postprocess_cmd,
                              prepostprocess_dropout)


def encoder(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd="n",
            postprocess_cmd="da"):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd, )
        enc_input = enc_output
    enc_output = pre_process_layer(enc_output, preprocess_cmd,
                                   prepostprocess_dropout)
    return enc_output


def wrap_encoder_forFeature(src_vocab_size,
                            max_length,
                            n_layer,
                            n_head,
                            d_key,
                            d_value,
                            d_model,
                            d_inner_hid,
                            prepostprocess_dropout,
                            attention_dropout,
                            relu_dropout,
                            preprocess_cmd,
                            postprocess_cmd,
                            weight_sharing,
                            enc_inputs=None,
                            bos_idx=0):
    """
    The wrapper assembles together all needed layers for the encoder.
    img, src_pos, src_slf_attn_bias = enc_inputs
    img
    """

    conv_features, src_pos, src_slf_attn_bias = enc_inputs  #
    b, t, c = conv_features.shape

    enc_input = prepare_encoder(
        conv_features,
        src_pos,
        src_vocab_size,
        d_model,
        max_length,
        prepostprocess_dropout,
        bos_idx=bos_idx,
        word_emb_param_name="src_word_emb_table")

    enc_output = encoder(
        enc_input,
        src_slf_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd, )
    return enc_output


def wrap_encoder(src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 enc_inputs=None,
                 bos_idx=0):
    """
    The wrapper assembles together all needed layers for the encoder.
    img, src_pos, src_slf_attn_bias = enc_inputs
    img
    """

    src_word, src_pos, src_slf_attn_bias = enc_inputs  #

    enc_input = prepare_decoder(
        src_word,
        src_pos,
        src_vocab_size,
        d_model,
        max_length,
        prepostprocess_dropout,
        bos_idx=bos_idx,
        word_emb_param_name="src_word_emb_table")

    enc_output = encoder(
        enc_input,
        src_slf_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd, )
    return enc_output
