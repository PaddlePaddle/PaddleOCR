from functools import partial
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers

# Set seed for CE
dropout_seed = None


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


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(np.arange(
        num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


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
                weights,
                dropout_prob=dropout_rate,
                seed=dropout_seed,
                is_test=False)
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
            hidden, dropout_prob=dropout_rate, seed=dropout_seed, is_test=False)
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
                    out,
                    dropout_prob=dropout_rate,
                    seed=dropout_seed,
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def prepare_encoder(
        src_word,  #[b,t,c]
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

    src_word_emb = src_word  #layers.concat(res,axis=1)
    src_word_emb = layers.cast(src_word_emb, 'float32')
    # print("src_word_emb",src_word_emb)

    src_word_emb = layers.scale(x=src_word_emb, scale=src_emb_dim**0.5)
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            name=pos_enc_param_name, trainable=False))
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc
    return layers.dropout(
        enc_input, dropout_prob=dropout_rate, seed=dropout_seed,
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
    # print("target_word_emb",src_word_emb)
    src_word_emb = layers.scale(x=src_word_emb, scale=src_emb_dim**0.5)
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            name=pos_enc_param_name, trainable=False))
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc
    return layers.dropout(
        enc_input, dropout_prob=dropout_rate, seed=dropout_seed,
        is_test=False) if dropout_rate else enc_input


# prepare_encoder = partial(
#     prepare_encoder_decoder, pos_enc_param_name=pos_enc_param_names[0])
# prepare_decoder = partial(
#     prepare_encoder_decoder, pos_enc_param_name=pos_enc_param_names[1])


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


def decoder_layer(dec_input,
                  enc_output,
                  slf_attn_bias,
                  dec_enc_attn_bias,
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
                  cache=None,
                  gather_idx=None):
    """ The layer to be stacked in decoder part.
    The structure of this module is similar to that in the encoder part except
    a multi-head attention is added to implement encoder-decoder attention.
    """
    slf_attn_output = multi_head_attention(
        pre_process_layer(dec_input, preprocess_cmd, prepostprocess_dropout),
        None,
        None,
        slf_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache=cache,
        gather_idx=gather_idx)
    slf_attn_output = post_process_layer(
        dec_input,
        slf_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    enc_attn_output = multi_head_attention(
        pre_process_layer(slf_attn_output, preprocess_cmd,
                          prepostprocess_dropout),
        enc_output,
        enc_output,
        dec_enc_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache=cache,
        gather_idx=gather_idx,
        static_kv=True)
    enc_attn_output = post_process_layer(
        slf_attn_output,
        enc_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    ffd_output = positionwise_feed_forward(
        pre_process_layer(enc_attn_output, preprocess_cmd,
                          prepostprocess_dropout),
        d_inner_hid,
        d_model,
        relu_dropout, )
    dec_output = post_process_layer(
        enc_attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    return dec_output


def decoder(dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
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
            caches=None,
            gather_idx=None):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    """
    for i in range(n_layer):
        dec_output = decoder_layer(
            dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
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
            cache=None if caches is None else caches[i],
            gather_idx=gather_idx)
        dec_input = dec_output
    dec_output = pre_process_layer(dec_output, preprocess_cmd,
                                   prepostprocess_dropout)
    return dec_output


def make_all_inputs(input_fields):
    """
    Define the input data layers for the transformer model.
    """
    inputs = []
    for input_field in input_fields:
        input_var = layers.data(
            name=input_field,
            shape=input_descs[input_field][0],
            dtype=input_descs[input_field][1],
            lod_level=input_descs[input_field][2]
            if len(input_descs[input_field]) == 3 else 0,
            append_batch_size=False)
        inputs.append(input_var)
    return inputs


def make_all_py_reader_inputs(input_fields, is_test=False):
    reader = layers.py_reader(
        capacity=20,
        name="test_reader" if is_test else "train_reader",
        shapes=[input_descs[input_field][0] for input_field in input_fields],
        dtypes=[input_descs[input_field][1] for input_field in input_fields],
        lod_levels=[
            input_descs[input_field][2]
            if len(input_descs[input_field]) == 3 else 0
            for input_field in input_fields
        ])
    return layers.read_file(reader), reader


def transformer(src_vocab_size,
                trg_vocab_size,
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
                label_smooth_eps,
                bos_idx=0,
                use_py_reader=False,
                is_test=False):
    if weight_sharing:
        assert src_vocab_size == trg_vocab_size, (
            "Vocabularies in source and target should be same for weight sharing."
        )

    data_input_names = encoder_data_input_fields + \
                decoder_data_input_fields[:-1] + label_data_input_fields

    if use_py_reader:
        all_inputs, reader = make_all_py_reader_inputs(data_input_names,
                                                       is_test)
    else:
        all_inputs = make_all_inputs(data_input_names)
    # print("all inputs",all_inputs)
    enc_inputs_len = len(encoder_data_input_fields)
    dec_inputs_len = len(decoder_data_input_fields[:-1])
    enc_inputs = all_inputs[0:enc_inputs_len]
    dec_inputs = all_inputs[enc_inputs_len:enc_inputs_len + dec_inputs_len]
    label = all_inputs[-2]
    weights = all_inputs[-1]

    enc_output = wrap_encoder(
        src_vocab_size, 64, n_layer, n_head, d_key, d_value, d_model,
        d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout,
        preprocess_cmd, postprocess_cmd, weight_sharing, enc_inputs)

    predict = wrap_decoder(
        trg_vocab_size,
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
        dec_inputs,
        enc_output, )

    # Padding index do not contribute to the total loss. The weights is used to
    # cancel padding index in calculating the loss.
    if label_smooth_eps:
        label = layers.label_smooth(
            label=layers.one_hot(
                input=label, depth=trg_vocab_size),
            epsilon=label_smooth_eps)

    cost = layers.softmax_with_cross_entropy(
        logits=predict,
        label=label,
        soft_label=True if label_smooth_eps else False)
    weighted_cost = cost * weights
    sum_cost = layers.reduce_sum(weighted_cost)
    token_num = layers.reduce_sum(weights)
    token_num.stop_gradient = True
    avg_cost = sum_cost / token_num
    return sum_cost, avg_cost, predict, token_num, reader if use_py_reader else None


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

    if enc_inputs is None:
        # This is used to implement independent encoder program in inference.
        conv_features, src_pos, src_slf_attn_bias = make_all_inputs(
            encoder_data_input_fields)
    else:
        conv_features, src_pos, src_slf_attn_bias = enc_inputs  #
        b, t, c = conv_features.shape
        #"""
        #    insert cnn
        #"""
        #import basemodel
        # feat = basemodel.resnet_50(img)

        # mycrnn = basemodel.CRNN()
        # feat = mycrnn.ocr_convs(img,use_cudnn=TrainTaskConfig.use_gpu)
        # b, c, w, h = feat.shape
        # src_word = layers.reshape(feat, shape=[-1, c, w * h])

        #myconv8 = basemodel.conv8()
        #feat = myconv8.net(img )
        #b , c, h, w = feat.shape#h=6
        #print(feat)
        #layers.Print(feat,message="conv_feat",summarize=10)

        #feat =layers.conv2d(feat,c,filter_size =[4 , 1],act="relu")
        #feat = layers.pool2d(feat,pool_stride=(3,1),pool_size=(3,1))
        #src_word = layers.squeeze(feat,axes=[2]) #src_word  [-1,c,ww]

        #feat = layers.transpose(feat, [0,3,1,2])
        #src_word = layers.reshape(feat,[-1,w, c*h])
        #src_word = layers.im2sequence(
        #    input=feat,
        #    stride=[1, 1],
        #    filter_size=[feat.shape[2], 1])
        #layers.Print(src_word,message="src_word",summarize=10)

        # print('feat',feat)
        #print("src_word",src_word)

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
    if enc_inputs is None:
        # This is used to implement independent encoder program in inference.
        src_word, src_pos, src_slf_attn_bias = make_all_inputs(
            encoder_data_input_fields)
    else:
        src_word, src_pos, src_slf_attn_bias = enc_inputs  #
        #"""
        #    insert cnn
        #"""
        #import basemodel
        # feat = basemodel.resnet_50(img)

        # mycrnn = basemodel.CRNN()
        # feat = mycrnn.ocr_convs(img,use_cudnn=TrainTaskConfig.use_gpu)
        # b, c, w, h = feat.shape
        # src_word = layers.reshape(feat, shape=[-1, c, w * h])

        #myconv8 = basemodel.conv8()
        #feat = myconv8.net(img )
        #b , c, h, w = feat.shape#h=6
        #print(feat)
        #layers.Print(feat,message="conv_feat",summarize=10)

        #feat =layers.conv2d(feat,c,filter_size =[4 , 1],act="relu")
        #feat = layers.pool2d(feat,pool_stride=(3,1),pool_size=(3,1))
        #src_word = layers.squeeze(feat,axes=[2]) #src_word  [-1,c,ww]

        #feat = layers.transpose(feat, [0,3,1,2])
        #src_word = layers.reshape(feat,[-1,w, c*h])
        #src_word = layers.im2sequence(
        #    input=feat,
        #    stride=[1, 1],
        #    filter_size=[feat.shape[2], 1])
        #layers.Print(src_word,message="src_word",summarize=10)

        # print('feat',feat)
        #print("src_word",src_word)
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


def wrap_decoder(trg_vocab_size,
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
                 dec_inputs=None,
                 enc_output=None,
                 caches=None,
                 gather_idx=None,
                 bos_idx=0):
    """
    The wrapper assembles together all needed layers for the decoder.
    """
    if dec_inputs is None:
        # This is used to implement independent decoder program in inference.
        trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, enc_output = \
            make_all_inputs(decoder_data_input_fields)
    else:
        trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias = dec_inputs

    dec_input = prepare_decoder(
        trg_word,
        trg_pos,
        trg_vocab_size,
        d_model,
        max_length,
        prepostprocess_dropout,
        bos_idx=bos_idx,
        word_emb_param_name="src_word_emb_table"
        if weight_sharing else "trg_word_emb_table")
    dec_output = decoder(
        dec_input,
        enc_output,
        trg_slf_attn_bias,
        trg_src_attn_bias,
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
        caches=caches,
        gather_idx=gather_idx)
    return dec_output
    # Reshape to 2D tensor to use GEMM instead of BatchedGEMM
    dec_output = layers.reshape(
        dec_output, shape=[-1, dec_output.shape[-1]], inplace=True)
    if weight_sharing:
        predict = layers.matmul(
            x=dec_output,
            y=fluid.default_main_program().global_block().var(
                "trg_word_emb_table"),
            transpose_y=True)
    else:
        predict = layers.fc(input=dec_output,
                            size=trg_vocab_size,
                            bias_attr=False)
    if dec_inputs is None:
        # Return probs for independent decoder program.
        predict = layers.softmax(predict)
    return predict


def fast_decode(src_vocab_size,
                trg_vocab_size,
                max_in_len,
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
                beam_size,
                max_out_len,
                bos_idx,
                eos_idx,
                use_py_reader=False):
    """
    Use beam search to decode. Caches will be used to store states of history
    steps which can make the decoding faster.
    """
    data_input_names = encoder_data_input_fields + fast_decoder_data_input_fields

    if use_py_reader:
        all_inputs, reader = make_all_py_reader_inputs(data_input_names)
    else:
        all_inputs = make_all_inputs(data_input_names)

    enc_inputs_len = len(encoder_data_input_fields)
    dec_inputs_len = len(fast_decoder_data_input_fields)
    enc_inputs = all_inputs[0:enc_inputs_len]  #enc_inputs tensor
    dec_inputs = all_inputs[enc_inputs_len:enc_inputs_len +
                            dec_inputs_len]  #dec_inputs tensor

    enc_output = wrap_encoder(
        src_vocab_size,
        64,  ##to do !!!!!????
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
        enc_inputs,
        bos_idx=bos_idx)
    start_tokens, init_scores, parent_idx, trg_src_attn_bias = dec_inputs

    def beam_search():
        max_len = layers.fill_constant(
            shape=[1],
            dtype=start_tokens.dtype,
            value=max_out_len,
            force_cpu=True)
        step_idx = layers.fill_constant(
            shape=[1], dtype=start_tokens.dtype, value=0, force_cpu=True)
        cond = layers.less_than(x=step_idx, y=max_len)  # default force_cpu=True
        while_op = layers.While(cond)
        # array states will be stored for each step.
        ids = layers.array_write(
            layers.reshape(start_tokens, (-1, 1)), step_idx)
        scores = layers.array_write(init_scores, step_idx)
        # cell states will be overwrited at each step.
        # caches contains states of history steps in decoder self-attention
        # and static encoder output projections in encoder-decoder attention
        # to reduce redundant computation.
        caches = [
            {
                "k":  # for self attention
                layers.fill_constant_batch_size_like(
                    input=start_tokens,
                    shape=[-1, n_head, 0, d_key],
                    dtype=enc_output.dtype,
                    value=0),
                "v":  # for self attention
                layers.fill_constant_batch_size_like(
                    input=start_tokens,
                    shape=[-1, n_head, 0, d_value],
                    dtype=enc_output.dtype,
                    value=0),
                "static_k":  # for encoder-decoder attention
                layers.create_tensor(dtype=enc_output.dtype),
                "static_v":  # for encoder-decoder attention
                layers.create_tensor(dtype=enc_output.dtype)
            } for i in range(n_layer)
        ]

        with while_op.block():
            pre_ids = layers.array_read(array=ids, i=step_idx)
            # Since beam_search_op dosen't enforce pre_ids' shape, we can do
            # inplace reshape here which actually change the shape of pre_ids.
            pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
            pre_scores = layers.array_read(array=scores, i=step_idx)
            # gather cell states corresponding to selected parent
            pre_src_attn_bias = layers.gather(
                trg_src_attn_bias, index=parent_idx)
            pre_pos = layers.elementwise_mul(
                x=layers.fill_constant_batch_size_like(
                    input=pre_src_attn_bias,  # cann't use lod tensor here
                    value=1,
                    shape=[-1, 1, 1],
                    dtype=pre_ids.dtype),
                y=step_idx,
                axis=0)
            logits = wrap_decoder(
                trg_vocab_size,
                max_in_len,
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
                dec_inputs=(pre_ids, pre_pos, None, pre_src_attn_bias),
                enc_output=enc_output,
                caches=caches,
                gather_idx=parent_idx,
                bos_idx=bos_idx)
            # intra-beam topK
            topk_scores, topk_indices = layers.topk(
                input=layers.softmax(logits), k=beam_size)
            accu_scores = layers.elementwise_add(
                x=layers.log(topk_scores), y=pre_scores, axis=0)
            # beam_search op uses lod to differentiate branches.
            accu_scores = layers.lod_reset(accu_scores, pre_ids)
            # topK reduction across beams, also contain special handle of
            # end beams and end sentences(batch reduction)
            selected_ids, selected_scores, gather_idx = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=beam_size,
                end_id=eos_idx,
                return_parent_idx=True)
            layers.increment(x=step_idx, value=1.0, in_place=True)
            # cell states(caches) have been updated in wrap_decoder,
            # only need to update beam search states here.
            layers.array_write(selected_ids, i=step_idx, array=ids)
            layers.array_write(selected_scores, i=step_idx, array=scores)
            layers.assign(gather_idx, parent_idx)
            layers.assign(pre_src_attn_bias, trg_src_attn_bias)
            length_cond = layers.less_than(x=step_idx, y=max_len)
            finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
            layers.logical_and(x=length_cond, y=finish_cond, out=cond)

        finished_ids, finished_scores = layers.beam_search_decode(
            ids, scores, beam_size=beam_size, end_id=eos_idx)
        return finished_ids, finished_scores

    finished_ids, finished_scores = beam_search()
    return finished_ids, finished_scores, reader if use_py_reader else None
