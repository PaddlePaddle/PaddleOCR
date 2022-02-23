import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layers.utils import map_structure

__all__ = [
    "position_encoding_init", "WordEmbedding", "PositionalEmbedding",
    "CrossEntropyCriterion", "TransformerDecodeCell",
    "TransformerBeamSearchDecoder", "TransformerModel", "InferTransformerModel"
]


def position_encoding_init(n_position, d_pos_vec, dtype="float32"):
    """
    Generates the initial values for the sinusoidal position encoding table.
    This method follows the implementation in tensor2tensor, but is slightly
    different from the description in "Attention Is All You Need".

    Args:
        n_position (int): 
            The largest position for sequences, that is, the maximum length
            of source or target sequences.
        d_pos_vec (int): 
            The size of positional embedding vector. 
        dtype (str, optional): 
            The output `numpy.array`'s data type. Defaults to "float32".

    Returns:
        numpy.array: 
            The embedding table of sinusoidal position encoding with shape
            `[n_position, d_pos_vec]`.

    Example:
        .. code-block::

            from paddlenlp.transformers import position_encoding_init

            max_length = 256
            emb_dim = 512
            pos_table = position_encoding_init(max_length, emb_dim)
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(
        np.arange(num_timescales) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype(dtype)


class WordEmbedding(nn.Layer):
    """
    Word Embedding layer of Transformer. 

    This layer automatically constructs a 2D embedding matrix based on the
    input the size of vocabulary (`vocab_size`) and the size of each embedding
    vector (`emb_dim`). This layer lookups embeddings vector of ids provided
    by input `word`. 

    After the embedding, those weights are multiplied by `sqrt(d_model)` which is
    `sqrt(emb_dim)` in the interface. 

    .. math::

        Out = embedding(word) * sqrt(emb\_dim)

    Args:
        vocab_size (int):
            The size of vocabulary. 
        emb_dim (int):
            Dimensionality of each embedding vector.
        bos_id (int, optional):
            The start token id and also is used as padding id. Defaults to 0.
    """

    def __init__(self, vocab_size, emb_dim, bos_id=0):
        super(WordEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=bos_id,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0., emb_dim**-0.5)))

    def forward(self, word):
        r"""
        Computes word embedding.

        Args:
            word (Tensor):
                The input ids which indicates the sequences' words with shape
                `[batch_size, sequence_length]` whose data type can be
                int or int64.

        Returns:
            Tensor:
                The (scaled) embedding tensor of shape
                `(batch_size, sequence_length, emb_dim)` whose data type can be
                float32 or float64.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import WordEmbedding

                word_embedding = WordEmbedding(
                    vocab_size=30000,
                    emb_dim=512,
                    bos_id=0)

                batch_size = 5
                sequence_length = 10
                src_words = paddle.randint(low=3, high=30000, shape=[batch_size, sequence_length])
                src_emb = word_embedding(src_words)
        """
        word_emb = self.emb_dim**0.5 * self.word_embedding(word)
        return word_emb


class PositionalEmbedding(nn.Layer):
    """
    This layer produces sinusoidal positional embeddings of any length.
    While in `forward()` method, this layer lookups embeddings vector of
    ids provided by input `pos`.

    Args:
        emb_dim (int):
            The size of each embedding vector.
        max_length (int):
            The maximum length of sequences.
    """

    def __init__(self, emb_dim, max_length):
        super(PositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.pos_encoder = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=self.emb_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    position_encoding_init(max_length, self.emb_dim))))

    def forward(self, pos):
        r"""
        Computes positional embedding.

        Args:
            pos (Tensor):
                The input position ids with shape `[batch_size, sequence_length]` whose
                data type can be int or int64.

        Returns:
            Tensor:
                The positional embedding tensor of shape
                `(batch_size, sequence_length, emb_dim)` whose data type can be
                float32 or float64.
        
        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import PositionalEmbedding

                pos_embedding = PositionalEmbedding(
                    emb_dim=512,
                    max_length=256)

                batch_size = 5
                pos = paddle.tile(paddle.arange(start=0, end=50), repeat_times=[batch_size, 1])
                pos_emb = pos_embedding(pos)
        """
        pos_emb = self.pos_encoder(pos)
        pos_emb.stop_gradient = True
        return pos_emb


class CrossEntropyCriterion(nn.Layer):
    """
    Computes the cross entropy loss for given input with or without label smoothing.

    Args:
        label_smooth_eps (float, optional):
            The weight used to mix up the original ground-truth distribution
            and the fixed distribution. Defaults to None. If given, label smoothing
            will be applied on `label`.
        pad_idx (int, optional):
            The token id used to pad variant sequence. Defaults to 0. 
    """

    def __init__(self, label_smooth_eps=None, pad_idx=0):
        super(CrossEntropyCriterion, self).__init__()
        self.label_smooth_eps = label_smooth_eps
        self.pad_idx = pad_idx

    def forward(self, predict, label):
        r"""
        Computes cross entropy loss with or without label smoothing. 

        Args:
            predict (Tensor):
                The predict results of `TransformerModel` with shape
                `[batch_size, sequence_length, vocab_size]` whose data type can
                be float32 or float64.
            label (Tensor):
                The label for correspoding results with shape
                `[batch_size, sequence_length, 1]`.

        Returns:
            tuple:
                A tuple with items: (`sum_cost`, `avg_cost`, `token_num`).

                With the corresponding fields:

                - `sum_cost` (Tensor):
                    The sum of loss of current batch whose data type can be float32, float64.
                - `avg_cost` (Tensor):
                    The average loss of current batch whose data type can be float32, float64.
                    The relation between `sum_cost` and `avg_cost` can be described as:

                    .. math::

                        avg\_cost = sum\_cost / token\_num

                - `token_num` (Tensor):
                    The number of tokens of current batch. Its data type can be float32, float64.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import CrossEntropyCriterion

                criterion = CrossEntropyCriterion(label_smooth_eps=0.1, pad_idx=0)
                batch_size = 1
                seq_len = 2
                vocab_size = 30000
                predict = paddle.rand(shape=[batch_size, seq_len, vocab_size])
                label = paddle.randint(
                    low=3,
                    high=vocab_size,
                    shape=[batch_size, seq_len, 1])

                criterion(predict, label)
        """
        weights = paddle.cast(
            label != self.pad_idx, dtype=paddle.get_default_dtype())
        if self.label_smooth_eps:
            label = paddle.squeeze(label, axis=[2])
            label = F.label_smooth(
                label=F.one_hot(
                    x=label, num_classes=predict.shape[-1]),
                epsilon=self.label_smooth_eps)

        cost = F.cross_entropy(
            input=predict,
            label=label,
            reduction='none',
            soft_label=True if self.label_smooth_eps else False)
        weighted_cost = cost * weights
        sum_cost = paddle.sum(weighted_cost)
        token_num = paddle.sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, token_num


class TransformerDecodeCell(nn.Layer):
    """
    This layer wraps a Transformer decoder combined with embedding
    layer and output layer to produce logits from ids and position.

    Args:
        decoder (callable):
            Can be a `paddle.nn.TransformerDecoder` instance. Or a wrapper that includes an
            embedding layer accepting ids and positions and includes an
            output layer transforming decoder output to logits.
        word_embedding (callable, optional):
            Can be a `WordEmbedding` instance or a callable that accepts ids as
            arguments and return embeddings. It can be None if `decoder`
            includes a embedding layer. Defaults to None.
        pos_embedding (callable, optional):
            Can be a `PositionalEmbedding` instance or a callable that accepts position
            as arguments and return embeddings. It can be None if `decoder`
            includes a positional embedding layer. Defaults to None.
        linear (callable, optional):
            Can be a `paddle.nn.Linear` instance or a callable to transform decoder
            output to logits.
        dropout (float, optional):
            The dropout rate for the results of `word_embedding` and `pos_embedding`.
            Defaults to 0.1.
    """

    def __init__(self,
                 decoder,
                 word_embedding=None,
                 pos_embedding=None,
                 linear=None,
                 dropout=0.1):
        super(TransformerDecodeCell, self).__init__()
        self.decoder = decoder
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding
        self.linear = linear
        self.dropout = dropout

    def forward(self, inputs, states, static_cache, trg_src_attn_bias, memory,
                **kwargs):
        r"""
        Produces logits.

        Args:
            inputs (Tensor|tuple|list):
                A tuple/list includes target ids and positions. If `word_embedding` is None,
                then it should be a Tensor which means the input for decoder.
            states (list):
                It is a list and each element of the list is an instance
                of `paddle.nn.MultiheadAttention.Cache` for corresponding decoder
                layer. It can be produced by `paddle.nn.TransformerDecoder.gen_cache`.
            static_cache (list):
                It is a list and each element of the list is an instance of
                `paddle.nn.MultiheadAttention.StaticCache` for corresponding
                decoder layer. It can be produced by `paddle.nn.TransformerDecoder.gen_cache`.
            trg_src_attn_bias (Tensor):
                A tensor used in self attention to prevents attention to some unwanted
                positions, usually the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, target_length, target_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
            memory (Tensor):
                The output of Transformer encoder. It is a tensor with shape
                `[batch_size, source_length, d_model]` and its data type can be
                float32 or float64.

        Returns:
            tuple: 
                A tuple with items: `(outputs, new_states)`
                
                With the corresponding fields:

                - `outputs` (Tensor):
                    A float32 or float64 3D tensor representing logits shaped
                    `[batch_size, sequence_length, vocab_size]`.
                - `new_states` (Tensor):
                    This output has the same structure and data type with `states`
                    while the length is one larger since concatanating the
                    intermediate results of current step.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TransformerDecodeCell
                from paddlenlp.transformers import TransformerBeamSearchDecoder

                def decoder():
                    # do decoder
                    pass

                cell = TransformerDecodeCell(decoder())

                self.decode = TransformerBeamSearchDecoder(
                    cell, start_token=0, end_token=1, beam_size=4,
                    var_dim_in_state=2)
        """

        if states and static_cache:
            states = list(zip(states, static_cache))

        if self.word_embedding:
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs)

            word_emb = self.word_embedding(inputs[0])
            pos_emb = self.pos_embedding(inputs[1])
            word_emb = word_emb + pos_emb
            inputs = F.dropout(
                word_emb, p=self.dropout,
                training=False) if self.dropout else word_emb

            cell_outputs, new_states = self.decoder(inputs, memory, None,
                                                    trg_src_attn_bias, states)
        else:
            cell_outputs, new_states = self.decoder(inputs, memory, None,
                                                    trg_src_attn_bias, states)

        if self.linear:
            cell_outputs = self.linear(cell_outputs)

        new_states = [cache[0] for cache in new_states]

        return cell_outputs, new_states


class TransformerBeamSearchDecoder(nn.decode.BeamSearchDecoder):
    """
    This layer is a subclass of `BeamSearchDecoder` to make
    beam search adapt to Transformer decoder.

    Args:
        cell (`TransformerDecodeCell`):
            An instance of `TransformerDecoderCell`.
        start_token (int):
            The start token id.
        end_token (int):
            The end token id.
        beam_size (int):
            The beam width used in beam search.
        var_dim_in_state (int):
            Indicate which dimension of states is variant.
    """

    def __init__(self, cell, start_token, end_token, beam_size,
                 var_dim_in_state):
        super(TransformerBeamSearchDecoder,
              self).__init__(cell, start_token, end_token, beam_size)
        self.cell = cell
        self.var_dim_in_state = var_dim_in_state

    def _merge_batch_beams_with_var_dim(self, c):
        # Init length of cache is 0, and it increases with decoding carrying on,
        # thus need to reshape elaborately
        var_dim_in_state = self.var_dim_in_state + 1  # count in beam dim
        c = paddle.transpose(c,
                             list(range(var_dim_in_state, len(c.shape))) +
                             list(range(0, var_dim_in_state)))
        c = paddle.reshape(
            c, [0] * (len(c.shape) - var_dim_in_state
                      ) + [self.batch_size * self.beam_size] +
            [int(size) for size in c.shape[-var_dim_in_state + 2:]])
        c = paddle.transpose(
            c,
            list(range((len(c.shape) + 1 - var_dim_in_state), len(c.shape))) +
            list(range(0, (len(c.shape) + 1 - var_dim_in_state))))
        return c

    def _split_batch_beams_with_var_dim(self, c):
        var_dim_size = paddle.shape(c)[self.var_dim_in_state]
        c = paddle.reshape(
            c, [-1, self.beam_size] +
            [int(size)
             for size in c.shape[1:self.var_dim_in_state]] + [var_dim_size] +
            [int(size) for size in c.shape[self.var_dim_in_state + 1:]])
        return c

    @staticmethod
    def tile_beam_merge_with_batch(t, beam_size):
        r"""
        Tiles the batch dimension of a tensor. Specifically, this function takes
        a tensor t shaped `[batch_size, s0, s1, ...]` composed of minibatch 
        entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
        `[batch_size * beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Args:
            t (list|tuple):
                A list of tensor with shape `[batch_size, ...]`.
            beam_size (int):
                The beam width used in beam search.

        Returns:
            Tensor:
                A tensor with shape `[batch_size * beam_size, ...]`, whose
                data type is same as `t`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TransformerBeamSearchDecoder

                t = paddle.rand(shape=[10, 10])
                TransformerBeamSearchDecoder.tile_beam_merge_with_batch(t, beam_size=4)
        """
        return map_structure(
            lambda x: nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(x, beam_size),
            t)

    def step(self, time, inputs, states, **kwargs):
        """
        Perform a beam search decoding step, which uses cell to get probabilities,
        and follows a beam search step to calculate scores and select candidate token ids.

        Args:
             time(Tensor): An `int64` tensor with shape `[1]` provided by the caller,
                 representing the current time step number of decoding.
             inputs(Tensor): A tensor variable. It is same as `initial_inputs`
                 returned by `initialize()` for the first decoding step and
                 `next_inputs` returned by `step()` for the others.
             states(Tensor): A structure of tensor variables.
                 It is same as the `initial_cell_states` returned by `initialize()`
                 for the first decoding step and `next_states` returned by
                 `step()` for the others.
             kwargs(dict, optional): Additional keyword arguments, provided by the caller `dynamic_decode`.

        Returns:
             tuple: Returns tuple (``beam_search_output, beam_search_state, next_inputs, finished``).
             `beam_search_state` and `next_inputs` have the same structure,
             shape and data type as the input arguments states and inputs separately.
             `beam_search_output` is a namedtuple(including scores, predicted_ids, parent_ids as fields) of tensor variables,
             where `scores, predicted_ids, parent_ids` all has a tensor value shaped [batch_size, beam_size] with data type
             float32, int64, int64. `finished` is a bool tensor with shape [batch_size, beam_size].

         """
        # Steps for decoding.
        # Compared to RNN, Transformer has 3D data at every decoding step
        inputs = paddle.reshape(inputs, [-1, 1])  # token
        pos = paddle.ones_like(inputs) * time  # pos

        cell_states = map_structure(self._merge_batch_beams_with_var_dim,
                                    states.cell_states)

        cell_outputs, next_cell_states = self.cell((inputs, pos), cell_states,
                                                   **kwargs)

        # Squeeze to adapt to BeamSearchDecoder which use 2D logits
        cell_outputs = map_structure(
            lambda x: paddle.squeeze(x, [1]) if len(x.shape) == 3 else x,
            cell_outputs)
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams_with_var_dim,
                                         next_cell_states)

        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,
            next_cell_states=next_cell_states,
            beam_state=states)

        if kwargs.get("trg_word", None) is not None:
            if in_dygraph_mode():
                if paddle.shape(kwargs.get("trg_word"))[1] > time:
                    beam_search_output, beam_search_state = self.force_decoding(
                        beam_search_output, beam_search_state,
                        kwargs.get("trg_word"), kwargs.get("trg_length"), time)
            else:

                def condition(trg_word, time):
                    return paddle.shape(trg_word)[1] > time

                def default_fn(beam_search_output, beam_search_state):
                    return beam_search_output, beam_search_state

                from functools import partial
                beam_search_output, beam_search_state = paddle.static.nn.case(
                    [(condition(kwargs.get("trg_word"), time), partial(
                        self.force_decoding,
                        beam_search_output=beam_search_output,
                        beam_search_state=beam_search_state,
                        trg_word=kwargs.get("trg_word"),
                        trg_length=kwargs.get("trg_length"),
                        time=time))],
                    default=partial(
                        default_fn,
                        beam_search_output=beam_search_output,
                        beam_search_state=beam_search_state))

        next_inputs, finished = (beam_search_output.predicted_ids,
                                 beam_search_state.finished)

        return (beam_search_output, beam_search_state, next_inputs, finished)

    def force_decoding(self, beam_search_output, beam_search_state, trg_word,
                       trg_length, time):
        batch_size = paddle.shape(beam_search_output.predicted_ids)[0]
        beam_size = paddle.shape(beam_search_output.predicted_ids)[1]

        ids_dtype = beam_search_output.predicted_ids.dtype
        scores_dtype = beam_search_output.scores.dtype
        parent_ids = paddle.zeros(shape=[batch_size, 1], dtype=ids_dtype)
        scores = paddle.ones(
            shape=[batch_size, beam_size], dtype=scores_dtype) * -1e4
        scores = paddle.scatter(
            scores.flatten(),
            paddle.arange(
                0, batch_size * beam_size, step=beam_size, dtype="int64"),
            paddle.zeros([batch_size])).reshape([batch_size, beam_size])

        force_position = paddle.unsqueeze(trg_length > time, [1])
        # NOTE: When the date type of the input of paddle.tile is bool
        # and enable static mode, its stop_gradient must be True .
        force_position.stop_gradient = True
        force_position = paddle.tile(force_position, [1, beam_size])
        crt_trg_word = paddle.slice(
            trg_word, axes=[1], starts=[time], ends=[time + 1])
        crt_trg_word = paddle.tile(crt_trg_word, [1, beam_size])

        predicted_ids = paddle.where(force_position, crt_trg_word,
                                     beam_search_output.predicted_ids)
        scores = paddle.where(force_position, scores, beam_search_output.scores)
        parent_ids = paddle.where(force_position, parent_ids,
                                  beam_search_output.parent_ids)

        cell_states = beam_search_state.cell_states
        log_probs = paddle.where(force_position, scores,
                                 beam_search_state.log_probs)
        finished = beam_search_state.finished
        lengths = beam_search_state.lengths

        return self.OutputWrapper(scores, predicted_ids,
                                  parent_ids), self.StateWrapper(
                                      cell_states, log_probs, finished, lengths)


class TransformerModel(nn.Layer):
    """
    The Transformer model.

    This model is a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        src_vocab_size (int):
            The size of source vocabulary.
        trg_vocab_size (int):
            The size of target vocabulary.
        max_length (int):
            The maximum length of input sequences.
        num_encoder_layers (int):
            The number of sub-layers to be stacked in the encoder.
        num_decoder_layers (int):
            The number of sub-layers to be stacked in the decoder.
        n_head (int):
            The number of head used in multi-head attention.
        d_model (int):
            The dimension for word embeddings, which is also the last dimension of
            the input and output of multi-head attention, position-wise feed-forward
            networks, encoder and decoder.
        d_inner_hid (int):
            Size of the hidden layer in position-wise feed-forward networks.
        dropout (float):
            Dropout rates. Used for pre-process, activation and inside attention.
        weight_sharing (bool):
            Whether to use weight sharing. 
        attn_dropout (float):
            The dropout probability used in MHA to drop some attention target.
            If None, use the value of dropout. Defaults to None.
        act_dropout (float):
            The dropout probability used after FFN activation. If None, use
            the value of dropout. Defaults to None.
        bos_id (int, optional):
            The start token id and also be used as padding id. Defaults to 0.
        eos_id (int, optional):
            The end token id. Defaults to 1.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 weight_sharing,
                 attn_dropout=None,
                 act_dropout=None,
                 bos_id=0,
                 eos_id=1):
        super(TransformerModel, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout

        self.src_word_embedding = WordEmbedding(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
        self.src_pos_embedding = PositionalEmbedding(
            emb_dim=d_model, max_length=max_length)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            self.trg_word_embedding = self.src_word_embedding
            self.trg_pos_embedding = self.src_pos_embedding
        else:
            self.trg_word_embedding = WordEmbedding(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(
                emb_dim=d_model, max_length=max_length)

        self.transformer = paddle.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            activation="relu",
            normalize_before=True)

        if weight_sharing:
            self.linear = lambda x: paddle.matmul(x=x,
                                                  y=self.trg_word_embedding.word_embedding.weight,
                                                  transpose_y=True)
        else:
            self.linear = nn.Linear(
                in_features=d_model,
                out_features=trg_vocab_size,
                bias_attr=False)

    def forward(self, src_word, trg_word):
        r"""
        The Transformer forward methods. The input are source/target sequences, and
        returns logits.

        Args:
            src_word (Tensor):
                The ids of source sequences words. It is a tensor with shape
                `[batch_size, source_sequence_length]` and its data type can be
                int or int64.
            trg_word (Tensor):
                The ids of target sequences words. It is a tensor with shape
                `[batch_size, target_sequence_length]` and its data type can be
                int or int64.

        Returns:
            Tensor:
                Output tensor of the final layer of the model whose data
                type can be float32 or float64 with shape
                `[batch_size, sequence_length, vocab_size]`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TransformerModel

                transformer = TransformerModel(
                    src_vocab_size=30000,
                    trg_vocab_size=30000,
                    max_length=257,
                    num_encoder_layers=6,
                    num_decoder_layers=6,
                    n_head=8,
                    d_model=512,
                    d_inner_hid=2048,
                    dropout=0.1,
                    weight_sharing=True,
                    bos_id=0,
                    eos_id=1)

                batch_size = 5
                seq_len = 10
                predict = transformer(
                    src_word=paddle.randint(low=3, high=30000, shape=[batch_size, seq_len]),
                    trg_word=paddle.randint(low=3, high=30000, shape=[batch_size, seq_len]))
        """
        src_max_len = paddle.shape(src_word)[-1]
        trg_max_len = paddle.shape(trg_word)[-1]
        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        src_slf_attn_bias.stop_gradient = True
        trg_slf_attn_bias = self.transformer.generate_square_subsequent_mask(
            trg_max_len)
        trg_slf_attn_bias.stop_gradient = True
        trg_src_attn_bias = src_slf_attn_bias
        src_pos = paddle.cast(
            src_word != self.bos_id, dtype=src_word.dtype) * paddle.arange(
                start=0, end=src_max_len, dtype=src_word.dtype)
        trg_pos = paddle.cast(
            trg_word != self.bos_id, dtype=src_word.dtype) * paddle.arange(
                start=0, end=trg_max_len, dtype=trg_word.dtype)
        with paddle.static.amp.fp16_guard():
            src_emb = self.src_word_embedding(src_word)
            src_pos_emb = self.src_pos_embedding(src_pos)
            src_emb = src_emb + src_pos_emb
            enc_input = F.dropout(
                src_emb, p=self.dropout,
                training=self.training) if self.dropout else src_emb

            trg_emb = self.trg_word_embedding(trg_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb

            dec_output = self.transformer(
                enc_input,
                dec_input,
                src_mask=src_slf_attn_bias,
                tgt_mask=trg_slf_attn_bias,
                memory_mask=trg_src_attn_bias)

            predict = self.linear(dec_output)

        return predict


class InferTransformerModel(TransformerModel):
    """
    The Transformer model for auto-regressive generation.

    Args:
        src_vocab_size (int):
            The size of source vocabulary.
        trg_vocab_size (int):
            The size of target vocabulary.
        max_length (int):
            The maximum length of input sequences.
        num_encoder_layers (int):
            The number of sub-layers to be stacked in the encoder.
        num_decoder_layers (int):
            The number of sub-layers to be stacked in the decoder.
        n_head (int):
            The number of head used in multi-head attention.
        d_model (int):
            The dimension for word embeddings, which is also the last dimension of
            the input and output of multi-head attention, position-wise feed-forward
            networks, encoder and decoder.
        d_inner_hid (int):
            Size of the hidden layer in position-wise feed-forward networks.
        dropout (float):
            Dropout rates. Used for pre-process, activation and inside attention.
        weight_sharing (bool):
            Whether to use weight sharing. 
        attn_dropout (float):
            The dropout probability used in MHA to drop some attention target.
            If None, use the value of dropout. Defaults to None.
        act_dropout (float):
            The dropout probability used after FFN activition. If None, use
            the value of dropout. Defaults to None.
        bos_id (int, optional):
            The start token id and also is used as padding id. Defaults to 0.
        eos_id (int, optional):
            The end token id. Defaults to 1.
        beam_size (int, optional):
            The beam width for beam search. Defaults to 4. 
        max_out_len (int, optional):
            The maximum output length. Defaults to 256.
        output_time_major(bool, optional):
            Indicate the data layout of predicted
            Tensor. If `False`, the data layout would be batch major with shape
            `[batch_size, seq_len, beam_size]`. If  `True`, the data layout would
            be time major with shape `[seq_len, batch_size, beam_size]`. Default
            to `False`.
        beam_search_version (str):
            Specify beam search version. It should be in one
            of [`v1`, `v2`]. If `v2`, need to set `alpha`(default to 0.6) for length
            penalty. Default to `v1`.
        kwargs:
            The key word arguments can be `rel_len` and `alpha`:

            - `rel_len(bool, optional)`: Indicating whether `max_out_len` in
            is the length relative to that of source text. Only works in `v2`
            temporarily. It is suggest to set a small `max_out_len` and use
            `rel_len=True`. Default to False if not set.

            - `alpha(float, optional)`: The power number in length penalty
            calculation. Refer to `GNMT <https://arxiv.org/pdf/1609.08144.pdf>`_.
            Only works in `v2` temporarily. Default to 0.6 if not set.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 weight_sharing,
                 attn_dropout=None,
                 act_dropout=None,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256,
                 output_time_major=False,
                 beam_search_version='v1',
                 **kwargs):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.beam_size = args.pop("beam_size")
        self.max_out_len = args.pop("max_out_len")
        self.output_time_major = args.pop("output_time_major")
        self.dropout = dropout
        self.beam_search_version = args.pop('beam_search_version')
        kwargs = args.pop("kwargs")
        if self.beam_search_version == 'v2':
            self.alpha = kwargs.get("alpha", 0.6)
            self.rel_len = kwargs.get("rel_len", False)
        super(InferTransformerModel, self).__init__(**args)

        cell = TransformerDecodeCell(
            self.transformer.decoder, self.trg_word_embedding,
            self.trg_pos_embedding, self.linear, self.dropout)

        self.decode = TransformerBeamSearchDecoder(
            cell, bos_id, eos_id, beam_size, var_dim_in_state=2)

    def forward(self, src_word, trg_word=None):
        r"""
        The Transformer forward method.

        Args:
            src_word (Tensor):
                The ids of source sequence words. It is a tensor with shape
                `[batch_size, source_sequence_length]` and its data type can be
                int or int64.
            trg_word (Tensor):
                The ids of target sequence words. Normally, it should NOT be
                given. If it's given, force decoding with previous output token
                will be trigger. Defaults to None. 
        
        Returns:
            Tensor:
                An int64 tensor shaped indicating the predicted ids. Its shape is
                `[batch_size, seq_len, beam_size]` or `[seq_len, batch_size, beam_size]`
                according to `output_time_major`.
        
        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import InferTransformerModel

                transformer = InferTransformerModel(
                    src_vocab_size=30000,
                    trg_vocab_size=30000,
                    max_length=256,
                    num_encoder_layers=6,
                    num_decoder_layers=6,
                    n_head=8,
                    d_model=512,
                    d_inner_hid=2048,
                    dropout=0.1,
                    weight_sharing=True,
                    bos_id=0,
                    eos_id=1,
                    beam_size=4,
                    max_out_len=256)

                batch_size = 5
                seq_len = 10
                transformer(
                    src_word=paddle.randint(low=3, high=30000, shape=[batch_size, seq_len]))
        """
        if trg_word is not None:
            trg_length = paddle.sum(paddle.cast(
                trg_word != self.bos_id, dtype="int32"),
                                    axis=-1)
        else:
            trg_length = None

        if self.beam_search_version == 'v1':
            src_max_len = paddle.shape(src_word)[-1]
            src_slf_attn_bias = paddle.cast(
                src_word == self.bos_id,
                dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
            trg_src_attn_bias = src_slf_attn_bias
            src_pos = paddle.cast(
                src_word != self.bos_id, dtype=src_word.dtype) * paddle.arange(
                    start=0, end=src_max_len, dtype=src_word.dtype)

            # Run encoder
            src_emb = self.src_word_embedding(src_word)
            src_pos_emb = self.src_pos_embedding(src_pos)
            src_emb = src_emb + src_pos_emb
            enc_input = F.dropout(
                src_emb, p=self.dropout,
                training=False) if self.dropout else src_emb
            enc_output = self.transformer.encoder(enc_input, src_slf_attn_bias)

            # Init states (caches) for transformer, need to be updated according to selected beam
            incremental_cache, static_cache = self.transformer.decoder.gen_cache(
                enc_output, do_zip=True)

            static_cache, enc_output, trg_src_attn_bias = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                (static_cache, enc_output, trg_src_attn_bias), self.beam_size)

            rs, _ = nn.decode.dynamic_decode(
                decoder=self.decode,
                inits=incremental_cache,
                max_step_num=self.max_out_len,
                memory=enc_output,
                trg_src_attn_bias=trg_src_attn_bias,
                static_cache=static_cache,
                is_test=True,
                output_time_major=self.output_time_major,
                trg_word=trg_word,
                trg_length=trg_length)

            return rs

        elif self.beam_search_version == 'v2':
            finished_seq, finished_scores = self.beam_search_v2(
                src_word, self.beam_size, self.max_out_len, self.alpha,
                trg_word, trg_length)
            if self.output_time_major:
                finished_seq = finished_seq.transpose([2, 0, 1])
            else:
                finished_seq = finished_seq.transpose([0, 2, 1])

            return finished_seq

    def beam_search_v2(self,
                       src_word,
                       beam_size=4,
                       max_len=None,
                       alpha=0.6,
                       trg_word=None,
                       trg_length=None):
        """
        Beam search with the alive and finished two queues, both have a beam size
        capicity separately. It includes `grow_topk` `grow_alive` `grow_finish` as
        steps.
        1. `grow_topk` selects the top `2*beam_size` candidates to avoid all getting
        EOS.
        2. `grow_alive` selects the top `beam_size` non-EOS candidates as the inputs
        of next decoding step.
        3. `grow_finish` compares the already finished candidates in the finished queue
        and newly added finished candidates from `grow_topk`, and selects the top
        `beam_size` finished candidates.
        """

        def expand_to_beam_size(tensor, beam_size):
            tensor = paddle.unsqueeze(tensor, axis=1)
            tile_dims = [1] * len(tensor.shape)
            tile_dims[1] = beam_size
            return paddle.tile(tensor, tile_dims)

        def merge_beam_dim(tensor):
            shape = tensor.shape
            return paddle.reshape(tensor,
                                  [shape[0] * shape[1]] + list(shape[2:]))

        # run encoder
        src_max_len = paddle.shape(src_word)[-1]
        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
        src_slf_attn_bias.stop_gradient = True
        src_pos = paddle.cast(
            src_word != self.bos_id, dtype=src_word.dtype) * paddle.arange(
                start=0, end=src_max_len, dtype=src_word.dtype)
        src_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=self.training) if self.dropout else src_emb

        enc_output = self.transformer.encoder(enc_input, src_slf_attn_bias)

        # constant number
        inf = float(1. * 1e7)
        batch_size = enc_output.shape[0]
        max_len = (enc_output.shape[1] + 20) if max_len is None else (
            enc_output.shape[1] + max_len if self.rel_len else max_len)

        ### initialize states of beam search ###
        ## init for the alive ##
        initial_log_probs = paddle.assign(
            np.array(
                [[0.] + [-inf] * (beam_size - 1)], dtype="float32"))
        alive_log_probs = paddle.tile(initial_log_probs, [batch_size, 1])

        alive_seq = paddle.tile(
            paddle.cast(
                paddle.assign(np.array([[[self.bos_id]]])), src_word.dtype),
            [batch_size, beam_size, 1])

        ## init for the finished ##
        finished_scores = paddle.assign(
            np.array(
                [[-inf] * beam_size], dtype="float32"))
        finished_scores = paddle.tile(finished_scores, [batch_size, 1])

        finished_seq = paddle.tile(
            paddle.cast(
                paddle.assign(np.array([[[self.bos_id]]])), src_word.dtype),
            [batch_size, beam_size, 1])
        finished_flags = paddle.zeros_like(finished_scores)

        ### initialize inputs and states of transformer decoder ###
        ## init inputs for decoder, shaped `[batch_size*beam_size, ...]`
        pre_word = paddle.reshape(alive_seq[:, :, -1],
                                  [batch_size * beam_size, 1])
        trg_src_attn_bias = src_slf_attn_bias
        trg_src_attn_bias = merge_beam_dim(
            expand_to_beam_size(trg_src_attn_bias, beam_size))
        enc_output = merge_beam_dim(expand_to_beam_size(enc_output, beam_size))

        ## init states (caches) for transformer, need to be updated according to selected beam
        caches = self.transformer.decoder.gen_cache(enc_output, do_zip=False)

        if trg_word is not None:
            scores_dtype = finished_scores.dtype
            scores = paddle.ones(
                shape=[batch_size, beam_size * 2], dtype=scores_dtype) * -1e4
            scores = paddle.scatter(
                scores.flatten(),
                paddle.arange(
                    0,
                    batch_size * beam_size * 2,
                    step=beam_size * 2,
                    dtype=finished_seq.dtype),
                paddle.zeros([batch_size]))
            scores = paddle.reshape(scores, [batch_size, beam_size * 2])

        def update_states(caches, topk_coordinates, beam_size, batch_size):
            new_caches = []
            for cache in caches:
                k = gather_2d(
                    cache[0].k,
                    topk_coordinates,
                    beam_size,
                    batch_size,
                    need_unmerge=True)
                v = gather_2d(
                    cache[0].v,
                    topk_coordinates,
                    beam_size,
                    batch_size,
                    need_unmerge=True)
                new_caches.append((nn.MultiHeadAttention.Cache(k, v), cache[1]))
            return new_caches

        def get_topk_coordinates(beam_idx, beam_size, batch_size,
                                 dtype="int64"):
            batch_pos = paddle.arange(
                batch_size * beam_size, dtype=dtype) // beam_size
            batch_pos = paddle.reshape(batch_pos, [batch_size, beam_size])
            topk_coordinates = paddle.stack([batch_pos, beam_idx], axis=2)
            return topk_coordinates

        def gather_2d(tensor_nd,
                      topk_coordinates,
                      beam_size,
                      batch_size,
                      need_unmerge=False):

            new_tensor_nd = paddle.reshape(
                tensor_nd,
                shape=[batch_size, beam_size] +
                list(tensor_nd.shape[1:])) if need_unmerge else tensor_nd
            topk_seq = paddle.gather_nd(new_tensor_nd, topk_coordinates)
            return merge_beam_dim(topk_seq) if need_unmerge else topk_seq

        def early_finish(alive_log_probs, finished_scores,
                         finished_in_finished):
            max_length_penalty = np.power(((5. + max_len) / 6.), alpha)
            lower_bound_alive_scores = alive_log_probs[:,
                                                       0] / max_length_penalty
            lowest_score_of_fininshed_in_finished = paddle.min(
                finished_scores * finished_in_finished, 1)
            lowest_score_of_fininshed_in_finished += (
                1. - paddle.max(finished_in_finished, 1)) * -inf
            bound_is_met = paddle.all(
                paddle.greater_than(lowest_score_of_fininshed_in_finished,
                                    lower_bound_alive_scores))

            return bound_is_met

        def grow_topk(i, logits, alive_seq, alive_log_probs, states):
            """
            This function takes the current alive sequences, and grows them to topk
            sequences where k = 2*beam.
            """
            logits = paddle.reshape(logits, [batch_size, beam_size, -1])
            candidate_log_probs = paddle.log(F.softmax(logits, axis=2))
            log_probs = paddle.add(candidate_log_probs,
                                   alive_log_probs.unsqueeze(-1))

            # Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
            # https://arxiv.org/abs/1609.08144.
            length_penalty = paddle.pow((5.0 + i + 1.0) / 6.0, alpha)
            curr_scores = log_probs / length_penalty
            flat_curr_scores = paddle.reshape(curr_scores, [batch_size, -1])

            topk_scores, topk_ids = paddle.topk(
                flat_curr_scores, k=beam_size * 2)
            if topk_ids.dtype != alive_seq.dtype:
                topk_ids = paddle.cast(topk_ids, dtype=alive_seq.dtype)

            if trg_word is not None:
                topk_ids, topk_scores = force_decoding_v2(topk_ids, topk_scores,
                                                          i)

            topk_log_probs = topk_scores * length_penalty

            topk_beam_index = topk_ids // self.trg_vocab_size
            topk_ids = topk_ids % self.trg_vocab_size

            topk_coordinates = get_topk_coordinates(
                topk_beam_index,
                beam_size * 2,
                batch_size,
                dtype=alive_seq.dtype)
            topk_seq = gather_2d(alive_seq, topk_coordinates, beam_size,
                                 batch_size)
            topk_seq = paddle.concat(
                [
                    topk_seq, paddle.reshape(topk_ids,
                                             list(topk_ids.shape[:]) + [1])
                ],
                axis=2)
            states = update_states(states, topk_coordinates, beam_size,
                                   batch_size)
            eos = paddle.full(
                shape=paddle.shape(topk_ids),
                dtype=alive_seq.dtype,
                fill_value=self.eos_id)
            topk_finished = paddle.cast(paddle.equal(topk_ids, eos), "float32")

            # topk_seq: [batch_size, 2*beam_size, i+1]
            # topk_log_probs, topk_scores, topk_finished: [batch_size, 2*beam_size]
            return topk_seq, topk_log_probs, topk_scores, topk_finished, states

        def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished,
                       states):
            """
            Given sequences and scores, will gather the top k=beam size sequences
            """
            curr_scores += curr_finished * -inf
            _, topk_indexes = paddle.topk(curr_scores, k=beam_size)
            if topk_indexes.dtype != curr_seq.dtype:
                topk_indexes = paddle.cast(topk_indexes, dtype=curr_seq.dtype)

            topk_coordinates = get_topk_coordinates(
                topk_indexes, beam_size, batch_size, dtype=curr_seq.dtype)
            alive_seq = gather_2d(curr_seq, topk_coordinates, beam_size,
                                  batch_size)

            alive_log_probs = gather_2d(curr_log_probs, topk_coordinates,
                                        beam_size, batch_size)
            states = update_states(states, topk_coordinates, beam_size * 2,
                                   batch_size)

            return alive_seq, alive_log_probs, states

        def grow_finished(finished_seq, finished_scores, finished_flags,
                          curr_seq, curr_scores, curr_finished):
            """
            Given sequences and scores, will gather the top k=beam size sequences.
            """
            # finished scores
            finished_seq = paddle.concat(
                [
                    finished_seq, paddle.full(
                        shape=[batch_size, beam_size, 1],
                        dtype=finished_seq.dtype,
                        fill_value=self.eos_id)
                ],
                axis=2)
            curr_scores += (1. - curr_finished) * -inf
            curr_finished_seq = paddle.concat([finished_seq, curr_seq], axis=1)
            curr_finished_scores = paddle.concat(
                [finished_scores, curr_scores], axis=1)
            curr_finished_flags = paddle.concat(
                [finished_flags, curr_finished], axis=1)
            _, topk_indexes = paddle.topk(curr_finished_scores, k=beam_size)
            if topk_indexes.dtype != curr_seq.dtype:
                topk_indexes = paddle.cast(topk_indexes, dtype=curr_seq.dtype)

            topk_coordinates = get_topk_coordinates(
                topk_indexes, beam_size, batch_size, dtype=curr_seq.dtype)
            finished_seq = gather_2d(curr_finished_seq, topk_coordinates,
                                     beam_size, batch_size)
            finished_scores = gather_2d(curr_finished_scores, topk_coordinates,
                                        beam_size, batch_size)
            finished_flags = gather_2d(curr_finished_flags, topk_coordinates,
                                       beam_size, batch_size)

            return finished_seq, finished_scores, finished_flags

        def force_decoding_v2(topk_ids, topk_scores, time):
            beam_size = topk_ids.shape[1]
            if trg_word.shape[1] > time:
                force_position = paddle.unsqueeze(trg_length > time, [1])
                force_position.stop_gradient = True
                force_position = paddle.tile(force_position, [1, beam_size])

                crt_trg_word = paddle.slice(
                    trg_word, axes=[1], starts=[time], ends=[time + 1])
                crt_trg_word = paddle.tile(crt_trg_word, [1, beam_size])

                topk_ids = paddle.where(force_position, crt_trg_word, topk_ids)

                topk_scores = paddle.where(force_position, scores, topk_scores)

            return topk_ids, topk_scores

        def inner_loop(i, pre_word, alive_seq, alive_log_probs, finished_seq,
                       finished_scores, finished_flags, caches):
            trg_pos = paddle.full(
                shape=paddle.shape(pre_word),
                dtype=alive_seq.dtype,
                fill_value=i)
            trg_emb = self.trg_word_embedding(pre_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb

            logits, caches = self.transformer.decoder(
                dec_input, enc_output, None, trg_src_attn_bias, caches)
            logits = self.linear(logits)
            topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
                i, logits, alive_seq, alive_log_probs, caches)
            alive_seq, alive_log_probs, states = grow_alive(
                topk_seq, topk_scores, topk_log_probs, topk_finished, states)
            caches = states
            finished_seq, finished_scores, finished_flags = grow_finished(
                finished_seq, finished_scores, finished_flags, topk_seq,
                topk_scores, topk_finished)
            pre_word = paddle.reshape(alive_seq[:, :, -1],
                                      [batch_size * beam_size, 1])
            return (i + 1, pre_word, alive_seq, alive_log_probs, finished_seq,
                    finished_scores, finished_flags, caches)

        def is_not_finish(i, pre_word, alive_seq, alive_log_probs, finished_seq,
                          finished_scores, finished_flags, caches):
            return paddle.greater_than(
                i < max_len,
                early_finish(alive_log_probs, finished_scores, finished_flags))

        _, pre_word, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags, caches = paddle.static.nn.while_loop(
            is_not_finish,
            inner_loop, [
                paddle.zeros(
                    shape=[1],
                    dtype="int64"), pre_word, alive_seq, alive_log_probs,
                finished_seq, finished_scores, finished_flags, caches
            ])

        # (gongenlei) `paddle.where` doesn't support broadcast, so we need to use `paddle.unsqueeze`
        # and `paddle.tile` to make condition.shape same as X.shape. But when converting dygraph
        # to static  graph, `paddle.tile` will raise error.
        finished_flags = paddle.cast(finished_flags, dtype=finished_seq.dtype)
        neg_finished_flags = 1 - finished_flags
        finished_seq = paddle.multiply(
            finished_seq, finished_flags.unsqueeze(-1)) + paddle.multiply(
                alive_seq, neg_finished_flags.unsqueeze(-1))
        finished_scores = paddle.multiply(
            finished_scores,
            paddle.cast(
                finished_flags, dtype=finished_scores.dtype)) + paddle.multiply(
                    alive_log_probs,
                    paddle.cast(
                        neg_finished_flags, dtype=alive_log_probs.dtype))
        return finished_seq, finished_scores
