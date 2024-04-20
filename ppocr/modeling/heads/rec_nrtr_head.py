# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import Dropout, LayerNorm
import numpy as np
from ppocr.modeling.backbones.rec_svtrnet import Mlp, zeros_
from paddle.nn.initializer import XavierNormal as xavier_normal_


class Transformer(nn.Layer):
    """A transformer model. User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        beam_size=0,
        num_decoder_layers=6,
        max_len=25,
        dim_feedforward=1024,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        in_channels=0,
        out_channels=0,
        scale_embedding=True,
    ):
        super(Transformer, self).__init__()
        self.out_channels = out_channels + 1
        self.max_len = max_len
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model
        )

        if num_encoder_layers > 0:
            self.encoder = nn.LayerList(
                [
                    TransformerBlock(
                        d_model,
                        nhead,
                        dim_feedforward,
                        attention_dropout_rate,
                        residual_dropout_rate,
                        with_self_attn=True,
                        with_cross_attn=False,
                    )
                    for i in range(num_encoder_layers)
                ]
            )
        else:
            self.encoder = None

        self.decoder = nn.LayerList(
            [
                TransformerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    attention_dropout_rate,
                    residual_dropout_rate,
                    with_self_attn=True,
                    with_cross_attn=True,
                )
                for i in range(num_decoder_layers)
            ]
        )

        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels, bias_attr=False)
        w0 = np.random.normal(
            0.0, d_model**-0.5, (d_model, self.out_channels)
        ).astype(np.float32)
        self.tgt_word_prj.weight.set_value(w0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1])

        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src  # B N C
        else:
            memory = src  # B N C
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
        output = tgt
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, targets=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        Shape:
            - src: :math:`(B, sN, C)`.
            - tgt: :math:`(B, tN, C)`.
        Examples:
            >>> output = transformer_model(src, tgt)
        """

        if self.training:
            max_len = targets[1].max()
            tgt = targets[0][:, : 2 + max_len]
            return self.forward_train(src, tgt)
        else:
            if self.beam_size > 0:
                return self.forward_beam(src)
            else:
                return self.forward_test(src)

    def forward_test(self, src):
        bs = src.shape[0]
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src  # B N C
        else:
            memory = src
        dec_seq = paddle.full((bs, 1), 2, dtype=paddle.int64)
        dec_prob = paddle.full((bs, 1), 1.0, dtype=paddle.float32)
        for len_dec_seq in range(1, paddle.to_tensor(self.max_len)):
            dec_seq_embed = self.embedding(dec_seq)
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.shape[1])
            tgt = dec_seq_embed
            for decoder_layer in self.decoder:
                tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
            dec_output = tgt
            dec_output = dec_output[:, -1, :]
            word_prob = F.softmax(self.tgt_word_prj(dec_output), axis=-1)
            preds_idx = paddle.argmax(word_prob, axis=-1)
            if paddle.equal_all(
                preds_idx, paddle.full(preds_idx.shape, 3, dtype="int64")
            ):
                break
            preds_prob = paddle.max(word_prob, axis=-1)
            dec_seq = paddle.concat(
                [dec_seq, paddle.reshape(preds_idx, [-1, 1])], axis=1
            )
            dec_prob = paddle.concat(
                [dec_prob, paddle.reshape(preds_prob, [-1, 1])], axis=1
            )
        return [dec_seq, dec_prob]

    def forward_beam(self, images):
        """Translation work in one batch"""

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """Indicate the position of an instance in a tensor."""
            return {
                inst_idx: tensor_position
                for tensor_position, inst_idx in enumerate(inst_idx_list)
            }

        def collect_active_part(
            beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm
        ):
            """Collect tensor parts associated to active instances."""

            beamed_tensor_shape = beamed_tensor.shape
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (
                n_curr_active_inst * n_bm,
                beamed_tensor_shape[1],
                beamed_tensor_shape[2],
            )

            beamed_tensor = beamed_tensor.reshape([n_prev_active_inst, -1])
            beamed_tensor = beamed_tensor.index_select(curr_active_inst_idx, axis=0)
            beamed_tensor = beamed_tensor.reshape(new_shape)

            return beamed_tensor

        def collate_active_info(
            src_enc, inst_idx_to_position_map, active_inst_idx_list
        ):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.

            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [
                inst_idx_to_position_map[k] for k in active_inst_idx_list
            ]
            active_inst_idx = paddle.to_tensor(active_inst_idx, dtype="int64")
            active_src_enc = collect_active_part(
                src_enc.transpose([1, 0, 2]), active_inst_idx, n_prev_active_inst, n_bm
            ).transpose([1, 0, 2])
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )
            return active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
            inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm
        ):
            """Decode and update beam status, and then return active beam idx"""

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [
                    b.get_current_state() for b in inst_dec_beams if not b.done
                ]
                dec_partial_seq = paddle.stack(dec_partial_seq)
                dec_partial_seq = dec_partial_seq.reshape([-1, len_dec_seq])
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
                dec_seq = self.embedding(dec_seq)
                dec_seq = self.positional_encoding(dec_seq)
                tgt_mask = self.generate_square_subsequent_mask(dec_seq.shape[1])
                tgt = dec_seq
                for decoder_layer in self.decoder:
                    tgt = decoder_layer(tgt, enc_output, self_mask=tgt_mask)
                dec_output = tgt
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.softmax(self.tgt_word_prj(dec_output), axis=1)
                word_prob = paddle.reshape(word_prob, [n_active_inst, n_bm, -1])
                return word_prob

            def collect_active_inst_idx_list(
                inst_beams, word_prob, inst_idx_to_position_map
            ):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(
                        word_prob[inst_position]
                    )
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)
            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map
            )
            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]
                hyps = [
                    inst_dec_beams[inst_idx].get_hypothesis(i)
                    for i in tail_idxs[:n_best]
                ]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with paddle.no_grad():
            # -- Encode
            if self.encoder is not None:
                src = self.positional_encoding(images)
                src_enc = self.encoder(src)
            else:
                src_enc = images

            n_bm = self.beam_size
            src_shape = src_enc.shape
            inst_dec_beams = [Beam(n_bm) for _ in range(1)]
            active_inst_idx_list = list(range(1))
            # Repeat data for beam search
            src_enc = paddle.tile(src_enc, [1, n_bm, 1])
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )
            # Decode
            for len_dec_seq in range(1, paddle.to_tensor(self.max_len)):
                src_enc_copy = src_enc.clone()
                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    src_enc_copy,
                    inst_idx_to_position_map,
                    n_bm,
                )
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>
                src_enc, inst_idx_to_position_map = collate_active_info(
                    src_enc_copy, inst_idx_to_position_map, active_inst_idx_list
                )
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
        result_hyp = []
        hyp_scores = []
        for bs_hyp, score in zip(batch_hyp, batch_scores):
            l = len(bs_hyp[0])
            bs_hyp_pad = bs_hyp[0] + [3] * (25 - l)
            result_hyp.append(bs_hyp_pad)
            score = float(score) / l
            hyp_score = [score for _ in range(25)]
            hyp_scores.append(hyp_score)
        return [
            paddle.to_tensor(np.array(result_hyp), dtype=paddle.int64),
            paddle.to_tensor(hyp_scores),
        ]

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = paddle.zeros([sz, sz], dtype="float32")
        mask_inf = paddle.triu(
            paddle.full(shape=[sz, sz], dtype="float32", fill_value="-inf"), diagonal=1
        )
        mask = mask + mask_inf
        return mask.unsqueeze([0, 1])


class MultiheadAttention(nn.Layer):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim**-0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        qN = query.shape[1]

        if self.self_attn:
            qkv = (
                self.qkv(query)
                .reshape((0, qN, 3, self.num_heads, self.head_dim))
                .transpose((2, 0, 3, 1, 4))
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            kN = key.shape[1]
            q = (
                self.q(query)
                .reshape([0, qN, self.num_heads, self.head_dim])
                .transpose([0, 2, 1, 3])
            )
            kv = (
                self.kv(key)
                .reshape((0, kN, 2, self.num_heads, self.head_dim))
                .transpose((2, 0, 3, 1, 4))
            )
            k, v = kv[0], kv[1]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale

        if attn_mask is not None:
            attn += attn_mask

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, qN, self.embed_dim))
        x = self.out_proj(x)

        return x


class TransformerBlock(nn.Layer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        with_self_attn=True,
        with_cross_attn=False,
        epsilon=1e-5,
    ):
        super(TransformerBlock, self).__init__()
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=attention_dropout_rate, self_attn=with_self_attn
            )
            self.norm1 = LayerNorm(d_model, epsilon=epsilon)
            self.dropout1 = Dropout(residual_dropout_rate)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = (
                MultiheadAttention(  # for self_attn of encoder or cross_attn of decoder
                    d_model, nhead, dropout=attention_dropout_rate
                )
            )
            self.norm2 = LayerNorm(d_model, epsilon=epsilon)
            self.dropout2 = Dropout(residual_dropout_rate)

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            act_layer=nn.ReLU,
            drop=residual_dropout_rate,
        )

        self.norm3 = LayerNorm(d_model, epsilon=epsilon)

        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))

        if self.with_cross_attn:
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class PositionalEncoding(nn.Layer):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = paddle.zeros([max_len, dim])
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, dim, 2).astype("float32") * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = paddle.unsqueeze(pe, 0)
        pe = paddle.transpose(pe, [1, 0, 2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x.transpose([1, 0, 2])
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x).transpose([1, 0, 2])


class PositionalEncoding_2d(nn.Layer):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding_2d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = paddle.zeros([max_len, dim])
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, dim, 2).astype("float32") * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = paddle.transpose(paddle.unsqueeze(pe, 0), [1, 0, 2])
        self.register_buffer("pe", pe)

        self.avg_pool_1 = nn.AdaptiveAvgPool2D((1, 1))
        self.linear1 = nn.Linear(dim, dim)
        self.linear1.weight.data.fill_(1.0)
        self.avg_pool_2 = nn.AdaptiveAvgPool2D((1, 1))
        self.linear2 = nn.Linear(dim, dim)
        self.linear2.weight.data.fill_(1.0)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        w_pe = self.pe[: x.shape[-1], :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = paddle.transpose(w_pe, [1, 2, 0])
        w_pe = paddle.unsqueeze(w_pe, 2)

        h_pe = self.pe[: x.shape.shape[-2], :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = paddle.transpose(h_pe, [1, 2, 0])
        h_pe = paddle.unsqueeze(h_pe, 3)

        x = x + w_pe + h_pe
        x = paddle.transpose(
            paddle.reshape(x, [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]),
            [2, 0, 1],
        )

        return self.dropout(x)


class Embeddings(nn.Layer):
    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        w0 = np.random.normal(0.0, d_model**-0.5, (vocab, d_model)).astype(np.float32)
        self.embedding.weight.set_value(w0)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)


class Beam:
    """Beam search"""

    def __init__(self, size, device=False):
        self.size = size
        self._done = False
        # The score for each translation on the beam.
        self.scores = paddle.zeros((size,), dtype=paddle.float32)
        self.all_scores = []
        # The backpointers at each time-step.
        self.prev_ks = []
        # The outputs at each time-step.
        self.next_ys = [paddle.full((size,), 0, dtype=paddle.int64)]
        self.next_ys[0][0] = 2

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.shape[1]

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.reshape([-1])
        best_scores, best_scores_id = flat_beam_lk.topk(
            self.size, 0, True, True
        )  # 1st sort
        self.all_scores.append(self.scores)
        self.scores = best_scores
        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == 3:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return self.scores, paddle.to_tensor(
            [i for i in range(int(self.scores.shape[0]))], dtype="int32"
        )

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[2] + h for h in hyps]
            dec_seq = paddle.to_tensor(hyps, dtype="int64")
        return dec_seq

    def get_hypothesis(self, k):
        """Walk back to construct the full hypothesis."""
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return list(map(lambda x: x.item(), hyp[::-1]))
