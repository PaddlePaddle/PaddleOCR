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
import copy
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import LayerList
from paddle.nn.initializer import XavierNormal as xavier_uniform_
from paddle.nn import Dropout, Linear, LayerNorm, Conv2D
import numpy as np
from ppocr.modeling.heads.multiheadAttention import MultiheadAttentionOptim
from paddle.nn.initializer import Constant as constant_
from paddle.nn.initializer import XavierNormal as xavier_normal_

zeros_ = constant_(value=0.)
ones_ = constant_(value=1.)


class TransformerOptim(nn.Layer):
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

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 beam_size=0,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 attention_dropout_rate=0.0,
                 residual_dropout_rate=0.1,
                 custom_encoder=None,
                 custom_decoder=None,
                 in_channels=0,
                 out_channels=0,
                 dst_vocab_size=99,
                 scale_embedding=True):
        super(TransformerOptim, self).__init__()
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=dst_vocab_size,
            padding_idx=0,
            scale_embedding=scale_embedding)
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate,
            dim=d_model, )
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            if num_encoder_layers > 0:
                encoder_layer = TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, attention_dropout_rate,
                    residual_dropout_rate)
                self.encoder = TransformerEncoder(encoder_layer,
                                                  num_encoder_layers)
            else:
                self.encoder = None

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, attention_dropout_rate,
                residual_dropout_rate)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self._reset_parameters()
        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model, dst_vocab_size, bias_attr=False)
        w0 = np.random.normal(0.0, d_model**-0.5,
                              (d_model, dst_vocab_size)).astype(np.float32)
        self.tgt_word_prj.weight.set_value(w0)
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Conv2D):
            xavier_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]

        tgt_key_padding_mask = self.generate_padding_mask(tgt)
        tgt = self.embedding(tgt).transpose([1, 0, 2])
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[0])

        if self.encoder is not None:
            src = self.positional_encoding(src.transpose([1, 0, 2]))
            memory = self.encoder(src)
        else:
            memory = src.squeeze(2).transpose([2, 0, 1])
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None)
        output = output.transpose([1, 0, 2])
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, targets=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
        Examples:
            >>> output = transformer_model(src, tgt)
        """

        if self.training:
            max_len = targets[1].max()
            tgt = targets[0][:, :2 + max_len]
            return self.forward_train(src, tgt)
        else:
            if self.beam_size > 0:
                return self.forward_beam(src)
            else:
                return self.forward_test(src)

    def forward_test(self, src):
        bs = src.shape[0]
        if self.encoder is not None:
            src = self.positional_encoding(src.transpose([1, 0, 2]))
            memory = self.encoder(src)
        else:
            memory = src.squeeze(2).transpose([2, 0, 1])
        dec_seq = paddle.full((bs, 1), 2, dtype=paddle.int64)
        for len_dec_seq in range(1, 25):
            src_enc = memory.clone()
            tgt_key_padding_mask = self.generate_padding_mask(dec_seq)
            dec_seq_embed = self.embedding(dec_seq).transpose([1, 0, 2])
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.shape[
                0])
            output = self.decoder(
                dec_seq_embed,
                src_enc,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=None)
            dec_output = output.transpose([1, 0, 2])

            dec_output = dec_output[:,
                                    -1, :]  # Pick the last step: (bh * bm) * d_h
            word_prob = F.log_softmax(self.tgt_word_prj(dec_output), axis=1)
            word_prob = word_prob.reshape([1, bs, -1])
            preds_idx = word_prob.argmax(axis=2)

            if paddle.equal_all(
                    preds_idx[-1],
                    paddle.full(
                        preds_idx[-1].shape, 3, dtype='int64')):
                break

            preds_prob = word_prob.max(axis=2)
            dec_seq = paddle.concat(
                [dec_seq, preds_idx.reshape([-1, 1])], axis=1)

        return dec_seq

    def forward_beam(self, images):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {
                inst_idx: tensor_position
                for tensor_position, inst_idx in enumerate(inst_idx_list)
            }

        def collect_active_part(beamed_tensor, curr_active_inst_idx,
                                n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.shape
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.reshape(
                [n_prev_active_inst, -1])  #contiguous()
            beamed_tensor = beamed_tensor.index_select(
                paddle.to_tensor(curr_active_inst_idx), axis=0)
            beamed_tensor = beamed_tensor.reshape([*new_shape])

            return beamed_tensor

        def collate_active_info(src_enc, inst_idx_to_position_map,
                                active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.

            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [
                inst_idx_to_position_map[k] for k in active_inst_idx_list
            ]
            active_inst_idx = paddle.to_tensor(active_inst_idx, dtype='int64')
            active_src_enc = collect_active_part(
                src_enc.transpose([1, 0, 2]), active_inst_idx,
                n_prev_active_inst, n_bm).transpose([1, 0, 2])
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list)
            return active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output,
                             inst_idx_to_position_map, n_bm,
                             memory_key_padding_mask):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [
                    b.get_current_state() for b in inst_dec_beams if not b.done
                ]
                dec_partial_seq = paddle.stack(dec_partial_seq)

                dec_partial_seq = dec_partial_seq.reshape([-1, len_dec_seq])
                return dec_partial_seq

            def prepare_beam_memory_key_padding_mask(
                    inst_dec_beams, memory_key_padding_mask, n_bm):
                keep = []
                for idx in (memory_key_padding_mask):
                    if not inst_dec_beams[idx].done:
                        keep.append(idx)
                memory_key_padding_mask = memory_key_padding_mask[
                    paddle.to_tensor(keep)]
                len_s = memory_key_padding_mask.shape[-1]
                n_inst = memory_key_padding_mask.shape[0]
                memory_key_padding_mask = paddle.concat(
                    [memory_key_padding_mask for i in range(n_bm)], axis=1)
                memory_key_padding_mask = memory_key_padding_mask.reshape(
                    [n_inst * n_bm, len_s])  #repeat(1, n_bm)
                return memory_key_padding_mask

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm,
                             memory_key_padding_mask):
                tgt_key_padding_mask = self.generate_padding_mask(dec_seq)
                dec_seq = self.embedding(dec_seq).transpose([1, 0, 2])
                dec_seq = self.positional_encoding(dec_seq)
                tgt_mask = self.generate_square_subsequent_mask(dec_seq.shape[
                    0])
                dec_output = self.decoder(
                    dec_seq,
                    enc_output,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                ).transpose([1, 0, 2])
                dec_output = dec_output[:,
                                        -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(self.tgt_word_prj(dec_output), axis=1)
                word_prob = word_prob.reshape([n_active_inst, n_bm, -1])
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob,
                                             inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[
                        inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            memory_key_padding_mask = None
            word_prob = predict_word(dec_seq, enc_output, n_active_inst, n_bm,
                                     memory_key_padding_mask)
            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)
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
            #-- Encode

            if self.encoder is not None:
                src = self.positional_encoding(images.transpose([1, 0, 2]))
                src_enc = self.encoder(src).transpose([1, 0, 2])
            else:
                src_enc = images.squeeze(2).transpose([0, 2, 1])

            #-- Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s, d_h = src_enc.shape
            src_enc = paddle.concat([src_enc for i in range(n_bm)], axis=1)
            src_enc = src_enc.reshape([n_inst * n_bm, len_s, d_h]).transpose(
                [1, 0, 2])  #repeat(1, n_bm, 1)
            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list)
            #-- Decode
            for len_dec_seq in range(1, 25):
                src_enc_copy = src_enc.clone()
                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_enc_copy,
                    inst_idx_to_position_map, n_bm, None)
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>
                src_enc, inst_idx_to_position_map = collate_active_info(
                    src_enc_copy, inst_idx_to_position_map,
                    active_inst_idx_list)
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams,
                                                                1)
        result_hyp = []
        for bs_hyp in batch_hyp:
            bs_hyp_pad = bs_hyp[0] + [3] * (25 - len(bs_hyp[0]))
            result_hyp.append(bs_hyp_pad)
        return paddle.to_tensor(np.array(result_hyp), dtype=paddle.int64)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = paddle.zeros([sz, sz], dtype='float32')
        mask_inf = paddle.triu(
            paddle.full(
                shape=[sz, sz], dtype='float32', fill_value='-inf'),
            diagonal=1)
        mask = mask + mask_inf
        return mask

    def generate_padding_mask(self, x):
        padding_mask = x.equal(paddle.to_tensor(0, dtype=x.dtype))
        return padding_mask

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(nn.Layer):
    """TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        """Pass the input through the endocder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output,
                                    src_mask=None,
                                    src_key_padding_mask=None)

        return output


class TransformerDecoder(nn.Layer):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        output = tgt
        for i in range(self.num_layers):
            output = self.layers[i](
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)

        return output


class TransformerEncoderLayer(nn.Layer):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 attention_dropout_rate=0.0,
                 residual_dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttentionOptim(
            d_model, nhead, dropout=attention_dropout_rate)

        self.conv1 = Conv2D(
            in_channels=d_model,
            out_channels=dim_feedforward,
            kernel_size=(1, 1))
        self.conv2 = Conv2D(
            in_channels=dim_feedforward,
            out_channels=d_model,
            kernel_size=(1, 1))

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(residual_dropout_rate)
        self.dropout2 = Dropout(residual_dropout_rate)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the endocder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = src.transpose([1, 2, 0])
        src = paddle.unsqueeze(src, 2)
        src2 = self.conv2(F.relu(self.conv1(src)))
        src2 = paddle.squeeze(src2, 2)
        src2 = src2.transpose([2, 0, 1])
        src = paddle.squeeze(src, 2)
        src = src.transpose([2, 0, 1])

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Layer):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 attention_dropout_rate=0.0,
                 residual_dropout_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttentionOptim(
            d_model, nhead, dropout=attention_dropout_rate)
        self.multihead_attn = MultiheadAttentionOptim(
            d_model, nhead, dropout=attention_dropout_rate)

        self.conv1 = Conv2D(
            in_channels=d_model,
            out_channels=dim_feedforward,
            kernel_size=(1, 1))
        self.conv2 = Conv2D(
            in_channels=dim_feedforward,
            out_channels=d_model,
            kernel_size=(1, 1))

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(residual_dropout_rate)
        self.dropout2 = Dropout(residual_dropout_rate)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        """
        tgt2 = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # default
        tgt = tgt.transpose([1, 2, 0])
        tgt = paddle.unsqueeze(tgt, 2)
        tgt2 = self.conv2(F.relu(self.conv1(tgt)))
        tgt2 = paddle.squeeze(tgt2, 2)
        tgt2 = tgt2.transpose([2, 0, 1])
        tgt = paddle.squeeze(tgt, 2)
        tgt = tgt.transpose([2, 0, 1])

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return LayerList([copy.deepcopy(module) for i in range(N)])


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
            paddle.arange(0, dim, 2).astype('float32') *
            (-math.log(10000.0) / dim))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.transpose([1, 0, 2])
        self.register_buffer('pe', pe)

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
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


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
            paddle.arange(0, dim, 2).astype('float32') *
            (-math.log(10000.0) / dim))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose([1, 0, 2])
        self.register_buffer('pe', pe)

        self.avg_pool_1 = nn.AdaptiveAvgPool2D((1, 1))
        self.linear1 = nn.Linear(dim, dim)
        self.linear1.weight.data.fill_(1.)
        self.avg_pool_2 = nn.AdaptiveAvgPool2D((1, 1))
        self.linear2 = nn.Linear(dim, dim)
        self.linear2.weight.data.fill_(1.)

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
        w_pe = self.pe[:x.shape[-1], :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = w_pe.transpose([1, 2, 0])
        w_pe = w_pe.unsqueeze(2)

        h_pe = self.pe[:x.shape[-2], :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = h_pe.transpose([1, 2, 0])
        h_pe = h_pe.unsqueeze(3)

        x = x + w_pe + h_pe
        x = x.reshape(
            [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]).transpose(
                [2, 0, 1])

        return self.dropout(x)


class Embeddings(nn.Layer):
    def __init__(self, d_model, vocab, padding_idx, scale_embedding):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        w0 = np.random.normal(0.0, d_model**-0.5,
                              (vocab, d_model)).astype(np.float32)
        self.embedding.weight.set_value(w0)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)


class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False):

        self.size = size
        self._done = False
        # The score for each translation on the beam.
        self.scores = paddle.zeros((size, ), dtype=paddle.float32)
        self.all_scores = []
        # The backpointers at each time-step.
        self.prev_ks = []
        # The outputs at each time-step.
        self.next_ys = [paddle.full((size, ), 0, dtype=paddle.int64)]
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
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True,
                                                        True)  # 1st sort
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
            [i for i in range(self.scores.shape[0])], dtype='int32')

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
            dec_seq = paddle.to_tensor(hyps, dtype='int64')
        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return list(map(lambda x: x.item(), hyp[::-1]))
