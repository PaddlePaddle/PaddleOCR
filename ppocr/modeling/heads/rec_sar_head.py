from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F


class SAREncoder(nn.Layer):
    """
    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        enc_drop_rnn (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(self,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 enc_gru=False,
                 d_model=512,
                 d_enc=512,
                 mask=True,
                 **kwargs):
        super().__init__()
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_drop_rnn, (int, float))
        assert 0 <= enc_drop_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        self.enc_bi_rnn = enc_bi_rnn
        self.enc_drop_rnn = enc_drop_rnn
        self.mask = mask

        # LSTM Encoder
        if enc_bi_rnn:
            direction = 'bidirectional'
        else:
            direction = 'forward'
        kwargs = dict(
            input_size=d_model,
            hidden_size=d_enc,
            num_layers=2,
            time_major=False,
            dropout=enc_drop_rnn,
            direction=direction)
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # global feature transformation
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat, img_metas=None):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        h_feat = feat.shape[2]  # bsz c h w
        feat_v = F.max_pool2d(
            feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        feat_v = feat_v.squeeze(2)  # bsz * C * W
        feat_v = paddle.transpose(feat_v, perm=[0, 2, 1])  # bsz * W * C
        holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * C

        if valid_ratios is not None:
            valid_hf = []
            T = holistic_feat.shape[1]
            for i, valid_ratio in enumerate(valid_ratios):
                valid_step = min(T, math.ceil(T * valid_ratio)) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = paddle.stack(valid_hf, axis=0)
        else:
            valid_hf = holistic_feat[:, -1, :]  # bsz * C
        holistic_feat = self.linear(valid_hf)  # bsz * C

        return holistic_feat


class BaseDecoder(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                label=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode

        if train_mode:
            return self.forward_train(feat, out_enc, label, img_metas)
        return self.forward_test(feat, out_enc, img_metas)


class ParallelSARDecoder(BaseDecoder):
    """
    Args:
        out_channels (int): Output class number.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder.
        dec_drop_rnn (float): Dropout of RNN layer in decoder.
        dec_gru (bool): If True, use GRU, else LSTM in decoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        d_k (int): Dim of channels of attention module.
        pred_dropout (float): Dropout probability of prediction layer.
        max_seq_len (int): Maximum sequence length for decoding.
        mask (bool): If True, mask padding in feature map.
        start_idx (int): Index of start token.
        padding_idx (int): Index of padding token.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state.
    """

    def __init__(
            self,
            out_channels,  # 90 + unknown + start + padding
            enc_bi_rnn=False,
            dec_bi_rnn=False,
            dec_drop_rnn=0.0,
            dec_gru=False,
            d_model=512,
            d_enc=512,
            d_k=64,
            pred_dropout=0.1,
            max_text_length=30,
            mask=True,
            pred_concat=True,
            **kwargs):
        super().__init__()

        self.num_classes = out_channels
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.max_seq_len = max_text_length
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)

        # 2D attention layer
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2D(
            d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)

        # Decoder RNN layer
        if dec_bi_rnn:
            direction = 'bidirectional'
        else:
            direction = 'forward'

        kwargs = dict(
            input_size=encoder_rnn_out_size,
            hidden_size=encoder_rnn_out_size,
            num_layers=2,
            time_major=False,
            dropout=dec_drop_rnn,
            direction=direction)
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)

        # Decoder input embedding
        self.embedding = nn.Embedding(
            self.num_classes,
            encoder_rnn_out_size,
            padding_idx=self.padding_idx)

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_classes = self.num_classes - 1
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + d_enc
        else:
            fc_in_channel = d_model
        self.prediction = nn.Linear(fc_in_channel, pred_num_classes)

    def _2d_attention(self,
                      decoder_input,
                      feat,
                      holistic_feat,
                      valid_ratios=None):

        y = self.rnn_decoder(decoder_input)[0]
        # y: bsz * (seq_len + 1) * hidden_size

        attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
        bsz, seq_len, attn_size = attn_query.shape
        attn_query = paddle.unsqueeze(attn_query, axis=[3, 4])
        # (bsz, seq_len + 1, attn_size, 1, 1)

        attn_key = self.conv3x3_1(feat)
        # bsz * attn_size * h * w
        attn_key = attn_key.unsqueeze(1)
        # bsz * 1 * attn_size * h * w

        attn_weight = paddle.tanh(paddle.add(attn_key, attn_query))

        # bsz * (seq_len + 1) * attn_size * h * w
        attn_weight = paddle.transpose(attn_weight, perm=[0, 1, 3, 4, 2])
        # bsz * (seq_len + 1) * h * w * attn_size
        attn_weight = self.conv1x1_2(attn_weight)
        # bsz * (seq_len + 1) * h * w * 1
        bsz, T, h, w, c = attn_weight.shape
        assert c == 1

        if valid_ratios is not None:
            # cal mask of attention weight
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_weight[i, :, :, valid_width:, :] = float('-inf')

        attn_weight = paddle.reshape(attn_weight, [bsz, T, -1])
        attn_weight = F.softmax(attn_weight, axis=-1)

        attn_weight = paddle.reshape(attn_weight, [bsz, T, h, w, c])
        attn_weight = paddle.transpose(attn_weight, perm=[0, 1, 4, 2, 3])
        # attn_weight: bsz * T * c * h * w
        # feat: bsz * c * h * w
        attn_feat = paddle.sum(paddle.multiply(feat.unsqueeze(1), attn_weight),
                               (3, 4),
                               keepdim=False)
        # bsz * (seq_len + 1) * C

        # Linear transformation
        if self.pred_concat:
            hf_c = holistic_feat.shape[-1]
            holistic_feat = paddle.expand(
                holistic_feat, shape=[bsz, seq_len, hf_c])
            y = self.prediction(paddle.concat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)
        # bsz * (seq_len + 1) * num_classes
        if self.train_mode:
            y = self.pred_dropout(y)

        return y

    def forward_train(self, feat, out_enc, label, img_metas):
        '''
        img_metas: [label, valid_ratio]
        '''
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        label = label.cuda()
        lab_embedding = self.embedding(label)
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        in_dec = paddle.concat((out_enc, lab_embedding), axis=1)
        # bsz * (seq_len + 1) * C
        out_dec = self._2d_attention(
            in_dec, feat, out_enc, valid_ratios=valid_ratios)
        # bsz * (seq_len + 1) * num_classes

        return out_dec[:, 1:, :]  # bsz * seq_len * num_classes

    def forward_test(self, feat, out_enc, img_metas):
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        if img_metas is not None and self.mask:
            valid_ratios = img_metas[-1]

        seq_len = self.max_seq_len
        bsz = feat.shape[0]
        start_token = paddle.full(
            (bsz, ), fill_value=self.start_idx, dtype='int64')
        # bsz
        start_token = self.embedding(start_token)
        # bsz * emb_dim
        emb_dim = start_token.shape[1]
        start_token = start_token.unsqueeze(1)
        start_token = paddle.expand(start_token, shape=[bsz, seq_len, emb_dim])
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        decoder_input = paddle.concat((out_enc, start_token), axis=1)
        # bsz * (seq_len + 1) * emb_dim

        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(
                decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            char_output = decoder_output[:, i, :]  # bsz * num_classes
            char_output = F.softmax(char_output, -1)
            outputs.append(char_output)
            max_idx = paddle.argmax(char_output, axis=1, keepdim=False)
            char_embedding = self.embedding(max_idx)  # bsz * emb_dim
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding

        outputs = paddle.stack(outputs, 1)  # bsz * seq_len * num_classes

        return outputs


class SARHead(nn.Layer):
    def __init__(self,
                 out_channels,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 enc_gru=False,
                 dec_bi_rnn=False,
                 dec_drop_rnn=0.0,
                 dec_gru=False,
                 d_k=512,
                 pred_dropout=0.1,
                 max_text_length=30,
                 pred_concat=True,
                 **kwargs):
        super(SARHead, self).__init__()

        # encoder module
        self.encoder = SAREncoder(
            enc_bi_rnn=enc_bi_rnn, enc_drop_rnn=enc_drop_rnn, enc_gru=enc_gru)

        # decoder module
        self.decoder = ParallelSARDecoder(
            out_channels=out_channels,
            enc_bi_rnn=enc_bi_rnn,
            dec_bi_rnn=dec_bi_rnn,
            dec_drop_rnn=dec_drop_rnn,
            dec_gru=dec_gru,
            d_k=d_k,
            pred_dropout=pred_dropout,
            max_text_length=max_text_length,
            pred_concat=pred_concat)

    def forward(self, feat, targets=None):
        '''
        img_metas: [label, valid_ratio]
        '''
        holistic_feat = self.encoder(feat, targets)  # bsz c

        if self.training:
            label = targets[0]  # label
            label = paddle.to_tensor(label, dtype='int64')
            final_out = self.decoder(
                feat, holistic_feat, label, img_metas=targets)
        if not self.training:
            final_out = self.decoder(
                feat,
                holistic_feat,
                label=None,
                img_metas=targets,
                train_mode=False)
            # (bsz, seq_len, num_classes)

        return final_out
