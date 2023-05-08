# for SPTS build model
import sys
import copy
import math
import paddle
from typing import Optional

from paddle import nn, Tensor
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
import paddle.nn.functional as F
import numpy as np


weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
bias_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (paddle.triu(paddle.ones([sz, sz])) == 1).transpose([1, 0])
    mask_shape = mask.shape
    if not min(mask_shape) > 0:
        if max(mask_shape) > 0:
            mask_shape = [max(mask_shape), max(mask_shape)]
        else:
            mask_shape = [1, 1]
    masks = paddle.zeros(mask_shape).astype('float32')
    masks[~mask] = float('-inf')
    return masks


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])


def build_position_embedding(config):
    """建立pos_embed"""
    N_steps = config['tfm_hidden_dim'] // 2
    if config['position_embedding'] in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {config['position_embedding']}")

    return position_embedding


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Transformer(nn.Layer):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 dropout, normalize_before, pad_token_id, num_classes, max_position_embeddings, 
                 return_intermediate_dec, num_bins, eos_index, activation="relu"):
        super(Transformer, self).__init__()
        self.embedding = DecoderEmbeddings(num_classes, d_model, pad_token_id, max_position_embeddings, dropout)
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        self.nhead = nhead
        self.d_model = d_model
        self.num_bins = num_bins
        self.eos_index = eos_index
        self.num_encoder_layers = num_encoder_layers
        self.max_position_embeddings = max_position_embeddings

    # @paddle.jit.to_static()
    def forward(self, src, mask, pos_embed, seq, vocab_embed):
        bs, _, _, _ = src.shape
        src = src.flatten(2).transpose([0, 2, 1])

        pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])
        # mask = mask.reshape([mask.shape[0], -1])
        N = paddle.shape(mask)[0]
        mask = paddle.reshape(mask, [N, -1])

        masks = paddle.zeros(mask.shape)
        masks[mask] = float('-inf')# bool转为inf
        mask = masks
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed.astype("float16"))
        else:
            memory = src

        query_embed = self.embedding.position_embeddings.weight.unsqueeze(0)
        # query_embed = paddle.concat([query_embed for _ in range(bs)], axis=1)
        if self.training:
            train_tgt = self.embedding(seq)
            train_hs = self.decoder(train_tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed[:, :train_tgt.shape[1], :],
                          tgt_mask=generate_square_subsequent_mask(train_tgt.shape[1]))
            return vocab_embed(train_hs)
        else:
            probs = []
            for i in range(self.max_position_embeddings):
                eval_tgt = self.embedding(seq)

                eval_hs = self.decoder(eval_tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed[:, :i+1, :],
                          tgt_mask=generate_square_subsequent_mask(i+1))

                out = vocab_embed(eval_hs[:, -1, :])
                out = F.softmax(out)

                # bins chars eos sos padding
                if i % 27 == 0: # coordinate or eos
                    out[:, self.num_bins:self.eos_index] = 0
                    out[:, self.eos_index+1:] = 0
                elif i % 27 == 1: # coordinate
                    out = out[:, :self.num_bins]
                else: # chars
                    out[:, :self.num_bins] = 0
                    out[:, self.eos_index:] = 0

                prob, extra_seq = out.topk(axis=-1, k=1)
                seq = paddle.concat([seq.astype("int64"), extra_seq.astype("int64")], axis=1)
                probs.append(prob)
                if extra_seq[0] == self.eos_index:
                    break

            seq = seq[:, 1:] # remove start index
            return seq, paddle.concat(probs, axis=-1)


class DecoderEmbeddings(nn.Layer):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super(DecoderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id, weight_attr=weight_attr)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim, weight_attr=weight_attr)

        self.LayerNorm = paddle.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.shape
        seq_length = input_shape[1]

        position_ids = paddle.arange(seq_length)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x.astype("int64"))
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TransformerEncoder(nn.Layer):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Layer):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model        
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr=weight_attr, bias_attr=bias_attr)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        # bs, km_len = src_key_padding_mask.shape
        # km2attn_mask = paddle.expand(src_key_padding_mask, 
        #                              [self.self_attn.num_heads, bs, km_len])
        tmp_src_key_padding_mask = paddle.unsqueeze(paddle.unsqueeze(src_key_padding_mask, axis=1), axis=1)
        src2 = self.self_attn(query=q, key=k, value=src2, attn_mask=tmp_src_key_padding_mask)

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr=weight_attr, bias_attr=bias_attr)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        query=self.with_pos_embed(tgt2, query_pos)
        key=self.with_pos_embed(memory, pos)
        value=memory
        attn_mask=memory_key_padding_mask

        tmp_attn_mask = paddle.unsqueeze(paddle.unsqueeze(attn_mask, axis=1), axis=1)
        tgt2 = self.multihead_attn(query=query, key=key, value=value, attn_mask=tmp_attn_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class Position(nn.Layer):
    def __init__(self, config):
        super(Position, self).__init__()
        self.position_embedding = build_position_embedding(config)

    def forward(self, data):
        src, mask = data[0], data[2]
        # src, mask = data['image'], data['mask']
        pos, mask = self.position_embedding(src, mask)
        return src, mask, pos


class MLP(nn.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(nn.Linear(n, k) for n, k in zip(
            [input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Layer):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, feature, mask):
        # mask = F.interpolate(mask[None].astype("float32"), \
        #     size=feature.shape[-2:]).astype(bool)[0]

        tmp_mask = mask[None].astype("float32")
        tmp_feature_shape = feature.shape[-2:]
        # if len(tmp_mask.shape) != 4:
        #     tmp_mask = paddle.randn([1, 1, 672, 1194])
        # if len(tmp_feature_shape) != 2:
        #     tmp_feature_shape = [25, 45]
        mask = F.interpolate(tmp_mask, size=tmp_feature_shape).astype(bool)[0]
        # mask = paddle.zeros(tmp_feature_shape).unsqueeze(0).astype(bool)

        not_mask = ~mask
        y_embed = not_mask.astype("float32").cumsum(1)
        x_embed = not_mask.astype("float32").cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(self.num_pos_feats, dtype=paddle.float32)
        dim_t = self.temperature ** (2 * (dim_t / 2).floor() / 
                                     self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.stack((pos_x[:, :, :, 0::2].sin(), 
                              pos_x[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos_y = paddle.stack((pos_y[:, :, :, 0::2].sin(), 
                              pos_y[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
        return pos, mask


class SPTS(nn.Layer):
    def __init__(self, **kwargs):
        super(SPTS, self).__init__()
        self.backbone = ResNet(BottleneckBlock, depth=50, with_pool=False, num_classes=-1)
        self.input_proj = nn.Conv2D(self.backbone.inplanes,
                                    kwargs['Position']['tfm_hidden_dim'],
                                    kernel_size=1)
        self.position = Position(kwargs['Position'])
        self.init_param(kwargs)

        # paddle.set_default_dtype("float64")
        self.transformer = Transformer(d_model=self.d_model,
                        nhead=self.nhead,
                        num_encoder_layers=self.num_encoder_layers,
                        num_decoder_layers=self.num_decoder_layers,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.normalize_before,
                        pad_token_id=self.padding_index,
                        num_classes=self.num_classes,
                        max_position_embeddings=self.max_position_embeddings,
                        return_intermediate_dec=False,
                        num_bins=self.num_bins,
                        eos_index=self.eos_index)
        self.vocab_embed = MLP(self.d_model, self.d_model, self.num_classes, 3)
        self.out_channels = kwargs['in_channels']

    def init_param(self, config):
        self.num_bins = config['Transformer']['num_bins']
        self.max_num_text_ins = config['Transformer']['max_num_text_ins']
        self.chars = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
        num_char_classes = len(self.chars) + 1 # unknown
        recog_pad_index = self.num_bins + num_char_classes
        self.eos_index = recog_pad_index + 1
        self.sos_index = self.eos_index + 1
        self.padding_index = self.sos_index + 1
        self.num_classes = self.padding_index + 1
        self.max_position_embeddings = (2 + 25) * self.max_num_text_ins + 1

        self.d_model = config['Transformer']['tfm_hidden_dim']
        self.nhead = config['Transformer']['nhead']
        self.num_encoder_layers = config['Transformer']['num_encoder_layers']
        self.num_decoder_layers = config['Transformer']['num_decoder_layers']
        self.dim_feedforward = config['Transformer']['dim_feedforward']
        self.dropout = config['Transformer']['dropout']
        self.normalize_before = config['Transformer']['normalize_before']
        self.return_intermediate_dec = config['Transformer']['return_intermediate_dec']

    # @paddle.jit.to_static
    def forward(self, data):
        if self.training:
            image = data[0]
            data[0] = paddle.concat([i for i in image], axis=0)
            sequence = data[1]
            data[1] = paddle.concat([i for i in sequence], axis=0)
            mask = data[2]
            data[2] = paddle.concat([i for i in mask], axis=0)
        else:
            data[0], data[1], data[2] = data[0][0], data[1][0], data[2][0]

        img, seq = data[0], data[1]
        outputs = self.backbone(img)

        outputs = self.input_proj(outputs)

        data[0] = outputs
        src, mask, pos = self.position(data)
        if self.training:
            # sequence = seq[:, 0, :].astype("int")
            sequence = seq[:, 0, :].astype("int")
        else:
            sequence = seq

        outputs = self.transformer(src,
                                    mask,
                                    pos,
                                    sequence,
                                    self.vocab_embed)
        return outputs
