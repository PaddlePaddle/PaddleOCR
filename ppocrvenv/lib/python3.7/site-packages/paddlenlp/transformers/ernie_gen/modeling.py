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
import os
import io
import copy
import logging
import six
import json

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddlenlp.utils.env import MODEL_HOME
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.log import logger
from paddlenlp.transformers import BertPretrainedModel, ElectraPretrainedModel, RobertaPretrainedModel, ErniePretrainedModel

from ..utils import InitTrackerMeta, fn_args_to_dict

__all__ = ["ErnieGenPretrainedModel", "ErnieForGeneration", "ErnieGenModel"]


def _build_linear(n_in, n_out, name, init):
    return nn.Linear(
        n_in,
        n_out,
        weight_attr=paddle.ParamAttr(
            name='%s.w_0' % name if name is not None else None,
            initializer=init),
        bias_attr='%s.b_0' % name if name is not None else None, )


def _build_ln(n_in, name):
    return nn.LayerNorm(
        normalized_shape=n_in,
        weight_attr=paddle.ParamAttr(
            name='%s_layer_norm_scale' % name if name is not None else None,
            initializer=nn.initializer.Constant(1.)),
        bias_attr=paddle.ParamAttr(
            name='%s_layer_norm_bias' % name if name is not None else None,
            initializer=nn.initializer.Constant(1.)), )


def append_name(name, postfix):
    if name is None:
        ret = None
    elif name == '':
        ret = postfix
    else:
        ret = '%s_%s' % (name, postfix)
    return ret


class AttentionLayer(nn.Layer):
    def __init__(self, cfg, name=None):
        super(AttentionLayer, self).__init__()
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        n_head = cfg['num_attention_heads']
        assert d_model % n_head == 0
        d_model_q = cfg.get('query_hidden_size_per_head',
                            d_model // n_head) * n_head
        d_model_v = cfg.get('value_hidden_size_per_head',
                            d_model // n_head) * n_head
        self.n_head = n_head
        self.d_key = d_model_q // n_head
        self.q = _build_linear(d_model, d_model_q,
                               append_name(name, 'query_fc'), initializer)
        self.k = _build_linear(d_model, d_model_q,
                               append_name(name, 'key_fc'), initializer)
        self.v = _build_linear(d_model, d_model_v,
                               append_name(name, 'value_fc'), initializer)
        self.o = _build_linear(d_model_v, d_model,
                               append_name(name, 'output_fc'), initializer)
        self.dropout = nn.Dropout(p=cfg['attention_probs_dropout_prob'])

    def forward(self, queries, keys, values, attn_bias, past_cache):
        assert len(queries.shape) == len(keys.shape) == len(values.shape) == 3
        #bsz, q_len, q_dim = queries.shape
        #bsz, k_len, k_dim = keys.shape
        #bsz, v_len, v_dim = values.shape
        #assert k_len == v_len

        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        cache = (k, v)
        if past_cache is not None:
            cached_k, cached_v = past_cache
            k = paddle.concat([cached_k, k], 1)
            v = paddle.concat([cached_v, v], 1)

        q = q.reshape(
            [0, 0, self.n_head, q.shape[-1] // self.n_head]).transpose(
                [0, 2, 1, 3])  #[batch, head, seq, dim]
        k = k.reshape(
            [0, 0, self.n_head, k.shape[-1] // self.n_head]).transpose(
                [0, 2, 1, 3])  #[batch, head, seq, dim]
        v = v.reshape(
            [0, 0, self.n_head, v.shape[-1] // self.n_head]).transpose(
                [0, 2, 1, 3])  #[batch, head, seq, dim]

        q = q.scale(self.d_key**-0.5)
        score = q.matmul(k, transpose_y=True)
        if attn_bias is not None:
            score += attn_bias
        score = F.softmax(score)
        score = self.dropout(score)

        out = score.matmul(v).transpose([0, 2, 1, 3])
        out = out.reshape([0, 0, out.shape[2] * out.shape[3]])
        out = self.o(out)
        return out, cache


class PositionwiseFeedForwardLayer(nn.Layer):
    def __init__(self, cfg, name=None):
        super(PositionwiseFeedForwardLayer, self).__init__()
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)
        self.act = getattr(paddle.nn.functional, cfg['hidden_act'])
        self.i = _build_linear(
            d_model,
            d_ffn,
            append_name(name, 'fc_0'),
            initializer, )
        self.o = _build_linear(d_ffn, d_model,
                               append_name(name, 'fc_1'), initializer)
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs):
        hidden = self.act(self.i(inputs))
        hidden = self.dropout(hidden)
        out = self.o(hidden)
        return out


class ErnieEncoderLayer(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieEncoderLayer, self).__init__()
        d_model = cfg['hidden_size']
        self.attn = AttentionLayer(
            cfg, name=append_name(name, 'multi_head_att'))
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'))
        self.ffn = PositionwiseFeedForwardLayer(
            cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'))
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs, attn_bias=None, past_cache=None):
        attn_out, cache = self.attn(
            inputs, inputs, inputs, attn_bias,
            past_cache=past_cache)  #self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm

        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden, cache


class ErnieEncoderStack(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieEncoderStack, self).__init__()
        n_layers = cfg['num_hidden_layers']
        self.block = nn.LayerList([
            ErnieEncoderLayer(cfg, append_name(name, 'layer_%d' % i))
            for i in range(n_layers)
        ])

    def forward(self, inputs, attn_bias=None, past_cache=None):
        if past_cache is not None:
            assert isinstance(
                past_cache, tuple
            ), 'unknown type of `past_cache`, expect tuple or list. got %s' % repr(
                type(past_cache))
            past_cache = list(zip(*past_cache))
        else:
            past_cache = [None] * len(self.block)
        cache_list_k, cache_list_v, hidden_list = [], [], [inputs]

        for b, p in zip(self.block, past_cache):
            inputs, cache = b(inputs, attn_bias=attn_bias, past_cache=p)
            cache_k, cache_v = cache
            cache_list_k.append(cache_k)
            cache_list_v.append(cache_v)
            hidden_list.append(inputs)

        return inputs, hidden_list, (cache_list_k, cache_list_v)


@six.add_metaclass(InitTrackerMeta)
class ErnieGenPretrainedModel(object):
    r"""
    An abstract class for pretrained ErnieGen models. It provides ErnieGen related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """
    model_config_file = "model_config.json"
    ernie_gen_pretrained_init_configuration = {
        "ernie-gen-base-en": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 1024,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "ernie-gen-large-en": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "ernie-gen-large-en-430g": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    ernie_gen_pretrained_resource_files_map = {
        "model_state": {
            "ernie-gen-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-gen-base/ernie_gen_base.pdparams",
            "ernie-gen-large-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-gen-large/ernie_gen_large.pdparams",
            "ernie-gen-large-430g-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-gen-large-430g/ernie_gen_large_430g.pdparams",
        }
    }

    # Support more model to warm start.
    pretrained_init_configuration = {
        ** ernie_gen_pretrained_init_configuration, **
        BertPretrainedModel.pretrained_init_configuration, **
        ElectraPretrainedModel.pretrained_init_configuration, **
        RobertaPretrainedModel.pretrained_init_configuration, **
        ErniePretrainedModel.pretrained_init_configuration
    }
    pretrained_resource_files_map = {
        "model_state": {
            ** ernie_gen_pretrained_resource_files_map["model_state"], **
            BertPretrainedModel.pretrained_resource_files_map["model_state"], **
            ElectraPretrainedModel.pretrained_resource_files_map["model_state"],
            **
            RobertaPretrainedModel.pretrained_resource_files_map["model_state"],
            ** ErniePretrainedModel.pretrained_resource_files_map["model_state"]
        }
    }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        resource_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                resource_files[file_id] = map_list[
                    pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(
                cls.pretrained_init_configuration[
                    pretrained_model_name_or_path])
        else:
            if os.path.isdir(pretrained_model_name_or_path):
                for file_id, file_name in cls.resource_files_names.items():
                    full_file_name = os.path.join(pretrained_model_name_or_path,
                                                  file_name)
                    resource_files[file_id] = full_file_name
                resource_files["model_config_file"] = os.path.join(
                    pretrained_model_name_or_path, cls.model_config_file)
            else:
                raise ValueError(
                    "Calling {}.from_pretrained() with a model identifier or the "
                    "path to a directory instead. The supported model "
                    "identifiers are as follows: {}".format(
                        cls.__name__, cls.pretrained_init_configuration.keys()))

        default_root = os.path.join(MODEL_HOME, pretrained_model_name_or_path)
        resolved_resource_files = {}
        for file_id, file_path in resource_files.items():
            path = os.path.join(default_root, file_path.split('/')[-1])
            if file_path is None or os.path.isfile(file_path):
                resolved_resource_files[file_id] = file_path
            elif os.path.exists(path):
                logger.info("Already cached %s" % path)
                resolved_resource_files[file_id] = path
            else:
                logger.info("Downloading %s and saved to %s" %
                            (file_path, default_root))
                resolved_resource_files[file_id] = get_path_from_url(
                    file_path, default_root)

        # Prepare model initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        model_config_file = resolved_resource_files.pop("model_config_file",
                                                        None)
        if model_config_file is not None:
            with io.open(model_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration

        # import pdb; pdb.set_trace()
        if not os.path.exists(resolved_resource_files[file_id]):
            raise ValueError('pretrain dir not found: %s' %
                             resolved_resource_files[file_id])

        name_prefix = kwargs.pop('name', None)
        model = cls(init_kwargs, name=name_prefix)

        weight_path = list(resolved_resource_files.values())[0]
        logger.info('loading pretrained model from %s' % weight_path)

        if os.path.exists(weight_path):
            m = paddle.load(weight_path)
            params_name = list(m.keys())
            if 'mlm.weight' not in params_name:
                # ernie_gen is not implemented with paddle.transformer.
                # So, when loading the params saved by paddle.transformer, we should convert the params name.
                # We will update ernie_gen with paddle.transformer in the future.
                name_index_begin = params_name[0].index('.') + 1
                for old_name in params_name:
                    new_name = old_name[name_index_begin:].replace("embeddings.word_embeddings","word_emb").replace("embeddings.position_embeddings","pos_emb")\
                        .replace("embeddings.token_type_embeddings","sent_emb").replace("embeddings.layer_norm","ln").replace("encoder.layers","encoder_stack.block")\
                            .replace("self_attn","attn").replace("k_proj","k").replace("q_proj","q").replace("v_proj","v").replace("out_proj","o")\
                                .replace("linear1","ffn.i").replace("linear2","ffn.o").replace("norm1","ln1").replace("norm2","ln2").replace("pooler.dense","pooler")
                    m[new_name] = m.pop(old_name)
            for k, v in model.state_dict().items():
                if k not in m:
                    logger.info('param:%s not set in pretrained model, skip' %
                                k)
                    m[k] = v  # FIXME: no need to do this in the future
            model.set_state_dict(m)
        else:
            raise ValueError('weight file not found in pretrain dir: %s' %
                             weight_path)
        return model

    def save_pretrained(self, save_directory):
        """
        Save model configuration and related resources (model state) to files
        under `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving directory ({}) should be a directory".format(save_directory)
        # save model config
        model_config_file = os.path.join(save_directory, self.model_config_file)
        model_config = self.init_config
        # If init_config contains a Layer, use the layer's init_config to save
        for key, value in model_config.items():
            if key == "init_args":
                args = []
                for arg in value:
                    args.append(
                        arg.init_config
                        if isinstance(arg, ErnieGenPretrainedModel) else arg)
                model_config[key] = tuple(args)
            elif isinstance(value, ErnieGenPretrainedModel):
                model_config[key] = value.init_config
        with io.open(model_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_config, ensure_ascii=False))
        # save model
        file_name = os.path.join(save_directory,
                                 list(self.resource_files_names.values())[0])
        paddle.save(self.state_dict(), file_name)

    def _wrap_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the prtrained model instance.
        """
        init_dict = fn_args_to_dict(original_init, *args, **kwargs)
        self.config = init_dict


class ErnieModel(nn.Layer, ErnieGenPretrainedModel):
    def __init__(self, cfg, name=None):
        """
        Fundamental pretrained Ernie model
        """
        logger.debug('init ErnieModel with config: %s' % repr(cfg))
        nn.Layer.__init__(self)
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
        d_sent = cfg.get("sent_type_vocab_size") or cfg['type_vocab_size']
        self.n_head = cfg['num_attention_heads']
        self.return_additional_info = cfg.get('return_additional_info', False)
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])

        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'))
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=paddle.ParamAttr(
                name=append_name(name, 'word_embedding'),
                initializer=initializer))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=paddle.ParamAttr(
                name=append_name(name, 'pos_embedding'),
                initializer=initializer))
        self.sent_emb = nn.Embedding(
            d_sent,
            d_emb,
            weight_attr=paddle.ParamAttr(
                name=append_name(name, 'sent_embedding'),
                initializer=initializer))
        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)

        self.encoder_stack = ErnieEncoderStack(cfg,
                                               append_name(name, 'encoder'))

    def forward(self,
                src_ids,
                sent_ids=None,
                pos_ids=None,
                input_mask=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
        """
        Args:
            src_ids (Tensor):
                Indices of input sequence tokens in the vocabulary.
                They are numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            sent_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            pos_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `[batch_size, num_tokens]` and dtype as int64. Defaults to `None`.
            input_mask(Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                We use whole-word-mask in ERNIE, so the whole word will have the same value. For example, "使用" as a word,
                "使" and "用" will have the same value.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            attn_bias(Tensor, optional):
                3D version of `input_mask`, if set, overrides `input_mask`;
                if set not False, attention mask willed not be applied.
            past_cache(Tensor, optional, tuple of two lists: cached key and cached value,
                Each is a list of `Variable`s of shape `[batch_size, seq_len, hidden_size]`:
                cached key/value tensor that will be concated to generated key/value when performing self attention.
                if set, `attn_bias` should not be None.

        Returns:
            tuple: Returns tuple (`encoded`, `additional_info`).

            With the fields:

            - `encoded`(Tensor):
                The output logits of transformer stack.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `additional_info` (dict):
                Additional middle level info, inclues all hidden stats and k/v caches.
        """
        assert len(
            src_ids.
            shape) == 2, 'expect src_ids.shape = [batch, sequecen], got %s' % (
                repr(src_ids.shape))
        assert attn_bias is not None if past_cache else True, 'if `past_cache` is specified; attn_bias should not be None'
        d_seqlen = paddle.shape(src_ids)[1]
        if pos_ids is None:
            pos_ids = paddle.arange(
                0, d_seqlen, 1, dtype='int32').reshape([1, -1]).cast('int64')
        if attn_bias is None:
            if input_mask is None:
                input_mask = paddle.cast(src_ids != 0, 'float32')
            assert len(input_mask.shape) == 2
            input_mask = input_mask.unsqueeze(-1)
            attn_bias = input_mask.matmul(input_mask, transpose_y=True)
            if use_causal_mask:
                sequence = paddle.reshape(
                    paddle.arange(
                        0, d_seqlen, 1, dtype='float32') + 1., [1, 1, -1, 1])
                causal_mask = (sequence.matmul(
                    1. / sequence, transpose_y=True) >= 1.).cast('float32')
                attn_bias *= causal_mask
        else:
            assert len(
                attn_bias.shape
            ) == 3, 'expect attn_bias tobe rank 3, got %r' % attn_bias.shape
        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = attn_bias.unsqueeze(1).tile(
            [1, self.n_head, 1, 1])  # avoid broadcast =_=

        if sent_ids is None:
            sent_ids = paddle.zeros_like(src_ids)

        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
        sent_embedded = self.sent_emb(sent_ids)
        embedded = src_embedded + pos_embedded + sent_embedded

        embedded = self.dropout(self.ln(embedded))

        encoded, hidden_list, cache_list = self.encoder_stack(
            embedded, attn_bias, past_cache=past_cache)

        additional_info = {
            'hiddens': hidden_list,
            'caches': cache_list,
        }

        return encoded, additional_info


class ErnieForGeneration(ErnieModel):
    """
    Ernie Model for sequence to sequence generation.

    This model inherits from :class:`~paddlenlp.transformers.ernie.modeling.ErnieModel`.
    Refer to the superclass documentation for the generic methods.

    """

    def __init__(self, cfg, name=None):
        super(ErnieForGeneration, self).__init__(cfg, name=name)
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_vocab = cfg['vocab_size']

        self.mlm = _build_linear(
            d_model,
            d_model,
            append_name(name, 'mask_lm_trans_fc'),
            initializer, )
        self.act = getattr(paddle.nn.functional, cfg['hidden_act'])
        self.mlm_ln = _build_ln(
            d_model, name=append_name(name, 'mask_lm_trans'))
        self.mlm_bias = paddle.create_parameter(
            dtype='float32',
            shape=[d_vocab],
            attr=paddle.ParamAttr(
                name=append_name(name, 'mask_lm_out_fc.b_0'),
                initializer=nn.initializer.Constant(value=0.0)),
            is_bias=True, )

    def forward(self, *args, **kwargs):
        """
        Args:
            tgt_labels(Tensor, optional):
                The ground truth target sequence id (hard label) or distribution (soft label).
                It's data type should be `int64` and has a shape of [batch_size, sequence_length] or
                [batch_size, sequence_length, sequence_length].
            tgt_pos(Tensor, optional):
                Index of tgt_labels in `src_ids`.
                It's data type should be `int64` and has a shape of [n_targets, 2]).
            encode_only(bool, optional):
                Whether the model will output the logits or only encode the inputs.
                If `encode_only` is `True`, `loss` and `logits_2d` will not be returned.

        Returns:
            tuple: Returns tuple (`None`, `None`, `info`) if `encode_only` is `True`,
            returns (`output_ids`, `logits`, `info`) if `tgt_labels` or `tgt_pos` is `None`,
            else, returns (`loss`, `logits_2d`, `info`).

            With the fields:

            - `info`(dict):
                 Middle level info, includes all hidden stats and k/v caches.

            - `output_ids`(Tensor):
                The output index. Its data type should be float32 and its shape is [batch_size].
                If `encode_only`, returns None.

            - `logits`(Tensor):
                Logits for every targets.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
                If `encode_only`, returns None.

            - `loss`(Tensor):
                Cross entropy loss mean over every target label.
                If `encode_only`, returns None.

            - `logits_2d`(Tensor):
                Logits for every targets if `tgt_labels` or `tgt_pos` is not `None` .
                Its data type should be float32 and its shape is [batch_size, sequence_length].

        """
        tgt_labels = kwargs.pop('tgt_labels', None)
        tgt_pos = kwargs.pop('tgt_pos', None)
        encode_only = kwargs.pop('encode_only', False)
        encoded, info = ErnieModel.forward(self, *args, **kwargs)
        if encode_only:
            return None, None, info
        if tgt_labels is None or tgt_pos is None:
            encoded = self.act(self.mlm(encoded))
            encoded = self.mlm_ln(encoded)
            logits = encoded.matmul(
                self.word_emb.weight, transpose_y=True) + self.mlm_bias
            output_ids = logits.argmax(-1)
            return output_ids, logits, info
        else:
            encoded_2d = encoded.gather_nd(tgt_pos)
            encoded_2d = self.act(self.mlm(encoded_2d))
            encoded_2d = self.mlm_ln(encoded_2d)
            logits_2d = encoded_2d.matmul(
                self.word_emb.weight, transpose_y=True) + self.mlm_bias
            if len(tgt_labels.shape) == 1:
                tgt_labels = paddle.reshape(tgt_labels, [-1, 1])

            loss = F.cross_entropy(
                logits_2d,
                tgt_labels,
                reduction="none",
                soft_label=(tgt_labels.shape[-1] != 1))

            return loss, logits_2d, info


ErnieGenModel = ErnieForGeneration
