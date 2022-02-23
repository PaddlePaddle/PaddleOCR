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
import os
import shutil
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import (TransformerModel, WordEmbedding,
                                    PositionalEmbedding, position_encoding_init,
                                    InferTransformerModel, GPTModel)
from paddlenlp.ops import (InferTransformerDecoding, InferGptDecoding,
                           InferUnifiedDecoding, InferBartDecoding,
                           InferMBartDecoding)

from .encoder import enable_faster_encoder, disable_faster_encoder
from paddlenlp.ops.ext_utils import load
from paddlenlp.utils.log import logger
from paddlenlp.transformers import (GPTChineseTokenizer, GPTTokenizer,
                                    UnifiedTransformerPretrainedModel,
                                    UNIMOPretrainedModel, BartPretrainedModel,
                                    GPTPretrainedModel, MBartPretrainedModel)


class FasterTransformer(TransformerModel):
    """
    FasterTransformer is a faster version for generation with the Transformer
    model. It uses a custom op based on and enhancing NV FasterTransformer to
    do fast generation.

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
        decoding_strategy (str, optional):
            Indicating the strategy of decoding. It can be 'beam_search', 'beam_search_v2',
            'topk_sampling' and 'topp_sampling'. For beam search strategies,
            'v2' would select the top `beam_size * 2` beams and process the top
            `beam_size` alive and finish beams in them separately, while 'v1'
            would only select the top `beam_size` beams and mix up the alive and
            finish beams. 'v2' always searchs more and get better results, since
            the alive beams would always be `beam_size` while the number of alive
            beams in `v1` might decrease when meeting the end token. However,
            'v2' always generates longer results thus might do more calculation
            and be slower.
        beam_size (int, optional):
            The beam width for beam search. Defaults to 4. 
        topk (int, optional):
            The number of highest probability tokens to keep for top-k sampling.
            Defaults to 4. 
        topp (float, optional):
            The most probable tokens whose cumulative probability is not less than
            `topp` are kept for top-p sampling. Defaults to 4. 
        max_out_len (int, optional):
            The maximum output length. Defaults to 256.
        diversity_rate (float, optional):
            Refer to `A Simple, Fast Diverse Decoding Algorithm for Neural Generation <https://arxiv.org/abs/1611.08562>`_
            for details. Bigger `diversity_rate` would lead to more diversity.
            if `diversity_rate == 0` is equivalent to naive BeamSearch. Default
            to 0 if not set.
        use_fp16_decoding(bool, optional):
            Whether to use fp16 for decoding. 
        enable_faster_encoder(bool, optional):
            Whether to use the faster version of encoder. This is experimental option for now.
            Defaults to False.
        use_fp16_encoder(bool, optional):
            Whether to use fp16 for encoder. Only works when enable_faster_encoder is True.
            Defaults to False.
        rel_len(bool, optional):
            Indicating whether `max_out_len` in is the length relative to that
            of source text. Only works in `v2` temporarily. It is suggest to set
            a small `max_out_len` and use `rel_len=True`. Default to False if
            not set.
        alpha(float, optional):
            The power number in length penalty calculation. Only works in `v2`
            temporarily. Refer to `GNMT <https://arxiv.org/pdf/1609.08144.pdf>`_.
            Default to 0.6 if not set.
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
                 decoding_strategy="beam_search",
                 beam_size=4,
                 topk=1,
                 topp=0.0,
                 max_out_len=256,
                 diversity_rate=0.0,
                 decoding_lib=None,
                 use_fp16_decoding=False,
                 enable_faster_encoder=False,
                 use_fp16_encoder=False,
                 rel_len=False,
                 alpha=0.6):
        # if decoding_lib is None:
        #     raise ValueError(
        #         "The args decoding_lib must be set to use FasterTransformer. ")
        # elif not os.path.exists(decoding_lib):
        #     raise ValueError("The path to decoding lib is not exist.")

        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.decoding_strategy = args.pop("decoding_strategy")
        self.beam_size = args.pop("beam_size")
        self.topk = args.pop("topk")
        self.topp = args.pop("topp")
        self.max_out_len = args.pop("max_out_len")
        self.diversity_rate = args.pop("diversity_rate")
        self.decoding_lib = args.pop("decoding_lib")
        self.use_fp16_decoding = args.pop("use_fp16_decoding")
        self.enable_faster_encoder = args.pop("enable_faster_encoder")
        self.use_fp16_encoder = args.pop("use_fp16_encoder")
        self.rel_len = args.pop("rel_len")
        self.alpha = args.pop("alpha")
        self.dropout = dropout
        self.weight_sharing = weight_sharing
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.bos_id = bos_id
        self.max_length = max_length
        super(FasterTransformer, self).__init__(**args)

        if self.enable_faster_encoder:
            logger.warning(
                "enable_faster_encoder is an experimental option and subject to change."
            )
        elif self.use_fp16_encoder:
            self.use_fp16_encoder = False

        self.decoding_linear = nn.Linear(
            in_features=d_model, out_features=trg_vocab_size)

        if weight_sharing:
            self.trg_word_embedding = WordEmbedding(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(
                emb_dim=d_model, max_length=max_length)

        self.decoding = InferTransformerDecoding(
            decoder=self.transformer.decoder,
            word_embedding=self.trg_word_embedding.word_embedding,
            positional_embedding=self.trg_pos_embedding.pos_encoder,
            linear=self.decoding_linear,
            num_decoder_layers=num_decoder_layers,
            n_head=n_head,
            d_model=d_model,
            bos_id=bos_id,
            eos_id=eos_id,
            decoding_strategy=decoding_strategy,
            beam_size=beam_size,
            topk=topk,
            topp=topp,
            max_out_len=max_out_len,
            diversity_rate=self.diversity_rate,
            decoding_lib=self.decoding_lib,
            use_fp16_decoding=self.use_fp16_decoding,
            rel_len=self.rel_len,
            alpha=self.alpha)

    def forward(self, src_word, trg_word=None):
        src_max_len = paddle.shape(src_word)[-1]
        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        src_pos = paddle.cast(
            src_word != self.bos_id, dtype=src_word.dtype) * paddle.arange(
                start=0, end=src_max_len)

        # Run encoder
        src_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout,
            training=False) if self.dropout else src_emb

        if self.enable_faster_encoder and self.use_fp16_encoder:
            enc_input = paddle.cast(enc_input, dtype="float16")

        enc_output = self.transformer.encoder(enc_input, src_slf_attn_bias)

        if self.use_fp16_decoding and enc_output.dtype != paddle.float16:
            enc_output = paddle.cast(enc_output, dtype="float16")
        elif not self.use_fp16_decoding and enc_output.dtype != paddle.float32:
            enc_output = paddle.cast(enc_output, dtype="float32")

        mem_seq_lens = paddle.sum(paddle.cast(
            src_word != self.bos_id, dtype="int32"),
                                  dtype="int32",
                                  axis=1)
        ids = self.decoding(enc_output, mem_seq_lens, trg_word=trg_word)

        return ids

    def load(self, init_from_params):
        # Load the trained model
        assert init_from_params, (
            "Please set init_from_params to load the infer model.")

        model_dict = paddle.load(init_from_params, return_numpy=True)

        # To set weight[padding_idx] to 0.
        model_dict["trg_word_embedding.word_embedding.weight"][
            self.bos_id] = [0] * self.d_model

        # Dealing with weight sharing.
        if self.weight_sharing:
            model_dict["decoding_linear.weight"] = np.transpose(model_dict[
                "trg_word_embedding.word_embedding.weight"])
        else:
            model_dict["decoding_linear.weight"] = model_dict["linear.weight"]

        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)
        model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)

        if self.decoding._fuse_qkv:
            for item in self.state_dict():
                if "decoder" in item and "self_attn.q_proj" in item:
                    num_layer = item.split(".")[3]
                    param_type = item.split(".")[-1]

                    model_dict["decoding.slf_q_" + param_type + "_" +
                               num_layer] = np.concatenate(
                                   (model_dict[item], model_dict[
                                       "transformer.decoder.layers." + num_layer
                                       + ".self_attn.k_proj." + param_type],
                                    model_dict["transformer.decoder.layers." +
                                               num_layer + ".self_attn.v_proj."
                                               + param_type]),
                                   axis=-1)

        if self.use_fp16_decoding:
            for item in self.state_dict():
                if "decoder" in item or "decoding.slf" in item:
                    model_dict[item] = np.float16(model_dict[item])
            model_dict["decoding_linear.weight"] = np.float16(model_dict[
                "decoding_linear.weight"])
            model_dict["trg_word_embedding.word_embedding.weight"] = np.float16(
                model_dict["trg_word_embedding.word_embedding.weight"])
            model_dict["trg_pos_embedding.pos_encoder.weight"] = np.float16(
                model_dict["trg_pos_embedding.pos_encoder.weight"])
            model_dict["decoding_linear.bias"] = np.zeros(
                [self.trg_vocab_size], dtype="float16")

        self.load_dict(model_dict)

    def export_params(self, init_from_params, place):
        '''
        This method is used for load static graph from dygraph checkpoint
        or export inference model using static graph. 

        Args:
            init_from_params (string):
                The path to dygraph checkpoint. 
            place (paddle.Place):
                The place to execute static graph. 
        
        Example:
            .. code-block::
                paddle.enable_static()
                place = "gpu"
                place = paddle.set_device(place)
                reader.adapt_vocab_size(args)

                test_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                with paddle.static.program_guard(test_program, startup_program):
                    src_word = paddle.static.data(
                        name="src_word", shape=[None, None], dtype="int64")

                    # Define model
                    transformer = FasterTransformer(
                        src_vocab_size=args.src_vocab_size,
                        trg_vocab_size=args.trg_vocab_size,
                        max_length=args.max_length + 1,
                        num_encoder_layers=args.n_layer,
                        num_decoder_layers=args.n_layer,
                        n_head=args.n_head,
                        d_model=args.d_model,
                        d_inner_hid=args.d_inner_hid,
                        dropout=args.dropout,
                        weight_sharing=args.weight_sharing,
                        bos_id=args.bos_idx,
                        eos_id=args.eos_idx,
                        decoding_strategy=args.decoding_strategy,
                        beam_size=args.beam_size,
                        max_out_len=args.max_out_len,
                        decoding_lib=args.decoding_lib,
                        use_fp16_decoding=args.use_fp16_decoding,
                        rel_len=args.use_rel_len,
                        alpha=args.alpha)

                    finished_seq = transformer(src_word=src_word)

                test_program = test_program.clone(for_test=True)

                exe = paddle.static.Executor(place)
                exe.run(startup_program)

                # Load checkpoint.
                transformer.export_params(
                    init_from_params=os.path.join(args.init_from_params,
                                                "transformer.pdparams"),
                    place=place)

                paddle.static.save_inference_model(
                    os.path.join(args.inference_model_dir, "transformer"),
                    feed_vars=src_word,
                    fetch_vars=finished_seq,
                    executor=exe,
                    program=test_program)
        '''
        # Load the trained model
        assert init_from_params, (
            "Please set init_from_params to load the infer model.")

        model_dict = paddle.load(init_from_params, return_numpy=True)

        # To set weight[padding_idx] to 0.
        model_dict["trg_word_embedding.word_embedding.weight"][
            self.bos_id] = [0] * self.d_model

        # Dealing with weight sharing.
        if self.weight_sharing:
            model_dict["decoding_linear.weight"] = np.transpose(model_dict[
                "trg_word_embedding.word_embedding.weight"])
        else:
            model_dict["decoding_linear.weight"] = model_dict["linear.weight"]

        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)
        model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
            self.max_length, self.d_model)

        if self.decoding._fuse_qkv:
            for item in self.state_dict():
                if "decoder" in item and "self_attn.q_proj" in item:
                    num_layer = item.split(".")[3]
                    param_type = item.split(".")[-1]

                    model_dict["decoding.slf_q_" + param_type + "_" +
                               num_layer] = np.concatenate(
                                   (model_dict[item], model_dict[
                                       "transformer.decoder.layers." + num_layer
                                       + ".self_attn.k_proj." + param_type],
                                    model_dict["transformer.decoder.layers." +
                                               num_layer + ".self_attn.v_proj."
                                               + param_type]),
                                   axis=-1)

        if self.use_fp16_decoding:
            for item in self.state_dict():
                if "decoder" in item or "decoding.slf" in item:
                    model_dict[item] = np.float16(model_dict[item])
            model_dict["decoding_linear.weight"] = np.float16(model_dict[
                "decoding_linear.weight"])
            model_dict["trg_word_embedding.word_embedding.weight"] = np.float16(
                model_dict["trg_word_embedding.word_embedding.weight"])
            model_dict["trg_pos_embedding.pos_encoder.weight"] = np.float16(
                model_dict["trg_pos_embedding.pos_encoder.weight"])
            model_dict["decoding_linear.bias"] = np.zeros(
                [self.trg_vocab_size], dtype="float16")

        for item in self.state_dict():
            param = self
            attr_list = item.split(".")
            for attr in attr_list:
                param = getattr(param, attr)
            param_name = param.name
            var = paddle.static.global_scope().find_var(param_name).get_tensor()
            var.set(model_dict[item], place)


class TransformerGenerator(paddle.nn.Layer):
    """
    The Transformer model for auto-regressive generation with beam search. It wraps
    `FasterTransformer` and `InferTransformerModel`, and automatically chioces using
    `FasterTransformer` (with jit building) or the slower verison `InferTransformerModel`.

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
        bos_id (int, optional):
            The start token id and also is used as padding id. Defaults to 0.
        eos_id (int, optional):
            The end token id. Defaults to 1.
        beam_size (int, optional):
            The beam width for beam search. Defaults to 4. 
        max_out_len (int, optional):
            The maximum output length. Defaults to 256.
        kwargs:
            The key word arguments can be `output_time_major`, `use_ft`, `use_fp16_decoding`,
            `rel_len`, `alpha`:

            - `output_time_major(bool, optional)`: Indicate the data layout of predicted
            Tensor. If `False`, the data layout would be batch major with shape
            `[batch_size, seq_len, beam_size]`. If  `True`, the data layout would
            be time major with shape `[seq_len, batch_size, beam_size]`. Default
            to `False`. 

            - `use_ft(bool, optional)`: Whether to use FasterTransformer
            for decoding. Default to True if not set.

            - `use_fp16_decoding(bool, optional)`: Whether to use fp16
            for decoding.  Only works when using FasterTransformer.

            - `beam_search_version(str, optional)`: Indicating the strategy of
            beam search. It can be 'v1' or 'v2'. 'v2' would select the top
            `beam_size * 2` beams and process the top `beam_size` alive and
            finish beams in them separately, while 'v1' would only select the
            top `beam_size` beams and mix up the alive and finish beams. 'v2' always
            searchs more and get better results, since the alive beams would
            always be `beam_size` while the number of alive beams in `v1` might
            decrease when meeting the end token. However, 'v2' always generates
            longer results thus might do more calculation and be slower.

            - `rel_len(bool, optional)`: Indicating whether `max_out_len` in is
            the length relative to that of source text. Only works in `v2` temporarily.
            It is suggest to set a small `max_out_len` and use `rel_len=True`.
            Default to False if not set.

            - `alpha(float, optional)`: The power number in length penalty
            calculation. Refer to `GNMT <https://arxiv.org/pdf/1609.08144.pdf>`_.
            Only works in `v2` temporarily. Default to 0.6 if not set.
        
            - diversity_rate(float, optional): Refer to `A Simple, Fast Diverse
            Decoding Algorithm for Neural Generation <https://arxiv.org/abs/1611.08562>`_
            for details. Bigger `diversity_rate` would lead to more diversity.
            if `diversity_rate == 0` is equivalent to naive BeamSearch. Default
            to 0 if not set. **NOTE**: Only works when using FasterTransformer
            temporarily.
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
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256,
                 **kwargs):
        logger.warning(
            "TransformerGenerator is an experimental API and subject to change.")
        # `kwargs` can include output_time_major, use_fp16_decoding, topk, topp.
        # The later three arguments can only work when using FasterTransformer,
        # and expose topk, topp later.
        super(TransformerGenerator, self).__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.output_time_major = kwargs.pop("output_time_major", True)
        # Only works for FasterTransformer.
        # TODO: original version supports diversity rate.
        diversity_rate = kwargs.pop("diversity_rate", 0.0)
        use_fp16_decoding = kwargs.pop("use_fp16_decoding", False)
        use_ft = kwargs.pop("use_ft", True)
        beam_search_version = kwargs.pop("beam_search_version", "v1")
        rel_len = kwargs.pop("rel_len", False)
        alpha = kwargs.pop("alpha", 0.6)

        if use_ft:
            try:
                decoding_strategy = ("beam_search_v2"
                                     if beam_search_version == "v2" else
                                     "beam_search")
                self.transformer = FasterTransformer(
                    src_vocab_size=src_vocab_size,
                    trg_vocab_size=trg_vocab_size,
                    max_length=max_length,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    n_head=n_head,
                    d_model=d_model,
                    d_inner_hid=d_inner_hid,
                    dropout=dropout,
                    weight_sharing=weight_sharing,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    beam_size=beam_size,
                    max_out_len=max_out_len,
                    diversity_rate=diversity_rate,
                    decoding_strategy=decoding_strategy,
                    use_fp16_decoding=use_fp16_decoding,
                    rel_len=rel_len,
                    alpha=alpha)
            except Exception:
                logger.warning(
                    "Exception occurs when using FasterTransformer. " \
                    "The original forward will be involved. ")
                if diversity_rate != 0:
                    logger.warning(
                        "diversity_rate would not work since it is only " \
                        "supported by FasterTransformer temporarily.")
                self.transformer = InferTransformerModel(
                    src_vocab_size=src_vocab_size,
                    trg_vocab_size=trg_vocab_size,
                    max_length=max_length,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    n_head=n_head,
                    d_model=d_model,
                    d_inner_hid=d_inner_hid,
                    dropout=dropout,
                    weight_sharing=weight_sharing,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    beam_size=beam_size,
                    max_out_len=max_out_len,
                    output_time_major=self.output_time_major,
                    beam_search_version=beam_search_version,
                    rel_len=rel_len,
                    alpha=alpha)
        else:
            if diversity_rate != 0:
                logger.warning(
                    "diversity_rate would not work since it is only " \
                    "supported by FasterTransformer temporarily.")
            self.transformer = InferTransformerModel(
                src_vocab_size=src_vocab_size,
                trg_vocab_size=trg_vocab_size,
                max_length=max_length,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                n_head=n_head,
                d_model=d_model,
                d_inner_hid=d_inner_hid,
                dropout=dropout,
                weight_sharing=weight_sharing,
                bos_id=bos_id,
                eos_id=eos_id,
                beam_size=beam_size,
                max_out_len=max_out_len,
                output_time_major=self.output_time_major,
                beam_search_version=beam_search_version,
                rel_len=rel_len,
                alpha=alpha)

    def forward(self, src_word, trg_word=None):
        r"""
        Performs decoding for transformer model.

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
                according to `output_time_major`. While, when using FasterTransformer
                and beam search v2, the beam dimension would be doubled to include
                both the top `beam_size` alive and finish beams, thus the tensor
                shape is `[batch_size, seq_len, beam_size * 2]` or `[seq_len, batch_size, beam_size * 2]`.
        
        Example:
            .. code-block::

                import paddle
                from paddlenlp.ops import TransformerGenerator

                transformer = TransformerGenerator(
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
        out = self.transformer(src_word, trg_word=trg_word)
        # TODO(guosheng): FasterTransformer has an output with layout
        # `[seq_len, batch_size, beam_size]`. While the output layout of
        # original one is `[batch_size, seq_len, beam_size]`. Maybe we need
        # unify them later.
        if not self.output_time_major and isinstance(self.transformer,
                                                     FasterTransformer):
            out = paddle.transpose(out, [1, 0, 2])
        return out

    def load(self, path):
        if isinstance(self.transformer, FasterTransformer):
            self.transformer.load(path)
        else:
            model_dict = paddle.load(path)
            self.transformer.load_dict(model_dict)


class FasterGPT(GPTPretrainedModel):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        super(FasterGPT, self).__init__()
        self._model = model
        self.use_fp16_decoding = use_fp16_decoding
        self.decoding = InferGptDecoding(
            model=model,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding)

    def forward(self,
                input_ids,
                seq_len=None,
                attention_mask=None,
                top_k=4,
                top_p=0.0,
                max_length=256,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                forced_eos_token_id=None,
                temperature=0,
                decode_strategy="sample",
                num_return_sequences=1,
                **model_kwargs):
        if input_ids.dtype == paddle.int64:
            input_ids = paddle.cast(input_ids, "int32")

        # change top_p to zero if not using top_p sampling for FT
        if decode_strategy == "greedy_search":
            top_p = 0.0
            top_k = 1
        if top_p == 1.0:
            top_p = 0.0
        if seq_len is None:
            seq_len = paddle.sum(paddle.cast(
                input_ids != pad_token_id, dtype="int32"),
                                 axis=-1,
                                 dtype="int32")

            if bos_token_id == pad_token_id and paddle.sum(
                    paddle.any(input_ids == pad_token_id), dtype="int64") > 0:
                seq_len = seq_len + 1

        if num_return_sequences > 1:
            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                seq_len=seq_len,
                attention_mask=attention_mask)
            seq_len = model_kwargs["seq_len"]
            attention_mask = model_kwargs.get("attention_mask", None)

        return self.decoding(
            input_ids,
            mem_seq_len=seq_len,
            attention_mask=attention_mask,
            topk=top_k,
            topp=top_p,
            max_out_len=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            forced_eos_token_id=forced_eos_token_id,
            temperature=temperature)

    def export_params(self, state_to_load, place):
        for item in state_to_load:
            param_data = np.array(state_to_load[item])
            if self.use_fp16_decoding:
                param_data = np.float16(param_data)

            param = self
            attr_list = item.split(".")
            attr_list = ["decoding", "model"] + attr_list
            for attr in attr_list:
                param = getattr(param, attr)
            param_name = param.name
            var = paddle.static.global_scope().find_var(param_name).get_tensor()
            var.set(param_data, place)

    def save_resources(self, tokenizer, path):
        vocab_file = os.path.join(path, "vocab.txt")
        if isinstance(tokenizer, GPTTokenizer):
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for token in tokenizer.encoder:
                    f.write(token + '\n')
            merges_file = os.path.join(path, "merges.txt")
            shutil.copyfile(tokenizer._merges_file, merges_file)
        elif isinstance(tokenizer, GPTChineseTokenizer):
            tokenizer.save_resources(path)

    generate = forward


class FasterUnifiedTransformer(UnifiedTransformerPretrainedModel):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        super(FasterUnifiedTransformer, self).__init__()
        self._model = model
        self.vocab_size = model.lm_head.decoder_bias.shape[0]
        self.unk_token_id = self._model.unk_token_id
        self.mask_token_id = self._model.mask_token_id
        self.bos_token_id = self._model.bos_token_id
        self.pad_token_id = self._model.pad_token_id
        self.logits_mask = self.generate_logits_mask(use_fp16_decoding)
        self._n_head = self._model.num_attention_heads
        self._hidden_dims = self._model.hidden_size
        self._normalize_before = self._model.normalize_before
        self._size_per_head = self._hidden_dims // self._n_head
        self._n_layer = self._model.num_hidden_layers
        self._hidden_act = self._model.hidden_act

        self.decoding = InferUnifiedDecoding(
            model=self._model,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding,
            logits_mask=self.logits_mask,
            n_head=self._n_head,
            hidden_dims=self._hidden_dims,
            size_per_head=self._size_per_head,
            n_layer=self._n_layer,
            unk_id=self.unk_token_id,
            mask_id=self.mask_token_id,
            normalize_before=self._normalize_before,
            hidden_act=self._hidden_act)

    def prepare_inputs_for_generation(self, input_ids, token_type_ids,
                                      position_ids, attention_mask, **kwargs):
        input_ids = input_ids[:, :-1]
        decoding_type_id = token_type_ids[:, -1]
        token_type_ids = token_type_ids[:, :-1]
        position_ids = position_ids[:, :-1]
        attention_mask = attention_mask[:, :, :-1, :-1]
        seq_len = kwargs.get("seq_len") - 1

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "seq_len": seq_len,
            "decoding_type_id": paddle.cast(
                decoding_type_id, dtype="int32")
        }

    def generate_logits_mask(self, use_fp16_decoding):
        # pre-process distribution
        logits_mask = np.zeros(shape=[self.vocab_size], dtype=np.float32)

        if use_fp16_decoding:
            logits_mask[self.unk_token_id] = -1e4
            logits_mask[self.bos_token_id] = -1e4
            logits_mask[self.pad_token_id] = -1e4
        else:
            logits_mask[self.unk_token_id] = -1e9
            logits_mask[self.bos_token_id] = -1e9
            logits_mask[self.pad_token_id] = -1e9

        logits_mask_t = paddle.assign(logits_mask)
        if use_fp16_decoding:
            return paddle.cast(logits_mask_t, dtype="float16")
        else:
            return logits_mask_t

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                seq_len=None,
                max_length=128,
                top_k=4,
                top_p=0.0,
                decode_strategy="sampling",
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                num_beams=4,
                diversity_rate=0.0,
                temperature=1.0,
                num_return_sequences=1,
                length_penalty=0.6,
                early_stopping=False,
                forced_eos_token_id=None,
                **model_kwargs):

        if seq_len is None:
            assert input_ids is not None, "You have to specify either input_ids when generating seq_len."
            seq_len = paddle.sum(paddle.cast(
                input_ids != self.pad_token_id, dtype="int32"),
                                 axis=-1,
                                 keepdim=True,
                                 dtype="int32")
        if decode_strategy.startswith("beam_search"):
            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                seq_len=seq_len)
        elif decode_strategy == "sampling":
            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                seq_len=seq_len)
        elif decode_strategy == "greedy_search":
            model_kwargs = {
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "seq_len": seq_len
            }
        else:
            raise ValueError(
                "Only greedy search, beam search and sampling are supported. ")

        model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                          **model_kwargs)

        seq_len = model_inputs.pop('seq_len')
        decoding_type_id = model_inputs.pop('decoding_type_id')

        outputs = self._model(**model_inputs)
        if isinstance(outputs, tuple):
            caches = outputs[1]
        else:
            raise RuntimeError('Not support.')

        cache_k = [c.k for c in caches]
        cache_v = [c.v for c in caches]

        return self.decoding(
            cache_k=cache_k,
            cache_v=cache_v,
            memory_seq_lens=seq_len,
            beam_size=num_beams,
            diversity_rate=diversity_rate,
            topk=top_k,
            topp=top_p,
            decoding_strategy=decode_strategy,
            max_out_len=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            length_penalty=length_penalty,
            decoding_type_id=decoding_type_id,
            pos_bias=True,
            forced_eos_token_id=forced_eos_token_id,
            early_stopping=early_stopping)

    generate = forward


class FasterUNIMOText(UNIMOPretrainedModel):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        super(FasterUNIMOText, self).__init__()
        self._model = model
        self.unk_token_id = self._model.unk_token_id
        self.mask_token_id = self._model.mask_token_id
        self.bos_token_id = self._model.bos_token_id
        self.pad_token_id = self._model.pad_token_id
        self.vocab_size = model.lm_head.decoder_bias.shape[0]

        self.logits_mask = self.generate_logits_mask(use_fp16_decoding)
        self._n_head = self._model.num_attention_heads
        self._hidden_dims = self._model.hidden_size
        self._normalize_before = self._model.normalize_before
        self._size_per_head = self._hidden_dims // self._n_head
        self._n_layer = self._model.num_hidden_layers
        self._hidden_act = self._model.hidden_act

        self.decoding = InferUnifiedDecoding(
            model=self._model,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding,
            logits_mask=self.logits_mask,
            n_head=self._n_head,
            hidden_dims=self._hidden_dims,
            size_per_head=self._size_per_head,
            n_layer=self._n_layer,
            unk_id=self.unk_token_id,
            mask_id=self.mask_token_id,
            normalize_before=self._normalize_before,
            hidden_act=self._hidden_act)

    def prepare_inputs_for_generation(self, input_ids, token_type_ids,
                                      position_ids, attention_mask, **kwargs):
        input_ids = input_ids[:, :-1]
        decoding_type_id = token_type_ids[:, -1]
        token_type_ids = token_type_ids[:, :-1]
        position_ids = position_ids[:, :-1]
        attention_mask = attention_mask[:, :, :-1, :-1]
        seq_len = kwargs.get("seq_len") - 1

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "seq_len": seq_len,
            "decoding_type_id": paddle.cast(
                decoding_type_id, dtype="int32")
        }

    def generate_logits_mask(self, use_fp16_decoding):
        # pre-process distribution
        logits_mask = np.zeros(shape=[self.vocab_size], dtype=np.float32)

        if use_fp16_decoding:
            logits_mask[self.unk_token_id] = -1e4
            logits_mask[self.bos_token_id] = -1e4
            logits_mask[self.pad_token_id] = -1e4
        else:
            logits_mask[self.unk_token_id] = -1e9
            logits_mask[self.bos_token_id] = -1e9
            logits_mask[self.pad_token_id] = -1e9

        logits_mask_t = paddle.assign(logits_mask)
        if use_fp16_decoding:
            return paddle.cast(logits_mask_t, dtype="float16")
        else:
            return logits_mask_t

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                seq_len=None,
                max_length=128,
                top_k=4,
                top_p=0.0,
                num_beams=4,
                decode_strategy="sampling",
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                diversity_rate=0.0,
                temperature=1.0,
                num_return_sequences=1,
                length_penalty=0.6,
                early_stopping=False,
                forced_eos_token_id=None,
                **model_kwargs):

        if seq_len is None:
            assert input_ids is not None, "You have to specify either input_ids when generating seq_len."
            seq_len = paddle.sum(paddle.cast(
                input_ids != self.pad_token_id, dtype="int32"),
                                 axis=-1,
                                 keepdim=True,
                                 dtype="int32")
        if decode_strategy.startswith("beam_search"):
            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                seq_len=seq_len)
        elif decode_strategy == "sampling":
            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                seq_len=seq_len)
        elif decode_strategy == "greedy_search":
            model_kwargs = {
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "seq_len": seq_len
            }
        else:
            raise ValueError(
                "Only greedy search, beam search and sampling are supported. ")

        model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                          **model_kwargs)
        seq_len = model_inputs.pop('seq_len')
        decoding_type_id = model_inputs.pop('decoding_type_id')

        outputs = self._model(**model_inputs)
        if isinstance(outputs, tuple):
            caches = outputs[1]
        else:
            raise RuntimeError('Not support.')

        cache_k = [c.k for c in caches]
        cache_v = [c.v for c in caches]

        return self.decoding(
            cache_k=cache_k,
            cache_v=cache_v,
            memory_seq_lens=seq_len,
            beam_size=num_beams,
            diversity_rate=diversity_rate,
            topk=top_k,
            topp=top_p,
            decoding_strategy=decode_strategy,
            max_out_len=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            length_penalty=length_penalty,
            decoding_type_id=decoding_type_id,
            forced_eos_token_id=forced_eos_token_id,
            pos_bias=False,
            early_stopping=early_stopping)

    generate = forward


class FasterBART(BartPretrainedModel):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        super(FasterBART, self).__init__()
        self.use_fp16_decoding = use_fp16_decoding
        self._model = model
        if use_fp16_decoding:
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Assign(
                model.bart.encoder.embed_tokens.weight))
            model.bart.encoder.embed_tokens = nn.Embedding(
                *model.bart.encoder.embed_tokens.weight.shape,
                weight_attr=weight_attr)
        self.encoder = model.bart.get_encoder()
        self.decoder = model.bart.get_decoder()
        self.pad_token_id = model.bart.config['pad_token_id']

        self.decoding = InferBartDecoding(
            model=self._model,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self,
                input_ids=None,
                encoder_output=None,
                seq_len=None,
                num_beams=4,
                top_k=1,
                top_p=0.0,
                decode_strategy="beam_search",
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                decoder_start_token_id=None,
                max_length=256,
                diversity_rate=0.0,
                length_penalty=0.6,
                num_return_sequences=1,
                early_stopping=False,
                forced_eos_token_id=None,
                **model_kwargs):

        if encoder_output is None:
            self.encoder = enable_faster_encoder(self.encoder)
            assert input_ids is not None, "You have to specify either input_ids or encoder_output."
            encoder_output = self.prepare_encoder_decoder_kwargs_for_generation(
                input_ids, model_kwargs)["encoder_output"]
            self.encoder = disable_faster_encoder(self.encoder)
        if seq_len is None:
            assert input_ids is not None, "You have to specify either input_ids when generating seq_len."
            seq_len = paddle.sum(paddle.cast(
                input_ids != self.pad_token_id, dtype="int32"),
                                 axis=-1,
                                 keepdim=True,
                                 dtype="int32")
        if self.use_fp16_decoding:
            encoder_output = paddle.cast(encoder_output, "float16")
        if decode_strategy.startswith("beam_search") and num_beams > 1:
            encoder_output, expanded_kwargs = self.expand_inputs_for_generation(
                encoder_output, expand_size=num_beams, seq_len=seq_len)
            seq_len = expanded_kwargs["seq_len"]
        elif decode_strategy == "sampling" and num_return_sequences > 1:
            encoder_output, expanded_kwargs = self.expand_inputs_for_generation(
                encoder_output,
                expand_size=num_return_sequences,
                seq_len=seq_len)
            seq_len = expanded_kwargs["seq_len"]
        if decoder_start_token_id is not None:
            bos_token_id = decoder_start_token_id
        return self.decoding(
            enc_output=encoder_output,
            memory_seq_lens=seq_len,
            beam_size=num_beams,
            top_k=top_k,
            decoding_strategy=decode_strategy,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            top_p=top_p,
            max_out_len=max_length,
            diversity_rate=diversity_rate,
            alpha=length_penalty,
            early_stopping=early_stopping,
            forced_eos_token_id=forced_eos_token_id)

    generate = forward


class FasterMBART(MBartPretrainedModel):
    def __init__(self, model, decoding_lib=None, use_fp16_decoding=False):
        super(FasterMBART, self).__init__()
        self.use_fp16_decoding = use_fp16_decoding
        self._model = model
        if use_fp16_decoding:
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Assign(
                model.mbart.encoder.embed_tokens.weight))
            model.mbart.encoder.embed_tokens = nn.Embedding(
                *model.mbart.encoder.embed_tokens.weight.shape,
                weight_attr=weight_attr)
        self.encoder = model.mbart.get_encoder()
        self.decoder = model.mbart.get_decoder()
        self.pad_token_id = model.mbart.config['pad_token_id']

        self.decoding = InferMBartDecoding(
            model=self._model,
            decoding_lib=decoding_lib,
            use_fp16_decoding=use_fp16_decoding,
            hidden_act=model.mbart.config['activation_function'])

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self,
                input_ids=None,
                encoder_output=None,
                seq_len=None,
                forced_bos_token_id=None,
                num_beams=4,
                top_k=1,
                top_p=0.0,
                decode_strategy="beam_search_v3",
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                decoder_start_token_id=None,
                max_length=256,
                diversity_rate=0.0,
                length_penalty=0.6,
                temperature=1.0,
                num_return_sequences=1,
                early_stopping=False,
                forced_eos_token_id=None,
                **model_kwargs):

        bos_token_id = bos_token_id if bos_token_id is not None else getattr(
            self._model, 'bos_token_id', None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(
            self._model, 'eos_token_id', None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(
            self._model, 'pad_token_id', None)
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else getattr(
            self._model, 'decoder_start_token_id', None)

        #(gongenlei) Not enable_faster_encoder temporarily
        if encoder_output is None:
            self.encoder = enable_faster_encoder(self.encoder)
            assert input_ids is not None, "You have to specify either input_ids or encoder_output."
            encoder_output = self.prepare_encoder_decoder_kwargs_for_generation(
                input_ids, model_kwargs)["encoder_output"]
            self.encoder = disable_faster_encoder(self.encoder)
        batch_size = paddle.shape(encoder_output)[0]
        if seq_len is None:
            assert input_ids is not None, "You have to specify either input_ids when generating seq_len."
            seq_len = paddle.sum(paddle.cast(
                input_ids != self.pad_token_id, dtype="int32"),
                                 axis=-1,
                                 keepdim=True,
                                 dtype="int32")
        if self.use_fp16_decoding:
            encoder_output = paddle.cast(encoder_output, "float16")
        if decode_strategy.startswith("beam_search") and num_beams > 1:
            encoder_output, expanded_kwargs = self.expand_inputs_for_generation(
                encoder_output, expand_size=num_beams, seq_len=seq_len)
            seq_len = expanded_kwargs["seq_len"]
        elif decode_strategy == "sampling" and num_return_sequences > 1:
            encoder_output, expanded_kwargs = self.expand_inputs_for_generation(
                encoder_output,
                expand_size=num_return_sequences,
                seq_len=seq_len)
            seq_len = expanded_kwargs["seq_len"]
        if decoder_start_token_id is not None:
            bos_token_id = decoder_start_token_id

        if forced_bos_token_id is not None:
            if decode_strategy == "sampling":
                trg_word = paddle.full(
                    [batch_size * num_return_sequences, 1],
                    forced_bos_token_id,
                    dtype="int32")
            else:
                trg_word = paddle.full(
                    [batch_size, 1], forced_bos_token_id, dtype="int32")
        else:
            trg_word = paddle.zeros([0])

        return self.decoding(
            enc_output=encoder_output,
            memory_seq_lens=seq_len,
            beam_size=num_beams,
            trg_word=trg_word,
            top_k=top_k,
            top_p=top_p,
            decoding_strategy=decode_strategy,
            diversity_rate=diversity_rate,
            max_out_len=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            alpha=length_penalty,
            temperature=temperature,
            early_stopping=early_stopping)

    generate = forward
