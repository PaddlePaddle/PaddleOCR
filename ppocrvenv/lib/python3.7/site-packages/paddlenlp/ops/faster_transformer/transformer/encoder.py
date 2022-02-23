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

import paddle
from paddle.fluid.layer_helper import LayerHelper
import paddle.nn as nn
from paddle.nn import TransformerEncoder, TransformerEncoderLayer
from paddle.utils.cpp_extension import load_op_meta_info_and_register_op

from paddlenlp.utils.log import logger
from paddlenlp.ops.ext_utils import load
from paddlenlp.ops.faster_transformer.transformer.decoding import transfer_param


def infer_transformer_encoder(
        input,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        attn_out_weight,
        attn_out_bias,
        attn_mask,
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        #   sequence_id_offset,
        #   trt_seqlen_offset,
        #   amax_list,
        n_head,
        size_per_head,
        n_layer=12,
        use_gelu=True,
        remove_padding=False,
        int8_mode=0,
        layer_idx=0,
        allow_gemm_test=False,
        use_trt_kernel=False,
        normalize_before=False):
    """
    Fusion Encoder API intergrating Encoder inference in FasterTransformer. It
    accepts the weight and bias of TransformerEncoder and some other parameters
    for inference.
    """
    helper = LayerHelper('fusion_encoder', **locals())
    inputs = {
        'Input': input,
        'SelfQueryWeight': q_weight,
        'SelfQueryBias': q_bias,
        'SelfKeyWeight': k_weight,
        'SelfKeyBias': k_bias,
        'SelfValueWeight': v_weight,
        'SelfValueBias': v_bias,
        'SelfAttnOutputWeight': attn_out_weight,
        'SelfAttnOutputBias': attn_out_bias,
        "SelfAttnMask": attn_mask,
        'SelfAttnOutputLayernormWeight': norm1_weight,
        'SelfAttnOutputLayernormBias': norm1_bias,
        'OutputLayernormWeight': norm2_weight,
        'OutputLayernormBias': norm2_bias,
        'FFNInterWeight': ffn_inter_weight,
        'FFNInterBias': ffn_inter_bias,
        'FFNOutputWeight': ffn_out_weight,
        'FFNOutputBias': ffn_out_bias,
        # "SequenceIdOffset": sequence_id_offset,
        # "TRTSeqLenOffset": trt_seqlen_offset,
        # 'AmaxList': amax_list
    }
    attrs = {
        'head_num': n_head,
        'size_per_head': size_per_head,
        'use_gelu': use_gelu,
        "remove_padding": remove_padding,
        'int8_mode': int8_mode,
        'num_layer': n_layer,
        'layer_idx': layer_idx,
        'allow_gemm_test': allow_gemm_test,
        'use_trt_kernel': use_trt_kernel,
        'normalize_before': normalize_before
    }
    encoder_out = helper.create_variable(dtype=input.dtype)
    outputs = {"EncoderOut": encoder_out}

    helper.append_op(
        type='fusion_encoder', inputs=inputs, outputs=outputs, attrs=attrs)
    return encoder_out


def encoder_layer_forward(self,
                          src,
                          src_mask,
                          cache=None,
                          sequence_id_offset=None,
                          trt_seq_len=None):
    """
    Redefines `forward` function of `paddle.nn.TransformerEncoderLayer` for
    integrating FasterTransformer for inference.

    The original `forward` function would not be replaced unless
    `enable_faster_encoder` is called by objects of its base class. After
    replacing, objects of `paddle.nn.TransformerEncoderLayer` also have the
    same member variables as before.

    After inference, `disable_faster_encoder` could be called to restore the
    `forward` function of `paddle.nn.TransformerEncoder` and
    `paddle.nn.TransformerEncoderLayer`.

    Args:
        src (Tensor):
            The input of Transformer encoder layer. It is a tensor with shape
            `[batch_size, sequence_length, d_model]`. The data type should be
            float32 or float64.
        src_mask (Tensor, optional):
            A tensor used in multi-head attention to prevents attention to some
            unwanted positions, usually the paddings or the subsequent
            positions. It is a tensor with shape `[batch_size, 1, 1, sequence_length]`.
            When the data type is bool, the unwanted positions have `False`
            values and the others have `True` values. When the data type is int,
            the unwanted positions have 0 values and the others have 1 values.
            When the data type is float, the unwanted positions have `-INF`
            values and the others have 0 values. It can be None when nothing
            wanted or needed to be prevented attention to. Defaults to None.

    Returns:
        src(Tensor|tuple):
            It is a tensor that has the same shape and data type as `enc_input`,
            representing the output of Transformer encoder layer. Or a tuple if
            `cache` is not None, except for encoder layer output, the tuple
            includes the new cache which is same as input `cache` argument but
            `incremental_cache` has an incremental length. See
            `paddle.nn.MultiHeadAttention.gen_cache` and
            `paddle.nn.MultiHeadAttention.forward` for more details.
    """
    if cache is not None:
        raise NotImplementedError("cache in encoder is not supported now")

    src = infer_transformer_encoder(
        input=src,
        q_weight=self.self_attn.q_proj.weight,
        q_bias=self.self_attn.q_proj.bias,
        k_weight=self.self_attn.k_proj.weight,
        k_bias=self.self_attn.k_proj.bias,
        v_weight=self.self_attn.v_proj.weight,
        v_bias=self.self_attn.v_proj.bias,
        attn_out_weight=self.self_attn.out_proj.weight,
        attn_out_bias=self.self_attn.out_proj.bias,
        attn_mask=src_mask,
        norm1_weight=self.norm1.weight,
        norm1_bias=self.norm1.bias,
        norm2_weight=self.norm2.weight,
        norm2_bias=self.norm2.bias,
        ffn_inter_weight=self.linear1.weight,
        ffn_inter_bias=self.linear1.bias,
        ffn_out_weight=self.linear2.weight,
        ffn_out_bias=self.linear2.bias,
        # sequence_id_offset=paddle.to_tensor([]),
        # trt_seqlen_offset=paddle.to_tensor([]),
        # amax_list=paddle.to_tensor([]),  # int8 mode is not supported.
        n_head=self._config['nhead'],
        size_per_head=self._config['d_model'] // self._config['nhead'],
        use_gelu=self._config['activation'] == 'gelu',
        normalize_before=self._config['normalize_before'] == True)

    return src


def encoder_forward(self, src, src_mask=None, cache=None):
    """
    Redefines `forward` function of `paddle.nn.TransformerEncoder` for
    integrating FasterTransformer for inference.

    The original `forward` function would not be replaced unless
    `enable_faster_encoder` is called by objects of its base class. After
    replacing, objects of `paddle.nn.TransformerEncoder` also have the same
    member variables as before.

    After inference, `disable_faster_encoder` could be called to restore the
    `forward` function of `paddle.nn.TransformerEncoder` and
    `paddle.nn.TransformerEncoderLayer`.

    Args:
        src (Tensor):
            The input of Transformer encoder. It is a tensor
            with shape `[batch_size, sequence_length, d_model]`. The data
            type should be float32 or float16.
        src_mask (Tensor, optional):
            A tensor used in multi-head attention to prevents attention to
            some unwanted positions, usually the paddings or the subsequent
            positions. It is a tensor with shape `[batch_size, 1, 1, sequence_length]`.
            The data type must be float, the unwanted positions have `-INF` values or other non-zeros
            and the wanted positions must be 0.0.
    Returns:
        output (Tensor|tuple):
            It is a tensor that has the same shape and data type as `src`,
            representing the output of Transformer encoder. Or a tuple if
            `cache` is not None, except for encoder output, the tuple includes
            the new cache which is same as input `cache` argument but
            `incremental_cache` in it has an incremental length. See
            `paddle.nn.MultiHeadAttention.gen_cache` and
            `paddle.nn.MultiHeadAttention.forward` for more details.
    """

    if src_mask.dtype == paddle.float16:
        src_mask = paddle.cast(src_mask, "float32")

    src_mask = src_mask == 0.0
    src_mask = paddle.cast(src_mask, src.dtype)

    # transpose_src_mask: [batch_size, 1, sequence_length, 1]
    transpose_src_mask = paddle.transpose(src_mask, perm=[0, 1, 3, 2])

    # src_mask: [batch_size, 1, sequence_length, sequence_length]
    src_mask = src_mask * transpose_src_mask
    output = src
    for i, layer in enumerate(self.layers):
        output = layer(output, src_mask)
    if self.norm is not None:
        output = self.norm(output)
    return output


def enable_faster_encoder(self, use_fp16=False, encoder_lib=None):
    """
    Compiles fusion encoder operator intergrated FasterTransformer using the
    method of JIT(Just-In-Time) and replaces the `forward` function of
    `paddle.nn.TransformerEncoder` and `paddle.nn.TransformerEncoderLayer`
    objects inherited from `self` to support inference using FasterTransformer.

    Examples:

        .. code-block:: python

            from paddlenlp.ops import enable_faster_encoder, disable_faster_encoder

            model.eval()
            model = enable_faster_encoder(model)
            enc_out = model(src, src_mask)
            model = disable_faster_encoder(model)
    """

    def init_func(layer):
        if isinstance(layer, TransformerEncoderLayer):
            is_usable = True
            if layer._config['bias_attr'] == False:
                logger.warning("`False` for paddle.nn.TransformerEncoder's" \
                               " parameter `bias_attr` is not supported in " \
                               "FasterTransformer by now. The original forward" \
                               " will be involved.")
                is_usable = False
            if layer._config['activation'] not in ('relu', 'gelu'):
                logger.warning("Only 'relu' or 'gelu' is supported by now. " \
                                "The original forward will be involved.")
                is_usable = False
            if is_usable:
                layer.forward = layer._ft_forward
        elif isinstance(layer, TransformerEncoder):
            layer.forward = layer._ft_forward
            if use_fp16:
                convert_to_fp16(layer)

    if not self.training:
        try:
            # Pass decoding lib to prevent re-building encoder.
            # Todo: check weather decoding lib have contained encoder or not.
            if encoder_lib is not None:
                if "FasterTransformer" not in LOADED_EXT.keys():
                    ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                        decoding_lib)
                    LOADED_EXT["FasterTransformer"] = ops
            else:
                load("FasterTransformer", verbose=True)
        except Exception:
            logger.warning(
                "Exception occurs when using FasterEncoder. " \
                "The original forward will be involved. ")
            return self
        for layer in self.children():
            layer.apply(init_func)
    return self


def disable_faster_encoder(self):
    """
    Restores the original `forward` function of `paddle.nn.TransformerEncoder`
    and `paddle.nn.TransformerEncoderLayer` objects inherited from `self`.

    Examples:

        .. code-block:: python

            from paddlenlp.ops import enable_faster_encoder, disable_faster_encoder

            model.eval()
            model = enable_faster_encoder(model)
            enc_out = model(src, src_mask)
            model = disable_faster_encoder(model)
    """

    def init_func(layer):
        if isinstance(layer, (TransformerEncoderLayer, TransformerEncoder)):
            layer.forward = layer._ori_forward

    for layer in self.children():
        layer.apply(init_func)
    return self


def convert_to_fp16(transformer_encoder):
    """ Convert paddle.nn.TransformerEncoder's parameter from float32 to float16

    Args:
        transformer_encoder (obeject, paddle.nn.TransformerEncoder):
            The object to be converted to float16 inplaced, it must be an isinstance
            of paddle.nn.TransformerEncoder.
    """
    if not isinstance(transformer_encoder, paddle.nn.TransformerEncoder):
        logger.warning(
            "transformer_encoder is not isinstance of paddle.nn.TransformerEncoder, return itself with no parameters convertion.".
            format)
        return transformer_encoder
    else:
        encoder_layers = transformer_encoder.layers

        for mod in encoder_layers:
            mod.norm1.weight = transfer_param(
                mod.norm1.weight, restore_data=True)
            mod.norm1.bias = transfer_param(
                mod.norm1.bias, is_bias=True, restore_data=True)
            mod.norm2.weight = transfer_param(
                mod.norm2.weight, restore_data=True)
            mod.norm2.bias = transfer_param(
                mod.norm2.bias, is_bias=True, restore_data=True)

            mod.linear1.weight = transfer_param(
                mod.linear1.weight, restore_data=True)
            mod.linear1.bias = transfer_param(
                mod.linear1.bias, is_bias=True, restore_data=True)

            mod.self_attn.q_proj.weight = transfer_param(
                mod.self_attn.q_proj.weight, restore_data=True)
            mod.self_attn.q_proj.bias = transfer_param(
                mod.self_attn.q_proj.bias, is_bias=True, restore_data=True)
            mod.self_attn.k_proj.weight = transfer_param(
                mod.self_attn.k_proj.weight, restore_data=True)
            mod.self_attn.k_proj.bias = transfer_param(
                mod.self_attn.k_proj.bias, is_bias=True, restore_data=True)
            mod.self_attn.v_proj.weight = transfer_param(
                mod.self_attn.v_proj.weight, restore_data=True)
            mod.self_attn.v_proj.bias = transfer_param(
                mod.self_attn.v_proj.bias, is_bias=True, restore_data=True)
            mod.self_attn.out_proj.weight = transfer_param(
                mod.self_attn.out_proj.weight, restore_data=True)
            mod.self_attn.out_proj.bias = transfer_param(
                mod.self_attn.out_proj.bias, is_bias=True, restore_data=True)

            mod.linear2.weight = transfer_param(
                mod.linear2.weight, restore_data=True)
            mod.linear2.bias = transfer_param(
                mod.linear2.bias, is_bias=True, restore_data=True)
        logger.info(
            "Convert transformer_encoder's parameters from float32 to float16 succeessfully."
        )
