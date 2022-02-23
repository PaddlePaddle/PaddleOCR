#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode, default_main_program
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.fluid import core, dygraph_utils
from paddle import _C_ops

__all__ = []


def _verify_dropout_rate(dropout_rate):
    if not isinstance(dropout_rate, (float, int)):
        raise TypeError("dropout_rate argument should be a number")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate argument should between 0 and 1")


def fused_feedforward(x,
                      linear1_weight,
                      linear2_weight,
                      linear1_bias=None,
                      linear2_bias=None,
                      ln1_scale=None,
                      ln1_bias=None,
                      ln2_scale=None,
                      ln2_bias=None,
                      dropout1_rate=0.5,
                      dropout2_rate=0.5,
                      activation="relu",
                      ln1_epsilon=1e-5,
                      ln2_epsilon=1e-5,
                      pre_layer_norm=False,
                      training=True,
                      mode='upscale_in_train',
                      name=None):
    """
    This is a fusion operator to compute feed forward layer in transformer model architecture.
    This operator only supports running on GPU. The function of the operator is consistent with
    the following pseudo code:

    .. code-block:: python

        residual = src;
        if pre_layer_norm:
            src = layer_norm(src)
        src = linear(dropout(activation(dropout(linear(src)))))
        if not pre_layer_norm:
            src = layer_norm(out)

    Args:
        x (Tensor): the input tensor could be 3-D tensor, the input data type could be float16, float32 or float64, the shape is`[batch\_size, sequence\_length, d_model]`.
        linear1_weight (Tensor): The weight of first linear, the data type is same as `x`, the shape is `[d\_model, dim\_feedforward]`.
        linear2_weight (Tensor): The weight of second linear, the data type is same as `x`, the shape is `[dim\_feedforward, d\_model]`.
        linear1_bias (Tensor, optional): The bias of first linear, the data type is same as `x`, the shape is `[dim_feedforward]`. Default None.
        linear2_bias (Tensor, optional): The bias of second linear, the data type is same as `x`, the shape is `[d_model]`. Default None.
        ln1_scale (Tensor, optional): the weight of first layer_norm, the data type is float32 or float64, the shape is same as `x`. Default None.
        ln1_bias (Tensor, optional): The bias of first layer_norm, the data type is float32 or float64, the shape is `[d\_model]`. Default None.
        ln2_scale (Tensor, optional): The weight of second layer_norm, the data type is float32 or float64, the shape is same as `x`. Default None.
        ln2_bias (Tensor, optional): The bias of second layer_norm, the data type is float32 or float64, the shape is `[d\_model]`. Default None.
        dropout1_rate (float, optional): The first dropout probability of setting units to zero. Default 0.5.
        dropout2_rate (float, optional): The second dropout probability of setting units to zero. Default 0.5.
        activation (str, optional): The activation. Default "relu".
        ln1_epsilon (float, optional): Small float of first layer_norm added to denominator to avoid dividing by zero. Default is 1e-5.
        ln2_epsilon (float, optional): Small float of second layer_norm added to denominator to avoid dividing by zero. Default is 1e-5.
        pre_layer_norm (bool, optional): add layer_norm in the pre-processing stage or post-processing state.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default True.
        mode (str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor, the data type and shape is same as `x`.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            import numpy as np
            x_data = np.random.random((1, 8, 8)).astype("float32")
            linear1_weight_data = np.random.random((8, 8)).astype("float32")
            linear2_weight_data = np.random.random((8, 8)).astype("float32")
            x = paddle.to_tensor(x_data)
            linear1_weight = paddle.to_tensor(linear1_weight_data)
            linear2_weight = paddle.to_tensor(linear2_weight_data)
            out = paddle.incubate.nn.functional.fused_feedforward(x, linear1_weight, linear2_weight)
            print(out.numpy().shape)
            # (1, 8, 8)
    """
    _verify_dropout_rate(dropout1_rate)
    _verify_dropout_rate(dropout2_rate)

    seed = None
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'")
    mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

    if in_dygraph_mode():
        if default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        out, _, _, _, _, _, _, _, _, _, _ = _C_ops.fused_feedforward(
            x, None, None, linear1_weight, linear1_bias, linear2_weight,
            linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias,
            'pre_layer_norm', pre_layer_norm, 'ln1_epsilon', ln1_epsilon,
            'ln2_epsilon', ln2_epsilon, 'act_method', activation,
            'dropout1_rate', dropout1_rate, 'dropout2_rate', dropout2_rate,
            "dropout1_is_test", not training, "dropout2_is_test", not training,
            "dropout1_fix_seed", seed is not None, "dropout2_fix_seed",
            seed is not None, "dropout1_seed", seed
            if seed is not None else 0, "dropout2_seed", seed
            if seed is not None else 0, 'dropout1_implementation', mode,
            'dropout2_implementation', mode)
        return out

    helper = LayerHelper("fused_feedforward")
    dtype = x.dtype
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'fused_feedforward')
    check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'],
                'fused_feedforward')

    out = helper.create_variable_for_type_inference(x.dtype)
    dropout1_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    dropout2_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    ln1_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    linear1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout2_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)

    if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
        seed = helper.main_program.random_seed

    helper.append_op(
        type='fused_feedforward',
        inputs={
            'X': x,
            'Linear1Weight': linear1_weight,
            'Linear1Bias': linear1_bias,
            'Linear2Weight': linear2_weight,
            'Linear2Bias': linear2_bias,
            'Ln1Scale': ln1_scale,
            'Ln1Bias': ln1_bias,
            'Ln2Scale': ln2_scale,
            'Ln2Bias': ln2_bias,
        },
        outputs={
            'Out': out,
            'Dropout1Mask': dropout1_mask,
            'Dropout2Mask': dropout2_mask,
            'Ln1Mean': ln1_mean,
            'Ln1Variance': ln1_variance,
            'Ln2Mean': ln2_mean,
            'Ln2Variance': ln2_variance,
            'Linear1Out': linear1_out,
            'Ln1Out': ln1_out,
            'Dropout1Out': dropout1_out,
            'Dropout2Out': dropout2_out,
        },
        attrs={
            'dropout1_rate': dropout1_rate,
            'dropout2_rate': dropout2_rate,
            'act_method': activation,
            'pre_layer_norm': pre_layer_norm,
            'ln1_epsilon': ln1_epsilon,
            'ln2_epsilon': ln2_epsilon,
            'dropout1_is_test': not training,
            'dropout2_is_test': not training,
            'dropout1_fix_seed': seed is not None,
            'dropout2_fix_seed': seed is not None,
            'dropout1_seed': seed if seed is not None else 0,
            'dropout2_seed': seed if seed is not None else 0,
            'dropout1_implementation': mode,
            'dropout2_implementation': mode
        })
    return out


def fused_multi_head_attention(x,
                               qkv_weight,
                               linear_weight,
                               pre_layer_norm=False,
                               pre_ln_scale=None,
                               pre_ln_bias=None,
                               ln_scale=None,
                               ln_bias=None,
                               pre_ln_epsilon=1e-05,
                               qkv_bias=None,
                               linear_bias=None,
                               attn_mask=None,
                               dropout_rate=0.5,
                               attn_dropout_rate=0.5,
                               ln_epsilon=1e-05,
                               training=True,
                               mode='upscale_in_train',
                               name=None):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces. This API only
    support self_attention. The pseudo code is as follows:

    .. code-block:: python

    	if pre_layer_norm:
    	    out = layer_norm(x)
            out = linear(out) + qkv) + bias
    	else:
	    out = linear(x) + bias
    	out = transpose(out, perm=[2, 0, 3, 1, 4])
    	# extract q, k and v from out.
    	q = out[0:1,::]
    	k = out[1:2,::]
    	v = out[2:3,::]
    	out = q * k^t
    	out = attn_mask + out
    	out = softmax(out)
    	out = dropout(out)
    	out = out * v
    	out = transpose(out, perm=[0, 2, 1, 3])
    	out = out_linear(out)
    	if pre_layer_norm:
    	    out = x + dropout(linear_bias + out)
	else:
    	    out = layer_norm(x + dropout(linear_bias + out))

    Parameters:
        x (Tensor): The input tensor of fused_multi_head_attention. The shape is
            `[batch\_size, sequence\_len, embed\_dim]`.
        qkv_weight (Tensor): The qkv weight tensor. The shape is `[3, num_head, dim_head, dim_embed]`.
        linear_weight (Tensor): The linear weight tensor. The shape is `[embed_dim, embed_dim]`.
        pre_layer_norm (bool, optional): whether it is pre_layer_norm (True) or post_layer_norm architecture
	    (False). Default False.
        pre_ln_scale (Tensor, optional): The weight tensor of pre layernorm. Default None.
        pre_ln_bias (Tensor, optional): The bias tensor of pre layernorm. Default None.
        ln_scale (Tensor, optional): The weight tensor of layernorm. Default None.
        ln_bias (Tensor, optional): The bias tensor of layernorm. Default None.
        pre_ln_epsilon (float, optional): Small float value added to denominator of the pre layer_norm
            to avoid dividing by zero. Default is 1e-5.
        qkv_bias (Tensor, optional): The bias of qkv computation. The shape is `[3, num_head, dim_head]`.
            Default None.
        linear_bias (Tensor, optional): The bias of linear. The shape is `[embed_dim]`. Default None.
        attn_mask (Tensor, optional):  A tensor used in multi-head attention to prevents attention to
 	    some unwanted positions, usually the paddings or the subsequent positions. It is a tensor
            with shape broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`. When the
            data type is bool, the unwanted positions have `False` values and the others have `True` values.
            When the data type is int, the unwanted positions have 0 values and the others have 1 values.
            When the data type is float, the unwanted positions have `-INF` values and the others have 0 values.
            It can be None when nothing wanted or needed to be prevented attention to. Default None.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        attn_dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout in attention.
            0 for no dropout. Default 0.5.
        ln_epsilon (float, optional): Small float value added to denominator of layer_norm
            to avoid dividing by zero. Default is 1e-5.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default True.
        mode (str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor, the data type and shape is same as `x`.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            import paddle.incubate.nn.functional as F

            # input: [batch_size, seq_len, embed_dim]
            x = paddle.rand(shape=(2, 4, 128), dtype="float32")
            # qkv_weight: [3, num_head, head_dim, embed_dim]
            qkv_weight = paddle.rand(shape=(3, 4, 32, 128), dtype="float32")
            # qkv_bias: [3, num_head, head_dim]
            qkv_bias = paddle.rand(shape=(3, 4, 32), dtype="float32")
            # linear_weight: [embed_dim, embed_dim]
            linear_weight = paddle.rand(shape=(128, 128), dtype="float32")
            # linear_bias: [embed_dim]
            linear_bias = paddle.rand(shape=[128], dtype="float32")
            # self attention mask: [batch_size, num_heads, seq_len, seq_len]
            attn_mask = paddle.rand(shape=(2, 4, 4, 4), dtype="float32")

            # output: [batch_size, seq_len, embed_dim]
            output = F.fused_multi_head_attention(
                x, qkv_weight, linear_weight, False,
                None, None, None, None, 1e-5, qkv_bias,
                linear_bias, attn_mask)
            # [2, 4, 128]
            print(output.shape)
    """

    seed = None
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'")
    mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

    if in_dygraph_mode():
        if default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        # pre_ln_mean, pre_ln_variance, pre_ln_out, qkv_out, qkv_bias_out, transpose_out, qk_out,
        # qktv_out, softmax_out, attn_dropout_mask_out, attn_dropout_out, attn_mask_out, fmha_out,
        # linear_out, dropout_mask_out, ln_mean_out, ln_var_out, bias_dropout_residual_out, final_out
        assert len(qkv_weight.shape
                   ) == 4, "The dims of the shape of qkv_weight should be 4."
        assert qkv_weight.shape[
            0] == 3, "The shape of qkv_weight should be [3, num_head, head_dim, embed_dim]."
        assert qkv_weight.shape[3] == x.shape[
            2], "The 3rd dim of qkv_weight and 2nd dim of x should be the same, i.e., embed_dim."
        assert qkv_weight.shape[1] * qkv_weight.shape[2] == qkv_weight.shape[
            3], "embed_dim must be divisible by num_heads."

        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, final_out = _C_ops.fused_attention(
            x, pre_ln_scale, pre_ln_bias, qkv_weight, qkv_bias, attn_mask,
            linear_weight, linear_bias, ln_scale, ln_bias, 'pre_layer_norm',
            pre_layer_norm, 'epsilon', pre_ln_epsilon, 'dropout_rate',
            dropout_rate, 'attn_dropout_rate', attn_dropout_rate, 'ln_epsilon',
            ln_epsilon, 'attn_dropout_is_test', not training, 'dropout_is_test',
            not training, 'attn_dropout_fix_seed', seed is not None,
            'dropout_fix_seed', seed is not None, 'attn_dropout_seed', seed
            if seed is not None else 0, 'dropout_seed', seed
            if seed is not None else 0, 'attn_dropout_implementation', mode,
            'dropout_implementation', mode)
        return final_out
    else:
        helper = LayerHelper('fused_multi_head_attention', **locals())
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'fused_multihead_attention')
        check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'],
                    'fused_multi_head_attention')

        # set inputs
        inputs = dict()
        inputs['X'] = [x]
        if pre_ln_scale:
            inputs['LnScale'] = [pre_ln_scale]
        if pre_ln_bias:
            inputs['LnBias'] = [pre_ln_bias]
        inputs['QKVW'] = [qkv_weight]
        if qkv_bias is not None:
            inputs['QKVBias'] = [qkv_bias]
        inputs['SrcMask'] = attn_mask
        inputs['OutLinearW'] = [linear_weight]
        if linear_bias is not None:
            inputs['OutLinearBias'] = [linear_bias]
        if ln_scale:
            inputs['Ln2Scale'] = [ln_scale]
        if ln_bias:
            inputs['Ln2Bias'] = [ln_bias]

        if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
            seed = helper.main_program.random_seed

        # set attrs
        attrs = {
            'pre_layer_norm': pre_layer_norm,
            'epsilon': pre_ln_epsilon,
            'ln_epsilon': ln_epsilon,
            'dropout_rate': dropout_rate,
            'attn_dropout_rate': attn_dropout_rate,
            'attn_dropout_is_test': not training,
            'dropout_is_test': not training,
            'attn_dropout_fix_seed': seed is not None,
            'dropout_fix_seed': seed is not None,
            'attn_dropout_seed': seed if seed is not None else 0,
            'dropout_seed': seed if seed is not None else 0,
            'attn_dropout_implementation': mode,
            'dropout_implementation': mode,
        }

        # set outputs
        pre_ln_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        pre_ln_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        pre_ln_out = helper.create_variable_for_type_inference(dtype=dtype)

        qkv_out = helper.create_variable_for_type_inference(dtype=dtype)
        qkv_bias_out = helper.create_variable_for_type_inference(dtype=dtype)

        transpose_out = helper.create_variable_for_type_inference(dtype=dtype)
        qk_out = helper.create_variable_for_type_inference(dtype=dtype)
        qktv_out = helper.create_variable_for_type_inference(dtype=dtype)
        softmax_out = helper.create_variable_for_type_inference(dtype=dtype)
        attn_dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)
        attn_dropout_out = helper.create_variable_for_type_inference(
            dtype=dtype)
        attn_mask_out = helper.create_variable_for_type_inference(dtype=dtype)
        fmha_out = helper.create_variable_for_type_inference(dtype=dtype)
        out_linear_out = helper.create_variable_for_type_inference(dtype=dtype)
        dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)
        ln_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        ln_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        bias_dropout_residual_out = helper.create_variable_for_type_inference(
            dtype=dtype)
        final_out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='fused_attention',
            inputs=inputs,
            outputs={
                "LnMean": pre_ln_mean_out,
                "LnVariance": pre_ln_variance_out,
                "LnOut": pre_ln_out,
                "QKVOut": qkv_out,
                "QKVBiasOut": qkv_bias_out,
                "TransposeOut2": transpose_out,
                "QKOut": qk_out,
                "QKTVOut": qktv_out,
                "SoftmaxOut": softmax_out,
                "AttnDropoutMaskOut": attn_dropout_mask_out,
                "AttnDropoutOut": attn_dropout_out,
                "SrcMaskOut": attn_mask_out,
                "FMHAOut": fmha_out,
                "OutLinearOut": out_linear_out,
                "DropoutMaskOut": dropout_mask_out,
                "Ln2Mean": ln_mean_out,
                "Ln2Variance": ln_variance_out,
                "BiasDropoutResidualOut": bias_dropout_residual_out,
                'Y': final_out
            },
            attrs=attrs)
        return final_out
