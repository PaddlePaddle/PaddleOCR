# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
"""
This code is refer from:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/donut/modeling_donut_swin.py

"""

import collections.abc
from collections import OrderedDict
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddle.nn.initializer import (
    TruncatedNormal,
    Constant,
    Normal,
    KaimingUniform,
    XavierUniform,
)

zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
kaiming_normal_ = KaimingUniform(nonlinearity="relu")
trunc_normal_ = TruncatedNormal(std=0.02)
xavier_uniform_ = XavierUniform()

# General docstring
_CONFIG_FOR_DOC = "DonutSwinConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "https://huggingface.co/naver-clova-ix/donut-base"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]


class DonutSwinConfig(object):
    model_type = "donut-swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                print(f"Can't set {key} with value {value} for {self}")
                raise err


@dataclass
# Copied from transformers.models.swin.modeling_swin.SwinEncoderOutput with Swin->DonutSwin
class DonutSwinEncoderOutput(OrderedDict):
    last_hidden_state = None
    hidden_states = None
    attentions = None
    reshaped_hidden_states = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self):
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
# Copied from transformers.models.swin.modeling_swin.SwinModelOutput with Swin->DonutSwin
class DonutSwinModelOutput(OrderedDict):
    last_hidden_state = None
    pooler_output = None
    hidden_states = None
    attentions = None
    reshaped_hidden_states = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self):
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.reshape(
        [
            batch_size,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            num_channels,
        ]
    )
    windows = input_feature.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, num_channels]
    )
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.reshape(
        [
            -1,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            num_channels,
        ]
    )
    windows = windows.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, height, width, num_channels]
    )
    return windows


# Copied from transformers.models.swin.modeling_swin.SwinEmbeddings with Swin->DonutSwin
class DonutSwinEmbeddings(nn.Layer):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = DonutSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        if use_mask_token:
            self.mask_token = paddle.create_parameter(
                [1, 1, config.embed_dim], dtype="float32"
            )
            zeros_(self.mask_token)
        else:
            self.mask_token = None
        if config.use_absolute_embeddings:
            self.position_embeddings = paddle.create_parameter(
                [1, num_patches + 1, config.embed_dim], dtype="float32"
            )
            zeros_(self.position_embedding)
        else:
            self.position_embeddings = None

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values, bool_masked_pos=None):

        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)

        batch_size, seq_len, _ = embeddings.shape

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, output_dimensions


class MyConv2d(nn.Conv2D):
    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding="SAME",
        dilation=1,
        groups=1,
        bias_attr=False,
        eps=1e-6,
    ):
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr,
        )
        self.weight = paddle.create_parameter(
            [out_channels, in_channel, kernel_size[0], kernel_size[1]], dtype="float32"
        )
        self.bias = paddle.create_parameter([out_channels], dtype="float32")
        ones_(self.weight)
        zeros_(self.bias)

    def forward(self, x):
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self._stride,
            self._padding,
            self._dilation,
            self._groups,
        )
        return x


# Copied from transformers.models.swin.modeling_swin.SwinPatchEmbeddings
class DonutSwinPatchEmbeddings(nn.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.is_export = config.is_export
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.projection = nn.Conv2D(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            if self.is_export:
                pad_values = paddle.to_tensor(pad_values, dtype="int32")
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            if self.is_export:
                pad_values = paddle.to_tensor(pad_values, dtype="int32")
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values) -> Tuple[paddle.Tensor, Tuple[int]]:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)

        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose([0, 2, 1])

        return embeddings, output_dimensions


# Copied from transformers.models.swin.modeling_swin.SwinPatchMerging
class DonutSwinPatchMerging(nn.Layer):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Layer`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(
        self,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: nn.Layer = nn.LayerNorm,
        is_export=False,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)
        self.is_export = is_export

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            if self.is_export:
                pad_values = paddle.to_tensor(pad_values, dtype="int32")
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(
        self, input_feature: paddle.Tensor, input_dimensions: Tuple[int, int]
    ) -> paddle.Tensor:
        height, width = input_dimensions
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.reshape([batch_size, height, width, num_channels])

        input_feature = self.maybe_pad(input_feature, height, width)
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        input_feature = paddle.concat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1
        )
        input_feature = input_feature.reshape(
            [batch_size, -1, 4 * num_channels]
        )  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(
    input: paddle.Tensor, drop_prob: float = 0.0, training: bool = False
) -> paddle.Tensor:
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (
        input.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(
        shape,
        dtype=input.dtype,
    )
    random_tensor.floor_()  # binarize
    output = input / keep_prob * random_tensor
    return output


# Copied from transformers.models.swin.modeling_swin.SwinDropPath
class DonutSwinDropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class DonutSwinSelfAttention(nn.Layer):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )
        self.relative_position_bias_table = paddle.create_parameter(
            [(2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads],
            dtype="float32",
        )
        zeros_(self.relative_position_bias_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = paddle.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.query = nn.Linear(
            self.all_head_size, self.all_head_size, bias_attr=config.qkv_bias
        )
        self.key = nn.Linear(
            self.all_head_size, self.all_head_size, bias_attr=config.qkv_bias
        )
        self.value = nn.Linear(
            self.all_head_size, self.all_head_size, bias_attr=config.qkv_bias
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ) -> Tuple[paddle.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer.transpose([0, 1, 3, 2]))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape([-1])
        ]
        relative_position_bias = relative_position_bias.reshape(
            [
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ]
        )

        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in DonutSwinModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.reshape(
                [
                    batch_size // mask_shape,
                    mask_shape,
                    self.num_attention_heads,
                    dim,
                    dim,
                ]
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(
                0
            )
            attention_scores = attention_scores.reshape(
                [-1, self.num_attention_heads, dim, dim]
            )

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = tuple(context_layer.shape[:-2]) + (
            self.all_head_size,
        )
        context_layer = context_layer.reshape(new_context_layer_shape)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput
class DonutSwinSelfOutput(nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor
    ) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinAttention with Swin->DonutSwin
class DonutSwinAttention(nn.Layer):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        self.self = DonutSwinSelfAttention(config, dim, num_heads, window_size)
        self.output = DonutSwinSelfOutput(config, dim)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ) -> Tuple[paddle.Tensor]:
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate
class DonutSwinIntermediate(nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput
class DonutSwinOutput(nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinLayer with Swin->DonutSwin
class DonutSwinLayer(nn.Layer):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, epsilon=config.layer_norm_eps)
        self.attention = DonutSwinAttention(
            config, dim, num_heads, window_size=self.window_size
        )
        self.drop_path = (
            DonutSwinDropPath(config.drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )
        self.layernorm_after = nn.LayerNorm(dim, epsilon=config.layer_norm_eps)
        self.intermediate = DonutSwinIntermediate(config, dim)
        self.output = DonutSwinOutput(config, dim)
        self.is_export = config.is_export

    def set_shift_and_window_size(self, input_resolution):
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(input_resolution)

    def get_attn_mask_export(self, height, width, dtype):

        attn_mask = None
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        img_mask = paddle.zeros((1, height, width, 1), dtype=dtype)
        count = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                if self.shift_size > 0:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1
        if paddle.to_tensor(self.shift_size > 0).cast(paddle.bool):
            # calculate attention mask for SW-MSA
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(
                [-1, self.window_size * self.window_size]
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = paddle.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(
                [-1, self.window_size * self.window_size]
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_bottom, 0, pad_right, 0, 0)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask=None,
        output_attentions=False,
        always_partition=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.shape
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)

        hidden_states = hidden_states.reshape([batch_size, height, width, channels])

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape

        # cyclic shift
        if self.shift_size > 0:
            shift_value = (-self.shift_size, -self.shift_size)
            if self.is_export:
                shift_value = paddle.to_tensor(shift_value, dtype="int32")
            shifted_hidden_states = paddle.roll(
                hidden_states, shifts=shift_value, axis=(1, 2)
            )
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(
            shifted_hidden_states, self.window_size
        )
        hidden_states_windows = hidden_states_windows.reshape(
            [-1, self.window_size * self.window_size, channels]
        )
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)

        attention_outputs = self.attention(
            hidden_states_windows,
            attn_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        attention_windows = attention_output.reshape(
            [-1, self.window_size, self.window_size, channels]
        )
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )
        # reverse cyclic shift
        if self.shift_size > 0:
            shift_value = (self.shift_size, self.shift_size)
            if self.is_export:
                shift_value = paddle.to_tensor(shift_value, dtype="int32")
            attention_windows = paddle.roll(
                shifted_windows, shifts=shift_value, axis=(1, 2)
            )
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.reshape(
            [batch_size, height * width, channels]
        )
        hidden_states = shortcut + self.drop_path(attention_windows)
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)
        layer_outputs = (
            (layer_output, attention_outputs[1])
            if output_attentions
            else (layer_output,)
        )
        return layer_outputs


# Copied from transformers.models.swin.modeling_swin.SwinStage with Swin->DonutSwin
class DonutSwinStage(nn.Layer):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.LayerList(
            [
                DonutSwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )
        self.is_export = config.is_export

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=nn.LayerNorm,
                is_export=self.is_export,
            )
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask=None,
        output_attentions=False,
        always_partition=False,
    ) -> Tuple[paddle.Tensor]:
        height, width = input_dimensions

        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
                always_partition,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(
                hidden_states_before_downsampling, input_dimensions
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# Copied from transformers.models.swin.modeling_swin.SwinEncoder with Swin->DonutSwin
class DonutSwinEncoder(nn.Layer):
    def __init__(self, config, grid_size):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        dpr = [
            x.item()
            for x in paddle.linspace(0, config.drop_path_rate, sum(config.depths))
        ]
        self.layers = nn.LayerList(
            [
                DonutSwinStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[
                        sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])
                    ],
                    downsample=(
                        DonutSwinPatchMerging
                        if (i_layer < self.num_layers - 1)
                        else None
                    ),
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        output_hidden_states_before_downsampling=False,
        always_partition=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )

            hidden_states = layer_outputs[0]

            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.reshape(
                    [
                        batch_size,
                        *(output_dimensions[0], output_dimensions[1]),
                        hidden_size,
                    ]
                )
                reshaped_hidden_state = reshaped_hidden_state.transpose([0, 3, 1, 2])
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.reshape(
                    [batch_size, *input_dimensions, hidden_size]
                )
                reshaped_hidden_state = reshaped_hidden_state.transpose([0, 3, 1, 2])
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return DonutSwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class DonutSwinPreTrainedModel(nn.Layer):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DonutSwinConfig
    base_model_prefix = "swin"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2D)):
            normal_ = Normal(mean=0.0, std=self.config.initializer_range)
            normal_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)

    def post_init(self):
        self.apply(self._initialize_weights)

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask


class DonutSwinModel(DonutSwinPreTrainedModel):
    def __init__(
        self,
        in_channels=3,
        hidden_size=1024,
        num_layers=4,
        num_heads=[4, 8, 16, 32],
        add_pooling_layer=True,
        use_mask_token=False,
        is_export=False,
    ):
        super().__init__()
        donut_swin_config = {
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "use_bfloat16": False,
            "tf_legacy_loss": False,
            "pruned_heads": {},
            "tie_word_embeddings": True,
            "chunk_size_feed_forward": 0,
            "is_encoder_decoder": False,
            "is_decoder": False,
            "cross_attention_hidden_size": None,
            "add_cross_attention": False,
            "tie_encoder_decoder": False,
            "max_length": 20,
            "min_length": 0,
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "num_return_sequences": 1,
            "output_scores": False,
            "return_dict_in_generate": False,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "remove_invalid_values": False,
            "exponential_decay_length_penalty": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "architectures": None,
            "finetuning_task": None,
            "id2label": {0: "LABEL_0", 1: "LABEL_1"},
            "label2id": {"LABEL_0": 0, "LABEL_1": 1},
            "tokenizer_class": None,
            "prefix": None,
            "bos_token_id": None,
            "pad_token_id": None,
            "eos_token_id": None,
            "sep_token_id": None,
            "decoder_start_token_id": None,
            "task_specific_params": None,
            "problem_type": None,
            "_name_or_path": "",
            "_commit_hash": None,
            "_attn_implementation_internal": None,
            "transformers_version": None,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "path_norm": True,
            "use_2d_embeddings": False,
            "image_size": [420, 420],
            "patch_size": 4,
            "num_channels": in_channels,
            "embed_dim": 128,
            "depths": [2, 2, 14, 2],
            "num_heads": num_heads,
            "window_size": 5,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "drop_path_rate": 0.1,
            "hidden_act": "gelu",
            "use_absolute_embeddings": False,
            "layer_norm_eps": 1e-05,
            "initializer_range": 0.02,
            "is_export": is_export,
        }

        config = DonutSwinConfig(**donut_swin_config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = DonutSwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = DonutSwinEncoder(config, self.embeddings.patch_grid)

        self.pooler = nn.AdaptiveAvgPool1D(1) if add_pooling_layer else None
        self.out_channels = hidden_size
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        input_data=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, DonutSwinModelOutput]:
        r"""
        bool_masked_pos (`paddle.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        if self.training:
            pixel_values, label, attention_mask = input_data
        else:
            if isinstance(input_data, list):
                pixel_values = input_data[0]
            else:
                pixel_values = input_data
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = paddle.repeat_interleave(pixel_values, repeats=3, axis=1)

        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose([0, 2, 1]))
            pooled_output = paddle.flatten(pooled_output, 1)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output

        donut_swin_output = DonutSwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
        if self.training:
            return donut_swin_output, label, attention_mask
        else:
            return donut_swin_output
