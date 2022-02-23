#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: import all neural network related api under this directory,
# including layers, linear, conv, rnn etc.

from .activation import elu  # noqa: F401
from .activation import elu_  # noqa: F401
from .activation import gelu  # noqa: F401
from .activation import hardshrink  # noqa: F401
from .activation import hardtanh  # noqa: F401
from .activation import hardsigmoid  # noqa: F401
from .activation import hardswish  # noqa: F401
from .activation import leaky_relu  # noqa: F401
from .activation import log_sigmoid  # noqa: F401
from .activation import maxout  # noqa: F401
from .activation import prelu  # noqa: F401
from .activation import relu  # noqa: F401
from .activation import relu_  # noqa: F401
from .activation import relu6  # noqa: F401
from .activation import selu  # noqa: F401
from .activation import sigmoid  # noqa: F401
from .activation import silu  # noqa: F401
from .activation import softmax  # noqa: F401
from .activation import softmax_  # noqa: F401
from .activation import softplus  # noqa: F401
from .activation import softshrink  # noqa: F401
from .activation import softsign  # noqa: F401
from .activation import swish  # noqa: F401
from .activation import mish  # noqa: F401
from .activation import tanh  # noqa: F401
from .activation import tanh_  # noqa: F401
from .activation import tanhshrink  # noqa: F401
from .activation import thresholded_relu  # noqa: F401
from .activation import log_softmax  # noqa: F401
from .activation import glu  # noqa: F401
from .activation import gumbel_softmax  # noqa: F401
from .common import dropout  # noqa: F401
from .common import dropout2d  # noqa: F401
from .common import dropout3d  # noqa: F401
from .common import alpha_dropout  # noqa: F401
from .common import label_smooth  # noqa: F401
from .common import pad  # noqa: F401
from .common import cosine_similarity  # noqa: F401
from .common import unfold  # noqa: F401
from .common import interpolate  # noqa: F401
from .common import upsample  # noqa: F401
from .common import bilinear  # noqa: F401
from .common import class_center_sample  # noqa: F401
from .conv import conv1d  # noqa: F401
from .conv import conv1d_transpose  # noqa: F401
from .common import linear  # noqa: F401
from .conv import conv2d  # noqa: F401
from .conv import conv2d_transpose  # noqa: F401
from .conv import conv3d  # noqa: F401
from .conv import conv3d_transpose  # noqa: F401
from .extension import diag_embed  # noqa: F401
from .extension import sequence_mask
from .loss import binary_cross_entropy  # noqa: F401
from .loss import binary_cross_entropy_with_logits  # noqa: F401
from .loss import cross_entropy  # noqa: F401
from .loss import dice_loss  # noqa: F401
from .loss import hsigmoid_loss  # noqa: F401
from .loss import kl_div  # noqa: F401
from .loss import l1_loss  # noqa: F401
from .loss import log_loss  # noqa: F401
from .loss import margin_ranking_loss  # noqa: F401
from .loss import mse_loss  # noqa: F401
from .loss import nll_loss  # noqa: F401
from .loss import npair_loss  # noqa: F401
from .loss import sigmoid_focal_loss  # noqa: F401
from .loss import smooth_l1_loss  # noqa: F401
from .loss import softmax_with_cross_entropy  # noqa: F401
from .loss import margin_cross_entropy  # noqa: F401
from .loss import square_error_cost  # noqa: F401
from .loss import ctc_loss  # noqa: F401
from .norm import batch_norm  # noqa: F401
from .norm import instance_norm  # noqa: F401
from .norm import layer_norm  # noqa: F401
from .norm import local_response_norm  # noqa: F401
from .norm import normalize  # noqa: F401
from .pooling import avg_pool1d  # noqa: F401
from .pooling import avg_pool2d  # noqa: F401
from .pooling import avg_pool3d  # noqa: F401
from .pooling import max_pool1d  # noqa: F401
from .pooling import max_pool2d  # noqa: F401
from .pooling import max_pool3d  # noqa: F401

from .pooling import adaptive_max_pool1d  # noqa: F401
from .pooling import adaptive_max_pool2d  # noqa: F401
from .pooling import adaptive_max_pool3d  # noqa: F401
from .pooling import adaptive_avg_pool1d  # noqa: F401
from .pooling import adaptive_avg_pool2d  # noqa: F401
from .pooling import adaptive_avg_pool3d  # noqa: F401
from .pooling import max_unpool2d  # noqa: F401

from .vision import affine_grid  # noqa: F401
from .vision import grid_sample  # noqa: F401
from .vision import pixel_shuffle  # noqa: F401
from .input import one_hot  # noqa: F401
from .input import embedding  # noqa: F401
from ...fluid.layers import gather_tree  # noqa: F401
from ...fluid.layers import temporal_shift  # noqa: F401

from .sparse_attention import sparse_attention

__all__ = [     #noqa
           'conv1d',
           'conv1d_transpose',
           'conv2d',
           'conv2d_transpose',
           'conv3d',
           'conv3d_transpose',
           'elu',
           'elu_',
           'gelu',
           'hardshrink',
           'hardtanh',
           'hardsigmoid',
           'hardswish',
           'leaky_relu',
           'log_sigmoid',
           'maxout',
           'prelu',
           'relu',
           'relu_',
           'relu6',
           'selu',
           'softmax',
           'softmax_',
           'softplus',
           'softshrink',
           'softsign',
           'sigmoid',
           'silu',
           'swish',
           'mish',
           'tanh',
           'tanh_',
           'tanhshrink',
           'thresholded_relu',
           'log_softmax',
           'glu',
           'gumbel_softmax',
           'diag_embed',
           'sequence_mask',
           'dropout',
           'dropout2d',
           'dropout3d',
           'alpha_dropout',
           'label_smooth',
           'linear',
           'pad',
           'unfold',
           'interpolate',
           'upsample',
           'bilinear',
           'cosine_similarity',
           'avg_pool1d',
           'avg_pool2d',
           'avg_pool3d',
           'max_pool1d',
           'max_pool2d',
           'max_pool3d',
           'max_unpool2d',
           'adaptive_avg_pool1d',
           'adaptive_avg_pool2d',
           'adaptive_avg_pool3d',
           'adaptive_max_pool1d',
           'adaptive_max_pool2d',
           'adaptive_max_pool3d',
           'binary_cross_entropy',
           'binary_cross_entropy_with_logits',
           'cross_entropy',
           'dice_loss',
           'hsigmoid_loss',
           'kl_div',
           'l1_loss',
           'log_loss',
           'mse_loss',
           'margin_ranking_loss',
           'nll_loss',
           'npair_loss',
           'sigmoid_focal_loss',
           'smooth_l1_loss',
           'softmax_with_cross_entropy',
           'margin_cross_entropy',
           'square_error_cost',
           'ctc_loss',
           'affine_grid',
           'grid_sample',
           'local_response_norm',
           'pixel_shuffle',
           'embedding',
           'gather_tree',
           'one_hot',
           'normalize',
           'temporal_shift',
           'batch_norm',
           'layer_norm',
           'instance_norm',
           'class_center_sample',
           'sparse_attention',
]
