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

from ..fluid.dygraph.layers import Layer  # noqa: F401
from ..fluid.dygraph.container import LayerList  # noqa: F401
from ..fluid.dygraph.container import ParameterList  # noqa: F401
from ..fluid.dygraph.container import Sequential  # noqa: F401

from .clip import ClipGradByGlobalNorm  # noqa: F401
from .clip import ClipGradByNorm  # noqa: F401
from .clip import ClipGradByValue  # noqa: F401
from .decode import BeamSearchDecoder  # noqa: F401
from .decode import dynamic_decode  # noqa: F401
from .layer.activation import ELU  # noqa: F401
from .layer.activation import GELU  # noqa: F401
from .layer.activation import Tanh  # noqa: F401
from .layer.activation import Hardshrink  # noqa: F401
from .layer.activation import Hardswish  # noqa: F401
from .layer.activation import Hardtanh  # noqa: F401
from .layer.activation import PReLU  # noqa: F401
from .layer.activation import ReLU  # noqa: F401
from .layer.activation import ReLU6  # noqa: F401
from .layer.activation import SELU  # noqa: F401
from .layer.activation import Silu  # noqa: F401
from .layer.activation import LeakyReLU  # noqa: F401
from .layer.activation import Sigmoid  # noqa: F401
from .layer.activation import Hardsigmoid  # noqa: F401
from .layer.activation import LogSigmoid  # noqa: F401
from .layer.activation import Softmax  # noqa: F401
from .layer.activation import Softplus  # noqa: F401
from .layer.activation import Softshrink  # noqa: F401
from .layer.activation import Softsign  # noqa: F401
from .layer.activation import Swish  # noqa: F401
from .layer.activation import Mish  # noqa: F401
from .layer.activation import Tanhshrink  # noqa: F401
from .layer.activation import ThresholdedReLU  # noqa: F401
from .layer.activation import LogSoftmax  # noqa: F401
from .layer.activation import Maxout  # noqa: F401
from .layer.common import Pad1D  # noqa: F401
from .layer.common import Pad2D  # noqa: F401
from .layer.common import Pad3D  # noqa: F401
from .layer.common import CosineSimilarity  # noqa: F401
from .layer.common import Embedding  # noqa: F401
from .layer.common import Linear  # noqa: F401
from .layer.common import Identity  # noqa: F401
from .layer.common import Flatten  # noqa: F401
from .layer.common import Upsample  # noqa: F401
from .layer.common import UpsamplingNearest2D  # noqa: F401
from .layer.common import UpsamplingBilinear2D  # noqa: F401
from .layer.common import Bilinear  # noqa: F401
from .layer.common import Dropout  # noqa: F401
from .layer.common import Dropout2D  # noqa: F401
from .layer.common import Dropout3D  # noqa: F401
from .layer.common import AlphaDropout  # noqa: F401
from .layer.common import Unfold  # noqa: F401

from .layer.pooling import AvgPool1D  # noqa: F401
from .layer.pooling import AvgPool2D  # noqa: F401
from .layer.pooling import AvgPool3D  # noqa: F401
from .layer.pooling import MaxPool1D  # noqa: F401
from .layer.pooling import MaxPool2D  # noqa: F401
from .layer.pooling import MaxPool3D  # noqa: F401
from .layer.pooling import MaxUnPool2D  # noqa: F401
from .layer.pooling import AdaptiveAvgPool1D  # noqa: F401
from .layer.pooling import AdaptiveAvgPool2D  # noqa: F401
from .layer.pooling import AdaptiveAvgPool3D  # noqa: F401
from .layer.pooling import AdaptiveMaxPool1D  # noqa: F401
from .layer.pooling import AdaptiveMaxPool2D  # noqa: F401
from .layer.pooling import AdaptiveMaxPool3D  # noqa: F401

from .layer.conv import Conv1D  # noqa: F401
from .layer.conv import Conv2D  # noqa: F401
from .layer.conv import Conv3D  # noqa: F401
from .layer.conv import Conv1DTranspose  # noqa: F401
from .layer.conv import Conv2DTranspose  # noqa: F401
from .layer.conv import Conv3DTranspose  # noqa: F401

from .layer.loss import BCEWithLogitsLoss  # noqa: F401
from .layer.loss import CrossEntropyLoss  # noqa: F401
from .layer.loss import HSigmoidLoss  # noqa: F401
from .layer.loss import MSELoss  # noqa: F401
from .layer.loss import L1Loss  # noqa: F401
from .layer.loss import NLLLoss  # noqa: F401
from .layer.loss import BCELoss  # noqa: F401
from .layer.loss import KLDivLoss  # noqa: F401
from .layer.loss import MarginRankingLoss  # noqa: F401
from .layer.loss import CTCLoss  # noqa: F401
from .layer.loss import SmoothL1Loss  # noqa: F401
from .layer.norm import BatchNorm  # noqa: F401
from .layer.norm import SyncBatchNorm  # noqa: F401
from .layer.norm import GroupNorm  # noqa: F401
from .layer.norm import LayerNorm  # noqa: F401
from .layer.norm import SpectralNorm  # noqa: F401
from .layer.norm import InstanceNorm1D  # noqa: F401
from .layer.norm import InstanceNorm2D  # noqa: F401
from .layer.norm import InstanceNorm3D  # noqa: F401
from .layer.norm import BatchNorm1D  # noqa: F401
from .layer.norm import BatchNorm2D  # noqa: F401
from .layer.norm import BatchNorm3D  # noqa: F401
from .layer.norm import LocalResponseNorm  # noqa: F401

from .layer.rnn import RNNCellBase  # noqa: F401
from .layer.rnn import SimpleRNNCell  # noqa: F401
from .layer.rnn import LSTMCell  # noqa: F401
from .layer.rnn import GRUCell  # noqa: F401
from .layer.rnn import RNN  # noqa: F401
from .layer.rnn import BiRNN  # noqa: F401
from .layer.rnn import SimpleRNN  # noqa: F401
from .layer.rnn import LSTM  # noqa: F401
from .layer.rnn import GRU  # noqa: F401

from .layer.transformer import MultiHeadAttention  # noqa: F401
from .layer.transformer import TransformerEncoderLayer  # noqa: F401
from .layer.transformer import TransformerEncoder  # noqa: F401
from .layer.transformer import TransformerDecoderLayer  # noqa: F401
from .layer.transformer import TransformerDecoder  # noqa: F401
from .layer.transformer import Transformer  # noqa: F401
from .layer.distance import PairwiseDistance  # noqa: F401

from .layer.vision import PixelShuffle  # noqa: F401
from .layer.container import LayerDict  # noqa: F401

from .utils.spectral_norm_hook import spectral_norm

# TODO: remove loss, keep it for too many used in unitests
from .layer import loss  # noqa: F401

from . import utils  # noqa: F401
from . import functional  # noqa: F401
from . import initializer  # noqa: F401
from . import quant  # noqa: F401

#TODO: remove 'diag_embed', 'remove_weight_norm', 'weight_norm' months later.
import paddle.utils.deprecated as deprecated


@deprecated(
    since="2.0.0",
    update_to="paddle.nn.funcitional.diag_embed",
    level=1,
    reason="diag_embed in paddle.nn will be removed in future")
def diag_embed(*args):
    '''
        alias name of paddle.nn.functional.diag_embed
    '''
    return functional.diag_embed(*args)


@deprecated(
    since="2.0.0",
    update_to="paddle.nn.utils.remove_weight_norm",
    level=1,
    reason="remove_weight_norm in paddle.nn will be removed in future")
def remove_weight_norm(*args):
    '''
        alias name of paddle.nn.utils.remove_weight_norm
    '''
    return utils.remove_weight_norm(*args)


@deprecated(
    since="2.0.0",
    update_to="paddle.nn.utils.weight_norm",
    level=1,
    reason="weight_norm in paddle.nn will be removed in future")
def weight_norm(*args):
    '''
        alias name of paddle.nn.utils.weight_norm
    '''
    return utils.weight_norm(*args)


__all__ = [     #noqa
           'BatchNorm',
           'GroupNorm',
           'LayerNorm',
           'SpectralNorm',
           'BatchNorm1D',
           'BatchNorm2D',
           'BatchNorm3D',
           'InstanceNorm1D',
           'InstanceNorm2D',
           'InstanceNorm3D',
           'SyncBatchNorm',
           'LocalResponseNorm',
           'Embedding',
           'Linear',
           'Upsample',
           'UpsamplingNearest2D',
           'UpsamplingBilinear2D',
           'Pad1D',
           'Pad2D',
           'Pad3D',
           'CosineSimilarity',
           'Dropout',
           'Dropout2D',
           'Dropout3D',
           'Bilinear',
           'AlphaDropout',
           'Unfold',
           'RNNCellBase',
           'SimpleRNNCell',
           'LSTMCell',
           'GRUCell',
           'RNN',
           'BiRNN',
           'SimpleRNN',
           'LSTM',
           'GRU',
           'dynamic_decode',
           'MultiHeadAttention',
           'Maxout',
           'Softsign',
           'Transformer',
           'MSELoss',
           'LogSigmoid',
           'BeamSearchDecoder',
           'ClipGradByNorm',
           'ReLU',
           'PairwiseDistance',
           'BCEWithLogitsLoss',
           'SmoothL1Loss',
           'MaxPool3D',
           'AdaptiveMaxPool2D',
           'Hardshrink',
           'Softplus',
           'KLDivLoss',
           'AvgPool2D',
           'L1Loss',
           'LeakyReLU',
           'AvgPool1D',
           'AdaptiveAvgPool3D',
           'AdaptiveMaxPool3D',
           'NLLLoss',
           'Conv1D',
           'Sequential',
           'Hardswish',
           'Conv1DTranspose',
           'AdaptiveMaxPool1D',
           'TransformerEncoder',
           'Softmax',
           'ParameterList',
           'Conv2D',
           'Softshrink',
           'Hardtanh',
           'TransformerDecoderLayer',
           'CrossEntropyLoss',
           'GELU',
           'SELU',
           'Silu',
           'Conv2DTranspose',
           'CTCLoss',
           'ThresholdedReLU',
           'AdaptiveAvgPool2D',
           'MaxPool1D',
           'Layer',
           'TransformerDecoder',
           'Conv3D',
           'Tanh',
           'Conv3DTranspose',
           'Flatten',
           'AdaptiveAvgPool1D',
           'Tanhshrink',
           'HSigmoidLoss',
           'PReLU',
           'TransformerEncoderLayer',
           'AvgPool3D',
           'MaxPool2D',
           'MarginRankingLoss',
           'LayerList',
           'ClipGradByValue',
           'BCELoss',
           'Hardsigmoid',
           'ClipGradByGlobalNorm',
           'LogSoftmax',
           'Sigmoid',
           'Swish',
           'Mish',
           'PixelShuffle',
           'ELU',
           'ReLU6',
           'LayerDict'
]
