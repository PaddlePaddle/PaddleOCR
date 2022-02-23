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

# TODO: define activation functions of neural network

from . import rnn  # noqa: F401
from . import transformer  # noqa: F401
from . import container  # noqa: F401

from .activation import PReLU  # noqa: F401
from .activation import ReLU  # noqa: F401
from .activation import ReLU6  # noqa: F401
from .activation import LeakyReLU  # noqa: F401
from .activation import Sigmoid  # noqa: F401
from .activation import Softmax  # noqa: F401
from .activation import LogSoftmax  # noqa: F401
from .common import Bilinear  # noqa: F401
from .common import Pad1D  # noqa: F401
from .common import Pad2D  # noqa: F401
from .common import Pad3D  # noqa: F401
from .common import CosineSimilarity  # noqa: F401
from .common import Embedding  # noqa: F401
from .common import Linear  # noqa: F401
from .common import Identity  # noqa: F401
from .common import Flatten  # noqa: F401
from .common import Upsample  # noqa: F401
from .common import Dropout  # noqa: F401
from .common import Dropout2D  # noqa: F401
from .common import Dropout3D  # noqa: F401
from .common import AlphaDropout  # noqa: F401
from .common import Upsample  # noqa: F401
from .common import UpsamplingBilinear2D  # noqa: F401
from .common import UpsamplingNearest2D  # noqa: F401
from .pooling import AvgPool1D  # noqa: F401
from .pooling import AvgPool2D  # noqa: F401
from .pooling import AvgPool3D  # noqa: F401
from .pooling import MaxPool1D  # noqa: F401
from .pooling import MaxPool2D  # noqa: F401
from .pooling import MaxPool3D  # noqa: F401
from .pooling import AdaptiveAvgPool1D  # noqa: F401
from .pooling import AdaptiveAvgPool2D  # noqa: F401
from .pooling import AdaptiveAvgPool3D  # noqa: F401
from .pooling import AdaptiveMaxPool1D  # noqa: F401
from .pooling import AdaptiveMaxPool2D  # noqa: F401
from .pooling import AdaptiveMaxPool3D  # noqa: F401
from .pooling import MaxUnPool2D  # noqa: F401
from .conv import Conv1D  # noqa: F401
from .conv import Conv2D  # noqa: F401
from .conv import Conv3D  # noqa: F401
from .conv import Conv1DTranspose  # noqa: F401
from .conv import Conv2DTranspose  # noqa: F401
from .conv import Conv3DTranspose  # noqa: F401
from .loss import BCEWithLogitsLoss  # noqa: F401
from .loss import CrossEntropyLoss  # noqa: F401
from .loss import MSELoss  # noqa: F401
from .loss import L1Loss  # noqa: F401
from .loss import NLLLoss  # noqa: F401
from .loss import BCELoss  # noqa: F401
from .loss import KLDivLoss  # noqa: F401
from .loss import MarginRankingLoss  # noqa: F401
from .loss import CTCLoss  # noqa: F401
from .loss import SmoothL1Loss  # noqa: F401
from .norm import BatchNorm1D  # noqa: F401
from .norm import BatchNorm2D  # noqa: F401
from .norm import BatchNorm3D  # noqa: F401
from .norm import SyncBatchNorm  # noqa: F401
from .norm import GroupNorm  # noqa: F401
from .norm import LayerNorm  # noqa: F401
from .norm import SpectralNorm  # noqa: F401
from .norm import LocalResponseNorm  # noqa: F401

from .vision import PixelShuffle  # noqa: F401
from .distance import PairwiseDistance  # noqa: F401
from .container import LayerDict  # noqa: F401

__all__ = []
