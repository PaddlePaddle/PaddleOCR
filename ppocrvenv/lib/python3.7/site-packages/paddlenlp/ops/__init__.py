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

from .faster_transformer.transformer.decoding import *
from .faster_transformer.transformer.faster_transformer import *
from .faster_transformer.transformer.decoder import *
from .faster_transformer.transformer.encoder import *
from .einsum import *
from .distributed import *
from . import optimizer

paddle.nn.TransformerEncoderLayer._ft_forward = encoder_layer_forward
paddle.nn.TransformerEncoder._ft_forward = encoder_forward

paddle.nn.TransformerEncoderLayer._ori_forward = paddle.nn.TransformerEncoderLayer.forward
paddle.nn.TransformerEncoder._ori_forward = paddle.nn.TransformerEncoder.forward
