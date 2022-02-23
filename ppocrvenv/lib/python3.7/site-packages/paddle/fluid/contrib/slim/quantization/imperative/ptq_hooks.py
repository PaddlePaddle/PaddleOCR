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

import paddle
import math
import numpy as np
from . import ptq_config
from .ptq_registry import PTQRegistry


def quant_forward_post_hook(layer, inputs, outputs):
    """
    The forward_post_hook for PTQ.
    """
    assert hasattr(layer, '_quant_config'), \
        "The layer should have _quant_config attr"

    qc = layer._quant_config
    if qc.enable_in_act_quantizer:
        qc.in_act_quantizer.sample_data(layer, inputs)
    qc.out_act_quantizer.sample_data(layer, (outputs, ))
