# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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

from __future__ import print_function

from .utils import calculate_density
from .utils import check_mask_1d
from .utils import get_mask_1d
from .utils import check_mask_2d
from .utils import get_mask_2d_greedy
from .utils import get_mask_2d_best
from .utils import create_mask
from .utils import check_sparsity
from .utils import MaskAlgo
from .utils import CheckMethod
from .asp import decorate
from .asp import prune_model
from .asp import set_excluded_layers
from .asp import reset_excluded_layers

__all__ = [
    'calculate_density', 'check_mask_1d', 'get_mask_1d', 'check_mask_2d',
    'get_mask_2d_greedy', 'get_mask_2d_best', 'create_mask', 'check_sparsity',
    'MaskAlgo', 'CheckMethod', 'decorate', 'prune_model', 'set_excluded_layers',
    'reset_excluded_layers'
]
