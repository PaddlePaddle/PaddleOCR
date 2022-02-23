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

from ...fluid.contrib.sparsity import calculate_density  #noqa: F401
from ...fluid.contrib.sparsity import decorate  #noqa: F401
from ...fluid.contrib.sparsity import prune_model  #noqa: F401
from ...fluid.contrib.sparsity import set_excluded_layers  #noqa: F401
from ...fluid.contrib.sparsity import reset_excluded_layers  #noqa: F401

__all__ = [     #noqa
    'calculate_density',
    'decorate',
    'prune_model',
    'set_excluded_layers',
    'reset_excluded_layers'
]
