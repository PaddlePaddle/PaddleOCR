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

# TODO: define all functions about input & output in this directory 

from ..fluid.io import DataLoader  # noqa: F401
from ..fluid.dataloader import Dataset  # noqa: F401
from ..fluid.dataloader import IterableDataset  # noqa: F401
from ..fluid.dataloader import BatchSampler  # noqa: F401
from ..fluid.dataloader import get_worker_info  # noqa: F401
from ..fluid.dataloader import TensorDataset  # noqa: F401
from ..fluid.dataloader import Sampler  # noqa: F401
from ..fluid.dataloader import SequenceSampler  # noqa: F401
from ..fluid.dataloader import RandomSampler  # noqa: F401
from ..fluid.dataloader import DistributedBatchSampler  # noqa: F401
from ..fluid.dataloader import ComposeDataset  # noqa: F401
from ..fluid.dataloader import ChainDataset  # noqa: F401
from ..fluid.dataloader import WeightedRandomSampler  # noqa: F401
from ..fluid.dataloader import Subset  # noqa: F401
from ..fluid.dataloader import random_split  # noqa: F401

__all__ = [ #noqa
           'Dataset',
           'IterableDataset',
           'TensorDataset',
           'ComposeDataset',
           'ChainDataset',
           'BatchSampler',
           'DistributedBatchSampler',
           'DataLoader',
           'get_worker_info',
           'Sampler',
           'SequenceSampler',
           'RandomSampler',
           'WeightedRandomSampler',
           'random_split',
           'Subset'
]
