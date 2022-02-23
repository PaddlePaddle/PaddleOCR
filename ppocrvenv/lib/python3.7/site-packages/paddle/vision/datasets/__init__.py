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

from .folder import DatasetFolder  # noqa: F401
from .folder import ImageFolder  # noqa: F401
from .mnist import MNIST  # noqa: F401
from .mnist import FashionMNIST  # noqa: F401
from .flowers import Flowers  # noqa: F401
from .cifar import Cifar10  # noqa: F401
from .cifar import Cifar100  # noqa: F401
from .voc2012 import VOC2012  # noqa: F401

__all__ = [ #noqa
    'DatasetFolder',
    'ImageFolder',
    'MNIST',
    'FashionMNIST',
    'Flowers',
    'Cifar10',
    'Cifar100',
    'VOC2012'
]
