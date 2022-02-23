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

from .optimizer import Optimizer  # noqa: F401
from .adagrad import Adagrad  # noqa: F401
from .adam import Adam  # noqa: F401
from .adamw import AdamW  # noqa: F401
from .adamax import Adamax  # noqa: F401
from .rmsprop import RMSProp  # noqa: F401
from .adadelta import Adadelta  # noqa: F401
from .sgd import SGD  # noqa: F401
from .momentum import Momentum  # noqa: F401
from .lamb import Lamb  # noqa: F401
from . import lr  # noqa: F401

__all__ = [     #noqa
           'Optimizer',
           'Adagrad',
           'Adam',
           'AdamW',
           'Adamax',
           'RMSProp',
           'Adadelta',
           'SGD',
           'Momentum',
           'Lamb'
]
