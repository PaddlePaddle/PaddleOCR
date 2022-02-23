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

from .viterbi_decode import ViterbiDecoder, viterbi_decode
from .datasets import Conll05st  # noqa: F401
from .datasets import Imdb  # noqa: F401
from .datasets import Imikolov  # noqa: F401
from .datasets import Movielens  # noqa: F401
from .datasets import UCIHousing  # noqa: F401
from .datasets import WMT14  # noqa: F401
from .datasets import WMT16  # noqa: F401

__all__ = [ #noqa
           'Conll05st',
           'Imdb',
           'Imikolov',
           'Movielens',
           'UCIHousing',
           'WMT14',
           'WMT16',
           'ViterbiDecoder',
           'viterbi_decode'
]
