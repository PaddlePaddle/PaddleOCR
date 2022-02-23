# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
Dataset package.
"""

import paddle.dataset.mnist  # noqa: F401
import paddle.dataset.imikolov  # noqa: F401
import paddle.dataset.imdb  # noqa: F401
import paddle.dataset.cifar  # noqa: F401
import paddle.dataset.movielens  # noqa: F401
import paddle.dataset.conll05  # noqa: F401
import paddle.dataset.uci_housing  # noqa: F401
import paddle.dataset.wmt14  # noqa: F401
import paddle.dataset.wmt16  # noqa: F401
import paddle.dataset.flowers  # noqa: F401
import paddle.dataset.voc2012  # noqa: F401
import paddle.dataset.image  # noqa: F401

# set __all__ as empty for not showing APIs under paddle.dataset
__all__ = []
