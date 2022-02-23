# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

from .tensor.linalg import cholesky  # noqa: F401
from .tensor.linalg import norm  # noqa: F401
from .tensor.linalg import eig  # noqa: F401
from .tensor.linalg import cond  # noqa: F401
from .tensor.linalg import matrix_power  # noqa: F401
from .tensor.linalg import solve  # noqa: F401
from .tensor import inverse as inv  # noqa: F401
from .tensor.linalg import eigvals  # noqa: F401
from .tensor.linalg import multi_dot  # noqa: F401
from .tensor.linalg import matrix_rank
from .tensor.linalg import svd
from .tensor.linalg import qr
from .tensor.linalg import eigh  # noqa: F401
from .tensor.linalg import eigvalsh
from .tensor.linalg import det
from .tensor.linalg import slogdet
from .tensor.linalg import pinv
from .tensor.linalg import triangular_solve

__all__ = [
    'cholesky',  #noqa
    'norm',
    'cond',
    'inv',
    'eig',
    'eigvals',
    'multi_dot',
    'matrix_rank',
    'svd',
    'qr',
    'matrix_power',
    'det',
    'slogdet',
    'eigh',
    'eigvalsh',
    'pinv',
    'solve',
    'triangular_solve',
]
