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

# TODO: import framework api under this directory 

from . import random  # noqa: F401
from .random import seed  # noqa: F401
from .framework import get_default_dtype  # noqa: F401
from .framework import set_default_dtype  # noqa: F401
from .framework import set_grad_enabled  # noqa: F401

from ..fluid.param_attr import ParamAttr  # noqa: F401
from ..fluid.layers.tensor import create_parameter  # noqa: F401
from ..fluid.core import CPUPlace  # noqa: F401
from ..fluid.core import CUDAPlace  # noqa: F401
from ..fluid.core import CUDAPinnedPlace  # noqa: F401
from ..fluid.core import NPUPlace  # noqa: F401
from ..fluid.core import VarBase  # noqa: F401

from paddle.fluid import core  # noqa: F401
from ..fluid.dygraph.base import no_grad_ as no_grad  # noqa: F401
from ..fluid.dygraph.base import grad  # noqa: F401
from .io import save  # noqa: F401
from .io import load  # noqa: F401
from ..fluid.dygraph.parallel import DataParallel  # noqa: F401

__all__ = []
