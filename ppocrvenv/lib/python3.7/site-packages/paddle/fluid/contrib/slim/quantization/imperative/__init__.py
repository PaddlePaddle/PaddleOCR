#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import qat
from .qat import *

from . import ptq
from .ptq import *

from . import ptq_config
from .ptq_config import *

from . import ptq_quantizer
from .ptq_quantizer import *

from . import ptq_registry
from .ptq_registry import *

__all__ = []
__all__ += qat.__all__
__all__ += ptq.__all__
__all__ += ptq_config.__all__
__all__ += ptq_quantizer.__all__
__all__ += ptq_registry.__all__
