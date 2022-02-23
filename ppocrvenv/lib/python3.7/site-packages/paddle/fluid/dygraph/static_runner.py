# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.dygraph.jit import _SaveLoadConfig
from paddle.fluid.dygraph.io import TranslatedLayer


# NOTE: This class will be deprecated later.
# It is kept here because PaddleHub is already using this API.
class StaticModelRunner(object):
    """
    A Dynamic graph Layer for loading inference program and related parameters,
    and then performing fine-tune training or inference.

    .. note::
        This is a temporary API, which will be deprecated later, please use 
        `fluid.dygraph.jit.load` to achieve the same function.
    """

    def __new__(cls, model_dir, model_filename=None, params_filename=None):
        configs = _SaveLoadConfig()
        if model_filename is not None:
            configs.model_filename = model_filename
        if params_filename is not None:
            configs.params_filename = params_filename
        return TranslatedLayer._construct(model_dir, configs)
