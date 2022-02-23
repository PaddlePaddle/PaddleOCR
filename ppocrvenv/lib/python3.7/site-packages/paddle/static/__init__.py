#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#   Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from . import amp  # noqa: F401
from . import sparsity  # noqa: F401
from . import nn  # noqa: F401
from .io import save_inference_model  # noqa: F401
from .io import load_inference_model  # noqa: F401
from .io import deserialize_persistables  # noqa: F401
from .io import serialize_persistables  # noqa: F401
from .io import deserialize_program  # noqa: F401
from .io import serialize_program  # noqa: F401
from .io import load_from_file  # noqa: F401
from .io import save_to_file  # noqa: F401
from .io import normalize_program  # noqa: F401
from ..fluid import Scope  # noqa: F401
from .input import data  # noqa: F401
from .input import InputSpec  # noqa: F401
from ..fluid.executor import Executor  # noqa: F401
from ..fluid.executor import global_scope  # noqa: F401
from ..fluid.executor import scope_guard  # noqa: F401
from ..fluid.backward import append_backward  # noqa: F401
from ..fluid.backward import gradients  # noqa: F401
from ..fluid.compiler import BuildStrategy  # noqa: F401
from ..fluid.compiler import CompiledProgram  # noqa: F401
from ..fluid.compiler import ExecutionStrategy  # noqa: F401
from ..fluid.framework import default_main_program  # noqa: F401
from ..fluid.framework import default_startup_program  # noqa: F401
from ..fluid.framework import device_guard  # noqa: F401
from ..fluid.framework import Program  # noqa: F401
from ..fluid.framework import name_scope  # noqa: F401
from ..fluid.framework import program_guard  # noqa: F401
from ..fluid.framework import cpu_places  # noqa: F401
from ..fluid.framework import cuda_places  # noqa: F401
from ..fluid.framework import xpu_places  # noqa: F401
from ..fluid.framework import Variable  # noqa: F401
from ..fluid.layers.control_flow import Print  # noqa: F401
from ..fluid.layers.nn import py_func  # noqa: F401
from ..fluid.parallel_executor import ParallelExecutor  # noqa: F401
from ..fluid.param_attr import WeightNormParamAttr  # noqa: F401
from ..fluid.optimizer import ExponentialMovingAverage  # noqa: F401
from ..fluid.io import save  # noqa: F401
from ..fluid.io import load  # noqa: F401
from ..fluid.io import load_program_state  # noqa: F401
from ..fluid.io import set_program_state  # noqa: F401

from ..fluid.io import load_vars  # noqa: F401
from ..fluid.io import save_vars  # noqa: F401

from ..fluid.layers import create_parameter  # noqa: F401
from ..fluid.layers import create_global_var  # noqa: F401
from ..fluid.layers.metric_op import auc  # noqa: F401
from ..fluid.layers.metric_op import accuracy  # noqa: F401

__all__ = [     #noqa
           'append_backward',
           'gradients',
           'Executor',
           'global_scope',
           'scope_guard',
           'BuildStrategy',
           'CompiledProgram',
           'Print',
           'py_func',
           'ExecutionStrategy',
           'name_scope',
           'ParallelExecutor',
           'program_guard',
           'WeightNormParamAttr',
           'ExponentialMovingAverage',
           'default_main_program',
           'default_startup_program',
           'Program',
           'data',
           'InputSpec',
           'save',
           'load',
           'save_inference_model',
           'load_inference_model',
           'serialize_program',
           'serialize_persistables',
           'save_to_file',
           'deserialize_program',
           'deserialize_persistables',
           'load_from_file',
           'normalize_program',
           'load_program_state',
           'set_program_state',
           'cpu_places',
           'cuda_places',
           'xpu_places',
           'Variable',
           'create_global_var',
           'accuracy',
           'auc',
           'device_guard',
           'create_parameter'
]
