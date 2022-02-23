# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from .amp_optimizer import AMPOptimizer
from .asp_optimizer import ASPOptimizer
from .recompute_optimizer import RecomputeOptimizer
from .gradient_merge_optimizer import GradientMergeOptimizer
from .graph_execution_optimizer import GraphExecutionOptimizer
from .parameter_server_optimizer import ParameterServerOptimizer
from .pipeline_optimizer import PipelineOptimizer
from .localsgd_optimizer import LocalSGDOptimizer
from .localsgd_optimizer import AdaptiveLocalSGDOptimizer
from .lars_optimizer import LarsOptimizer
from .parameter_server_graph_optimizer import ParameterServerGraphOptimizer
from .dgc_optimizer import DGCOptimizer
from .lamb_optimizer import LambOptimizer
from .fp16_allreduce_optimizer import FP16AllReduceOptimizer
from .sharding_optimizer import ShardingOptimizer
from .dygraph_optimizer import HybridParallelOptimizer
from .dygraph_optimizer import HybridParallelGradScaler
from .tensor_parallel_optimizer import TensorParallelOptimizer
from .raw_program_optimizer import RawProgramOptimizer
