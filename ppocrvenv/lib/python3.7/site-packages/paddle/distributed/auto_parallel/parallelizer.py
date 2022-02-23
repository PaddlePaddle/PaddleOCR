#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.distributed.fleet import cloud_utils
import paddle.fluid.core as core
from .context import DistributedContext
from .context import get_default_distributed_context
from .completion import complete_annotation
from .partitioner import Partitioner
from .process import get_all_process_groups


class AutoParallelizer:
    """
    AutoParallelizer is the main controller class to do the auto parallel process.
    And the auto parallel process will be triggered in the wrapped parallelize function.
    To facilitate the auto parallelization, it will contain information about program, cluster and the
    related context. In this basic version, the program information will be retrevied from 
    Fleet object, and the cluster information can be retrevied in the new created Cluster object,
    and the context information can be retrevied in the new created DistributedContext. 
    """

    def __init__(self, fleet):
        self._fleet = fleet
        self._optimizer = self._fleet.user_defined_optimizer
        self._dist_strategy = self._fleet._user_defined_strategy
        # self._dist_context = DistributedContext()
        self._dist_context = get_default_distributed_context()

    def _remove_distributed_attrs(self, main_program):
        suffix = core.kAutoParallelSuffix()
        # distributed attributes for variable have been removed
        # in previous process.
        for block in main_program.blocks:
            for op in block.ops:
                for attr_name in op.attr_names:
                    if suffix in attr_name:
                        op._remove_attr(attr_name)

    def parallelize(self,
                    loss,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None):
        self._original_main_program = loss.block.program
        # For now, we only allow user to use the default startup and main program
        assert startup_program is not None
        if startup_program == None:
            self._original_startup_program = \
                paddle.static.default_startup_program().clone(for_test=False)
            startup_program = paddle.static.default_startup_program()
        else:
            self._original_startup_program = \
                startup_program.clone(for_test=False)

        # Annotation completion
        completed_main_program = complete_annotation(
            self._original_main_program, self._dist_context)

        # Logical partition 
        rank = paddle.distributed.get_rank()
        partitioner = Partitioner(self._dist_strategy, self._dist_context, rank)
        partitioned_main_prog, partitioned_startup_prog = partitioner.transpile_forward(
            completed_main_program, startup_program)
        dist_params_grads = partitioner.apply_backward(
            loss, completed_main_program, startup_program,
            partitioned_main_prog, partitioned_startup_prog)
        dist_optimize_ops = partitioner.apply_optimize(
            self._optimizer, dist_params_grads, partitioned_main_prog,
            partitioned_startup_prog)

        # Traverse different rank programs and traverse each op of them,
        # instantiate communication by process_mapping.
        all_process_groups = get_all_process_groups()
        for process_group in all_process_groups:
            process_group.instantiate()

        # The last step: remove all distributed attributes to be compatiable
        # with inference.
        self._remove_distributed_attrs(partitioned_main_prog)

        return dist_optimize_ops, dist_params_grads, partitioned_startup_prog, partitioned_main_prog
