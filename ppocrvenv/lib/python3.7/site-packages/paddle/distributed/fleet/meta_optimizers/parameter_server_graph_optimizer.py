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

from paddle import fluid
from paddle.fluid import compiler
from .parameter_server_optimizer import ParameterServerOptimizer

__all__ = []


class ParameterServerGraphOptimizer(ParameterServerOptimizer):
    def __init__(self, optimizer):
        super(ParameterServerGraphOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _can_apply(self):
        if self.role_maker._is_collective:
            return False

        k_steps = self.user_defined_strategy.a_sync_configs["k_steps"]
        if k_steps < 0:
            return False

        if self.role_maker._is_server():
            return False

        if self.role_maker._is_heter_parameter_server_mode:
            return False

        return True

    def _disable_strategy(self, dist_strategy):
        return

    def _enable_strategy(self, dist_strategy, context):
        # only open up the async mode for auto-parallel
        return

    def _is_graph_out(self):
        return True

    def _try_to_compile(self, main_program, loss):
        dist_strategy = self._get_distributed_strategy()

        build_strategy = dist_strategy.get_build_strategy()
        exec_strategy = dist_strategy.get_execute_strategy()

        self._compiled_program = compiler.CompiledProgram(main_program)

        self._compiled_program.with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            share_vars_from=None)

        return self._compiled_program

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        program = loss.block.program
        compiled_program = self._try_to_compile(program, loss)
        program._graph = compiled_program
        # just return self.optimizer_ops and self.param_grads
        return None, None
