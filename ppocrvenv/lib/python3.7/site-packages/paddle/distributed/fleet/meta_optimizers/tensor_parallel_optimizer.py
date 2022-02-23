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

from __future__ import print_function
from __future__ import division

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper, is_update_op, is_loss_grad_op, is_backward_op, is_optimizer_op

__all__ = []


class TensorParallelOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(TensorParallelOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
            "LarsOptimizer",
            "LambOptimizer",
        ]
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]
        self.mp_ring_id = 0
        self.global_ring_id = 1
        self.dp_ring_id = 2

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(TensorParallelOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        self.mp_degree = user_defined_strategy.tensor_parallel_configs[
            'tensor_parallel_degree']

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.tensor_parallel == True:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.tensor_parallel = False
        dist_strategy.tensor_parallel_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.tensor_parallel = True
        dist_strategy.tensor_parallel_configs = {"tensor_parallel_degree": 1, }

    def _broadcast_params(self, ring_id, mp_mode):
        block = self.startup_program.global_block()
        param = None
        for param in block.iter_parameters():
            if param.is_distributed and mp_mode:
                continue

            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })

        if not param: return  # no parameter on this device
        block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': param},
            outputs={'Out': param},
            attrs={'ring_id': ring_id,
                   OP_ROLE_KEY: OpRole.Forward})

    def _get_process_group_info(self):
        # global ring info
        self.global_endpoints = self.endpoints
        self.global_rank = self.rank
        self.global_nranks = self.nranks

        # model parallel ring info
        self.mp_rank = self.rank % self.mp_degree
        self.mp_nranks = self.mp_degree
        mp_group = self.rank // self.mp_degree
        self.mp_endpoints = [
            self.endpoints[i] for i in range(self.global_nranks)
            if i // self.mp_degree == mp_group
        ]

        # data parallel ring info
        if self.nranks > self.mp_degree:
            self.dp_rank = self.rank // self.mp_degree
            self.dp_nranks = self.nranks // self.mp_degree
            start_index = self.rank % self.mp_degree
            self.dp_endpoints = [
                self.endpoints[start_index + i * self.mp_degree]
                for i in range(self.dp_nranks)
            ]

    def _init_process_group(self):
        self._get_process_group_info()
        collective_helper = CollectiveHelper(self.role_maker, wait_port=False)

        # Create global ring for all gpus
        collective_helper._init_communicator(
            self.startup_program, self.current_endpoint, self.global_endpoints,
            self.global_rank, self.global_ring_id, True, self.global_ring_id,
            True)

        # Create model parallel ring for all gpus
        collective_helper._init_communicator(
            self.startup_program, self.current_endpoint, self.mp_endpoints,
            self.mp_rank, self.mp_ring_id, True, self.global_ring_id, True)
        self._broadcast_params(self.mp_ring_id, mp_mode=True)

        # Create dp rings
        if self.nranks > self.mp_degree:
            collective_helper._init_communicator(
                self.startup_program, self.current_endpoint, self.dp_endpoints,
                self.dp_rank, self.dp_ring_id, True, self.global_ring_id, True)
            self._broadcast_params(self.dp_ring_id, mp_mode=False)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.endpoints[self.role_maker._worker_index()]
        self.startup_program = startup_program
        if startup_program is None:
            self.startup_program = fluid.default_startup_program()

        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, self.startup_program, parameter_list, no_grad_set)

        self.main_program = loss.block.program
        self.nranks = len(self.endpoints)
        self.rank = self.role_maker._worker_index()

        self._init_process_group()

        assert self.nranks % self.mp_degree == 0

        if self.nranks > self.mp_degree:
            # data parallelism
            dp_degree = self.nranks // self.mp_degree
            self._transpile_main_program(loss, dp_degree)
        return optimize_ops, params_grads

    def _transpile_main_program(self, loss, dp_degree):
        self._insert_loss_grad_ops(loss, dp_degree)
        self._insert_allreduce_ops(loss, self.dp_ring_id)

    def _insert_loss_grad_ops(self, loss, dp_degree):
        """
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        """
        block = loss.block
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / dp_degree,
                        OP_ROLE_KEY: OpRole.Backward
                    })
                break

    def _insert_allreduce_ops(self, loss, ring_id):
        block = loss.block
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.attr(OP_ROLE_VAR_KEY)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                offset = idx
                for i in range(0, len(op_role_var), 2):
                    param = block.vars[op_role_var[i]]
                    grad = block.vars[op_role_var[i + 1]]
                    if offset == idx:
                        offset += 1
                        block._insert_op(
                            offset,
                            type='c_sync_calc_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
                            attrs={OP_ROLE_KEY: OpRole.Backward})
                        offset += 1

                    block._insert_op(
                        offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Backward
                        })

        if grad is None:
            return

        for idx, op in list(enumerate(block.ops)):
            if is_optimizer_op(op):
                block._insert_op(
                    idx,
                    type='c_sync_comm_stream',
                    inputs={'X': grad},
                    outputs={'Out': grad},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
                break
