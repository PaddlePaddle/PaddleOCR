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

from __future__ import print_function
from __future__ import division
import os

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from ..base.private_helper_function import wait_server_ready
from paddle.fluid.optimizer import PipelineOptimizer as PO
from .meta_optimizer_base import MetaOptimizerBase
from .common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper, is_loss_grad_op, is_backward_op, is_optimizer_op

__all__ = []


class PipelineOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(PipelineOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
        ]
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]
        self.global_ring_id = 1
        self.dp_ring_id = 2
        self.start_pipeline_ring_id = 20  # Just a magic number

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(PipelineOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)
        self.micro_batch_size = user_defined_strategy.pipeline_configs[
            'micro_batch_size']
        self.num_microbatches = user_defined_strategy.pipeline_configs[
            'accumulate_steps']
        self.schedule_mode = user_defined_strategy.pipeline_configs[
            'schedule_mode']
        self.use_sharding = user_defined_strategy.sharding

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        # FIXME revise for hybrid parallelism
        if self.use_sharding:
            return False

        if self.user_defined_strategy.pipeline == True:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.pipeline = False
        dist_strategy.pipeline_configs = {
            "micro_batch_size": 1,
            "accumulate_steps": 1,
            "schedule_mode": "1F1B",
        }

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.pipeline = True
        dist_strategy.pipeline_configs = {
            "micro_batch_size": 1,
            "accumulate_steps": 1,
            "schedule_mode": "1F1B",
        }

    def _broadcast_params(self, ring_id):
        block = self.startup_program.global_block()
        param = None
        for param in block.iter_parameters():
            if param.is_distributed:
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

        # data parallel ring info
        if self.pipeline_num > 1:
            self.dp_rank = self.rank // self.inner_parallelism
            self.dp_nranks = self.nranks // self.inner_parallelism
            start_index = self.rank % self.inner_parallelism
            self.dp_endpoints = [
                self.endpoints[start_index + i * self.inner_parallelism]
                for i in range(self.pipeline_num)
            ]

    def _init_process_group(self, pipeline_pair, pipeline_ring_map):
        self._get_process_group_info()
        collective_helper = CollectiveHelper(self.role_maker, wait_port=False)
        # Create global ring for all gpus (ring_id = 0)
        collective_helper._init_communicator(
            self.startup_program, self.current_endpoint, self.global_endpoints,
            self.global_rank, self.global_ring_id, True, self.global_ring_id,
            True)
        # Create pipeline rings
        if self.inner_parallelism > 1:
            pipeline_id = self.rank // self.inner_parallelism
            start_index = pipeline_id * self.inner_parallelism
            for pair in pipeline_pair:
                pair_key = pair[0] * 1000 + pair[1]
                ring_id = pipeline_ring_map[pair_key]
                assert ring_id >= self.start_pipeline_ring_id
                first_node = pair[0] + start_index
                second_node = pair[1] + start_index
                if self.rank != first_node and self.rank != second_node:
                    collective_helper._init_communicator(
                        self.startup_program, None, None, None, None, False,
                        self.global_ring_id, True)
                    continue
                pipeline_endpoints = [
                    self.endpoints[first_node], self.endpoints[second_node]
                ]
                pipeline_rank = 0 if self.rank == first_node else 1
                pipeline_nranks = 2
                collective_helper._init_communicator(
                    self.startup_program, self.current_endpoint,
                    pipeline_endpoints, pipeline_rank, ring_id, False,
                    self.global_ring_id, True)

        # Create dp rings
        if self.pipeline_num > 1:
            collective_helper._init_communicator(
                self.startup_program, self.current_endpoint, self.dp_endpoints,
                self.dp_rank, self.dp_ring_id, True, self.global_ring_id, True)
            self._broadcast_params(self.dp_ring_id)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.endpoints[self.role_maker._worker_index()]
        self.rank = self.role_maker._worker_index()
        self.nranks = self.role_maker._worker_num()

        self.wrapped_opt = PO(self.inner_opt,
                              num_microbatches=self.num_microbatches)
        orig_startup_program = startup_program if startup_program else fluid.default_startup_program(
        )
        block = loss.block
        program = block.program

        program._pipeline_opt = dict()
        program._pipeline_opt['local_rank'] = self.rank
        program._pipeline_opt['global_ring_id'] = self.global_ring_id
        program._pipeline_opt['ring_id'] = self.start_pipeline_ring_id
        program._pipeline_opt['micro_batch_size'] = self.micro_batch_size
        program._pipeline_opt['schedule_mode'] = self.schedule_mode
        program._pipeline_opt['use_sharding'] = False
        program._pipeline_opt['mp_degree'] = 1
        program._pipeline_opt['mp_rank'] = 0
        optimize_ops, params_grads, prog_list, pp_pair, ring_map = self.wrapped_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)
        self.startup_program = orig_startup_program._pipeline_opt[
            'startup_program']
        self.inner_parallelism = program._pipeline_opt['inner_parallelism']
        assert self.nranks % self.inner_parallelism == 0
        assert prog_list
        self.pipeline_num = len(self.endpoints) // self.inner_parallelism

        self._init_process_group(pp_pair, ring_map)

        self.main_program_list = prog_list
        self.main_program = program
        if self.pipeline_num > 1:
            self._transpile_main_program(loss)
        return optimize_ops, params_grads

    def _transpile_main_program(self, loss):
        self._insert_loss_grad_ops(loss, self.pipeline_num)
        self._insert_allreduce_ops(self.dp_ring_id)

    def _insert_loss_grad_ops(self, loss, pipeline_num):
        """
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        """
        block = self.main_program_list[-1].global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / pipeline_num,
                        OP_ROLE_KEY: OpRole.Backward
                    })

    def _insert_allreduce_ops(self, ring_id):
        block = self.main_program._pipeline_opt['section_program'].global_block(
        )
        origin_block = self.main_program.global_block()
        grad = None
        processed_param_name = set()
        first_optimize_op_idx = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and not first_optimize_op_idx:
                first_optimize_op_idx = idx + 1
                # no optimize phase
                if first_optimize_op_idx == len(block.ops): return
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                offset = 0
                for i in range(0, len(op_role_var), 2):
                    param_name = op_role_var[i]
                    param = block.vars[op_role_var[i]]
                    if param_name in processed_param_name: continue
                    processed_param_name.add(param_name)
                    grad_name = op_role_var[i + 1]
                    if not 'MERGED' in grad_name: grad_name += '@MERGED'
                    grad = block.vars[grad_name]
                    origin_param = origin_block.vars[op_role_var[i]]
                    if origin_param.is_distributed:
                        continue

                    block._insert_op(
                        first_optimize_op_idx + offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Optimize
                        })
