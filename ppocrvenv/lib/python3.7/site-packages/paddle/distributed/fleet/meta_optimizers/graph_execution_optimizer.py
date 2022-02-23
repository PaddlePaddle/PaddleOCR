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

import copy
import paddle
from paddle.fluid.framework import core
from paddle.fluid import compiler
from .meta_optimizer_base import MetaOptimizerBase
from ..base.private_helper_function import wait_server_ready
import logging
from paddle.static import BuildStrategy

__all__ = []


class GraphExecutionOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(GraphExecutionOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        self.meta_optimizers_black_list = []

    def _is_graph_out(self):
        return True

    def _can_apply(self):
        """
        Basically, this is PE, and almost all programs can be executed here
        """
        if not self.role_maker._is_collective:
            # update me. currently, if parameter server is used
            # graph execution optimizer can not be applied
            return False
        return not self.user_defined_strategy.without_graph_optimization

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        pass

    # should fix the variable
    def _setup_nccl_op(self, startup_program, main_program, build_strategy):
        trainer_endpoints = self.role_maker._get_trainer_endpoints()
        other_trainers = copy.copy(trainer_endpoints)

        trainer_id = self.role_maker._worker_index()
        current_endpoint = self.role_maker._get_trainer_endpoints()[trainer_id]
        other_trainers.remove(current_endpoint)

        trainer_endpoints_env = ",".join(trainer_endpoints)
        trainers_num = self.role_maker._worker_num()

        # NOTE(wangxi): npu don't need to wait server ready
        if trainer_id == 0 and not paddle.is_compiled_with_npu():
            wait_server_ready(other_trainers)

        if core.is_compiled_with_cuda():
            comm_id_var = startup_program.global_block().create_var(
                name="NCCLID", persistable=True, type=core.VarDesc.VarType.RAW)

            for i in range(1, build_strategy.nccl_comm_num):
                startup_program.global_block().create_var(
                    name="NCCLID_{}".format(i),
                    persistable=True,
                    type=core.VarDesc.VarType.RAW)

            if build_strategy.use_hierarchical_allreduce:
                for i in range(0, build_strategy.nccl_comm_num):
                    startup_program.global_block().create_var(
                        name="Hierarchical_inter_NCCLID_{}".format(i),
                        persistable=True,
                        type=core.VarDesc.VarType.RAW)
                    startup_program.global_block().create_var(
                        name="Hierarchical_exter_NCCLID_{}".format(i),
                        persistable=True,
                        type=core.VarDesc.VarType.RAW)

            startup_program.global_block().append_op(
                type="gen_nccl_id",
                inputs={},
                outputs={"NCCLID": comm_id_var},
                attrs={
                    "trainers": trainer_endpoints,
                    "trainer_id": trainer_id,
                    "nccl_comm_num": build_strategy.nccl_comm_num,
                    "use_hierarchical_allreduce":
                    build_strategy.use_hierarchical_allreduce,
                    "hierarchical_allreduce_inter_ranks":
                    build_strategy.hierarchical_allreduce_inter_nranks
                })
        elif core.is_compiled_with_xpu():
            comm_id_var = startup_program.global_block().create_var(
                name="BKCLID", persistable=True, type=core.VarDesc.VarType.RAW)

            #NOTE(liuyuhui) Baidu Kunlun Communication Library(BKCL) currently do not support multi machines.
            assert build_strategy.bkcl_comm_num == 1, \
                "Baidu Kunlun Communication Library(BKCL) currently do not support multi machines."
            for i in range(1, build_strategy.bkcl_comm_num):
                startup_program.global_block().create_var(
                    name="BKCLID_{}".format(i),
                    persistable=True,
                    type=core.VarDesc.VarType.RAW)

            startup_program.global_block().append_op(
                type="gen_bkcl_id",
                inputs={},
                outputs={"BKCLID": comm_id_var},
                attrs={
                    "trainers": trainer_endpoints,
                    "trainer_id": trainer_id,
                    "nccl_comm_num": build_strategy.nccl_comm_num,
                    "use_hierarchical_allreduce":
                    build_strategy.use_hierarchical_allreduce,
                    "hierarchical_allreduce_inter_ranks":
                    build_strategy.hierarchical_allreduce_inter_nranks
                })
        else:
            raise ValueError(
                "comm_id must be generated in paddlepaddle-xpu or paddlepaddle-gpu."
            )

    def _try_to_compile(self, startup_program, main_program, loss):
        dist_strategy = self.user_defined_strategy
        local_build_strategy = dist_strategy.build_strategy

        local_build_strategy.use_hierarchical_allreduce = \
            dist_strategy.use_hierarchical_allreduce
        local_build_strategy.hierarchical_allreduce_inter_nranks = \
            dist_strategy.hierarchical_allreduce_inter_nranks
        local_build_strategy.sync_batch_norm = \
            dist_strategy.sync_batch_norm
        local_build_strategy.fuse_all_reduce_ops = \
            dist_strategy.fuse_all_reduce_ops
        local_build_strategy.nccl_comm_num = \
            dist_strategy.nccl_comm_num

        gradient_scale_configs = self.user_defined_strategy.gradient_scale_configs
        scale_strategys = {
            'avg': BuildStrategy.GradientScaleStrategy.CoeffNumDevice,
            'sum': BuildStrategy.GradientScaleStrategy.One,
            'customized': BuildStrategy.GradientScaleStrategy.Customized,
        }
        assert gradient_scale_configs['scale_strategy'] in scale_strategys, \
            "gradient_scale_configs.scale_strategy must be 'avg', 'sum' or 'customized'"
        local_build_strategy.gradient_scale_strategy = \
            scale_strategys[gradient_scale_configs['scale_strategy']]

        if self.user_defined_strategy.recompute == True:
            logging.warn(
                "set enable_sequential_execution=True since you have enable the recompute strategy"
            )
            local_build_strategy.enable_sequential_execution = True

        exe_strategy = self.user_defined_strategy.execution_strategy
        worker_num = self.role_maker._worker_num()
        node_num = self.role_maker._node_num()

        if self.role_maker._is_collective:
            assert worker_num >= 1, "nccl2 worker_num must >= 1, now:{}" % worker_num

        if worker_num <= 1:
            # local mode
            if local_build_strategy.nccl_comm_num > 1:
                logging.warn("set nccl_comm_num=1 since you only have 1 node.")
            local_build_strategy.nccl_comm_num = 1

        if node_num <= 1:
            if local_build_strategy.use_hierarchical_allreduce:
                logging.warn(
                    "set hierachical_allreduce=False since you only have 1 node."
                )
            local_build_strategy.use_hierarchical_allreduce = False

        sync_allreduce = dist_strategy.sync_nccl_allreduce
        if sync_allreduce:
            exe_strategy.num_threads = max(
                local_build_strategy.nccl_comm_num + 1,
                exe_strategy.num_threads)
            if local_build_strategy.nccl_comm_num > 1:
                logging.warn(
                    "nccl_comm_num > 1, you may need to set sync_nccl_allreduce=False to ensure that different nccl comms can overlap"
                )

        sync_batch_norm = local_build_strategy.sync_batch_norm
        if sync_batch_norm:
            local_build_strategy.nccl_comm_num = 1
            local_build_strategy.use_hierarchical_allreduce = False
            exe_strategy.num_threads = 1
            logging.warn(
                "use sync_batch_norm will hang when set num_threads > 1, so "
                "set num_threads=1, nccl_comm_num=1, hierachical_allreduce=False."
            )

        # NOTE. compatible with compiler, otherwise these values will be overwritten by compiler
        main_program._nccl_comm_num = local_build_strategy.nccl_comm_num
        main_program._use_hierarchical_allreduce = local_build_strategy.use_hierarchical_allreduce
        main_program._hierarchical_allreduce_inter_nranks = local_build_strategy.hierarchical_allreduce_inter_nranks

        # TODO(guru4elephant): should be an independent optimizer
        if worker_num > 1:
            self._setup_nccl_op(startup_program, main_program,
                                local_build_strategy)

        local_build_strategy.num_trainers = self.role_maker._worker_num()
        local_build_strategy.trainer_id = self.role_maker._worker_index()
        local_build_strategy.trainers_endpoints = self.role_maker._get_trainer_endpoints(
        )
        local_build_strategy.enable_backward_optimizer_op_deps = True

        self._compiled_program = compiler.CompiledProgram(main_program)

        self._compiled_program.with_data_parallel(
            loss_name=loss.name,
            build_strategy=local_build_strategy,
            exec_strategy=exe_strategy,
            share_vars_from=None)

        return self._compiled_program

    def _disable_strategy(self, dist_strategy):
        # TODO(guru4elephant): should close all PE related flags here
        return

    def _enable_strategy(self, dist_strategy, context):
        # by default, graph execution strategy is enabled
        return

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        if startup_program == None:
            startup_program = paddle.static.default_startup_program()
        compiled_program = self._try_to_compile(startup_program,
                                                loss.block.program, loss)
        loss.block.program._graph = compiled_program

        # just return self.optimizer_ops and self.param_grads
        return None, None
