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
from .meta_optimizer_base import MetaOptimizerBase
from paddle.fluid import core
import subprocess
import re
import os
import platform
from ..base.private_helper_function import wait_server_ready

__all__ = []


class ParameterServerOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ParameterServerOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(ParameterServerOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

        #self.micro_batch_size = user_defined_strategy.pipeline_configs[
        #    'micro_batch_size']
        self.num_microbatches = user_defined_strategy.pipeline_configs[
            'accumulate_steps']

    def _is_graph_out(self):
        return False

    def _can_apply(self):
        if self.role_maker._is_collective:
            return False

        k_steps = self.user_defined_strategy.a_sync_configs["k_steps"]
        return True if k_steps >= 0 else False

    def get_dist_env(self):
        trainer_id = int(os.getenv('PADDLE_TRAINER_ID', '0'))
        trainer_endpoints = ''
        current_endpoint = ''
        num_trainers = 0
        if os.getenv('PADDLE_TRAINER_ENDPOINTS'):
            trainer_endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS')
            current_endpoint = trainer_endpoints.split(',')[trainer_id]
            num_trainers = len(trainer_endpoints.split(','))

        return {
            'trainer_id': trainer_id,
            'num_trainers': num_trainers,
            'current_endpoint': current_endpoint,
            'trainer_endpoints': trainer_endpoints
        }

    def _get_distributed_strategy(self):
        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

        k_steps = self.user_defined_strategy.a_sync_configs["k_steps"]
        strategy = None

        if not self.user_defined_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_sync_strategy()

        if self.user_defined_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_async_strategy()

        if self.user_defined_strategy.a_sync and k_steps > 0:
            strategy = StrategyFactory.create_geo_strategy(k_steps)

        if not strategy:
            raise ValueError("k_steps must be invalid value, please check")

        return strategy

    def _build_trainer_programs(self, compiled_config):
        from paddle.fluid.incubate.fleet.parameter_server.ir import trainer_pass as worker

        _main = compiled_config.origin_main_program.clone()
        _startup = compiled_config.origin_startup_program.clone()

        use_ps_gpu = self.user_defined_strategy.a_sync_configs["use_ps_gpu"]

        if not compiled_config.is_geo_mode():
            from paddle.fluid.incubate.fleet.parameter_server.ir.public import _add_lr_decay_table_pass
            _add_lr_decay_table_pass(
                _main, compiled_config,
                self.user_defined_strategy.a_sync_configs["lr_decay_steps"])

            # for main program
            _main = worker.distributed_ops_pass(_main, compiled_config,
                                                use_ps_gpu)
            if not use_ps_gpu:
                _main = worker.delete_optimizer_pass(_main, compiled_config)
                _main = worker.append_send_ops_pass(_main, compiled_config)
                _startup = worker.delete_extra_optimizes_pass(_startup,
                                                              compiled_config)

                # for startup program
            _startup = worker.fake_init_ops_pass(_startup, compiled_config)
            if use_ps_gpu:
                _main = worker.ps_gpu_pass(_main)
                from paddle.fluid.transpiler.collective import SingleProcessMultiThread
                t = SingleProcessMultiThread()
                env = self.get_dist_env()
                t.transpile(
                    startup_program=_startup,
                    main_program=_main,
                    rank=env["trainer_id"],
                    endpoints=env["trainer_endpoints"],
                    current_endpoint=env['current_endpoint'],
                    wait_port=False)

            compiled_config.set_origin_ps_main_program(_main)
            compiled_config.set_origin_ps_startup_program(_startup)
            # for heter program
            if self.role_maker._is_heter_parameter_server_mode:
                from paddle.fluid.incubate.fleet.parameter_server.ir import heter_trainer_pass as heter_worker
                if self.role_maker._is_heter_worker():
                    # for heter worker
                    stage_id = self.role_maker._get_stage_id()
                    device = self.role_maker._heter_device_type().lower()
                    _main = heter_worker.split_heter_worker_ops_pass(
                        _main, compiled_config, stage_id, device)
                else:
                    # for default worker
                    _main = heter_worker.split_trainer_ops_pass(_main,
                                                                compiled_config)
        else:
            _main = worker.append_send_ops_pass(_main, compiled_config)
            _startup = _startup
            compiled_config.set_origin_ps_main_program(_main)
            compiled_config.set_origin_ps_startup_program(_startup)

        launch_barrier = self.user_defined_strategy.a_sync_configs[
            "launch_barrier"]
        launch_barrier_flag = int(os.getenv("FLAGS_LAUNCH_BARRIER", "1"))
        if launch_barrier and launch_barrier_flag:
            # for trainer wait server ready
            wait_server_ready(self.role_maker._get_pserver_endpoints())

            # for ps-heter mode, wait heter worker ready
            # if self.role_maker._is_heter_parameter_server_mode and self.role_maker._is_worker(
            # ):
            #     wait_server_ready(self.role_maker._get_heter_worker_endpoints())

        return _main, _startup

    def _build_pserver_programs(self, compiled_config):
        _main = fluid.Program()
        _startup = fluid.Program()

        from paddle.fluid.incubate.fleet.parameter_server.ir import pserver_pass as server

        if not compiled_config.is_geo_mode():

            from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops
            is_sgd_adam = False

            main_program = compiled_config.get_origin_main_program()
            ops = _get_optimize_ops(main_program)

            if len(ops) == 0:
                return _main, _startup

            from paddle.fluid.incubate.fleet.parameter_server.ir.public import _add_lr_decay_table_pass
            lr_decay_steps = self.user_defined_strategy.a_sync_configs[
                "lr_decay_steps"]
            _add_lr_decay_table_pass(main_program, compiled_config,
                                     lr_decay_steps)

            for op in ops:
                if op.type in ["sgd", "adam"]:
                    is_sgd_adam = True
                    break

            if is_sgd_adam:
                return _main, _startup

            _main = server.add_listen_and_serv_pass(_main, compiled_config)
            _main = server.add_rpc_global_flags_pass(_main, compiled_config)
            _main = server.add_optimizer_pass(_main, compiled_config)
            _main = server.large_scale_sparse_pass(_main, _main,
                                                   compiled_config, False)
            _startup = server.build_pserver_startup_program_pass(
                _startup, _main, compiled_config)
            _startup = server.large_scale_sparse_pass(_startup, _main,
                                                      compiled_config, True)

            if not compiled_config.is_sync_mode():
                _main = server.delete_unused_in_main_pass(_main,
                                                          compiled_config)

            _startup = server.delete_unused_in_startup_pass(_startup, _main,
                                                            compiled_config)
        else:
            _main = server.add_listen_and_serv_pass(_main, compiled_config)
            _main = server.add_rpc_global_flags_pass(_main, compiled_config)
            _main = server.add_geo_optimizer_pass(_main, compiled_config)
            _startup = server.build_pserver_startup_program_pass(
                _startup, _main, compiled_config)
            _startup = server.delete_unused_in_startup_pass(_startup, _main,
                                                            compiled_config)

        return _main, _startup

    def _can_apply_geo(self, dist_strategy, program):
        def get_sys_free_mem():
            plat = platform.system()
            if platform.system() == "Darwin":
                vm = subprocess.Popen(
                    ['vm_stat'], stdout=subprocess.PIPE).communicate()[0]
                # Process vm_stat
                vmLines = vm.split('\n')
                sep = re.compile(r':[\s]+')
                vmStats = {}
                for row in range(1, len(vmLines) - 2):
                    rowText = vmLines[row].strip()
                    rowElements = sep.split(rowText)
                    vmStats[(rowElements[0]
                             )] = int(rowElements[1].strip(r'\.')) * 4096
                return vmStats["Pages free"]
            elif platform.system() == "Linux":
                mems = {}
                with open('/proc/meminfo', 'rb') as f:
                    for line in f:
                        fields = line.split()
                        mems[fields[0]] = int(fields[1]) * 1024
                free = mems[b'MemFree:']
                return free
            else:
                raise ValueError(
                    "%s platform is unsupported is parameter server optimizer" %
                    (platform.system()))

        if not isinstance(self.inner_opt, fluid.optimizer.SGDOptimizer):
            return False

        free = get_sys_free_mem()

        from paddle.fluid.incubate.fleet.parameter_server.ir import vars_metatools

        processed_var_names = set(["@EMPTY@"])
        param_memory_size = 0
        for varname in program.global_block().vars:
            var = program.global_block().vars[varname]
            if not var.persistable or var.desc.type(
            ) != core.VarDesc.VarType.LOD_TENSOR:
                continue
            param = vars_metatools.create_var_struct(var)
            param_memory_size += param.m_size
            processed_var_names.add(varname)

        upper_mem_use = param_memory_size * 5.0

        program_tmp_vars = dict()
        eval_batch_size = 1024
        for op in program.global_block().ops:
            for var_name in op.output_arg_names:
                if var_name in processed_var_names:
                    continue
                processed_var_names.add(var_name)
                var = program.global_block().vars[var_name]

                if var.desc.type() != core.VarDesc.VarType.LOD_TENSOR:
                    continue

                data_count = 1
                neg_dim_count = 0
                for x in var.shape:
                    if x < 0:
                        if neg_dim_count >= 1:
                            raise ValueError(
                                "Var %s has more than one negative dim." %
                                (var_name))
                        neg_dim_count += 1
                        data_count *= (-x)
                    else:
                        data_count *= x
                program_tmp_vars[var_name] = (
                    data_count, neg_dim_count,
                    vars_metatools.dtype_to_size[var.dtype])

        for varname in program_tmp_vars:
            data_count, neg_dim_count, type_size = program_tmp_vars[varname]
            if neg_dim_count == 1:
                data_count *= eval_batch_size
            var_memory = data_count * type_size
            upper_mem_use += var_memory

        if upper_mem_use < free:
            return True
        else:
            return False

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.inner_opt.minimize(loss, startup_program, parameter_list,
                                no_grad_set)
        strategy = self._get_distributed_strategy()

        _origin_main_program = loss.block.program
        _origin_startup_program = startup_program
        from paddle.fluid.incubate.fleet.parameter_server.ir import public as public

        compiled_config = public.CompileTimeStrategy(_origin_main_program,
                                                     _origin_startup_program,
                                                     strategy, self.role_maker)
        compiled_config.strategy = strategy

        if self.role_maker._is_worker() or self.role_maker._is_heter_worker():
            main_program, startup_program = self._build_trainer_programs(
                compiled_config)
            if self.role_maker._is_heter_parameter_server_mode:
                _origin_startup_program._heter_pipeline_opt = {
                    "startup_program": startup_program,
                    "pipeline_stage": int(self.role_maker._get_stage_id()) - 1,
                    "heter_place": self.role_maker._heter_device(),
                }

                loss.block.program._heter_pipeline_opt = {
                    "trainer": "HeterPipelineTrainer",
                    "device_worker": "HeterSection",
                    "trainers": self.role_maker._get_stage_trainers(
                    ),  ## trainer num in each stage
                    "trainer_id": int(self.role_maker._role_id()),
                    "pipeline_stage": int(self.role_maker._get_stage_id()) - 1,
                    "num_pipeline_stages":
                    int(self.role_maker._get_num_stage()),
                    "section_program": main_program,
                    "num_microbatches": self.num_microbatches,
                    "heter_place": self.role_maker._heter_device(),
                }
            else:
                loss.block.program = main_program
                fluid.framework.switch_startup_program(startup_program)

        elif self.role_maker._is_server():
            main_program, startup_program = self._build_pserver_programs(
                compiled_config)
            loss.block.program = main_program
            fluid.framework.switch_startup_program(startup_program)
        return None, None

    def _disable_strategy(self, dist_strategy):
        #if self.role_maker._is_heter_parameter_server_mode:
        #    dist_strategy.pipeline = False
        #    dist_strategy.pipeline_configs = {
        #        "micro_batch_size": 1,
        #        "accumulate_steps": 1,
        #    }
        dist_strategy.a_sync = False
        a_sync_configs = dist_strategy.a_sync_configs
        a_sync_configs["k_steps"] = -1
        dist_strategy.a_sync_configs = a_sync_configs

    def _enable_strategy(self, dist_strategy, context):
        #if self.role_maker._is_heter_parameter_server_mode:
        #    dist_strategy.pipeline = True
        #    dist_strategy.pipeline_configs = {
        #        "micro_batch_size": 1,
        #        "accumulate_steps": 1,
        #    }
        a_sync_configs = dist_strategy.a_sync_configs
        if a_sync_configs["k_steps"] >= 0:
            return

        dist_strategy.a_sync = True
        a_sync_configs = dist_strategy.a_sync_configs

        is_geo = self._can_apply_geo(dist_strategy,
                                     context["origin_main_program"])

        if is_geo:
            a_sync_configs["k_steps"] = 800
        else:
            a_sync_configs["k_steps"] = 0
        dist_strategy.a_sync_configs = a_sync_configs
