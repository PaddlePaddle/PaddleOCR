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
"""
Convert the fluid program to distributed data-parallelism programs.
"""

import os
import sys
import warnings

from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import default_main_program
from paddle.fluid.framework import default_startup_program
from paddle.fluid.framework import Program
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.executor import Executor
from paddle.fluid.parallel_executor import ParallelExecutor
from paddle.fluid.optimizer import Optimizer

from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

from paddle.fluid.incubate.fleet.base.fleet_base import Fleet
from paddle.fluid.incubate.fleet.base.mode import Mode
from paddle.fluid.incubate.fleet.base.role_maker import MPISymetricRoleMaker

from paddle.fluid.incubate.fleet.parameter_server import version
from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_lr_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _has_global_step
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import TrainerRuntimeConfig, DistributedStrategy, \
    SyncStrategy, AsyncStrategy, HalfAsyncStrategy, GeoStrategy, StrategyFactory

from paddle.fluid.transpiler.details.checkport import wait_server_ready

from paddle.fluid.incubate.fleet.parameter_server.mode import PSMode
from paddle.fluid.incubate.fleet.base.fleet_base import DistributedOptimizer

from paddle.fluid.incubate.fleet.parameter_server.ir import trainer_pass as worker
from paddle.fluid.incubate.fleet.parameter_server.ir import pserver_pass as server
from paddle.fluid.incubate.fleet.parameter_server.ir import public as public


class FleetTranspiler(Fleet):
    """
    A subclass for compatibility with fluid.transpiler.DistributeTranspiler.
    """

    def __init__(self):
        super(FleetTranspiler, self).__init__(Mode.TRANSPILER)

        self._inner_mode = None

        if version.is_transpiler():
            self._inner_mode = PSMode.TRANSPILER
        else:
            self._inner_mode = PSMode.PSLIB

        self._strategy = None
        self._transpiler = None
        self._origin_main_program = None
        self._origin_startup_program = None
        self._communicator = None
        self.startup_program = None
        self.main_program = None

        self._opt_info = None
        self._local_ip = 0
        self._fleet_ptr = None
        self._main_programs = []
        self._scopes = []
        self._client2client_request_timeout_ms = 500000
        self._client2client_connect_timeout_ms = 10000
        self._client2client_max_retry = 3

    def init(self, role_maker=None):
        if role_maker is None:
            role_maker = MPISymetricRoleMaker()
        super(FleetTranspiler, self).init(role_maker)
        self._fleet_ptr = core.Fleet()

    def _init_transpiler_worker(self):
        """
        `init_worker` has many many functions to do before training,
        first, wait for all parameter servers launch completely.
        second, run executor to initialize startup program
        third, wait for all worker initialize completely.

        Returns:
            None
        """

        def sync_strategy_envs():
            kwargs = {}
            kwargs[
                "pserver_endpoints"] = self._role_maker.get_pserver_endpoints()
            kwargs["trainer_id"] = self._role_maker.worker_index()
            return kwargs

        def geo_strategy_envs():
            def get_sparse_attrs():
                opt_init_map = {}
                opt_init_map["gaussian_random"] = ["seed", "mean", "std"]
                opt_init_map["fill_constant"] = ["value"]
                opt_init_map["uniform_random"] = ["seed", "min", "max"]
                opt_init_map[
                    "truncated_gaussian_random"] = ["seed", "mean", "std"]

                dist_varnames = get_sparse_tablenames(self._origin_main_program,
                                                      True)
                sparse_varnames = get_sparse_tablenames(
                    self._origin_main_program, False)

                if len(dist_varnames) != 0:
                    raise ValueError(
                        "GeoStrategy can not support large scale embeding now, please use fluid.layers.embedding"
                    )

                init_attrs = []
                for value_name in sparse_varnames:
                    value_var = self._origin_main_program.global_block().vars[
                        value_name]
                    value_attr = [
                        value_name,
                        ",".join([str(dim) for dim in value_var.shape])
                    ]
                    for op in self._origin_startup_program.global_block().ops:
                        if op.type in opt_init_map.keys(
                        ) and value_name == op.output("Out")[0]:
                            init_attr = [op.type]
                            for attr in opt_init_map[op.type]:
                                init_attr.append(str(op.attr(attr)))
                            value_attr.append("&".join(init_attr))
                            init_attrs.append(":".join(value_attr))
                            break
                return "#".join(init_attrs)

            kwargs = {}
            kwargs["trainers"] = self.worker_num()
            kwargs["sparse_attrs"] = get_sparse_attrs()
            return kwargs

        # if MPISymetricRoleMaker is defined
        # we suppose a user wants to submit job on mpi cluster

        if isinstance(self._role_maker, MPISymetricRoleMaker):
            # check whether server has been initialized
            wait_server_ready(self.server_endpoints(to_string=False))

        trainer_config = self._strategy.get_trainer_runtime_config()

        print(trainer_config)

        lrs = _has_global_step(_get_lr_ops(self._origin_main_program))

        if lrs > 0:
            kwargs = {"need_global_step": "1"}
        else:
            kwargs = {"need_global_step": "0"}

        if isinstance(self._strategy, GeoStrategy):
            geo_kwargs = geo_strategy_envs()
            kwargs.update(geo_kwargs)
        if isinstance(self._strategy, SyncStrategy):
            sync_kwargs = sync_strategy_envs()
            kwargs.update(sync_kwargs)

        kwargs = kwargs if kwargs else None

        send_ctx = fleet.compiled_config.get_communicator_send_context()

        if self.compiled_config.is_geo_mode():
            recv_ctx = fleet.compiled_config.get_communicator_recv_context(
                recv_type=4)
        else:
            recv_ctx = fleet.compiled_config.get_communicator_recv_context(
                recv_type=1)

        from paddle.fluid.communicator import Communicator
        self._communicator = Communicator(
            trainer_config.mode, kwargs,
            trainer_config.get_communicator_flags())

        self._communicator.init_with_ctx(send_ctx, recv_ctx)

        if not self._communicator.is_running():
            self._communicator.start()
        else:
            raise ValueError(
                "Communicator can only be inited once, please check")

    def init_worker(self):
        """
        `init_worker` has many many functions to do before training,
        first, wait for all parameter servers launch completely.
        second, run executor to initialize startup program
        third, wait for all worker initialize completely.

        Returns:
            None
        """
        if self._inner_mode == PSMode.TRANSPILER:
            self._init_transpiler_worker()
        else:
            raise NotImplementedError("add implement later")

    def _init_transpiler_server(self, model_dir=None):
        if not self.startup_program:
            raise ValueError(
                "startup_program is None, need invoke DistributedOptimizer.minimize first"
            )

        self._executor.run(self.startup_program)

        if model_dir:
            if not os.path.isdir(model_dir):
                raise ValueError("There is no directory named '%s'", model_dir)

            sparse_varnames = self.compiled_config.get_sparse_varname_on_ps(
                True)
            distribtued_varnames = self.compiled_config.get_sparse_varname_on_ps(
                False)

            remaining_vars = list(
                filter(
                    FleetTranspiler.__exclude_vars(sparse_varnames +
                                                   distribtued_varnames),
                    self.main_program.list_vars()))

            fluid.io.load_vars(
                self._executor,
                main_program=self.main_program,
                dirname=model_dir,
                vars=remaining_vars)

            self._load_sparse_params(
                dirname=model_dir, varnames=sparse_varnames)

            # todo(tangwei12) load distributed vars
            # self._load_sparse_params(dirname=model_dir, varnames=distribtued_varnames)

    def init_server(self, model_dir=None, **kwargs):
        """
        `init_server` has many many functions to do before start pserver,
        first, run executor to initialize startup program,
        second, if the `model_dir` is not empty, it will load parameters from it for increment training.

        Args:
            model_dir(str): The directory path.

        Returns:
            None
        """

        if self._inner_mode == PSMode.TRANSPILER:
            self._init_transpiler_server(model_dir)
        else:
            raise NotImplementedError("add implement later")

    def run_server(self):
        """
        `run_server` execute executor to start pserver main program.

        Returns:
            None
        """

        if self._inner_mode == PSMode.TRANSPILER:
            if not self.main_program:
                raise ValueError(
                    "main_program is None, need invoke DistributedOptimizer.minimize first"
                )

            self._executor.run(self.main_program)
        else:
            raise NotImplementedError("add implement later")

    def stop_worker(self):
        """
        Close this executor.

        For the distributed training, this method would free the resource on PServers related to
        the current Trainer.

        Returns:
            None
        """

        if self._inner_mode == PSMode.TRANSPILER:
            self._communicator.stop()
            if isinstance(self._role_maker, MPISymetricRoleMaker):
                self._role_maker._finalize()
            self._executor.close()
        else:
            raise NotImplementedError("add implement later")

    def distributed_optimizer(self, optimizer, strategy=None):
        """
        Optimizer for distributed training.

        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.

        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(DistributeTranspilerConfig): Extra properties for distributed optimizer.

        Returns:
            TranspilerOptimizer: subclass of DistributedOptimizer.
        """

        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer must be an instance of Optimizer")
        if not self._is_initialized:
            raise ValueError(
                "fleet.init(role) to initialize before optimizer.minimize(loss)")

        if not strategy:
            _strategy = StrategyFactory.create_async_strategy()

        if isinstance(strategy, DistributedStrategy):
            _strategy = strategy
        elif isinstance(strategy, DistributeTranspilerConfig):
            if strategy.sync_mode:
                _strategy = SyncStrategy()
            else:
                if strategy.runtime_split_send_recv:
                    if strategy.geo_sgd_mode:
                        _strategy = GeoStrategy(strategy.geo_sgd_need_push_nums)
                    elif strategy.half_async:
                        _strategy = HalfAsyncStrategy()
                    else:
                        _strategy = AsyncStrategy()
                else:
                    _strategy = HalfAsyncStrategy()
                    # for half_async compatibility
                    strategy.half_async = True
                    strategy.runtime_split_send_recv = True
            _strategy.set_program_config(strategy)
        elif isinstance(strategy, dict):
            if self._inner_mode != PSMode.PSLIB:
                raise TypeError("Dict strategy can only be used at PSLIB Mode")

            _strategy = StrategyFactory.create_async_strategy()
            _strategy.set_pslib_runtime_config(strategy)
        else:
            raise TypeError(
                "strategy must be an instance of DistributeTranspilerConfig, DistributedStrategy"
            )

        self._strategy = _strategy
        self._optimizer = ParameterServerOptimizer(optimizer, _strategy)
        return self._optimizer

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        """
        Prune the given `main_program` to build a new program especially for inference,
        and then save it and all related parameters to given `dirname` by the `executor`.
        """

        if self._inner_mode == PSMode.PSLIB:
            raise NotImplementedError("add implement later")

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save_inference_model() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save_inference_model() function, executor must be as Executor type"
            )

        # Todo(MrChengmo): support recv&save GPU-Kernel for ps-gpu model save
        if not isinstance(executor.place, fluid.CPUPlace):
            save_executor = Executor(fluid.CPUPlace())
        else:
            save_executor = executor

        if main_program is not None:
            if isinstance(main_program, CompiledProgram):
                raise TypeError(
                    "in fleet.save_inference_model() function, main_program must be as Program type, CompiledProgram is not allowed"
                )
            fluid.io.save_inference_model(dirname, feeded_var_names,
                                          target_vars, executor, main_program,
                                          None, None, export_for_deployment)
        else:
            fluid.io.save_inference_model(dirname, feeded_var_names,
                                          target_vars, executor,
                                          self._origin_main_program, None, None,
                                          export_for_deployment, True)

            model_basename = "__model__"
            model_filename = os.path.join(dirname, model_basename)

            with open(model_filename, "rb") as f:
                program_desc_str = f.read()

            program = Program.parse_from_string(program_desc_str)
            program._copy_dist_param_info_from(self.main_program)
            self.save_persistables(executor, dirname, program)

    def _load_sparse_params(self, dirname, varnames):
        from paddle.fluid.communicator import LargeScaleKV
        scale_kv = LargeScaleKV()
        for varname in varnames:
            origin_varname, _, _ = public._get_varname_parts(varname)
            sparse_dir = os.path.join(dirname, origin_varname, varname)
            scale_kv.load(varname, sparse_dir)

    def _get_optimizer_status(self, op, param_name):
        supported_opts = [
            "sgd", "adam", "adagrad", "adamax", "momentum", "lars_momentum",
            "rmsprop", "decayed_adagrad", "ftrl"
        ]

        reshaped_val_map = {}
        reshaped_val_map["sgd"] = []
        reshaped_val_map["adam"] = ["moment1_0", "moment2_0"]
        reshaped_val_map["adagrad"] = ["moment_0"]
        reshaped_val_map["adamax"] = ["moment_0", "inf_norm_0"]
        reshaped_val_map["momentum"] = ["velocity_0"]
        reshaped_val_map["lars_momentum"] = ["velocity_0"]
        reshaped_val_map[
            "rmsprop"] = ["momentum_0", "mean_square_0", "mean_grad_0"]
        reshaped_val_map["decayed_adagrad"] = ["moment_0"]
        reshaped_val_map["ftrl"] = ["squared_0", "linear_0"]

        orishaped_val_map = {}
        orishaped_val_map["adam"] = ["beta1_pow_acc_0", "beta2_pow_acc_0"]
        orishaped_val_map["adamax"] = ["beta1_pow_acc_0"]

        if op not in supported_opts:
            raise ValueError(
                "fleet can not support optimizer: {}, only this can be supported: {}".
                format(op, supported_opts))

        reshaped_names = [
            param_name + "_" + val for val in reshaped_val_map[op]
        ]

        if op not in orishaped_val_map:
            origin_names = []
        else:
            origin_names = [
                param_name + "_" + val for val in orishaped_val_map[op]
            ]
        return reshaped_names, origin_names

    def _get_optimizer_op(self, param_name):
        opts = public._get_optimize_ops(self._origin_main_program)
        for op in opts:
            if "Param" in op.input_names and \
                    "LearningRate" in op.input_names and op.input("Param")[0] == param_name:
                return op

    def _save_dense_params(self, executor, dirname, context, main_program):
        self._communicator.recv()

        prog = Program()
        block = prog.global_block()
        local_vars = []

        for name, var_ctx in context.items():
            if len(var_ctx.origin_varnames()) != 1:
                raise ValueError("Dense can not support split now.")

            varname = var_ctx.origin_varnames()[0]
            local_vars.append(varname)

            optimizer = self._get_optimizer_op(varname)
            reshaped_varnames, origin_varnames = self._get_optimizer_status(
                optimizer.type, varname)

            for var_name in [varname] + reshaped_varnames + origin_varnames:
                var = self._origin_main_program.global_block().vars[var_name]
                block.append_op(
                    type='recv_save',
                    attrs={
                        "trainer_id": self._role_maker.worker_index(),
                        "shape": var.shape,
                        "slice_shapes":
                        [",".join([str(i) for i in var.shape])],
                        "slice_varnames": [var.name],
                        "remote_varnames": [var.name],
                        "is_sparse": False,
                        "endpoints": var_ctx.split_endpoints(),
                        "file_path": os.path.join(dirname, var.name)
                    })

        executor.run(prog)
        return local_vars

    def _save_sparse_params(self, executor, dirname, context, main_program):
        prog = Program()
        block = prog.global_block()
        local_vars = []

        for name, var_ctx in context.items():
            if len(var_ctx.origin_varnames()) != 1:
                raise ValueError("Dense can not support split now.")

            varname = var_ctx.origin_varnames()[0]
            local_vars.append(varname)

            optimizer = self._get_optimizer_op(varname)
            reshaped_varnames, origin_varnames = self._get_optimizer_status(
                optimizer.type, varname)

            var = self._origin_main_program.global_block().vars[varname]
            slice_shapes = []
            dims1 = ",".join([str(i) for i in var.shape[1:]])

            for section in var_ctx.sections():
                slice_shapes.append(str(section) + dims1)

            block.append_op(
                type='recv_save',
                attrs={
                    "trainer_id": self._role_maker.worker_index(),
                    "shape": var.shape,
                    "slice_shapes": slice_shapes,
                    "slice_varnames": var_ctx.split_varnames(),
                    "remote_varnames": var_ctx.split_varnames(),
                    "is_sparse": True,
                    "endpoints": var_ctx.split_endpoints(),
                    "pserver_num":
                    len(self._role_maker.get_pserver_endpoints()),
                    "file_path": os.path.join(dirname, var.name)
                })

            for reshaped_varname in reshaped_varnames:
                var = self._origin_main_program.global_block().vars[
                    reshaped_varname]

                slice_varnames = []
                remote_varnames = []
                for i in range(len(var_ctx.split_varnames())):
                    slice_varnames.append("{}.block{}".format(reshaped_varname,
                                                              i))
                    remote_varnames.append(reshaped_varname)

                block.append_op(
                    type='recv_save',
                    attrs={
                        "trainer_id": self._role_maker.worker_index(),
                        "shape": var.shape,
                        "slice_shapes": slice_shapes,
                        "slice_varnames": slice_varnames,
                        "remote_varnames": remote_varnames,
                        "is_sparse": True,
                        "endpoints": var_ctx.split_endpoints(),
                        "pserver_num":
                        len(self._role_maker.get_pserver_endpoints()),
                        "file_path": os.path.join(dirname, var.name)
                    })

            for origin_varname in origin_varnames:
                var = self._origin_main_program.global_block().vars[
                    origin_varname]

                block.append_op(
                    type='recv_save',
                    attrs={
                        "trainer_id": self._role_maker.worker_index(),
                        "shape": var.shape,
                        "slice_shapes":
                        [",".join([str(i) for i in var.shape])],
                        "slice_varnames": [origin_varname],
                        "remote_varnames": [origin_varname],
                        "is_sparse": False,
                        "endpoints": var_ctx.split_endpoints()[:1],
                        "file_path": os.path.join(dirname, var.name)
                    })
        executor.run(prog)
        return context.keys()

    def _save_distributed_params(self, executor, dirname, context,
                                 main_program):
        prog = Program()
        block = prog.global_block()

        for name, var_ctx in context.items():
            block.append_op(
                type='checkpoint_notify',
                attrs={
                    "varname": name,
                    "is_slice": True,
                    "slice_varnames": var_ctx.split_varnames(),
                    "remote_varnames": var_ctx.split_varnames(),
                    "endpoints": var_ctx.split_endpoints(),
                    "dirname": dirname
                })

        executor.run(prog)
        return context.keys()

    def _save_distributed_persistables(self, executor, dirname, main_program):
        dense_ctx = fleet.compiled_config.get_communicator_recv_context(
            recv_type=1)

        sparse_ctx = fleet.compiled_config.get_communicator_recv_context(
            recv_type=2)

        distributed_ctx = fleet.compiled_config.get_communicator_recv_context(
            recv_type=3)

        recv_dense_varnames = self._save_dense_params(executor, dirname,
                                                      dense_ctx, main_program)

        recv_sparse_varnames = self._save_sparse_params(
            executor, dirname, sparse_ctx, main_program)

        recv_distributed_varnames = self._save_distributed_params(
            executor, dirname, distributed_ctx, main_program)

        saved_varnames = recv_dense_varnames + list(
            recv_sparse_varnames) + list(recv_distributed_varnames)

        remaining_vars = list(
            filter(
                FleetTranspiler.__exclude_vars(saved_varnames),
                main_program.list_vars()))

        fluid.io.save_vars(
            executor,
            main_program=main_program,
            dirname=dirname,
            vars=remaining_vars)

    def save_persistables(self, executor, dirname, main_program=None, **kwargs):
        """
        This function filters out all variables with `persistable==True` from the
        give `main_program` and then saves these variables to the folder `dirname`
        or file `filename`.

        The `dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set `filename` None;
if you would like to save all variables in a
        single file, use `filename` to specify the file name.
        """

        if self._inner_mode == PSMode.PSLIB:
            raise NotImplementedError("add implement later")

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save_persistables() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save_persistables() function, executor must be as Executor type"
            )
        # Todo(MrChengmo): support recv&save GPU-Kernel for ps-gpu model save
        if not isinstance(executor.place, fluid.CPUPlace):
            save_executor = Executor(fluid.CPUPlace())
        else:
            save_executor = executor

        if main_program is None:
            main_program = self.main_program

        if isinstance(main_program, CompiledProgram):
            raise TypeError(
                "in fleet.save_persistables() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        self._save_distributed_persistables(save_executor, dirname,
                                            main_program)

    @staticmethod
    def __exclude_vars(exclude_var_names=[]):
        def is_valid(var):
            if var.name in exclude_var_names:
                return False

            origin_varname, _, _ = public._get_varname_parts(var.name)
            if origin_varname.endswith("@GRAD"):
                return False

            if origin_varname == "learning_rate_0":
                return False

            if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                    var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                    var.desc.type() == core.VarDesc.VarType.READER:
                return False
            return var.persistable

        return is_valid


# fleet is a global instance for parameter server.
fleet = FleetTranspiler()


class ParameterServerOptimizer(DistributedOptimizer):
    """
    DistributedOptimizer is a wrapper for paddle.fluid.optimizer
    A user should pass a paddle.fluid.optimizer to DistributedOptimizer
    minimize() function is implemented.
    DistributedOptimizer is the starting point for a user who wants to
    run distributed training. The optimized information will be stored in
    Fleet() instance who holds the global information about current distributed
    training.

    Args:
        optimizer(Optimizer): subclass of Optimizer.
        strategy(DistributeTranspilerConfig): instance of DistributeTranspilerConfig.

    Returns:
        None
    """

    def __init__(self, optimizer, strategy, mode=PSMode.TRANSPILER):
        super(ParameterServerOptimizer, self).__init__(optimizer, strategy)
        self._mode = mode
        if self._mode == PSMode.PSLIB:
            self._optimizer_name = "Distributed%s" % optimizer.type.capitalize()
            if optimizer.type != "adam":
                print("Currently, distributed optimizer only support Adam"
                      "Will config built-in adam for you."
                      "We will support more functions in DistributedOptimizer",
                      sys.stderr)
                self._optimizer_name = "DistributedAdam"

            self._optimizer = globals()[self._optimizer_name](optimizer)
        else:
            self._optimizer = optimizer

        self._window = 1
        self.type = "downpour"
        self.data_norm_name = [
            ".batch_size", ".batch_square_sum", ".batch_sum",
            ".batch_size@GRAD", ".batch_square_sum@GRAD", ".batch_sum@GRAD"
        ]

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        raise NotImplementedError()

    def apply_gradients(self, params_grads):
        raise NotImplementedError()

    def _build_trainer_programs(self, compiled_config):
        _main = fleet._origin_main_program.clone()
        _startup = fleet._origin_startup_program.clone()

        if not compiled_config.is_geo_mode():
            # for main program
            _main = worker.delete_optimizer_pass(_main, compiled_config)
            _main = worker.distributed_ops_pass(_main, compiled_config)
            _main = worker.append_send_ops_pass(_main, compiled_config)

            # for startup program
            _startup = worker.fake_init_ops_pass(_startup, compiled_config)
            _startup = worker.init_from_server_pass(_startup, compiled_config)
            _startup = worker.delet_extra_optimizes_pass(_startup,
                                                         compiled_config)
        else:
            _main = worker.append_send_ops_pass(_main, compiled_config)
            _startup = _startup

        return _main, _startup

    def _build_pserver_programs(self, compiled_config):
        _main = fluid.Program()
        _startup = fluid.Program()

        if not compiled_config.is_geo_mode():
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
            _main = server.large_scale_sparse_pass(_main, _main,
                                                   compiled_config, False)
            _startup = server.build_pserver_startup_program_pass(
                _startup, _main, compiled_config)
            _startup = server.large_scale_sparse_pass(_startup, _main,
                                                      compiled_config, True)
            _startup = server.delete_unused_in_startup_pass(_startup, _main,
                                                            compiled_config)

        return _main, _startup

    def minimize(self,
                 losses,
                 scopes=None,
                 startup_programs=None,
                 parameter_list=None,
                 no_grad_set=None):

        if isinstance(losses, list):
            raise ValueError("need implement later")

        self._optimizer.minimize(losses, startup_programs, parameter_list,
                                 no_grad_set)

        fleet._origin_main_program = default_main_program().clone(
            for_test=False)
        fleet._origin_startup_program = default_startup_program().clone(
            for_test=False)

        compiled_config = public.CompileTimeStrategy(
            fleet._origin_main_program, fleet._origin_startup_program,
            self._strategy, fleet._role_maker)

        fleet.compiled_config = compiled_config
        fleet.main_program, fleet.startup_program = \
            self._build_trainer_programs(compiled_config) if fleet.is_worker() \
            else self._build_pserver_programs(compiled_config)
