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

import os
import warnings

import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import Program
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.executor import Executor
from paddle.fluid.parallel_executor import ParallelExecutor
from paddle.fluid.framework import Variable, Parameter

from .runtime_base import RuntimeBase
from ..base.private_helper_function import wait_server_ready

__all__ = []


class ParameterServerRuntime(RuntimeBase):
    def __init__(self):
        super(ParameterServerRuntime, self).__init__()
        self._communicator = None

    def _set_basic_info(self, context):
        self.context = context
        self.role_maker = context["role_maker"]
        self.origin_main_program = context["origin_main_program"]
        self.origin_startup_program = context["origin_startup_program"]
        self.async_strategy = self._get_distributed_strategy()
        self.compiled_strategy = self.build_compiled_startegy()

    def _get_distributed_strategy(self):
        strategy = None

        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

        dist_strategy = self.context["valid_strategy"]
        k_steps = dist_strategy.a_sync_configs["k_steps"]

        if not dist_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_sync_strategy()

        if dist_strategy.a_sync and k_steps == 0:
            strategy = StrategyFactory.create_async_strategy()

        if dist_strategy.a_sync and k_steps > 0:
            strategy = StrategyFactory.create_geo_strategy(k_steps)

        if not strategy:
            raise ValueError("k_steps must be invalid value, please check")

        return strategy

    def build_compiled_startegy(self):
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import CompileTimeStrategy

        compiled_config = CompileTimeStrategy(
            self.origin_main_program, self.origin_main_program,
            self.async_strategy, self.role_maker)
        return compiled_config

    def _load_sparse_params(self,
                            executor,
                            dirname,
                            varnames,
                            main_program=None):
        assert vars != None
        check_vars = []
        load_prog = Program()
        load_block = load_prog.global_block()

        def _in_varnames(var):
            return var.name in varnames

        load_vars = list(
            filter(_in_varnames, fluid.default_main_program().list_vars()))
        if main_program is None:
            main_program = self.origin_main_program

        from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_varname_parts
        for each_var in load_vars:
            assert isinstance(each_var, Variable)

            origin_varname, _, _ = _get_varname_parts(each_var.name)

            new_var = fluid.io._clone_var_in_block_(load_block, each_var)
            var_path = os.path.join(dirname, origin_varname)
            if not os.path.exists(var_path):
                raise ValueError("SelectedRows var {} can not find at {}".
                                 format(new_var.name, var_path))

            if os.path.isfile(var_path):
                load_block.append_op(
                    type='sparse_tensor_load',
                    inputs={},
                    outputs={'Out': [new_var]},
                    attrs={
                        'file_path': os.path.join(dirname, origin_varname),
                        'node_index': self.role_maker._server_index(),
                        'node_num': self.role_maker._server_num(),
                        'shape': each_var.shape
                    })
            check_vars.append(each_var)

        executor.run(load_prog)

    def _load_distributed_params(self, dirname, varnames):
        from paddle.fluid.communicator import LargeScaleKV
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_varname_parts

        scale_kv = LargeScaleKV()
        for varname in varnames:
            origin_varname, _, _ = _get_varname_parts(varname)
            sparse_dir = os.path.join(dirname, origin_varname, varname)
            scale_kv.load(varname, sparse_dir)

    @staticmethod
    def __exclude_vars(exclude_var_names=[]):
        def is_valid(var):
            if var.name in exclude_var_names:
                return False

            from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_varname_parts

            origin_varname, _, _ = _get_varname_parts(var.name)
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

    def _init_worker(self):
        def sync_strategy_envs():
            kwargs = {}
            kwargs[
                "pserver_endpoints"] = self.role_maker._get_pserver_endpoints()
            kwargs["trainer_id"] = self.role_maker._worker_index()
            return kwargs

        def geo_strategy_envs():
            from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames

            def get_sparse_attrs():
                opt_init_map = {}
                opt_init_map["gaussian_random"] = ["seed", "mean", "std"]
                opt_init_map["fill_constant"] = ["value"]
                opt_init_map["uniform_random"] = ["seed", "min", "max"]
                opt_init_map[
                    "truncated_gaussian_random"] = ["seed", "mean", "std"]

                dist_varnames = get_sparse_tablenames(self.origin_main_program,
                                                      True)
                sparse_varnames = get_sparse_tablenames(
                    self.origin_main_program, False)

                if len(dist_varnames) != 0:
                    raise ValueError(
                        "GeoStrategy can not support large scale embeding now, please use fluid.layers.embedding"
                    )

                init_attrs = []
                for value_name in sparse_varnames:
                    value_var = self.origin_main_program.global_block().vars[
                        value_name]
                    value_attr = [
                        value_name,
                        ",".join([str(dim) for dim in value_var.shape])
                    ]
                    for op in self.origin_startup_program.global_block().ops:
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
            kwargs["trainers"] = self.role_maker._worker_num()
            kwargs["sparse_attrs"] = get_sparse_attrs()
            return kwargs

        from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_lr_ops, _has_global_step

        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import \
            SyncStrategy, GeoStrategy

        trainer_config = self.async_strategy.get_trainer_runtime_config()
        print(trainer_config)

        dist_strategy = self.context["valid_strategy"]
        launch_barrier = dist_strategy.a_sync_configs["launch_barrier"]
        if launch_barrier:
            # for trainer wait server ready
            wait_server_ready(self.role_maker._get_pserver_endpoints())

            # for ps-heter mode, wait heter worker ready
            if self.role_maker._is_heter_parameter_server_mode and self.role_maker._is_worker(
            ):
                wait_server_ready(self.role_maker._get_heter_worker_endpoints())

        lrs = _has_global_step(_get_lr_ops(self.origin_main_program))

        if lrs:
            kwargs = {"need_global_step": "1"}
        else:
            kwargs = {"need_global_step": "0"}

        if isinstance(self.async_strategy, GeoStrategy):
            geo_kwargs = geo_strategy_envs()
            kwargs.update(geo_kwargs)
        if isinstance(self.async_strategy, SyncStrategy):
            sync_kwargs = sync_strategy_envs()
            kwargs.update(sync_kwargs)

        kwargs = kwargs if kwargs else None

        send_ctx = self.compiled_strategy.get_communicator_send_context()

        if self.compiled_strategy.is_geo_mode():
            recv_ctx = self.compiled_strategy.get_communicator_recv_context(
                recv_type=4)
        else:
            recv_ctx = self.compiled_strategy.get_communicator_recv_context(
                recv_type=1)

        from paddle.fluid.communicator import Communicator
        self._communicator = Communicator(
            trainer_config.mode, kwargs,
            trainer_config.get_communicator_flags())
        self._communicator.init_with_ctx(send_ctx, recv_ctx)

        if not self._communicator.is_running():
            self._communicator.start()
        else:
            warnings.warn("communicator has been initialized, skip")

    def _get_executor(self):
        executor = fluid.Executor(fluid.CPUPlace())
        if self.role_maker._is_heter_parameter_server_mode:
            heter_worker_device_guard = self.context[
                "valid_strategy"].a_sync_configs[
                    "heter_worker_device_guard"].upper()
            if heter_worker_device_guard not in ["GPU", "XPU", "CPU"]:
                raise ValueError("Heter Worker Not Support Device {}".format(
                    heter_worker_device_guard))
            if self.role_maker._is_heter_worker():
                if heter_worker_device_guard == "GPU":
                    executor = Executor(
                        fluid.CUDAPlace(
                            int(os.getenv("FLAGS_selected_gpus", "0"))))
                elif heter_worker_device_guard == "XPU":
                    executor = Executor(
                        fluid.XPUPlace(
                            int(os.getenv("FLAGS_selected_xpus", "0"))))
        return executor

    def _init_server(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError("init server can only accept 1 args: `dirname`")
        elif len(args) == 1:
            model_dirname = args[0]
        else:
            model_dirname = None

        executor = self._get_executor()
        if self.role_maker._is_heter_worker() and self.context[
                "valid_strategy"].a_sync_configs["launch_barrier"]:
            # for heter trainer wait server ready
            wait_server_ready(self.role_maker._get_pserver_endpoints())
        executor.run(fluid.default_startup_program())

        if self.role_maker._is_heter_worker():
            self._init_worker()
            return

        sparse_varnames = self.compiled_strategy.get_sparse_varname_on_ps(False)
        sparse_related_optimize_varnames = []
        for var_name in sparse_varnames:
            sparse_related_optimize_varnames += self.compiled_strategy.get_optimize_varname_on_ps(
                var_name)
        sparse_related_optimize_varnames = list(
            set(sparse_related_optimize_varnames))
        distribtued_varnames = self.compiled_strategy.get_sparse_varname_on_ps(
            True)
        distributed_related_optimize_varnames = []
        for var_name in distribtued_varnames:
            distributed_related_optimize_varnames += self.compiled_strategy.get_optimize_varname_on_ps(
                var_name)
        distributed_related_optimize_varnames = list(
            set(distributed_related_optimize_varnames))

        remaining_vars = list(
            filter(
                ParameterServerRuntime.__exclude_vars(
                    sparse_varnames + distribtued_varnames +
                    sparse_related_optimize_varnames +
                    distributed_related_optimize_varnames),
                fluid.default_main_program().list_vars()))

        if not model_dirname:
            return

        if not os.path.isdir(model_dirname):
            raise ValueError("There is no directory named '%s'", model_dirname)

        # load dense
        fluid.io.load_vars(
            executor,
            main_program=fluid.default_main_program(),
            dirname=model_dirname,
            vars=remaining_vars)

        # load sparse
        self._load_sparse_params(
            executor=executor,
            dirname=model_dirname,
            varnames=sparse_varnames + sparse_related_optimize_varnames)

        # load large scale
        self._load_distributed_params(
            dirname=model_dirname,
            varnames=distribtued_varnames +
            distributed_related_optimize_varnames)

    def _run_server(self):
        executor = self._get_executor()
        executor.run(fluid.default_main_program())

    def _stop_worker(self):
        self._communicator.stop()
        executor = self._get_executor()
        executor.close()

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
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops

        opts = _get_optimize_ops(self.origin_main_program)
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
                var = self.origin_main_program.global_block().vars[var_name]
                block.append_op(
                    type='recv_save',
                    attrs={
                        "trainer_id": self.role_maker._worker_index(),
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

            var = self.origin_main_program.global_block().vars[varname]
            slice_shapes = []
            dims1 = ",".join([str(i) for i in var.shape[1:]])

            for section in var_ctx.sections():
                slice_shapes.append(str(section) + dims1)

            block.append_op(
                type='recv_save',
                attrs={
                    "trainer_id": self.role_maker._worker_index(),
                    "shape": var.shape,
                    "slice_shapes": slice_shapes,
                    "slice_varnames": var_ctx.split_varnames(),
                    "remote_varnames": var_ctx.split_varnames(),
                    "is_sparse": True,
                    "endpoints": var_ctx.split_endpoints(),
                    "pserver_num":
                    len(self.role_maker._get_pserver_endpoints()),
                    "file_path": os.path.join(dirname, var.name)
                })

            for reshaped_varname in reshaped_varnames:
                var = self.origin_main_program.global_block().vars[
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
                        "trainer_id": self.role_maker._worker_index(),
                        "shape": var.shape,
                        "slice_shapes": slice_shapes,
                        "slice_varnames": slice_varnames,
                        "remote_varnames": remote_varnames,
                        "is_sparse": True,
                        "endpoints": var_ctx.split_endpoints(),
                        "pserver_num":
                        len(self.role_maker._get_pserver_endpoints()),
                        "file_path": os.path.join(dirname, var.name)
                    })

            for origin_varname in origin_varnames:
                var = self.origin_main_program.global_block().vars[
                    origin_varname]

                block.append_op(
                    type='recv_save',
                    attrs={
                        "trainer_id": self.role_maker._worker_index(),
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

    def _save_distributed_params(self, executor, dirname, context, mode):
        prog = Program()
        block = prog.global_block()

        for name, var_ctx in context.items():
            block.append_op(
                type='checkpoint_notify',
                attrs={
                    "varname": name,
                    "mode": mode,
                    "slice_varnames": var_ctx.split_varnames(),
                    "remote_varnames": var_ctx.split_varnames(),
                    "endpoints": var_ctx.split_endpoints(),
                    "dirname": dirname
                })

        executor.run(prog)
        return context.keys()

    def _save_distributed_persistables(self, executor, dirname, main_program,
                                       mode):
        dense_ctx = self.compiled_strategy.get_communicator_recv_context(
            recv_type=1, use_origin_program=True)

        sparse_ctx = self.compiled_strategy.get_communicator_recv_context(
            recv_type=2, use_origin_program=True)

        distributed_ctx = self.compiled_strategy.get_communicator_recv_context(
            recv_type=3, use_origin_program=True)

        recv_dense_varnames = self._save_dense_params(executor, dirname,
                                                      dense_ctx, main_program)

        recv_sparse_varnames = self._save_sparse_params(
            executor, dirname, sparse_ctx, main_program)

        recv_distributed_varnames = self._save_distributed_params(
            executor, dirname, distributed_ctx, mode)

        saved_varnames = recv_dense_varnames + list(
            recv_sparse_varnames) + list(recv_distributed_varnames)

        remaining_vars = list(
            filter(
                ParameterServerRuntime.__exclude_vars(saved_varnames),
                main_program.list_vars()))

        fluid.io.save_vars(
            executor,
            main_program=main_program,
            dirname=dirname,
            vars=remaining_vars)

    def _ps_inference_save_persistables(self,
                                        executor,
                                        dirname,
                                        main_program=None,
                                        mode=0,
                                        **kwargs):
        """
        This function filters out all variables with `persistable==True` from the
        give `main_program` and then saves these variables to the folder `dirname`
        or file `filename`.

        The `dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set `filename` None; if you would like to save all variables in a
        single file, use `filename` to specify the file name.
        """

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save_persistables() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save_persistables() function, executor must be as Executor type"
            )

        if main_program is None:
            main_program = self.compiled_strategy.get_origin_ps_main_program()

        if isinstance(main_program, CompiledProgram):
            raise TypeError(
                "in fleet.save_persistables() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        self._save_distributed_persistables(executor, dirname, main_program,
                                            mode)

    def _ps_inference_save_inference_model(self,
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

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save_inference_model() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save_inference_model() function, executor must be as Executor type"
            )

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
                                          self.origin_main_program, None, None,
                                          export_for_deployment, True)

            model_basename = "__model__"
            model_filename = os.path.join(dirname, model_basename)

            with open(model_filename, "rb") as f:
                program_desc_str = f.read()

            program = Program.parse_from_string(program_desc_str)
            program._copy_dist_param_info_from(fluid.default_main_program())
            self._ps_inference_save_persistables(
                executor, dirname, program, mode=0)

    def _save_inference_model(self, *args, **kwargs):
        self._ps_inference_save_inference_model(*args, **kwargs)

    def _save_persistables(self, *args, **kwargs):
        self._ps_inference_save_persistables(*args, **kwargs)
