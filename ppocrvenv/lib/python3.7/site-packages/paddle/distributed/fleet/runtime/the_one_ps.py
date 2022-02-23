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

import warnings

import os
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


def conv_indent(indent):
    return "".join([" "] * indent)


PSERVER_SAVE_SUFFIX = ".shard"


def parse_table_class(varname, o_main_program):
    from paddle.fluid.incubate.fleet.parameter_server.ir.public import is_distributed_sparse_op
    from paddle.fluid.incubate.fleet.parameter_server.ir.public import is_sparse_op

    for op in o_main_program.global_block().ops:
        if not is_distributed_sparse_op(op) and not is_sparse_op(op):
            continue

        param_name = op.input("W")[0]

        if param_name == varname and op.type == "lookup_table" or op.type == "lookup_table_v2":
            if op.has_attr('table_class') and op.attr("table_class") != "none":
                return op.attr('table_class')
            else:
                return "CommonSparseTable"


class Accessor:
    def __init__(self):
        self.accessor_class = ""
        self.optimizer = None
        self.feature_dim = -1
        self.embedding_dim = -1
        self.optimizer = None

    def to_string(self, indent):
        accessor_str = "{}accessor {{{}\n{}}}"
        attrs = ""
        attrs += "accessor_class: \"{}\" ".format(self.accessor_class)
        attrs += "fea_dim: {} ".format(self.feature_dim)
        attrs += "embedx_dim: {} ".format(self.embedding_dim)
        attrs += "\n"
        if self.optimizer is not None:
            attrs += self.optimizer.to_string(indent)
        return accessor_str.format(
            conv_indent(indent), attrs, conv_indent(indent))


class CommonAccessor:
    def __init__(self):
        self.accessor_class = ""
        self.table_name = None
        self.entry = None
        self.attrs = []
        self.params = []
        self.dims = []
        self.trainer_num = 0
        self.sync = "false"
        self.initializers = []
        self.opt_input_map = {}
        self.opt_attr_map = {}
        self.opt_init_map = {}
        self.define_optimize_map()

    def define_optimize_map(self):
        opt_input_map = {}
        opt_input_map["sgd"] = [("Param", None), ("LearningRate", 1)]
        opt_input_map["adam"] = [("Param", None), ("Moment1", None),
                                 ("Moment2", None), ("Beta1Pow", 1),
                                 ("Beta2Pow", 1), ("LearningRate", 1)]
        opt_input_map["sum"] = [("Param", None)]
        opt_input_map["naive_adagrad"] = [("Param", None), ("G2Sum", 1),
                                          ("LearningRate", 1)]

        opt_attr_map = {}
        opt_attr_map["sgd"] = []
        opt_attr_map["sum"] = []
        opt_attr_map["naive_adagrad"] = []
        opt_attr_map["adam"] = [("beta1", "f"), ("beta2", "f"),
                                ("epsilon", "f")]

        opt_init_map = {}
        opt_init_map["gaussian_random"] = ["seed", "mean", "std"]
        opt_init_map["fill_constant"] = ["value"]
        opt_init_map["uniform_random"] = ["seed", "min", "max"]
        opt_init_map["truncated_gaussian_random"] = ["seed", "mean", "std"]

        self.opt_attr_map = opt_attr_map
        self.opt_input_map = opt_input_map
        self.opt_init_map = opt_init_map

    def parse_entry(self, varname, o_main_program):
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import is_distributed_sparse_op
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import is_sparse_op

        for op in o_main_program.global_block().ops:
            if not is_distributed_sparse_op(op) and not is_sparse_op(op):
                continue

            param_name = op.input("W")[0]

            if param_name == varname and op.type == "lookup_table":
                self.entry = op.attr('entry')
                break

            if param_name == varname and op.type == "lookup_table_v2":
                self.entry = "none"
                break

    def get_shard(self, total_dim, shard_num, pserver_id):
        # remainder = total_dim % shard_num
        blocksize = int(total_dim / shard_num + 1)

        if blocksize * (pserver_id + 1) <= total_dim:
            return blocksize
        else:
            if blocksize * pserver_id < total_dim:
                return total_dim - blocksize * pserver_id
            else:
                return 0

    def get_initializer_attr(self, value_name, o_startup_program):
        l_in = "&"
        attr_str = ""

        origin_var_name = value_name
        for op in o_startup_program.global_block().ops:
            if op.type in self.opt_init_map.keys(
            ) and origin_var_name == op.output("Out")[0]:
                init_attr = [op.type]
                for attr in self.opt_init_map[op.type]:
                    init_attr.append(str(op.attr(attr)))
                attr_str = l_in.join(init_attr)
                break
        return attr_str

    def parse_by_optimizer(self, grad_name, is_sparse, total_dims,
                           compiled_strategy):
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops
        param_name = compiled_strategy.grad_name_to_param_name[grad_name]
        main_program, startup_program = compiled_strategy.get_origin_programs()
        pserver_id = compiled_strategy.get_role_id()
        pserver_num = len(compiled_strategy.get_ps_endpoints())
        optimizer_ops = _get_optimize_ops(main_program)
        oop = None

        for op in optimizer_ops:
            if ("Param" in op.input_names) and (
                    op.input("Param")[0] == param_name):
                oop = op
                break

        if oop is None:
            raise ValueError("can not find optimizer for {}".format(grad_name))

        params = []
        dims = []
        attrs = []
        initializers = []

        self.trainer_num = compiled_strategy.get_trainers()

        if compiled_strategy.is_geo_mode():
            param_varnames = self.opt_input_map["sum"]
            attr_varnames = self.opt_attr_map["sum"]
            self.accessor_class = "sum"
        elif compiled_strategy.use_ps_gpu and is_sparse:
            param_varnames = self.opt_input_map["naive_adagrad"]
            attr_varnames = self.opt_attr_map["naive_adagrad"]
            self.accessor_class = "sgd"
        else:
            param_varnames = self.opt_input_map[oop.type]
            attr_varnames = self.opt_attr_map[oop.type]
            self.accessor_class = oop.type

        for (formal_name, shape) in param_varnames:
            params.append(formal_name)
            if formal_name == "G2Sum":
                dims.append(1)
                initializer = "fill_constant&0"
                initializers.append(initializer)
            else:
                param = main_program.global_block().vars[oop.input(formal_name)[
                    0]]
                if formal_name == "LearningRate" and param.name != "learning_rate_0":
                    warnings.warn("will support decay soon")
                    param = main_program.global_block().vars["learning_rate_0"]

                if shape is None:
                    if is_sparse:
                        shape = total_dims
                    else:
                        shape = self.get_shard(total_dims, pserver_num,
                                               pserver_id)
                dims.append(shape)

                initializer = self.get_initializer_attr(param.name,
                                                        startup_program)
                initializers.append(initializer)

        for (attr_varname, type_) in attr_varnames:
            value = oop.attr(attr_varname)
            attrs.append("&".join([attr_varname, type_, str(value)]))

        self.params = params
        self.dims = dims
        self.initializers = initializers
        self.attrs = attrs

    def to_string(self, indent):
        accessor_str = "{}common {{{}\n{}}}"
        attrs = ""
        attrs += "name: \"{}\" ".format(self.accessor_class)

        if self.table_name:
            attrs += "table_name: \"{}\" ".format(self.table_name)

        if self.entry:
            attrs += "entry: \"{}\" ".format(self.entry)
        attrs += "trainer_num: {} ".format(self.trainer_num)
        attrs += "sync: {} ".format(self.sync)

        for param in self.params:
            attrs += "params: \"{}\" ".format(param)

        for dim in self.dims:
            attrs += "dims: {} ".format(dim)

        for initializer in self.initializers:
            attrs += "initializers: \"{}\" ".format(initializer)

        attrs += "\n"
        return accessor_str.format(
            conv_indent(indent), attrs, conv_indent(indent))


class Tensor:
    def __init__(self):
        self.main_program_id = None
        self.startup_program_id = None
        self.feed_var_name = None
        self.fetch_var_name = None
        self.tensor_table_class = False

    def to_string(self, indent):
        program_str = "{}tensor {{{}\n{}}}"
        attrs = ""
        attrs += "feed_var_name: \"{}\" ".format(str(self.feed_var_name))
        attrs += "fetch_var_name: \"{}\" ".format(str(self.fetch_var_name))
        attrs += "startup_program_id: {} ".format(str(self.startup_program_id))
        attrs += "main_program_id: {} ".format(str(self.main_program_id))
        attrs += "tensor_table_class: \"{}\" ".format(
            str(self.tensor_table_class))
        attrs += "\n"
        return program_str.format(
            conv_indent(indent), attrs, conv_indent(indent))


class Table:
    def __init__(self):
        self.id = -1
        self.table_class = None
        self.shard_num = -1
        self.type = None
        self.accessor = None
        self.common = None
        self.tensor = None

    def to_string(self, indent):
        table_str = "{}downpour_table_param {{{}\n{}}}"

        attrs = ""
        attrs += "table_id: {} ".format(self.id)
        attrs += "table_class: \"{}\" ".format(self.table_class)
        attrs += "shard_num: {} ".format(self.shard_num)
        attrs += "type: {}".format(self.type)
        attrs += "\n"
        indent += 2

        if self.accessor is not None:
            attrs += self.accessor.to_string(indent)
            attrs += "\n"

        if self.tensor is not None:
            attrs += self.tensor.to_string(indent)
            attrs += "\n"

        if self.common is not None:
            attrs += self.common.to_string(indent)
            attrs += "\n"

        return table_str.format(conv_indent(indent), attrs, conv_indent(indent))


class Service:
    def __init__(self):
        self.server_class = "BrpcPsServer"
        self.client_class = "BrpcPsClient"
        self.service_class = "BrpcPsService"
        self.start_server_port = 0
        self.server_thread_num = 12

    def to_string(self, indent):
        service_str = "{}service_param {{{}\n{}}}"

        attrs = ""
        attrs += "server_class: \"{}\" ".format(self.server_class)
        attrs += "client_class: \"{}\" ".format(self.client_class)
        attrs += "service_class: \"{}\" ".format(self.service_class)
        attrs += "start_server_port: {} ".format(self.start_server_port)
        attrs += "server_thread_num: {} ".format(self.server_thread_num)

        return service_str.format(
            conv_indent(indent), attrs, conv_indent(indent))


class DownpourServer:
    def __init__(self):
        self.service = None
        self.tables = []

    def set_service_param(self, service):
        self.service = service

    def append_tables(self, table):
        if not isinstance(table, Table):
            raise ValueError("only support instance Table")
        self.tables.append(table)

    def to_string(self, indent):
        server_str = "{}downpour_server_param {{{}\n{}}}"

        table_strs = ""
        indent += 2

        table_strs += "\n"
        table_strs += self.service.to_string(indent)

        for table in self.tables:
            table_strs += "\n"
            table_strs += table.to_string(indent)
        return server_str.format(
            conv_indent(indent), table_strs, conv_indent(indent))


class Server:
    def __init__(self):
        self.servers = []

    def add_server(self, server):
        if not isinstance(server, DownpourServer):
            raise ValueError("only support instance DownpourServer")
        self.servers.append(server)

    def __str__(self):
        server_str = "server_param {{{}\n}}"
        indent = 2
        servers_str = ""
        for server in self.servers:
            servers_str += "\n"
            servers_str += server.to_string(indent)

        return server_str.format(servers_str)


class DownpourWorker:
    def __init__(self):
        self.tables = []

    def append_tables(self, table):
        if not isinstance(table, Table):
            raise ValueError("only support instance Table")
        self.tables.append(table)

    def to_string(self, indent):
        worker_str = "{}downpour_worker_param {{{}\n{}}}"
        table_strs = ""
        indent += 2
        for table in self.tables:
            table_strs += "\n"
            table_strs += table.to_string(indent)

        return worker_str.format(
            conv_indent(indent), table_strs, conv_indent(indent))


class Worker:
    def __init__(self):
        self.workers = []

    def add_worker(self, worker):
        if not isinstance(worker, DownpourWorker):
            raise ValueError("only support instance DownpourWorker")
        self.workers.append(worker)

    def __str__(self):
        worker_str = "worker_param {{{}\n}}"
        indent = 2
        workers_str = ""
        for worker in self.workers:
            workers_str += "\n"
            workers_str += worker.to_string(indent)

        return worker_str.format(workers_str)


class TheOnePSRuntime(RuntimeBase):
    def __init__(self):
        super(TheOnePSRuntime, self).__init__()
        self._communicator = None
        self._server = None
        self._worker = fluid.core.DistFleetWrapper()
        self._server_sub_program = []
        self._heter_client = None

    def _set_basic_info(self, context):
        self.context = context
        self.role_maker = context["role_maker"]
        self.origin_main_program = context["origin_main_program"]
        self.origin_startup_program = context["origin_startup_program"]
        self.async_strategy = self._get_distributed_strategy()
        self.compiled_strategy = self.build_compiled_startegy()

    def _get_distributed_strategy(self):
        strategy = None

        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import \
            StrategyFactory

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

        if dist_strategy.a_sync_configs["use_ps_gpu"]:
            strategy.use_ps_gpu = True
        return strategy

    def build_compiled_startegy(self):
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import CompileTimeStrategy

        compiled_config = CompileTimeStrategy(
            self.origin_main_program, self.origin_main_program,
            self.async_strategy, self.role_maker)
        if self.async_strategy.use_ps_gpu:
            compiled_config.use_ps_gpu = True
        return compiled_config

    def _init_worker(self):
        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import \
            SyncStrategy, GeoStrategy

        is_sync = self.compiled_strategy.is_sync_mode()
        worker = self._get_fleet_proto(is_server=False, is_sync=is_sync)
        server = self._get_fleet_proto(is_server=True, is_sync=is_sync)

        dist_strategy = self.context["valid_strategy"]
        use_ps_gpu = dist_strategy.a_sync_configs["use_ps_gpu"]
        if use_ps_gpu:
            main_program = self.context['loss'].block.program
            if not main_program._fleet_opt:
                main_program._fleet_opt = {}
            main_program._fleet_opt["use_ps_gpu"] = True
            gpus_env = os.getenv("FLAGS_selected_gpus")
            main_program._fleet_opt[
                "worker_places"] = [int(s) for s in gpus_env.split(",")]

        def sync_strategy_envs():
            kwargs = {}
            kwargs[
                "pserver_endpoints"] = self.role_maker._get_pserver_endpoints()
            kwargs["trainer_id"] = self.role_maker._worker_index()
            return kwargs

        proto_txt = str(worker) + "\n" + str(server)

        debug = bool(int(os.getenv("PSERVER_DEBUG", "0")))

        if debug:
            print("worker: \n{}".format(proto_txt))

        endpoints = self.compiled_strategy.get_ps_endpoints()

        string_hosts = []
        for idx, ep in enumerate(endpoints):
            host, port = ep.split(":")
            pshost = fluid.core.PSHost(host, int(port), idx)
            string_hosts.append(pshost.serialize_to_string())

        dense_map = self.compiled_strategy.get_the_one_recv_context(
            split_dense_table=self.role_maker._is_heter_parameter_server_mode)
        send_ctx = self.compiled_strategy.get_the_one_send_context(
            split_dense_table=self.role_maker._is_heter_parameter_server_mode,
            use_origin_program=self.role_maker._is_heter_parameter_server_mode,
            ep_list=endpoints)
        trainer_config = self.async_strategy.get_trainer_runtime_config()

        debug = bool(int(os.getenv("PSERVER_DEBUG", "0")))

        if debug:
            print("worker: \n{}".format(proto_txt))
            print("communicator send_ctx:")
            for key in send_ctx:
                print("{}: {}".format(key, send_ctx[key]))
            for key in dense_map:
                print("{}: {}".format(key, dense_map[key]))

        kwargs = {}
        kwargs['need_global_step'] = "0"
        kwargs["trainer_id"] = self.role_maker._role_id()
        kwargs["trainers"] = self.role_maker._worker_num()
        #if self.role_maker._is_heter_worker():
        #    kwargs["trainer_id"] += kwargs["trainers"]

        for table in server.servers[0].tables:
            if table.table_class == "BarrierTable":
                kwargs["barrier_table_id"] = table.id
                break

        if isinstance(self.async_strategy, SyncStrategy):
            sync_kwargs = sync_strategy_envs()
            kwargs.update(sync_kwargs)

        from paddle.fluid.communicator import Communicator, HeterClient
        self._communicator = Communicator(
            trainer_config.mode, kwargs,
            trainer_config.get_communicator_flags())
        self._communicator.init_with_ctx(send_ctx, dense_map, proto_txt,
                                         string_hosts, fluid.global_scope())

        dist_strategy = self.context["valid_strategy"]

        is_test = bool(int(os.getenv("TEST_MODE", "0")))

        if self.role_maker._is_first_worker(
        ) and self.role_maker._is_heter_parameter_server_mode:
            # for ps-heter mode load all parameters on first_worker
            init_params = self.compiled_strategy.get_the_one_recv_context(
                split_dense_table=True, use_origin_program=True)
        else:
            init_params = dense_map

        if not is_test:
            self._communicator.init_params(init_params)

        if not self._communicator.is_running():
            self._communicator.start()
        else:
            warnings.warn("communicator has been initialized, skip")

        launch_barrier = dist_strategy.a_sync_configs["launch_barrier"]
        launch_barrier_flag = int(os.getenv("FLAGS_LAUNCH_BARRIER", "1"))
        if launch_barrier and launch_barrier_flag:
            # for trainer wait server ready
            wait_server_ready(self.role_maker._get_pserver_endpoints())
            if self.role_maker._is_heter_parameter_server_mode and self.role_maker._get_next_trainers(
            ) != []:
                wait_server_ready(self.role_maker._get_next_trainers())
            if self.role_maker._is_heter_parameter_server_mode:
                previous_trainers = []
                if self.role_maker._get_previous_trainers() != []:
                    previous_trainers = self.role_maker._get_previous_trainers()
                next_trainers = []
                if self.role_maker._get_next_trainers() != []:
                    next_trainers = self.role_maker._get_next_trainers()
                self._heter_client = HeterClient(next_trainers,
                                                 previous_trainers,
                                                 self.role_maker._role_id())

    def _push_sparse_param(self,
                           var_name,
                           table_id=-1,
                           scope=fluid.global_scope()):
        self._communicator.push_sparse_param(var_name, table_id, scope)

    def _get_executor(self):
        executor = fluid.Executor(fluid.CPUPlace())
        if self.role_maker._is_heter_parameter_server_mode:
            if self.role_maker._is_heter_worker():
                heter_device_type = self.role_maker._heter_device_type().upper()
                if heter_device_type not in ["GPU", "XPU", "CPU"]:
                    raise ValueError("Heter Worker Not Support Device {}".
                                     format(device_type))
                if heter_device_type == "GPU":
                    executor = Executor(
                        fluid.CUDAPlace(
                            int(os.getenv("FLAGS_selected_gpus", "0"))))
                elif heter_device_type == "XPU":
                    executor = Executor(
                        fluid.XPUPlace(
                            int(os.getenv("FLAGS_selected_xpus", "0"))))
        return executor

    def _get_fleet_proto(self, is_server, is_sync):
        def _build_merge_accessor(ctx):
            accessor = Accessor()
            accessor.accessor_class = "CommMergeAccessor"
            accessor.optimizer = None

            if ctx.is_sparse():
                accessor.feature_dim = ctx.sections()[0]
                accessor.embedding_dim = ctx.sections()[1]
            else:
                accessor.feature_dim = ctx.sections()[0]
                accessor.embedding_dim = 1

            return accessor

        def _build_barrier_table(idx):
            table = Table()
            table.id = idx
            table.type = "PS_OTHER_TABLE"
            table.table_class = "BarrierTable"
            table.shard_num = 256

            accessor = Accessor()
            accessor.accessor_class = "CommMergeAccessor"
            accessor.optimizer = None
            accessor.feature_dim = 0
            accessor.embedding_dim = 0
            table.accessor = accessor

            common = CommonAccessor()
            common.table_name = "barrier_table"
            trainer_num = self.compiled_strategy.get_trainers()
            if self.role_maker._is_heter_parameter_server_mode:
                trainer_num += len(self.role_maker._get_heter_worker_endpoints(
                ))
            common.trainer_num = trainer_num
            common.attrs = ""
            common.dims = []
            common.params = []
            table.common = common
            return table

        def _build_tensor_table(idx, tensor_dict):
            table = Table()
            table.id = idx
            table.type = "PS_OTHER_TABLE"
            table.table_class = tensor_dict["tensor_table_class"]
            table.shard_num = 256

            accessor = Accessor()
            accessor.accessor_class = "CommMergeAccessor"
            accessor.optimizer = None
            accessor.feature_dim = 0
            accessor.embedding_dim = 0
            table.accessor = accessor

            common = CommonAccessor()
            common.table_name = tensor_dict["feed_var_name"]
            common.trainer_num = self.compiled_strategy.get_trainers()
            common.attrs = ""
            common.dims = []
            common.params = []
            table.common = common

            tensor = Tensor()
            tensor.main_program_id = tensor_dict["main_program_id"]
            tensor.startup_program_id = tensor_dict["startup_program_id"]
            tensor.feed_var_name = tensor_dict["feed_var_name"]
            tensor.fetch_var_name = tensor_dict["fetch_var_name"]
            tensor.tensor_table_class = tensor_dict["tensor_table_class"]
            table.tensor = tensor

            return table

        def _add_tensor_table(tables):
            tensor_table_dict = self.compiled_strategy.get_tensor_table_dict()
            program_idx = 0
            for table_name in tensor_table_dict:
                if tensor_table_dict[table_name]["startup_program"] != None:
                    tensor_table_dict[table_name][
                        "startup_program_id"] = program_idx
                    self._server_sub_program.append(tensor_table_dict[
                        table_name]["startup_program"].desc)
                    program_idx += 1
                if tensor_table_dict[table_name]["main_program"] != None:
                    tensor_table_dict[table_name][
                        "main_program_id"] = program_idx
                    self._server_sub_program.append(tensor_table_dict[
                        table_name]["main_program"].desc)
                    program_idx += 1
                # Todo: Hard code for lr_decay table apply table id
                new_table = _build_tensor_table(
                    len(tables), tensor_table_dict[table_name])
                tables.append(new_table)
            return tables

        def _get_tables():
            send_ctx = self.compiled_strategy.get_the_one_send_context(
                use_origin_program=True,
                split_dense_table=self.role_maker.
                _is_heter_parameter_server_mode)

            tables = []
            for idx, (name, ctx) in enumerate(send_ctx.items()):
                if ctx.is_tensor_table() or len(ctx.origin_varnames()) < 1:
                    continue

                table = Table()
                table.id = ctx.table_id()
                common = CommonAccessor()

                if ctx.is_sparse():
                    table.type = "PS_SPARSE_TABLE"
                    table.shard_num = 256

                    common.table_name = self.compiled_strategy.grad_name_to_param_name[
                        ctx.origin_varnames()[0]]

                    if self.compiled_strategy.is_geo_mode():
                        table.table_class = "SparseGeoTable"
                    else:
                        table.table_class = parse_table_class(
                            common.table_name, self.origin_main_program)

                else:
                    table.type = "PS_DENSE_TABLE"
                    table.table_class = "CommonDenseTable"
                    table.shard_num = 256
                    common.table_name = "MergedDense"

                common.parse_by_optimizer(ctx.origin_varnames()[0],
                                          ctx.is_sparse(),
                                          ctx.sections()[1] if ctx.is_sparse()
                                          else ctx.sections()[0],
                                          self.compiled_strategy)

                if ctx.is_sparse():
                    common.parse_entry(common.table_name,
                                       self.origin_main_program)

                if is_sync:
                    common.sync = "true"
                else:
                    common.sync = "false"

                table.common = common

                accessor = _build_merge_accessor(ctx)
                table.accessor = accessor
                tables.append(table)

            tensor_table_dict = self.compiled_strategy.get_tensor_table_dict()
            if len(tensor_table_dict) > 0:
                tables = _add_tensor_table(tables)
            else:
                empty_porgram = Program()
                self._server_sub_program.append(empty_porgram.desc)

            barrier_table = _build_barrier_table(len(tables))
            tables.append(barrier_table)
            return tables

        if is_server:
            server = Server()
            downpour_server = DownpourServer()

            service = Service()
            dist_strategy = self.context["valid_strategy"]
            use_ps_gpu = dist_strategy.a_sync_configs["use_ps_gpu"]
            if use_ps_gpu:
                service.server_class = "PsLocalServer"
                service.client_class = "PsLocalClient"
            downpour_server.set_service_param(service)

            tables = _get_tables()
            downpour_server.tables = tables
            server.add_server(downpour_server)
            return server
        else:
            worker = Worker()
            downpour_worker = DownpourWorker()

            tables = _get_tables()
            downpour_worker.tables = tables
            worker.add_worker(downpour_worker)
            return worker

    def _init_server(self, dirname=None, var_names=None, **kwargs):
        role_id = self.compiled_strategy.get_role_id()
        endpoints = self.compiled_strategy.get_ps_endpoints()
        is_sync = self.compiled_strategy.is_sync_mode()
        trainers = self.compiled_strategy.get_trainers()
        if self.role_maker._is_heter_parameter_server_mode:
            trainers += len(self.role_maker._get_heter_worker_endpoints())
        server = self._get_fleet_proto(is_server=True, is_sync=is_sync)
        proto_txt = str(server)

        debug = bool(int(os.getenv("PSERVER_DEBUG", "0")))
        if debug:
            print("server: \n{}".format(proto_txt))

        string_hosts = []
        for idx, ep in enumerate(endpoints):
            host, port = ep.split(":")
            pshost = fluid.core.PSHost(host, int(port), idx)
            string_hosts.append(pshost.serialize_to_string())

        self._server = fluid.core.DistFleetWrapper()
        self._server.init_server(proto_txt, string_hosts, role_id, trainers,
                                 self._server_sub_program)

        from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames

        dist_varnames = get_sparse_tablenames(self.origin_main_program, True)
        sparse_varnames = get_sparse_tablenames(self.origin_main_program, False)

        distributed_varnames = dist_varnames + sparse_varnames

        if var_names is None:
            load_varnames = distributed_varnames
        else:
            for var_name in var_names:
                if var_name not in distributed_varnames:
                    raise ValueError(
                        "fleet.init server can only load sparse variables in {}".
                        format(distributed_varnames))
            load_varnames = var_names

        if dirname is None or not load_varnames:
            return

        sparse_table_maps = {}
        for table in server.servers[0].tables:
            if table.type == "PS_SPARSE_TABLE" and table.common is not None:
                sparse_table_maps[table.common.table_name] = table.id

        dirname = os.path.normpath(dirname)
        pserver_id = self.role_maker._role_id()

        for var_name in load_varnames:
            table_id = sparse_table_maps[var_name]
            # path = os.path.join(dirname, var_name + PSERVER_SAVE_SUFFIX,
            #                     "{}.block{}.txt".format(var_name, pserver_id))
            # meta = os.path.join(dirname, var_name + PSERVER_SAVE_SUFFIX,
            #                     "{}.block{}.meta".format(var_name, pserver_id))
            self._server.load_sparse(dirname, "0", table_id)

    def _run_server(self):
        ep = self.compiled_strategy.get_ps_endpoint()
        host, port = ep.split(":")
        self._server.run_server(host, int(port))

    def _stop_worker(self):
        self._communicator.stop()
        if self.role_maker._is_heter_parameter_server_mode:
            assert self._heter_client != None, "heter client should not be None in heterps mode"
            self._heter_client.stop()
        #executor = self._get_executor()
        #executor.close()

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

    def _save_sparse_params(self, executor, dirname, context, main_program,
                            mode):
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames
        distributed_varnames = get_sparse_tablenames(
            self.compiled_strategy.origin_main_program, True)
        values = []
        for id, names in context.items():
            if names[0] not in distributed_varnames:
                # only save sparse param to local
                self._worker.recv_and_save_model(id, dirname)
            # save sparse & distributed param on server
            self._worker.save_one_model(id, dirname, mode)
            values.extend(names)
        return values

    def _save_distributed_persistables(self,
                                       executor,
                                       dirname,
                                       main_program,
                                       mode=0):

        denses = self.compiled_strategy.get_the_one_recv_context(
            is_dense=True,
            split_dense_table=self.role_maker._is_heter_parameter_server_mode,
            use_origin_program=True)
        sparses = self.compiled_strategy.get_the_one_recv_context(
            is_dense=False,
            split_dense_table=self.role_maker._is_heter_parameter_server_mode,
            use_origin_program=True)

        sparse_varnames = self._save_sparse_params(executor, dirname, sparses,
                                                   main_program, mode)

        recv_dense_varnames = []
        for id, names in denses.items():
            recv_dense_varnames.extend(names)

        saved_varnames = sparse_varnames

        remaining_vars = list(
            filter(
                TheOnePSRuntime.__exclude_vars(saved_varnames),
                main_program.list_vars()))

        self._communicator.pull_dense(denses)

        import paddle
        for var in remaining_vars:
            # if var.name not in recv_dense_varnames:
            #     continue
            tensor = var.get_value()
            paddle.save(
                tensor, os.path.join(dirname, var.name), use_binary_format=True)

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
                "in fleet.save() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save() function, executor must be as Executor type")

        if main_program is None:
            main_program = self.compiled_strategy.get_origin_ps_main_program()

        if isinstance(main_program, CompiledProgram):
            raise TypeError(
                "in fleet.save() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        # Todo(MrChengmo): Save optimizer status
        self._save_distributed_persistables(executor, dirname, main_program,
                                            mode)

    def _ps_inference_save_inference_model(self,
                                           executor,
                                           dirname,
                                           feeded_var_names,
                                           target_vars,
                                           main_program=None,
                                           export_for_deployment=True,
                                           mode=0):
        """
        Prune the given `main_program` to build a new program especially for inference,
        and then save it and all related parameters to given `dirname` by the `executor`.
        """

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save() function, executor must be as Executor type")

        import paddle
        program = self.origin_main_program if main_program is None else main_program

        if isinstance(program, CompiledProgram):
            raise TypeError(
                "in fleet.save() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        feed_vars = [
            program.global_block().var(name) for name in feeded_var_names
        ]

        infer_program = paddle.static.normalize_program(program, feed_vars,
                                                        target_vars)

        infer_program._copy_dist_param_info_from(program)

        model_basename = "__model__"
        model_basename = os.path.join(dirname, model_basename)
        paddle.save(infer_program, model_basename)

        self._ps_inference_save_persistables(executor, dirname, infer_program,
                                             mode)

    def _save_inference_model(self, *args, **kwargs):
        self._ps_inference_save_inference_model(*args, **kwargs)

    def _save_persistables(self, *args, **kwargs):
        self._ps_inference_save_persistables(*args, **kwargs)

    def _load_sparse_params(self, dirname, context, main_program, mode):
        from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames
        distributed_varnames = get_sparse_tablenames(
            self.compiled_strategy.origin_main_program, True)
        values = []
        for id, names in context.items():
            if names[0] not in distributed_varnames:
                # TODO: only load sparse param from local
                warnings.warn("varname is not in distributed_varnames, pass")
            # load sparse & distributed param on server
            self._worker.load_one_table(id, dirname, mode)
            values.extend(names)
        return values

    def _load_distributed_persistables(self, dirname, main_program=None,
                                       mode=0):
        if main_program is None:
            main_program = self.compiled_strategy.get_origin_ps_main_program()

        if isinstance(main_program, CompiledProgram):
            raise TypeError(
                "in fleet.save() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        denses = self.compiled_strategy.get_the_one_recv_context(
            is_dense=True,
            split_dense_table=self.role_maker._is_heter_parameter_server_mode,
            use_origin_program=True)
        sparses = self.compiled_strategy.get_the_one_recv_context(
            is_dense=False,
            split_dense_table=self.role_maker._is_heter_parameter_server_mode,
            use_origin_program=True)

        sparse_varnames = self._load_sparse_params(dirname, sparses,
                                                   main_program, mode)

        recv_dense_varnames = []
        for id, names in denses.items():
            recv_dense_varnames.extend(names)

        loaded_varnames = sparse_varnames

        remaining_vars = list(
            filter(
                TheOnePSRuntime.__exclude_vars(loaded_varnames),
                main_program.list_vars()))

        import paddle
        for var in remaining_vars:
            if var.name not in recv_dense_varnames:
                continue
            tensor = paddle.load(os.path.join(dirname, var.name))
            var.set_value(tensor)

        self._communicator.init_params(denses)

    def load_model(self, path, mode):
        self._load_distributed_persistables(path, mode=mode)

    def _shrink(self, threshold):
        import paddle.distributed.fleet as fleet
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            sparses = self.compiled_strategy.get_the_one_recv_context(
                is_dense=False,
                split_dense_table=self.role_maker.
                _is_heter_parameter_server_mode,
                use_origin_program=True)

            for id, names in sparses.items():
                self._worker.shrink_sparse_table(id, threshold)
        fleet.util.barrier()
