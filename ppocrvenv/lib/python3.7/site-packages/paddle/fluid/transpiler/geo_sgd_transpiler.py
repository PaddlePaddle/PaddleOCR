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
# limitations under the License.

from __future__ import print_function
"""
Steps to transpile trainer:
1. split variable to multiple blocks, aligned by product(dim[1:]) (width).
2. create delta variable in global scope which used to send
3. add send op to send sparse ids to communicator

Steps to transpile pserver:
1. create new program for parameter server.
2. create params variables that assigned to current server instance.
3. create a sub-block in the server side program
4. append sum ops that should run on current server instance.
5. add listen_and_serv op
"""
import sys
import collections
import six
import numpy as np

from .ps_dispatcher import RoundRobin, PSDispatcher
from .. import core, framework
from ..framework import Program, default_main_program, \
    default_startup_program, Block, Parameter
from .details import wait_server_ready, VarsDistributed
from .details import delete_ops
from ..distribute_lookup_table import find_distributed_lookup_table
from .distribute_transpiler import DistributeTranspiler, DistributeTranspilerConfig, slice_variable, same_or_split_var, ServerRuntimeConfig
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode

RPC_OP_ROLE_ATTR_NAME = op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName(
)
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC


class GeoSgdTranspiler(DistributeTranspiler):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = DistributeTranspilerConfig()
        self._set_server_config()

        if self.config.split_method is None:
            self.config.split_method = RoundRobin

        assert (self.config.min_block_size >= 8192)
        assert (self.config.split_method.__bases__[0] == PSDispatcher)

    def transpile(self,
                  trainer_id,
                  program=None,
                  pservers="127.0.0.1:6174",
                  trainers=1,
                  sync_mode=False,
                  startup_program=None,
                  current_endpoint="127.0.0.1:6174"):
        if program is None:
            program = default_main_program()
        if startup_program is None:
            startup_program = default_startup_program()
        self.origin_program = program
        self.startup_program = startup_program
        self.origin_startup_program = self.startup_program.clone()

        self.trainer_num = trainers
        # geo-sgd only supply async-mode
        self.sync_mode = False
        self.trainer_id = trainer_id
        pserver_endpoints = pservers.split(",")
        self.pserver_endpoints = pserver_endpoints
        self.vars_overview = VarsDistributed()
        self.optimize_ops, self.params_grads = self._get_optimize_pass()
        ps_dispatcher = self.config.split_method(self.pserver_endpoints)
        self.param_name_to_grad_name = dict()
        self.grad_name_to_param_name = dict()
        for param_var, grad_var in self.params_grads:
            self.param_name_to_grad_name[param_var.name] = grad_var.name
            self.grad_name_to_param_name[grad_var.name] = param_var.name

        # distribute lookup table
        self.table_name = find_distributed_lookup_table(self.origin_program)
        self.has_distributed_lookup_table = self.table_name != None
        self.origin_program._distributed_lookup_table = self.table_name if self.table_name else None

        # add distributed attrs to program
        self.origin_program._is_distributed = True
        self.origin_program._endpoints = self.pserver_endpoints
        self.origin_program._ps_endpoint = current_endpoint
        self.origin_program._is_chief = self.trainer_id == 0

        # program info send to geo-sgd communicator
        self.vars_info = collections.OrderedDict()
        self.split_to_origin_mapping = collections.OrderedDict()
        self.delta_vars_list = []
        self.sparse_var_list = []
        self.sparse_var_splited_list = []

        # split and create vars, then put split vars in dicts for later use.
        # step 1. split and create vars, then put split vars in dicts for later use.
        self._init_splited_vars()

        # step 3. create send recv var (param after optimize)
        send_vars = []
        ps_dispatcher.reset()
        param_var_mapping_items = list(six.iteritems(self.param_var_mapping))
        # send_vars is the parameter which split by communicator and send to pserver,not the origin parameter
        for _, splited_vars in param_var_mapping_items:
            for _, var in enumerate(splited_vars):
                send_vars.append(var)

        recv_vars = send_vars

        ps_dispatcher.reset()
        eplist = ps_dispatcher.dispatch(recv_vars)
        for i, ep in enumerate(eplist):
            self.param_opt_ep_mapping[ep]["params"].append(recv_vars[i])
            distributed_var = self.vars_overview.get_distributed_var_by_slice(
                recv_vars[i].name)
            distributed_var.endpoint = ep
            origin_name = self.split_to_origin_mapping[recv_vars[i].name]
            self.vars_info[origin_name]["epmap"].append(ep)
        self.origin_program._parameters_on_pservers = self.vars_overview

        # send sparse id to communicator
        self.sparse_var = []
        self.sparse_tables = []
        unique_sparse_var = {}
        for op in self.origin_program.global_block().ops:
            if "is_sparse" in op.all_attrs():
                if op.type == "lookup_table":
                    op._set_attr('remote_prefetch', False)
                for input_var_name, sparse_var_name in zip(
                        op.input("Ids"), op.input("W")):
                    if sparse_var_name in self.sparse_var_list:
                        if input_var_name in unique_sparse_var:
                            if unique_sparse_var[
                                    input_var_name] == sparse_var_name:
                                continue
                        input_var = program.global_block().var(input_var_name)
                        self.sparse_var.append(input_var)
                        self.sparse_tables.append(sparse_var_name)
                        unique_sparse_var[input_var_name] = sparse_var_name

        # batch training loop end flag
        dummy_output = program.global_block().create_var(
            name=framework.generate_control_dev_var_name())
        program.global_block().append_op(
            type="send",
            inputs={"X": self.sparse_var},
            outputs={"Out": dummy_output},
            attrs={"send_varnames": self.sparse_tables})

        # add param_init flag in trainer startup program
        self.trainer_startup_program = self._get_trainer_startup_program(
            recv_vars=recv_vars, eplist=eplist)
        for delta_var in self.delta_vars_list:
            self.trainer_startup_program.global_block().create_var(
                name=delta_var.name,
                persistable=delta_var.persistable,
                dtype=delta_var.dtype,
                type=delta_var.type,
                shape=delta_var.shape)
        dummy_output = self.trainer_startup_program.global_block().create_var(
            name=framework.generate_control_dev_var_name())
        param_init = self.trainer_startup_program.global_block().create_var(
            name="param_init")
        self.trainer_startup_program.global_block().append_op(
            type="send",
            inputs={"X": [param_init]},
            outputs={"Out": dummy_output},
            attrs={"send_varnames": [param_init.name]})

    def _get_vars_info(self):
        return self.vars_info

    def get_trainer_program(self, wait_port=True):
        if wait_port:
            wait_server_ready(self.pserver_endpoints)
        return self.origin_program

    def get_pserver_programs(self, endpoint):
        pserver_prog = self.get_pserver_program(endpoint)
        self.param_grad_ep_mapping = self.param_opt_ep_mapping
        pserver_startup = self.get_startup_program(
            endpoint, pserver_program=pserver_prog)
        return pserver_prog, pserver_startup

    def get_pserver_program(self, endpoint):
        # step1
        pserver_program = Program()
        pserver_program.random_seed = self.origin_program.random_seed
        pserver_program._copy_dist_param_info_from(self.origin_program)

        # step2: Create vars to receive vars at parameter servers.
        recv_inputs = []
        for v in self.param_opt_ep_mapping[endpoint]["params"]:
            self._clone_var(pserver_program.global_block(), v)

        optimize_block = []
        param_to_block_id = []
        sparse_grad_to_param = []

        # append op to the current block
        pre_block_idx = pserver_program.num_blocks - 1
        for var in self.param_opt_ep_mapping[endpoint]["params"]:
            per_opt_block = pserver_program._create_block(pre_block_idx)
            optimize_block.append(per_opt_block)
            var_name = var.name
            pserver_block = per_opt_block.program.global_block()
            param = pserver_block.vars[var_name]

            delta_var_name = "%s.delta" % (param.name)
            if var.name in self.sparse_var_splited_list:
                delta_type = core.VarDesc.VarType.SELECTED_ROWS
                sparse_grad_to_param.append(":".join(
                    [delta_var_name, param.name]))
            else:
                delta_type = param.type
            delta_var = pserver_block.create_var(
                name=delta_var_name,
                persistable=False,
                type=delta_type,
                dtype=param.dtype,
                shape=param.shape)

            per_opt_block.append_op(
                type="sum",
                inputs={"X": [param, delta_var]},
                outputs={"Out": param})
            param_to_block_id.append(delta_var_name + ":" + str(
                per_opt_block.idx))

        attrs = {
            "optimize_blocks": optimize_block,
            "endpoint": endpoint,
            "Fanin": self.trainer_num,
            "distributed_mode": DistributedMode.GEO,
            "grad_to_block_id": param_to_block_id,
            "sparse_grad_to_param": sparse_grad_to_param,
            "rpc_get_thread_num": self.server_config._rpc_get_thread_num,
            "rpc_send_thread_num": self.server_config._rpc_send_thread_num,
            "rpc_prefetch_thread_num":
            self.server_config._rpc_prefetch_thread_num
        }

        # step5 append the listen_and_serv op
        pserver_program.global_block().append_op(
            type="listen_and_serv",
            inputs={'X': recv_inputs},
            outputs={},
            attrs=attrs)

        pserver_program._sync_with_cpp()
        # save pserver program to generate pserver side startup relatively.
        self.pserver_program = pserver_program
        return pserver_program

    def _init_splited_vars(self):
        param_list = []
        grad_list = []
        param_grad_set = set()
        # step 1. create param_list
        for p, g in self.params_grads:
            if type(p) == Parameter and p.trainable == False:
                continue
            if p.name not in param_grad_set:
                param_list.append(p)
                param_grad_set.add(p.name)
            if g.name not in param_grad_set:
                grad_list.append(g)
                param_grad_set.add(g.name)
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                self.sparse_var_list.append(p.name)

        # step 2. Slice vars into numbers of piece with block_size
        # when we slice var up into blocks, we will slice the var according to
        # pserver services' count. A pserver may have two or more listening ports.
        param_blocks = slice_variable(param_list,
                                      len(self.pserver_endpoints),
                                      self.config.min_block_size)

        # step 3. Create split param from split blocks
        # origin_param_name -> [splited_param_vars]
        # Todo: update _create_vars_from_blocklist
        self.param_var_mapping = self._create_vars_from_blocklist(
            self.origin_program, param_blocks)

        # step 4. Create mapping of endpoint -> split var to create pserver side program
        self.param_opt_ep_mapping = collections.OrderedDict()
        [
            self.param_opt_ep_mapping.update({
                ep: {
                    "params": [],
                }
            }) for ep in self.pserver_endpoints
        ]

        # step 5. Create delta var of Geo-Sgd & record vars information
        for origin_name, splited_vars in self.param_var_mapping.items():
            origin_var = self.origin_program.global_block().var(origin_name)
            self.vars_info[origin_name] = collections.OrderedDict()
            self.vars_info[origin_name]["var_names"] = []
            vars_section = self._get_splited_var_sections(splited_vars)
            self.vars_info[origin_name]["sections"] = [
                str(i) for i in vars_section
            ]
            self.vars_info[origin_name]["epmap"] = []
            self.vars_info[origin_name]["is_sparse"] = []
            # todo: add var shape(may be no need,because recv scope have)
            if origin_name in self.sparse_var_list:
                delta_type = core.VarDesc.VarType.SELECTED_ROWS
                self.vars_info[origin_name]["is_sparse"].append("True")
            else:
                delta_type = origin_var.type
                self.vars_info[origin_name]["is_sparse"].append("False")

            delta_var = self.origin_program.global_block().create_var(
                name=".".join([origin_name, "delta"]),
                persistable=False,
                dtype=origin_var.dtype,
                type=delta_type,
                shape=origin_var.shape)

            self.delta_vars_list.append(delta_var)

            for splited_var in splited_vars:
                is_slice, block_id, offset = self._get_slice_var_info(
                    splited_var)
                self.vars_overview.add_distributed_var(
                    origin_var=origin_var,
                    slice_var=splited_var,
                    block_id=block_id,
                    offset=offset,
                    is_slice=is_slice,
                    vtype="Param")
                self.split_to_origin_mapping[splited_var.name] = origin_name
                if origin_name in self.sparse_var_list:
                    self.sparse_var_splited_list.append(splited_var.name)
                self.vars_info[origin_name]["var_names"].append(
                    splited_var.name)
                if len(splited_vars) != 1:
                    self.origin_program.global_block().create_var(
                        name=".".join([splited_var.name, "delta"]),
                        persistable=False,
                        dtype=splited_var.dtype,
                        type=delta_type,
                        shape=splited_var.shape)
