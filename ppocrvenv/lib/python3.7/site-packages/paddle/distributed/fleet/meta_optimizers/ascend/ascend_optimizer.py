# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.core as core
import numpy as np
from . import ascend_parser
from paddle.distributed import fleet
import hccl.manage.api as hccl
from collections import namedtuple

HcomGroupConfig = namedtuple('HcomGroupConfig', ['name', 'nranks', 'rank_ids'])

__all__ = []


class AscendIRParser(object):
    def __init__(self, auto_dp=False, world_rank_size=1):
        self.graph_idx = 0
        self.hcom_endpoints = {}
        self.groups_to_create = []
        self._auto_dp = auto_dp
        self._world_rank_size = world_rank_size

    def _construct_input_map(self, input_varlist):
        ret_map = {}
        ge_in_operator = []
        for id, var in enumerate(input_varlist):
            if var.is_data:  # input data
                ge_input = core.GEOperatorFactory.create_operator(
                    var.name, "Data").set_attr_int32("index", id)
                ret_map[var.name] = ge_input
                ge_in_operator.append(ge_input)
            else:  # param, learning ...
                ge_input = core.GEOperatorFactory.create_operator(var.name,
                                                                  "Variable")
                ge_input.update_output_desc("y",
                                            core.GETensorDesc(
                                                core.GEShape(var.shape),
                                                core.GEFormat.FORMAT_ND,
                                                core.GEDataType.DT_FLOAT))
                ret_map[var.name] = ge_input
        return ge_in_operator, ret_map

    def _endpoint_to_world_rank_id(self, endpoint):
        world_endpoints = fleet.worker_endpoints()
        assert endpoint in world_endpoints, "endpoint (%s) not in worker_endpoints (%s) " % (
            endpoint, fleet.world_device_ids())
        return world_endpoints.index(endpoint)

    def parse_op(self, op):
        if op.type == 'c_gen_nccl_id':
            endpoint = op.attr("endpoint")
            other_endpoints = op.attr("other_endpoints")
            rank = op.attr("rank")

            nccl_id = op.output_arg_names[0]

            # c_gen_nccl_id operator splits endpoints into local endpoint and other_endpoints
            # we should combine these together to produce world_rank_ids 
            self.hcom_endpoints[nccl_id] = other_endpoints[:]
            self.hcom_endpoints[nccl_id].insert(rank, endpoint)

            print("nccl_id (%s) registered endpoints %s" %
                  (nccl_id, self.hcom_endpoints[nccl_id]))
        elif op.type == 'c_comm_init':
            nccl_id = op.input_arg_names[0]
            nranks = op.attr("nranks")
            assert nranks == len(self.hcom_endpoints[
                nccl_id]), "nranks doesn't match endpoint count"
            rank = op.attr("rank")
            ring_id = op.attr("ring_id")

            group_name = "hcom_group_" + str(ring_id)
            global_rank_ids = [
                self._endpoint_to_world_rank_id(endpoint)
                for endpoint in self.hcom_endpoints[nccl_id]
            ]
            self.groups_to_create.append(
                HcomGroupConfig(
                    name=group_name, nranks=nranks, rank_ids=global_rank_ids))
            print("append to create group: %s, with rank_ids: %s" %
                  (group_name, global_rank_ids))
        elif op.type in ascend_parser.registerd_op:
            op_parser = self.parser_factory.create_parse(
                ascend_parser.registerd_op[op.type])
            op_parser.apply(op)
        else:
            assert False, "Op[%s] has not been registered, so we have to skip it" % (
                op.type)

    def _parse_program(self,
                       graph_name,
                       program,
                       input_varlist=[],
                       fetch_list=[]):
        begin_graph_idx = self.graph_idx
        ge_in_operator = []
        ge_out_operator = []
        self.var2geop = {}

        block = program.global_block()
        if len(block.ops) == 0:
            print("There is no ops in program %s" % (graph_name))
            return []

        graph = core.GEGraph(graph_name)

        ge_in_operator, self.var2geop = self._construct_input_map(input_varlist)

        self.parser_factory = ascend_parser.AscendParserFactory(graph,
                                                                self.var2geop)
        for i, curop in list(enumerate(block.ops)):
            self.parse_op(curop)

        # Set fetch_var for GE
        for e in fetch_list:
            name = e
            if not isinstance(e, str):
                name = e.name
            ge_out_operator.append(self.var2geop[name])

        # (Debug) If you want to print back prop vars, append/assign the varname in ge_out_operator here, such as:
        # if graph_name == "main":
        #     ge_out_operator.append(self.var2geop["reduce_sum_0.tmp_0@GRAD"])

        # Add ops that may be input of a graph, such as const.
        for varname, geop in self.var2geop.items():
            if varname.startswith("geinput"):
                ge_in_operator.append(geop)

        graph.set_inputs(ge_in_operator).set_outputs(ge_out_operator)

        # Remove ops of origin program
        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)

        input_varlist = [var for var in input_varlist if var.is_data]

        block.append_op(
            type="ascend_trigger",
            inputs={"FeedList": input_varlist},
            outputs={"FetchList": fetch_list},
            attrs={'graph_idx': self.graph_idx})
        self.graph_idx += 1
        return graph

    def parse_program(self, startup_program, main_program, input_varlist,
                      fetch_list):
        startup_graph = self._parse_program("startup", startup_program)
        main_graph = self._parse_program("main", main_program, input_varlist,
                                         fetch_list)
        if self._auto_dp and self._world_rank_size > 1:
            assert len(self.groups_to_create
                       ) == 0, "can't parse program under auto_dp mode"

            from paddle.distributed import fleet
            self.groups_to_create.append(
                HcomGroupConfig(
                    name="hcom_group_0",
                    nranks=fleet.world_size(),
                    rank_ids=[x for x in range(fleet.world_size())]))

        return startup_graph, main_graph


# AscendOptimizer is a wrapper for basic optimizer now
# We will make it part of fleet meta_optimizer in the future
class AscendOptimizer(Optimizer):
    def __init__(self, optimizer, fetch_list=[]):
        self.inner_opt = optimizer
        self.fetch_list = fetch_list
        self.ascend_instance = None

    def __del__(self):
        print("begin AscendOptimizer del")
        if self.ascend_instance is not None:
            self.ascend_instance.destroy_global_resources()
        core.ge_finalize()
        print("end AscendOptimizer del")

    def _can_apply(self):
        if not self.user_defined_strategy.ascend:
            return False
        # TODO(hutuxian): other check here
        return True

    def _disable_strategy(self, dist_strategy):
        dist_strategy.ascend = False
        dist_strategy.ascend_configs = {}

    def _get_input_varlist(self, program):
        ret_list = []
        for var in program.list_vars():
            if var.is_data or var.persistable:
                ret_list.append(var)
        return ret_list

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 auto_dp=False,
                 rank_table_file=None,
                 precision_mode="must_keep_origin_dtype"):
        minimized = None
        if self.inner_opt:
            minimized = self.inner_opt.minimize(
                loss, startup_program=startup_program)

        self.ascend_instance = core.AscendInstance()

        from paddle.distributed import fleet
        if auto_dp and fleet.world_size() > 1:
            from paddle.fluid.transpiler import ascend_transpiler
            t = ascend_transpiler.AscendTranspiler(startup_program,
                                                   loss.block.program)
            t.transpile()
            #print(loss.block.program)

        # Config about Graph Engine can be found in https://support.huaweicloud.com/
        config = {
            "ge.exec.deviceId": str(fleet.local_device_ids()),
            "ge.graphRunMode": "1",
            "ge.exec.precision_mode": precision_mode,
        }
        # if multi trainers
        if rank_table_file and fleet.world_size() > 1:
            config["ge.exec.rankTableFile"] = rank_table_file
            config["ge.exec.rankId"] = str(fleet.worker_index())
            config["ge.exec.isUseHcom"] = "1"
            config["ge.exec.deployMode"] = "0"
        print("ge_initialize config:", config)
        core.ge_initialize(config)

        # Init Session
        self.ascend_instance.init_global_resources()

        main_block = loss.block
        self.parser = AscendIRParser(
            auto_dp=auto_dp, world_rank_size=fleet.world_size())

        input_varlist = self._get_input_varlist(main_block.program)

        startup_graph, main_graph = self.parser.parse_program(
            startup_program, main_block.program, input_varlist, self.fetch_list)

        for cfg in self.parser.groups_to_create:
            print("create group (%s), nranks: %d, rank_ids: %s" %
                  (cfg.name, cfg.nranks, cfg.rank_ids))
            hccl.create_group(cfg.name, cfg.nranks, cfg.rank_ids)

        self.ascend_instance.add_ascend_subgraph(0, startup_graph)
        self.ascend_instance.add_ascend_subgraph(1, main_graph)

        return minimized
