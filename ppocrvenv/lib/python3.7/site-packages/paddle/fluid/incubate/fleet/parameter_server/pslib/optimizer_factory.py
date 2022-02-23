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
"""Optimizer Factory."""

__all__ = ["DistributedAdam", "FLEET_GLOBAL_DICT"]
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_inputs
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_outputs
from google.protobuf import text_format
from collections import OrderedDict
import copy
from .node import DownpourWorker, DownpourServer
from . import ps_pb2 as pslib
import os
import logging

OpRole = core.op_proto_and_checker_maker.OpRole
# this dict is for store info about pull/push sparse ops.
FLEET_GLOBAL_DICT = {
    # global settings
    "enable": False,
    "emb_to_table": {},
    "emb_to_accessor": {},
    "emb_to_size": {},
    # current embedding settings
    "cur_sparse_id": 0,
    "cur_accessor": "",
    "click_name": "",
    "scale_sparse_grad": None,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class DistributedOptimizerImplBase(object):
    """
    DistributedOptimizerImplBase
    base class of optimizers
    """

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._learning_rate = optimizer._learning_rate
        self._regularization = optimizer.regularization

    def minimize(self,
                 losses,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Args:
            losses(Variable): loss variable defined by user
            startup_program(Program): startup program that defined by user
            parameter_list(str list): parameter names defined by users
            no_grad_set(set): a set of variables that is defined by users
                so that these variables do not need gradient computation
        """
        pass


class DistributedAdam(DistributedOptimizerImplBase):
    """
    DistributedAdam
    adam optimizer in distributed training
    """

    def __init__(self, optimizer):
        # todo(guru4elephant): add more optimizers here as argument
        # todo(guru4elephant): make learning_rate as a variable
        super(DistributedAdam, self).__init__(optimizer)
        self._window = 1
        self.type = "downpour"
        self.data_norm_name = [
            ".batch_size", ".batch_square_sum", ".batch_sum",
            ".batch_size@GRAD", ".batch_square_sum@GRAD", ".batch_sum@GRAD"
        ]
        self.supported_embedding_types = [
            "lookup_table", "pull_sparse", "pull_sparse_v2", "pull_box_sparse"
        ]
        self.supported_embedding_grad_types = [
            "lookup_table_grad", "push_sparse", "push_sparse_v2"
        ]
        op_maker = core.op_proto_and_checker_maker
        self.op_role_key = op_maker.kOpRoleAttrName()

    def _find_distributed_lookup_table_inputs(self, program, table_names):
        """
        Find input variable of distribute lookup table in program.
        We could support multi-distribute table now.
        Args:
            program(Program): given program, locate distributed lookup table
            table_name(str): given table names that is found beforehand
        Returns:
            inputs
        """
        local_vars = program.current_block().vars
        inputs_dict = dict()
        for table_name in table_names:
            inputs_dict[table_name] = []

        for op in program.global_block().ops:
            if op.type in self.supported_embedding_types:
                if op.input("W")[0] in table_names:
                    inputs_dict[op.input("W")[0]].extend(
                        [local_vars[name] for name in op.input("Ids")])
        return inputs_dict

    def _find_distributed_lookup_table_outputs(self, program, table_names):
        """
        Find output variable of distribute lookup table in program.
        We could support multi-distribute table now.
        Args:
            programs(Program): given program, locate distributed lookup table
            table_name(str): given table name that is found beforehand
        Returns:
            outputs
        """
        local_vars = program.current_block().vars
        outputs_dict = dict()
        for table_name in table_names:
            outputs_dict[table_name] = []

        for op in program.global_block().ops:
            if op.type in self.supported_embedding_types:
                if op.input("W")[0] in table_names:
                    outputs_dict[op.input("W")[0]].extend(
                        [local_vars[name] for name in op.output("Out")])
        return outputs_dict

    def _find_distributed_lookup_table_grads(self, program, table_names):
        local_vars = program.current_block().vars
        grads_dict = dict()
        for table_name in table_names:
            grads_dict[table_name] = []

        for op in program.global_block().ops:
            if op.type in self.supported_embedding_grad_types:
                if op.input("W")[0] in table_names:
                    grads_dict[op.input("W")[0]].extend(
                        [local_vars[name] for name in op.input("Out@GRAD")])
        return grads_dict

    def _is_optimizer_op(self, op):
        return self.op_role_key in op.attr_names and \
                int(op.all_attrs()[self.op_role_key]) & int(OpRole.Optimize)

    def _remove_optimize_op_for_embedding(self, loss, table_name):
        """
        find multi-sparse-table
        """
        table_name = [name + "@GRAD" for name in table_name]
        need_remove_op_index = []
        block = loss.block.program.global_block()
        for ids, op in list(enumerate(block.ops)):
            if self._is_optimizer_op(op):
                if op.input("Grad")[0] in table_name:
                    need_remove_op_index.append(ids)

        need_remove_op_index.sort(reverse=True)
        for index in need_remove_op_index:
            block._remove_op(index)

    def _find_multi_distributed_lookup_table(self, losses):
        """
        find multi-sparse-table
        """
        table_names = set()
        cnt = 0
        tmp_list = []
        ret_list = []
        for loss in losses:
            for op in loss.block.program.global_block().ops:
                if op.type in self.supported_embedding_types:
                    if op.attr('is_distributed') is True:
                        table_name = op.input("W")[0]
                        if table_name not in table_names:
                            table_names.add(table_name)
                            tmp_list.append([table_name, cnt])
                            cnt += 1
        tmp_list.sort(key=lambda k: k[1])
        for x in tmp_list:
            ret_list.append(x[0])
        return ret_list

    def _if_last_block(self, op, _equal_dict):
        # for conditional_block op 
        cond_str = op.input('Cond')[0]
        bool_test = False
        if cond_str.startswith('equal'):
            bool_test = True
        vars_ = op.input('Input')
        equal_keys = _equal_dict.keys()
        for var_cond in vars_:
            if var_cond in equal_keys:
                if bool_test:
                    print("the conditional block is error")
                return False
        return True

    def _generte_cond_para_map(self, op, _fill_value_dict, _equal_fill_dict,
                               _now_program, _all_params):
        # generate cond value to parameter map recursively
        cond_str = op.input('Cond')[0]
        vars_ = op.input('Input')

        if self._if_last_block(op, _equal_fill_dict):
            vars_ = op.input('Input')
            cond_key = ""
            if cond_str.startswith('equal'):
                cond_key = int(_fill_value_dict[_equal_fill_dict[cond_str]])
            else:
                cond_key = -1
            p_list = []
            for var_cond in vars_:
                if var_cond in _all_params:
                    p_list.append(var_cond)

            self._cond_params[cond_key] = p_list
            self._other_params.extend(p_list)
        else:
            ops_cond = _now_program.block(int(op.attr('sub_block').id)).ops
            for op in ops_cond:
                if op.type == 'conditional_block':
                    self._generte_cond_para_map(op, _fill_value_dict,
                                                _equal_fill_dict, _now_program,
                                                _all_params)

    def _has_conditional_block(self, loss):
        now_program = loss.block.program
        root_block = now_program.block(0)
        ops_ = root_block.ops
        for op in ops_:
            if op.type == 'conditional_block':
                return True
        return False

    def _check_params_grads(self, params, grads):
        if len(params) != len(grads):
            raise ValueError("params size != grads size, %s vs %s" %
                             (len(params), len(grads)))

        pname2grad = dict()
        for i in range(len(params)):
            pname = params[i].name
            gname = grads[i].name
            if pname != gname[:-5]:
                raise ValueError(" params != grads , %s vs %s" % (pname, gname))
            pname2grad[pname] = grads[i]

        return pname2grad

    def _generate_multi_dense_table(self,
                                    params,
                                    grads,
                                    cond_params,
                                    other_params,
                                    sparse_table_names,
                                    dense_table_id=0):
        # generate multi dense table by cond value
        pname2grad = self._check_params_grads(params, grads)
        root_params_list = []
        root_grads_list = []
        dense_tables = []
        for i, p in enumerate(params):
            if p.name not in other_params and p.name not in sparse_table_names:
                root_params_list.append(p)
                root_grads_list.append(grads[i])
        if len(root_params_list) > 0:
            dense_tables.append(dense_table_id)
            dense_table_id += 1
        lists_params = [[] for i in range(len(cond_params.keys()))]
        lists_grads = [[] for i in range(len(cond_params.keys()))]

        key_id = 0
        name2key = dict()
        cond2denseid = dict()
        for key, value in cond_params.items():
            cond2denseid[key] = dense_table_id
            dense_tables.append(dense_table_id)
            dense_table_id += 1
            for v in value:
                name2key[v] = key_id
            key_id += 1

        for p in params:
            if p.name in other_params:
                lists_params[name2key[p.name]].append(p)
                lists_grads[name2key[p.name]].append(pname2grad[p.name])

        return dense_tables, cond2denseid, lists_params, lists_grads, root_params_list, root_grads_list

    def _gen_distributed_emb_to_size_dict(self, program):
        d_size = dict()
        local_vars = program.current_block().vars

        for op in program.global_block().ops:
            if op.type in self.supported_embedding_types:
                if op.attr('is_distributed') is True:
                    table_name = op.input("W")[0]
                    emb_size = local_vars[table_name].shape[-1]
                    if d_size.get(table_name) is None:
                        d_size[table_name] = emb_size
                    elif d_size[table_name] != emb_size:
                        raise ValueError("embedding size error: %s vs %s" %
                                         (emb_size, d_size[table_name]))

        return d_size

    def _check_config_fleet_with_program_op(self, strategy, table_name,
                                            emb_to_size):
        if strategy.get(table_name) is None:
            strategy[table_name] = dict()
        st = strategy[table_name]

        accessor = "DownpourCtrAccessor"
        if st.get("sparse_accessor_class") is not None:
            accessor = st["sparse_accessor_class"]

        # set sparse_embedx_dim in strategy,
        # user do not have to set it in config_fleet
        if accessor == "DownpourFeatureValueAccessor" \
                or accessor == "DownpourCtrAccessor" \
                or accessor == "DownpourDoubleUnitAccessor" \
                or accessor == "DownpourUnitAccessor":
            if st.get("sparse_embedx_dim") is not None \
                    and st["sparse_embedx_dim"] != emb_to_size[table_name] - 3:
                raise ValueError("fleet config sparse_embedx_dim=%s not"
                                 " equal to embedding dim - 3 = %s" %
                                 (st["sparse_embedx_dim"],
                                  emb_to_size[table_name] - 3))
            if st.get("sparse_embedx_dim") is None:
                logger.warning(
                    "sparse embedding dim for table name '{}' is: {}, while sparse_embedx_dim "
                    "with same sparse table name is not set in config_fleet.py. "
                    "Hence automatically set sparse_embedx_dim = {} - 3.".
                    format(table_name, emb_to_size[table_name], emb_to_size[
                        table_name]))
                st["sparse_embedx_dim"] = emb_to_size[table_name] - 3
        elif accessor == "DownpourSparseValueAccessor":
            if st.get("sparse_embedx_dim") is not None \
                    and st["sparse_embedx_dim"] != emb_to_size[table_name]:
                raise ValueError("fleet config sparse_embedx_dim=%s not"
                                 " equal to embedding dim = %s" %
                                 (st["sparse_embedx_dim"],
                                  emb_to_size[table_name]))
            if st.get("sparse_embedx_dim") is None:
                logger.warning(
                    "sparse embedding dim for table name '{}' is: {}, while sparse_embedx_dim "
                    "with same sparse table name is not set in config_fleet.py. "
                    "Hence automatically set sparse_embedx_dim = {}.".format(
                        table_name, emb_to_size[table_name], emb_to_size[
                            table_name]))
                st["sparse_embedx_dim"] = emb_to_size[table_name]

        return strategy

    def _minimize(self,
                  losses,
                  startup_program=None,
                  parameter_list=None,
                  no_grad_set=None,
                  strategy={}):
        """
        DownpounSGD is a distributed optimizer so
        that user can call minimize to generate backward
        operators and optimization operators within minimize function
        Args:
            loss(Variable): loss variable defined by user
            startup_program(Program): startup program that defined by user
            parameter_list(str list): parameter names defined by users
            no_grad_set(set): a set of variables that is defined by users
            so that these variables do not need gradient computation
            strategy(dict): user-defined properties
        Returns:
            [optimize_ops, grads_and_weights]
        """
        # sparse table names of each program
        prog_id_to_sparse_table = OrderedDict()
        # inputs_dict and outputs_dict of sparse tables of each program
        prog_id_to_inputs_dict = OrderedDict()
        prog_id_to_outputs_dict = OrderedDict()
        # related to PSParameter
        ps_param = pslib.PSParameter()
        # related to ServerParameter
        server = DownpourServer()
        # program to worker (related to DownpourTrainerParameter)
        prog_id_to_worker = OrderedDict()
        # param_grads of each program
        prog_id_to_param_grads = OrderedDict()
        # sparse_grads of each program
        prog_id_to_sparse_grads = OrderedDict()
        # unique program set
        program_id_set = set()

        sparse_table_to_index = OrderedDict()
        sparse_table_index = 0
        for num in range(len(losses)):
            loss = losses[num]
            parameters = None
            if parameter_list != None:
                parameters = parameter_list[num]
            prog_id = str(id(loss.block.program))
            # param_grads of program
            params_grads = sorted(
                fluid.backward.append_backward(loss, parameters, no_grad_set),
                key=lambda x: x[0].name)

            flag_use_ps_gpu = strategy.get("use_ps_gpu", False)
            if flag_use_ps_gpu:
                if not isinstance(startup_program, list):
                    startup_program = [startup_program]
                optimizer = copy.deepcopy(self._optimizer)
                optimize_ops = optimizer.apply_optimize(
                    loss,
                    startup_program=startup_program[num],
                    params_grads=params_grads)
                embedding_table = self._find_multi_distributed_lookup_table(
                    [loss])
                self._remove_optimize_op_for_embedding(loss, embedding_table)
            # has condition_block op means multi-task 
            flag_multi_task = self._has_conditional_block(loss)
            if flag_multi_task:
                self._cond_params = dict()
                self._other_params = []
                now_program = loss.block.program
                root_block = now_program.block(0)
                all_params = []
                for par in root_block.all_parameters():
                    all_params.append(par.name)

                ops_ = root_block.ops
                fill_value_dict = dict()
                equal_fill_dict = dict()
                for op in ops_:
                    # conditional_block op must has fill_constant and equal op
                    if op.type == 'fill_constant':
                        fill_value_dict[op.output('Out')[0]] = op.attr('value')
                    if op.type == 'equal':
                        equal_fill_dict[op.output('Out')[0]] = op.input('Y')[0]
                    if op.type == 'conditional_block':
                        self._generte_cond_para_map(op, fill_value_dict,
                                                    equal_fill_dict,
                                                    now_program, all_params)

            if prog_id not in program_id_set:
                program_id_set.add(prog_id)
                sparse_table = self._find_multi_distributed_lookup_table([loss])
                prog_id_to_sparse_table[prog_id] = sparse_table

                # get sparse_table_to_index
                for tn in sparse_table:
                    if sparse_table_to_index.get(tn) is None:
                        sparse_table_to_index[tn] = sparse_table_index
                        sparse_table_index += 1

                # get {table_name: emb_size} dict from program ops
                emb_to_size = self._gen_distributed_emb_to_size_dict(
                    loss.block.program)

                # get inputs_dict
                inputs_dict = self._find_distributed_lookup_table_inputs(
                    loss.block.program, sparse_table)
                prog_id_to_inputs_dict[prog_id] = inputs_dict
                # get outputs_dict
                outputs_dict = self._find_distributed_lookup_table_outputs(
                    loss.block.program, sparse_table)
                prog_id_to_outputs_dict[prog_id] = outputs_dict

                prog_id_to_worker[prog_id] = DownpourWorker(self._window)

                grads_dict = self._find_distributed_lookup_table_grads(
                    loss.block.program, sparse_table)
                prog_id_to_sparse_grads[prog_id] = grads_dict

            if prog_id not in prog_id_to_param_grads:
                prog_id_to_param_grads[prog_id] = []
            prog_id_to_param_grads[prog_id].append(params_grads)

        #if strategy.get("parallel_compute")

        # if user specify a fleet_desc.prototxt file, then load the file
        # instead of creating default fleet_desc.prototxt.
        # user can specify server_param or trainer_param or fs_client_param.
        if strategy.get("fleet_desc_file") is not None:
            fleet_desc_file = strategy["fleet_desc_file"]
            with open(fleet_desc_file) as f:
                text_format.Merge(f.read(), ps_param)
            server.get_desc().CopyFrom(ps_param.server_param)
            if len(ps_param.trainer_param) == 1:
                for k in prog_id_to_worker:
                    prog_id_to_worker[k].get_desc().CopyFrom(
                        ps_param.trainer_param[0])
            else:
                if len(ps_param.trainer_param) != len(prog_id_to_worker):
                    raise ValueError(
                        "trainer param size != program size, %s vs %s" %
                        (len(ps_param.trainer_param), len(prog_id_to_worker)))
                idx = 0
                # prog_id_to_worker is OrderedDict
                for k in prog_id_to_worker:
                    prog_id_to_worker[k].get_desc().CopyFrom(
                        ps_param.trainer_param[idx])
                    idx += 1

        # check config in op defination and fleet config
        if FLEET_GLOBAL_DICT["enable"]:
            one_slot = None
            strategy["device_worker"] = "Hogwild"
            emb_to_table = FLEET_GLOBAL_DICT["emb_to_table"]
            emb_to_accessor = FLEET_GLOBAL_DICT["emb_to_accessor"]
            emb_to_size = FLEET_GLOBAL_DICT["emb_to_size"]
            if len(sparse_table_to_index) != len(emb_to_table):
                raise ValueError(
                    "sparse tables from  program != sparse tables from op: %s "
                    "vs %s" % (len(sparse_table_to_index), len(emb_to_table)))
            for key in sparse_table_to_index:
                if key not in emb_to_table or \
                                sparse_table_to_index[key] != emb_to_table[key]:
                    print("sparse_table_to_index ", sparse_table_to_index)
                    print("emb_to_table ", emb_to_table)
                    raise ValueError("key error: %s" % key)
                if strategy.get(key) is None:
                    strategy[key] = dict()
                st = strategy[key]

                accessor = None
                if st.get("sparse_accessor_class") is not None:
                    accessor = st["sparse_accessor_class"]
                tables = \
                    server.get_desc().downpour_server_param.downpour_table_param
                for table in tables:
                    if table.table_id == sparse_table_to_index[key]:
                        accessor = table.accessor.accessor_class
                        break

                for loss in losses:
                    for op in loss.block.program.global_block().ops:
                        if op.type in self.supported_embedding_types:
                            if accessor is not None \
                                    and op.has_attr("AccessorClass"):
                                op._set_attr("AccessorClass", accessor)
                            if one_slot is None:
                                one_slot = loss.block.program. \
                                    global_block().var(op.input("Ids")[0])

                # if accessor is None, use default accessor in op definition
                if accessor is None:
                    accessor = emb_to_accessor[key]
                # set sparse_embedx_dim in strategy,
                # user do not have to set it in config_fleet
                if accessor == "DownpourFeatureValueAccessor" \
                        or accessor == "DownpourCtrAccessor" \
                        or accessor == "DownpourDoubleUnitAccessor" \
                        or accessor == "DownpourUnitAccessor":
                    if st.get("sparse_embedx_dim") is not None \
                            and st["sparse_embedx_dim"] != emb_to_size[key] - 3:
                        raise ValueError("fleet config sparse_embedx_dim=%s not"
                                         " equal to embedding size - 3 = %s" %
                                         (st["sparse_embedx_dim"],
                                          emb_to_size[key] - 3))
                    st["sparse_embedx_dim"] = emb_to_size[key] - 3
                elif accessor == "DownpourSparseValueAccessor":
                    if st.get("sparse_embedx_dim") is not None \
                            and st["sparse_embedx_dim"] != emb_to_size[key]:
                        raise ValueError("fleet config sparse_embedx_dim=%s not"
                                         " equal to embedding size = %s" %
                                         (st["sparse_embedx_dim"],
                                          emb_to_size[key]))
                    st["sparse_embedx_dim"] = emb_to_size[key]

        # ServerParameter add all sparse tables
        for tn in sparse_table_to_index:
            sparse_table_index = sparse_table_to_index[tn]
            st = self._check_config_fleet_with_program_op(strategy, tn,
                                                          emb_to_size)
            if st.get(tn) is not None:
                server.add_sparse_table(sparse_table_index, st[tn])
            else:
                server.add_sparse_table(sparse_table_index, None)

        # each DownpourTrainerParameter add its own sparse tables
        program_id_set.clear()
        for loss in losses:
            prog_id = str(id(loss.block.program))
            if prog_id not in program_id_set:
                program_id_set.add(prog_id)
                worker = prog_id_to_worker[prog_id]
                inputs_dict = prog_id_to_inputs_dict[prog_id]
                outputs_dict = prog_id_to_outputs_dict[prog_id]
                for tn in prog_id_to_sparse_table[prog_id]:
                    sparse_table_index = sparse_table_to_index[tn]
                    grads_dict = prog_id_to_sparse_grads[prog_id]
                    worker.add_sparse_table(sparse_table_index, inputs_dict[tn],
                                            outputs_dict[tn], grads_dict[tn])

        dense_start_table_id = len(sparse_table_to_index)
        dense_table_index = len(sparse_table_to_index)
        program_configs = {}
        # ServerParameter add all dense tables
        # each DownpourTrainerParameter add its own dense tables
        program_id_set.clear()
        for loss_index in range(len(losses)):
            program_id = str(id(losses[loss_index].block.program))
            if program_id not in program_id_set:
                program_id_set.add(program_id)
                worker = prog_id_to_worker[program_id]
                sparse_table_names = prog_id_to_sparse_table[program_id]
                sparse_table_index = \
                    [sparse_table_to_index[i] for i in sparse_table_names]

                program_configs[program_id] = {
                    "pull_sparse": [t_index for t_index in sparse_table_index],
                    "push_sparse": [t_index for t_index in sparse_table_index]
                }

                params_grads = prog_id_to_param_grads[program_id]
                for pg in params_grads:
                    params = []
                    grads = []
                    data_norm_params = []
                    data_norm_grads = []
                    for i in pg:
                        is_data_norm_data = False
                        for data_norm_name in self.data_norm_name:
                            if i[0].name.endswith(data_norm_name):
                                is_data_norm_data = True
                                data_norm_params.append(i[0])
                        if not is_data_norm_data:
                            params.append(i[0])

                    for i in pg:
                        is_data_norm_data = False
                        for data_norm_grad in self.data_norm_name:
                            if i[0].name.endswith(data_norm_grad):
                                is_data_norm_data = True
                                data_norm_grads.append(i[1])
                        if not is_data_norm_data:
                            grads.append(i[1])
                    # for new dense table
                    multi_task_dense_tables_push = []
                    multi_task_dense_tables_pull = []
                    if flag_multi_task:
                        dense_tables, cond2denseid, lists_params, lists_grads, root_params_list, root_grads_list = self._generate_multi_dense_table(
                            params, grads, self._cond_params,
                            self._other_params, sparse_table_names,
                            dense_table_index)
                        program_configs[program_id][
                            'cond2denseid'] = cond2denseid
                        multi_task_dense_tables_push = dense_tables
                        multi_task_dense_tables_pull = dense_tables[:]

                    if strategy.get('dense_table') is not None:
                        if flag_multi_task:
                            server_dense_table_index = dense_table_index
                            if len(root_params_list) > 0:
                                server.add_dense_table(
                                    server_dense_table_index, root_params_list,
                                    root_grads_list, strategy['dense_table'],
                                    sparse_table_names)
                                server_dense_table_index += 1

                            for i in range(len(lists_params)):
                                server.add_dense_table(
                                    server_dense_table_index, lists_params[i],
                                    lists_grads[i], strategy['dense_table'],
                                    sparse_table_names)
                                server_dense_table_index += 1
                        else:
                            server.add_dense_table(
                                dense_table_index, params, grads,
                                strategy['dense_table'], sparse_table_names)

                    else:
                        server.add_dense_table(dense_table_index, params, grads,
                                               None, sparse_table_names)

                    if flag_multi_task:

                        if len(root_params_list) > 0:
                            worker.add_dense_table(
                                dense_table_index, self._learning_rate,
                                root_params_list, root_grads_list,
                                dense_start_table_id, sparse_table_names)
                            dense_table_index += 1

                        for i in range(len(lists_params)):
                            worker.add_dense_table(
                                dense_table_index, self._learning_rate,
                                lists_params[i], lists_grads[i],
                                dense_start_table_id, sparse_table_names)
                            dense_table_index += 1

                        dense_table_index -= 1
                    else:
                        worker.add_dense_table(
                            dense_table_index, self._learning_rate, params,
                            grads, dense_start_table_id, sparse_table_names)

                    if FLEET_GLOBAL_DICT["enable"]:
                        cur_prog = losses[loss_index].block.program
                        cur_prog.global_block().append_op(
                            type="push_dense",
                            inputs={"Ids": one_slot},
                            attrs={
                                "InputNames": [i.name for i in grads],
                                "TableId": dense_table_index,
                                "ScaleDataNorm":
                                strategy.get("scale_datanorm", -1)
                            })

                    if "pull_dense" in program_configs[
                            program_id] and "push_dense" in program_configs[
                                program_id] and len(program_configs[program_id][
                                    "pull_dense"]) > 0:
                        if flag_multi_task:
                            program_configs[program_id]["pull_dense"].extend(
                                multi_task_dense_tables_pull)
                            program_configs[program_id]["push_dense"].extend(
                                multi_task_dense_tables_push)
                        else:
                            program_configs[program_id]["pull_dense"].extend(
                                [dense_table_index])
                            program_configs[program_id]["push_dense"].extend(
                                [dense_table_index])
                    else:
                        if flag_multi_task:
                            program_configs[program_id][
                                "pull_dense"] = multi_task_dense_tables_pull
                            program_configs[program_id][
                                "push_dense"] = multi_task_dense_tables_push
                        else:
                            program_configs[program_id][
                                "pull_dense"] = [dense_table_index]
                            program_configs[program_id][
                                "push_dense"] = [dense_table_index]

                    if len(data_norm_params) != 0 and len(data_norm_grads) != 0:
                        dense_table_index += 1
                        if strategy.get('datanorm_table') is not None:
                            server.add_data_norm_table(
                                dense_table_index, self._learning_rate,
                                data_norm_params, data_norm_grads,
                                strategy['datanorm_table'], sparse_table_names)
                        else:
                            server.add_data_norm_table(
                                dense_table_index, self._learning_rate,
                                data_norm_params, data_norm_grads, None,
                                sparse_table_names)

                        worker.add_dense_table(
                            dense_table_index, self._learning_rate,
                            data_norm_params, data_norm_grads,
                            dense_start_table_id, sparse_table_names)

                        if FLEET_GLOBAL_DICT["enable"]:
                            cur_prog = losses[loss_index].block.program
                            cur_prog.global_block().append_op(
                                type="push_dense",
                                inputs={"Ids": one_slot},
                                attrs={
                                    "InputNames":
                                    [i.name for i in data_norm_grads],
                                    "TableId": dense_table_index,
                                    "ScaleDataNorm":
                                    strategy.get("scale_datanorm", -1)
                                })

                        program_configs[program_id]["pull_dense"].extend(
                            [dense_table_index])
                        program_configs[program_id]["push_dense"].extend(
                            [dense_table_index])
                    dense_table_index += 1

            # Todo(guru4elephant): figure out how to support more sparse parameters
            # currently only support lookup_table
            worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
            if len(worker.get_desc().skip_op) == 0:
                worker.get_desc().skip_op.extend(worker_skipped_ops)

        ps_param.server_param.CopyFrom(server.get_desc())
        # prog_id_to_worker is OrderedDict
        if len(ps_param.trainer_param) == 0:
            for k in prog_id_to_worker:
                tp = ps_param.trainer_param.add()
                tp.CopyFrom(prog_id_to_worker[k].get_desc())

        if strategy.get("fs_uri") is not None:
            ps_param.fs_client_param.uri = strategy["fs_uri"]
        elif ps_param.fs_client_param.uri == "":
            ps_param.fs_client_param.uri = "hdfs://your_hdfs_uri"
        if strategy.get("fs_user") is not None:
            ps_param.fs_client_param.user = strategy["fs_user"]
        elif ps_param.fs_client_param.user == "":
            ps_param.fs_client_param.user = "your_hdfs_user"
        if strategy.get("fs_passwd") is not None:
            ps_param.fs_client_param.passwd = strategy["fs_passwd"]
        elif ps_param.fs_client_param.passwd == "":
            ps_param.fs_client_param.passwd = "your_hdfs_passwd"
        if strategy.get("fs_hadoop_bin") is not None:
            ps_param.fs_client_param.hadoop_bin = strategy["fs_hadoop_bin"]
        elif ps_param.fs_client_param.hadoop_bin == "":
            ps_param.fs_client_param.hadoop_bin = "$HADOOP_HOME/bin/hadoop"

        opt_info = {}
        opt_info["program_id_to_worker"] = prog_id_to_worker
        opt_info["program_configs"] = program_configs
        opt_info["trainer"] = strategy.get("trainer", "DistMultiTrainer")
        opt_info["device_worker"] = strategy.get("device_worker", "DownpourSGD")
        opt_info["optimizer"] = "DownpourSGD"
        opt_info["fleet_desc"] = ps_param
        opt_info["worker_skipped_ops"] = worker_skipped_ops
        opt_info["use_cvm"] = strategy.get("use_cvm", False)
        opt_info["no_cvm"] = strategy.get("no_cvm", False)
        opt_info["scale_sparse_gradient_with_batch_size"] = strategy.get(
            "scale_sparse_gradient_with_batch_size", True)
        opt_info["worker_class"] = strategy.get("worker_class",
                                                "DownpourWorker")
        opt_info["stat_var_names"] = strategy.get("stat_var_names", [])
        opt_info["local_tables"] = strategy.get("local_tables", [])
        opt_info["async_tables"] = strategy.get("async_tables", [])
        opt_info["async_tables"] = strategy.get("async_tables", [])
        opt_info["scale_datanorm"] = strategy.get("scale_datanorm", -1)
        opt_info["check_nan_var_names"] = strategy.get("check_nan_var_names",
                                                       [])
        opt_info["dump_slot"] = False
        opt_info["dump_converter"] = ""
        opt_info["dump_fields"] = strategy.get("dump_fields", [])
        opt_info["dump_file_num"] = strategy.get("dump_file_num", 16)
        opt_info["user_define_dump_filename"] = strategy.get(
            "user_define_dump_filename", "")
        opt_info["dump_fields_path"] = strategy.get("dump_fields_path", "")
        opt_info["dump_param"] = strategy.get("dump_param", [])
        gpus_env = os.getenv("FLAGS_selected_gpus", "0")
        opt_info["worker_places"] = [int(s) for s in gpus_env.split(",")]
        opt_info["use_ps_gpu"] = strategy.get("use_ps_gpu", False)
        if server._server.downpour_server_param.downpour_table_param[
                0].accessor.accessor_class in [
                    "DownpourCtrAccessor", "DownpourCtrDoubleAccessor",
                    "DownpourUnitAccessor", "DownpourDoubleUnitAccessor"
                ]:
            opt_info["dump_slot"] = True
        elif server._server.downpour_server_param.downpour_table_param[
                0].accessor.accessor_class == "DownpourSparseValueAccessor":
            opt_info["no_cvm"] = True
        opt_info["adjust_ins_weight"] = strategy.get("adjust_ins_weight", {})
        opt_info["copy_table"] = strategy.get("copy_table", {})
        opt_info["loss_names"] = strategy.get("loss_names", [])

        for loss in losses:
            loss.block.program._fleet_opt = opt_info

        param_grads_list = []
        for loss in losses:
            prog_id = str(id(loss.block.program))
            param_grads_list.append(prog_id_to_param_grads[prog_id])
        return None, param_grads_list, opt_info
