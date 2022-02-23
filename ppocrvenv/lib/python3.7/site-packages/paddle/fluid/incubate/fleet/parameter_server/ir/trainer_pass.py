# -*- coding: UTF-8 -*-
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

from __future__ import print_function
import os
import six
import collections
import warnings
import math

from functools import reduce
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import paddle.compat as cpt

from paddle.fluid.transpiler.details.program_utils import delete_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_lr_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode

OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "gradient_clip"
STEP_COUNTER = "@PS_STEP_COUNTER@"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()

SPARSE_OP_TYPE_DICT = {"lookup_table": "W", "lookup_table_v2": "W"}
DEVICE_LIST = ["cpu", "gpu", "xpu"]
COMMUNICATE_OPS_TYPE = ["send", "recv", "fetch_barrier", "send_barrier"]
DEFAULT_DEVICE = 'cpu'


def delete_optimizer_pass(program, config):
    def _delete_optimizer_op_and_vars(_program, optimize_ops):
        optimize_vars = []
        optimize_op_role_vars = []
        optimize_need_delete_vars = []

        for op in optimize_ops:
            optimize_vars.extend(op.input_arg_names)
            optimize_op_role_vars.extend(op.attr("op_role_var"))

        optimize_vars = list(set(optimize_vars))
        optimize_op_role_vars = list(set(optimize_op_role_vars))

        for var in optimize_vars:
            if var not in optimize_op_role_vars:
                optimize_need_delete_vars.append(var)
        need_delete_optimize_vars = list(set(optimize_need_delete_vars))

        delete_ops(_program.global_block(), optimize_ops)
        for var in need_delete_optimize_vars:
            if _program.global_block().has_var(var):
                _program.global_block()._remove_var(var)

    def _add_lr_var(main_program, compiled_config):
        # Todo: hard code for pe
        lr_var = compiled_config.origin_main_program.global_block().vars[
            "learning_rate_0"]
        main_program.global_block().create_var(
            name=lr_var.name,
            shape=lr_var.shape,
            dtype=lr_var.dtype,
            type=lr_var.type,
            lod_level=lr_var.lod_level,
            persistable=True)

    optimizer_ops = _get_optimize_ops(program)
    lr_ops = _get_lr_ops(program)
    optimizer_ops.extend(lr_ops)
    _delete_optimizer_op_and_vars(program, optimizer_ops)

    if hasattr(config.origin_main_program, 'lr_sheduler'):
        _add_lr_var(program, config)

    return program


def distributed_ops_pass(program, config, use_ps_gpu=False):
    trainer_id = config.get_role_id()
    send_ctx = config.get_the_one_send_context(
        split_dense_table=config.is_heter_ps_mode)

    def _get_pull_sparse_ops(_program):
        pull_sparse_ops = {}
        for op in _program.global_block().ops:
            if op.type in SPARSE_OP_TYPE_DICT.keys() \
                    and op.attr('remote_prefetch') is True:
                param_name = op.input(SPARSE_OP_TYPE_DICT[op.type])[0]
                if config.is_heter_ps_mode:
                    # trick for matchnet, need to modify
                    param_name += op.input("Ids")[0][0]
                ops = pull_sparse_ops.get(param_name, [])
                ops.append(op)
                pull_sparse_ops[param_name] = ops
        return pull_sparse_ops

    def _pull_sparse_fuse(_program, pull_sparse_ops, use_ps_gpu):
        for param, ops in pull_sparse_ops.items():
            all_ops = program.global_block().ops
            op_idxs = [all_ops.index(op) for op in ops]
            op_device = ""
            if config.is_heter_ps_mode:
                op_device = ops[0].attr("op_device")
            inputs = [
                program.global_block().vars[op.input("Ids")[0]] for op in ops
            ]
            w = program.global_block().vars[ops[0].input("W")[0]]

            grad_name = config.param_name_to_grad_name[w.name]

            table_id = -1

            for name, ctx in send_ctx.items():
                if grad_name in ctx.origin_varnames():
                    table_id = ctx.table_id()

            if table_id == -1:
                raise ValueError(
                    "can not find suitable sparse table, please check")

            padding_idx = ops[0].attr("padding_idx")
            is_distributed = ops[0].attr("is_distributed")
            op_type = ops[0].type

            outputs = [
                program.global_block().vars[op.output("Out")[0]] for op in ops
            ]

            for idx in op_idxs[::-1]:
                program.global_block()._remove_op(idx)

            inputs_idxs = [-1] * len(inputs)
            outputs_idxs = [-1] * len(outputs)

            for idx, op in enumerate(program.global_block().ops):
                for i in range(0, len(op.output_names)):
                    outs = op.output(op.output_names[i])
                    for in_id, in_var in enumerate(inputs):
                        if in_var.name in outs:
                            inputs_idxs[in_id] = idx
                for i in range(0, len(op.input_names)):
                    ins = op.input(op.input_names[i])
                    for out_id, out_var in enumerate(outputs):
                        if out_var.name in ins:
                            outputs_idxs[out_id] = idx

            if min(outputs_idxs) - max(inputs_idxs) >= 1:

                if max(inputs_idxs) == -1:
                    distributed_idx = min(op_idxs)
                else:
                    distributed_idx = max(inputs_idxs) + 1

                if use_ps_gpu:
                    program.global_block()._insert_op(
                        index=distributed_idx,
                        type="pull_box_sparse",
                        inputs={"Ids": inputs,
                                'W': w},
                        outputs={"Out": outputs},
                        attrs={
                            "size": w.shape[1],
                            "is_distributed": True,
                            "is_sparse": True
                        })
                else:
                    program.global_block()._insert_op(
                        index=distributed_idx,
                        type="distributed_lookup_table",
                        inputs={"Ids": inputs,
                                'W': w},
                        outputs={"Outputs": outputs},
                        attrs={
                            "is_distributed": is_distributed,
                            "padding_idx": padding_idx,
                            "table_id": table_id,
                            "lookup_table_version": op_type,
                            "op_device": op_device
                        })
            else:
                for i in range(len(inputs_idxs)):
                    distributed_idx = op_idxs[i] + 1

                    program.global_block()._insert_op(
                        index=distributed_idx,
                        type="distributed_lookup_table",
                        inputs={"Ids": [inputs[i]],
                                'W': w},
                        outputs={"Outputs": [outputs[i]]},
                        attrs={
                            "is_distributed": is_distributed,
                            "padding_idx": padding_idx,
                            "table_id": table_id,
                            "lookup_table_version": op_type,
                            "op_device": op_device
                        })

    pull_sparse_ops = _get_pull_sparse_ops(program)
    _pull_sparse_fuse(program, pull_sparse_ops, use_ps_gpu)
    return program


def append_send_ops_pass(program, config):
    mode = config.get_distributed_mode()
    trainer_id = config.get_role_id()

    def _append_send_op(union_vars, queue, is_sparse, table_id):

        if queue == STEP_COUNTER:
            send_input_vars = []
        else:
            send_input_vars = [
                program.global_block().vars[union_var]
                for union_var in union_vars
            ]

        dummy_output = []
        if mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
            dummy_output = program.global_block().create_var(
                name=framework.generate_control_dev_var_name())

        program.global_block().append_op(
            type="send",
            inputs={"X": send_input_vars},
            outputs={"Out": dummy_output},
            attrs={
                "send_varnames": [queue],
                "is_sparse": is_sparse,
                "table_id": table_id,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

        return dummy_output

    def _append_barrier_op(dummys):
        program.global_block().append_op(
            type="send_barrier",
            inputs={"X": dummys},
            outputs={"Out": []},
            attrs={
                "trainer_id": trainer_id,
                "half_async": True,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

    dummys = []

    sends = config.get_the_one_trainer_send_context(
        split_dense_table=config.is_heter_ps_mode)

    for merged_name, send in sends.items():
        is_sparse = 1 if send.is_sparse() else 0
        is_sparse = 2 if send.is_distributed() else is_sparse
        dummys.append(
            _append_send_op(send.origin_varnames(), merged_name, is_sparse,
                            send.table_id()))

    if mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
        _append_barrier_op(dummys)

    return program


def init_from_server_pass(program, config):
    # 0' trainer do not need barrier, it will call barrier at the end init_worker
    if config.role_maker._is_first_worker():
        return program

    fetch_barrier_out = program.global_block().create_var(
        name=framework.generate_control_dev_var_name())

    program.global_block().append_op(
        type="fetch_barrier",
        inputs={},
        outputs={"Out": fetch_barrier_out},
        attrs={
            "endpoints": config.get_ps_endpoints(),
            "trainer_id": config.get_role_id(),
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
        })
    return program


def fake_init_ops_pass(program, config):
    origin_program = config.get_origin_main_program()

    def _get_sparse_table_names():
        dist_varnames = get_sparse_tablenames(origin_program, True)
        sparse_varnames = get_sparse_tablenames(origin_program, False)
        return list(set(dist_varnames + sparse_varnames))

    def _fake_init_sparsetable(sparse_table_names):
        # delete table init op
        for table_name in sparse_table_names:
            table_var = program.global_block().vars[table_name]
            table_param_init_op = []
            for op in program.global_block().ops:
                if table_name in op.output_arg_names:
                    table_param_init_op.append(op)
            init_op_num = len(table_param_init_op)
            if init_op_num != 1:
                raise ValueError("table init op num should be 1, now is " + str(
                    init_op_num))
            table_init_op = table_param_init_op[0]
            program.global_block().append_op(
                type="fake_init",
                inputs={},
                outputs={"Out": table_var},
                attrs={"shape": table_init_op.attr('shape')})
            delete_ops(program.global_block(), table_param_init_op)

    sparse_tables = _get_sparse_table_names()
    _fake_init_sparsetable(sparse_tables)

    return program


def ps_gpu_pass(program):
    def _add_push_box_sparse_op(program):
        op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        for op in program.global_block().ops:
            if op.type != "pull_box_sparse":
                continue
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, cpt.to_text(set()), [])
            for op_desc in grad_op_desc:
                new_op_desc = program.global_block().desc.append_op()
                new_op_desc.copy_from(op_desc)
                new_op_desc._set_attr(op_role_attr_name, backward)

    def _remove_lookup_table_grad_op_and_var(program):
        lookup_table_grad_var = {}
        remove_op_index = []
        remove_var = []
        for idx, op in list(enumerate(program.global_block().ops)):
            if op.type == "lookup_table_grad":
                for name in op.output("W@GRAD"):
                    lookup_table_grad_var[name] = 1
                    remove_op_index.append(idx)
                    remove_var.append(name)
                for name in op.input("W"):
                    lookup_table_grad_var[name] = 1

        for idx, op in list(enumerate(program.global_block().ops)):
            if op.type == "pull_box_sparse":
                continue
            for key_name in op.input_names:
                for var in op.input(key_name):
                    if var in lookup_table_grad_var:
                        remove_op_index.append(idx)
                        break

        remove_op_index = list(set(remove_op_index))
        remove_op_index.sort(reverse=True)
        for idx in remove_op_index:
            program.global_block()._remove_op(idx)
        for name in remove_var:
            program.global_block()._remove_var(name)

    def _remove_optimizer_var(program):

        embedding_w = {}
        for idx, op in list(enumerate(program.global_block().ops)):
            if op.type == "lookup_table_grad":
                for name in op.input("W"):
                    embedding_w[name] = 1

        optimize_vars = []
        optimize_op_role_vars = []
        optimize_need_delete_vars = []
        for op in _get_optimize_ops(program):
            for name in op.input("Param"):
                if name in embedding_w:
                    optimize_op_role_vars.extend(op.attr("op_role_var"))
                    for key_name in op.input_names:
                        if key_name == "LearningRate":
                            continue
                        for var in op.input(key_name):
                            optimize_vars.append(var)

        optimize_vars = list(set(optimize_vars))
        optimize_op_role_vars = list(set(optimize_op_role_vars))

        for var in optimize_vars:
            if var not in optimize_op_role_vars:
                optimize_need_delete_vars.append(var)
        need_delete_optimize_vars = list(set(optimize_need_delete_vars))

        for name in need_delete_optimize_vars:
            if program.global_block().has_var(name):
                program.global_block()._remove_var(name)

    _add_push_box_sparse_op(program)
    _remove_optimizer_var(program)
    _remove_lookup_table_grad_op_and_var(program)
    return program


def delete_extra_optimizes_pass(program, config):
    optimize_vars = []
    optimize_op_role_vars = []
    optimize_need_delete_vars = []

    origin_program = config.get_origin_main_program()
    for op in _get_optimize_ops(origin_program):
        optimize_vars.extend(op.input_arg_names)
        optimize_op_role_vars.extend(op.attr("op_role_var"))

    optimize_vars = list(set(optimize_vars))
    optimize_op_role_vars = list(set(optimize_op_role_vars))
    for var in optimize_vars:
        if var not in optimize_op_role_vars:
            optimize_need_delete_vars.append(var)
    need_delete_optimize_vars = list(set(optimize_need_delete_vars))

    init_ops = []
    for var in need_delete_optimize_vars:
        param_init_op = []
        for op in program.global_block().ops:
            if var in op.output_arg_names:
                param_init_op.append(op)
        init_ops.extend(param_init_op)
    delete_ops(program.global_block(), init_ops)

    for var in need_delete_optimize_vars:
        if program.global_block().has_var(var):
            program.global_block()._remove_var(var)

    return program


def find_heter_ops(program, default_device="cpu"):
    if default_device not in DEVICE_LIST:
        raise ValueError("Given device {} is not in device list {}".format(
            default_device, DEVICE_LIST))

    def _is_heter_op(op, current_heter_device, default_device="cpu"):
        heter_devices = list(DEVICE_LIST)
        heter_devices.remove(default_device)
        op_device = op.attr("op_device")
        op_type = op.type
        if op_device in heter_devices:
            return True
        elif op_type in COMMUNICATE_OPS_TYPE and current_heter_device != default_device:
            # for distributed communciate ops: send & recv & barrier etc.
            # Todo: need update this method
            #op._set_attr('op_device', current_heter_device)
            return True
        elif op_device == None or op_device == default_device:
            op._set_attr('op_device', default_device)
            return False
        return False

    def _is_same_device(op, pre_device, default_device="cpu"):
        op_device = op.attr("op_device")
        if op_device == pre_device:
            return True
        if pre_device == default_device:
            return True
        return False

    def _append_heter_op(op, current_heter_block_ops, heter_ops):
        op_device = op.attr("op_device")
        if op_device not in heter_ops:
            heter_ops[op_device] = {}
        current_heter_block_ops.append(op)

    origin_porgram = program.clone()
    block = program.global_block()
    '''
       re-place sum op to fix bug for union forward backward op
    '''
    var2idx = {}
    op_list = list(block.ops)
    op_size = len(op_list)

    for i in range(op_size - 1, -1, -1):
        op_list = list(block.ops)
        op = op_list[i]
        if "_grad" in op.type:
            forward_op_type = op.type.split("_grad")[0]
            if forward_op_type in SPARSE_OP_TYPE_DICT.keys() \
                and op.attr('remote_prefetch') is True:
                param_name = op.input(SPARSE_OP_TYPE_DICT[forward_op_type])[0]
                if param_name in var2idx:
                    ## insert sum op & remove sum op from var2idx and origin place
                    op_list = list(block.ops)
                    sum_op = op_list[var2idx[param_name]]
                    sum_op_inputs = {
                        sum_op.input_names[0]: [
                            block.vars[input]
                            for input in sum_op.input_arg_names
                        ]
                    }
                    sum_op_outputs = {
                        sum_op.output_names[0]: [
                            block.vars[output]
                            for output in sum_op.output_arg_names
                        ]
                    }
                    block._insert_op(
                        index=i + 1,
                        type=sum_op.type,
                        inputs=sum_op_inputs,
                        outputs=sum_op_outputs,
                        attrs=sum_op.all_attrs())
                    block._remove_op(var2idx[param_name] + 1)
                    var2idx.pop(param_name)
                    for var_ in var2idx:
                        var2idx[var_] += 1
            elif forward_op_type == "elementwise_mul":
                """
                get output varname of pre op

                """
                output_vars_no_grad = []
                for key in op.output_names:
                    for varname in op.output(key):
                        if varname == "@EMPTY@":
                            continue
                        if "lod_tensor_blocking_queue" in varname:
                            continue
                        output_vars_no_grad.append(varname.split("@GRAD")[0])
                for no_grad_var in output_vars_no_grad:
                    if no_grad_var in var2idx:
                        """
                       insert sum op & remove sum op from var2idx and origin place
  
                       """
                        op_list = list(block.ops)
                        sum_op = op_list[var2idx[no_grad_var]]
                        sum_op_inputs = {
                            sum_op.input_names[0]: [
                                block.vars[input]
                                for input in sum_op.input_arg_names
                            ]
                        }
                        sum_op_outputs = {
                            sum_op.output_names[0]: [
                                block.vars[output]
                                for output in sum_op.output_arg_names
                            ]
                        }
                        block._insert_op(
                            index=i + 1,
                            type=sum_op.type,
                            inputs=sum_op_inputs,
                            outputs=sum_op_outputs,
                            attrs=sum_op.all_attrs())
                        block._remove_op(var2idx[no_grad_var] + 1)
                        var2idx.pop(no_grad_var)
                        for var_ in var2idx:
                            var2idx[var_] += 1
        else:
            if op.type == "sum":
                var = op.output("Out")[0]
                if "@GRAD" in var:
                    origin_var = var.split("@GRAD")[0]
                    pre_op = op_list[i - 1]
                    if "_grad" in pre_op.type:
                        forward_op_type = pre_op.type.split("_grad")[0]
                        if forward_op_type in SPARSE_OP_TYPE_DICT.keys() \
                            and pre_op.attr('remote_prefetch') is True:
                            param_name = pre_op.input(SPARSE_OP_TYPE_DICT[
                                forward_op_type])[0]
                            if param_name == origin_var and op.attr(
                                    "op_device") == pre_op.attr("op_device"):
                                continue
                            else:
                                var2idx[origin_var] = i
                        elif forward_op_type == "elementwise_mul":
                            output_vars = []
                            for key in pre_op.output_names:
                                for varname in pre_op.output(key):
                                    if varname == "@EMPTY@":
                                        continue
                                    if "lod_tensor_blocking_queue" in varname:
                                        continue
                                    output_vars.append(varname)
                            input_vars = []
                            for key in op.input_names:
                                for varname in op.input(key):
                                    if varname == "@EMPTY@":
                                        continue
                                    if "lod_tensor_blocking_queue" in varname:
                                        continue
                                    input_vars.append(varname)
                            is_match = False
                            for varname in output_vars:
                                if varname in input_vars:
                                    is_match = True
                                    break
                            if is_match:
                                continue
                            else:
                                var2idx[origin_var] = i
                    else:
                        var2idx[origin_var] = i

    origin_porgram = program.clone()
    block = program.global_block()

    program_block_ops = []
    default_ops = {default_device: {}}
    heter_ops = {}
    block_index = 0

    current_heter_block_ops = []
    current_default_block_ops = []
    current_heter_device = default_device
    is_heter = False
    for op in block.ops:
        if _is_heter_op(op, current_heter_device, default_device):
            # for gpu/xpu-op
            is_heter = True

            # for cpu-op block append
            if len(current_default_block_ops) > 1:
                default_ops[default_device][
                    block_index] = current_default_block_ops
                program_block_ops.append(current_default_block_ops)
                current_default_block_ops = []
                block_index += 1

            if _is_same_device(op, current_heter_device, default_device):
                # for gpu-op, gpu-op -> gpu-op,...
                current_heter_device = op.attr("op_device")
                _append_heter_op(op, current_heter_block_ops, heter_ops)
            else:
                # for gpu-op -> xpu-op, ...
                op_device = current_heter_block_ops[0].attr("op_device")
                heter_ops[op_device][block_index] = current_heter_block_ops
                program_block_ops.append(current_heter_block_ops)
                block_index += 1
                current_heter_block_ops = []
                current_heter_device = op.attr("op_device")
                _append_heter_op(op, current_heter_block_ops, heter_ops)

        elif is_heter:
            # for gpu/xpu-op -> cpu-op
            op_device = current_heter_block_ops[0].attr("op_device")
            heter_ops[op_device][block_index] = current_heter_block_ops
            program_block_ops.append(current_heter_block_ops)
            block_index += 1
            current_heter_block_ops = []
            current_heter_device = default_device
            is_heter = False
            current_default_block_ops.append(op)
        else:
            # for cpu-op
            current_default_block_ops.append(op)

    if current_default_block_ops != []:
        default_ops[default_device][block_index] = current_default_block_ops
        program_block_ops.append(current_default_block_ops)

    if current_heter_block_ops != []:
        op_device = current_heter_block_ops[0].attr("op_device")
        heter_ops[op_device][block_index] = current_heter_block_ops
        program_block_ops.append(current_heter_block_ops)

    if len(heter_ops) == 0:
        warnings.warn(
            "No heterogeneous OP was found in your program , "
            " please using fluid.device_guard() to run OPs on different device.")

    total_heter_ops = 0
    heter_blocks = 0
    for device in heter_ops.keys():
        heter_block_dict = heter_ops[device]
        heter_blocks += len(heter_block_dict)
        for _, heter_block in heter_block_dict.items():
            total_heter_ops += len(heter_block)
    print(
        "There are {} OPs in your main_program, and contains {} heter-OPs which is made up of {} heter-blocks.".
        format(len(block.ops), total_heter_ops, heter_blocks))

    return origin_porgram, heter_ops, default_ops, program_block_ops


def create_heter_program(program, config, heter_program, program_block_ops_list,
                         heter_ops, block_var_detail, current_device, stage_id):
    # This function mainly includes the following contents:
    # 1. For every heter block:
    #     a) copy heter device op from origin program
    #     b) create variables which belong to heter op:
    #         -> if variable is persistable, clone it in global_scope
    #         -> if variable is temp, create it in heter block
    #     c) create communicate related op as follow:
    #         joint_var.0_1 -> slice -> reshape -> origin_var
    #         origin_var -> origin_program
    #         reshape -> concat -> joint_var.1_2
    #     d) copy send op from origin program for var@grad which loacted in current heter block
    #     e) re-check every op in current blcok if its device is not current heter devie
    # 2. Create send op for step counter in last heter-block
    # 3. Create Listen&Serv OP and Send&Recv OP for distributed training
    # 4. update CompileTimeStrategy for heter_program

    optimizer_block = []
    grad_to_block_id = []
    send_grad_var_list = []

    pre_block_idx = heter_program.num_blocks - 1
    stage_id = int(stage_id)
    print("stage id", stage_id)
    heter_block_ops_forward = program_block_ops_list[stage_id - 1]["forward"]

    heter_block_ops_backward = program_block_ops_list[stage_id - 1]["backward"]

    heter_block = heter_program._create_block(pre_block_idx)
    optimizer_block.append(heter_block)
    for _, op in enumerate(heter_block_ops_forward):
        block_append_op(heter_program, program, heter_block, op)

    entrance_vars = block_var_detail[stage_id - 1]["forward"]["entrance"]
    add_vars_by_var_list(entrance_vars, program, heter_program, heter_block)
    exit_vars = block_var_detail[stage_id - 1]["forward"]["exit"]
    add_vars_by_var_list(exit_vars, program, heter_program, heter_block)

    first_op_index_fp = len(heter_block.ops)

    if stage_id < len(program_block_ops_list):

        heter_block_bp = heter_program._create_block(pre_block_idx)
        optimizer_block.append(heter_block_bp)

        for _, op in enumerate(heter_block_ops_backward):
            block_append_op(heter_program, program, heter_block_bp, op)

        bp_entrance_vars = block_var_detail[stage_id - 1]["backward"][
            "entrance"]
        add_vars_by_var_list(bp_entrance_vars, program, heter_program,
                             heter_block_bp)
        bp_exit_vars = block_var_detail[stage_id - 1]["backward"]["exit"]
        add_vars_by_var_list(bp_exit_vars, program, heter_program,
                             heter_block_bp)
        backward_comm_info = get_communicate_var_info(
            program, stage_id, bp_entrance_vars, type="backward")

        grad_to_block_id.append(backward_comm_info["block_input_var_name"] + ":"
                                + str(heter_block_bp.idx))

    else:
        for _, op in enumerate(heter_block_ops_backward):
            block_append_op(heter_program, program, heter_block, op)

        bp_entrance_vars = block_var_detail[stage_id - 1]["backward"][
            "entrance"]
        add_vars_by_var_list(bp_entrance_vars, program, heter_program,
                             heter_block)
        bp_exit_vars = block_var_detail[stage_id - 1]["backward"]["exit"]
        add_vars_by_var_list(bp_exit_vars, program, heter_program, heter_block)

        heter_block_bp = heter_block

    forward_comm_info = get_communicate_var_info(
        program, stage_id, entrance_vars, type="forward")

    grad_to_block_id.append(forward_comm_info["block_input_var_name"] + ":" +
                            str(heter_block.idx))

    first_op_index_bp = len(heter_block_bp.ops)

    if stage_id <= len(block_var_detail) - 1:
        static_var = insert_communicate_op(program, config, heter_block,
                                           stage_id, first_op_index_fp,
                                           block_var_detail, current_device)
    static_var_bp = insert_communicate_op(
        program, config, heter_block_bp, stage_id, first_op_index_bp,
        block_var_detail, current_device, False)

    # add send op
    send_grad_var_list = add_heter_send_op(
        program, heter_program, heter_block_bp, block_var_detail[stage_id - 1])

    # ---------------
    # add step conter
    send_input_vars = []
    dummy_output = []
    pserver_endpoints = config.get_ps_endpoints()

    # optimizer_block[-1].append_op(
    #     type="send",
    #     inputs={"X": send_input_vars},
    #     outputs={"Out": dummy_output},
    #     attrs={
    #         "send_varnames": [STEP_COUNTER],
    #         "merge_add": True,
    #         "use_send_handler": False,
    #         "endpoints": pserver_endpoints
    #     })

    # add info in listen&serv
    attrs = {
        #"mode": "sync",
        #"trainers": config.get_trainers(),
        #"trainer_id": config.get_role_id() + config.get_trainers(),
        "message_to_block_id": grad_to_block_id,
        "optimize_blocks": optimizer_block,
        # runtime attribute
        "endpoint": config.get_heter_worker_endpoint(),
        "fanin": len(config.get_previous_stage_trainers()),
        "pserver_id": config.get_role_id(),
        "distributed_mode": config.get_distributed_mode(),
        "rpc_exec_thread_num": int(os.getenv("CPU_NUM", 32)),
        RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
    }
    # append the listen_and_serv op
    heter_program.global_block().append_op(
        type="heter_listen_and_serv", inputs={'X': []}, outputs={}, attrs=attrs)
    check_heter_compile_time_strategy(program, config, send_grad_var_list)


def check_heter_compile_time_strategy(program, config, send_grad_var_list):
    origin_grad_var_list = []
    for _, var_grad in config.merged_variables_pairs:
        origin_grad_var_list.append(var_grad.merged_var.name)

    origin_grad_var_list = list(set(origin_grad_var_list))
    send_grad_var_list = list(set(send_grad_var_list))
    useless_grad_var_list = list(
        set(origin_grad_var_list) - set(send_grad_var_list))

    for useless_grad_var in useless_grad_var_list:
        config.remove_var_pair_by_grad(useless_grad_var)


def create_trainer_program(program, origin_program, config,
                           program_block_ops_list, block_var_detail):
    # This function mainly includes the following contents:
    # 1. For every heter block in origin program
    #     a) delete heter op and related variables
    #     b) add send&recv op
    #     c) add communicate ops as follows:
    #         origin_var -> reshape -> concat -> joint_var.0_1
    #         send&recv op(send joint_var.0_1; recv joint_var.1_2)
    #         joint_var.1_2 -> slice -> reshape -> origin_var
    #     d) remove send op which related var@grad is not in trainer program
    # 2. check every op's device
    static_var = []
    for heter_block_index in range(1, len(program_block_ops_list)):
        ops_list = program_block_ops_list[heter_block_index][
            "forward"] + program_block_ops_list[heter_block_index]["backward"]
        static_var += replace_ops_by_communicate_op(
            program, config, heter_block_index, ops_list, block_var_detail)
        remove_trainer_send_op(program, config, heter_block_index,
                               block_var_detail)

    optimizer_block = []
    grad_to_block_id = []

    bp_ops_list = program_block_ops_list[0]["backward"]
    delete_same_ops(program.global_block(), bp_ops_list)
    delete_trainer_useless_var(config, program, static_var)
    backward_block = create_backward_block(program, origin_program, config,
                                           bp_ops_list, block_var_detail)

    bp_entrance_vars = block_var_detail[0]["backward"]["entrance"]
    backward_comm_info = get_communicate_var_info(
        origin_program, 1, bp_entrance_vars, type="backward")

    grad_to_block_id.append(backward_comm_info["block_input_var_name"] + ":" +
                            str(backward_block.idx))
    optimizer_block.append(backward_block)

    attrs = {
        #"mode": "sync",
        #"trainers": config.get_trainers(),
        #"trainer_id": config.get_role_id(),
        "message_to_block_id": grad_to_block_id,
        "optimize_blocks": optimizer_block,
        # runtime attribute
        "endpoint": config.get_trainer_endpoint(),  ## get trainer endpoint
        "fanin": 0,  ## get heter worker
        "pserver_id": config.get_role_id(),
        "distributed_mode": config.get_distributed_mode(),
        "rpc_exec_thread_num": int(os.getenv("CPU_NUM", 32)),
        RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
    }
    # append the listen_and_serv op
    program.global_block()._insert_op(
        index=0,
        type="heter_listen_and_serv",
        inputs={'X': []},
        outputs={},
        attrs=attrs)

    ## TODO add check for bp block
    check_op_device(program.global_block(), DEFAULT_DEVICE)


def insert_communicate_op(orign_program,
                          config,
                          heter_block,
                          stage_id,
                          first_op_index,
                          block_var_detail,
                          device,
                          is_forward=True):

    if is_forward:
        next_heter_worker_endpoints = config.get_next_stage_trainers()
        previous_heter_worker_endpoints = config.get_previous_stage_trainers()
        entrance_var = block_var_detail[stage_id]["forward"]["entrance"]
        comm_info = get_communicate_var_info(orign_program, stage_id + 1,
                                             entrance_var)

    else:
        next_heter_worker_endpoints = config.get_next_stage_trainers()
        #if next_heter_worker_endpoints == "":
        #    next_heter_worker_endpoints = []
        previous_heter_worker_endpoints = config.get_previous_stage_trainers()
        entrance_var = block_var_detail[stage_id - 1]["backward"]["exit"]
        comm_info = get_communicate_var_info(orign_program, stage_id - 1,
                                             entrance_var, "backward")

    heter_block._insert_op(
        index=first_op_index,
        type="send_and_recv",
        inputs={"X": heter_block.vars[entrance_var[0]]},
        outputs={"Out": []},
        attrs={
            "mode": "forward" if is_forward else "backward",
            "send_var_name": entrance_var + ["microbatch_id"],
            "recv_var_name": [],
            "message_name": comm_info["block_input_var_name"],
            "next_endpoints": next_heter_worker_endpoints,
            "previous_endpoints": previous_heter_worker_endpoints,
            "trainer_id": config.get_role_id(),
            "op_device": device,
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
        })

    return entrance_var


def create_backward_block(program, origin_program, config, bp_ops_list,
                          block_var_detail):
    pre_block_idx = program.num_blocks - 1
    heter_block = program._create_block(pre_block_idx)

    for _, op in enumerate(bp_ops_list):
        if op.type == "send":
            send_varnames = op.attr('send_varnames')
            is_skip = False
            for varname in send_varnames:
                if varname not in program.global_block(
                ).vars and varname not in heter_block.vars:
                    is_skip = True
                    break
            if is_skip == True:
                continue
        block_append_op(program, origin_program, heter_block, op)

    entrance_vars = block_var_detail[0]["backward"]["entrance"]
    add_vars_by_var_list(entrance_vars, origin_program, program, heter_block)
    exit_vars = block_var_detail[0]["backward"]["exit"]
    add_vars_by_var_list(exit_vars, origin_program, program, heter_block)
    return heter_block


def replace_ops_by_communicate_op(program, config, heter_block_index, ops_list,
                                  block_var_detail):
    all_op = program.global_block().ops
    start_op = ops_list[0]
    first_op_idx = -1
    for op in all_op:
        if is_same_op(op, start_op):
            first_op_idx = all_op.index(op)
            break
    assert first_op_idx != -1
    delete_same_ops(program.global_block(), ops_list)

    entrance_var = []

    if heter_block_index == 1:
        mode = config.get_distributed_mode()
        next_heter_worker_endpoints = config.get_next_stage_trainers()

        entrance_var = block_var_detail[heter_block_index]["forward"][
            "entrance"]

        comm_info = get_communicate_var_info(program, heter_block_index + 1,
                                             entrance_var)
        program.global_block()._insert_op(
            index=first_op_idx,
            type="send_and_recv",
            inputs={"X": program.global_block().vars[entrance_var[0]]},
            outputs={"Out": []},
            attrs={
                "mode": "forward",
                "send_var_name": entrance_var + ["microbatch_id"],
                "recv_var_name": [],
                "message_name": comm_info["block_input_var_name"],
                "next_endpoints": next_heter_worker_endpoints,
                "previous_endpoints": [],
                "trainer_id": config.get_role_id(),
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

    return entrance_var


def remove_trainer_send_op(program, config, heter_block_index,
                           block_var_detail):

    # if trainer do FF->BP->SEND, it has follow vars: var, var@GRAD
    # if trainer only do SEND, it has one var: var@GRAD
    # Delete Send op ,if trainer doesn't has pair var (var<->var@GRAD)
    persistables = block_var_detail[heter_block_index]["forward"]["persistables"] + \
                   block_var_detail[heter_block_index]["backward"]["persistables"]
    need_remove_send_op = []
    need_remove_grad_var = []
    for op in find_send_op(program):
        input_list, _ = find_op_input_output(program,
                                             program.global_block(), op)
        for var_name in input_list:
            origin_var_name = var_name.split("@GRAD")[0]
            if origin_var_name in persistables:
                need_remove_send_op.append(op)
                need_remove_grad_var.append(var_name)
    need_remove_send_op = list(set(need_remove_send_op))
    delete_ops(program.global_block(), need_remove_send_op)
    for grad_var_name in need_remove_grad_var:
        config.remove_var_pair_by_grad(grad_var_name)


def add_heter_send_op(program, heter_program, block, block_var_detail):
    def _get_send_op_dict():
        send_op_dict = {}
        send_op_list = find_send_op(program)
        for op in send_op_list:
            input_list, _ = find_op_input_output(program,
                                                 program.global_block(), op)
            for var in input_list:
                send_op_dict[var] = op
        return send_op_dict

    # send_Op = { inputs{'X':[]},
    #             outputs{'Out':dummy_output},
    #             attrs{'send_varnames'"[]",
    #                   'is_sparse':int,
    #                   'table_id':int } }
    send_grad_var_list = []
    send_op_dict = _get_send_op_dict()
    table_dict = {}
    for persistable_var in block_var_detail["backward"]["persistables"]:
        # check var_name ==  var@GRAD
        if "@GRAD" not in persistable_var:
            continue
        if "GRAD" != persistable_var.split("@")[-1]:
            continue
        if persistable_var not in send_op_dict:
            continue
        send_op = send_op_dict[persistable_var]
        is_sparse = send_op.attr('is_sparse')
        table_id = send_op.attr('table_id')
        send_varnames = send_op.attr('send_varnames')
        send_grad_var_list.append(persistable_var)
        if table_id not in table_dict:
            table_dict[table_id] = {}
            table_dict[table_id]['var_list'] = []
            table_dict[table_id]['is_sparse'] = is_sparse
            table_dict[table_id]['send_varnames'] = send_varnames
        table_dict[table_id]['var_list'].append(persistable_var)

    for table_id in table_dict:
        dummy_output = block.create_var(
            name=framework.generate_control_dev_var_name())
        send_input_vars = [
            block.vars[union_var]
            for union_var in table_dict[table_id]['var_list']
        ]
        block.append_op(
            type="send",
            inputs={"X": send_input_vars},
            outputs={"Out": dummy_output},
            attrs={
                "send_varnames": table_dict[table_id]['send_varnames'],
                "is_sparse": is_sparse,
                "table_id": table_id,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

    return send_grad_var_list


def find_send_op(program):
    send_op_list = []
    for op in program.global_block().ops:
        if op.type == "send":
            send_op_list.append(op)
    return send_op_list


def get_communicate_var_info(program,
                             block_index,
                             entrance_var_list,
                             type="forward"):
    input_var_reshape_dim = []
    input_var_reshape_name = []

    if type == "forward":
        block_input_var_name = "forward_joint_{}_{}@Heter".format(
            block_index - 1, block_index)
    else:
        block_input_var_name = "backward_joint_{}_{}@Heter".format(
            block_index + 1, block_index)

    entrance_var_list.sort()
    # input
    # Heter_SERVER_BLOCK_index@JOINT_VAR -> slice -> var@Heter_SERVER_BLOCK@INPUT_RESHAPE_VAR -> reshape -> var
    for name in entrance_var_list:
        var = program.global_block().vars[name]
        shape = var.shape
        # if len(shape) < 2 or shape[0] != -1:
        #     raise ValueError(
        #         "Variable {} not support heter training. its shape is {}".
        #         format(name, shape))
        recv_var_dim = -1 * reduce(lambda x, y: x * y, shape)
        input_var_reshape_dim.append(recv_var_dim)
        input_var_reshape_name.append("{}.input_reshape@Heter".format(name))

    # output
    # var -> reshape -> var@Heter_SERVER_BLOCK@INPUT_RESHAPE_VAR -> concat -> Heter_SERVER_BLOCK_index@JOINT_VAR
    #for var_name in exit_var_list:
    #    var = program.global_block().vars[var_name]
    #    shape = var.shape
    #    # if len(shape) < 2 or shape[0] != -1:
    #    #     raise ValueError(
    #    #         "Variable {} not support heter training. its shape is {}".
    #    #         format(var_name, shape))
    #    send_reshape_dim = -1 * reduce(lambda x, y: x * y, shape)
    #    output_var_reshape_dim.append(send_reshape_dim)
    #    output_var_reshape_name.append("{}.output_reshape@Heter".format(
    #        var_name))

    info = {
        "input_var_reshape_dim": input_var_reshape_dim,
        "input_var_reshape_name": input_var_reshape_name,
        "block_input_var_name": block_input_var_name,
        #    "output_var_reshape_dim": output_var_reshape_dim,
        #    "output_var_reshape_name": output_var_reshape_name,
        #    "block_output_var_name": block_output_var_name
    }

    return info


def union_forward_gradient_op(program_block_ops_list):
    """
    before analyzing the input & output of each block in program_block_list, we should
    union the forward op and corresponding gradient op to elimincate the uneccessary variable
    transmit
    """
    """
    fix for 2emb model, re-place sum op

    """
    block_length = len(program_block_ops_list)
    '''
    ## get the final part
    final_part_idx = -1 
    for i in range(block_length):
        op_list = program_block_ops_list[i]
        for op in op_list:
           if "_grad" in op.type:
              final_part_idx = i
              break
        if final_part_idx != -1:
            break
    
    ## eliminate wrong partition because of sum op
    ## lookup_table_v2_grad
    ## every looup_table_v2_grad op block should follow a sum op
    var2idx  = {}

    for i in range(final_part_idx, block_length):
        op_list = program_block_ops_list[i]
        for j in range(len(op_list) - 1, -1, -1):
            op = op_list[j]
            #if op.type == "lookup_table_v2_grad":
            #   if j < len(op_list) - 1):
            #   else:
            #      ## get var and record place
            if _grad in op.type:
                forward_op_type = op.type.split("_grad")[0]
                if forward_op_type in SPARSE_OP_TYPE_DICT.keys() \
                    and op.attr('remote_prefetch') is True:
                    param_name = op.input(SPARSE_OP_TYPE_DICT[forward_op_type])[0]
                    
                    var2idx[] = [i,j] ## 
    
    '''

    union_program_block_ops_list = []
    assert block_length % 2 != 0, "the length of program_block_ops_list should be odd"
    for i in range(0, block_length // 2):
        block_op_list = {"forward": program_block_ops_list[i]}
        block_op_list.update({
            "backward": program_block_ops_list[block_length - 1 - i]
        })
        union_program_block_ops_list.append(block_op_list)

    block_op_list = {"forward": [], "backward": []}
    for op in program_block_ops_list[block_length // 2]:
        if not "_grad" in op.type and not (op.type == "sum"):
            block_op_list["forward"].append(op)
        else:
            block_op_list["backward"].append(op)
    union_program_block_ops_list.append(block_op_list)
    return union_program_block_ops_list


def find_block_joints(program, program_block_ops_list, heter_ops):
    block_var_detail = find_entrance_exit_private(program,
                                                  program_block_ops_list)
    block_var_detail = entrance_exit_check(program, program_block_ops_list,
                                           block_var_detail, heter_ops)
    block_var_detail = delete_block_useless_exit(
        program, program_block_ops_list, block_var_detail)

    return block_var_detail


def find_entrance_exit_private(program, program_block_ops_list):
    block_var_detail = []
    persistables = []
    for index, block_op_list in enumerate(program_block_ops_list):
        ## forward
        block_input, block_output = find_ops_list_input_output(
            program, block_op_list["forward"])
        persistables = screen_persistables(
            program, block_input) + screen_persistables(program, block_output)
        # find entrance & exit
        block_private_vars = list(set(block_input) & set(block_output))
        block_entrance = list(set(block_input) - set(block_private_vars))
        block_exit = list(set(block_output) - set(block_private_vars))
        detail = {
            "forward": {
                "entrance": block_entrance,
                "exit": block_exit,
                "private": block_private_vars,
                "persistables": persistables
            }
        }

        ## backward
        bp_block_input, bp_block_output = find_ops_list_input_output(
            program, block_op_list["backward"])
        bp_persistables = screen_persistables(
            program, bp_block_input) + screen_persistables(program,
                                                           bp_block_output)
        # find entrance & exit
        bp_block_private_vars = list(set(bp_block_input) & set(bp_block_output))
        bp_block_entrance = list(
            set(bp_block_input) - set(bp_block_private_vars))
        bp_block_exit = list(set(bp_block_output) - set(bp_block_private_vars))
        detail.update({
            "backward": {
                "entrance": bp_block_entrance,
                "exit": bp_block_exit,
                "private": bp_block_private_vars,
                "persistables": bp_persistables
            }
        })
        block_var_detail.append(detail)
    return block_var_detail


def entrance_exit_check(program, program_block_ops_list, block_var_detail,
                        heter_ops):
    for index in range(len(block_var_detail) - 1, -1, -1):
        if index - 1 < 0:
            break
        previous_block_exit = block_var_detail[index - 1]["forward"]["exit"]
        previous_block_exit.sort()
        current_block_entrance = block_var_detail[index]["forward"]["entrance"]

        backward_entrance = block_var_detail[index]["backward"]["entrance"]

        forward_all = block_var_detail[index]["forward"][
            "entrance"] + block_var_detail[index]["forward"][
                "private"] + block_var_detail[index]["forward"]["exit"]

        for var in backward_entrance:
            if not ("@GRAD" in var) and not (var in forward_all):
                current_block_entrance.append(var)

        current_block_entrance.sort()

        if previous_block_exit == current_block_entrance:
            continue
        exist_vars = list(
            set(previous_block_exit) & set(current_block_entrance))
        need_add_vars = list(set(current_block_entrance) - set(exist_vars))
        # var in different stage should not be ignored, since they are not placed in the same program & device
        #need_add_vars = find_need_var_from_previous_block(
        #    need_add_vars, block_var_detail, index, heter_ops)

        previous_block_private = block_var_detail[index - 1]["forward"][
            "private"]
        previous_block_entrance = block_var_detail[index - 1]["forward"][
            "entrance"]
        for var in need_add_vars:
            if var not in previous_block_private and var not in previous_block_entrance:
                previous_block_entrance.append(var)
            previous_block_exit.append(var)
            if not var in current_block_entrance:
                current_block_entrance.append(var)

    for index in range(0, len(block_var_detail) - 1, 1):
        previous_block_exit = block_var_detail[index + 1]["backward"]["exit"]
        previous_block_exit.sort()
        current_block_entrance = block_var_detail[index]["backward"]["entrance"]

        current_block_entrance.sort()

        if previous_block_exit == current_block_entrance:
            continue
        exist_vars = list(
            set(previous_block_exit) & set(current_block_entrance))
        need_add_vars = list(set(current_block_entrance) - set(exist_vars))
        need_ignore_vars = []
        for var in need_add_vars:
            if not "@GRAD" in var:
                need_ignore_vars.append(var)
        need_add_vars = list(
            set(need_add_vars).difference(set(need_ignore_vars)))
        previous_block_private = block_var_detail[index + 1]["backward"][
            "private"]
        previous_block_entrance = block_var_detail[index + 1]["backward"][
            "entrance"]
        for var in need_add_vars:
            if var not in previous_block_private and var not in previous_block_entrance:
                previous_block_entrance.append(var)
            previous_block_exit.append(var)
    return block_var_detail


def find_need_var_from_previous_block(need_add_vars, block_var_detail,
                                      current_index, heter_ops):
    # create index_device_map
    index_device_map = {}
    for index in range(len(block_var_detail)):
        index_device_map[index] = DEFAULT_DEVICE
    for device in heter_ops:
        for index in heter_ops[device].keys():
            if index < len(block_var_detail):
                index_device_map[index] = device

    pre_index = current_index - 1
    need_ignore_var = []

    # if need_add_var in current device, no need communicate
    for var in need_add_vars:
        while (pre_index >= 0):
            previous_block_private = block_var_detail[pre_index]["private"]
            previous_block_exit = block_var_detail[pre_index]["exit"]
            previous_block_entrance = block_var_detail[pre_index]["entrance"]
            total_var = previous_block_private + previous_block_exit + previous_block_entrance
            if var in total_var:
                if index_device_map[current_index] == index_device_map[
                        pre_index] and index_device_map[
                            current_index] == DEFAULT_DEVICE:
                    need_ignore_var.append(var)
                    break
            pre_index -= 1

    need_add_vars = list(set(need_add_vars).difference(set(need_ignore_var)))
    return need_add_vars


def delete_block_useless_exit(program, program_block_ops_list,
                              block_var_detail):
    ## forward
    for index in range(len(block_var_detail)):
        if index == len(block_var_detail) - 1:
            break
        current_block_exit = block_var_detail[index]["forward"]["exit"]
        next_block_entrance = block_var_detail[index + 1]["forward"]["entrance"]
        need_delete_var = []
        for var in current_block_exit:
            if var not in next_block_entrance:
                need_delete_var.append(var)

        for var in need_delete_var:
            current_block_exit.remove(var)
    ## backward
    for index in range(len(block_var_detail) - 1, -1, -1):
        if index - 1 < 0:
            break
        current_block_exit = block_var_detail[index]["backward"]["exit"]
        next_block_entrance = block_var_detail[index - 1]["backward"][
            "entrance"]
        need_delete_var = []
        for var in current_block_exit:
            if var not in next_block_entrance:
                need_delete_var.append(var)
        for var in need_delete_var:
            current_block_exit.remove(var)

    return block_var_detail


def check_op_device(block, device):
    for op in block.ops:
        op._set_attr('op_device', device)


def screen_persistables(program, var_list):
    need_remove = []
    for var_name in var_list:
        if "@GRAD" in var_name:
            if "GRAD" != var_name.split("@")[-1]:
                continue
            origin_var_name = var_name.split("@GRAD")[0]
            var = program.global_block().vars[origin_var_name]
        else:
            var = program.global_block().vars[var_name]

        if fluid.io.is_persistable(var):
            need_remove.append(var_name)

    for var_name in need_remove:
        var_list.remove(var_name)
    return need_remove


def insert_reshape_op(program,
                      block,
                      index,
                      var_name,
                      new_var_name,
                      new_var_shape=None):
    input_var = block.vars[var_name]

    if new_var_name not in block.vars:
        out = block.create_var(
            name=new_var_name,
            shape=new_var_shape,
            dtype=input_var.dtype,
            type=input_var.type)
    else:
        out = block.vars[new_var_name]
        new_var_shape = out.shape

    x_shape = block.create_var(
        name="{}.xshape@Heter".format(var_name), dtype=input_var.dtype)
    block._insert_op(
        index=index,
        type="reshape2",
        inputs={"X": input_var},
        attrs={'shape': new_var_shape},
        outputs={"Out": out,
                 "XShape": x_shape})


def insert_send_concat_op(program, block, index, var_name_list, new_var_name,
                          new_var_shape):
    input_var_list = [block.vars[var_name] for var_name in var_name_list]

    out = program.global_block().create_var(
        name=new_var_name,
        shape=new_var_shape,
        dtype=input_var_list[0].dtype,
        type=input_var_list[0].type)

    block._insert_op(
        index=index,
        type='concat',
        inputs={"X": input_var_list},
        outputs={'Out': [out]},
        attrs={'axis': -1,
               'use_stack': False})


def insert_recv_slice_op(program, block, index, var_name, var_shape, dtype,
                         type, new_var_name_list, new_var_shape_list):

    if var_name not in program.global_block().vars:
        input_var = program.global_block().create_var(
            name=var_name, shape=var_shape, dtype=dtype, type=type)
    else:
        input_var = program.global_block().vars[var_name]

    out_list = []
    for i in range(len(new_var_name_list)):
        if new_var_name_list[i] not in block.vars:
            out = block.create_var(
                name=new_var_name_list[i],
                shape=new_var_shape_list[i],
                dtype=input_var.dtype,
                type=input_var.type)
        else:
            out = block.vars[new_var_name_list[i]]
        out_list.append(out)

    start_index = 0
    end_index = 0
    for i in range(len(new_var_name_list)):
        starts = []
        ends = []
        attrs = {'axes': [1]}
        end_index += new_var_shape_list[i][1]
        starts.append(start_index)
        ends.append(end_index)
        attrs['starts'] = starts
        attrs['ends'] = ends

        block._insert_op(
            index=index,
            type='slice',
            inputs={'Input': input_var},
            attrs=attrs,
            outputs={'Out': out_list[i]})
        start_index = end_index
        index += 1


def add_heter_trainer_useful_vars(config, program, heter_program, heter_block,
                                  static_var):
    static_var = list(set(static_var))
    for var_name in static_var:
        if var_name not in heter_program.global_block(
        ).vars and var_name not in heter_block.vars:
            var = program.global_block().vars[var_name]
            if var.persistable:
                heter_program.global_block()._clone_variable(
                    var, force_persistable=False)
            else:
                heter_block._clone_variable(var, force_persistable=False)


def delete_trainer_useless_var(config, program, static_var):
    static_var = list(set(static_var))
    program_useful_var_list = []
    for op in program.global_block().ops:
        input_var_list, output_var_list = find_op_input_output(
            program, program.global_block(), op)
        op_var_list = list(set(input_var_list).union(set(output_var_list)))
        program_useful_var_list = list(
            set(program_useful_var_list).union(set(op_var_list)))
    program_useful_var_list += static_var
    program_useless_var_list = list(
        set(get_vars_name_in_block(program.global_block())).difference(
            set(program_useful_var_list)))
    for var in program_useless_var_list:
        program.global_block()._remove_var(var)
    return program_useless_var_list


def block_append_op(program, origin_program, block, op):

    merge_ordereddict = origin_program.global_block().vars.copy()
    merge_ordereddict.update(block.vars)
    inputs = _get_input_map_from_op(merge_ordereddict, op)
    for key, varlist in six.iteritems(inputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if var.name not in program.global_block(
            ).vars and var.name not in block.vars:
                if var.persistable:
                    program.global_block()._clone_variable(
                        var, force_persistable=False)
                else:
                    block._clone_variable(var, force_persistable=False)

    outputs = _get_output_map_from_op(origin_program.global_block().vars, op)
    for key, varlist in six.iteritems(outputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if var.name not in program.global_block(
            ).vars and var.name not in block.vars:
                if var.persistable:
                    program.global_block()._clone_variable(
                        var, force_persistable=False)
                else:
                    block._clone_variable(var, force_persistable=False)

    if "_grad" not in op.type:
        # for forward op
        return block.append_op(
            type=op.type, inputs=inputs, outputs=outputs, attrs=op.all_attrs())
    else:
        # for grad op
        op_desc = op.desc
        op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()

        # append grad op
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(op_desc)
        new_op_desc._set_attr(op_role_attr_name, backward)

        # set device gard
        if op.desc.has_attr(device_attr_name):
            op_device = op_desc.attr(device_attr_name)
            new_op_desc._set_attr(device_attr_name, op_device)
        block._sync_with_cpp()


def add_vars_by_var_list(var_name_list, origin_program, program, block):
    for var_name in var_name_list:
        if var_name not in program.global_block(
        ).vars and var_name not in block.vars:
            var = origin_program.global_block().vars[var_name]
            if var.persistable:
                program.global_block()._clone_variable(
                    var, force_persistable=False)
            else:
                block._clone_variable(var, force_persistable=False)


def get_varlist_from_op_map(var_map):
    var_list = []
    for key, varlist in six.iteritems(var_map):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for i in range(len(varlist)):
            var = varlist[i]
            var_list.append(var.name)
    return var_list


def find_ops_list_input_output(program, ops_list):
    input_var_list = []
    output_var_list = []
    for op in ops_list:
        inputs = _get_input_map_from_op(program.global_block().vars, op)
        input_var_list += get_varlist_from_op_map(inputs)
        outputs = _get_output_map_from_op(program.global_block().vars, op)
        output_var_list += get_varlist_from_op_map(outputs)

    input_var_list = list(set(input_var_list))
    output_var_list = list(set(output_var_list))
    return input_var_list, output_var_list


def find_op_input_output(program, block, op):
    input_var_list = []
    output_var_list = []
    inputs = _get_input_map_from_op(block.vars, op)
    input_var_list += get_varlist_from_op_map(inputs)
    outputs = _get_output_map_from_op(block.vars, op)
    output_var_list += get_varlist_from_op_map(outputs)
    input_var_list = list(set(input_var_list))
    output_var_list = list(set(output_var_list))
    return input_var_list, output_var_list


def get_vars_name_in_block(block):
    vars_list = block.vars.keys()
    vars_name_list = [var_name for var_name in vars_list]
    return vars_name_list


def is_same_op(op1, op2):
    if str(op1) != str(op2):
        return False
    return True


def _get_input_map_from_op(varmap, op):
    """Returns a dict from op input name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.input_names:
        vars = []
        for varname in op.input(key):
            if varname == "@EMPTY@":
                continue
            if "lod_tensor_blocking_queue" in varname:
                continue
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def _get_output_map_from_op(varmap, op):
    """Returns a dict from op output name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.output_names:
        vars = []
        for varname in op.output(key):
            if varname == "@EMPTY@":
                continue
            if "lod_tensor_blocking_queue" in varname:
                continue
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def delete_same_ops(block, ops):
    for op in ops:
        try:
            for origin_op in block.ops:
                if is_same_op(origin_op, op):
                    idx = list(block.ops).index(origin_op)
                    block._remove_op(idx)
                    break
        except Exception as e:
            print(e)
