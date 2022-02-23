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

import collections
import six

from paddle.fluid import core
from paddle.fluid.framework import Block

from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _orig_varname
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_varname_parts
from paddle.fluid.incubate.fleet.parameter_server.ir.public import is_distributed_sparse_op
from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablename
from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_lr_ops

LEARNING_RATE_DECAY_COUNTER = "@LR_DECAY_COUNTER@"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched


def _is_optimizer_op(op):
    if "Param" in op.input_names and \
            "LearningRate" in op.input_names:
        return True
    return False


def _same_or_split_var(p_name, var_name):
    return p_name == var_name or p_name.startswith(var_name + ".block")


def _get_optimizer_input_shape(op_type, varkey, orig_shape, param_shape):
    """
    Returns the shape for optimizer inputs that need to be reshaped when
    Param and Grad is split to multiple servers. 
    """
    # HACK(typhoonzero) : Should use functions of corresponding optimizer in
    # optimizer.py to get the shape, do not bind this in the transpiler.
    if op_type == "adam":
        if varkey in ["Moment1", "Moment2"]:
            return param_shape
    elif op_type == "adagrad":
        if varkey == "Moment":
            return param_shape
    elif op_type == "adamax":
        if varkey in ["Moment", "InfNorm"]:
            return param_shape
    elif op_type in ["momentum", "lars_momentum"]:
        if varkey == "Velocity":
            return param_shape
    elif op_type == "rmsprop":
        if varkey in ["Moment", "MeanSquare"]:
            return param_shape
    elif op_type == "decayed_adagrad":
        if varkey == "Moment":
            return param_shape
    elif op_type == "ftrl":
        if varkey in ["SquaredAccumulator", "LinearAccumulator"]:
            return param_shape
    elif op_type == "sgd":
        pass
    else:
        raise ValueError(
            "Not supported optimizer for distributed training: %s" % op_type)
    return orig_shape


def _append_pserver_non_opt_ops(optimize_block, opt_op, origin_program, config):
    def _get_pserver_grad_param_var(var, var_dict):
        """
        Return pserver side grad/param variable, return None
        if the variable is not grad/param, e.g.

            a@GRAD -> a@GRAD.block0
            a@GRAD -> a@GRAD (a is not split)
            fc_0.w_0 -> fc_0.w_0.block_0
            fc_0.w_0 -> fc_0.w_0 (weight is not split)
            _generated_var_123 -> None
        """

        grad_block = None
        for _, g in six.iteritems(var_dict):
            if _orig_varname(g.name) == _orig_varname(var.name):
                # skip per trainer vars
                if g.name.find(".trainer_") == -1:
                    # only param or grads have split blocks
                    ovar_name = _orig_varname(g.name)
                    if ovar_name in config.param_grad_ep_mapping:
                        grad_block = g
                        break
                    elif ovar_name in config.grad_param_mapping:
                        grad_block = g
                        break

        return grad_block

    program = optimize_block.program
    # Append the ops for parameters that do not need to be optimized / updated
    inputs = _get_input_map_from_op(origin_program.global_block().vars, opt_op)
    for key, varlist in six.iteritems(inputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for i in range(len(varlist)):
            var = varlist[i]
            # for ops like clipping and weight decay, get the split var(xxx.block0)
            # for inputs / outputs
            grad_block = _get_pserver_grad_param_var(
                var, program.global_block().vars)
            if grad_block:
                varlist[i] = grad_block
            elif var.name not in program.global_block().vars:
                tmpvar = program.global_block()._clone_variable(var)
                varlist[i] = tmpvar
            else:
                varlist[i] = program.global_block().vars[var.name]
        inputs[key] = varlist

    outputs = _get_output_map_from_op(origin_program.global_block().vars,
                                      opt_op)
    for key, varlist in six.iteritems(outputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for i in range(len(varlist)):
            var = varlist[i]
            grad_block = _get_pserver_grad_param_var(
                var, program.global_block().vars)
            if grad_block:
                varlist[i] = grad_block
            elif var.name not in program.global_block().vars:
                tmpvar = program.global_block()._clone_variable(var)
                varlist[i] = tmpvar
            else:
                varlist[i] = program.global_block().vars[var.name]
        outputs[key] = varlist

    return optimize_block.append_op(
        type=opt_op.type,
        inputs=inputs,
        outputs=outputs,
        attrs=opt_op.all_attrs())


def _append_pserver_ops(optimize_block, opt_op, endpoint, grad_to_block_id,
                        origin_program, merged_var, sparse_grad_to_param,
                        config):
    program = optimize_block.program
    pserver_block = program.global_block()
    new_inputs = collections.OrderedDict()

    def _get_param_block(opt_op):
        # param is already created on global program
        unmerged_vars = []
        merged_vars = []
        merged_ordervars = []

        param_vars = [
            p for p in config.param_grad_ep_mapping[endpoint]["params"]
        ]

        for var in param_vars:
            name = var.name
            orig_varname = _orig_varname(name)

            for pairs in config.merged_variables_pairs:
                merged_p = pairs[0]
                if merged_p.merged_var.name == orig_varname:
                    if merged_p.merged_var.name == merged_p.ordered_vars[
                            0].name:
                        unmerged_vars.append(merged_p.ordered_vars[0])
                    else:
                        merged_vars.append(merged_p.merged_var)
                        merged_ordervars.append(merged_p.ordered_vars[0])
                    break

        param_name = opt_op.input("Param")[0]

        for i in range(len(unmerged_vars)):
            if _same_or_split_var(param_name, unmerged_vars[i].name):
                for var in param_vars:
                    if _same_or_split_var(var.name, unmerged_vars[i].name):
                        return var

        for i in range(len(merged_ordervars)):
            if _same_or_split_var(param_name, merged_ordervars[i].name):
                for var in param_vars:
                    if _same_or_split_var(var.name, merged_vars[i].name):
                        return var
        return None

    for key in opt_op.input_names:
        if key == "Grad":
            # Note !!This is for l2decay on sparse gradient, \
            # because it will create a new tensor for
            # decayed gradient but not inplace modify the origin one
            origin_grad_name = opt_op.input(key)[0]
            if core.kNewGradSuffix(
            ) in origin_grad_name and pserver_block.has_var(origin_grad_name):
                new_grad = pserver_block.var(origin_grad_name)
                new_inputs[key] = new_grad
            else:
                new_inputs[key] = merged_var
        elif key == "Param":
            param_block = _get_param_block(opt_op)

            if not param_block:
                return
            tmpvar = pserver_block.create_var(
                name=param_block.name,
                persistable=True,
                dtype=param_block.dtype,
                shape=param_block.shape)
            new_inputs[key] = tmpvar

        elif key == "LearningRate":
            # learning rate variable has already be created by non - optimize op,
            # don't create it once again.
            lr_varname = opt_op.input(key)[0]
            if lr_varname in pserver_block.vars:
                new_inputs[key] = pserver_block.vars[opt_op.input(key)[0]]
            else:
                origin_var = origin_program.global_block().vars[lr_varname]
                tmpvar = pserver_block.create_var(
                    name=origin_var.name,
                    persistable=origin_var.persistable,
                    dtype=origin_var.dtype,
                    shape=origin_var.shape)
                new_inputs[key] = tmpvar

    for key in opt_op.input_names:
        new_shape = None
        if key in [
                "Param", "Grad", "LearningRate", "MasterParam", "Beta1Tensor",
                "Beta2Tensor"
        ]:
            continue
        var = origin_program.global_block().vars[opt_op.input(key)[0]]
        param_var = new_inputs["Param"]
        # update accumulator variable shape
        new_shape = _get_optimizer_input_shape(opt_op.type, key, var.shape,
                                               param_var.shape)
        tmpvar = pserver_block.create_var(
            name=var.name,
            persistable=var.persistable,
            dtype=var.dtype,
            shape=new_shape)
        new_inputs[key] = tmpvar

    # change output's ParamOut variable
    outputs = _get_output_map_from_op(origin_program.global_block().vars,
                                      opt_op)
    outputs["ParamOut"] = new_inputs["Param"]
    optimize_block.append_op(
        type=opt_op.type,
        inputs=new_inputs,
        outputs=outputs,
        attrs=opt_op.all_attrs())

    # record sparse grad to param name
    if new_inputs["Grad"].type == core.VarDesc.VarType.SELECTED_ROWS:
        sparse_grad_to_param.append(
            str(new_inputs["Grad"].name) + ":" + str(new_inputs["Param"].name))


def _get_input_map_from_op(varmap, op):
    """Returns a dict from op input name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.input_names:
        vars = []
        for varname in op.input(key):
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
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def get_op_by_type(block, op_type):
    for op in block.ops:
        if op.type == op_type:
            return op
    raise ValueError("add_listen_and_serv_pass must at first")


def add_listen_and_serv_pass(program, config):
    attrs = {
        "grad_to_block_id": None,
        "sparse_grad_to_param": None,
        "lr_decay_block_id": None,
        "dense_optimize_blocks": None,
        "sparse_optimize_blocks": None,

        # runtime attribute
        "endpoint": config.get_ps_endpoint(),
        "pserver_id": config.get_role_id(),
        "Fanin": config.get_trainers(),
        "distributed_mode": config.get_distributed_mode(),
        "rpc_get_thread_num": -1,
        "rpc_send_thread_num": -1,
        "rpc_prefetch_thread_num": -1
    }

    # step5 append the listen_and_serv op
    program.global_block().append_op(
        type="listen_and_serv", inputs={'X': []}, outputs={}, attrs=attrs)

    return program


def add_rpc_global_flags_pass(program, config):
    server_runtime = config.get_server_runtime_config()
    send_threads = server_runtime._rpc_send_thread_num
    get_threads = server_runtime._rpc_get_thread_num
    pull_threads = server_runtime._rpc_prefetch_thread_num

    op = get_op_by_type(program.global_block(), "listen_and_serv")

    if get_threads < 1 or send_threads < 1 or pull_threads < 1:
        raise ValueError(
            "error arguments in get_threads/send_threads/pull_threads")

    op._set_attr("rpc_get_thread_num", get_threads)
    op._set_attr("rpc_send_thread_num", send_threads)
    op._set_attr("rpc_prefetch_thread_num", pull_threads)

    return program


def _clone_var(block, var, persistable=True):
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=persistable)


def add_optimizer_pass(program, config):
    def _append_pserver_grad_merge_ops(optimize_block, grad_varname_for_block,
                                       endpoint, grad_to_block_id):
        trainers = config.get_trainers()

        program = optimize_block.program
        pserver_block = program.global_block()
        grad_block = None

        for g in config.param_grad_ep_mapping[endpoint]["grads"]:
            if _orig_varname(g.name) == \
                    _orig_varname(grad_varname_for_block):
                grad_block = g
                break

        if not grad_block:
            # do not append this op if current endpoint
            # is not dealing with this grad block
            return None

        orig_varname, block_name, trainer_name = _get_varname_parts(
            grad_block.name)

        if block_name:
            merged_var_name = '.'.join([orig_varname, block_name])
        else:
            merged_var_name = orig_varname

        merged_var = pserver_block.create_var(
            name=grad_block.name,
            persistable=True,
            type=grad_block.type,
            dtype=grad_block.dtype,
            shape=grad_block.shape)

        grad_to_block_id.append(merged_var.name + ":" + str(optimize_block.idx))
        if config.is_sync_mode() and trainers > 1:
            vars2merge = []
            for i in range(trainers):
                per_trainer_name = "%s.trainer_%d" % \
                                   (merged_var_name, i)
                per_trainer_var = pserver_block.create_var(
                    name=per_trainer_name,
                    persistable=False,
                    type=grad_block.type,
                    dtype=grad_block.dtype,
                    shape=grad_block.shape)
                vars2merge.append(per_trainer_var)

            optimize_block.append_op(
                type="sum",
                inputs={"X": vars2merge},
                outputs={"Out": merged_var},
                attrs={"use_mkldnn": False})
            optimize_block.append_op(
                type="scale",
                inputs={"X": merged_var},
                outputs={"Out": merged_var},
                attrs={"scale": 1.0 / float(trainers)})
        return merged_var

    origin_program = config.get_origin_main_program()
    origin_program = origin_program.clone()
    ps_endpoint = config.get_ps_endpoint()

    opt_op_on_pserver = []
    # Iterate through the ops, and if an op and the optimize ops
    # which located on current pserver are in one set, then
    # append it into the sub program.
    global_ops = []
    # sparse grad name to param name
    sparse_grad_to_param = []

    def _is_opt_op_on_pserver(endpoint, op):
        param_names = [
            p.name for p in config.param_grad_ep_mapping[endpoint]["params"]
        ]

        unmerged_varnames = []
        merged_varnames = []
        merged_ordernames = []

        for name in param_names:
            orig_varname = _orig_varname(name)

            for pairs in config.merged_variables_pairs:
                merged_p = pairs[0]
                if merged_p.merged_var.name == orig_varname:
                    if merged_p.merged_var.name == merged_p.ordered_vars[
                            0].name:
                        unmerged_varnames.append(merged_p.ordered_vars[0].name)
                    else:
                        merged_varnames.append(merged_p.merged_var.name)
                        merged_ordernames.append(merged_p.ordered_vars[0].name)
                    break

        param = op.input("Param")[0]

        if param in unmerged_varnames:
            return True

        for i in range(len(merged_ordernames)):
            if param == merged_ordernames[i]:
                merged_p = merged_varnames[i]
                merged_g = "{}@GRAD".format(merged_varnames[i])
                op._set_attr(OP_ROLE_VAR_ATTR_NAME, [merged_p, merged_g])
                return True
        return False

    def __append_optimize_op__(op, block, grad_to_block_id, merged_var, lr_ops):
        if _is_optimizer_op(op):
            _append_pserver_ops(block, op, ps_endpoint, grad_to_block_id,
                                origin_program, merged_var,
                                sparse_grad_to_param, config)
        elif op not in lr_ops:
            _append_pserver_non_opt_ops(block, op, origin_program, config)

    optimize_ops = _get_optimize_ops(origin_program)
    for _, op in enumerate(optimize_ops):
        if _is_optimizer_op(op) and _is_opt_op_on_pserver(ps_endpoint, op):
            opt_op_on_pserver.append(op)

    # append lr decay ops to the child block if exists
    lr_ops = _get_lr_ops(origin_program)
    has_lr_decay = True if len(lr_ops) > 0 else False
    lr_decay_block_id = -1
    optimize_blocks = []

    if has_lr_decay > 0:
        counter_increment_idx = -1
        for idx, op in enumerate(lr_ops):
            if op.type != 'increment':
                continue
            counter = op.input("X")[0]
            if counter == LEARNING_RATE_DECAY_COUNTER:
                counter_increment_idx = idx
                break

        if counter_increment_idx != -1:
            lr_ops.pop(counter_increment_idx)

        lr_decay_block = program._create_block(program.num_blocks - 1)
        optimize_blocks.append(lr_decay_block)
        for op in lr_ops:
            cloned_op = _append_pserver_non_opt_ops(lr_decay_block, op,
                                                    origin_program, config)
            # append sub blocks to pserver_program in lr_decay_op
            # todo(tangwei12): __clone_lr_op_sub_block__
        lr_decay_block_id = lr_decay_block.idx

    # append op to the current block
    grad_to_block_id = []
    pre_block_idx = program.num_blocks - 1

    for idx, opt_op in enumerate(opt_op_on_pserver):
        per_opt_block = program._create_block(pre_block_idx)
        optimize_blocks.append(per_opt_block)
        optimize_target_param_name = opt_op.attr(OP_ROLE_VAR_ATTR_NAME)[0]
        # append grad merging ops before clip and weight decay
        # e.g.merge grad->L2Decay op->clip op->optimize
        merged_var = None
        for _, op in enumerate(optimize_ops):
            # find the origin grad var before clipping / L2Decay,
            # merged_var should be the input var name of L2Decay
            grad_varname_for_block = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
            if op.attr(OP_ROLE_VAR_ATTR_NAME)[0] == optimize_target_param_name:
                merged_var = _append_pserver_grad_merge_ops(
                    per_opt_block, grad_varname_for_block, ps_endpoint,
                    grad_to_block_id)
                if merged_var:
                    break  # append optimize op once then append other ops.

        if merged_var:
            for _, op in enumerate(optimize_ops):
                # optimizer is connected to itself
                if op.attr(OP_ROLE_VAR_ATTR_NAME)[0] == optimize_target_param_name and \
                        op not in global_ops:
                    __append_optimize_op__(op, per_opt_block, grad_to_block_id,
                                           merged_var, lr_ops)

    # dedup grad to ids list
    grad_to_block_id = list(set(grad_to_block_id))
    # append global ops
    if global_ops:
        opt_state_block = program._create_block(program.num_blocks - 1)
        optimize_blocks.append(opt_state_block)
        for glb_op in global_ops:
            __append_optimize_op__(glb_op, opt_state_block, grad_to_block_id,
                                   None, lr_ops)

    if len(optimize_blocks) == 0:
        pre_block_idx = program.num_blocks - 1
        empty_block = program._create_block(pre_block_idx)
        optimize_blocks.append(empty_block)

    op = get_op_by_type(program.global_block(), "listen_and_serv")
    op._set_attr("optimize_blocks", optimize_blocks)
    op._set_attr("grad_to_block_id", grad_to_block_id)
    op._set_attr("sparse_grad_to_param", sparse_grad_to_param)
    op._set_attr("lr_decay_block_id", lr_decay_block_id)
    return program


def large_scale_sparse_pass(program, main_program, config, is_startup=False):
    opt_value_map = {}
    opt_value_map["sgd"] = ["Param"]
    opt_value_map["adam"] = ["Param", "Moment1", "Moment2"]
    opt_value_map["adagrad"] = ["Param", "Moment"]
    opt_value_map["adamax"] = ["Param", "Moment", "InfNorm"]
    opt_value_map["momentum"] = ["Param", "Velocity"]
    opt_value_map["lars_momentum"] = ["Param", "Velocity"]
    opt_value_map["rmsprop"] = ["Param", "Moment", "MeanSquare"]
    opt_value_map["decayed_adagrad"] = ["Param", "Moment"]
    opt_value_map["ftrl"] = ["Param", "SquaredAccumulator", "LinearAccumulator"]

    geo_value_map = {}
    geo_value_map["sum"] = "Param"

    opt_init_map = {}
    opt_init_map["gaussian_random"] = ["seed", "mean", "std"]
    opt_init_map["fill_constant"] = ["value"]
    opt_init_map["uniform_random"] = ["seed", "min", "max"]
    opt_init_map["truncated_gaussian_random"] = ["seed", "mean", "std"]

    def get_entry_attr(param_name):
        origin_name = _orig_varname(param_name)
        o_main_program = config.get_origin_main_program()
        for op in o_main_program.global_block().ops:
            if is_distributed_sparse_op(op) and get_sparse_tablename(
                    op) == origin_name:
                entry = op.attr("entry")
                return entry

    def get_initializer_attrs(acture_value_names):
        l_sep = ","
        l_in = "&"
        init_attrs = []
        o_startup_program = config.get_origin_startup_program()

        for value_name in acture_value_names:
            origin_var_name = _orig_varname(value_name)
            for op in o_startup_program.global_block().ops:
                if op.type in opt_init_map.keys(
                ) and origin_var_name == op.output("Out")[0]:
                    init_attr = [op.type]
                    for attr in opt_init_map[op.type]:
                        init_attr.append(str(op.attr(attr)))
                    init_attrs.append(l_in.join(init_attr))
                    break

        return l_sep.join(init_attrs)

    def get_optimizer_values(block):
        value_names = []
        acture_names = []
        value_dims = []
        grad = None
        opt_idx = -1
        fuse = False

        for op in block.ops:
            opt_idx += 1

            if op.type not in opt_value_map.keys():
                continue

            if op.type in ["sgd", "adam"]:
                fuse = True

            grad = main_program.global_block().vars[op.input("Grad")[0]]

            for value in opt_value_map[op.type]:
                var = main_program.global_block().vars[op.input(value)[0]]
                if len(var.shape) != 2:
                    raise ValueError("sparse param's dimension must be 2")

                value_names.append(value)
                value_dims.append(var.shape[1])
                acture_names.append(var.name)

            if value_names:
                break
        return grad, opt_idx, value_names, value_dims, acture_names, fuse

    def add_fuse_large_scale_op(block, global_block, table_name, value_names,
                                acture_names, grad, is_entry, opt_idx):

        op = block.ops[opt_idx]

        if op.type == "sgd":
            grad = main_program.global_block().vars[op.input("Grad")[0]]
            lr = main_program.global_block().vars[op.input("LearningRate")[0]]

            block._insert_op(
                opt_idx,
                type="lookup_sparse_table_fuse_sgd",
                inputs={"Grad": grad,
                        "LearningRate": lr},
                attrs={
                    "is_entry": is_entry,
                    "tablename": table_name,
                    "value_names": value_names
                })

        elif op.type == "adam":
            grad = main_program.global_block().vars[op.input("Grad")[0]]
            lr = main_program.global_block().vars[op.input("LearningRate")[0]]
            beta1_pow = main_program.global_block().vars[op.input("Beta1Pow")[
                0]]
            beta2_pow = main_program.global_block().vars[op.input("Beta2Pow")[
                0]]
            beta1_pow_o = main_program.global_block().vars[op.output(
                "Beta1PowOut")[0]]
            beta2_pow_o = main_program.global_block().vars[op.output(
                "Beta2PowOut")[0]]

            beta1 = op.attr('beta1')
            beta2 = op.attr('beta2')
            epsilon = op.attr('epsilon')

            block._insert_op(
                opt_idx,
                type="lookup_sparse_table_fuse_adam",
                inputs={
                    "Grad": grad,
                    "LearningRate": lr,
                    "Beta1Pow": beta1_pow,
                    "Beta2Pow": beta2_pow
                },
                outputs={
                    "Beta1PowOut": beta1_pow_o,
                    "Beta2PowOut": beta2_pow_o
                },
                attrs={
                    "beta1": beta1,
                    "beta2": beta2,
                    "epsilon": epsilon,
                    "is_entry": is_entry,
                    "tablename": table_name,
                    "value_names": value_names
                })
        else:
            raise ValueError("only support sgd/adam optimizer now")

    def add_large_scale_op(block, global_block, table_name, value_names,
                           acture_names, grad, is_entry, opt_idx):
        ids = global_block.create_var(
            name="kSparseIDs@{}".format(table_name),
            persistable=False,
            dtype="int64",
            shape=[1, 1],
            lod_level=0)

        # insert grad split to ids and tensor op
        block._insert_op(
            opt_idx,
            type="lookup_sparse_table_grad_split",
            inputs={"Grad": grad},
            outputs={"Row": ids,
                     "Value": grad},
            attrs={"tablename": table_name,
                   "is_entry": is_entry})

        # insert read at first
        vars = [global_block.vars[acture_name] for acture_name in acture_names]
        block._insert_op(
            opt_idx + 1,
            type="lookup_sparse_table_read",
            inputs={"Ids": ids},
            outputs={"Out": vars},
            attrs={"tablename": table_name,
                   "value_names": value_names})

        # append write at last
        inputs = {"Ids": ids, "In": vars}

        block.append_op(
            type="lookup_sparse_table_write",
            inputs=inputs,
            outputs={},
            attrs={"tablename": table_name,
                   "value_names": value_names})

    op = get_op_by_type(main_program.global_block(), "listen_and_serv")

    param_blockid_map = {}
    grad_blockid_map = {}
    grad_to_params = op.attr('sparse_grad_to_param')
    grad_to_block_ids = op.attr('grad_to_block_id')

    origin_program = config.get_origin_main_program()
    sparse_varnames = get_sparse_tablenames(origin_program, False)

    for grad_to_block_id in grad_to_block_ids:
        grad, blockid = grad_to_block_id.split(":")
        grad_blockid_map[grad] = int(blockid)

    for grad_to_param in grad_to_params:
        grad, param = grad_to_param.split(":")

        if _orig_varname(param) in sparse_varnames:
            continue

        param_blockid_map[param] = grad_blockid_map[grad]

    if not is_startup:
        for param, blockid in param_blockid_map.items():
            opt_block = program.block(blockid)

            grad, opt_idx, value_names, value_dims, acture_names, fuse = \
                get_optimizer_values(opt_block)

            entry_attr = get_entry_attr(param)
            is_entry = False if entry_attr == "none" else True

            if fuse:
                add_fuse_large_scale_op(opt_block,
                                        program.global_block(), param,
                                        value_names, acture_names, grad,
                                        is_entry, opt_idx)
            else:
                add_large_scale_op(opt_block,
                                   program.global_block(), param, value_names,
                                   acture_names, grad, is_entry, opt_idx)
    else:
        large_scale_kv_metas = []
        for param, blockid in param_blockid_map.items():
            opt_block = main_program.block(blockid)

            grad, opt_idx, value_names, value_dims, acture_names, fuse = \
                get_optimizer_values(opt_block)

            entry_attr = get_entry_attr(param)

            if fuse:
                # remove origin optimzier op
                opt_block._remove_op(opt_idx)

            # training/infer
            mode = "0"
            names_str = ",".join(value_names)
            dims_str = ",".join([str(dim) for dim in value_dims])
            ids_name = "kSparseIDs@{}".format(param)
            cached_str = ",".join(acture_names + [ids_name])
            init_attr_str = get_initializer_attrs(acture_names)

            meta_str = ":".join([
                param, names_str, dims_str, mode, grad.name, cached_str,
                init_attr_str, entry_attr
            ])
            print("large_scale_metas: {}".format(meta_str))
            large_scale_kv_metas.append(meta_str)

        program.global_block().append_op(
            type="lookup_sparse_table_init",
            inputs=None,
            outputs=None,
            attrs={"large_scale_metas": large_scale_kv_metas})

    # todo: need delete unused var.
    return program


def get_distributed_from_listen_and_serv(program, origin_program):
    op = get_op_by_type(program.global_block(), "listen_and_serv")
    sparse_varnames = get_sparse_tablenames(origin_program, True)
    sparse_params = []
    grad_to_params = op.attr('sparse_grad_to_param')
    for grad_to_param in grad_to_params:
        _, param = grad_to_param.split(":")
        if _orig_varname(param) in sparse_varnames:
            sparse_params.append(param)
    return sparse_params


def delete_unused_in_main_pass(program, config):
    origin_program = config.get_origin_main_program()
    sparse_params = get_distributed_from_listen_and_serv(program,
                                                         origin_program)

    for var in sparse_params:
        if program.global_block().has_var(var):
            program.global_block()._remove_var(var)
    return program


def delete_unused_in_startup_pass(program, main_program, config):
    origin_program = config.get_origin_main_program()
    sparse_params = get_distributed_from_listen_and_serv(main_program,
                                                         origin_program)
    remove_ops = []

    for op in program.global_block().ops:
        if op.type in ["recv", "fetch_barrier", "concat"]:
            continue

        for key in op.output_names:
            if op.output(key)[0] in sparse_params:
                remove_ops.append(op)

    all_ops = program.global_block().ops
    op_idxs = [all_ops.index(op) for op in remove_ops]

    for idx in op_idxs[::-1]:
        program.global_block()._remove_op(idx)

    for var in sparse_params:
        if program.global_block().has_var(var):
            program.global_block()._remove_var(var)

    return program


def build_pserver_startup_program_pass(program, p_main_program, config):
    ps_endpoint = config.get_ps_endpoint()
    o_startup_program = config.get_origin_startup_program()
    program.random_seed = o_startup_program.random_seed
    params = config.param_grad_ep_mapping[ps_endpoint]["params"]
    merged_ordervars = []

    for var in params:
        name = var.name
        orig_varname = _orig_varname(name)

        for pairs in config.merged_variables_pairs:
            merged_p = pairs[0]
            if merged_p.merged_var.name == orig_varname:
                if merged_p.merged_var.name != merged_p.ordered_vars[0].name:
                    merged_ordervars.append(merged_p.ordered_vars[0])
                break

    def _get_splited_name_and_shape(varname):
        for splited_param in params:
            pname = splited_param.name
            if _same_or_split_var(pname, varname) and varname != pname:
                return pname, splited_param.shape

            for idx, ordered in enumerate(merged_ordervars):
                if _same_or_split_var(varname, ordered.name):
                    return pname, splited_param.shape

        return "", []

    # 1. create vars in pserver program to startup program
    pserver_vars = p_main_program.global_block().vars

    created_var_map = collections.OrderedDict()
    for _, var in six.iteritems(pserver_vars):
        tmpvar = program.global_block()._clone_variable(var)
        created_var_map[var.name] = tmpvar

    # 2. rename op outputs
    for op in o_startup_program.global_block().ops:
        new_outputs = collections.OrderedDict()
        # do not append startup op if var is not on this pserver
        op_on_pserver = False
        # TODO(gongwb) : remove this line.
        if op.type not in ["recv", "fetch_barrier", "concat"]:
            for key in op.output_names:
                newname, _ = _get_splited_name_and_shape(op.output(key)[0])
                if newname:
                    op_on_pserver = True
                    new_outputs[key] = created_var_map[newname]
                elif op.output(key)[0] in pserver_vars:
                    op_on_pserver = True
                    new_outputs[key] = pserver_vars[op.output(key)[0]]

        if op_on_pserver:
            # most startup program ops have no inputs
            new_inputs = _get_input_map_from_op(pserver_vars, op)

            if op.type in [
                    "gaussian_random", "fill_constant", "uniform_random",
                    "truncated_gaussian_random"
            ]:
                op._set_attr("shape", list(new_outputs["Out"].shape))

            program.global_block().append_op(
                type=op.type,
                inputs=new_inputs,
                outputs=new_outputs,
                attrs=op.all_attrs())

    return program


def add_geo_optimizer_pass(program, config):
    endpoint = config.get_ps_endpoint()
    params = [p for p in config.param_grad_ep_mapping[endpoint]["params"]]

    sparse_tablenames = get_sparse_tablenames(config.get_origin_main_program(),
                                              False)

    for param in params:
        _clone_var(program.global_block(), param)

    optimize_block = []
    sparse_grad_to_param = []
    param_to_block_id = []
    pre_block_idx = program.num_blocks - 1

    for param in params:
        per_opt_block = program._create_block(pre_block_idx)
        optimize_block.append(per_opt_block)
        var_name = param.name
        pserver_block = per_opt_block.program.global_block()
        param = pserver_block.vars[var_name]

        delta_var_name = "%s.delta" % (param.name)
        origin_varname = _orig_varname(param.name)

        if origin_varname in sparse_tablenames:
            sparse_grad_to_param.append(":".join([delta_var_name, param.name]))

        delta_var = pserver_block.create_var(
            name=delta_var_name,
            persistable=False,
            type=param.type,
            dtype=param.dtype,
            shape=param.shape)

        per_opt_block.append_op(
            type="sum",
            inputs={"X": [param, delta_var]},
            outputs={"Out": param})

        param_to_block_id.append(delta_var_name + ":" + str(per_opt_block.idx))

    op = get_op_by_type(program.global_block(), "listen_and_serv")
    op._set_attr("optimize_blocks", optimize_block)
    op._set_attr("grad_to_block_id", param_to_block_id)
    op._set_attr("sparse_grad_to_param", sparse_grad_to_param)

    return program
