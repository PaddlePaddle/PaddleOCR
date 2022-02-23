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
import paddle
from paddle.fluid import core, unique_name
from functools import reduce
from paddle.distributed.fleet.meta_optimizers.common import is_loss_grad_op, is_backward_op
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY

import re
import os


def check_broadcast(block):
    """
    if a var is broadcasted, it should have a sync_comm before
    this var is used, if not, raise error.
    if the broadcasted var has a fill_constant op, the fill_constant
    op should stay forward before the broadcast op, and before a
    sync_calc op. Otherwise, raise error.

    should ignore and skip broadcast_op of inner_parallelism (e.g. Megatron)
    """
    broadcast_vars = {}
    for idx, op in enumerate(block.ops):
        if op.type == "c_broadcast":
            if op.all_attrs()["use_calc_stream"] == False:
                var_name = op.desc.input_arg_names()[0]
                if "@BroadCast" in var_name:
                    if var_name in broadcast_vars:
                        raise ValueError("var_name areadly exist: {}"
                                         "the old pos is {}, the new pos is {}".
                                         format(var_name, broadcast_vars[
                                             var_name]["broadcast_pos"], idx))
                    broadcast_vars[var_name] = {
                        "fill_constant_pos": -1,
                        "broadcast_pos": idx,
                    }

    for idx, op in enumerate(block.ops):
        if op.type == "fill_constant":
            var_name = op.desc.output_arg_names()[0]
            if var_name in broadcast_vars:
                broadcast_vars[var_name]["fill_constant_pos"] = idx
            continue

    last_sync_comm_op_idx = -1
    last_sync_calc_op_idx = -1
    for idx, op in enumerate(block.ops):
        if op.type == "c_sync_comm_stream":
            last_sync_comm_op_idx = idx
            continue
        if op.type == "c_sync_calc_stream":
            last_sync_calc_op_idx = idx
            continue
        if op.type == "c_broadcast":
            if op.all_attrs()["use_calc_stream"] == False:
                var_name = op.desc.input_arg_names()[0]
                if "@BroadCast" in var_name:
                    if broadcast_vars[var_name]["fill_constant_pos"] != -1:
                        assert (last_sync_calc_op_idx != -1)
                        assert (broadcast_vars[var_name]["fill_constant_pos"] <
                                last_sync_calc_op_idx)
                        assert (last_sync_calc_op_idx < idx)
                    continue
        for input_name in op.desc.input_arg_names():
            if input_name in broadcast_vars:
                assert (broadcast_vars[input_name]["broadcast_pos"] != -1)
                assert (broadcast_vars[input_name]["broadcast_pos"] <
                        last_sync_comm_op_idx)
                assert (last_sync_comm_op_idx < idx)
    return


def check_allreduce_sum(block, shard, sharding_ring_id, dp_ring_id=-1):
    """
    the op order should be:
        grad:
            - 0: op that generate Var
            - 1: sync_calc
            - 2: reduce_sum_sharding (allreduce --> reduce)
            - 3: sync_comm
            - 4: allreuce_sum_dp (dp_grads)
            - 5: sync_comm (dp_grads)
            - 6: op that use Var (dp_grads & sum)

    should ignore and skip allreduce_op of inner_parallelism (e.g. Megatron)
    """
    vars_status = {}
    dp_grads_status = {}
    idx_last_grad_allreduce = -1
    idx_amp_allreduce = -1
    idx_gradient_clip_allreduce = -1

    for idx, op in enumerate(block.ops):
        # sharding use both allreduce and reduce to sync grad
        if op.type == "c_allreduce_sum" or op.type == "c_reduce_sum":
            if op.all_attrs()["use_calc_stream"] == False:
                ring_id = op.desc.attr("ring_id")
                var_name = op.desc.input_arg_names()[0]
                param = var_name.split("@")[0]

                assert 'sum' in var_name or ("@GRAD" in var_name)
                if 'sum' in var_name or (not shard.has_param(param)):
                    vars_status[var_name] = -1
                else:
                    dp_grads_status[var_name] = -1

                if ring_id != sharding_ring_id:
                    assert shard.has_param(param)
                    assert ring_id == dp_ring_id

                if "sum" in var_name:
                    idx_amp_allreduce = idx
                elif "@GRAD":
                    idx_last_grad_allreduce = idx

        if op.type == "c_allreduce_max":
            idx_gradient_clip_allreduce = idx

    for op in block.ops:
        if op.type == "c_sync_calc_stream":
            for var_name in vars_status:
                if var_name in vars_status and vars_status[var_name] == 0:
                    vars_status[var_name] = 1
            for var_name in dp_grads_status:
                if var_name in dp_grads_status and dp_grads_status[
                        var_name] == 0:
                    dp_grads_status[var_name] = 1
        # check sharding allreduce and  reduce but skip megatron allreduce
        elif op.type == "c_allreduce_sum" or op.type == "c_reduce_sum":
            if op.all_attrs()["use_calc_stream"] == False:
                var_name = op.desc.input_arg_names()[0]
                ring_id = op.desc.attr("ring_id")
                if ring_id == sharding_ring_id:
                    assert op.type == "c_reduce_sum", "Grad in Sharding group should be reduce rather than allreduce"
                    if var_name in vars_status:
                        _status = vars_status[var_name]
                    else:
                        _status = dp_grads_status[var_name]
                    if _status == -1:
                        raise ValueError("{} is not generated, but you are"
                                         "trying to all-reduce it".format(
                                             var_name))
                    if _status == 0:
                        raise ValueError("There should be a sync_calc op "
                                         "after generate Var: {} and before the"
                                         "c_allreduce_sum op".format(var_name))
                    assert (_status == 1)
                    if var_name in vars_status:
                        vars_status[var_name] = 2
                    else:
                        dp_grads_status[var_name] = 2
                else:
                    assert ring_id == dp_ring_id
                    param = var_name.split("@")[0]
                    assert shard.has_param(param)
                    assert dp_grads_status[var_name] == 3
                    dp_grads_status[var_name] = 4

        elif op.type == "c_sync_comm_stream":
            var_name = op.desc.input_arg_names()[0]
            ring_id = op.desc.attr("ring_id")
            if ring_id == sharding_ring_id:
                for var_name in op.desc.input_arg_names():
                    if var_name in vars_status:
                        assert vars_status[var_name] == 2
                        vars_status[var_name] = 3
                    elif var_name in dp_grads_status:
                        assert dp_grads_status[var_name] == 2
                        dp_grads_status[var_name] = 3
            else:
                for var_name in op.desc.input_arg_names():
                    param = var_name.split("@")[0]
                    assert ring_id == dp_ring_id
                    assert shard.has_param(param)
                    assert dp_grads_status[var_name] == 4
                    dp_grads_status[var_name] = 5
        else:
            for input_name in op.desc.input_arg_names():
                if input_name in vars_status:
                    if vars_status[input_name] != 3:
                        raise ValueError("There should be a sync_comm op "
                                         "after allreduce the Var: {}".format(
                                             input_name))
                    raise ValueError(
                        "The reduce output grad [{}] should NOT be be used in Non-root rank.".
                        format(input_name))
                if input_name in dp_grads_status:
                    if dp_ring_id == -1:
                        if dp_grads_status[input_name] != 3:
                            raise ValueError("There should be a sync_comm op "
                                             "after allreduce the Var: {}".
                                             format(input_name))
                    else:
                        if dp_grads_status[input_name] != 5:
                            raise ValueError(
                                "The grad in shard should be allreduce and sync"
                                "twice before usage {}".format(input_name))

            for output_name in op.desc.output_arg_names():
                if output_name in vars_status and \
                    vars_status[output_name] == -1:
                    vars_status[output_name] = 0
                if output_name in dp_grads_status and  \
                    dp_grads_status[output_name] == -1:
                    dp_grads_status[output_name] = 0

    # check sharding with amp
    if idx_amp_allreduce != -1:
        assert idx_amp_allreduce > idx_last_grad_allreduce

    # check sharding with gradient_clip_by_global_norm
    if idx_gradient_clip_allreduce != -1:
        assert idx_gradient_clip_allreduce > idx_last_grad_allreduce

    return


def get_valid_op_role(block, insert_idx):
    """
    return OpRole.Forward or OpRole.Backward
    """
    op_role = block.ops[insert_idx].attr('op_role')
    if (insert_idx >= len(block.ops)) or (
            op_role in [int(OpRole.Backward), int(OpRole.Optimize)]):
        return OpRole.Backward
    if op_role in [int(OpRole.Forward), int(OpRole.Loss)]:
        return OpRole.Forward

    return get_valid_op_role(block, insert_idx + 1)


def insert_sync_calc_op(block, insert_idx, calc_dep_vars):
    """
    _insert_sync_calc_op
    """
    op_role = get_valid_op_role(block, insert_idx)
    block._insert_op_without_sync(
        insert_idx,
        type='c_sync_calc_stream',
        inputs={'X': calc_dep_vars},
        outputs={'Out': calc_dep_vars},
        attrs={OP_ROLE_KEY: op_role})
    return


def insert_sync_comm_op(block, insert_idx, ring_id, comm_dep_vars):
    """
    insert sync_comm_op for single var
    """
    op_role = get_valid_op_role(block, insert_idx)
    block._insert_op_without_sync(
        insert_idx,
        type='c_sync_comm_stream',
        inputs={'X': comm_dep_vars},
        outputs={'Out': comm_dep_vars},
        attrs={'ring_id': ring_id,
               OP_ROLE_KEY: op_role})
    return 1


def insert_sync_comm_ops(block, insert_idx, ring_id, comm_dep_vars):
    """
    insert sync_comm_op for vars
    """
    # NOTE (JZ-LIANG) to be check, may result undefined case 
    if len(comm_dep_vars) == 0:
        return 0

    op_role = get_valid_op_role(block, insert_idx)
    block._insert_op_without_sync(
        insert_idx,
        type='c_sync_comm_stream',
        inputs={'X': comm_dep_vars},
        outputs={'Out': comm_dep_vars},
        attrs={'ring_id': int(ring_id),
               OP_ROLE_KEY: op_role})
    return 1


def insert_fill_constant_ops(block, insert_idx, fill_constant_vars):
    """
    _add_fill_constant_ops
    """
    op_role = get_valid_op_role(block, insert_idx)
    for broadcast_name in fill_constant_vars:
        broadcast_var = block.var(broadcast_name)
        block._insert_op_without_sync(
            insert_idx,
            type="fill_constant",
            outputs={"Out": broadcast_var.name},
            attrs={
                "shape": broadcast_var.shape,
                "dtype": broadcast_var.dtype,
                "value": 0.0,
                OP_ROLE_KEY: op_role
            })
    return


def insert_cast_ops(block, insert_idx, cast_ops):
    """
    _add_cast_ops
    """
    op_role = get_valid_op_role(block, insert_idx)
    for fp16_name, fp32_name in cast_ops.items():
        block._insert_op_without_sync(
            insert_idx,
            type="cast",
            inputs={"X": fp32_name},
            outputs={"Out": fp16_name},
            attrs={
                "in_dtype": core.VarDesc.VarType.FP32,
                "out_dtype": core.VarDesc.VarType.FP16,
                OP_ROLE_KEY: op_role
            })
    return


def insert_allreduce_ops(block,
                         insert_idx,
                         ring_id,
                         allreduce_vars,
                         op_role=OpRole.Backward,
                         use_calc_stream=False,
                         user_defined_strategy=None):
    """
    _add_allreduce_ops
    """
    if len(allreduce_vars) == 0:
        return

    if user_defined_strategy and \
            user_defined_strategy.fuse_all_reduce_ops and \
            not user_defined_strategy.fuse_grad_merge:
        # If fuse_grad_merge is enable, the grad vars have already been fused during
        # gradient merge pass, therefore, those vars are not need to be fused here
        insert_fused_allreduce_ops(block, insert_idx, ring_id, allreduce_vars,
                                   op_role, use_calc_stream,
                                   user_defined_strategy.fuse_grad_size_in_MB)
    else:
        for var in allreduce_vars:
            block._insert_op_without_sync(
                insert_idx,
                type='c_allreduce_sum',
                inputs={'X': var},
                outputs={'Out': var},
                attrs={
                    'ring_id': ring_id,
                    'use_calc_stream': use_calc_stream,
                    OP_ROLE_KEY: op_role
                })

    return


class FuseHelper(object):
    @staticmethod
    def get_fused_groups(block, vars_name, fuse_size=32.):
        """ coalesce tensor, get fused group """
        groups = []
        cur_size = 0.
        last_dtype = None
        for var_name in vars_name:
            real_var = block.var(var_name)
            var_size = get_var_size(real_var)
            if cur_size + var_size > fuse_size \
                    or len(groups) == 0 \
                    or real_var.dtype != last_dtype:
                groups.append([real_var])
                cur_size = var_size
                last_dtype = real_var.dtype
            else:
                groups[-1].append(real_var)
                cur_size += var_size
        return groups

    @staticmethod
    def insert_coalesce_tensor(block,
                               index,
                               groups,
                               op_role=OpRole.Backward,
                               prefix="Output"):
        fused_vars = []
        insert_num = 0
        for group in groups:
            assert len(group) >= 1
            if len(group) == 1:
                # no need fuse
                fused_vars.append(group[0])
                continue

            fused_var = block.create_var(
                name=unique_name.generate('Fused{}_{}'.format(prefix, group[0]
                                                              .name)),
                dtype=group[0].dtype,
                persistable=False,
                stop_gradient=True)
            fused_vars.append(fused_var)
            block._insert_op_without_sync(
                index,
                type="coalesce_tensor",
                inputs={"Input": group},
                outputs={"Output": group,
                         "FusedOutput": fused_var},
                attrs={
                    "copy_data": True,
                    "use_align": True,
                    "dtype": group[0].dtype,
                    OP_ROLE_KEY: op_role
                })
            insert_num += 1
        return fused_vars, insert_num


def insert_fused_allreduce_ops(block,
                               insert_idx,
                               ring_id,
                               allreduce_vars,
                               op_role=OpRole.Backward,
                               use_calc_stream=False,
                               fuse_grad_size_in_MB=32):
    groups = FuseHelper.get_fused_groups(block, allreduce_vars,
                                         fuse_grad_size_in_MB)

    fused_vars, insert_num = FuseHelper.insert_coalesce_tensor(
        block, insert_idx, groups, op_role, prefix="Grad")

    for fused_var in fused_vars:
        block._insert_op_without_sync(
            insert_idx + insert_num,
            type='c_allreduce_sum',
            inputs={'X': fused_var},
            outputs={'Out': fused_var},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': use_calc_stream,
                OP_ROLE_KEY: op_role
            })
        if not use_calc_stream:
            block._insert_op_without_sync(
                insert_idx + insert_num,
                type='c_sync_calc_stream',
                inputs={'X': fused_var},
                outputs={'Out': fused_var},
                attrs={OP_ROLE_KEY: op_role})


def insert_fused_reduce_ops(block,
                            insert_idx,
                            ring_id,
                            reduce_vars,
                            shard,
                            op_role=OpRole.Backward,
                            use_calc_stream=False,
                            rank=None,
                            fuse_grad_size=32):
    nranks = shard.worker_num
    device_to_vars = [[] for _ in range(nranks)]

    for var in reduce_vars:
        root_id = get_grad_device(var, shard)
        assert 0 <= root_id < nranks, "root_id should >=0 and < nranks, " \
            "but now nranks={}, the root_id of var={} is {}"\
            .format(nranks, var, root_id)
        device_to_vars[root_id].append(var)

    for root_id, vars_name in enumerate(device_to_vars):
        groups = FuseHelper.get_fused_groups(block, vars_name, fuse_grad_size)

        fused_vars, insert_num = FuseHelper.insert_coalesce_tensor(
            block, insert_idx, groups, op_role, prefix="Grad")

        for fused_var in fused_vars:
            block._insert_op_without_sync(
                insert_idx + insert_num,
                type='c_reduce_sum',
                inputs={'X': fused_var},
                outputs={'Out': fused_var},
                attrs={
                    'ring_id': ring_id,
                    'root_id': root_id,
                    'use_calc_stream': use_calc_stream,
                    OP_ROLE_KEY: op_role
                })
            if not use_calc_stream:
                block._insert_op_without_sync(
                    insert_idx + insert_num,
                    type='c_sync_calc_stream',
                    inputs={'X': fused_var},
                    outputs={'Out': fused_var},
                    attrs={OP_ROLE_KEY: op_role})

    return [] if rank is None else device_to_vars[rank]


def insert_reduce_ops(block,
                      insert_idx,
                      ring_id,
                      reduce_vars,
                      shard,
                      op_role=OpRole.Backward,
                      use_calc_stream=False,
                      rank=None,
                      strategy=None):
    """
    _add_reduce_ops
    """
    if strategy and strategy.fuse_all_reduce_ops and \
            not strategy.fuse_grad_merge:
        return insert_fused_reduce_ops(block, insert_idx, ring_id, reduce_vars,
                                       shard, op_role, use_calc_stream, rank,
                                       strategy.fuse_grad_size_in_MB)

    grad_in_this_device = []
    for var in reduce_vars:
        grad_var = var
        if strategy and strategy.fuse_all_reduce_ops and \
                strategy.fuse_grad_merge:
            # TODO(wangxi): if support fp16_allreduce, need be
            # 'FusedMergedGrad.cast_fp16._'
            grad_var = var.replace('FusedMergedGrad_', '')
        root_id = get_grad_device(grad_var, shard)
        assert root_id >= 0, "root id should be a positive int, but now root id is {}".format(
            root_id)
        if rank is not None and rank == root_id:
            grad_in_this_device.append(var)
        block._insert_op_without_sync(
            insert_idx,
            type='c_reduce_sum',
            inputs={'X': var},
            outputs={'Out': var},
            attrs={
                'ring_id': ring_id,
                'root_id': root_id,
                'use_calc_stream': use_calc_stream,
                OP_ROLE_KEY: op_role
            })

    return grad_in_this_device


def insert_fused_broadcast_param_ops(block,
                                     insert_idx,
                                     ring_id,
                                     params,
                                     shard,
                                     op_role=OpRole.Optimize,
                                     use_calc_stream=False,
                                     rank=None,
                                     fuse_size=32):
    nranks = shard.worker_num
    device_to_vars = [[] for _ in range(nranks)]

    for var in params:
        root_id = shard.device(var)
        assert 0 <= root_id < nranks, "root_id should >=0 and < nranks, " \
            "but now nranks={}, the root_id of var={} is {}"\
            .format(nranks, var, root_id)
        device_to_vars[root_id].append(var)

    for root_id, vars_name in enumerate(device_to_vars):
        groups = FuseHelper.get_fused_groups(block, vars_name, fuse_size)

        fused_vars, insert_num = FuseHelper.insert_coalesce_tensor(
            block, insert_idx, groups, op_role, prefix="Param")

        for fused_var in fused_vars:
            block._insert_op_without_sync(
                insert_idx + insert_num,
                type='c_broadcast',
                inputs={'X': fused_var},
                outputs={'Out': fused_var},
                attrs={
                    'ring_id': ring_id,
                    'root': root_id,
                    'use_calc_stream': use_calc_stream,
                    OP_ROLE_KEY: op_role
                })
            if not use_calc_stream:
                block._insert_op_without_sync(
                    insert_idx + insert_num,
                    type='c_sync_calc_stream',
                    inputs={'X': fused_var},
                    outputs={'Out': fused_var},
                    attrs={OP_ROLE_KEY: op_role})

    return [] if rank is None else device_to_vars[rank]


def insert_broadcast_param_ops(block,
                               insert_idx,
                               ring_id,
                               params,
                               shard,
                               op_role=OpRole.Optimize,
                               use_calc_stream=False,
                               rank=None,
                               strategy=None):
    """
    add broadcast param ops
    """
    if strategy and strategy.fuse_all_reduce_ops:
        # TODO(wangxi): put fused var in startup_program, only need exec once
        return insert_fused_broadcast_param_ops(
            block, insert_idx, ring_id, params, shard, op_role, use_calc_stream,
            rank, strategy.fuse_grad_size_in_MB)

    param_in_this_device = []
    for param in params:
        root_id = shard.device(param)
        assert root_id >= 0, "root id should be a positive int, but now root id is {}".format(
            root_id)
        if rank is not None and rank == root_id:
            param_in_this_device.append(param)
        block._insert_op_without_sync(
            insert_idx,
            type='c_broadcast',
            inputs={'X': param},
            outputs={'Out': param},
            attrs={
                'ring_id': ring_id,
                'root': root_id,
                'use_calc_stream': use_calc_stream,
                OP_ROLE_KEY: op_role
            })

    return param_in_this_device


def get_grad_device(grad_name, shard):
    assert "@GRAD" in grad_name, "[{}] should be a grad variable.".format(
        grad_name)
    base_name = None
    # NOTE: mind the traversal order
    possible_suffixes = [
        # sharding gm
        '.cast_fp16@GRAD@MERGED',
        '.cast_fp16@GRAD',
        # pipeline
        '@GRAD@MERGED@FP16',
        '@GRAD@MERGED',
        '@GRAD',
    ]
    for suffix in possible_suffixes:
        if suffix in grad_name:
            base_name = re.sub(suffix, '', grad_name)
            break

    assert base_name in shard.global_param2device, "[{}] should be a param variable.".format(
        base_name)

    return shard.global_param2device[base_name]


def get_first_check_finite_and_unscale_op_idx(block, raise_error=True):

    for idx, op in enumerate(block.ops):
        if op.type == "check_finite_and_unscale":
            return idx

    if raise_error:
        raise ValueError(
            "amp is turned on but check_finite_and_unscale op does not exist in main block"
        )

    return -1


def get_first_optimize_op_idx(block):
    first_opt_op_idx = None
    for index, op in reversed(tuple(enumerate(block.ops))):
        if is_backward_op(op) and first_opt_op_idx is None:
            first_opt_op_idx = index + 1
            break
    return first_opt_op_idx


def insert_broadcast_ops(block, insert_idx, ring_id, broadcast2root):
    """
    _add_broadcast_ops
    """
    op_role = get_valid_op_role(block, insert_idx)
    for broadcast_name, root_device in broadcast2root:
        block._insert_op_without_sync(
            insert_idx,
            type='c_broadcast',
            inputs={'X': broadcast_name},
            outputs={'Out': broadcast_name},
            attrs={
                'ring_id': ring_id,
                'root': root_device,
                OP_ROLE_KEY: op_role
            })

    return


DtypeToSize = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    core.VarDesc.VarType.UINT8: 1,
}


def get_var_size(param):
    """
    input:
        - param: var
    return:
        var size in MB
    """
    assert -1 not in param.shape
    return reduce(lambda x, y: x * y,
                  param.shape) * DtypeToSize[param.dtype] / 1024.0 / 1024.0


def insert_scale_loss_grad_ops(block, scale=1.0):
    '''
    In order to keep the learning rate consistent in different numbers of
    training workers, we scale the loss grad by the number of workers
    '''
    for idx, op in reversed(list(enumerate(block.ops))):
        if is_loss_grad_op(op):
            assert op.type == 'fill_constant', \
                "loss_grad_op must be fill_constant op, " \
                "but this op is {}".format(op.type)
            assert op.has_attr('value')
            loss_scale = float(op.attr('value'))
            loss_scale = loss_scale / scale
            op._set_attr('value', loss_scale)
            break


def comm_analyse(main_program):
    """
    Analyse the parameter size that need to be broadcast/allreduce during sharding training 
    """
    reduce_vars = {}
    broadcast_vars = {}
    block = main_program.global_block()
    for op in block.ops:
        if op.type == "c_broadcast":
            var_name = op.desc.input_arg_names()[0]
            # convert MB to KB
            broadcast_vars[var_name] = get_var_size(block.var(
                var_name)) * 1024.0
        elif op.type == "c_allreduce_sum":
            var_name = op.desc.input_arg_names()[0]
            reduce_vars[var_name] = get_var_size(block.var(var_name)) * 1024.0

    varsize_count = {}
    gap = 1

    for k, v in broadcast_vars.items():
        print("broadcast: {}: {} KB".format(k, v))
        if (int(v / gap) in varsize_count):
            varsize_count[int(v / gap)] += 1
        else:
            varsize_count[int(v / gap)] = 1

    for k, v in reduce_vars.items():
        print("allreduce: {}: {} KB".format(k, v))
        if (int(v / gap) in varsize_count):
            varsize_count[int(v / gap)] += 1
        else:
            varsize_count[int(v / gap)] = 1

    with open("nccl_size.txt", 'w') as f:
        sorted_varsize = sorted(varsize_count.items(), key=lambda x: x[0])
        for varsize, count in sorted_varsize:
            print("NCCL size {}~{} KB: {}".format(varsize, varsize + 1, count))
            f.write("NCCL size {}~{} KB: {}\n".format(varsize, varsize + 1,
                                                      count))


def add_sync_comm(program, sharding_ring_id):
    """
    When clone a test prog by clone from the sharding main prog, 
    part of the sync_comm op maybe be pruned by mistake, this function
    add the sync_comm op for the test prog.

    """
    #NOTE (liangjianzhong): only support one comm stream by now, use more than one
    # comm streams will cause error. should be revise in future.

    assert sharding_ring_id >= 0, "sharding_ring_id should larger than zero"
    block = program.global_block()
    not_sync_vars = set([])
    for op in block.ops:
        if op.type in ["c_broadcast", "c_allreduce"]:
            for input_name in op.desc.input_arg_names():
                not_sync_vars.add(input_name)
        if op.type == "c_sync_comm_stream":
            for input_name in op.desc.input_arg_names():
                not_sync_vars.remove(input_name)
    if not_sync_vars:
        block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': list(not_sync_vars)},
            outputs={'Out': list(not_sync_vars)},
            attrs={
                'ring_id': sharding_ring_id,
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            })
    return


def save_persistables(exe, dirname, main_program, filename=None):
    """
    When use sharding, part of persistable vars are unique and are partitioned in different ranks,
    and part of persistable vars are duplicated and exist in all the ranks with different values.
    This function handles the model saving for sharding training.
    """
    # TODO (JZ-LIANG) revise this for uniform mixed parallelism
    if main_program._pipeline_opt:
        main_program = main_program._pipeline_opt['section_program']

    def is_opt_vars(var):
        # NOTE(JZ-LIANG): The checks should be updated when add new compatible optimizer
        # now only Momentum and adam are compatible with sharding
        checks = [
            "_moment1_0", "_moment2_0", "_beta1_pow_acc_0", "_beta2_pow_acc_0",
            "_velocity_0"
        ]
        for check in checks:
            if var.name.endswith(check):
                return True
        return False

    def is_gradient_merge_vars(var):
        # NOTE(JZ-LIANG): to revise save/load logic in framework instead of write this naive rule

        return var.name.endswith("@GradiantMerge")

    def is_trainable(var):
        return isinstance(var,
                          paddle.fluid.framework.Parameter) and var.trainable

    def sharding_predicate(var):
        return is_trainable(var) or is_opt_vars(var) or is_gradient_merge_vars(
            var)

    if int(os.environ.get('PADDLE_TRAINER_ID', 0)) == 0:
        paddle.fluid.io.save_persistables(
            exe, dirname, main_program=main_program, filename=None)
    else:
        paddle.fluid.io.save_vars(
            exe,
            dirname,
            main_program=main_program,
            predicate=sharding_predicate,
            filename=None)

    return


def append_naive_sync(block, sync_var, ring_id):
    # NOTE (JZ-LIANG) update this to use barrier sync for more elegent logic
    # sync within global 
    block.append_op(
        type="fill_constant",
        outputs={"Out": sync_var},
        attrs={
            "shape": sync_var.shape,
            "dtype": sync_var.dtype,
            "value": int(1),
        })
    block.append_op(
        type='c_allreduce_sum',
        inputs={'X': sync_var},
        outputs={'Out': sync_var},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': True,
            OP_ROLE_KEY: OpRole.Forward
        })
    block.append_op(
        type='c_sync_calc_stream',
        inputs={'X': [sync_var]},
        outputs={'Out': [sync_var]},
        attrs={OP_ROLE_KEY: OpRole.Forward})
