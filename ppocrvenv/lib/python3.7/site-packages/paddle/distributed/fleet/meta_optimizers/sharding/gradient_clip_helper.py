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

from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

__all__ = []


class GradientClipHelper(object):
    def __init__(self, mp_ring_id):
        self.mp_ring_id = mp_ring_id

    def _is_gradient_clip_op(self, op):
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/gradient_clip")

    def prune_gradient_clip(self, block, shard, ring_ids):
        """
        prune gradient_clip related ops for params that not belong to cur shard
        prune: square, reduce_sum, elementwise_mul
        keep: sum, sqrt, elementwise_max, elementwise_div
        """
        deperated_vars = set()
        deperate_op_idx = set()
        reversed_x_paramname = []
        global_norm_sum_op_idx = -1
        for idx, op in enumerate(block.ops):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == "sum":
                global_norm_sum_op_idx = idx
                continue
            deperate_op = False
            for input_name in op.desc.input_arg_names():
                if input_name in deperated_vars:
                    deperate_op = True
                # TODO (JZ-LIANG) revise this for uniform mixed parallelism
                if "@MERGED" in input_name:
                    param_name = input_name.strip("@GRAD@MERGED")
                else:
                    param_name = input_name.strip("@GRAD")
                if shard.is_param(param_name) and \
                  not shard.has_param(param_name):
                    deperate_op = True
                elif shard.is_param(param_name):
                    reversed_x_paramname.append(param_name)

            if deperate_op:
                deperate_op_idx.add(idx)
                for output_name in op.desc.output_arg_names():
                    if output_name not in op.desc.input_arg_names():
                        deperated_vars.add(output_name)

        # NOTE(wangxi): If only have 2 sharding, and 1 param.
        # sharding 0 will not deperated_vars, will return, only
        # sharding 1 will insert allreduce, then hang.
        if not deperated_vars and global_norm_sum_op_idx == -1:
            # got no gradient_clip op
            return

        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_gradient_clip_op(op):
                continue
            if idx in deperate_op_idx:
                block._remove_op(idx, sync=False)
                continue
            if op.type == "sum":
                reversed_inputs = []
                global_norm_sum_op_idx = idx
                for input_name in op.desc.input_arg_names():
                    if input_name not in deperated_vars:
                        reversed_inputs.append(input_name)

                op.desc.set_input("X", reversed_inputs)
                assert (len(op.desc.output_arg_names()) == 1)
                sum_res = op.desc.output_arg_names()[0]

                # NOTE(wangxi): If we have 2 param, but sharding is 4,
                # then the sum op in some cards will not have input.
                # So we use fill_constant_op to set `sum_var` to zero,
                # which does not affect correctness.
                if len(reversed_inputs) == 0:
                    sum_var = block.var(sum_res)
                    namescope = op.attr("op_namescope")

                    block._remove_op(idx, sync=False)
                    op = block._insert_op_without_sync(
                        idx,
                        type='fill_constant',
                        inputs={},
                        outputs={'Out': sum_res},
                        attrs={
                            'shape': sum_var.shape,
                            'dtype': sum_var.dtype,
                            'value': 0.0,
                            OP_ROLE_KEY: OpRole.Optimize
                        })
                    op._set_attr('op_namescope', namescope)

                # allreduce(mp)->allreduce(sharding)->allreduce(pp)
                idx_offset = 1
                for ring_id in ring_ids:
                    if ring_id == -1: continue
                    # this allreduce should not overlap with calc and should be scheduled in calc stream
                    block._insert_op_without_sync(
                        idx + idx_offset,
                        type='c_allreduce_sum',
                        inputs={'X': sum_res},
                        outputs={'Out': sum_res},
                        attrs={
                            'ring_id': ring_id,
                            'op_namescope': "/gradient_clip_model_parallelism",
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Optimize,
                        })
                    idx_offset += 1

        # the grad sum here should take the all and only param in the current shard
        to_check_param = set(reversed_x_paramname)
        should_check_param = set(shard.global_params).intersection(set(
            [param for param, worker_idx in shard.global_param2device.items() \
                if worker_idx == shard.worker_idx]))
        assert to_check_param == should_check_param, "amp check_finite_and_unscale \
        checking miss [{}] and got unexpected [{}]".format(
            should_check_param - to_check_param,
            to_check_param - should_check_param)

        for var_name in deperated_vars:
            block._remove_var(var_name, sync=False)
        block._sync_with_cpp()
        return

    # TODO (JZ-LIANG) revise this for uniform mixed parallelism
    def sync_global_norm(self, block, ring_ids, mp_rank):
        """
        prune gradient_clip related ops for params that not belong to cur shard
        prune: square, reduce_sum, elementwise_mul
        keep: sum, sqrt, elementwise_max, elementwise_div
        """
        is_clip_grad_by_global_norm = False
        for idx, op in list(enumerate(block.ops)):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                is_clip_grad_by_global_norm = True
                break
        if not is_clip_grad_by_global_norm:
            # TODO(Yuang Liu): need some extra handles when clip_grad_norm for mp
            return

        removed_op_idx = set()
        removed_tmp_var = set()
        for idx, op in list(enumerate(block.ops)):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                break
            for input_name in op.input_arg_names:
                input_var = block.var(input_name)
                # NOTE: when mp_degree > 1, some vars will be split into each mp rank.
                # However, there still some vars such as Scale, Bias are not split.
                # Those not be split vars should only be counted once during grad clip
                # by global norm. Those vars either doesn't have is_distributed attr
                # or the is_distributed attr has been set as False.
                # Therefore, we prune those duplicated vars for grad clip.
                if mp_rank >= 1 and (not (hasattr(input_var, 'is_distributed')
                                          and input_var.is_distributed)):
                    removed_op_idx.add(idx)
                    for output_name in op.output_arg_names:
                        removed_tmp_var.add(output_name)

        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_gradient_clip_op(op):
                continue
            if idx in removed_op_idx:
                block._remove_op(idx, sync=False)

        for var_name in removed_tmp_var:
            block._remove_var(var_name, sync=False)

        for idx, op in list(enumerate(block.ops)):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                # If mp_rank == 0, no extra handles, just allreduce
                # If mp_rank >= 1, some extra handles is needed
                sum_rst_var = block.var(op.output_arg_names[0])
                if mp_rank >= 1:
                    reserved_vars = []
                    for input_name in op.input_arg_names:
                        if input_name not in removed_tmp_var:
                            reserved_vars.append(input_name)

                    if len(reserved_vars) > 0:
                        op.desc.set_input("X", reserved_vars)
                    else:
                        # If all input of sum op should be removed, then remove the sum op.
                        # And set the output's value of sum to 0.
                        namescope = op.attr("op_namescope")
                        block._remove_op(idx, sync=False)
                        fill_constant_op = block._insert_op_without_sync(
                            idx,
                            type='fill_constant',
                            inputs={},
                            outputs={'Out': sum_rst_var},
                            attrs={
                                'shape': sum_rst_var.shape,
                                'dtype': sum_rst_var.dtype,
                                'value': 0.0,
                                OP_ROLE_KEY: OpRole.Optimize
                            })
                        fill_constant_op._set_attr('op_namescope', namescope)
                self._insert_allreduce(block, ring_ids, idx, sum_rst_var)
                break

    @staticmethod
    def _insert_allreduce(block, ring_ids, idx, var):
        for ring_id in ring_ids:
            if ring_id == -1:
                continue

            idx = idx + 1
            block._insert_op_without_sync(
                idx,
                type='c_allreduce_sum',
                inputs={'X': var},
                outputs={'Out': var},
                attrs={
                    'ring_id': ring_id,
                    'op_namescope': "/gradient_clip_model_parallelism",
                    'use_calc_stream': True,
                    OP_ROLE_KEY: OpRole.Optimize,
                })
