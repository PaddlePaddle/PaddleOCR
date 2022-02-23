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

from ..common import is_optimizer_op, OP_ROLE_KEY, OpRole, is_update_op
from paddle.fluid import core, unique_name

__all__ = []


class OffloadHelper(object):
    cpu_place_type = 0
    cuda_place_type = 1
    cuda_pinned_place_type = 2

    def __init__(self):
        pass
        "0: dst is on CPUPlace. "
        "1: dst is on CUDAPlace. "
        "2: dst is on CUDAPinnedPlace. "

    def _insert_cast_op(self, block, idx, src_name, dst_name):
        src_var = block.var(src_name)
        if not block.has_var(dst_name):
            block.create_var(
                name=dst_name,
                shape=src_var.shape,
                dtype=core.VarDesc.VarType.FP16,
                persistable=True)
        dst_var = block.var(dst_name)
        assert dst_var.dtype == core.VarDesc.VarType.FP16
        block._insert_op_without_sync(
            idx,
            type='cast',
            inputs={'X': src_var},
            outputs={'Out': dst_var},
            attrs={
                'in_dtype': src_var.dtype,
                'out_dtype': dst_var.dtype,
                OP_ROLE_KEY: OpRole.Optimize
            })

    def _insert_memcpy_op(self, block, idx, src_name, dst_name, dst_place_type):
        src_var = block.var(src_name)
        dst_var = block.var(dst_name)
        block._insert_op_without_sync(
            idx,
            type='memcpy',
            inputs={'X': src_var},
            outputs={'Out': dst_var},
            attrs={
                'dst_place_type': dst_place_type,
                OP_ROLE_KEY: OpRole.Optimize,
            })

    def _insert_fetch_op(self, block, idx, src_name, dst_name):
        self._insert_memcpy_op(block, idx, src_name, dst_name,
                               OffloadHelper.cuda_place_type)

    def _insert_offload_op(self, block, idx, src_name, dst_name):
        self._insert_memcpy_op(block, idx, src_name, dst_name,
                               OffloadHelper.cuda_pinned_place_type)

    def _get_offload_var_name(self, name):
        return unique_name.generate(name + '@offload')

    def _create_offload_var(self, var_name, offload_var_name, blocks):
        for block in blocks:
            var = block.var(var_name)
            var.persistable = False
            offload_var = block.create_var(
                name=offload_var_name,
                shape=var.shape,
                dtype=var.dtype,
                persistable=True)

    def offload_fp32param(self, block, startup_block, offload=True):
        """
        (p_fp16) = cast(p)
        (p_fp16_recompute) = cast(p)
        (pout,) = adam(p)
        ===========================>
        rename(p_fp16_recompute, p_fp16)

        (p,) = prefetch(p@offload)
        (pout,) = adam(p)
        (p_fp16) = cast(p)
        (p@offload) = memcpy(p)
        """
        param_to_idx = dict()
        param_to_fp16 = dict()
        # recompute_var which need rename to fp16_param
        fp16_param_to_recompute = dict()
        recompute_to_fp16 = dict()

        def remove_param(input_name):
            param_to_idx.pop(input_name)
            if input_name in param_to_fp16:
                fp16_param = param_to_fp16.pop(input_name)
                if fp16_param in fp16_param_to_recompute:
                    recompute = fp16_param_to_recompute.pop(fp16_param)
                    recompute_to_fp16.pop(recompute)

        # step1: record param
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_update_op(op):
                param = op.desc.input("Param")[0]
                param_to_idx[param] = idx

        # step2: remove param which can't offload and
        #        record param->fp16param, fp16param->recompute_var
        for idx, op in enumerate(block.ops):
            if is_optimizer_op(op):
                break
            # TODO (Yuang Liu): tmp solution for fuse_grad_merge + optimize_cast
            if not offload and op.type == 'coalesce_tensor':
                continue
            for input_name in op.desc.input_arg_names():
                if input_name not in param_to_idx:
                    continue

                # param which will be used by fp32 op
                if op.type != 'cast':
                    remove_param(input_name)
                    continue

                # param is only used by cast op,
                # which to cast fp32_param to fp16_param
                output_name = op.output_arg_names[0]
                if 'cast_fp16' not in output_name:
                    remove_param(input_name)
                    continue

                if 'subprog' not in output_name:
                    assert output_name == input_name + '.cast_fp16'
                    assert input_name not in param_to_fp16, \
                        "There must be only one cast op from fp32 param to fp16 param."
                    param_to_fp16[input_name] = output_name
                else:
                    # fp16-->recompute_var
                    assert input_name in param_to_fp16, \
                        "param must first be cast to fp16"
                    fp16_param = param_to_fp16[input_name]
                    fp16_param_to_recompute[fp16_param] = output_name
                    recompute_to_fp16[output_name] = fp16_param

        param_name_to_offload_name = dict()
        # step3: main_block add offload, cast op
        # change recompute to fp16, remove cast(param) to fp16
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_update_op(op):
                param = op.desc.input("Param")[0]
                if param not in param_to_idx: continue
                # step3.1: create offload_var
                offload_var_name = self._get_offload_var_name(param)
                param_name_to_offload_name[param] = offload_var_name
                if offload:
                    self._create_offload_var(param, offload_var_name,
                                             [block, startup_block])

                    # step3.2: insert cast op and offload op
                    self._insert_offload_op(block, idx + 1, param,
                                            offload_var_name)

                assert param in param_to_fp16
                fp16_param_name = param_to_fp16[param]
                fp16_param_var = block.var(fp16_param_name)
                fp16_param_var.persistable = True
                self._insert_cast_op(block, idx + 1, param,
                                     param_to_fp16[param])

                if offload:
                    # step3.3: insert fetch op
                    self._insert_fetch_op(block, idx, offload_var_name, param)
                continue

            # step3.4: remove cast op
            if op.type == 'cast':
                input_name = op.desc.input_arg_names()[0]
                if input_name in param_to_idx:
                    block._remove_op(idx, sync=False)
                    continue

            # step3.5: change recompute_param to fp16_param
            for input_name in op.desc.input_arg_names():
                if input_name in recompute_to_fp16:
                    op._rename_input(input_name, recompute_to_fp16[input_name])
            for output_name in op.desc.output_arg_names():
                if output_name in recompute_to_fp16:
                    op._rename_output(output_name,
                                      recompute_to_fp16[output_name])

        # step4: remove recompute_param
        for name in recompute_to_fp16.keys():
            block._remove_var(name, sync=False)

        # step5: startup_block add offload
        visited_vars = set()
        for idx, op in reversed(list(enumerate(startup_block.ops))):
            for out_name in op.output_arg_names:
                if out_name in visited_vars:
                    continue

                if out_name in param_name_to_offload_name:
                    var_name = out_name
                    if offload:
                        offload_var_name = param_name_to_offload_name[var_name]
                        self._insert_offload_op(startup_block, idx + 1,
                                                var_name, offload_var_name)
                    self._insert_cast_op(startup_block, idx + 1, var_name,
                                         param_to_fp16[var_name])

                visited_vars.add(out_name)

        block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def cast_fp32param_in_optimize(self, block, startup_block):
        """
        (p_fp16) = cast(p)
        (p_fp16_recompute) = cast(p)
        (pout,) = adam(p)
        ===========================>
        rename(p_fp16_recompute, p_fp16)

        (pout,) = adam(p)
        (p_fp16) = cast(p)
        """
        self.offload_fp32param(block, startup_block, offload=False)

    def offload(self, block, startup_block):
        """
        (m1, m2) = prefetch(m1@offload, m2@offload)
        (m1out, m2out, pout) = adam(m1, m2, p)
        (m1@offload, m2@offload) = memcpy(m1, m2)
        """
        vars_name_to_offload_name = dict()

        # main_block add offload
        for idx, op in reversed(list(enumerate(block.ops))):
            if not is_optimizer_op(op):
                break

            vars_name = []
            if op.type == "adam":
                # {Moment1Out = [''], Moment2Out = [''], ParamOut = ['']} =
                # adam(inputs={Moment1 = [''], Moment2 = [''], Param = ['']})
                vars_name.append(op.desc.input("Moment1")[0])
                vars_name.append(op.desc.input("Moment2")[0])
            elif op.type == 'momentum':
                pass
            elif op.type == 'lars':
                pass
            elif op.type == 'lamb':
                pass

            # step1: create and init offload_var
            for var_name in vars_name:
                assert var_name not in vars_name_to_offload_name

                offload_var_name = self._get_offload_var_name(var_name)
                vars_name_to_offload_name[var_name] = offload_var_name

                self._create_offload_var(var_name, offload_var_name,
                                         [block, startup_block])

            # step2: insert offload op
            for var_name in vars_name:
                offload_var_name = vars_name_to_offload_name[var_name]
                self._insert_offload_op(block, idx + 1, var_name,
                                        offload_var_name)

            # step3: insert fetch op
            for var_name in vars_name:
                offload_var_name = vars_name_to_offload_name[var_name]
                self._insert_fetch_op(block, idx, offload_var_name, var_name)

        # startup_block add offload
        visited_vars = set()
        for idx, op in reversed(list(enumerate(startup_block.ops))):
            for out_name in op.output_arg_names:
                if out_name in visited_vars:
                    continue

                if out_name in vars_name_to_offload_name:
                    var_name = out_name
                    offload_var_name = vars_name_to_offload_name[var_name]
                    # insert offload op after var is generated
                    self._insert_offload_op(startup_block, idx + 1, var_name,
                                            offload_var_name)
                visited_vars.add(out_name)

        block._sync_with_cpp()
        startup_block._sync_with_cpp()
