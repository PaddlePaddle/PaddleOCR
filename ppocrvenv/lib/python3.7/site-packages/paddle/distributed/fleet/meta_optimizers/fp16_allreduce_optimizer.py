#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import core, framework, unique_name
from .meta_optimizer_base import MetaOptimizerBase

__all__ = []


class FP16AllReduceOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(FP16AllReduceOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = [
            "LarsOptimizer",
            "LambOptimizer",
            "RecomputeOptimizer",
            "LocalSGDOptimizer",
            "GradientMergeOptimizer",
            "GraphExecutionOptimizer",
            "AdaptiveLocalSGDOptimizer",
        ]
        self.meta_optimizers_black_list = ["DGCOptimizer"]

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(FP16AllReduceOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.fp16_allreduce:
            return True

        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.fp16_allreduce = False

    def _enable_strategy(self, dist_strategy, context=None):
        dist_strategy.fp16_allreduce = True

    @staticmethod
    def fp16_compression(param_and_grads):
        """
        Compress fp32 gradients to fp16 during allreduce.
        """
        op_maker = core.op_proto_and_checker_maker

        new_param_and_grads = []  # param, grad, is_cast
        # cast grad from fp32->fp16 before allreduce,
        for param, grad in param_and_grads:
            if grad is None or grad.dtype != core.VarDesc.VarType.FP32:
                new_param_and_grads.append((param, grad, False))
                continue

            op = grad.op
            block = grad.block
            var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
            if param.name not in var_attr:
                new_param_and_grads.append((param, grad, False))
                continue

            # remove (param, grad) from op_role_var
            var_attr.remove(param.name)
            var_attr.remove(grad.name)
            if len(var_attr) > 1:
                op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
            else:
                op._remove_attr(op_maker.kOpRoleVarAttrName())

            new_grad = block.create_var(
                name=unique_name.generate(grad.name + ".cast_fp16"),
                dtype=core.VarDesc.VarType.FP16,
                persistable=False,
                stop_gradient=True)

            with block.program._backward_role_guard():
                cast_op = block.append_op(
                    type="cast",
                    inputs={"X": grad},
                    outputs={"Out": new_grad},
                    attrs={
                        "in_dtype": core.VarDesc.VarType.FP32,
                        "out_dtype": core.VarDesc.VarType.FP16
                    },
                    stop_gradient=True)

                backward = op_maker.OpRole.Backward
                cast_op._set_attr(op_maker.kOpRoleAttrName(), backward)
                cast_op._set_attr(op_maker.kOpRoleVarAttrName(),
                                  [param.name, new_grad.name])
                new_grad.op = cast_op

            new_param_and_grads.append((param, new_grad, True))

        ret_param_and_grads = []
        # cast grad from fp16->fp32 after allreduce.
        # NOTE. Now we split fp16 compression into two for loops,
        # if we do not separate them, fuse allreduce will wrong.
        # This must be the problem of fuse allreduce pass, need
        # fixed in future.
        for param, grad, cast in new_param_and_grads:
            if not cast:
                ret_param_and_grads.append((param, grad))
                continue

            block = grad.block
            new_grad = block.create_var(
                name=unique_name.generate(grad.name + ".cast_fp32"),
                dtype=core.VarDesc.VarType.FP32,
                persistable=False,
                stop_gradient=True)

            with block.program._optimized_guard(
                [param, grad]), framework.name_scope('fp16_allreduce'):
                cast_op = block.append_op(
                    type="cast",
                    inputs={"X": grad},
                    outputs={"Out": new_grad},
                    attrs={
                        "in_dtype": core.VarDesc.VarType.FP16,
                        "out_dtype": core.VarDesc.VarType.FP32
                    },
                    stop_gradient=True)
            ret_param_and_grads.append((param, new_grad))

        return ret_param_and_grads

    def apply_optimize(self, loss, startup_program, params_grads):
        new_params_grads = self.fp16_compression(params_grads)
        return self.inner_opt.apply_optimize(
            loss,
            startup_program=startup_program,
            params_grads=new_params_grads)
