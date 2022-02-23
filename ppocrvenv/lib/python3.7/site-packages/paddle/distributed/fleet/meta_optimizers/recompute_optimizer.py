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

from paddle.fluid.optimizer import RecomputeOptimizer as RO
from .meta_optimizer_base import MetaOptimizerBase

__all__ = []


class RecomputeOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(RecomputeOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.wrapped_opt = None
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = [
            "LarsOptimizer",
            "LambOptimizer",
            "GraphExecutionOptimizer",
            "DGCOptimizer",
        ]
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(RecomputeOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _init_wrapped_opt(self):
        if self.wrapped_opt is not None:
            return

        configs = self.user_defined_strategy.recompute_configs
        self.wrapped_opt = RO(self.inner_opt)
        self.wrapped_opt._set_checkpoints(list(configs["checkpoints"]))
        if configs["enable_offload"]:
            self.wrapped_opt._enable_offload()
            # TODO(JZ-LIANG) might found a way to infer the checkpoint shape automatically
            checkpoint_shapes = list(configs["checkpoint_shape"])
            self.wrapped_opt.checkpoint_shape = checkpoint_shapes

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.recompute == True:
            if len(self.user_defined_strategy.recompute_configs[
                    "checkpoints"]) == 0:
                return False
            else:
                return True

    def _disable_strategy(self, dist_strategy):
        dist_strategy.recompute = False
        dist_strategy.recompute_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        # we do not support automatically recompute checkpoints currently
        return

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        # maybe inner_opt of other meta optimizer
        self._init_wrapped_opt()
        return self.wrapped_opt.backward(loss, startup_program, parameter_list,
                                         no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        return self.wrapped_opt.apply_gradients(params_grads=params_grads)

    def apply_optimize(self, loss, startup_program, params_grads):
        return self.wrapped_opt.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self._init_wrapped_opt()
        optimize_ops, params_grads = \
            self.wrapped_opt.minimize(loss, startup_program,
                                      parameter_list, no_grad_set)
        return optimize_ops, params_grads
