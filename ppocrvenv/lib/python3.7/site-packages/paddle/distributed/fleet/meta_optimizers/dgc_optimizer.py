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

from paddle.fluid.optimizer import Momentum, DGCMomentumOptimizer
from .meta_optimizer_base import MetaOptimizerBase
import logging

__all__ = []


class DGCOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(DGCOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.dgc_opt = None
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(DGCOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _init_dgc_opt(self):
        if self.dgc_opt is not None:
            return

        opt = self.inner_opt

        if not self.role_maker._is_collective:
            return

        if not isinstance(opt, Momentum):
            return

        configs = self.user_defined_strategy.dgc_configs
        if len(configs['sparsity']) == 0:
            # default is [0.999]
            configs['sparsity'] = [0.999]

        self.dgc_opt = DGCMomentumOptimizer(
            learning_rate=opt._learning_rate,
            momentum=opt._momentum,
            rampup_begin_step=configs['rampup_begin_step'],
            rampup_step=configs['rampup_step'],
            sparsity=configs['sparsity'],
            parameter_list=opt._parameter_list,
            use_nesterov=opt._use_nesterov,
            num_trainers=self.role_maker._worker_num(),
            regularization=opt.regularization,
            grad_clip=opt._grad_clip,
            name=opt._name)

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.dgc:
            if not isinstance(self.inner_opt, Momentum):
                logging.warn("dgc only works on Momentum optimizer")
                return False
            if self.role_maker._worker_num() <= 1:
                logging.warn("dgc only works on multi cards")
                return False

            return True

        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.dgc = False
        dist_strategy.dgc_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.dgc = True
        dist_strategy.dgc_configs = {"rampup_begin_step": 0, "rampup_step": 1}

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        self._init_dgc_opt()
        return self.dgc_opt.backward(loss, startup_program, parameter_list,
                                     no_grad_set, callbacks)

    def apply_gradients(self, params_grads):
        self._init_dgc_opt()
        return self.dgc_opt.apply_gradients(params_grads=params_grads)

    def apply_optimize(self, loss, startup_program, params_grads):
        self._init_dgc_opt()
        return self.dgc_opt.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self._init_dgc_opt()
        optimize_ops, params_grads = \
            self.dgc_opt.minimize(loss, startup_program,
                                  parameter_list, no_grad_set)
        return optimize_ops, params_grads
