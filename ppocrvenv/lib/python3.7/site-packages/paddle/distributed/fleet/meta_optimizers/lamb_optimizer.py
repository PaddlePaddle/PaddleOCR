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

from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.optimizer import LambOptimizer as LAMB
from .meta_optimizer_base import MetaOptimizerBase
import logging

__all__ = []


class LambOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(LambOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.lamb_opt = None
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = ["GraphExecutionOptimizer"]
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(LambOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

        opt = self.inner_opt
        if not isinstance(opt, AdamOptimizer):
            return

        configs = self.user_defined_strategy.lamb_configs
        if len(configs['exclude_from_weight_decay']) == 0:
            _exclude_from_weight_decay_fn = None
        else:

            def exclude_fn(param):
                exclude_list = configs['exclude_from_weight_decay']
                for name in exclude_list:
                    if param.name.endswith(name):
                        return True
                return False

            _exclude_from_weight_decay_fn = exclude_fn

        self.lamb_opt = LAMB(
            learning_rate=opt._learning_rate,
            lamb_weight_decay=configs['lamb_weight_decay'],
            beta1=opt._beta1,
            beta2=opt._beta2,
            epsilon=opt._epsilon,
            parameter_list=opt._parameter_list,
            regularization=opt.regularization,
            grad_clip=opt._grad_clip,
            exclude_from_weight_decay_fn=_exclude_from_weight_decay_fn,
            name=opt._name)

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.lamb:
            if not isinstance(self.inner_opt, AdamOptimizer):
                logging.warn(
                    "lamb need the inner optimizer to be AdamOptimizer optimizer but got {}.".
                    format(self.inner_opt.type))
                return False
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.lamb = False
        dist_strategy.lamb_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.lamb = True
        dist_strategy.lamb_configs = {
            "lamb_weight_decay": 0.01,
            "exclude_from_weight_decay": []
        }

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self.lamb_opt.backward(loss, startup_program, parameter_list,
                                      no_grad_set, callbacks)

    # the following function will be used by AMP if both LARS and AMP are turn on together.
    def apply_gradients(self, params_grads):
        return self.lamb_opt.apply_gradients(params_grads=params_grads)

    def apply_optimize(self, loss, startup_program, params_grads):
        return self.lamb_opt.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        optimize_ops, params_grads = \
            self.lamb_opt.minimize(loss, startup_program,
                                   parameter_list, no_grad_set)
        return optimize_ops, params_grads
