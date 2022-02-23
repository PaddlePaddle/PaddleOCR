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
# limitations under the License.

from paddle.fluid.optimizer import Optimizer

__all__ = []


class MetaOptimizerBase(Optimizer):
    def __init__(self, optimizer):
        self.inner_opt = optimizer
        self._learning_rate = self.inner_opt._learning_rate
        self._learning_rate_map = self.inner_opt._learning_rate_map
        self.meta_optimizers_white_list = []
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        self.loss = loss
        self.role_maker = role_maker
        self.user_defined_optimizer = user_defined_optimizer
        self.user_defined_strategy = user_defined_strategy

    def _update_inner_optimizer(self, optimizer):
        self.inner_opt = optimizer

    def _can_apply(self):
        return False

    def _is_graph_out(self):
        return False

    def _can_update(self, optimizer):
        if str(optimizer.__class__.__name__) in self.meta_optimizers_white_list:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        raise NotImplementedError("you should implement disable strategy in {}".
                                  format(type(self).__name__))

    def _enable_strategy(self, dist_strategy, context=None):
        raise NotImplementedError("you should implement enable strategy in {}".
                                  format(type(self).__name__))

    def apply_gradients(self, params_grads):
        return self.inner_opt.apply_gradients(params_grads=params_grads)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        return self.inner_opt.backward(loss, startup_program, parameter_list,
                                       no_grad_set, callbacks)

    def apply_optimize(self, loss, startup_program, params_grads):
        return self.inner_opt.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        optimize_ops, params_grads = self.minimize_impl(
            loss, startup_program, parameter_list, no_grad_set)
        return optimize_ops, params_grads
