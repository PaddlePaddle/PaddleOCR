# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from paddle.fluid.contrib.sparsity.asp import ASPHelper
from .meta_optimizer_base import MetaOptimizerBase

__all__ = []


class ASPOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ASPOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = [
            "AMPOptimizer", "LarsOptimizer", "LambOptimizer",
            "GraphExecutionOptimizer", "RecomputeOptimizer",
            "GradientMergeOptimizer"
        ]
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(ASPOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False

        if self.user_defined_strategy.asp:
            return True

        return False

    def _disable_strategy(self, dist_strategy):
        dist_strategy.asp = False

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.asp = True

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):

        optimize_ops, params_grads = ASPHelper._minimize(
            self.inner_opt,
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        return optimize_ops, params_grads
