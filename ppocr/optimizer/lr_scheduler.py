# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from paddle.optimizer.lr import LRScheduler


class CyclicalCosineDecay(LRScheduler):
    def __init__(
        self, learning_rate, T_max, cycle=1, last_epoch=-1, eta_min=0.0, verbose=False
    ):
        """
        Cyclical cosine learning rate decay
        A learning rate which can be referred in https://arxiv.org/pdf/2012.12645.pdf
        Args:
            learning rate(float): learning rate
            T_max(int): maximum epoch num
            cycle(int): period of the cosine decay
            last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
            eta_min(float): minimum learning rate during training
            verbose(bool): whether to print learning rate for each epoch
        """
        super(CyclicalCosineDecay, self).__init__(learning_rate, last_epoch, verbose)
        self.cycle = cycle
        self.eta_min = eta_min

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        reletive_epoch = self.last_epoch % self.cycle
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * reletive_epoch / self.cycle)
        )
        return lr


class OneCycleDecay(LRScheduler):
    """
    One Cycle learning rate decay
    A learning rate which can be referred in https://arxiv.org/abs/1708.07120
    Code referred in https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    """

    def __init__(
        self,
        max_lr,
        epochs=None,
        steps_per_epoch=None,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
    ):
        # Validate total_steps
        if epochs <= 0 or not isinstance(epochs, int):
            raise ValueError(
                "Expected positive integer epochs, but got {}".format(epochs)
            )
        if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
            raise ValueError(
                "Expected positive integer steps_per_epoch, but got {}".format(
                    steps_per_epoch
                )
            )
        self.total_steps = epochs * steps_per_epoch

        self.max_lr = max_lr
        self.initial_lr = self.max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor

        if three_phase:
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": self.initial_lr,
                    "end_lr": self.max_lr,
                },
                {
                    "end_step": float(2 * pct_start * self.total_steps) - 2,
                    "start_lr": self.max_lr,
                    "end_lr": self.initial_lr,
                },
                {
                    "end_step": self.total_steps - 1,
                    "start_lr": self.initial_lr,
                    "end_lr": self.min_lr,
                },
            ]
        else:
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": self.initial_lr,
                    "end_lr": self.max_lr,
                },
                {
                    "end_step": self.total_steps - 1,
                    "start_lr": self.max_lr,
                    "end_lr": self.min_lr,
                },
            ]

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(
                "Expected float between 0 and 1 pct_start, but got {}".format(pct_start)
            )

        # Validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                "anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(
                    anneal_strategy
                )
            )
        elif anneal_strategy == "cos":
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == "linear":
            self.anneal_func = self._annealing_linear

        super(OneCycleDecay, self).__init__(max_lr, last_epoch, verbose)

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_lr(self):
        computed_lr = 0.0
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                "Tried to step {} times. The specified number of total steps is {}".format(
                    step_num + 1, self.total_steps
                )
            )
        start_step = 0
        for i, phase in enumerate(self._schedule_phases):
            end_step = phase["end_step"]
            if step_num <= end_step or i == len(self._schedule_phases) - 1:
                pct = (step_num - start_step) / (end_step - start_step)
                computed_lr = self.anneal_func(phase["start_lr"], phase["end_lr"], pct)
                break
            start_step = phase["end_step"]

        return computed_lr


class TwoStepCosineDecay(LRScheduler):
    def __init__(
        self, learning_rate, T_max1, T_max2, eta_min=0, last_epoch=-1, verbose=False
    ):
        if not isinstance(T_max1, int):
            raise TypeError(
                "The type of 'T_max1' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max1)
            )
        if not isinstance(T_max2, int):
            raise TypeError(
                "The type of 'T_max2' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max2)
            )
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingDecay' must be 'float, int', but received %s."
                % type(eta_min)
            )
        assert T_max1 > 0 and isinstance(
            T_max1, int
        ), " 'T_max1' must be a positive integer."
        assert T_max2 > 0 and isinstance(
            T_max2, int
        ), " 'T_max1' must be a positive integer."
        self.T_max1 = T_max1
        self.T_max2 = T_max2
        self.eta_min = float(eta_min)
        super(TwoStepCosineDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.T_max1:
            if self.last_epoch == 0:
                return self.base_lr
            elif (self.last_epoch - 1 - self.T_max1) % (2 * self.T_max1) == 0:
                return (
                    self.last_lr
                    + (self.base_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max1))
                    / 2
                )

            return (1 + math.cos(math.pi * self.last_epoch / self.T_max1)) / (
                1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max1)
            ) * (self.last_lr - self.eta_min) + self.eta_min
        else:
            if (self.last_epoch - 1 - self.T_max2) % (2 * self.T_max2) == 0:
                return (
                    self.last_lr
                    + (self.base_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max2))
                    / 2
                )

            return (1 + math.cos(math.pi * self.last_epoch / self.T_max2)) / (
                1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max2)
            ) * (self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        if self.last_epoch <= self.T_max1:
            return (
                self.eta_min
                + (self.base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max1))
                / 2
            )
        else:
            return (
                self.eta_min
                + (self.base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max2))
                / 2
            )
