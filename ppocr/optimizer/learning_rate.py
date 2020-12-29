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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from paddle.optimizer import lr
from .lr_scheduler import CyclicalCosineDecay


class Linear(object):
    """
    Linear learning rate decay
    Args:
        lr (float): The initial learning rate. It is a python float number.
        epochs(int): The decay step size. It determines the decay cycle.
        end_lr(float, optional): The minimum final learning rate. Default: 0.0001.
        power(float, optional): Power of polynomial. Default: 1.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 epochs,
                 step_each_epoch,
                 end_lr=0.0,
                 power=1.0,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Linear, self).__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs * step_each_epoch
        self.end_lr = end_lr
        self.power = power
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.epochs,
            end_lr=self.end_lr,
            power=self.power,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


class Cosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Cosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = lr.CosineAnnealingDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


class Step(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_size,
                 step_each_epoch,
                 gamma,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Step, self).__init__()
        self.step_size = step_each_epoch * step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = lr.StepDecay(
            learning_rate=self.learning_rate,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


class Piecewise(object):
    """
    Piecewise learning rate decay
    Args:
        boundaries(list): A list of steps numbers. The type of element in the list is python int.
        values(list): A list of learning rate values that will be picked during different epoch boundaries.
            The type of element in the list is python float.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 step_each_epoch,
                 decay_epochs,
                 values,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Piecewise, self).__init__()
        self.boundaries = [step_each_epoch * e for e in decay_epochs]
        self.values = values
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = lr.PiecewiseDecay(
            boundaries=self.boundaries,
            values=self.values,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.values[0],
                last_epoch=self.last_epoch)
        return learning_rate


class CyclicalCosine(object):
    """
    Cyclical cosine learning rate decay
    Args:
        learning_rate(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        cycle(int): period of the cosine learning rate
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 cycle,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(CyclicalCosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)
        self.cycle = round(cycle * step_each_epoch)

    def __call__(self):
        learning_rate = CyclicalCosineDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            cycle=self.cycle,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate
