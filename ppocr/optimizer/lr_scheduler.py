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
    def __init__(self,
                 learning_rate,
                 T_max,
                 cycle=1,
                 last_epoch=-1,
                 eta_min=0.0,
                 verbose=False):
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
        super(CyclicalCosineDecay, self).__init__(learning_rate, last_epoch,
                                                  verbose)
        self.cycle = cycle
        self.eta_min = eta_min

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        reletive_epoch = self.last_epoch % self.cycle
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                (1 + math.cos(math.pi * reletive_epoch / self.cycle))
        return lr
