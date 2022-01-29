# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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


class EtaTimer:
    def __init__(self, total_steps, slide_ratio=0.9):
        """A surplus training time evaluator.
        Args:
            total_steps (int): total train steps in whole training stage.
            slide_ratio (float): determine the smoothness of the strategy of 
                surplus time calculation, bigger is smoother, its default 
                value is 0.9.
        """
        self.total_steps = total_steps
        self.surplus_steps = total_steps
        self.seconds = 0
        self.slide_ratio = slide_ratio
        self.step = 1

    def get_eta(self, time_seconds_cost, steps_cost):
        """Accroding to
        Args:
            time_seconds_cost: time cost in a certain stage, unit second.
            steps_cost: in `time_seconds_cost`, how many train(or eval) steps 
                has been consumed.
        """
        self.surplus_steps -= steps_cost
        self.seconds = self.slide_ratio * self.seconds + \
            (1 - self.slide_ratio) * time_seconds_cost / steps_cost
        no_seconds_bias = self.seconds / (1 - self.slide_ratio**self.step)
        ret = self._format_seconds(int(self.surplus_steps * no_seconds_bias))
        self.step += 1
        return ret

    def reset(self, new_total_steps=None):
        """Reset member variables of EtaTimer class.
        Args:
            new_total_steps: new value of self.surplus_steps, if set None, 
                only reset to original total_steps
        """
        self.surplus_steps = \
            new_total_steps if new_total_steps is not None else self.total_steps
        self.seconds = 0
        self.step = 1

    def _format_seconds(self, seconds):
        """Transform seconds to day|hour|minute|second format.
        Args:
            seconds: input total seconds.
        """
        seconds = max(0, seconds)  # in the final step, seconds maybe negative
        d, s = divmod(seconds, 86400)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f'{d}d{h}h{m}m{s}s'
