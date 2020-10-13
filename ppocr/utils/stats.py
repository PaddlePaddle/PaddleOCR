# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import numpy as np
import datetime

__all__ = ['TrainingStats', 'Time']


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = collections.deque(maxlen=window_size)

    def add_value(self, value):
        self.deque.append(value)

    def get_median_value(self):
        return np.median(self.deque)


def Time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


class TrainingStats(object):
    def __init__(self, window_size, stats_keys):
        self.window_size = window_size
        self.smoothed_losses_and_metrics = {
            key: SmoothedValue(window_size)
            for key in stats_keys
        }

    def update(self, stats):
        for k, v in stats.items():
            if k not in self.smoothed_losses_and_metrics:
                self.smoothed_losses_and_metrics[k] = SmoothedValue(
                    self.window_size)
            self.smoothed_losses_and_metrics[k].add_value(v)

    def get(self, extras=None):
        stats = collections.OrderedDict()
        if extras:
            for k, v in extras.items():
                stats[k] = v
        for k, v in self.smoothed_losses_and_metrics.items():
            stats[k] = round(v.get_median_value(), 6)

        return stats

    def log(self, extras=None):
        d = self.get(extras)
        strs = ', '.join(str(dict({x: y})).strip('{}') for x, y in d.items())
        return strs
