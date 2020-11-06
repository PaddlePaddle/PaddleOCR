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

import copy

__all__ = ['build_optimizer']


def build_lr_scheduler(lr_config, epochs, step_each_epoch):
    from . import learning_rate
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = getattr(learning_rate, lr_name)(**lr_config)()
    else:
        lr = lr_config['learning_rate']
    return lr


def build_optimizer(config, epochs, step_each_epoch, parameters):
    from . import regularizer, optimizer
    config = copy.deepcopy(config)
    # step1 build lr
    lr = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch)

    # step2 build regularization
    if 'regularizer' in config and config['regularizer'] is not None:
        reg_config = config.pop('regularizer')
        reg_name = reg_config.pop('name') + 'Decay'
        reg = getattr(regularizer, reg_name)(**reg_config)()
    else:
        reg = None

    # step3 build optimizer
    optim_name = config.pop('name')
    optim = getattr(optimizer, optim_name)(learning_rate=lr,
                                           weight_decay=reg,
                                           **config)
    return optim(parameters), lr
