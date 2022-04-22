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

import paddle
from paddle import nn

from .rec_ctc_loss import CTCLoss
from .rec_sar_loss import SARLoss


class MultiLoss(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_funcs = {}
        self.loss_list = kwargs.pop('loss_config_list')
        self.weight_1 = kwargs.get('weight_1', 1.0)
        self.weight_2 = kwargs.get('weight_2', 1.0)
        self.gtc_loss = kwargs.get('gtc_loss', 'sar')
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def forward(self, predicts, batch):
        self.total_loss = {}
        total_loss = 0.0
        # batch [image, label_ctc, label_sar, length, valid_ratio]
        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                loss = loss_func(predicts['ctc'],
                                 batch[:2] + batch[3:])['loss'] * self.weight_1
            elif name == 'SARLoss':
                loss = loss_func(predicts['sar'],
                                 batch[:1] + batch[2:])['loss'] * self.weight_2
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiLoss yet'.format(name))
            self.total_loss[name] = loss
            total_loss += loss
        self.total_loss['loss'] = total_loss
        return self.total_loss
