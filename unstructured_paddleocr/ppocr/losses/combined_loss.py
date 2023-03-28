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

import paddle
import paddle.nn as nn

from .rec_ctc_loss import CTCLoss
from .center_loss import CenterLoss
from .ace_loss import ACELoss
from .rec_sar_loss import SARLoss

from .distillation_loss import DistillationCTCLoss
from .distillation_loss import DistillationSARLoss
from .distillation_loss import DistillationDMLLoss
from .distillation_loss import DistillationDistanceLoss, DistillationDBLoss, DistillationDilaDBLoss
from .distillation_loss import DistillationVQASerTokenLayoutLMLoss, DistillationSERDMLLoss
from .distillation_loss import DistillationLossFromOutput
from .distillation_loss import DistillationVQADistanceLoss


class CombinedLoss(nn.Layer):
    """
    CombinedLoss:
        a combionation of loss function
    """

    def __init__(self, loss_config_list=None):
        super().__init__()
        self.loss_func = []
        self.loss_weight = []
        assert isinstance(loss_config_list, list), (
            'operator config should be a list')
        for config in loss_config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            self.loss_weight.append(param.pop("weight"))
            self.loss_func.append(eval(name)(**param))

    def forward(self, input, batch, **kargs):
        loss_dict = {}
        loss_all = 0.
        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, batch, **kargs)
            if isinstance(loss, paddle.Tensor):
                loss = {"loss_{}_{}".format(str(loss), idx): loss}

            weight = self.loss_weight[idx]

            loss = {key: loss[key] * weight for key in loss}

            if "loss" in loss:
                loss_all += loss["loss"]
            else:
                loss_all += paddle.add_n(list(loss.values()))
            loss_dict.update(loss)
        loss_dict["loss"] = loss_all
        return loss_dict
