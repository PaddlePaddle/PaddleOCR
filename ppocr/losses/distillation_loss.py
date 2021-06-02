#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn

from .rec_ctc_loss import CTCLoss
from .basic_loss import DMLLoss


class DistillationDMLLoss(DMLLoss):
    """
    """

    def __init__(self,
                 model_name_list1=[],
                 model_name_list2=[],
                 key=None,
                 name="loss_dml"):
        super().__init__(name=name)
        if not isinstance(model_name_list1, (list, )):
            model_name_list1 = [model_name_list1]
        if not isinstance(model_name_list2, (list, )):
            model_name_list2 = [model_name_list2]

        assert len(model_name_list1) == len(model_name_list2)
        self.model_name_list1 = model_name_list1
        self.model_name_list2 = model_name_list2
        self.key = key

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx in range(len(self.model_name_list1)):
            out1 = predicts[self.model_name_list1[idx]]
            out2 = predicts[self.model_name_list2[idx]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                assert len(loss) == 1
                loss = list(loss.values())[0]
            loss_dict["{}_{}".format(self.name, idx)] = loss
        return loss_dict


class DistillationCTCLoss(CTCLoss):
    def __init__(self, model_name_list=[], key=None, name="loss_ctc"):
        super().__init__()
        self.model_name_list = model_name_list
        self.key = key
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for model_name in self.model_name_list:
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            loss = super().forward(out, batch)
            if isinstance(loss, dict):
                assert len(loss) == 1
                loss = list(loss.values())[0]
            loss_dict["{}_{}".format(self.name, model_name)] = loss
        return loss_dict
