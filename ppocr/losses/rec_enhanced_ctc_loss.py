# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from .ace_loss import ACELoss
from .center_loss import CenterLoss
from .rec_ctc_loss import CTCLoss


class EnhancedCTCLoss(nn.Layer):
    def __init__(self,
                 use_focal_loss=False,
                 use_ace_loss=False,
                 ace_loss_weight=0.1,
                 use_center_loss=False,
                 center_loss_weight=0.05,
                 num_classes=6625,
                 feat_dim=96,
                 init_center=False,
                 center_file_path=None,
                 **kwargs):
        super(EnhancedCTCLoss, self).__init__()
        self.ctc_loss_func = CTCLoss(use_focal_loss=use_focal_loss)

        self.use_ace_loss = False
        if use_ace_loss:
            self.use_ace_loss = use_ace_loss
            self.ace_loss_func = ACELoss()
            self.ace_loss_weight = ace_loss_weight

        self.use_center_loss = False
        if use_center_loss:
            self.use_center_loss = use_center_loss
            self.center_loss_func = CenterLoss(
                num_classes=num_classes,
                feat_dim=feat_dim,
                init_center=init_center,
                center_file_path=center_file_path)
            self.center_loss_weight = center_loss_weight

    def __call__(self, predicts, batch):
        loss = self.ctc_loss_func(predicts, batch)["loss"]

        if self.use_center_loss:
            center_loss = self.center_loss_func(
                predicts, batch)["loss_center"] * self.center_loss_weight
            loss = loss + center_loss

        if self.use_ace_loss:
            ace_loss = self.ace_loss_func(
                predicts, batch)["loss_ace"] * self.ace_loss_weight
            loss = loss + ace_loss

        return {'enhanced_ctc_loss': loss}
