#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss


class DBLoss(object):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self, params):
        super(DBLoss, self).__init__()
        self.balance_loss = params['balance_loss']
        self.main_loss_type = params['main_loss_type']

        self.alpha = params['alpha']
        self.beta = params['beta']
        self.ohem_ratio = params['ohem_ratio']

    def __call__(self, predicts, labels):
        label_shrink_map = labels['shrink_map']
        label_shrink_mask = labels['shrink_mask']
        label_threshold_map = labels['threshold_map']
        label_threshold_mask = labels['threshold_mask']
        pred = predicts['maps']
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]

        loss_shrink_maps = BalanceLoss(
            shrink_maps,
            label_shrink_map,
            label_shrink_mask,
            balance_loss=self.balance_loss,
            main_loss_type=self.main_loss_type,
            negative_ratio=self.ohem_ratio)
        loss_threshold_maps = MaskL1Loss(threshold_maps, label_threshold_map,
                                         label_threshold_mask)
        loss_binary_maps = DiceLoss(binary_maps, label_shrink_map,
                                    label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        loss_all = loss_shrink_maps + loss_threshold_maps\
            + loss_binary_maps
        losses = {'total_loss':loss_all,\
            "loss_shrink_maps":loss_shrink_maps,\
            "loss_threshold_maps":loss_threshold_maps,\
            "loss_binary_maps":loss_binary_maps}
        return losses
