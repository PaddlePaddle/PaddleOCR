# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/basic_loss.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle
from paddle import nn
import paddle.nn.functional as F


class BalanceLoss(nn.Layer):
    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 negative_ratio=3,
                 return_origin=False,
                 eps=1e-6,
                 **kwargs):
        """
               The BalanceLoss for Differentiable Binarization text detection
               args:
                   balance_loss (bool): whether balance loss or not, default is True
                   main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
                       'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
                   negative_ratio (int|float): float, default is 3.
                   return_origin (bool): whether return unbalanced loss or not, default is False.
                   eps (float): default is 1e-6.
               """
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == "Euclidean":
            self.loss = nn.MSELoss()
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction='none')
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = [
                'CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss', 'MaskL1Loss'
            ]
            raise Exception(
                "main_loss_type in BalanceLoss() can only be one of {}".format(
                    loss_type))

    def forward(self, pred, gt, mask=None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(
            min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = paddle.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            # negative_loss, _ = paddle.topk(negative_loss, k=negative_count_int)
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class DiceLoss(nn.Layer):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        DiceLoss function.
        """

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = paddle.sum(pred * gt * mask)

        union = paddle.sum(pred * mask) + paddle.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Layer):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        Mask L1 Loss
        """
        loss = (paddle.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = paddle.mean(loss)
        return loss


class BCELoss(nn.Layer):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss
