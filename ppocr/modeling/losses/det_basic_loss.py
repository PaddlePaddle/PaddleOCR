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

import numpy as np

import paddle.fluid as fluid


def BalanceLoss(pred,
                gt,
                mask,
                balance_loss=True,
                main_loss_type="DiceLoss",
                negative_ratio=3,
                return_origin=False,
                eps=1e-6):
    """
    The BalanceLoss for Differentiable Binarization text detection
    args:
        pred (variable): predicted feature maps.
        gt (variable): ground truth feature maps.
        mask (variable): masked maps.
        balance_loss (bool): whether balance loss or not, default is True
        main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
            'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
        negative_ratio (int|float): float, default is 3.
        return_origin (bool): whether return unbalanced loss or not, default is False.
        eps (float): default is 1e-6.
    return: (variable) balanced loss
    """
    positive = gt * mask
    negative = (1 - gt) * mask

    positive_count = fluid.layers.reduce_sum(positive)
    positive_count_int = fluid.layers.cast(positive_count, dtype=np.int32)
    negative_count = min(
        fluid.layers.reduce_sum(negative), positive_count * negative_ratio)
    negative_count_int = fluid.layers.cast(negative_count, dtype=np.int32)

    if main_loss_type == "CrossEntropy":
        loss = fluid.layers.cross_entropy(input=pred, label=gt, soft_label=True)
        loss = fluid.layers.reduce_mean(loss)
    elif main_loss_type == "Euclidean":
        loss = fluid.layers.square(pred - gt)
        loss = fluid.layers.reduce_mean(loss)
    elif main_loss_type == "DiceLoss":
        loss = DiceLoss(pred, gt, mask)
    elif main_loss_type == "BCELoss":
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(pred, label=gt)
    elif main_loss_type == "MaskL1Loss":
        loss = MaskL1Loss(pred, gt, mask)
    else:
        loss_type = [
            'CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss', 'MaskL1Loss'
        ]
        raise Exception("main_loss_type in BalanceLoss() can only be one of {}".
                        format(loss_type))

    if not balance_loss:
        return loss

    positive_loss = positive * loss
    negative_loss = negative * loss
    negative_loss = fluid.layers.reshape(negative_loss, shape=[-1])
    negative_loss, _ = fluid.layers.topk(negative_loss, k=negative_count_int)
    balance_loss = (fluid.layers.reduce_sum(positive_loss) +
                    fluid.layers.reduce_sum(negative_loss)) / (
                        positive_count + negative_count + eps)

    if return_origin:
        return balance_loss, loss
    return balance_loss


def DiceLoss(pred, gt, mask, weights=None, eps=1e-6):
    """
    DiceLoss function.
    """

    assert pred.shape == gt.shape
    assert pred.shape == mask.shape
    if weights is not None:
        assert weights.shape == mask.shape
        mask = weights * mask
    intersection = fluid.layers.reduce_sum(pred * gt * mask)

    union = fluid.layers.reduce_sum(pred * mask) + fluid.layers.reduce_sum(
        gt * mask) + eps
    loss = 1 - 2.0 * intersection / union
    assert loss <= 1
    return loss


def MaskL1Loss(pred, gt, mask, eps=1e-6):
    """
    Mask L1 Loss
    """
    loss = fluid.layers.reduce_sum((fluid.layers.abs(pred - gt) * mask)) / (
        fluid.layers.reduce_sum(mask) + eps)
    loss = fluid.layers.reduce_mean(loss)
    return loss
