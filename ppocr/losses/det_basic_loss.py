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
    def __init__(
        self,
        balance_loss=True,
        main_loss_type="DiceLoss",
        negative_ratio=3,
        return_origin=False,
        eps=1e-6,
        **kwargs,
    ):
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
            self.loss = BCELoss(reduction="none")
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = [
                "CrossEntropy",
                "DiceLoss",
                "Euclidean",
                "BCELoss",
                "MaskL1Loss",
            ]
            raise Exception(
                "main_loss_type in BalanceLoss() can only be one of {}".format(
                    loss_type
                )
            )

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
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))
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
                positive_count + negative_count + self.eps
            )
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
    def __init__(self, reduction="mean"):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


class MaskedFocalLoss(nn.Layer):
    """
    Binary Focal Loss with mask support, designed for text segmentation tasks.

    Focal Loss addresses class imbalance by down-weighting easy examples and
    focusing training on hard examples:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Compared to OHEM (which hard-selects a fixed ratio of negatives), Focal Loss
    applies a continuous per-pixel weight that gracefully scales with difficulty,
    making it a strictly superior drop-in for the OHEM + DiceLoss pattern when
    DiceLoss returns a scalar and OHEM has no discriminating effect.

    Args:
        alpha (float): Balancing factor for the positive (text) class.
            Since text pixels are a small minority, alpha > 0.5 gives them
            higher weight. Default: 0.75.
        gamma (float): Focusing parameter. gamma=0 reduces to masked BCE.
            gamma=2 is the standard value from the original Focal Loss paper.
            Default: 2.0.
        eps (float): Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super(MaskedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        Args:
            pred  (Tensor): Predicted probability map, shape (B, H, W), in [0, 1].
                            (i.e. after sigmoid — the direct output of DBHead.binarize)
            gt    (Tensor): Binary ground-truth map, shape (B, H, W), values 0 or 1.
            mask  (Tensor): Valid-pixel mask, shape (B, H, W), values 0 or 1.
                            Pixels with mask=0 are ignored regions (e.g. too-small text).
        Returns:
            Tensor: Scalar focal loss averaged over valid (mask=1) pixels.
        """
        # F.sigmoid_focal_loss expects a logit (pre-sigmoid) input and applies
        # sigmoid internally using the numerically stable log-sum-exp form:
        #   log(σ(x)) = -softplus(-x),  log(1-σ(x)) = -softplus(x)
        # This avoids the log(0) issue of the manual implementation.
        # Since pred is already a probability (post-sigmoid from DBHead), we
        # invert it: logit = log(p / (1-p)).  The round-trip is numerically safe
        # after clamping, and the stable path inside paddle takes over from there.
        pred = paddle.clip(pred, self.eps, 1.0 - self.eps)
        logit = paddle.log(pred / (1.0 - pred))

        # Per-pixel focal loss, shape (B, H, W)
        # reduction='none' so we can apply the mask ourselves
        loss = F.sigmoid_focal_loss(
            logit,
            gt,
            normalizer=None,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="none",
        )

        # Average over valid (mask=1) pixels only
        return (loss * mask).sum() / (mask.sum() + self.eps)


class DiceFocalLoss(nn.Layer):
    """
    Combined DiceLoss + MaskedFocalLoss for binary text segmentation.

    Rationale for the combination:
    - DiceLoss optimizes the global F1 / region overlap between prediction and GT.
      It is naturally robust to class imbalance (text vs background) because it
      normalizes by the sum of both sets, not by pixel count.
    - MaskedFocalLoss provides per-pixel supervision with adaptive hard-example
      weighting. It compensates for DiceLoss being a global metric that cannot
      distinguish which specific pixels are mispredicted.
    Together they provide complementary supervision: DiceLoss for global shape
    quality, FocalLoss for pixel-level precision on ambiguous boundaries.

    This design follows the Dice + Focal combination used in mmsegmentation and
    segmentation_models_pytorch for binary segmentation with class imbalance.

    This class is a drop-in replacement for BalanceLoss when main_loss_type is
    'DiceLoss' — both share the same forward(pred, gt, mask) signature and
    return a scalar.

    Args:
        dice_weight  (float): Weight for the DiceLoss term. Default: 1.0.
        focal_weight (float): Weight for the MaskedFocalLoss term. Default: 1.0.
        focal_alpha  (float): Positive-class balancing factor for FocalLoss.
            Default: 0.75.
        focal_gamma  (float): Focusing exponent for FocalLoss. Default: 2.0.
        eps          (float): Numerical stability constant. Default: 1e-6.
    """

    def __init__(
        self,
        dice_weight=1.0,
        focal_weight=1.0,
        focal_alpha=0.75,
        focal_gamma=2.0,
        eps=1e-6,
    ):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(eps=eps)
        self.focal_loss = MaskedFocalLoss(alpha=focal_alpha, gamma=focal_gamma, eps=eps)

    def forward(self, pred, gt, mask=None):
        """
        Args:
            pred (Tensor): Predicted probability map, shape (B, H, W), in [0, 1].
            gt   (Tensor): Binary ground-truth shrink map, shape (B, H, W).
            mask (Tensor): Valid-pixel mask, shape (B, H, W).
        Returns:
            Tensor: Scalar combined loss.
        """
        loss_dice = self.dice_loss(pred, gt, mask)
        loss_focal = self.focal_loss(pred, gt, mask)
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal
