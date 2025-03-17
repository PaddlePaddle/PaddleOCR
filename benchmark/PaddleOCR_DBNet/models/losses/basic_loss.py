# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 14:39
# @Author  : zhoujun
import paddle
import paddle.nn as nn


class BalanceCrossEntropyLoss(nn.Layer):
    """
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    """

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(
        self,
        pred: paddle.Tensor,
        gt: paddle.Tensor,
        mask: paddle.Tensor,
        return_origin=False,
    ):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())
        negative_count = min(
            int(negative.sum()), int(positive_count * self.negative_ratio)
        )
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction="none")
        positive_loss = loss * positive
        negative_loss = loss * negative
        negative_loss, _ = negative_loss.reshape([-1]).topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + self.eps
        )

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Layer):
    """
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity between tow heatmaps.
    """

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: paddle.Tensor, gt, mask, weights=None):
        """
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        """
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if len(pred.shape) == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Layer):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: paddle.Tensor, gt, mask):
        loss = (paddle.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss
