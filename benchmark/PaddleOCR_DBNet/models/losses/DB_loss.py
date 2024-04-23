import paddle
from models.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss


class DBLoss(paddle.nn.Layer):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction="mean", eps=1e-06):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ["mean", "sum"], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]
        loss_shrink_maps = self.bce_loss(
            shrink_maps, batch["shrink_map"], batch["shrink_mask"]
        )
        loss_threshold_maps = self.l1_loss(
            threshold_maps, batch["threshold_map"], batch["threshold_mask"]
        )
        metrics = dict(
            loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps
        )
        if pred.shape[1] > 2:
            loss_binary_maps = self.dice_loss(
                binary_maps, batch["shrink_map"], batch["shrink_mask"]
            )
            metrics["loss_binary_maps"] = loss_binary_maps
            loss_all = (
                self.alpha * loss_shrink_maps
                + self.beta * loss_threshold_maps
                + loss_binary_maps
            )
            metrics["loss"] = loss_all
        else:
            metrics["loss"] = loss_shrink_maps
        return metrics
