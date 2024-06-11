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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn import L1Loss
from paddle.nn import MSELoss as L2Loss
from paddle.nn import SmoothL1Loss


class CELoss(nn.Layer):
    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        loss_dict = {}
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                label = F.softmax(label, axis=-1)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(x, label=label, soft_label=soft_label)
        return loss


class KLJSLoss(object):
    def __init__(self, mode="kl"):
        assert mode in [
            "kl",
            "js",
            "KL",
            "JS",
        ], "mode can only be one of ['kl', 'KL', 'js', 'JS']"
        self.mode = mode

    def __call__(self, p1, p2, reduction="mean", eps=1e-5):
        if self.mode.lower() == "kl":
            loss = paddle.multiply(p2, paddle.log((p2 + eps) / (p1 + eps) + eps))
            loss += paddle.multiply(p1, paddle.log((p1 + eps) / (p2 + eps) + eps))
            loss *= 0.5
        elif self.mode.lower() == "js":
            loss = paddle.multiply(
                p2, paddle.log((2 * p2 + eps) / (p1 + p2 + eps) + eps)
            )
            loss += paddle.multiply(
                p1, paddle.log((2 * p1 + eps) / (p1 + p2 + eps) + eps)
            )
            loss *= 0.5
        else:
            raise ValueError(
                "The mode.lower() if KLJSLoss should be one of ['kl', 'js']"
            )

        if reduction == "mean":
            loss = paddle.mean(loss, axis=[1, 2])
        elif reduction == "none" or reduction is None:
            return loss
        else:
            loss = paddle.sum(loss, axis=[1, 2])

        return loss


class DMLLoss(nn.Layer):
    """
    DMLLoss
    """

    def __init__(self, act=None, use_log=False):
        super().__init__()
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(axis=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.use_log = use_log
        self.jskl_loss = KLJSLoss(mode="kl")

    def _kldiv(self, x, target):
        eps = 1.0e-10
        loss = target * (paddle.log(target + eps) - x)
        # batch mean loss
        loss = paddle.sum(loss) / loss.shape[0]
        return loss

    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1) + 1e-10
            out2 = self.act(out2) + 1e-10
        if self.use_log:
            # for recognition distillation, log is needed for feature map
            log_out1 = paddle.log(out1)
            log_out2 = paddle.log(out2)
            loss = (self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0
        else:
            # for detection distillation log is not needed
            loss = self.jskl_loss(out1, out2)
        return loss


class DistanceLoss(nn.Layer):
    """
    DistanceLoss:
        mode: loss mode
    """

    def __init__(self, mode="l2", **kargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == "l2":
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(**kargs)

    def forward(self, x, y):
        return self.loss_func(x, y)


class LossFromOutput(nn.Layer):
    def __init__(self, key="loss", reduction="none"):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def forward(self, predicts, batch):
        loss = predicts
        if self.key is not None and isinstance(predicts, dict):
            loss = loss[self.key]
        if self.reduction == "mean":
            loss = paddle.mean(loss)
        elif self.reduction == "sum":
            loss = paddle.sum(loss)
        return {"loss": loss}


class KLDivLoss(nn.Layer):
    """
    KLDivLoss
    """

    def __init__(self):
        super().__init__()

    def _kldiv(self, x, target, mask=None):
        eps = 1.0e-10
        loss = target * (paddle.log(target + eps) - x)
        if mask is not None:
            loss = loss.flatten(0, 1).sum(axis=1)
            loss = loss.masked_select(mask).mean()
        else:
            # batch mean loss
            loss = paddle.sum(loss) / loss.shape[0]
        return loss

    def forward(self, logits_s, logits_t, mask=None):
        log_out_s = F.log_softmax(logits_s, axis=-1)
        out_t = F.softmax(logits_t, axis=-1)
        loss = self._kldiv(log_out_s, out_t, mask)
        return loss


class DKDLoss(nn.Layer):
    """
    KLDivLoss
    """

    def __init__(self, temperature=1.0, alpha=1.0, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(axis=1, keepdim=True)
        t2 = (t * mask2).sum(axis=1, keepdim=True)
        rt = paddle.concat([t1, t2], axis=1)
        return rt

    def _kl_div(self, x, label, mask=None):
        y = (label * (paddle.log(label + 1e-10) - x)).sum(axis=1)
        if mask is not None:
            y = y.masked_select(mask).mean()
        else:
            y = y.mean()
        return y

    def forward(self, logits_student, logits_teacher, target, mask=None):
        gt_mask = F.one_hot(target.reshape([-1]), num_classes=logits_student.shape[-1])
        other_mask = 1 - gt_mask
        logits_student = logits_student.flatten(0, 1)
        logits_teacher = logits_teacher.flatten(0, 1)
        pred_student = F.softmax(logits_student / self.temperature, axis=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, axis=1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = paddle.log(pred_student)
        tckd_loss = self._kl_div(log_pred_student, pred_teacher) * (self.temperature**2)
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, axis=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, axis=1
        )
        nckd_loss = self._kl_div(log_pred_student_part2, pred_teacher_part2) * (
            self.temperature**2
        )

        loss = self.alpha * tckd_loss + self.beta * nckd_loss

        return loss
