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
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/DB_loss.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np


# 在线难样本挖掘
def ohem_single(score, gt_text, training_mask):
    # 预测kernel， 真实kernel， 外边界mask

    # 真实kernel的像素 - (真实kernel且是外边界) = 真实kernel-真实kernel与外边界重复的部分 ～= 真实kernel 基本上没有重复？
    pos_num = int(paddle.sum(gt_text > 0.5)) - int(
        paddle.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    #print(torch.sum(gt_text), pos_num)

    if pos_num == 0:  #没有kernel？这个情况会被触发吗？
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = paddle.cast(
            selected_mask.reshape(
                (1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
        return selected_mask

    # 不是真实kernel且不是外边界 = 文字区域之外的像素？
    neg_num = int(paddle.sum((gt_text <= 0.5) & (training_mask > 0.5)))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = paddle.cast(
            selected_mask.reshape(
                (1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
        return selected_mask

    # 取难训练的neg
    # 取预测概率高的负样本，容易识别成正样本的负样本，作为hard example
    neg_score = score[(gt_text <= 0.5) & (training_mask > 0.5)]
    neg_score_sorted = paddle.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    # 外边界之外的像素，选择条件：kernel区域，或预测分数大于阈值的文字框之外的区域
    selected_mask = ((score >= threshold) |
                     (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = paddle.cast(
        selected_mask.reshape(
            (1, selected_mask.shape[0], selected_mask.shape[1])), "float32")
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    # 4, 640, 640
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(
            ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[
                i, :, :]))

    selected_masks = paddle.cast(paddle.concat(selected_masks, 0), "float32")
    return selected_masks


EPS = 1e-6


def iou_single(a, b, mask, n_class):
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []
    # 每一类算iou，然后求平均
    for i in range(n_class):
        inter = paddle.cast(((a == i) & (b == i)), "float32")
        union = paddle.cast(((a == i) | (b == i)), "float32")

        miou.append(paddle.sum(inter) / (paddle.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = a.reshape((batch_size, -1))
    b = b.reshape((batch_size, -1))
    mask = mask.reshape((batch_size, -1))

    iou = paddle.zeros((batch_size, ), dtype="float32")
    #iou = a.new_zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = paddle.mean(iou)
    return iou


class DiceLoss(nn.Layer):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.shape[0]
        input = F.sigmoid(input)  # 这里会转成0-1之间

        input = input.reshape((batch_size, -1))
        target = paddle.cast(target.reshape((batch_size, -1)), "float32")
        mask = paddle.cast(mask.reshape((batch_size, -1)), "float32")

        input = input * mask
        target = target * mask

        a = paddle.sum(input * target, axis=1)
        b = paddle.sum(input * input, axis=1) + 0.001
        c = paddle.sum(target * target, axis=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = paddle.mean(loss)

        return loss


class SmoothL1Loss(nn.Layer):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

        # NOTE: 这一块反向可能会有问题
        # 原始
        #self.coord = nn.Parameter(torch.zeros([640, 640, 2]).long(), requires_grad=False)
        # for i in range(640):
        #     for j in range(640):
        #         self.coord[i, j, 0] = j
        #         self.coord[i, j, 1] = i
        # self.coord.data = self.coord.reshape((-1, 2)) # (h*w, 2)

        # =========写法一============
        np_coord = np.zeros(shape=[640, 640, 2], dtype=np.int64)
        for i in range(640):
            for j in range(640):
                np_coord[i, j, 0] = j
                np_coord[i, j, 1] = i
        np_coord = np_coord.reshape((-1, 2))

        self.coord = self.create_parameter(
            shape=[640 * 640, 2],
            dtype="int32",  # NOTE: "int64报错" 2.3.1也有这个bug
            default_initializer=nn.initializer.Assign(value=np_coord))
        self.coord.stop_gradient = True
        #------------------

        #==========写法二=======================
        # self.coord = self.create_parameter(shape=[640*640, 2], 
        #                             dtype="int64", # NOTE: "int64报错"
        #                             default_initializer=nn.initializer.Constant(value=0))
        # self.coord.stop_gradient = True       
        # np_coord = np.zeros(shape=[640, 640, 2], dtype=np.int64)
        # for i in range(640):
        #     for j in range(640):
        #         np_coord[i, j, 0] = j
        #         np_coord[i, j, 1] = i
        # np_coord = np_coord.reshape((-1, 2))
        # tensor_coord = paddle.to_tensor(np_coord)
        # paddle.nn.utils.vector_to_parameters(tensor_coord, self.coord)

    def forward_single(self, input, target, mask, beta=1.0, eps=1e-6):
        batch_size = input.shape[0]

        diff = paddle.abs(input - target) * mask.unsqueeze(1)
        loss = paddle.where(diff < beta, 0.5 * diff * diff / beta,
                            diff - 0.5 * beta)
        loss = paddle.cast(loss.reshape((batch_size, -1)), "float32")
        mask = paddle.cast(mask.reshape((batch_size, -1)), "float32")
        loss = paddle.sum(loss, axis=-1)
        loss = loss / (mask.sum(axis=-1) + eps)

        return loss

    def select_single(self, distance, gt_instance, gt_kernel_instance,
                      training_mask):

        with paddle.no_grad():
            # 写法3=========
            # 由于paddle slice功能缺失，替代 distance[:, self.coord[:, 1], self.coord[:, 0]]
            select_distance_list = []
            for i in range(2):
                tmp1 = distance[i, :]
                tmp2 = tmp1[self.coord[:, 1], self.coord[:, 0]]
                select_distance_list.append(tmp2.unsqueeze(0))
            select_distance = paddle.concat(select_distance_list, axis=0)
            #print(select_distance.shape)

            off_points = paddle.cast(
                self.coord, "float32") + 10 * select_distance.transpose((1, 0))

            off_points = paddle.cast(off_points, "int64")
            off_points = paddle.clip(
                off_points, 0, distance.shape[-1] - 1)  # 预测值截断， 0,640，不能超出图像大小
            # gt_instance和预测的gt_instance不相等的时候才要计算loss

            selected_mask = (
                gt_instance[self.coord[:, 1], self.coord[:, 0]] !=
                gt_kernel_instance[off_points[:, 1], off_points[:, 0]])
            selected_mask = paddle.cast(
                selected_mask.reshape((1, -1, distance.shape[-1])), "int64")
            selected_training_mask = selected_mask * training_mask

            return selected_training_mask

    def forward(self,
                distances,
                gt_instances,
                gt_kernel_instances,
                training_masks,
                gt_distances,
                reduce=True):

        selected_training_masks = []
        for i in range(distances.shape[0]):
            selected_training_masks.append(
                self.select_single(distances[i, :, :, :], gt_instances[i, :, :],
                                   gt_kernel_instances[i, :, :], training_masks[
                                       i, :, :]))
        selected_training_masks = paddle.cast(
            paddle.concat(selected_training_masks, 0), "float32")

        loss = self.forward_single(distances, gt_distances,
                                   selected_training_masks, self.beta)
        loss = self.loss_weight * loss

        with paddle.no_grad():
            batch_size = distances.shape[0]
            false_num = selected_training_masks.reshape((batch_size, -1))
            false_num = false_num.sum(axis=-1)
            total_num = paddle.cast(
                training_masks.reshape((batch_size, -1)), "float32")
            total_num = total_num.sum(axis=-1)
            iou_text = (total_num - false_num) / (total_num + 1e-6)

        if reduce:
            loss = paddle.mean(loss)

        return loss, iou_text


class CTLoss(nn.Layer):
    def __init__(self):
        super(CTLoss, self).__init__()
        self.kernel_loss = DiceLoss()
        self.loc_loss = SmoothL1Loss(beta=0.1, loss_weight=0.05)

    # def _upsample(self, x, size, scale=1):
    #     _, _, H, W = size
    #     return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    # def _upsample(self, x, scale=1):
    #     return F.upsample(x, scale_factor=scale, mode='bilinear')

    def forward(self, preds, batch):
        imgs = batch[0]
        #print(preds)
        out = preds['maps']
        gt_kernels, training_masks, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances = batch[
            1:]
        # print('==1==', out.shape)
        # out = self._upsample(out, imgs.shape)
        # out = self._upsample(out, scale=4)
        # print('==2==', out.shape)
        # exit()
        # output
        # out: 4,3,640,640
        # gt_kernels = torch.rand(4, 640, 640).long(), 单词区域收缩得到的kernel区域，不区分实例。背景0，单词区域1
        # training_masks = torch.rand(4, 640, 640).long(), 外边界mask。外边界0，其余1。mask掉外边界(可能因为标注导致不准确的位置)
        # gt_instances = torch.rand(4, 640, 640).long(), 单词区域，区分实例。背景0，单词区域1,2,..
        # gt_kernel_instances = torch.rand(4, 640, 640).long(), 单词区域收缩得到的kernel区域，区分实例。背景0，kernel区域1,2,..
        # training_mask_distances = torch.rand(4, 640, 640).long(), 特殊单词或未标注的单词。特殊单词或未标注单词填充为0，其余为1
        # gt_distances = torch.rand(4, 2, 640, 640).float()) 外边界mask和内边界mask的差？

        kernels = out[:, 0, :, :]  # 预测kernel图 （没过softmax，不是概率）
        distances = out[:, 1:, :, :]

        # kernel loss
        selected_masks = ohem_batch(kernels, gt_kernels, training_masks)
        # # kernel和背景中select的负样本算loss，外边界忽略了不算loss

        loss_kernel = self.kernel_loss(
            kernels, gt_kernels, selected_masks, reduce=False)
        # # 这里为什么取大于0？ 就是认为：大于0是kernel，小于0是背景
        iou_kernel = iou(paddle.cast((kernels > 0), "int64"),
                         gt_kernels,
                         training_masks,
                         reduce=False)
        losses = dict(
            loss_kernels=loss_kernel,
            #            iou_kernel=iou_kernel
        )

        # # loc loss
        #loc_loss = SmoothL1Loss(beta=0.1, loss_weight=0.05)
        loss_loc, iou_text = self.loc_loss(
            distances,
            gt_instances,
            gt_kernel_instances,
            training_mask_distances,
            gt_distances,
            reduce=False)
        losses.update(dict(
            loss_loc=loss_loc,
            # iou_text=iou_text
        ))
        loss_all = loss_kernel + loss_loc
        #print(loss_kernel, iou_kernel, loss_loc, iou_text)
        losses = {'loss': loss_all}
        return losses
