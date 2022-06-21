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
from paddle.nn import functional as F

class TableAttentionLoss(nn.Layer):
    def __init__(self, structure_weight, loc_weight, use_giou=False, giou_weight=1.0, **kwargs):
        super(TableAttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight
        self.use_giou = use_giou
        self.giou_weight = giou_weight
        
    def giou_loss(self, preds, bbox, eps=1e-7, reduction='mean'):
        '''
        :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :return: loss
        '''
        ix1 = paddle.maximum(preds[:, 0], bbox[:, 0])
        iy1 = paddle.maximum(preds[:, 1], bbox[:, 1])
        ix2 = paddle.minimum(preds[:, 2], bbox[:, 2])
        iy2 = paddle.minimum(preds[:, 3], bbox[:, 3])

        iw = paddle.clip(ix2 - ix1 + 1e-3, 0., 1e10)
        ih = paddle.clip(iy2 - iy1 + 1e-3, 0., 1e10)

        # overlap
        inters = iw * ih

        # union
        uni = (preds[:, 2] - preds[:, 0] + 1e-3) * (preds[:, 3] - preds[:, 1] + 1e-3
            ) + (bbox[:, 2] - bbox[:, 0] + 1e-3) * (
            bbox[:, 3] - bbox[:, 1] + 1e-3) - inters + eps

        # ious
        ious = inters / uni

        ex1 = paddle.minimum(preds[:, 0], bbox[:, 0])
        ey1 = paddle.minimum(preds[:, 1], bbox[:, 1])
        ex2 = paddle.maximum(preds[:, 2], bbox[:, 2])
        ey2 = paddle.maximum(preds[:, 3], bbox[:, 3])
        ew = paddle.clip(ex2 - ex1 + 1e-3, 0., 1e10)
        eh = paddle.clip(ey2 - ey1 + 1e-3, 0., 1e10)

        # enclose erea
        enclose = ew * eh + eps
        giou = ious - (enclose - uni) / enclose

        loss = 1 - giou

        if reduction == 'mean':
            loss = paddle.mean(loss)
        elif reduction == 'sum':
            loss = paddle.sum(loss)
        else:
            raise NotImplementedError
        return loss

    def forward(self, predicts, batch):
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1].astype("int64")
        structure_targets = structure_targets[:, 1:]
        if len(batch) == 6:
            structure_mask = batch[5].astype("int64")
            structure_mask = structure_mask[:, 1:]
            structure_mask = paddle.reshape(structure_mask, [-1])
        structure_probs = paddle.reshape(structure_probs, [-1, structure_probs.shape[-1]])
        structure_targets = paddle.reshape(structure_targets, [-1])
        structure_loss = self.loss_func(structure_probs, structure_targets)
        
        if len(batch) == 6:
             structure_loss = structure_loss * structure_mask
            
#         structure_loss = paddle.sum(structure_loss) * self.structure_weight
        structure_loss = paddle.mean(structure_loss) * self.structure_weight
        
        loc_preds = predicts['loc_preds']
        loc_targets = batch[2].astype("float32")
        loc_targets_mask = batch[4].astype("float32")
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]
        loc_loss = F.mse_loss(loc_preds * loc_targets_mask, loc_targets) * self.loc_weight
        if self.use_giou:
            loc_loss_giou = self.giou_loss(loc_preds * loc_targets_mask, loc_targets) * self.giou_weight
            total_loss = structure_loss + loc_loss + loc_loss_giou
            return {'loss':total_loss, "structure_loss":structure_loss, "loc_loss":loc_loss, "loc_loss_giou":loc_loss_giou}
        else:
            total_loss = structure_loss + loc_loss            
            return {'loss':total_loss, "structure_loss":structure_loss, "loc_loss":loc_loss}