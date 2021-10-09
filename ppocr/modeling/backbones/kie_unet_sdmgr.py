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

__all__ = ["Kie_backbone"]


class Encoder(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2D(
            num_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(num_filters, act='relu')

        self.conv2 = nn.Conv2D(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(num_filters, act='relu')

        self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_pooled = self.pool(x)

        return x, x_pooled


class Decoder(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(Decoder, self).__init__()
        self.up = nn.Conv2DTranspose(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=2,
            stride=2)
        self.conv1 = nn.Conv2D(
            num_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(num_filters, act='relu')

        self.conv2 = nn.Conv2D(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(num_filters, act='relu')

    def forward(self, inputs_prev, inputs):
        x = self.up(inputs)
        x = paddle.concat([inputs_prev, x], axis=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class UNet(nn.Layer):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = Encoder(num_channels=3, num_filters=16)
        self.down2 = Encoder(num_channels=16, num_filters=32)
        self.down3 = Encoder(num_channels=32, num_filters=64)
        self.down4 = Encoder(num_channels=64, num_filters=128)
        self.down5 = Encoder(num_channels=128, num_filters=256)

        self.up4 = Decoder(256, 128)
        self.up3 = Decoder(128, 64)
        self.up2 = Decoder(64, 32)
        self.up1 = Decoder(32, 16)
        self.out_channels = 16

    def forward(self, inputs):
        x1, x = self.down1(inputs)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x5, x = self.down5(x)

        x = self.up4(x4, x5)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        return x


class Kie_backbone(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super(Kie_backbone, self).__init__()
        self.out_channels = 16
        self.img_feat = UNet()
        self.maxpool = nn.MaxPool2D(kernel_size=7)

    def bbox2roi(self, bbox_list):
        rois_list = []
        rois_num = []
        for img_id, bboxes in enumerate(bbox_list):
            rois_num.append(bboxes.shape[0])
            rois_list.append(bboxes)
        rois = paddle.concat(rois_list, 0)
        rois_num = paddle.to_tensor(rois_num, dtype='int32')
        return rois, rois_num

    def pre_process(self, relations, texts, gt_bboxes, tag):
        relations, texts, gt_bboxes, tag = relations.numpy(), texts.numpy(
        ), gt_bboxes.numpy(), tag.numpy().tolist()
        temp_relations, temp_texts, temp_gt_bboxes = [], [], []
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            temp_relations.append(
                paddle.to_tensor(
                    relations[i, :num, :num, :], dtype='float32'))
            temp_texts.append(
                paddle.to_tensor(
                    texts[i, :num, :recoder_len], dtype='float32'))
            temp_gt_bboxes.append(
                paddle.to_tensor(
                    gt_bboxes[i, :num, ...], dtype='float32'))
        return temp_relations, temp_texts, temp_gt_bboxes

    def forward(self, inputs):
        img, relations, texts, gt_bboxes, tag = inputs[0], inputs[1], inputs[
            2], inputs[3], inputs[5]
        relations, texts, gt_bboxes = self.pre_process(relations, texts,
                                                       gt_bboxes, tag)
        x = self.img_feat(img)
        boxes, rois_num = self.bbox2roi(gt_bboxes)
        feats = paddle.fluid.layers.roi_align(
            x,
            boxes,
            spatial_scale=1.0,
            pooled_height=7,
            pooled_width=7,
            rois_num=rois_num)
        feats = self.maxpool(feats).squeeze(-1).squeeze(-1)
        return [relations, texts, feats]
