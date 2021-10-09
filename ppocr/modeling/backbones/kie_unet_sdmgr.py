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
import numpy as np
import cv2

__all__ = ["Kie_backbone"]


class Encoder(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2D(
            num_channels,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn1 = nn.BatchNorm(num_filters, act='relu')

        self.conv2 = nn.Conv2D(
            num_filters,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
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

        self.conv1 = nn.Conv2D(
            num_channels,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn1 = nn.BatchNorm(num_filters, act='relu')

        self.conv2 = nn.Conv2D(
            num_filters,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn2 = nn.BatchNorm(num_filters, act='relu')

        self.conv0 = nn.Conv2D(
            num_channels,
            num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)
        self.bn0 = nn.BatchNorm(num_filters, act='relu')

    def forward(self, inputs_prev, inputs):
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = paddle.nn.functional.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
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

        self.up1 = Decoder(32, 16)
        self.up2 = Decoder(64, 32)
        self.up3 = Decoder(128, 64)
        self.up4 = Decoder(256, 128)
        self.out_channels = 16

    def forward(self, inputs):
        x1, _ = self.down1(inputs)
        _, x2 = self.down2(x1)
        _, x3 = self.down3(x2)
        _, x4 = self.down4(x3)
        _, x5 = self.down5(x4)

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

    def pre_process(self, img, relations, texts, gt_bboxes, tag, img_size):
        img, relations, texts, gt_bboxes, tag, img_size = img.numpy(
        ), relations.numpy(), texts.numpy(), gt_bboxes.numpy(), tag.numpy(
        ).tolist(), img_size.numpy()
        temp_relations, temp_texts, temp_gt_bboxes = [], [], []
        h, w = int(np.max(img_size[:, 0])), int(np.max(img_size[:, 1]))
        img = paddle.to_tensor(img[:, :, :h, :w])
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
        return img, temp_relations, temp_texts, temp_gt_bboxes

    def forward(self, images, inputs):
        img = images
        relations, texts, gt_bboxes, tag, img_size = inputs[0], inputs[
            1], inputs[2], inputs[4], inputs[-1]
        img, relations, texts, gt_bboxes = self.pre_process(
            img, relations, texts, gt_bboxes, tag, img_size)
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
