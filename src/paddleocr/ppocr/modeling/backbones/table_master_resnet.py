# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/mmocr/models/textrecog/backbones/table_resnet_extra.py
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, gcb_config=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False
        )
        self.bn1 = nn.BatchNorm2D(planes, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn2 = nn.BatchNorm2D(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        self.gcb_config = gcb_config

        if self.gcb_config is not None:
            gcb_ratio = gcb_config["ratio"]
            gcb_headers = gcb_config["headers"]
            att_scale = gcb_config["att_scale"]
            fusion_type = gcb_config["fusion_type"]
            self.context_block = MultiAspectGCAttention(
                inplanes=planes,
                ratio=gcb_ratio,
                headers=gcb_headers,
                att_scale=att_scale,
                fusion_type=fusion_type,
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.gcb_config is not None:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_gcb_config(gcb_config, layer):
    if gcb_config is None or not gcb_config["layers"][layer]:
        return None
    else:
        return gcb_config


class TableResNetExtra(nn.Layer):
    def __init__(self, layers, in_channels=3, gcb_config=None):
        assert len(layers) >= 4

        super(TableResNetExtra, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv2D(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn1 = nn.BatchNorm2D(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(
            64, 128, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn2 = nn.BatchNorm2D(128)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(
            BasicBlock,
            256,
            layers[0],
            stride=1,
            gcb_config=get_gcb_config(gcb_config, 0),
        )

        self.conv3 = nn.Conv2D(
            256, 256, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn3 = nn.BatchNorm2D(256)
        self.relu3 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.layer2 = self._make_layer(
            BasicBlock,
            256,
            layers[1],
            stride=1,
            gcb_config=get_gcb_config(gcb_config, 1),
        )

        self.conv4 = nn.Conv2D(
            256, 256, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn4 = nn.BatchNorm2D(256)
        self.relu4 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.layer3 = self._make_layer(
            BasicBlock,
            512,
            layers[2],
            stride=1,
            gcb_config=get_gcb_config(gcb_config, 2),
        )

        self.conv5 = nn.Conv2D(
            512, 512, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn5 = nn.BatchNorm2D(512)
        self.relu5 = nn.ReLU()

        self.layer4 = self._make_layer(
            BasicBlock,
            512,
            layers[3],
            stride=1,
            gcb_config=get_gcb_config(gcb_config, 3),
        )

        self.conv6 = nn.Conv2D(
            512, 512, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn6 = nn.BatchNorm2D(512)
        self.relu6 = nn.ReLU()

        self.out_channels = [256, 256, 512]

    def _make_layer(self, block, planes, blocks, stride=1, gcb_config=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, gcb_config=gcb_config)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f = []
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        f.append(x)

        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        f.append(x)

        x = self.maxpool3(x)

        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.layer4(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        f.append(x)
        return f


class MultiAspectGCAttention(nn.Layer):
    def __init__(
        self,
        inplanes,
        ratio,
        headers,
        pooling_type="att",
        att_scale=False,
        fusion_type="channel_add",
    ):
        super(MultiAspectGCAttention, self).__init__()
        assert pooling_type in ["avg", "att"]

        assert fusion_type in ["channel_add", "channel_mul", "channel_concat"]
        assert (
            inplanes % headers == 0 and inplanes >= 8
        )  # inplanes must be divided by headers evenly

        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = False

        self.single_header_inplanes = int(inplanes / headers)

        if pooling_type == "att":
            self.conv_mask = nn.Conv2D(self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(axis=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2D(1)

        if fusion_type == "channel_add":
            self.channel_add_conv = nn.Sequential(
                nn.Conv2D(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),
                nn.Conv2D(self.planes, self.inplanes, kernel_size=1),
            )
        elif fusion_type == "channel_concat":
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2D(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),
                nn.Conv2D(self.planes, self.inplanes, kernel_size=1),
            )
            # for concat
            self.cat_conv = nn.Conv2D(2 * self.inplanes, self.inplanes, kernel_size=1)
        elif fusion_type == "channel_mul":
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2D(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),
                nn.Conv2D(self.planes, self.inplanes, kernel_size=1),
            )

    def spatial_pool(self, x):
        batch, channel, height, width = x.shape
        if self.pooling_type == "att":
            # [N*headers, C', H , W] C = headers * C'
            x = x.reshape(
                [batch * self.headers, self.single_header_inplanes, height, width]
            )
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            # input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.reshape(
                [batch * self.headers, self.single_header_inplanes, height * width]
            )

            # [N*headers, 1, C', H * W]
            input_x = input_x.unsqueeze(1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.reshape(
                [batch * self.headers, 1, height * width]
            )

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / paddle.sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = paddle.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            context = context.reshape(
                [batch, self.headers * self.single_header_inplanes, 1, 1]
            )
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x

        if self.fusion_type == "channel_mul":
            # [N, C, 1, 1]
            channel_mul_term = F.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == "channel_add":
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape

            out = paddle.concat(
                [out, channel_concat_term.expand([-1, -1, H, W])], axis=1
            )
            out = self.cat_conv(out)
            out = F.layer_norm(out, [self.inplanes, H, W])
            out = F.relu(out)

        return out
