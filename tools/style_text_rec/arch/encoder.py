# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from arch.base_module import SNConv, SNConvTranspose, ResBlock


class Encoder(nn.Layer):
    def __init__(self, name, in_channels, encode_dim, use_bias, norm_layer,
                 act, act_attr, conv_block_dropout, conv_block_num,
                 conv_block_dilation):
        super(Encoder, self).__init__()
        self._pad2d = paddle.nn.Pad2D([3, 3, 3, 3], mode="replicate")
        self._in_conv = SNConv(
            name=name + "_in_conv",
            in_channels=in_channels,
            out_channels=encode_dim,
            kernel_size=7,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._down1 = SNConv(
            name=name + "_down1",
            in_channels=encode_dim,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._down2 = SNConv(
            name=name + "_down2",
            in_channels=encode_dim * 2,
            out_channels=encode_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._down3 = SNConv(
            name=name + "_down3",
            in_channels=encode_dim * 4,
            out_channels=encode_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        conv_blocks = []
        for i in range(conv_block_num):
            conv_blocks.append(
                ResBlock(
                    name="{}_conv_block_{}".format(name, i),
                    channels=encode_dim * 4,
                    norm_layer=norm_layer,
                    use_dropout=conv_block_dropout,
                    use_dilation=conv_block_dilation,
                    use_bias=use_bias))
        self._conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):
        out_dict = dict()
        x = self._pad2d(x)
        out_dict["in_conv"] = self._in_conv.forward(x)
        out_dict["down1"] = self._down1.forward(out_dict["in_conv"])
        out_dict["down2"] = self._down2.forward(out_dict["down1"])
        out_dict["down3"] = self._down3.forward(out_dict["down2"])
        out_dict["res_blocks"] = self._conv_blocks.forward(out_dict["down3"])
        return out_dict


class EncoderUnet(nn.Layer):
    def __init__(self, name, in_channels, encode_dim, use_bias, norm_layer,
                 act, act_attr):
        super(EncoderUnet, self).__init__()
        self._pad2d = paddle.nn.Pad2D([3, 3, 3, 3], mode="replicate")
        self._in_conv = SNConv(
            name=name + "_in_conv",
            in_channels=in_channels,
            out_channels=encode_dim,
            kernel_size=7,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._down1 = SNConv(
            name=name + "_down1",
            in_channels=encode_dim,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._down2 = SNConv(
            name=name + "_down2",
            in_channels=encode_dim * 2,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._down3 = SNConv(
            name=name + "_down3",
            in_channels=encode_dim * 2,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._down4 = SNConv(
            name=name + "_down4",
            in_channels=encode_dim * 2,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._up1 = SNConvTranspose(
            name=name + "_up1",
            in_channels=encode_dim * 2,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._up2 = SNConvTranspose(
            name=name + "_up2",
            in_channels=encode_dim * 4,
            out_channels=encode_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)

    def forward(self, x):
        output_dict = dict()
        x = self._pad2d(x)
        output_dict['in_conv'] = self._in_conv.forward(x)
        output_dict['down1'] = self._down1.forward(output_dict['in_conv'])
        output_dict['down2'] = self._down2.forward(output_dict['down1'])
        output_dict['down3'] = self._down3.forward(output_dict['down2'])
        output_dict['down4'] = self._down4.forward(output_dict['down3'])
        output_dict['up1'] = self._up1.forward(output_dict['down4'])
        output_dict['up2'] = self._up2.forward(
            paddle.concat(
                (output_dict['down3'], output_dict['up1']), axis=1))
        output_dict['concat'] = paddle.concat(
            (output_dict['down2'], output_dict['up2']), axis=1)
        return output_dict
