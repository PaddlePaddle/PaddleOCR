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


class Decoder(nn.Layer):
    def __init__(self, name, encode_dim, out_channels, use_bias, norm_layer,
                 act, act_attr, conv_block_dropout, conv_block_num,
                 conv_block_dilation, out_conv_act, out_conv_act_attr):
        super(Decoder, self).__init__()
        conv_blocks = []
        for i in range(conv_block_num):
            conv_blocks.append(
                ResBlock(
                    name="{}_conv_block_{}".format(name, i),
                    channels=encode_dim * 8,
                    norm_layer=norm_layer,
                    use_dropout=conv_block_dropout,
                    use_dilation=conv_block_dilation,
                    use_bias=use_bias))
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self._up1 = SNConvTranspose(
            name=name + "_up1",
            in_channels=encode_dim * 8,
            out_channels=encode_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._up2 = SNConvTranspose(
            name=name + "_up2",
            in_channels=encode_dim * 4,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._up3 = SNConvTranspose(
            name=name + "_up3",
            in_channels=encode_dim * 2,
            out_channels=encode_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._pad2d = paddle.nn.Pad2D([1, 1, 1, 1], mode="replicate")
        self._out_conv = SNConv(
            name=name + "_out_conv",
            in_channels=encode_dim,
            out_channels=out_channels,
            kernel_size=3,
            use_bias=use_bias,
            norm_layer=None,
            act=out_conv_act,
            act_attr=out_conv_act_attr)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = paddle.concat(x, axis=1)
        output_dict = dict()
        output_dict["conv_blocks"] = self.conv_blocks.forward(x)
        output_dict["up1"] = self._up1.forward(output_dict["conv_blocks"])
        output_dict["up2"] = self._up2.forward(output_dict["up1"])
        output_dict["up3"] = self._up3.forward(output_dict["up2"])
        output_dict["pad2d"] = self._pad2d.forward(output_dict["up3"])
        output_dict["out_conv"] = self._out_conv.forward(output_dict["pad2d"])
        return output_dict


class DecoderUnet(nn.Layer):
    def __init__(self, name, encode_dim, out_channels, use_bias, norm_layer,
                 act, act_attr, conv_block_dropout, conv_block_num,
                 conv_block_dilation, out_conv_act, out_conv_act_attr):
        super(DecoderUnet, self).__init__()
        conv_blocks = []
        for i in range(conv_block_num):
            conv_blocks.append(
                ResBlock(
                    name="{}_conv_block_{}".format(name, i),
                    channels=encode_dim * 8,
                    norm_layer=norm_layer,
                    use_dropout=conv_block_dropout,
                    use_dilation=conv_block_dilation,
                    use_bias=use_bias))
        self._conv_blocks = nn.Sequential(*conv_blocks)
        self._up1 = SNConvTranspose(
            name=name + "_up1",
            in_channels=encode_dim * 8,
            out_channels=encode_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._up2 = SNConvTranspose(
            name=name + "_up2",
            in_channels=encode_dim * 8,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._up3 = SNConvTranspose(
            name=name + "_up3",
            in_channels=encode_dim * 4,
            out_channels=encode_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._pad2d = paddle.nn.Pad2D([1, 1, 1, 1], mode="replicate")
        self._out_conv = SNConv(
            name=name + "_out_conv",
            in_channels=encode_dim,
            out_channels=out_channels,
            kernel_size=3,
            use_bias=use_bias,
            norm_layer=None,
            act=out_conv_act,
            act_attr=out_conv_act_attr)

    def forward(self, x, y, feature2, feature1):
        output_dict = dict()
        output_dict["conv_blocks"] = self._conv_blocks(
            paddle.concat(
                (x, y), axis=1))
        output_dict["up1"] = self._up1.forward(output_dict["conv_blocks"])
        output_dict["up2"] = self._up2.forward(
            paddle.concat(
                (output_dict["up1"], feature2), axis=1))
        output_dict["up3"] = self._up3.forward(
            paddle.concat(
                (output_dict["up2"], feature1), axis=1))
        output_dict["pad2d"] = self._pad2d.forward(output_dict["up3"])
        output_dict["out_conv"] = self._out_conv.forward(output_dict["pad2d"])
        return output_dict


class SingleDecoder(nn.Layer):
    def __init__(self, name, encode_dim, out_channels, use_bias, norm_layer,
                 act, act_attr, conv_block_dropout, conv_block_num,
                 conv_block_dilation, out_conv_act, out_conv_act_attr):
        super(SingleDecoder, self).__init__()
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
        self._up1 = SNConvTranspose(
            name=name + "_up1",
            in_channels=encode_dim * 4,
            out_channels=encode_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._up2 = SNConvTranspose(
            name=name + "_up2",
            in_channels=encode_dim * 8,
            out_channels=encode_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._up3 = SNConvTranspose(
            name=name + "_up3",
            in_channels=encode_dim * 4,
            out_channels=encode_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act=act,
            act_attr=act_attr)
        self._pad2d = paddle.nn.Pad2D([1, 1, 1, 1], mode="replicate")
        self._out_conv = SNConv(
            name=name + "_out_conv",
            in_channels=encode_dim,
            out_channels=out_channels,
            kernel_size=3,
            use_bias=use_bias,
            norm_layer=None,
            act=out_conv_act,
            act_attr=out_conv_act_attr)

    def forward(self, x, feature2, feature1):
        output_dict = dict()
        output_dict["conv_blocks"] = self._conv_blocks.forward(x)
        output_dict["up1"] = self._up1.forward(output_dict["conv_blocks"])
        output_dict["up2"] = self._up2.forward(
            paddle.concat(
                (output_dict["up1"], feature2), axis=1))
        output_dict["up3"] = self._up3.forward(
            paddle.concat(
                (output_dict["up2"], feature1), axis=1))
        output_dict["pad2d"] = self._pad2d.forward(output_dict["up3"])
        output_dict["out_conv"] = self._out_conv.forward(output_dict["pad2d"])
        return output_dict
