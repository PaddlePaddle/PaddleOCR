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

from arch.base_module import MiddleNet, ResBlock
from arch.encoder import Encoder
from arch.decoder import Decoder, DecoderUnet, SingleDecoder
from utils.load_params import load_dygraph_pretrain
from utils.logging import get_logger


class StyleTextRec(nn.Layer):
    def __init__(self, config):
        super(StyleTextRec, self).__init__()
        self.logger = get_logger()
        self.text_generator = TextGenerator(config["Predictor"][
            "text_generator"])
        self.bg_generator = BgGeneratorWithMask(config["Predictor"][
            "bg_generator"])
        self.fusion_generator = FusionGeneratorSimple(config["Predictor"][
            "fusion_generator"])
        bg_generator_pretrain = config["Predictor"]["bg_generator"]["pretrain"]
        text_generator_pretrain = config["Predictor"]["text_generator"][
            "pretrain"]
        fusion_generator_pretrain = config["Predictor"]["fusion_generator"][
            "pretrain"]
        load_dygraph_pretrain(
            self.bg_generator,
            self.logger,
            path=bg_generator_pretrain,
            load_static_weights=False)
        load_dygraph_pretrain(
            self.text_generator,
            self.logger,
            path=text_generator_pretrain,
            load_static_weights=False)
        load_dygraph_pretrain(
            self.fusion_generator,
            self.logger,
            path=fusion_generator_pretrain,
            load_static_weights=False)

    def forward(self, style_input, text_input):
        text_gen_output = self.text_generator.forward(style_input, text_input)
        fake_text = text_gen_output["fake_text"]
        fake_sk = text_gen_output["fake_sk"]
        bg_gen_output = self.bg_generator.forward(style_input)
        bg_encode_feature = bg_gen_output["bg_encode_feature"]
        bg_decode_feature1 = bg_gen_output["bg_decode_feature1"]
        bg_decode_feature2 = bg_gen_output["bg_decode_feature2"]
        fake_bg = bg_gen_output["fake_bg"]

        fusion_gen_output = self.fusion_generator.forward(fake_text, fake_bg)
        fake_fusion = fusion_gen_output["fake_fusion"]
        return {
            "fake_fusion": fake_fusion,
            "fake_text": fake_text,
            "fake_sk": fake_sk,
            "fake_bg": fake_bg,
        }


class TextGenerator(nn.Layer):
    def __init__(self, config):
        super(TextGenerator, self).__init__()
        name = config["module_name"]
        encode_dim = config["encode_dim"]
        norm_layer = config["norm_layer"]
        conv_block_dropout = config["conv_block_dropout"]
        conv_block_num = config["conv_block_num"]
        conv_block_dilation = config["conv_block_dilation"]
        if norm_layer == "InstanceNorm2D":
            use_bias = True
        else:
            use_bias = False
        self.encoder_text = Encoder(
            name=name + "_encoder_text",
            in_channels=3,
            encode_dim=encode_dim,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act="ReLU",
            act_attr=None,
            conv_block_dropout=conv_block_dropout,
            conv_block_num=conv_block_num,
            conv_block_dilation=conv_block_dilation)
        self.encoder_style = Encoder(
            name=name + "_encoder_style",
            in_channels=3,
            encode_dim=encode_dim,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act="ReLU",
            act_attr=None,
            conv_block_dropout=conv_block_dropout,
            conv_block_num=conv_block_num,
            conv_block_dilation=conv_block_dilation)
        self.decoder_text = Decoder(
            name=name + "_decoder_text",
            encode_dim=encode_dim,
            out_channels=int(encode_dim / 2),
            use_bias=use_bias,
            norm_layer=norm_layer,
            act="ReLU",
            act_attr=None,
            conv_block_dropout=conv_block_dropout,
            conv_block_num=conv_block_num,
            conv_block_dilation=conv_block_dilation,
            out_conv_act="Tanh",
            out_conv_act_attr=None)
        self.decoder_sk = Decoder(
            name=name + "_decoder_sk",
            encode_dim=encode_dim,
            out_channels=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act="ReLU",
            act_attr=None,
            conv_block_dropout=conv_block_dropout,
            conv_block_num=conv_block_num,
            conv_block_dilation=conv_block_dilation,
            out_conv_act="Sigmoid",
            out_conv_act_attr=None)

        self.middle = MiddleNet(
            name=name + "_middle_net",
            in_channels=int(encode_dim / 2) + 1,
            mid_channels=encode_dim,
            out_channels=3,
            use_bias=use_bias)

    def forward(self, style_input, text_input):
        style_feature = self.encoder_style.forward(style_input)["res_blocks"]
        text_feature = self.encoder_text.forward(text_input)["res_blocks"]
        fake_c_temp = self.decoder_text.forward([text_feature,
                                                 style_feature])["out_conv"]
        fake_sk = self.decoder_sk.forward([text_feature,
                                           style_feature])["out_conv"]
        fake_text = self.middle(paddle.concat((fake_c_temp, fake_sk), axis=1))
        return {"fake_sk": fake_sk, "fake_text": fake_text}


class BgGeneratorWithMask(nn.Layer):
    def __init__(self, config):
        super(BgGeneratorWithMask, self).__init__()
        name = config["module_name"]
        encode_dim = config["encode_dim"]
        norm_layer = config["norm_layer"]
        conv_block_dropout = config["conv_block_dropout"]
        conv_block_num = config["conv_block_num"]
        conv_block_dilation = config["conv_block_dilation"]
        self.output_factor = config.get("output_factor", 1.0)

        if norm_layer == "InstanceNorm2D":
            use_bias = True
        else:
            use_bias = False

        self.encoder_bg = Encoder(
            name=name + "_encoder_bg",
            in_channels=3,
            encode_dim=encode_dim,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act="ReLU",
            act_attr=None,
            conv_block_dropout=conv_block_dropout,
            conv_block_num=conv_block_num,
            conv_block_dilation=conv_block_dilation)

        self.decoder_bg = SingleDecoder(
            name=name + "_decoder_bg",
            encode_dim=encode_dim,
            out_channels=3,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act="ReLU",
            act_attr=None,
            conv_block_dropout=conv_block_dropout,
            conv_block_num=conv_block_num,
            conv_block_dilation=conv_block_dilation,
            out_conv_act="Tanh",
            out_conv_act_attr=None)

        self.decoder_mask = Decoder(
            name=name + "_decoder_mask",
            encode_dim=encode_dim // 2,
            out_channels=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            act="ReLU",
            act_attr=None,
            conv_block_dropout=conv_block_dropout,
            conv_block_num=conv_block_num,
            conv_block_dilation=conv_block_dilation,
            out_conv_act="Sigmoid",
            out_conv_act_attr=None)

        self.middle = MiddleNet(
            name=name + "_middle_net",
            in_channels=3 + 1,
            mid_channels=encode_dim,
            out_channels=3,
            use_bias=use_bias)

    def forward(self, style_input):
        encode_bg_output = self.encoder_bg(style_input)
        decode_bg_output = self.decoder_bg(encode_bg_output["res_blocks"],
                                           encode_bg_output["down2"],
                                           encode_bg_output["down1"])

        fake_c_temp = decode_bg_output["out_conv"]
        fake_bg_mask = self.decoder_mask.forward(encode_bg_output[
            "res_blocks"])["out_conv"]
        fake_bg = self.middle(
            paddle.concat(
                (fake_c_temp, fake_bg_mask), axis=1))
        return {
            "bg_encode_feature": encode_bg_output["res_blocks"],
            "bg_decode_feature1": decode_bg_output["up1"],
            "bg_decode_feature2": decode_bg_output["up2"],
            "fake_bg": fake_bg,
            "fake_bg_mask": fake_bg_mask,
        }


class FusionGeneratorSimple(nn.Layer):
    def __init__(self, config):
        super(FusionGeneratorSimple, self).__init__()
        name = config["module_name"]
        encode_dim = config["encode_dim"]
        norm_layer = config["norm_layer"]
        conv_block_dropout = config["conv_block_dropout"]
        conv_block_dilation = config["conv_block_dilation"]
        if norm_layer == "InstanceNorm2D":
            use_bias = True
        else:
            use_bias = False

        self._conv = nn.Conv2D(
            in_channels=6,
            out_channels=encode_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=paddle.ParamAttr(name=name + "_conv_weights"),
            bias_attr=False)

        self._res_block = ResBlock(
            name="{}_conv_block".format(name),
            channels=encode_dim,
            norm_layer=norm_layer,
            use_dropout=conv_block_dropout,
            use_dilation=conv_block_dilation,
            use_bias=use_bias)

        self._reduce_conv = nn.Conv2D(
            in_channels=encode_dim,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=paddle.ParamAttr(name=name + "_reduce_conv_weights"),
            bias_attr=False)

    def forward(self, fake_text, fake_bg):
        fake_concat = paddle.concat((fake_text, fake_bg), axis=1)
        fake_concat_tmp = self._conv(fake_concat)
        output_res = self._res_block(fake_concat_tmp)
        fake_fusion = self._reduce_conv(output_res)
        return {"fake_fusion": fake_fusion}
