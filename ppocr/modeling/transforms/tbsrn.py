# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
https://github.com/FudanVI/FudanOCR/blob/main/scene-text-telescope/model/tbsrn.py
"""

import math
import warnings
import numpy as np
import paddle
from paddle import nn
import string

warnings.filterwarnings("ignore")

from .tps_spatial_transformer import TPSSpatialTransformer
from .stn import STN as STNHead
from .tsrn import GruBlock, mish, UpsampleBLock
from ppocr.modeling.heads.sr_rensnet_transformer import (
    Transformer,
    LayerNorm,
    PositionwiseFeedForward,
    MultiHeadedAttention,
)


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = paddle.zeros([d_model, height, width])
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = paddle.exp(
        paddle.arange(0.0, d_model, 2, dtype="int64") * -(math.log(10000.0) / d_model)
    )
    pos_w = paddle.arange(0.0, width, dtype="float32").unsqueeze(1)
    pos_h = paddle.arange(0.0, height, dtype="float32").unsqueeze(1)

    pe[0:d_model:2, :, :] = (
        paddle.sin(pos_w * div_term).transpose([1, 0]).unsqueeze(1).tile([1, height, 1])
    )
    pe[1:d_model:2, :, :] = (
        paddle.cos(pos_w * div_term).transpose([1, 0]).unsqueeze(1).tile([1, height, 1])
    )
    pe[d_model::2, :, :] = (
        paddle.sin(pos_h * div_term).transpose([1, 0]).unsqueeze(2).tile([1, 1, width])
    )
    pe[d_model + 1 :: 2, :, :] = (
        paddle.cos(pos_h * div_term).transpose([1, 0]).unsqueeze(2).tile([1, 1, width])
    )

    return pe


class FeatureEnhancer(nn.Layer):
    def __init__(self):
        super(FeatureEnhancer, self).__init__()

        self.multihead = MultiHeadedAttention(h=4, d_model=128, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=128)

        self.pff = PositionwiseFeedForward(128, 128)
        self.mul_layernorm3 = LayerNorm(features=128)

        self.linear = nn.Linear(128, 64)

    def forward(self, conv_feature):
        """
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        """
        batch = conv_feature.shape[0]
        position2d = (
            positionalencoding2d(64, 16, 64)
            .cast("float32")
            .unsqueeze(0)
            .reshape([1, 64, 1024])
        )
        position2d = position2d.tile([batch, 1, 1])
        conv_feature = paddle.concat(
            [conv_feature, position2d], 1
        )  # batch, 128(64+64), 32, 128
        result = conv_feature.transpose([0, 2, 1])
        origin_result = result
        result = self.mul_layernorm1(
            origin_result + self.multihead(result, result, result, mask=None)[0]
        )
        origin_result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))
        result = self.linear(result)
        return result.transpose([0, 2, 1])


def str_filt(str_, voc_type):
    alpha_dict = {
        "digit": string.digits,
        "lower": string.digits + string.ascii_lowercase,
        "upper": string.digits + string.ascii_letters,
        "all": string.digits + string.ascii_letters + string.punctuation,
    }
    if voc_type == "lower":
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, "")
    str_ = str_.lower()
    return str_


class TBSRN(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        scale_factor=2,
        width=128,
        height=32,
        STN=True,
        srb_nums=5,
        mask=False,
        hidden_units=32,
        infer_mode=False,
    ):
        super(TBSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2D(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU(),
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, "block%d" % (i + 2), RecurrentResidualBlock(2 * hidden_units))

        setattr(
            self,
            "block%d" % (srb_nums + 2),
            nn.Sequential(
                nn.Conv2D(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                nn.BatchNorm2D(2 * hidden_units),
            ),
        )

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2D(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, "block%d" % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        self.out_channels = in_channels
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins),
            )

            self.stn_head = STNHead(
                in_channels=in_planes,
                num_ctrlpoints=num_control_points,
                activation="none",
            )
        self.infer_mode = infer_mode

        self.english_alphabet = (
            "-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        self.english_dict = {}
        for index in range(len(self.english_alphabet)):
            self.english_dict[self.english_alphabet[index]] = index
        transformer = Transformer(alphabet="-0123456789abcdefghijklmnopqrstuvwxyz")
        self.transformer = transformer
        for param in self.transformer.parameters():
            param.trainable = False

    def label_encoder(self, label):
        batch = len(label)

        length = [len(i) for i in label]
        length_tensor = paddle.to_tensor(length, dtype="int64")

        max_length = max(length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                input_tensor[i][j + 1] = self.english_dict[label[i][j]]

        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_dict[j])
        text_gt = paddle.to_tensor(text_gt, dtype="int64")

        input_tensor = paddle.to_tensor(input_tensor, dtype="int64")
        return length_tensor, input_tensor, text_gt

    def forward(self, x):
        output = {}
        if self.infer_mode:
            output["lr_img"] = x
            y = x
        else:
            output["lr_img"] = x[0]
            output["hr_img"] = x[1]
            y = x[0]
        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(y)
            y, _ = self.tps(y, ctrl_points_x)
        block = {"1": self.block1(y)}
        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, "block%d" % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, "block%d" % (self.srb_nums + 3))(
            (block["1"] + block[str(self.srb_nums + 2)])
        )

        sr_img = paddle.tanh(block[str(self.srb_nums + 3)])
        output["sr_img"] = sr_img

        if self.training:
            hr_img = x[1]

            # add transformer
            label = [str_filt(i, "lower") + "-" for i in x[2]]
            length_tensor, input_tensor, text_gt = self.label_encoder(label)
            hr_pred, word_attention_map_gt, hr_correct_list = self.transformer(
                hr_img, length_tensor, input_tensor
            )
            sr_pred, word_attention_map_pred, sr_correct_list = self.transformer(
                sr_img, length_tensor, input_tensor
            )
            output["hr_img"] = hr_img
            output["hr_pred"] = hr_pred
            output["text_gt"] = text_gt
            output["word_attention_map_gt"] = word_attention_map_gt
            output["sr_pred"] = sr_pred
            output["word_attention_map_pred"] = word_attention_map_pred

        return output


class RecurrentResidualBlock(nn.Layer):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2D(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(channels)
        self.gru2 = GruBlock(channels, channels)
        self.feature_enhancer = FeatureEnhancer()

        for p in self.parameters():
            if p.dim() > 1:
                paddle.nn.initializer.XavierUniform(p)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        size = residual.shape
        residual = residual.reshape([size[0], size[1], -1])
        residual = self.feature_enhancer(residual)
        residual = residual.reshape([size[0], size[1], size[2], size[3]])
        return x + residual
