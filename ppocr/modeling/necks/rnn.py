# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from ppocr.modeling.heads.rec_ctc_head import get_para_bias_attr
from ppocr.modeling.backbones.rec_svtrnet import Block, ConvBNLayer, trunc_normal_, zeros_, ones_


class Im2Seq(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        return x


class EncoderWithRNN(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class BidirectionalLSTM(nn.Layer):
    def __init__(self, input_size,
                 hidden_size,
                 output_size=None,
                 num_layers=1,
                 dropout=0,
                 direction=False,
                 time_major=False,
                 with_linear=False):
        super(BidirectionalLSTM, self).__init__()
        self.with_linear = with_linear
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           direction=direction,
                           time_major=time_major)

        # text recognition the specified structure LSTM with linear
        if self.with_linear:
            self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_feature):
        recurrent, _ = self.rnn(input_feature)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        if self.with_linear:
            output = self.linear(recurrent)     # batch_size x T x output_size
            return output
        return recurrent

class EncoderWithCascadeRNN(nn.Layer):
    def __init__(self, in_channels, hidden_size, out_channels, num_layers=2, with_linear=False):
        super(EncoderWithCascadeRNN, self).__init__()
        self.out_channels = out_channels[-1]
        self.encoder = nn.LayerList(
            [BidirectionalLSTM(
                in_channels if i == 0 else out_channels[i - 1], 
                hidden_size, 
                output_size=out_channels[i], 
                num_layers=1, 
                direction='bidirectional', 
                with_linear=with_linear) 
            for i in range(num_layers)]
        )
        

    def forward(self, x):
        for i, l in enumerate(self.encoder):
            x = l(x)
        return x


class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=0.00001, k=in_channels)
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='reduce_encoder_fea')

    def forward(self, x):
        x = self.fc(x)
        return x


class EncoderWithSVTR(nn.Layer):
    def __init__(
            self,
            in_channels,
            dims=64,  # XS
            depth=2,
            hidden_dims=120,
            use_guide=False,
            num_heads=8,
            qkv_bias=True,
            mlp_ratio=2.0,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path=0.,
            qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(
            in_channels, in_channels // 8, padding=1, act=nn.Swish)
        self.conv2 = ConvBNLayer(
            in_channels // 8, hidden_dims, kernel_size=1, act=nn.Swish)

        self.svtr_block = nn.LayerList([
            Block(
                dim=hidden_dims,
                num_heads=num_heads,
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.Swish,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer='nn.LayerNorm',
                epsilon=1e-05,
                prenorm=False) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dims, epsilon=1e-6)
        self.conv3 = ConvBNLayer(
            hidden_dims, in_channels, kernel_size=1, act=nn.Swish)
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(
            2 * in_channels, in_channels // 8, padding=1, act=nn.Swish)

        self.conv1x1 = ConvBNLayer(
            in_channels // 8, dims, kernel_size=1, act=nn.Swish)
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        # for use guide
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).transpose([0, 2, 1])
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # last stage
        z = z.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        z = self.conv3(z)
        z = paddle.concat((h, z), axis=1)
        z = self.conv1x1(self.conv4(z))
        return z


class SequenceEncoder(nn.Layer):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN,
                'svtr': EncoderWithSVTR,
                'cascadernn': EncoderWithCascadeRNN
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs)
            elif encoder_type == 'cascadernn':
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x
