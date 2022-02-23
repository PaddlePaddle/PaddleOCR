# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.nn as nn
from paddle.nn.utils import weight_norm

__all__ = ["TemporalBlock", "TCN"]


class Chomp1d(nn.Layer):
    """
    Remove the elements on the right.

    Args:
        chomp_size (int):
            The number of elements removed.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Layer):
    """
    The TCN block, consists of dilated causal conv, relu and residual block. 
    See the Figure 1(b) in https://arxiv.org/pdf/1803.01271.pdf for more details.

    Args:
        n_inputs (int):
            The number of channels in the input tensor.
        n_outputs (int):
            The number of filters.
        kernel_size (int):
            The filter size.
        stride (int):
            The stride size.
        dilation (int):
            The dilation size.
        padding (int):
            The size of zeros to be padded.
        dropout (float, optional):
            Probability of dropout the units. Defaults to 0.2.
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1D(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        # Chomp1d is used to make sure the network is causal.
        # We pad by (k-1)*d on the two sides of the input for convolution, 
        # and then use Chomp1d to remove the (k-1)*d output elements on the right.
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1D(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2)
        self.downsample = nn.Conv1D(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.set_value(
            paddle.tensor.normal(0.0, 0.01, self.conv1.weight.shape))
        self.conv2.weight.set_value(
            paddle.tensor.normal(0.0, 0.01, self.conv2.weight.shape))
        if self.downsample is not None:
            self.downsample.weight.set_value(
                paddle.tensor.normal(0.0, 0.01, self.downsample.weight.shape))

    def forward(self, x):
        """
        Args:
            x (Tensor):
                The input tensor with a shape  of [batch_size, input_channel, sequence_length].

        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Layer):
    def __init__(self, input_channel, num_channels, kernel_size=2, dropout=0.2):
        """
        Temporal Convolutional Networks is a simple convolutional architecture. It outperforms canonical recurrent networks
        such as LSTMs in many tasks. See https://arxiv.org/pdf/1803.01271.pdf for more details.

        Args:
            input_channel (int):
                The number of channels in the input tensor.
            num_channels (list | tuple):
                The number of channels in different layer.
            kernel_size (int, optional):
                The filter size.. Defaults to 2.
            dropout (float, optional):
                Probability of dropout the units.. Defaults to 0.2.
        """
        super(TCN, self).__init__()
        layers = nn.LayerList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_channel if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Apply temporal convolutional networks to the input tensor.

        Args:
            x (Tensor):
                The input tensor with a shape of [batch_size, input_channel, sequence_length].

        Returns:
            Tensor: The `output` tensor with a shape of [batch_size, num_channels[-1], sequence_length].
        """
        output = self.network(x)
        return output
