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
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/necks/fpn_unet.py
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class UpBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        self.conv1x1 = nn.Conv2D(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv3x3 = nn.Conv2D(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.deconv = nn.Conv2DTranspose(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        x = F.relu(self.conv1x1(x))
        x = F.relu(self.conv3x3(x))
        x = self.deconv(x)
        return x


class FPN_UNet(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert len(in_channels) == 4
        assert isinstance(out_channels, int)
        self.out_channels = out_channels

        blocks_out_channels = [out_channels] + [
            min(out_channels * 2**i, 256) for i in range(4)
        ]
        blocks_in_channels = (
            [blocks_out_channels[1]]
            + [in_channels[i] + blocks_out_channels[i + 2] for i in range(3)]
            + [in_channels[3]]
        )

        self.up4 = nn.Conv2DTranspose(
            blocks_in_channels[4],
            blocks_out_channels[4],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block3 = UpBlock(blocks_in_channels[3], blocks_out_channels[3])
        self.up_block2 = UpBlock(blocks_in_channels[2], blocks_out_channels[2])
        self.up_block1 = UpBlock(blocks_in_channels[1], blocks_out_channels[1])
        self.up_block0 = UpBlock(blocks_in_channels[0], blocks_out_channels[0])

    def forward(self, x):
        """
        Args:
            x (list[Tensor] | tuple[Tensor]): A list of four tensors of shape
                :math:`(N, C_i, H_i, W_i)`, representing C2, C3, C4, C5
                features respectively. :math:`C_i` should matches the number in
                ``in_channels``.

        Returns:
            Tensor: Shape :math:`(N, C, H, W)` where :math:`H=4H_0` and
            :math:`W=4W_0`.
        """
        c2, c3, c4, c5 = x

        x = F.relu(self.up4(c5))

        x = paddle.concat([x, c4], axis=1)
        x = F.relu(self.up_block3(x))

        x = paddle.concat([x, c3], axis=1)
        x = F.relu(self.up_block2(x))

        x = paddle.concat([x, c2], axis=1)
        x = F.relu(self.up_block1(x))

        x = self.up_block0(x)
        return x
