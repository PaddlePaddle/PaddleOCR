import paddle
from paddle import nn

# refer from: https://github.com/ViTAE-Transformer/I3CL/blob/736c80237f66d352d488e83b05f3e33c55201317/mmdet/models/detectors/intra_cl_module.py


class IntraCLBlock(nn.Layer):
    def __init__(self, in_channels=96, reduce_factor=4):
        super(IntraCLBlock, self).__init__()
        self.channels = in_channels
        self.rf = reduce_factor
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.conv1x1_reduce_channel = nn.Conv2D(
            self.channels,
            self.channels // self.rf,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv1x1_return_channel = nn.Conv2D(
            self.channels // self.rf,
            self.channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.v_layer_7x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(3, 0))
        self.v_layer_5x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0))
        self.v_layer_3x1 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0))

        self.q_layer_1x7 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(0, 3))
        self.q_layer_1x5 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 5),
            stride=(1, 1),
            padding=(0, 2))
        self.q_layer_1x3 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1))

        # base
        self.c_layer_7x7 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(3, 3))
        self.c_layer_5x5 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2))
        self.c_layer_3x3 = nn.Conv2D(
            self.channels // self.rf,
            self.channels // self.rf,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1))

        self.bn = nn.BatchNorm2D(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_new = self.conv1x1_reduce_channel(x)

        x_7_c = self.c_layer_7x7(x_new)
        x_7_v = self.v_layer_7x1(x_new)
        x_7_q = self.q_layer_1x7(x_new)
        x_7 = x_7_c + x_7_v + x_7_q

        x_5_c = self.c_layer_5x5(x_7)
        x_5_v = self.v_layer_5x1(x_7)
        x_5_q = self.q_layer_1x5(x_7)
        x_5 = x_5_c + x_5_v + x_5_q

        x_3_c = self.c_layer_3x3(x_5)
        x_3_v = self.v_layer_3x1(x_5)
        x_3_q = self.q_layer_1x3(x_5)
        x_3 = x_3_c + x_3_v + x_3_q

        x_relation = self.conv1x1_return_channel(x_3)

        x_relation = self.bn(x_relation)
        x_relation = self.relu(x_relation)

        return x + x_relation


def build_intraclblock_list(num_block):
    IntraCLBlock_list = nn.LayerList()
    for i in range(num_block):
        IntraCLBlock_list.append(IntraCLBlock())

    return IntraCLBlock_list