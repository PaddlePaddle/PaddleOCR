import math
import paddle
import paddle.nn.functional as F
from paddle import nn
from collections import OrderedDict
import sys
import numpy as np
from IPython import embed
import warnings
import math, copy
import cv2

warnings.filterwarnings("ignore")

from .tps_spatial_transformer import TPSSpatialTransformer
from .stn import STN as STN_model

def print_hook_fn(grad):
    tmp = grad

class TSRN(nn.Layer):
    def __init__(self, in_channels, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=False, hidden_units=32, **kwargs):
        super(TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2D(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2D(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2D(2*hidden_units)
                ))
        
        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2D(2*hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height//scale_factor, width//scale_factor]
        tps_outputsize = [height//scale_factor, width//scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STN_model(
                in_channels=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')
        self.out_channels=in_channels

    def forward(self, x):
        #x = x[1]
        # [256, 32, 128, 3]
        # embed()
        if self.stn and self.training:
            # print("x shape:", x.shape)
            _, ctrl_points_x = self.stn_head(x)
            # print("ctrl_poinsts_x:", np.sum(ctrl_points_x.numpy()))
            # print("ctrl_poinst_x:", ctrl_points_x.shape)
            # print("x:", x.shape)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}
        # print("block1:", np.sum(self.block1(x).numpy()))
        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])
            # print("block{}:{}".format(str(i + 2), np.sum(block[str(i + 2)].numpy())))

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        # print("block{}:{}".format(str(self.srb_nums + 3), np.sum(block[str(self.srb_nums + 3)].numpy())))

        output = paddle.tanh(block[str(self.srb_nums + 3)])
        # print("batch pics:", np.sum(output.numpy()))
        # batch pics: 196598.17
        ### visual data
        # for i in (range(output.shape[0])):
        #     fm = (output[i].numpy() * 255).transpose(1,2,0).astype(np.uint8)
        #     # fm = cv2.resize(fm, (128,32))
        #     print("fm shape:", fm.shape)
        #     cv2.imwrite("visual_data/SR_out_{}.jpg".format(i), fm)
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

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        #print("tsrn batch norm:", residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        #print("before contiguous:", np.sum(residual.numpy()))
        # todo residual
        residual = self.gru1(residual.transpose([0,1,3,2])).transpose([0,1,3,2])
        #print("after gru:", np.sum(residual.numpy()))

        return self.gru2(x + residual)


class UpsampleBLock(nn.Layer):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2D(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Layer):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (paddle.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, direction='bidirectional')

    def forward(self, x):
        # x: b, c, w, h
        x = self.conv1(x)
        x = x.transpose([0, 2, 3, 1])# b, w, h, c
        b = x.shape
        x = x.reshape([b[0] * b[1], b[2], b[3]]) # b*w, h, c
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.reshape([b[0], b[1], b[2], b[3]])
        x = x.transpose([0, 3, 1, 2])
        return x




if __name__ == '__main__':

    np.random.seed(50)
    data = np.random.randn(7, 3, 16, 64).astype("float32")
    data = paddle.to_tensor(data)
    print("inputdata:", np.sum(data.numpy()))
    tsrn_model = TSRN(3, STN=True)
    params = paddle.load('tsrn_paddle.pdparams')
    state_dict = tsrn_model.state_dict()
    # for k,v in state_dict.items():
    #     print(k)
    # for k,v in params.items():
    #     print(k)
    new_state_dict = {}
    for k1 in state_dict.keys():
        if k1 == "block1.1._weight":
            k2 = "block1.1.weight"
        else:
            k2 = k1
        if k2 not in params.keys():
            pass
            #print("The pretrained params {} not in model".format(k2))
        else:
            if list(state_dict[k1].shape) == list(params[k2].shape):
                new_state_dict[k1] = params[k2]
            else:
                print(
                    "The shape of model params {} {} not matched with loaded params {} {} !".
                    format(k1, state_dict[k1].shape, k1, params[k1].shape))
    tsrn_model.set_state_dict(new_state_dict)
    output = tsrn_model(data)
    print("output shape:", output.shape)

    print("output:", np.sum(output.numpy()))

