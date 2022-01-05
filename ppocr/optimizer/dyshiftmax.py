import paddle
import paddle.nn as nn
import numpy as np

from ppocr.modeling.backbones.det_mobilenet_v3 import make_divisible
import paddle.nn.functional as F


class h_sigmoid(nn.Layer):
    def __init__(self, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()
        self.h_max = h_max / 6

    def forward(self, x):
        return self.relu(x + 3) * self.h_max


class DYShiftMax(nn.Layer):
    def __init__(self,
                 inp,
                 oup,
                 reduction=4,
                 act_max=1.0,
                 act_relu=True,
                 init_a=[0.0, 0.0],
                 init_b=[0.0, 0.0],
                 relu_before_pool=False,
                 g=None,
                 expansion=False):
        super(DYShiftMax, self).__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(nn.ReLU() if relu_before_pool == True else
                                      nn.Sequential(), nn.AdaptiveAvgPool2D(1))

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        squeeze = make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(), nn.Linear(squeeze, oup * self.exp), h_sigmoid())

        if g is None:
            g = 1
        self.g = g[1]
        if self.g != 1 and expansion:
            self.g = inp // self.g

        self.gc = inp // self.g
        index = paddle.to_tensor([range(inp)])
        index = paddle.reshape(index, [1, inp, 1, 1])
        index = paddle.reshape(index, [1, self.g, self.gc, 1, 1])
        indexgs = paddle.split(index, [1, self.g - 1], axis=1)
        indexgs = paddle.concat((indexgs[1], indexgs[0]), axis=1)
        indexs = paddle.split(indexgs, [1, self.gc - 1], axis=2)
        indexs = paddle.concat((indexs[1], indexs[0]), axis=2)
        self.index = paddle.reshape(indexs, [inp])
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x

        b, c, _, _ = x_in.shape
        y = self.avg_pool(x_in)
        y = paddle.reshape(y, [b, c])
        y = self.fc(y)
        y = paddle.reshape(y, [b, self.oup * self.exp, 1, 1])
        y = (y - 0.5) * self.act_max

        n2, c2, h2, w2 = x_out.shape
        x2 = paddle.to_tensor(x_out.numpy()[:, self.index.numpy(), :, :])

        if self.exp == 4:
            temp = y.shape
            a1, b1, a2, b2 = paddle.split(y, temp[1] // self.oup, axis=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = paddle.maximum(z1, z2)

        elif self.exp == 2:
            temp = y.shape
            a1, b1 = paddle.split(y, temp[1] // self.oup, axis=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out
