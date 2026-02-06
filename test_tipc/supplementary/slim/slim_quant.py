import paddle
import numpy as np
import os
import paddle.nn as nn
import paddleslim


class PACT(paddle.nn.Layer):
    def __init__(self):
        super(PACT, self).__init__()
        alpha_attr = paddle.ParamAttr(
            name=self.full_name() + ".pact",
            initializer=paddle.nn.initializer.Constant(value=20),
            learning_rate=1.0,
            regularizer=paddle.regularizer.L2Decay(2e-5),
        )

        self.alpha = self.create_parameter(shape=[1], attr=alpha_attr, dtype="float32")

    def forward(self, x):
        out_left = paddle.nn.functional.relu(x - self.alpha)
        out_right = paddle.nn.functional.relu(-self.alpha - x)
        x = x - out_left + out_right
        return x


quant_config = {
    # weight preprocess type, default is None and no preprocessing is performed.
    "weight_preprocess_type": None,
    # activation preprocess type, default is None and no preprocessing is performed.
    "activation_preprocess_type": None,
    # weight quantize type, default is 'channel_wise_abs_max'
    "weight_quantize_type": "channel_wise_abs_max",
    # activation quantize type, default is 'moving_average_abs_max'
    "activation_quantize_type": "moving_average_abs_max",
    # weight quantize bit num, default is 8
    "weight_bits": 8,
    # activation quantize bit num, default is 8
    "activation_bits": 8,
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    "dtype": "int8",
    # window size for 'range_abs_max' quantization. default is 10000
    "window_size": 10000,
    # The decay coefficient of moving average, default is 0.9
    "moving_rate": 0.9,
    # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
    "quantizable_layer_type": ["Conv2D", "Linear"],
}
