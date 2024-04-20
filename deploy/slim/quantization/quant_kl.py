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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(__dir__, "..", "..", "..", "tools")))

import yaml
import paddle
import paddle.distributed as dist

paddle.seed(2)

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
import tools.program as program
import paddleslim
from paddleslim.dygraph.quant import QAT
import numpy as np

dist.get_world_size()


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


def sample_generator(loader):
    def __reader__():
        for indx, data in enumerate(loader):
            images = np.array(data[0])
            yield images

    return __reader__


def sample_generator_layoutxlm_ser(loader):
    def __reader__():
        for indx, data in enumerate(loader):
            input_ids = np.array(data[0])
            bbox = np.array(data[1])
            attention_mask = np.array(data[2])
            token_type_ids = np.array(data[3])
            images = np.array(data[4])
            yield [input_ids, bbox, attention_mask, token_type_ids, images]

    return __reader__


def main(config, device, logger, vdl_writer):
    # init dist environment
    if config["Global"]["distributed"]:
        dist.init_parallel_env()

    global_config = config["Global"]

    # build dataloader
    set_signal_handlers()
    config["Train"]["loader"]["num_workers"] = 0
    is_layoutxlm_ser = (
        config["Architecture"]["model_type"] == "kie"
        and config["Architecture"]["Backbone"]["name"] == "LayoutXLMForSer"
    )
    train_dataloader = build_dataloader(config, "Train", device, logger)
    if config["Eval"]:
        config["Eval"]["loader"]["num_workers"] = 0
        valid_dataloader = build_dataloader(config, "Eval", device, logger)
        if is_layoutxlm_ser:
            train_dataloader = valid_dataloader
    else:
        valid_dataloader = None

    paddle.enable_static()
    exe = paddle.static.Executor(device)

    if "inference_model" in global_config.keys():  # , 'inference_model'):
        inference_model_dir = global_config["inference_model"]
    else:
        inference_model_dir = os.path.dirname(global_config["pretrained_model"])
        if not (
            os.path.exists(os.path.join(inference_model_dir, "inference.pdmodel"))
            and os.path.exists(os.path.join(inference_model_dir, "inference.pdiparams"))
        ):
            raise ValueError(
                "Please set inference model dir in Global.inference_model or Global.pretrained_model for post-quantization"
            )

    if is_layoutxlm_ser:
        generator = sample_generator_layoutxlm_ser(train_dataloader)
    else:
        generator = sample_generator(train_dataloader)

    paddleslim.quant.quant_post_static(
        executor=exe,
        model_dir=inference_model_dir,
        model_filename="inference.pdmodel",
        params_filename="inference.pdiparams",
        quantize_model_path=global_config["save_inference_dir"],
        sample_generator=generator,
        save_model_filename="inference.pdmodel",
        save_params_filename="inference.pdiparams",
        batch_size=1,
        batch_nums=None,
    )


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config, device, logger, vdl_writer)
