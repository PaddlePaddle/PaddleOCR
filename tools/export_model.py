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

import argparse

from ppocr.utils.save_load import load_dygraph_pretrain
import paddle
from paddle.jit import to_static

from ppocr.utils.logging import get_logger
from ppocr.utils.save_load import init_model
from ppocr.modeling import build_model
from tools.program import load_config


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default='/home/zhoujun20/dygraph/PaddleOCR/output/rec/benckmark_Im2seq/inference'
    )
    parser.add_argument("--out_channels", type=int, default=37)
    return parser.parse_args()


class Model(paddle.nn.Layer):
    def __init__(self, model):
        super(Model, self).__init__()
        self.pre_model = model

    # Please modify the 'shape' according to actual needs
    @to_static(input_spec=[
        paddle.static.InputSpec(
            shape=[None, 3, 32, None], dtype='float32')
    ])
    def forward(self, inputs):
        x = self.pre_model(inputs)
        return x


def main():
    logger = get_logger()
    paddle.disable_static()
    FLAGS = parse_args()
    config = load_config(FLAGS.config)
    config['Architecture']["Head"]['out_channels'] = FLAGS.out_channels
    model = build_model(config['Architecture'])
    init_model(config, model, logger)
    model.eval()

    model = Model(model)
    paddle.jit.save(model, FLAGS.output_path)


if __name__ == "__main__":
    main()
