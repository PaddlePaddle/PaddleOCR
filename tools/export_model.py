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

import paddle
from paddle.jit import to_static

from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import init_model
from tools.program import load_config
from tools.program import merge_config


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        "-o", "--output_path", type=str, default='./output/infer/')
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
    FLAGS = parse_args()
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            config['Global'])

    # build model
    #for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])
    init_model(config, model, logger)
    model.eval()

    model = Model(model)
    paddle.jit.save(model, FLAGS.output_path)


if __name__ == "__main__":
    main()
