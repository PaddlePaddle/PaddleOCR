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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import argparse

import paddle
from paddle.jit import to_static

from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import init_model
from ppocr.utils.logging import get_logger
from tools.program import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        "-o", "--output_path", type=str, default='./output/infer/')
    return parser.parse_args()


def main():
    FLAGS = parse_args()
    config = load_config(FLAGS.config)
    logger = get_logger()
    # build post process

    post_process_class = build_post_process(config['PostProcess'],
                                            config['Global'])

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])
    init_model(config, model, logger)
    model.eval()

    save_path = '{}/inference'.format(FLAGS.output_path)
    infer_shape = [3, 32, 100] if config['Architecture'][
        'model_type'] != "det" else [3, 640, 640]
    model = to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + infer_shape, dtype='float32')
        ])
    paddle.jit.save(model, save_path)
    logger.info('inference model is saved to {}'.format(save_path))


if __name__ == "__main__":
    main()
