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
from tools.program import load_config, merge_config, ArgsParser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        "-o", "--output_path", type=str, default='./output/infer/')
    return parser.parse_args()


def main():
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
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

    save_path = '{}/inference'.format(config['Global']['save_inference_dir'])

    if config['Architecture']['algorithm'] == "SRN":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 64, 256], dtype='float32'), [
                    paddle.static.InputSpec(
                        shape=[None, 256, 1],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, 25, 1],
                            dtype="int64"), paddle.static.InputSpec(
                                shape=[None, 8, 25, 25], dtype="int64"),
                    paddle.static.InputSpec(
                        shape=[None, 8, 25, 25], dtype="int64")
                ]
        ]
        model = to_static(model, input_spec=other_shape)
    else:
        infer_shape = [3, -1, -1]
        if config['Architecture']['model_type'] == "rec":
            infer_shape = [3, 32, -1]  # for rec model, H must be 32
            if 'Transform' in config['Architecture'] and config['Architecture'][
                    'Transform'] is not None and config['Architecture'][
                        'Transform']['name'] == 'TPS':
                logger.info(
                    'When there is tps in the network, variable length input is not supported, and the input size needs to be the same as during training'
                )
                infer_shape[-1] = 100
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
