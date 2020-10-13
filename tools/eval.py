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
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import paddle
# paddle.manual_seed(2)

from ppocr.utils.logging import get_logger
from ppocr.data import build_dataloader
from ppocr.modeling import build_model
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
from ppocr.utils.utility import print_dict
import tools.program as program


def main():
    global_config = config['Global']
    # build dataloader
    eval_loader, _ = build_dataloader(config['EVAL'], device, False,
                                      global_config)

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))
    model = build_model(config['Architecture'])

    best_model_dict = init_model(config, model, logger)
    if len(best_model_dict):
        logger.info('metric in ckpt ***************')
        for k, v in best_model_dict.items():
            logger.info('{}:{}'.format(k, v))

    # build metric
    eval_class = build_metric(config['Metric'])

    # start eval
    metirc = program.eval(model, eval_loader, post_process_class, eval_class)
    logger.info('metric eval ***************')
    for k, v in metirc.items():
        logger.info('{}:{}'.format(k, v))


if __name__ == '__main__':
    device, config = program.preprocess()
    paddle.disable_static(device)

    logger = get_logger()
    print_dict(config, logger)
    main()
