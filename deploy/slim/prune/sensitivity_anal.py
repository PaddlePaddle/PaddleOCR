# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '..', '..', '..'))
sys.path.append(os.path.join(__dir__, '..', '..', '..', 'tools'))

import json
import cv2
import paddle
from paddle import fluid
import paddleslim as slim
from copy import deepcopy
from tools import program

import paddle
import paddle.distributed as dist
from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
import tools.program as program

dist.get_world_size()


def get_pruned_params(parameters, mode="det"):
    if mode == "det":
        skip_prune_params = [
            "conv2d_56.w_0", "conv2d_54.w_0", "conv2d_51.w_0",
            "conv_last_weights", "conv14_linear_weights",
            "conv13_expand_weights", "conv12_linear_weights",
            "conv12_expand_weights", "conv7_expand_weights",
            "conv8_expand_weights", "conv8_linear_weights",
            "conv5_linear_weights", "conv5_expand_weights",
            "conv3_linear_weights"
        ]
        skip_prune_params = skip_prune_params + ['conv2d_53.w_0']
    else:
        skip_prune_params = None
    params = []

    for param in parameters:
        if len(
                param.shape
        ) == 4 and 'depthwise' not in param.name and 'transpose' not in param.name and "conv2d_57" not in param.name and "conv2d_56" not in param.name:
            if param.name not in skip_prune_params:
                params.append(param.name)
    return params


def main(config, device, logger, vdl_writer):
    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()

    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])

    flops = paddle.flops(model, [1, 3, 640, 640])
    logger.info(f"FLOPs before pruning: {flops}")

    from paddleslim.dygraph import FPGMFilterPruner
    model.train()
    pruner = FPGMFilterPruner(model, [1, 3, 640, 640])

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = init_model(config, model, logger, optimizer)

    logger.info('train dataloader has {} iters, valid dataloader has {} iters'.
                format(len(train_dataloader), len(valid_dataloader)))

    def eval_fn():
        metric = program.eval(model, valid_dataloader, post_process_class,
                              eval_class)
        logger.info(f"metric['hmean']: {metric['hmean']}")
        return metric['hmean']

    pruner.sensitive(
        eval_func=eval_fn,
        sen_file="./sen.pickle",
        skip_vars=[
            "conv2d_57.w_0", "conv2d_transpose_2.w_0", "conv2d_transpose_3.w_0"
        ])

    params = get_pruned_params(model.parameters())
    ratios = {}
    # set the prune ratio is 0.2
    for param in params:
        ratios[param] = 0.2

    plan = pruner.prune_vars(ratios, [0])
    for param in model.parameters():
        if ("weights" in param.name and "conv" in param.name) or (
                "w_0" in param.name and "conv2d" in param.name):
            logger.info(f"{param.name}: {param.shape}")

    flops = paddle.flops(model, [1, 3, 640, 640])
    logger.info(f"FLOPs after pruning: {flops}")

    # start train
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer)


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config, device, logger, vdl_writer)
