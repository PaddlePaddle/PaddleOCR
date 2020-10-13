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

import yaml
import paddle
import paddle.distributed as dist

paddle.manual_seed(2)

from ppocr.utils.logging import get_logger
from ppocr.data import build_dataloader
from ppocr.modeling import build_model, build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
from ppocr.utils.utility import print_dict
import tools.program as program

dist.get_world_size()


def main(config, device, logger, vdl_writer):
    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()

    global_config = config['Global']
    # build dataloader
    train_loader, train_info_dict = build_dataloader(
        config['TRAIN'], device, global_config['distributed'], global_config)
    if config['EVAL']:
        eval_loader, _ = build_dataloader(config['EVAL'], device, False,
                                          global_config)
    else:
        eval_loader = None
    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))
    model = build_model(config['Architecture'])
    if config['Global']['distributed']:
        model = paddle.DataParallel(model)

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_loader),
        parameters=model.parameters())

    best_model_dict = init_model(config, model, logger, optimizer)

    # build loss
    loss_class = build_loss(config['Loss'])
    # build metric
    eval_class = build_metric(config['Metric'])

    # start train
    program.train(config, model, loss_class, optimizer, lr_scheduler,
                  train_loader, eval_loader, post_process_class, eval_class,
                  best_model_dict, logger, vdl_writer)


def test_reader(config, place, logger):
    train_loader = build_dataloader(config['TRAIN'], place)
    import time
    starttime = time.time()
    count = 0
    try:
        for data in train_loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(count,
                                                        len(data), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


def dis_main():
    device, config = program.preprocess()
    config['Global']['distributed'] = dist.get_world_size() != 1
    paddle.disable_static(device)

    # save_config
    os.makedirs(config['Global']['save_model_dir'], exist_ok=True)
    with open(
            os.path.join(config['Global']['save_model_dir'], 'config.yml'),
            'w') as f:
        yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)

    logger = get_logger(
        log_file='{}/train.log'.format(config['Global']['save_model_dir']))
    if config['Global']['use_visualdl']:
        from visualdl import LogWriter
        vdl_writer = LogWriter(logdir=config['Global']['save_model_dir'])
    else:
        vdl_writer = None
    print_dict(config, logger)
    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))

    main(config, device, logger, vdl_writer)
    # test_reader(config, place, logger)


if __name__ == '__main__':
    # main()
    # dist.spawn(dis_main, nprocs=2, selelcted_gpus='6,7')
    dis_main()
