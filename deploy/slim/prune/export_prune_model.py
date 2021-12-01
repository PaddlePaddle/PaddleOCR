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

import paddle
from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model

from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
import tools.program as program


def main(config, device, logger, vdl_writer):

    global_config = config['Global']

    # build dataloader
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])

    if config['Architecture']['model_type'] == 'det':
        input_shape = [1, 3, 640, 640]
    elif config['Architecture']['model_type'] == 'rec':
        input_shape = [1, 3, 32, 320]

    flops = paddle.flops(model, input_shape)
    logger.info("FLOPs before pruning: {}".format(flops))

    from paddleslim.dygraph import FPGMFilterPruner
    model.train()
    pruner = FPGMFilterPruner(model, input_shape)

    # build metric
    eval_class = build_metric(config['Metric'])

    def eval_fn():
        metric = program.eval(model, valid_dataloader, post_process_class,
                              eval_class)
        if config['Architecture']['model_type'] == 'det':
            main_indicator = 'hmean'
        else:
            main_indicator = 'acc'
        logger.info("metric[{}]: {}".format(main_indicator, metric[
            main_indicator]))
        return metric[main_indicator]

    params_sensitive = pruner.sensitive(
        eval_func=eval_fn,
        sen_file="./sen.pickle",
        skip_vars=[
            "conv2d_57.w_0", "conv2d_transpose_2.w_0", "conv2d_transpose_3.w_0"
        ])

    logger.info(
        "The sensitivity analysis results of model parameters saved in sen.pickle"
    )
    # calculate pruned params's ratio
    params_sensitive = pruner._get_ratios_by_loss(params_sensitive, loss=0.02)
    for key in params_sensitive.keys():
        logger.info("{}, {}".format(key, params_sensitive[key]))

    plan = pruner.prune_vars(params_sensitive, [0])

    flops = paddle.flops(model, input_shape)
    logger.info("FLOPs after pruning: {}".format(flops))

    # load pretrain model
    load_model(config, model)
    metric = program.eval(model, valid_dataloader, post_process_class,
                          eval_class)
    if config['Architecture']['model_type'] == 'det':
        main_indicator = 'hmean'
    else:
        main_indicator = 'acc'
    logger.info("metric['']: {}".format(main_indicator, metric[main_indicator]))

    # start export model
    from paddle.jit import to_static

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

    save_path = '{}/inference'.format(config['Global']['save_inference_dir'])
    paddle.jit.save(model, save_path)
    logger.info('inference model is saved to {}'.format(save_path))


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config, device, logger, vdl_writer)
