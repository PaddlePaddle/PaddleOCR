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
from tools.eval_utils.eval_det_utils import eval_det_run

from tools import program
from ppocr.utils.utility import initial_logger
from ppocr.data.reader_main import reader_main
from ppocr.utils.save_load import init_model
from ppocr.utils.character import CharacterOps
from ppocr.utils.utility import create_module
from ppocr.data.reader_main import reader_main

logger = initial_logger()


def get_pruned_params(program):
    params = []
    for param in program.global_block().all_parameters():
        if len(
                param.shape
        ) == 4 and 'depthwise' not in param.name and 'transpose' not in param.name:
            params.append(param.name)
    return params


def eval_function(eval_args, mode='eval'):
    exe = eval_args['exe']
    config = eval_args['config']
    eval_info_dict = eval_args['eval_info_dict']
    metrics = eval_det_run(exe, config, eval_info_dict, mode=mode)
    return metrics['hmean']


def main():
    # Run code with static graph mode.
    try:
        paddle.enable_static()
    except:
        pass

    config = program.load_config(FLAGS.config)
    program.merge_config(FLAGS.opt)
    logger.info(config)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    program.check_gpu(use_gpu)

    alg = config['Global']['algorithm']
    assert alg in ['EAST', 'DB', 'Rosetta', 'CRNN', 'STARNet', 'RARE']
    if alg in ['Rosetta', 'CRNN', 'STARNet', 'RARE']:
        config['Global']['char_ops'] = CharacterOps(config['Global'])

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    startup_prog = fluid.Program()
    eval_program = fluid.Program()
    eval_build_outputs = program.build(
        config, eval_program, startup_prog, mode='test')
    eval_fetch_name_list = eval_build_outputs[1]
    eval_fetch_varname_list = eval_build_outputs[2]
    eval_program = eval_program.clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    init_model(config, eval_program, exe)

    eval_reader = reader_main(config=config, mode="eval")
    eval_info_dict = {'program':eval_program,\
        'reader':eval_reader,\
        'fetch_name_list':eval_fetch_name_list,\
        'fetch_varname_list':eval_fetch_varname_list}
    eval_args = dict()
    eval_args = {'exe': exe, 'config': config, 'eval_info_dict': eval_info_dict}
    metrics = eval_function(eval_args)
    print("Baseline: {}".format(metrics))

    params = get_pruned_params(eval_program)
    print('Start to analyze')
    sens_0 = slim.prune.sensitivity(
        eval_program,
        place,
        params,
        eval_function,
        sensitivities_file="sensitivities_0.data",
        pruned_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        eval_args=eval_args,
        criterion='geometry_median')


if __name__ == '__main__':
    parser = program.ArgsParser()
    FLAGS = parser.parse_args()
    main()
