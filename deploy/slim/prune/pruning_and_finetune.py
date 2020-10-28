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
import numpy as np
__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '..', '..', '..'))
sys.path.append(os.path.join(__dir__, '..', '..', '..', 'tools'))

import tools.program as program
from paddle import fluid
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.data.reader_main import reader_main
from ppocr.utils.save_load import init_model
from ppocr.utils.character import CharacterOps
from ppocr.utils.utility import initial_logger
from paddleslim.prune import Pruner, save_model
from paddleslim.analysis import flops
from paddleslim.core.graph_wrapper import *
from paddleslim.prune import load_sensitivities, get_ratios_by_loss, merge_sensitive
logger = initial_logger()

skip_list = [
    'conv10_linear_weights', 'conv11_linear_weights', 'conv12_expand_weights',
    'conv12_linear_weights', 'conv12_se_2_weights', 'conv13_linear_weights',
    'conv2_linear_weights', 'conv4_linear_weights', 'conv5_expand_weights',
    'conv5_linear_weights', 'conv5_se_2_weights', 'conv6_linear_weights',
    'conv7_linear_weights', 'conv8_expand_weights', 'conv8_linear_weights',
    'conv9_expand_weights', 'conv9_linear_weights'
]


def main():
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
    startup_program = fluid.Program()
    train_program = fluid.Program()
    train_build_outputs = program.build(
        config, train_program, startup_program, mode='train')
    train_loader = train_build_outputs[0]
    train_fetch_name_list = train_build_outputs[1]
    train_fetch_varname_list = train_build_outputs[2]
    train_opt_loss_name = train_build_outputs[3]

    eval_program = fluid.Program()
    eval_build_outputs = program.build(
        config, eval_program, startup_program, mode='eval')
    eval_fetch_name_list = eval_build_outputs[1]
    eval_fetch_varname_list = eval_build_outputs[2]
    eval_program = eval_program.clone(for_test=True)

    train_reader = reader_main(config=config, mode="train")
    train_loader.set_sample_list_generator(train_reader, places=place)

    eval_reader = reader_main(config=config, mode="eval")

    exe = fluid.Executor(place)
    exe.run(startup_program)

    # compile program for multi-devices
    init_model(config, train_program, exe)

    sen = load_sensitivities("sensitivities_0.data")
    for i in skip_list:
        if i in sen.keys():
            sen.pop(i)
    back_bone_list = ['conv' + str(x) for x in range(1, 5)]
    for i in back_bone_list:
        for key in list(sen.keys()):
            if i + '_' in key:
                sen.pop(key)
    ratios = get_ratios_by_loss(sen, 0.03)
    logger.info("FLOPs before pruning: {}".format(flops(eval_program)))
    pruner = Pruner(criterion='geometry_median')
    print("ratios: {}".format(ratios))
    pruned_val_program, _, _ = pruner.prune(
        eval_program,
        fluid.global_scope(),
        params=ratios.keys(),
        ratios=ratios.values(),
        place=place,
        only_graph=True)

    pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=ratios.keys(),
        ratios=ratios.values(),
        place=place)
    logger.info("FLOPs after pruning: {}".format(flops(pruned_val_program)))
    train_compile_program = program.create_multi_devices_program(
        pruned_program, train_opt_loss_name)


    train_info_dict = {'compile_program':train_compile_program,\
        'train_program':pruned_program,\
        'reader':train_loader,\
        'fetch_name_list':train_fetch_name_list,\
        'fetch_varname_list':train_fetch_varname_list}

    eval_info_dict = {'program':pruned_val_program,\
        'reader':eval_reader,\
        'fetch_name_list':eval_fetch_name_list,\
        'fetch_varname_list':eval_fetch_varname_list}

    if alg in ['EAST', 'DB']:
        program.train_eval_det_run(
            config, exe, train_info_dict, eval_info_dict, is_slim="prune")
    else:
        program.train_eval_rec_run(config, exe, train_info_dict, eval_info_dict)


if __name__ == '__main__':
    parser = program.ArgsParser()
    FLAGS = parser.parse_args()
    main()
