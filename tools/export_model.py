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
import time
import multiprocessing
import numpy as np


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

import program
from paddle import fluid
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.utils.save_load import init_model
from ppocr.utils.character import CharacterOps
from ppocr.utils.utility import create_module


def main():
    config = program.load_config(FLAGS.config)
    program.merge_config(FLAGS.opt)
    logger.info(config)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    # program.check_gpu(True)
    use_gpu = False

    alg = config['Global']['algorithm']
    assert alg in ['EAST', 'DB', 'Rosetta', 'CRNN', 'STARNet', 'RARE']
    if alg in ['Rosetta', 'CRNN', 'STARNet', 'RARE']:
        config['Global']['char_ops'] = CharacterOps(config['Global'])

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    startup_prog = fluid.Program()
    eval_program = fluid.Program()

    feeded_var_names, target_vars, fetches_var_name = program.build_export(
        config, eval_program, startup_prog)
    eval_program = eval_program.clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    init_model(config, eval_program, exe)

    save_inference_dir = config['Global']['save_inference_dir']
    if not os.path.exists(save_inference_dir):
        os.makedirs(save_inference_dir)
    fluid.io.save_inference_model(
        dirname=save_inference_dir,
        feeded_var_names=feeded_var_names,
        main_program=eval_program,
        target_vars=target_vars,
        executor=exe,
        model_filename='model',
        params_filename='params')
    print("inference model saved in {}/model and {}/params".format(
        save_inference_dir, save_inference_dir))
    print("save success, output_name_list:", fetches_var_name)


if __name__ == '__main__':
    parser = program.ArgsParser()
    FLAGS = parser.parse_args()
    main()
