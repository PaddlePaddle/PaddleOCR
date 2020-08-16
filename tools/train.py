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
sys.path.append(os.path.join(__dir__, '..'))


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

import tools.program as program
from paddle import fluid
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.data.reader_main import reader_main
from ppocr.utils.save_load import init_model
from paddle.fluid.contrib.model_stat import summary


def main():
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
    train_compile_program = program.create_multi_devices_program(
        train_program, train_opt_loss_name)

    # dump mode structure
    if config['Global']['debug']:
        if train_alg_type == 'rec' and 'attention' in config['Global']['loss_type']:
            logger.warning('Does not suport dump attention...')
        else:
            summary(train_program)

    init_model(config, train_program, exe)

    train_info_dict = {'compile_program':train_compile_program,\
        'train_program':train_program,\
        'reader':train_loader,\
        'fetch_name_list':train_fetch_name_list,\
        'fetch_varname_list':train_fetch_varname_list}

    eval_info_dict = {'program':eval_program,\
        'reader':eval_reader,\
        'fetch_name_list':eval_fetch_name_list,\
        'fetch_varname_list':eval_fetch_varname_list}

    if train_alg_type == 'det':
        program.train_eval_det_run(config, exe, train_info_dict, eval_info_dict)
    else:
        program.train_eval_rec_run(config, exe, train_info_dict, eval_info_dict)


def test_reader():
    logger.info(config)
    train_reader = reader_main(config=config, mode="train")
    import time
    starttime = time.time()
    count = 0
    try:
        for data in train_reader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader:", count, len(data), batch_time)
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    startup_program, train_program, place, config, train_alg_type = program.preprocess()
    main()
#     test_reader()
