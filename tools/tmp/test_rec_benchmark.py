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

from paddle import fluid

from ppocr.utils.utility import load_config, merge_config
import ppocr.data.rec.reader_main as reader

from ppocr.utils.utility import ArgsParser
from ppocr.utils.character import CharacterOps, cal_predicts_accuracy
from ppocr.utils.check import check_gpu
from ppocr.utils.utility import create_module

from ppocr.utils.eval_utils import eval_run

from ppocr.utils.utility import initial_logger
logger = initial_logger()


def main():
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    char_ops = CharacterOps(config['Global'])
    config['Global']['char_num'] = char_ops.get_char_num()

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    if use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(
            os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    rec_model = create_module(config['Architecture']['function'])(params=config)

    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            eval_loader, eval_outputs = rec_model(mode="eval")
            eval_fetch_list = [v.name for v in eval_outputs]
    eval_prog = eval_prog.clone(for_test=True)

    exe.run(startup_prog)
    pretrain_weights = config['Global']['pretrain_weights']
    if pretrain_weights is not None:
        fluid.load(eval_prog, pretrain_weights)

    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867',\
        'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    eval_data_dir = config['TestReader']['lmdb_sets_dir']
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    eval_data_acc_info = {}
    for eval_data in eval_data_list:
        config['TestReader']['lmdb_sets_dir'] = \
            eval_data_dir + "/" + eval_data
        eval_reader = reader.train_eval_reader(
            config=config, char_ops=char_ops, mode="test")
        eval_loader.set_sample_list_generator(eval_reader, places=place)

        start_time = time.time()
        outs = eval_run(exe, eval_prog, eval_loader, eval_fetch_list, char_ops,
                        "best", "test")
        infer_time = time.time() - start_time
        eval_acc, acc_num, sample_num = outs
        total_forward_time += infer_time
        total_evaluation_data_number += sample_num
        total_correct_number += acc_num
        eval_data_acc_info[eval_data] = outs

    avg_forward_time = total_forward_time / total_evaluation_data_number
    avg_acc = total_correct_number * 1.0 / total_evaluation_data_number
    logger.info('-' * 50)
    strs = ""
    for eval_data in eval_data_list:
        eval_acc, acc_num, sample_num = eval_data_acc_info[eval_data]
        strs += "\n {}, accuracy:{:.6f}".format(eval_data, eval_acc)
    strs += "\n average, accuracy:{:.6f}, time:{:.6f}".format(avg_acc,
                                                              avg_forward_time)
    logger.info(strs)
    logger.info('-' * 50)


if __name__ == '__main__':
    parser = ArgsParser()
    FLAGS = parser.parse_args()
    main()
