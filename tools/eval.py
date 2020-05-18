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
from ppocr.data.reader_main import reader_main
from ppocr.utils.save_load import init_model
from eval_utils.eval_det_utils import eval_det_run
from eval_utils.eval_rec_utils import test_rec_benchmark
from eval_utils.eval_rec_utils import eval_rec_run
from ppocr.utils.character import CharacterOps


def main():
    config = program.load_config(FLAGS.config)
    program.merge_config(FLAGS.opt)
    logger.info(config)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    program.check_gpu(True)

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

    if alg in ['EAST', 'DB']:
        eval_reader = reader_main(config=config, mode="eval")
        eval_info_dict = {'program':eval_program,\
            'reader':eval_reader,\
            'fetch_name_list':eval_fetch_name_list,\
            'fetch_varname_list':eval_fetch_varname_list}
        metrics = eval_det_run(exe, config, eval_info_dict, "eval")
    else:
        reader_type = config['Global']['reader_yml']
        if "benchmark" not in reader_type:
            eval_reader = reader_main(config=config, mode="eval")
            eval_info_dict = {'program': eval_program, \
                              'reader': eval_reader, \
                              'fetch_name_list': eval_fetch_name_list, \
                              'fetch_varname_list': eval_fetch_varname_list}
            metrics = eval_rec_run(exe, config, eval_info_dict, "eval")
            print("Eval result:", metrics)
        else:
            eval_info_dict = {'program':eval_program,\
                'fetch_name_list':eval_fetch_name_list,\
                'fetch_varname_list':eval_fetch_varname_list}
            test_rec_benchmark(exe, config, eval_info_dict)


if __name__ == '__main__':
    parser = program.ArgsParser()
    FLAGS = parser.parse_args()
    main()
