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
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..', '..')))
sys.path.append(
    os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))


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
import paddle
from paddle import fluid
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.utils.save_load import init_model, load_params
from ppocr.utils.character import CharacterOps
from ppocr.utils.utility import create_module
from ppocr.data.reader_main import reader_main

from paddleslim.quant import quant_aware, convert
from paddle.fluid.layer_helper import LayerHelper
from eval_utils.eval_det_utils import eval_det_run
from eval_utils.eval_rec_utils import eval_rec_run
from eval_utils.eval_cls_utils import eval_cls_run


def main():
    # 1. quantization configs
    quant_config = {
        # weight quantize type, default is 'channel_wise_abs_max'
        'weight_quantize_type': 'channel_wise_abs_max',
        # activation quantize type, default is 'moving_average_abs_max'
        'activation_quantize_type': 'moving_average_abs_max',
        # weight quantize bit num, default is 8
        'weight_bits': 8,
        # activation quantize bit num, default is 8
        'activation_bits': 8,
        # ops of name_scope in not_quant_pattern list, will not be quantized
        'not_quant_pattern': ['skip_quant'],
        # ops of type in quantize_op_types, will be quantized
        'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
        # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
        'dtype': 'int8',
        # window size for 'range_abs_max' quantization. defaulf is 10000
        'window_size': 10000,
        # The decay coefficient of moving average, default is 0.9
        'moving_rate': 0.9,
    }
    # Run code with static graph mode.
    try:
        paddle.enable_static()
    except:
        pass

    startup_prog, eval_program, place, config, alg_type = program.preprocess()

    feeded_var_names, target_vars, fetches_var_name = program.build_export(
        config, eval_program, startup_prog)

    eval_program = eval_program.clone(for_test=True)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    eval_program = quant_aware(
        eval_program, place, quant_config, scope=None, for_test=True)

    init_model(config, eval_program, exe)

    # 2. Convert the program before save inference program
    #    The dtype of eval_program's weights is float32, but in int8 range.

    eval_program = convert(eval_program, place, quant_config, scope=None)

    eval_fetch_name_list = fetches_var_name
    eval_fetch_varname_list = [v.name for v in target_vars]
    eval_reader = reader_main(config=config, mode="eval")
    quant_info_dict = {'program':eval_program,\
        'reader':eval_reader,\
        'fetch_name_list':eval_fetch_name_list,\
        'fetch_varname_list':eval_fetch_varname_list}

    if alg_type == 'det':
        final_metrics = eval_det_run(exe, config, quant_info_dict, "eval")
    elif alg_type == 'cls':
        final_metrics = eval_cls_run(exe, quant_info_dict)
    else:
        final_metrics = eval_rec_run(exe, config, quant_info_dict, "eval")
    print(final_metrics)

    # 3. Save inference model
    model_path = "./quant_model"
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    fluid.io.save_inference_model(
        dirname=model_path,
        feeded_var_names=feeded_var_names,
        target_vars=target_vars,
        executor=exe,
        main_program=eval_program,
        model_filename=model_path + '/model',
        params_filename=model_path + '/params')
    print("model saved as {}".format(model_path))


if __name__ == '__main__':
    main()
