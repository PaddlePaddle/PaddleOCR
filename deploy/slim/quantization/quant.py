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

import tools.program as program
from paddle import fluid
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.data.reader_main import reader_main
from ppocr.utils.save_load import init_model
from paddle.fluid.contrib.model_stat import summary

# quant dependencies
import paddle
import paddle.fluid as fluid
from paddleslim.quant import quant_aware, convert
from paddle.fluid.layer_helper import LayerHelper


def pact(x):
    """
    Process a variable using the pact method you define
    Args:
        x(Tensor): Paddle Tensor, need to be preprocess before quantization
    Returns:
        The processed Tensor x.
    """
    helper = LayerHelper("pact", **locals())
    dtype = 'float32'
    init_thres = 20
    u_param_attr = fluid.ParamAttr(
        name=x.name + '_pact',
        initializer=fluid.initializer.ConstantInitializer(value=init_thres),
        regularizer=fluid.regularizer.L2Decay(0.0001),
        learning_rate=1)
    u_param = helper.create_parameter(attr=u_param_attr, shape=[1], dtype=dtype)
    x = fluid.layers.elementwise_sub(
        x, fluid.layers.relu(fluid.layers.elementwise_sub(x, u_param)))
    x = fluid.layers.elementwise_add(
        x, fluid.layers.relu(fluid.layers.elementwise_sub(-u_param, x)))
    return x


def get_optimizer():
    """
    Build a program using a model and an optimizer
    """
    return fluid.optimizer.AdamOptimizer(0.001)


def main():
    # Run code with static graph mode.
    try:
        paddle.enable_static()
    except:
        pass

    train_build_outputs = program.build(
        config, train_program, startup_program, mode='train')
    train_loader = train_build_outputs[0]
    train_fetch_name_list = train_build_outputs[1]
    train_fetch_varname_list = train_build_outputs[2]
    train_opt_loss_name = train_build_outputs[3]
    model_average = train_build_outputs[-1]

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

    # 2. quantization transform programs (training aware)
    #    Make some quantization transforms in the graph before training and testing.
    #    According to the weight and activation quantization type, the graph will be added
    #    some fake quantize operators and fake dequantize operators.
    act_preprocess_func = pact
    optimizer_func = get_optimizer
    executor = exe

    eval_program = quant_aware(
        eval_program,
        place,
        quant_config,
        scope=None,
        act_preprocess_func=act_preprocess_func,
        optimizer_func=optimizer_func,
        executor=executor,
        for_test=True)
    quant_train_program = quant_aware(
        train_program,
        place,
        quant_config,
        scope=None,
        act_preprocess_func=act_preprocess_func,
        optimizer_func=optimizer_func,
        executor=executor,
        for_test=False)

    # compile program for multi-devices
    train_compile_program = program.create_multi_devices_program(
        quant_train_program, train_opt_loss_name, for_quant=True)

    init_model(config, train_program, exe)

    train_info_dict = {'compile_program':train_compile_program,\
        'train_program':quant_train_program,\
        'reader':train_loader,\
        'fetch_name_list':train_fetch_name_list,\
        'fetch_varname_list':train_fetch_varname_list,\
        'model_average': model_average}

    eval_info_dict = {'program':eval_program,\
        'reader':eval_reader,\
        'fetch_name_list':eval_fetch_name_list,\
        'fetch_varname_list':eval_fetch_varname_list}

    if train_alg_type == 'det':
        program.train_eval_det_run(
            config, exe, train_info_dict, eval_info_dict, is_slim="quant")
    elif train_alg_type == 'rec':
        program.train_eval_rec_run(
            config, exe, train_info_dict, eval_info_dict, is_slim="quant")
    else:
        program.train_eval_cls_run(
            config, exe, train_info_dict, eval_info_dict, is_slim="quant")


if __name__ == '__main__':
    startup_program, train_program, place, config, train_alg_type = program.preprocess(
    )
    main()
