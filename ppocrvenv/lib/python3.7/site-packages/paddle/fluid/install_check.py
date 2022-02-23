# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import paddle
from .framework import Program, program_guard, unique_name, cuda_places, cpu_places
from .param_attr import ParamAttr
from .initializer import Constant
from . import layers
from . import backward
from .dygraph import Layer, nn
from . import executor
from . import optimizer
from . import core
from . import compiler
import logging
import numpy as np

__all__ = ['run_check']


class SimpleLayer(Layer):
    def __init__(self, input_size):
        super(SimpleLayer, self).__init__()
        self._linear1 = nn.Linear(
            input_size,
            3,
            param_attr=ParamAttr(initializer=Constant(value=0.1)))

    def forward(self, inputs):
        x = self._linear1(inputs)
        x = layers.reduce_sum(x)
        return x


def run_check():
    """To check whether install is successful
    This func should not be called only if you need to verify installation

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            fluid.install_check.run_check()

            # If installed successfully, output may be
            # Running Verify Fluid Program ... 
            # W0805 04:24:59.496919 35357 device_context.cc:268] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.2, Runtime API Version: 10.1
            # W0805 04:24:59.505594 35357 device_context.cc:276] device: 0, cuDNN Version: 7.6.
            # Your Paddle Fluid works well on SINGLE GPU or CPU.
            # Your Paddle Fluid works well on MUTIPLE GPU or CPU.
            # Your Paddle Fluid is installed successfully! Let's start deep Learning with Paddle Fluid now
    """
    paddle.enable_static()

    print("Running Verify Fluid Program ... ")

    device_list = []
    if core.is_compiled_with_cuda():
        try:
            core.get_cuda_device_count()
        except Exception as e:
            logging.warning(
                "You are using GPU version Paddle Fluid, But Your CUDA Device is not set properly"
                "\n Original Error is {}".format(e))
            return 0
        device_list = cuda_places()
    else:
        device_list = [core.CPUPlace(), core.CPUPlace()]

    np_inp_single = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inp = []
    for i in range(len(device_list)):
        inp.append(np_inp_single)
    np_inp_muti = np.array(inp)
    np_inp_muti = np_inp_muti.reshape(len(device_list), 2, 2)

    def test_parallerl_exe():
        train_prog = Program()
        startup_prog = Program()
        scope = core.Scope()
        with executor.scope_guard(scope):
            with program_guard(train_prog, startup_prog):
                with unique_name.guard():
                    build_strategy = compiler.BuildStrategy()
                    build_strategy.enable_inplace = True
                    inp = layers.data(name="inp", shape=[2, 2])
                    simple_layer = SimpleLayer(input_size=2)
                    out = simple_layer(inp)
                    exe = executor.Executor(
                        core.CUDAPlace(0) if core.is_compiled_with_cuda() and
                        (core.get_cuda_device_count() > 0) else core.CPUPlace())
                    loss = layers.mean(out)
                    loss.persistable = True
                    optimizer.SGD(learning_rate=0.01).minimize(loss)
                    startup_prog.random_seed = 1
                    compiled_prog = compiler.CompiledProgram(
                        train_prog).with_data_parallel(
                            build_strategy=build_strategy,
                            loss_name=loss.name,
                            places=device_list)
                    exe.run(startup_prog)

                    exe.run(compiled_prog,
                            feed={inp.name: np_inp_muti},
                            fetch_list=[loss.name])

    def test_simple_exe():
        train_prog = Program()
        startup_prog = Program()
        scope = core.Scope()
        with executor.scope_guard(scope):
            with program_guard(train_prog, startup_prog):
                with unique_name.guard():
                    inp0 = layers.data(
                        name="inp", shape=[2, 2], append_batch_size=False)
                    simple_layer0 = SimpleLayer(input_size=2)
                    out0 = simple_layer0(inp0)
                    param_grads = backward.append_backward(
                        out0,
                        parameter_list=[simple_layer0._linear1.weight.name])[0]
                    exe0 = executor.Executor(
                        core.CUDAPlace(0) if core.is_compiled_with_cuda() and
                        (core.get_cuda_device_count() > 0) else core.CPUPlace())
                    exe0.run(startup_prog)
                    exe0.run(feed={inp0.name: np_inp_single},
                             fetch_list=[out0.name, param_grads[1].name])

    test_simple_exe()

    print("Your Paddle Fluid works well on SINGLE GPU or CPU.")
    try:
        test_parallerl_exe()
        print("Your Paddle Fluid works well on MUTIPLE GPU or CPU.")
        print(
            "Your Paddle Fluid is installed successfully! Let's start deep Learning with Paddle Fluid now"
        )
    except Exception as e:
        logging.warning(
            "Your Paddle Fluid has some problem with multiple GPU. This may be caused by:"
            "\n 1. There is only 1 or 0 GPU visible on your Device;"
            "\n 2. No.1 or No.2 GPU or both of them are occupied now"
            "\n 3. Wrong installation of NVIDIA-NCCL2, please follow instruction on https://github.com/NVIDIA/nccl-tests "
            "\n to test your NCCL, or reinstall it following https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html"
        )

        print("\n Original Error is: {}".format(e))
        print(
            "Your Paddle Fluid is installed successfully ONLY for SINGLE GPU or CPU! "
            "\n Let's start deep Learning with Paddle Fluid now")

    paddle.disable_static()
