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

import os
import logging
import numpy as np

import paddle

__all__ = []


def _simple_network():
    """
    Define a simple network composed by a single linear layer.
    """
    input = paddle.static.data(
        name="input", shape=[None, 2, 2], dtype="float32")
    weight = paddle.create_parameter(
        shape=[2, 3],
        dtype="float32",
        attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.1)))
    bias = paddle.create_parameter(shape=[3], dtype="float32")
    linear_out = paddle.nn.functional.linear(x=input, weight=weight, bias=bias)
    out = paddle.tensor.sum(linear_out)
    return input, out, weight


def _prepare_data(device_count):
    """
    Prepare feeding data for simple network. The shape is [device_count, 2, 2].

    Args:
        device_count (int): The number of devices.
    """
    # Prepare the feeding data.
    np_input_single = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    if device_count == 1:
        return np_input_single.reshape(device_count, 2, 2)
    else:
        input_list = []
        for i in range(device_count):
            input_list.append(np_input_single)
        np_input_muti = np.array(input_list)
        np_input_muti = np_input_muti.reshape(device_count, 2, 2)
        return np_input_muti


def _is_cuda_available():
    """
    Check whether CUDA is avaiable.
    """
    try:
        assert len(paddle.static.cuda_places()) > 0
        return True
    except Exception as e:
        logging.warning(
            "You are using GPU version PaddlePaddle, but there is no GPU "
            "detected on your machine. Maybe CUDA devices is not set properly."
            "\n Original Error is {}".format(e))
        return False


def _run_dygraph_single(use_cuda):
    """
    Testing the simple network in dygraph mode using one CPU/GPU.

    Args:
        use_cuda (bool): Whether running with CUDA.
    """
    paddle.disable_static()
    if use_cuda:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    weight_attr = paddle.ParamAttr(
        name="weight", initializer=paddle.nn.initializer.Constant(value=0.5))
    bias_attr = paddle.ParamAttr(
        name="bias", initializer=paddle.nn.initializer.Constant(value=1.0))
    linear = paddle.nn.Linear(
        2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
    input_np = _prepare_data(1)
    input_tensor = paddle.to_tensor(input_np)
    linear_out = linear(input_tensor)
    out = paddle.tensor.sum(linear_out)
    out.backward()
    opt = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=linear.parameters())
    opt.step()


def _run_static_single(use_cuda):
    """
    Testing the simple network with executor running directly, using one CPU/GPU.

    Args:
        use_cuda (bool): Whether running with CUDA.
    """
    paddle.enable_static()
    with paddle.static.scope_guard(paddle.static.Scope()):
        train_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        startup_prog.random_seed = 1
        with paddle.static.program_guard(train_prog, startup_prog):
            input, out, weight = _simple_network()
            param_grads = paddle.static.append_backward(
                out, parameter_list=[weight.name])[0]

        exe = paddle.static.Executor(
            paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace())
        exe.run(startup_prog)
        exe.run(train_prog,
                feed={input.name: _prepare_data(1)},
                fetch_list=[out.name, param_grads[1].name])
    paddle.disable_static()


def _run_static_parallel(use_cuda, device_list):
    """
    Testing the simple network in data parallel mode, using multiple CPU/GPU.

    Args:
        use_cuda (bool): Whether running with CUDA.
        device_list (int): The specified devices.
    """
    paddle.enable_static()
    with paddle.static.scope_guard(paddle.static.Scope()):
        train_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(train_prog, startup_prog):
            input, out, _ = _simple_network()
            loss = paddle.tensor.mean(out)
            loss.persistable = True
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

        compiled_prog = paddle.static.CompiledProgram(
            train_prog).with_data_parallel(
                loss_name=loss.name, places=device_list)

        exe = paddle.static.Executor(
            paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace())
        exe.run(startup_prog)
        exe.run(compiled_prog,
                feed={input.name: _prepare_data(len(device_list))},
                fetch_list=[loss.name])
    paddle.disable_static()


def run_check():
    """
    Check whether PaddlePaddle is installed correctly and running successfully
    on your system.

    Examples:
        .. code-block:: python

            import paddle

            paddle.utils.run_check()
            # Running verify PaddlePaddle program ...
            # W1010 07:21:14.972093  8321 device_context.cc:338] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 10.1
            # W1010 07:21:14.979770  8321 device_context.cc:346] device: 0, cuDNN Version: 7.6.
            # PaddlePaddle works well on 1 GPU.
            # PaddlePaddle works well on 8 GPUs.
            # PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
    """

    print("Running verify PaddlePaddle program ... ")

    if paddle.is_compiled_with_cuda():
        use_cuda = _is_cuda_available()
    else:
        use_cuda = False

    if use_cuda:
        device_str = "GPU"
        device_list = paddle.static.cuda_places()
    else:
        device_str = "CPU"
        device_list = paddle.static.cpu_places(device_count=2)
    device_count = len(device_list)

    _run_static_single(use_cuda)
    _run_dygraph_single(use_cuda)
    print("PaddlePaddle works well on 1 {}.".format(device_str))

    try:
        _run_static_parallel(use_cuda, device_list)
        print("PaddlePaddle works well on {} {}s.".format(device_count,
                                                          device_str))
        print(
            "PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now."
        )
    except Exception as e:
        logging.warning(
            "PaddlePaddle meets some problem with {} {}s. This may be caused by:"
            "\n 1. There is not enough GPUs visible on your system"
            "\n 2. Some GPUs are occupied by other process now"
            "\n 3. NVIDIA-NCCL2 is not installed correctly on your system. Please follow instruction on https://github.com/NVIDIA/nccl-tests "
            "\n to test your NCCL, or reinstall it following https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html".
            format(device_count, device_str))

        logging.warning("\n Original Error is: {}".format(e))
        print("PaddlePaddle is installed successfully ONLY for single {}! "
              "Let's start deep learning with PaddlePaddle now.".format(
                  device_str))
