#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define random api
import paddle.fluid as fluid
from paddle.fluid import core

__all__ = []


def seed(seed):
    """

    Sets the seed for global default generator, which manages the random number generation.

    Args:
        seed(int): The random seed to set. It is recommend to set a large int number.

    Returns:
        Generator: The global default generator object.

    Examples:
        .. code-block:: python

            import paddle
            gen = paddle.seed(102)

    """
    #TODO(zhiqiu): 1. remove program.random_seed when all random-related op upgrade
    # 2. support gpu generator by global device 

    seed = int(seed)

    if core.is_compiled_with_cuda():
        for i in range(core.get_cuda_device_count()):
            core.default_cuda_generator(i)._is_init_py = True
            core.default_cuda_generator(i).manual_seed(seed)

    core.default_cpu_generator()._is_init_py = True
    return core.default_cpu_generator().manual_seed(seed)


def get_cuda_rng_state():
    """

    Get random state of cuda generators.

    Args:
        None

    Returns:
        GeneratorState:  object.

    Examples:
        .. code-block:: python

            import paddle
            sts = paddle.get_cuda_rng_state()

    """
    state_list = []
    if core.is_compiled_with_cuda():
        for i in range(core.get_cuda_device_count()):
            state_list.append(core.default_cuda_generator(i).get_state())

    return state_list


def set_cuda_rng_state(state_list):
    """

    Sets generator state for all cuda generators

    Args:
        state_list(list|tuple): The cuda states to set back to cuda generators. state_list is obtained from get_cuda_rng_state().

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            sts = paddle.get_cuda_rng_state()
            paddle.set_cuda_rng_state(sts)

    """
    if core.is_compiled_with_cuda():
        if not len(state_list) == core.get_cuda_device_count():
            raise ValueError(
                "Length of cuda state list shoule be equal to the cuda device count"
            )
        for i in range(core.get_cuda_device_count()):
            core.default_cuda_generator(i).set_state(state_list[i])


def _manual_program_seed(seed):
    """
    Sets global seed for generating random numbers.
  
    NOTE(zhiqiu): This is the original implemention of seed. Keeps it temporally
    since CUDA generator is not developed, so we need it in the unittest.

    Args:
        seed(int): The random seed to set. It is recommend to set a large int number.
    
    Returns:
        None
    """
    fluid.default_main_program().random_seed = seed
    fluid.default_startup_program().random_seed = seed
    program = fluid.Program()
    program.global_seed(seed)


def set_random_seed_generator(name, seed):
    core.set_random_seed_generator(name, seed)


def get_random_seed_generator(name):
    return core.get_random_seed_generator(name)
