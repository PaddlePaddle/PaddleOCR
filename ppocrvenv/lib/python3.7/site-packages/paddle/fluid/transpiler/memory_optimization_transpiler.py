#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import logging


def memory_optimize(input_program,
                    skip_opt_set=None,
                    print_log=False,
                    level=0,
                    skip_grads=True):
    """
	:api_attr: Static Graph

    This API is deprecated since 1.6. Please do not use it. The better
    memory optimization strategies are enabled by default.
    """
    logging.warn(
        'Caution! paddle.fluid.memory_optimize() is deprecated '
        'and not maintained any more, since it is not stable!\n'
        'This API would not take any memory optimizations on your Program '
        'now, since we have provided default strategies for you.\n'
        'The newest and stable memory optimization strategies (they are all '
        'enabled by default) are as follows:\n'
        ' 1. Garbage collection strategy, which is enabled by exporting '
        'environment variable FLAGS_eager_delete_tensor_gb=0 (0 is the '
        'default value).\n'
        ' 2. Inplace strategy, which is enabled by setting '
        'build_strategy.enable_inplace=True (True is the default value) '
        'when using CompiledProgram or ParallelExecutor.\n')


def release_memory(input_program, skip_opt_set=None):
    """
	:api_attr: Static Graph

    This API is deprecated since 1.6. Please do not use it. The better
    memory optimization strategies are enabled by default.
    """
    logging.warn('paddle.fluid.release_memory() is deprecated, it would not'
                 ' take any memory release on your program')
