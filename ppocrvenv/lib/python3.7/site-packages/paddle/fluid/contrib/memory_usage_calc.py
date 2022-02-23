#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
"""
This module provides a memory usage calculate function for user.
The purpose of this API is to allow users to estimate memory usage of
a program under a special batch size, then user can set appropriate
batch size to fully utilize a GPU.

This API is still under active development and may change drastically.
"""

from __future__ import print_function

import six

from .. import core
from ..framework import Program, Variable

__all__ = ['memory_usage']

dtype_to_size = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    core.VarDesc.VarType.UINT8: 1,
}

DEBUG = False


def memory_usage(program, batch_size):
    r"""
    Get the estimate memory usage of program with input batch size.

    Args:
        program(Program): The current Program.
        batch_size(int): The current input data batch_size.

    Returns:
        min_total_memory(float): the estimate memory usage lower bound.
        max_total_memory(float): the estimate memory usage upper bound.
        unit_str(string): the unit of estimate usage result.

    Examples:

        >>> import paddle.fluid as fluid
        >>> lower_usage, upper_usage, unit = fluid.contrib.memory_usage(
                fluid.default_main_program(), batch_size=10)
        >>> print "memory usage is about %.3f - %.3f %s" % \
                (lower_usage, upper_usage, unit)

    """

    # Parameters check
    if not isinstance(program, Program):
        raise TypeError(
            "Calculating Memory Usage requires Program as its Parameter."
            "But you passed in %s" % (type(program)))
    if batch_size <= 0:
        raise ValueError("The batch size need to be positive.")

    # Get the var_name list of first block and calculate
    total_memory = 0.0
    processed_var_names = set(["@EMPTY@"])
    for op in program.global_block().ops:
        for var_name in op.output_arg_names:
            if var_name in processed_var_names:
                continue
            processed_var_names.add(var_name)
            var = program.global_block().vars[var_name]
            if var.desc.type() != core.VarDesc.VarType.LOD_TENSOR:
                continue

            data_count = 1
            neg_dim_count = 0
            for x in var.shape:
                if x < 0:
                    if neg_dim_count >= 1:
                        raise ValueError(
                            "Var %s has more than one negative dim." %
                            (var_name))
                    neg_dim_count += 1
                    data_count *= batch_size * (-x)
                else:
                    data_count *= x
            var_memory = data_count * dtype_to_size[var.dtype]
            if DEBUG:
                print("%s memory usage: %d" % (var.name, var_memory))
            total_memory += var_memory
    if DEBUG:
        print("total memory usage: %.2f" % (total_memory))

    # Convert appropriate unit
    unit_str = "B"
    if total_memory > 1024:
        total_memory /= 1024
        unit_str = "KB"
        if total_memory > 1024:
            total_memory /= 1024
            unit_str = "MB"

    # Append extra memory consumption (5% - 10%)
    min_total_memory = total_memory * 1.05
    max_total_memory = total_memory * 1.1

    return min_total_memory, max_total_memory, unit_str
