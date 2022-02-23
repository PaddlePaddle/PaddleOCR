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

from __future__ import print_function
from collections import OrderedDict

from ..framework import Program

__all__ = ['op_freq_statistic']


def op_freq_statistic(program):
    """
    Statistics of Op frequency.

    Args:
        program(Program): The current Program.

    Returns:
        uni_op_freq(dict): the single op frequency.
        adj_2_op_freq(dict): the two adjacent ops frequency.

    Examples:

        >>> import paddle.fluid as fluid
        >>> uni_op_freq, adj_2_op_freq = fluid.contrib.op_freq_statistic(
        >>>        fluid.default_main_program())
        >>> for op_type, op_num in uni_op_freq:
        >>>     print("%s  \t  %d" % (op_type, op_num))
        >>> for op_type, op_num in adj_2_op_freq:
        >>>     print("%s  \t  %d" % (op_type, op_num))

    """

    if not isinstance(program, Program):
        raise TypeError("The input type should be Porgram."
                        "But you passed in %s" % (type(program)))

    uni_op_freq = OrderedDict()
    adj_2_op_freq = OrderedDict()
    op_in_ops = OrderedDict()

    parameters = [p.name for p in program.blocks[0].all_parameters()]

    # get uni_op_freq
    for op in program.global_block().ops:
        had_recorded = False
        for var_name in op.output_arg_names:
            if var_name in parameters:
                continue
            if not had_recorded and uni_op_freq.has_key(op.type):
                uni_op_freq[op.type] += 1
                had_recorded = True
            elif not had_recorded:
                uni_op_freq[op.type] = 1
                had_recorded = True

    # get adj_2_op_freq
    var_gen_op = {}
    for op in program.global_block().ops:
        for var_name in op.input_arg_names:
            if var_name in parameters:
                continue
            if var_gen_op.has_key(var_name):
                assert len(var_gen_op[var_name]) > 0
                if op_in_ops.has_key(op.type):
                    op_in_ops[op.type].append(var_gen_op[var_name][-1])
                else:
                    op_in_ops[op.type] = [var_gen_op[var_name][-1]]
            else:
                print("Var's generate op is not found,%s, %s" %
                      (var_name, op.type))

        for var_name in op.output_arg_names:
            if var_gen_op.has_key(var_name):
                var_gen_op[var_name].append(op.type)
            else:
                var_gen_op[var_name] = [op.type]

    for op, in_ops in op_in_ops.iteritems():
        for in_op in in_ops:
            op_op = in_op + "->" + op
            if adj_2_op_freq.has_key(op_op):
                adj_2_op_freq[op_op] += 1
            else:
                adj_2_op_freq[op_op] = 1

    uni_op_freq = sorted(
        uni_op_freq.items(), key=lambda item: item[1], reverse=True)
    adj_2_op_freq = sorted(
        adj_2_op_freq.items(), key=lambda item: item[1], reverse=True)

    return uni_op_freq, adj_2_op_freq
