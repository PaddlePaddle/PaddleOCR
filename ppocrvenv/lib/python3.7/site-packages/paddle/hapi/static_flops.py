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

import copy
import numpy as np
import paddle
from collections import OrderedDict
from paddle.static import Program, program_guard, Variable

__all__ = []


class VarWrapper(object):
    def __init__(self, var, graph):
        assert isinstance(var, Variable)
        assert isinstance(graph, GraphWrapper)
        self._var = var
        self._graph = graph

    def name(self):
        """
        Get the name of the variable.
        """
        return self._var.name

    def shape(self):
        """
        Get the shape of the varibale.
        """
        return self._var.shape


class OpWrapper(object):
    def __init__(self, op, graph):
        assert isinstance(graph, GraphWrapper)
        self._op = op
        self._graph = graph

    def type(self):
        """
        Get the type of this operator.
        """
        return self._op.type

    def inputs(self, name):
        """
        Get all the varibales by the input name.
        """
        if name in self._op.input_names:
            return [
                self._graph.var(var_name) for var_name in self._op.input(name)
            ]
        else:
            return []

    def outputs(self, name):
        """
        Get all the varibales by the output name.
        """
        return [self._graph.var(var_name) for var_name in self._op.output(name)]


class GraphWrapper(object):
    """
    It is a wrapper of paddle.fluid.framework.IrGraph with some special functions
    for paddle slim framework.

    Args:
        program(framework.Program): A program with 
        in_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
        out_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
    """

    def __init__(self, program=None, in_nodes=[], out_nodes=[]):
        """
        """
        super(GraphWrapper, self).__init__()
        self.program = Program() if program is None else program
        self.persistables = {}
        self.teacher_persistables = {}
        for var in self.program.list_vars():
            if var.persistable:
                self.persistables[var.name] = var
        self.compiled_graph = None
        in_nodes = [] if in_nodes is None else in_nodes
        out_nodes = [] if out_nodes is None else out_nodes
        self.in_nodes = OrderedDict(in_nodes)
        self.out_nodes = OrderedDict(out_nodes)
        self._attrs = OrderedDict()

    def ops(self):
        """
        Return all operator nodes included in the graph as a set.
        """
        ops = []
        for block in self.program.blocks:
            for op in block.ops:
                ops.append(OpWrapper(op, self))
        return ops

    def var(self, name):
        """
        Get the variable by variable name.
        """
        for block in self.program.blocks:
            if block.has_var(name):
                return VarWrapper(block.var(name), self)
        return None


def count_convNd(op):
    filter_shape = op.inputs("Filter")[0].shape()
    filter_ops = np.product(filter_shape[1:])
    bias_ops = 1 if len(op.inputs("Bias")) > 0 else 0
    output_numel = np.product(op.outputs("Output")[0].shape()[1:])
    total_ops = output_numel * (filter_ops + bias_ops)
    total_ops = abs(total_ops)
    return total_ops


def count_leaky_relu(op):
    total_ops = np.product(op.outputs("Output")[0].shape()[1:])
    return total_ops


def count_bn(op):
    output_numel = np.product(op.outputs("Y")[0].shape()[1:])
    total_ops = 2 * output_numel
    total_ops = abs(total_ops)
    return total_ops


def count_linear(op):
    total_mul = op.inputs("Y")[0].shape()[0]
    numel = np.product(op.outputs("Out")[0].shape()[1:])
    total_ops = total_mul * numel
    total_ops = abs(total_ops)
    return total_ops


def count_pool2d(op):
    input_shape = op.inputs("X")[0].shape()
    output_shape = op.outputs('Out')[0].shape()
    kernel = np.array(input_shape[2:]) // np.array(output_shape[2:])
    total_add = np.product(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = np.product(output_shape[1:])
    total_ops = kernel_ops * num_elements
    total_ops = abs(total_ops)
    return total_ops


def count_element_op(op):
    input_shape = op.inputs("X")[0].shape()
    total_ops = np.product(input_shape[1:])
    total_ops = abs(total_ops)
    return total_ops


def _graph_flops(graph, detail=False):
    assert isinstance(graph, GraphWrapper)
    flops = 0
    table = Table(["OP Type", 'Param name', "Flops"])
    for op in graph.ops():
        param_name = ''
        if op.type() in ['conv2d', 'depthwise_conv2d']:
            op_flops = count_convNd(op)
            flops += op_flops
            param_name = op.inputs("Filter")[0].name()
        elif op.type() == 'pool2d':
            op_flops = count_pool2d(op)
            flops += op_flops

        elif op.type() in ['mul', 'matmul']:
            op_flops = count_linear(op)
            flops += op_flops
            param_name = op.inputs("Y")[0].name()
        elif op.type() == 'batch_norm':
            op_flops = count_bn(op)
            flops += op_flops
        elif op.type().startswith('element'):
            op_flops = count_element_op(op)
            flops += op_flops
        if op_flops != 0:
            table.add_row([op.type(), param_name, op_flops])
        op_flops = 0
    if detail:
        table.print_table()
    return flops


def static_flops(program, print_detail=False):
    graph = GraphWrapper(program)
    return _graph_flops(graph, detail=print_detail)


class Table(object):
    def __init__(self, table_heads):
        self.table_heads = table_heads
        self.table_len = []
        self.data = []
        self.col_num = len(table_heads)
        for head in table_heads:
            self.table_len.append(len(head))

    def add_row(self, row_str):
        if not isinstance(row_str, list):
            print('The row_str should be a list')
        if len(row_str) != self.col_num:
            print(
                'The length of row data should be equal the length of table heads, but the data: {} is not equal table heads {}'.
                format(len(row_str), self.col_num))
        for i in range(self.col_num):
            if len(str(row_str[i])) > self.table_len[i]:
                self.table_len[i] = len(str(row_str[i]))
        self.data.append(row_str)

    def print_row(self, row):
        string = ''
        for i in range(self.col_num):
            string += '|' + str(row[i]).center(self.table_len[i] + 2)
        string += '|'
        print(string)

    def print_shelf(self):
        string = ''
        for length in self.table_len:
            string += '+'
            string += '-' * (length + 2)
        string += '+'
        print(string)

    def print_table(self):
        self.print_shelf()
        self.print_row(self.table_heads)
        self.print_shelf()
        for data in self.data:
            self.print_row(data)
        self.print_shelf()
