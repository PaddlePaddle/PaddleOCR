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

LOOKUP_TABLE_TYPE = "lookup_table"


def find_distributed_lookup_table_inputs(program, table_name):
    """
    Find input variable of distribute lookup table in program.
    We only support one distribute table now.
    Args:
    program(Program): given program, locate distributed lookup table
    table_name(str): given table name that is found beforehand
    Returns:
    inputs
    """
    local_vars = program.current_block().vars
    inputs = []
    for op in program.global_block().ops:
        if op.type == LOOKUP_TABLE_TYPE:
            if table_name == op.input("W")[0]:
                inputs.extend([local_vars[name] for name in op.input("Ids")])
    return inputs


def find_distributed_lookup_table_outputs(program, table_name):
    """
    Find output variable of distribute lookup table in program.
    We only support one distribute table now.
    Args:
    program(Program): given program, locate distributed lookup table
    table_name(str): given table name that is found beforehand
    Returns:
    outputs
    """
    local_vars = program.current_block().vars
    outputs = []
    for op in program.global_block().ops:
        if op.type == LOOKUP_TABLE_TYPE:
            if table_name == op.input("W")[0]:
                outputs.extend([local_vars[name] for name in op.output("Out")])
    return outputs


def find_distributed_lookup_table(program):
    """
    Find distribute lookup table in program.
    We only support one distribute table now.
    Args:
    program(Program): given program, locate distributed lookup table
    Returns:
    table_name or None
    """
    table_name = None

    for op in program.global_block().ops:
        if op.type == LOOKUP_TABLE_TYPE:
            if op.attr('is_distributed') is True:
                if table_name is None:
                    table_name = op.input("W")[0]
                if table_name != op.input("W")[0]:
                    raise RuntimeError("all distributed lookup_table_ops"
                                       " should have only one table")
            else:
                if table_name is not None:
                    assert op.input("W")[0] != table_name

    return table_name
