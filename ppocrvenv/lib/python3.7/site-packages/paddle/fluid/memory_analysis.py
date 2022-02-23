# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from . import core
import numpy as np


def get_var_and_memory_size(block, var_name, batch_size=None):
    var = block._find_var_recursive(var_name)
    assert var is not None, "Variable {} cannot be found".format(var_name)
    assert var.type == core.VarDesc.VarType.LOD_TENSOR, "Variable {} is not Tensor".format(
        var_name)
    shape = list(var.shape)
    if not shape:
        return var, 0

    has_none = False
    for i, s in enumerate(shape):
        if s is None or s < 0:
            assert not has_none
            shape[i] = batch_size
            has_none = True
    assert all(
        [s >= 0 for s in shape]), "shape {} is not deterministic".format(shape)
    mem_size = int(np.prod(shape)) * core.size_of_dtype(var.dtype)
    return var, mem_size


def pre_allocate_memory(size, place):
    t = core.LoDTensor()
    t._set_dims([size])
    t._mutable_data(place, core.VarDesc.VarType.INT8)
    del t


# NOTE: does not consider inplace yet. 
def get_max_memory_info(program, batch_size=None):
    assert program.num_blocks == 1, "only support to analysis program with only one block"
    cur_tmp_mem = 0
    max_tmp_mem = 0
    max_persistable_mem = 0
    visited_vars = set()
    alived_vars = []

    block = program.global_block()
    gc_vars = core._get_eager_deletion_vars(program.desc, [])[0]
    for i, op in enumerate(block.ops):
        var_names = op.input_arg_names + op.output_arg_names
        for var_name in var_names:
            if var_name in visited_vars:
                continue
            visited_vars.add(var_name)
            var, mem_size = get_var_and_memory_size(block, var_name, batch_size)
            if var.persistable:
                max_persistable_mem += mem_size
            else:
                cur_tmp_mem += mem_size
                max_tmp_mem = max(max_tmp_mem, cur_tmp_mem)

        cur_gc_vars = gc_vars[i]
        for var_name in var_names:
            if var_name not in cur_gc_vars:
                continue
            _, mem_size = get_var_and_memory_size(block, var_name, batch_size)
            cur_tmp_mem -= mem_size
    return max_tmp_mem, max_persistable_mem
