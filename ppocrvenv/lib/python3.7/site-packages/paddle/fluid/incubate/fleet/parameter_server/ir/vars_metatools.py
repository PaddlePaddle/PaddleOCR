# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from functools import reduce

from paddle.fluid.framework import Variable
from paddle.fluid import core

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


class VarBlock:
    def __init__(self, varname, offset, size):
        self.varname = varname
        # NOTE: real offset is offset * size
        self.offset = offset
        self.size = size

    def __str__(self):
        return "%s:%d:%d" % (self.varname, self.offset, self.size)


def create_var_struct(var):
    if var.type == core.VarDesc.VarType.SELECTED_ROWS:
        lod_level = None
    elif var.type == core.VarDesc.VarType.LOD_TENSOR:
        lod_level = var.lod_level
    else:
        raise ValueError("can only support SELECTED_ROWS/LOD_TENSOR now")

    return VarStruct(var.name, var.shape, var.dtype, var.type, lod_level,
                     var.persistable)


class VarStruct(object):
    """
    record part properties of a Variable in python.
    """

    def __init__(self, name, shape, dtype, type, lod_level, persistable):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.type = type
        self.lod_level = lod_level
        self.persistable = persistable
        self.m_size = 1
        self.m_size = reduce(lambda x, y: x * y, shape)
        self.m_size *= dtype_to_size[dtype]

    def __str__(self):
        return "N: {}, S: {}, D: {}, T: {}, LL: {}, P: {}, M: {}".format(
            self.name, self.shape, self.dtype, self.type, self.lod_level,
            self.persistable, self.m_size)


class VarDistributed(object):
    """
    a class to record the var distributed on parameter servers.
    the class will record the relationship between origin var and slice var.
    the slice var's properties, such as type/shape/offset/endpoint.
    """

    def __init__(self,
                 origin_var,
                 slice_var,
                 is_slice=None,
                 block_id=None,
                 offset=None,
                 vtype=None,
                 endpoint=None):
        """
        Args:
            origin_var(Variable|VarStruct): origin var properties
            slice_var(Variable|VarStruct): slice var properties
            is_slice(bool|None): slice or not, slice_var=True/False and its block size > 8192 are the judgement standard.
            block_id(int|None): the number about the slice var.
            offset(int|None): if the slice var is sliced, offset is the numel before the var.
            vtype(str|None): a tag, such as Optimizer/Param/RemoteProfetch.
            endpoint(str|None): which parameter the slice var on, such as "127.0.0.1:1001"
        """

        if isinstance(origin_var, Variable):
            self.origin = create_var_struct(origin_var)
        else:
            self.origin = origin_var

        if isinstance(slice_var, Variable):
            self.slice = create_var_struct(slice_var)
        else:
            self.slice = slice_var

        if self.equal(self.origin, self.slice):
            self.is_slice = False
            self.block_id = 0
            self.offset = 0
        else:
            self.is_slice = True
            self.block_id = 0
            self.offset = 0

        if is_slice is not None:
            self.is_slice = is_slice
        if block_id is not None:
            self.block_id = block_id
        if offset is not None:
            self.offset = offset

        self.vtype = vtype
        self.endpoint = endpoint

    @staticmethod
    def equal(var1, var2):
        """
        the two var is equal or not.
        Returns:
            bool: equal will return True else False
        """
        assert isinstance(var1, VarStruct) and isinstance(var2, VarStruct)

        return var1.name == var2.name and \
               var1.type == var2.type and \
               var1.shape == var2.shape and \
               var1.dtype == var2.dtype and \
               var1.lod_level == var2.lod_level and \
               var1.persistable == var2.persistable

    def __str__(self):
        origin_var_str = "{name} : fluid.{type}.shape{shape}.astype({dtype})". \
            format(i="{", e="}", name=self.origin.name, type=self.origin.type,
                   shape=self.origin.shape, dtype=self.origin.dtype)

        slice_var_str = "{name} : fluid.{type}.shape{shape}.astype({dtype})" \
                        ".slice({is_slice}).block({block_id}).offset({offset})". \
            format(i="{", e="}", name=self.slice.name, type=self.slice.type,
                   shape=self.slice.shape, dtype=self.slice.dtype,
                   is_slice=self.is_slice, block_id=self.block_id, offset=self.offset)

        return "var owned: {}, origin var: ( {} ), slice var: ( {} ), endpoint: {} ".format(
            self.vtype, origin_var_str, slice_var_str, self.endpoint)


class VarsDistributed(object):
    """
    a gather about VarDistributed with many methods to find distributed vars.
    through the class, we can get overview about the distributed parameters on parameter servers.
    this class may centralized and convenient for developer to manage and get variable's distribute.
    other module can also use this to find variables such io.py.
    """

    def __init__(self):
        self.distributed_vars = []

    def add_distributed_var(self,
                            origin_var,
                            slice_var,
                            is_slice=None,
                            block_id=None,
                            offset=None,
                            vtype=None,
                            endpoint=None):
        """
        add distributed var in this.

        Args:
            origin_var(Variable|VarStruct): origin var properties
            slice_var(Variable|VarStruct): slice var properties
            is_slice(bool|None): slice or not, slice_var=True/False and its block size > 8192 are the judgement standard.
            block_id(int|None): the number about the slice var.
            offset(int|None): if the slice var is sliced, offset is the numel before the var.
            vtype(str|None): a tag, such as Optimizer/Param/RemoteProfetch.
            endpoint(str|None): which parameter the slice var on, such as "127.0.0.1:1001"
        Returns:
            None
        """
        self.distributed_vars.append(
            VarDistributed(origin_var, slice_var, is_slice, block_id, offset,
                           vtype, endpoint))
