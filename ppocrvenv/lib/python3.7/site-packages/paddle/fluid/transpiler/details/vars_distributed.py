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
from paddle.fluid.framework import Variable


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
            self.origin = self.__create_var_struct(origin_var)
        else:
            self.origin = origin_var

        if isinstance(slice_var, Variable):
            self.slice = self.__create_var_struct(slice_var)
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
    def __create_var_struct(var):
        return VarStruct(var.name, var.shape, var.dtype, var.type,
                         var.lod_level, var.persistable)

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

    def get_distributed_var_by_slice(self, var_name):
        """
        get distributed var by conditions.

        Args:
            var_name(str): slice var name, such as "w.traier0.block1"
        Returns:
            VarDistributed: distributed var.
        """
        for dist_var in self.distributed_vars:
            if dist_var.slice.name == var_name:
                return dist_var
        return None

    @staticmethod
    def equal(var1, var2):
        """
        the two var is equal or not.
        Returns:
            bool: equal will return True else False
        """
        return var1.name == var2.name and \
               var1.type == var2.type and \
               var1.shape == var2.shape and \
               var1.dtype == var2.dtype and \
               var1.lod_level == var2.lod_level and \
               var1.persistable == var2.persistable

    def get_distributed_var_by_origin_and_ep(self, origin_var_name, endpoint):
        """
        get distributed var by conditions.

        Args:
            origin_var_name(str):
            endpoint(str): the parameter endpoint, such as "127.0.0.1:1001"
        Returns:
            VarDistributed: distributed var.
        """
        for dist_var in self.distributed_vars:
            if dist_var.origin.name == origin_var_name and dist_var.endpoint == endpoint:
                return dist_var
        return None

    def get_distributed_vars_by_vtypes(self, vtypes, groupby=False):
        """
        get distributed vars by conditions.

        Args:
            vtype(str|None): distributed var's vtype, such as "Optimizer", "RemotePrefetch"
            groupby(bool|False): group by origin var or not.

        Returns:
            list: distributed var list.
            dict: distributed var map when groupby=True
        """
        vtype_vars = []
        for var in self.distributed_vars:
            if var.vtype in vtypes:
                vtype_vars.append(var)
        if not groupby:
            return vtype_vars

        params_map = {}
        for var in vtype_vars:
            origin_var_name = var.origin.name

            if origin_var_name in params_map.keys():
                optimizers = params_map.get(origin_var_name)
            else:
                optimizers = []
            optimizers.append(var)
            params_map[origin_var_name] = optimizers
        return params_map

    def get_distributed_vars_by_ep(self, endpoint, vtype=None):
        """
        get distributed vars by conditions.

        Args:
            endpoint(str): the parameter server endpoint, such as "127.0.0.1:2001"
            vtype(str|None): distributed var's vtype, such as "Optimizer", "RemotePrefetch"

        Returns:
            list: distributed var list.
        """
        endpoint_vars = []
        for var in self.distributed_vars:
            if var.endpoint == endpoint:
                endpoint_vars.append(var)
        if not vtype:
            return endpoint_vars

        vtype_vars = []
        for var in endpoint_vars:
            if var.vtype == vtype:
                vtype_vars.append(var)
        return vtype_vars

    def overview(self):
        """
        get the overview string about all params on all parameter servers.

        Returns:
            Str: overview string.

        """
        vars_str = []
        for var in self.distributed_vars:
            vars_str.append(str(var))
        return "\n".join(vars_str)
