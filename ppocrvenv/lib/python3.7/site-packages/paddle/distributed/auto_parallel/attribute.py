#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import copy
from collections import defaultdict
from paddle.fluid import core


class TensorDistributedAttribute:
    def __init__(self, owner_tensor, owner_context):
        self._owner_tensor = owner_tensor
        self._owner_context = owner_context
        self._process_mesh = None
        self._dims_mapping = None
        self._shard_mask = None
        self._offload_device = None
        self._shape = None
        self._is_annotated = {}
        self._is_parameter = False

    def get_owner_tensor(self):
        return self._owner_tensor

    def get_owner_context(self):
        return self._owner_context

    def get_process_mesh(self):
        return self._process_mesh

    def set_process_mesh(self, process_mesh):
        self._process_mesh = copy.deepcopy(process_mesh)

    def get_dims_mapping(self):
        return self._dims_mapping

    def set_dims_mapping(self, dims_mapping):
        self._dims_mapping = copy.deepcopy(dims_mapping)

    def get_shard_mask(self):
        return self._shard_mask

    def set_shard_mask(self, shard_mask):
        self._shard_mask = copy.deepcopy(shard_mask)

    def get_offload_device(self):
        return self._offload_device

    def set_offload_device(self, offload_device):
        self._offload_device = copy.deepcopy(offload_device)

    def get_shape(self):
        return self._shape

    def set_shape(self, shape):
        self._shape = copy.deepcopy(shape)

    def is_annotated(self, dist_attr_name):
        return self._is_annotated.get(dist_attr_name, False)

    def mark_as_annotated(self, dist_attr_name):
        self._is_annotated[dist_attr_name] = True

    def is_parameter(self):
        return self._is_parameter

    def mark_as_parameter(self):
        self._is_parameter = True

    def is_valid(self):
        if self.get_owner_tensor().type == core.VarDesc.VarType.READER:
            return True
        tensor_shape = self.get_owner_tensor().desc.shape()
        if len(tensor_shape) != len(self.get_dims_mapping()):
            return False
        for i in range(len(self.get_dims_mapping())):
            if self.get_dims_mapping()[i] < -1 or self.get_dims_mapping()[
                    i] >= len(self.get_process_mesh().topology):
                return False
        for i in range(len(self.get_process_mesh().topology)):
            if self.get_dims_mapping().count(i) > 1:
                return False
        return True

    def __str__(self):
        str = "{{tensor name: {}, tensor id: {}".format(
            self.get_owner_tensor().desc.name(),
            self.get_owner_tensor().desc.id())
        if self.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", process_mesh ({}): {}".format(annotated_str,
                                                self.get_process_mesh())

        str += ", is_parameter: {}".format(self._is_parameter)

        if self.is_annotated("dims_mapping"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", dims_mapping ({}): {}".format(annotated_str,
                                                self.get_dims_mapping())

        if self.is_annotated("shard_mask"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", shard_mask ({}): {}".format(annotated_str,
                                              self.get_shard_mask())

        if self.is_annotated("offload_device"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", offload_device ({}): {} }}".format(annotated_str,
                                                     self.get_offload_device())
        return str

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # No need to copy the owner tensor and context
            if k == "_owner_tensor" or k == "_owner_context":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class OperatorDistributedAttribute:
    def __init__(self, owner_op, owner_context):
        self._owner_op = owner_op
        self._owner_context = owner_context
        self._process_mesh = None
        self._dims_mapping = {}
        self._shapes = {}
        self._is_annotated = {}
        self._is_parameters = {}
        self._pipeline_stage = None
        self._impl_idx = None

    def get_owner_op(self):
        return self._owner_op

    def get_owner_context(self):
        return self._owner_context

    def get_process_mesh(self):
        return self._process_mesh

    def set_process_mesh(self, process_mesh):
        self._process_mesh = copy.deepcopy(process_mesh)

    def get_input_dims_mapping(self, name):
        return self._dims_mapping.get("IN_" + name, None)

    def set_input_dims_mapping(self, name, dims_mapping):
        self._dims_mapping["IN_" + name] = copy.deepcopy(dims_mapping)

    def get_output_dims_mapping(self, name):
        return self._dims_mapping.get("OUT_" + name, None)

    def set_output_dims_mapping(self, name, dims_mapping):
        self._dims_mapping["OUT_" + name] = copy.deepcopy(dims_mapping)

    def get_impl_idx(self):
        return self._impl_idx

    def set_impl_idx(self, impl_idx):
        self._impl_idx = impl_idx

    def get_pipeline_stage(self):
        return self._pipeline_stage

    def set_pipeline_stage(self, pipeline_stage):
        self._pipeline_stage = copy.deepcopy(pipeline_stage)

    def get_input_shape(self, name):
        return self._shapes.get("IN_" + name, None)

    def set_input_shape(self, name, shape):
        self._shapes["IN_" + name] = copy.deepcopy(shape)

    def get_output_shape(self, name):
        return self._shapes.get("OUT_" + name, None)

    def set_output_shape(self, name, shape):
        self._shapes["OUT_" + name] = copy.deepcopy(shape)

    def is_annotated(self, attr_name):
        return self._is_annotated.get(attr_name, False)

    def mark_as_annotated(self, attr_name):
        self._is_annotated[attr_name] = True

    def is_annotated_input_dims_mapping(self, name):
        return self._is_annotated.get("IN_" + name, False)

    def mark_as_annotated_input_dims_mapping(self, name):
        self._is_annotated["IN_" + name] = True

    def is_annotated_output_dims_mapping(self, name):
        return self._is_annotated.get("OUT_" + name, False)

    def mark_as_annotated_output_dims_mapping(self, name):
        self._is_annotated["OUT_" + name] = True

    def is_parameter(self, name):
        return self._is_parameters.get(name, False)

    def mark_as_parameter(self, name):
        self._is_parameters[name] = True

    def is_valid(self):
        if "read" in self.get_owner_op().type:
            return True
        for name in self.get_owner_op().desc.input_arg_names():
            dims_mapping = self.get_input_dims_mapping(name)
            shape = self.get_input_shape(name)
            if len(shape) != len(dims_mapping):
                return False
            for i in range(len(dims_mapping)):
                if dims_mapping[i] < -1 or dims_mapping[i] >= len(
                        self.get_process_mesh().topology):
                    return False
            for i in range(len(self.get_process_mesh().topology)):
                if dims_mapping.count(i) > 1:
                    return False
        for name in self.get_owner_op().desc.output_arg_names():
            dims_mapping = self.get_output_dims_mapping(name)
            shape = self.get_output_shape(name)
            if len(shape) != len(dims_mapping):
                return False
            for i in range(len(dims_mapping)):
                if dims_mapping[i] < -1 or dims_mapping[i] >= len(
                        self.get_process_mesh().topology):
                    return False
            for i in range(len(self.get_process_mesh().topology)):
                if dims_mapping.count(i) > 1:
                    return False
        return True

    def __str__(self):
        str = "{{op type: {}, op id: {}".format(self.get_owner_op().desc.type(),
                                                self.get_owner_op().desc.id())

        if self.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", process_mesh ({}): {}".format(annotated_str,
                                                self.get_process_mesh())

        for arg_name in self.get_owner_op().desc.input_arg_names():
            dims_mapping = self.get_input_dims_mapping(arg_name)
            if self.is_annotated_input_dims_mapping(arg_name):
                annotated_str = "annotated"
            else:
                annotated_str = "non-annotated"
            if self.is_parameter(arg_name):
                is_parameter_str = "parameter"
            else:
                is_parameter_str = "non-parameter"
            str += ", {}'s dims_mapping (input, {}, {}): {}".format(
                arg_name, annotated_str, is_parameter_str, dims_mapping)

        for arg_name in self.get_owner_op().desc.output_arg_names():
            dims_mapping = self.get_output_dims_mapping(arg_name)
            if self.is_annotated_output_dims_mapping(arg_name):
                annotated_str = "annotated"
            else:
                annotated_str = "non-annotated"
            if self.is_parameter(arg_name):
                is_parameter_str = "parameter"
            else:
                is_parameter_str = "non-parameter"
            str += ", {}'s dims_mapping (output, {}, {}): {}".format(
                arg_name, annotated_str, is_parameter_str, dims_mapping)

        str += ", pipeline stage: {}".format(self._pipeline_stage)

        str += ", dist_impl idx: {} }}".format(self._impl_idx)

        return str

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # No need to copy the owner op and context
            if k == "_owner_op" or k == "_owner_context":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result
