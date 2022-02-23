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
from paddle.fluid import framework
from paddle.fluid import core
from .attribute import TensorDistributedAttribute
from .attribute import OperatorDistributedAttribute
from .utils import append_distributed_attr_suffix
from .interface import _g_process_mesh_map

# There always exists a default context for user. And user can set it to another one.
DEFAULT_DISTRIBUTED_CONTEXT = None


def get_default_distributed_context():
    global DEFAULT_DISTRIBUTED_CONTEXT
    if DEFAULT_DISTRIBUTED_CONTEXT is None:
        dist_context = DistributedContext()
        set_default_distributed_context(dist_context)
    return DEFAULT_DISTRIBUTED_CONTEXT


def set_default_distributed_context(dist_context):
    global DEFAULT_DISTRIBUTED_CONTEXT
    DEFAULT_DISTRIBUTED_CONTEXT = dist_context


class DistributedContext:
    """
    DistributedContext is used to collect related distributed information for program and graph.
    One auto-parallel run should use its own DistributedContext to avoid interfering other run.
    """

    def __init__(self):
        self._is_initialized_for_program = False
        self._is_initialized_for_graph = False
        self._tensor_distributed_attr_map_for_program = {}
        self._op_distributed_attr_map_for_program = {}
        self._tensor_distributed_attr_map_for_graph = {}
        self._op_distributed_attr_map_for_graph = {}
        # The following is a hard code and will be removed in the future
        self._data_parallel_axis = None
        self._model_parallel_axis = None
        self._process_mesh = _g_process_mesh_map.get(0, None)
        if self._process_mesh is not None:
            if self._process_mesh.ndim == 1:
                self._data_parallel_axis = 0
                self._model_parallel_axis = 0
            else:
                self._data_parallel_axis = 0
                self._model_parallel_axis = 1
        else:
            self._data_parallel_axis = -1
            self._model_parallel_axis = -1

    def is_initialized_for_program(self):
        return self._is_initialized_for_program

    def is_initialized_for_graph(self):
        return self._is_initialized_for_graph

    def get_tensor_distributed_attr_for_program(self, tensor):
        tensor_id = tensor.desc.id()
        tensor_dist_attr = self._tensor_distributed_attr_map_for_program.get(
            tensor_id, None)
        return tensor_dist_attr

    def set_tensor_distributed_attr_for_program(self, tensor, tensor_dist_attr):
        tensor_id = tensor.desc.id()
        self._tensor_distributed_attr_map_for_program[
            tensor_id] = tensor_dist_attr

    def get_op_distributed_attr_for_program(self, op):
        op_id = op.desc.id()
        op_dist_attr = self._op_distributed_attr_map_for_program.get(op_id,
                                                                     None)
        return op_dist_attr

    def set_op_distributed_attr_for_program(self, op, op_dist_attr):
        op_id = op.desc.id()
        self._op_distributed_attr_map_for_program[op_id] = op_dist_attr

    def get_tensor_distributed_attr_for_graph(self, tensor_node):
        tensor_node_id = tensor_node.id()
        tensor_dist_attr = self._tensor_distributed_attr_map_for_graph.get(
            tensor_node_id, None)
        return tensor_dist_attr

    def set_tensor_distributed_attr_for_graph(self, tensor_node,
                                              tensor_dist_attr):
        tensor_node_id = tensor_node.id()
        self._tensor_distributed_attr_map_for_graph[
            tensor_node_id] = tensor_dist_attr

    def get_op_distributed_attr_for_graph(self, op_node):
        op_node_id = op_node.id()
        op_dist_attr = self._op_distributed_attr_map_for_graph.get(op_node_id,
                                                                   None)
        return op_dist_attr

    def set_op_distributed_attr_for_graph(self, op_node, op_dist_attr):
        op_node_id = op_node.id()
        self._op_distributed_attr_map_for_graph[op_node_id] = op_dist_attr

    def set_process_mesh(self, process_mesh):
        self._process_mesh = process_mesh
        if self._process_mesh is not None:
            if self._process_mesh.ndim == 1:
                self._data_parallel_axis = 0
                self._model_parallel_axis = 0
            else:
                self._data_parallel_axis = 0
                self._model_parallel_axis = 1
        else:
            self._data_parallel_axis = -1
            self._model_parallel_axis = -1

    def initialize_distributed_attr_for_program(self, program):
        if self._is_initialized_for_program:
            return
        for block in program.blocks:
            for tensor in block.vars.values():
                # Since only tensors have distributed attributes, it's better to make sure var is a tensor
                tensor_dist_attr = self.get_tensor_distributed_attr_for_program(
                    tensor)
                if tensor_dist_attr is None:
                    tensor_dist_attr = TensorDistributedAttribute(tensor, self)
                    self._copy_distributed_attr_from_tensor_desc(
                        tensor.desc, tensor_dist_attr)
                    self.set_tensor_distributed_attr_for_program(
                        tensor, tensor_dist_attr)
                if tensor.type == core.VarDesc.VarType.READER:
                    tensor_dist_attr.set_shape([])
                else:
                    tensor_dist_attr.set_shape(tensor.desc.shape())
                if tensor_dist_attr.get_process_mesh() is not None:
                    tensor_dist_attr.mark_as_annotated("process_mesh")
                if tensor_dist_attr.get_dims_mapping() is None:
                    tensor_dims_mapping = [
                        -1 for _ in range(len(tensor_dist_attr.get_shape()))
                    ]
                    tensor_dist_attr.set_dims_mapping(tensor_dims_mapping)
                else:
                    tensor_dist_attr.mark_as_annotated("dims_mapping")
                if isinstance(tensor, framework.Parameter):
                    tensor_dist_attr.mark_as_parameter()
            for op in block.ops:
                op_dist_attr = self.get_op_distributed_attr_for_program(op)
                if op_dist_attr is None:
                    op_dist_attr = OperatorDistributedAttribute(op, self)
                    self._copy_distributed_attr_from_op_desc(op.desc,
                                                             op_dist_attr)
                    self.set_op_distributed_attr_for_program(op, op_dist_attr)
                # Default distributed implementation for all operators
                # This will be updated during the completion prcess
                op_dist_attr.set_impl_idx(-2)
                if op_dist_attr.get_process_mesh() is not None:
                    op_dist_attr.mark_as_annotated("process_mesh")
                for tensor_name in op.input_arg_names:
                    # There may be a better way to find the tensor by name
                    if op.type == "create_py_reader" \
                        or tensor.type == core.VarDesc.VarType.READER:
                        op_dist_attr.set_input_shape(tensor_name, [])
                    else:
                        tensor = op.block._var_recursive(tensor_name)
                        op_dist_attr.set_input_shape(tensor_name,
                                                     tensor.desc.shape())
                    if op_dist_attr.get_input_dims_mapping(tensor_name) is None:
                        tensor_dims_mapping = [
                            -1
                            for _ in range(
                                len(op_dist_attr.get_input_shape(tensor_name)))
                        ]
                        op_dist_attr.set_input_dims_mapping(tensor_name,
                                                            tensor_dims_mapping)
                    else:
                        op_dist_attr.mark_as_annotated_input_dims_mapping(
                            tensor_name)
                    if isinstance(tensor, framework.Parameter):
                        op_dist_attr.mark_as_parameter(tensor_name)
                for tensor_name in op.output_arg_names:
                    tensor = op.block._var_recursive(tensor_name)
                    if tensor.type == core.VarDesc.VarType.READER:
                        op_dist_attr.set_output_shape(tensor_name, [])
                    else:
                        op_dist_attr.set_output_shape(tensor_name,
                                                      tensor.desc.shape())
                    if op_dist_attr.get_output_dims_mapping(
                            tensor_name) is None:
                        tensor_dims_mapping = [
                            -1
                            for _ in range(
                                len(
                                    op_dist_attr.get_output_shape(tensor_name)))
                        ]
                        op_dist_attr.set_output_dims_mapping(
                            tensor_name, tensor_dims_mapping)
                    else:
                        op_dist_attr.mark_as_annotated_output_dims_mapping(
                            tensor_name)
                    if isinstance(tensor, framework.Parameter):
                        op_dist_attr.mark_as_parameter(tensor_name)
        self._is_initialized_for_program = True

    def finalize_distributed_attr_for_program(self, program):
        assert self._is_initialized_for_program, \
            "The program must initialize its distributed attribute before finalization."
        for block in program.blocks:
            for tensor in block.vars.values():
                tensor_dist_attr = self.get_tensor_distributed_attr_for_program(
                    tensor)
                if tensor_dist_attr is not None:
                    self._store_distributed_attr_to_tensor_desc(
                        tensor.desc, tensor_dist_attr)
            for op in block.ops:
                op_dist_attr = self.get_op_distributed_attr_for_program(op)
                if op_dist_attr is not None:
                    self._store_distributed_attr_to_op_desc(op.desc,
                                                            op_dist_attr)

    def _copy_distributed_attr_from_tensor_desc(self, desc, dist_attr):
        from paddle.distributed.auto_parallel.interface import _g_process_mesh_map
        attr_name = append_distributed_attr_suffix("mesh_id")
        if desc.has_attr(attr_name):
            mesh_id = desc.attr(attr_name)
            process_mesh = _g_process_mesh_map[mesh_id]
            copied_process_mesh = copy.deepcopy(process_mesh)
            dist_attr.set_process_mesh(copied_process_mesh)
        attr_name = append_distributed_attr_suffix("dim_mapping")
        if desc.has_attr(attr_name):
            dims_mapping = desc.attr(attr_name)
            copied_dims_mapping = copy.deepcopy(dims_mapping)
            dist_attr.set_dims_mapping(copied_dims_mapping)
        attr_name = append_distributed_attr_suffix("mask")
        if desc.has_attr(attr_name):
            shard_mask = desc.attr(attr_name)
            copied_shard_mask = copy.deepcopy(shard_mask)
            dist_attr.set_shard_mask(copied_shard_mask)
        attr_name = append_distributed_attr_suffix("offload_device")
        if desc.has_attr(attr_name):
            offload_device = desc.attr(attr_name)
            copied_offload_device = copy.deepcopy(offload_device)
            dist_attr.set_offload_device(copied_offload_device)

    def _copy_distributed_attr_from_op_desc(self, desc, dist_attr):
        from paddle.distributed.auto_parallel.interface import _g_process_mesh_map
        attr_name = append_distributed_attr_suffix("mesh_id")
        if desc.has_attr(attr_name):
            mesh_id = desc.attr(attr_name)
            process_mesh = _g_process_mesh_map[mesh_id]
            copied_process_mesh = copy.deepcopy(process_mesh)
            dist_attr.set_process_mesh(copied_process_mesh)
        for tensor_name in desc.input_arg_names():
            attr_name = append_distributed_attr_suffix("IN_" + tensor_name)
            if desc.has_attr(attr_name):
                dims_mapping = desc.attr(attr_name)
                copied_dims_mapping = copy.deepcopy(dims_mapping)
                dist_attr.set_input_dims_mapping(tensor_name,
                                                 copied_dims_mapping)
        for tensor_name in desc.output_arg_names():
            attr_name = append_distributed_attr_suffix("OUT_" + tensor_name)
            if desc.has_attr(attr_name):
                dims_mapping = desc.attr(attr_name)
                copied_dims_mapping = copy.deepcopy(dims_mapping)
                dist_attr.set_input_dims_mapping(tensor_name,
                                                 copied_dims_mapping)
        attr_name = append_distributed_attr_suffix("pipeline_stage")
        if desc.has_attr(attr_name):
            pipeline_stage = desc.attr(attr_name)
            copied_pipeline_stage = copy.deepcopy(pipeline_stage)
            dist_attr.set_pipeline_stage(copied_pipeline_stage)

    def _store_distributed_attr_to_tensor_desc(self, desc, dist_attr):
        process_mesh = dist_attr.get_process_mesh()
        if process_mesh is not None:
            attr_name = append_distributed_attr_suffix("mesh_id")
            desc._set_attr(attr_name, process_mesh._id)
        dims_mapping = dist_attr.get_dims_mapping()
        if dims_mapping is not None:
            attr_name = append_distributed_attr_suffix("dim_mapping")
            desc._set_attr(attr_name, dims_mapping)
        shard_mask = dist_attr.get_shard_mask()
        if shard_mask is not None:
            attr_name = append_distributed_attr_suffix("mask")
            desc._set_attr(attr_name, shard_mask)
        offload_device = dist_attr.get_offload_device()
        if offload_device is not None:
            attr_name = append_distributed_attr_suffix("offload_device")
            desc._set_attr(attr_name, offload_device)

    def _store_distributed_attr_to_op_desc(self, desc, dist_attr):
        process_mesh = dist_attr.get_process_mesh()
        if process_mesh is not None:
            attr_name = append_distributed_attr_suffix("mesh_id")
            desc._set_attr(attr_name, process_mesh._id)
        for tensor_name in desc.input_arg_names():
            dims_mapping = dist_attr.get_input_dims_mapping(tensor_name)
            if dims_mapping is not None:
                attr_name = append_distributed_attr_suffix("IN_" + tensor_name)
                desc._set_attr(attr_name, dims_mapping)
        for tensor_name in desc.output_arg_names():
            dims_mapping = dist_attr.get_output_dims_mapping(tensor_name)
            if dims_mapping is not None:
                attr_name = append_distributed_attr_suffix("OUT_" + tensor_name)
                desc._set_attr(attr_name, dims_mapping)
        pipeline_stage = dist_attr.get_pipeline_stage()
        if pipeline_stage is not None:
            attr_name = append_distributed_attr_suffix("pipeline_stage")
            desc._set_attr(attr_name, pipeline_stage)

    def initialize_distributed_attr_for_graph(self, graph):
        assert self._is_initialized_for_program, \
            "The program must initialize its distributed attribute before its graph."
        if self._is_initialized_for_graph:
            return
        all_nodes = graph.all_nodes()
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_desc = node.var()
                tensor_id = tensor_desc.id()
                tensor_dist_attr = self._tensor_distributed_attr_map_for_program[
                    tensor_id]
                assert tensor_dist_attr is not None, \
                    "Tensor must have a distributed attribute after the initialization for program."
                new_tensor_dist_attr = copy.deepcopy(tensor_dist_attr)
                self.set_tensor_distributed_attr_for_graph(node,
                                                           new_tensor_dist_attr)

            if node.is_op() and node.op() is not None:
                op_desc = node.op()
                op_id = op_desc.id()
                op_dist_attr = self._op_distributed_attr_map_for_program[op_id]
                assert op_dist_attr is not None, \
                    "Operator must have a distributed attribute after the initialization for program."
                new_op_dist_attr = copy.deepcopy(op_dist_attr)
                self.set_op_distributed_attr_for_graph(node, new_op_dist_attr)
        self._is_initialized_for_graph = True

    def clear_distributed_attr_for_program(self):
        self._tensor_distributed_attr_map_for_program.clear()
        self._op_distributed_attr_map_for_program.clear()

    def clear_distributed_attr_for_graph(self):
        self._tensor_distributed_attr_map_for_graph.clear()
        self._op_distributed_attr_map_for_graph.clear()

    def copy_distribute_attr_from_graph_to_program(self, graph, program):
        assert self._is_initialized_for_program and self._is_initialized_for_graph, \
            "The distribute attributes must be initialized both in its program and graph"
        updated_tensors = {}
        all_nodes = graph.all_nodes()
        for node in all_nodes:
            if node.is_var() and node.var() is not None:
                tensor_desc = node.var()
                tensor_id = tensor_desc.id()
                updated = updated_tensors.get(tensor_desc.name(), False)
                # If a var has multiples var nodes in graph, only use the first one for now
                if not updated:
                    tensor_dist_attr = self.get_tensor_distributed_attr_for_graph(
                        node)
                    new_tensor_dist_attr = copy.deepcopy(tensor_dist_attr)
                    self._tensor_distributed_attr_map_for_program[
                        tensor_id] = new_tensor_dist_attr
                    updated_tensors[tensor_desc.name()] = True
            if node.is_op() and node.op() is not None:
                op_desc = node.op()
                op_id = op_desc.id()
                op_dist_attr = self.get_op_distributed_attr_for_graph(node)
                new_op_dist_attr = copy.deepcopy(op_dist_attr)
                self._op_distributed_attr_map_for_program[
                    op_id] = new_op_dist_attr

    def amend_distributed_attr_for_program(self):
        for attr in self._tensor_distributed_attr_map_for_program.values():
            assert attr.is_valid(), \
                "Tensor's distributed attribute {} is not valid".format(attr)
            tensor_shape = attr.get_shape()
            dims_mapping = attr.get_dims_mapping()
            process_mesh_shape = attr.get_process_mesh().topology
            # If the dimension of tensor is less than the sharding dimension of process mesh,
            # we just amend the dimension mapping to -1. (Is this really OK?)
            for i in range(len(tensor_shape)):
                if dims_mapping[i] != -1 and tensor_shape[i] > 0 \
                    and process_mesh_shape[dims_mapping[i]] > tensor_shape[i]:
                    dims_mapping[i] = -1

        for attr in self._op_distributed_attr_map_for_program.values():
            assert attr.is_valid(), \
                "Operator's distributed attribute {} is not valid".format(attr)
            for arg_name in attr.get_owner_op().desc.input_arg_names():
                tensor_shape = attr.get_input_shape(arg_name)
                dims_mapping = attr.get_input_dims_mapping(arg_name)
                process_mesh_shape = attr.get_process_mesh().topology
                # If the dimension of tensor is less than the sharding dimension of process mesh,
                # we just amend the dimension mapping to -1. (Is this really OK?)
                for i in range(len(tensor_shape)):
                    if dims_mapping[i] != -1 and tensor_shape[i] > 0 \
                        and process_mesh_shape[dims_mapping[i]] > tensor_shape[i]:
                        dims_mapping[i] = -1

            for arg_name in attr.get_owner_op().desc.output_arg_names():
                tensor_shape = attr.get_output_shape(arg_name)
                dims_mapping = attr.get_output_dims_mapping(arg_name)
                process_mesh_shape = attr.get_process_mesh().topology
                # If the dimension of tensor is less than the sharding dimension of process mesh,
                # we just amend the dimension mapping to -1. (Is this really OK?)
                for i in range(len(tensor_shape)):
                    if dims_mapping[i] != -1 and tensor_shape[i] > 0 \
                        and process_mesh_shape[dims_mapping[i]] > tensor_shape[i]:
                        dims_mapping[i] = -1

    def _get_data_parallel_info(self):
        # This function is a hard code, and will be obsoleted in the future
        return self._data_parallel_axis, self._process_mesh

    def _get_model_parallel_info(self):
        # This function is a hard code, and will be obsoleted in the future
        return self._model_parallel_axis, self._process_mesh
