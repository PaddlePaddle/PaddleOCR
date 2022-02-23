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
# limitations under the License

from .common import DistributedOperator
from .common import DistributedOperatorImpl
from .common import register_distributed_operator
from .common import register_distributed_operator_impl
from .common import copy_distributed_attr_for_var
from .common import copy_distributed_attr_for_dist_op
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from paddle.fluid import core, unique_name
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from ..process import new_process_group
from ..utils import _get_comm_group


class DistributedEmbedding(DistributedOperator):
    def __init__(self, name):
        super(DistributedEmbedding, self).__init__()
        self._name = name


register_distributed_operator("lookup_table_v2",
                              DistributedEmbedding("embedding"))


# RowParallel
class DistributedEmbeddingImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedEmbeddingImpl, self).__init__()
        self._name = name
        self._forward_implemented = True
        self._backward_implemented = False

    def is_process_mesh_compatible(self, op_dist_attr):
        """ No restriction for now. """
        return True

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        if is_dim_replicate(w_dims_mapping[-2]) or is_dim_shard(w_dims_mapping[
                -1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in ids_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        op_desc = op_dist_attr.get_owner_op().desc
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        out_name = op_desc.output('Out')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        for i in range(len(ids_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [ids_dims_mapping, out_dims_mapping], [i, i])
            if dim_changed:
                changed = True

        dim_changed = compute_compatible_and_update_dim_mapping(
            [w_dims_mapping, out_dims_mapping], [-1, -1])
        if dim_changed:
            changed = True

        return changed

    def forward(self, serial_op):
        def static_handle(dst_block,
                          src_op,
                          op_dist_attr,
                          input_name_mapping,
                          output_name_mapping,
                          rank_id=0):
            assert len(
                input_name_mapping
            ) == 2, "row_parallel_embedding take 2 inputs variable but got {}".format(
                input_name_mapping)
            assert len(
                output_name_mapping
            ) == 1, "row_parallel_embedding take 2 inputs variable but got {}".format(
                output_name_mapping)
            assert len(
                input_name_mapping['Ids']
            ) == 1, "row_parallel_embedding input Ids take 1 variable but got {}".format(
                input_name_mapping['Ids'])
            assert len(
                input_name_mapping['W']
            ) == 1, "row_parallel_embedding input W take 1 variable but got {}".format(
                input_name_mapping['W'])
            assert len(
                output_name_mapping['Out']
            ) == 1, "row_parallel_embedding input Out take 1 variable but got {}".format(
                input_name_mapping['Out'])

            Ids_var = dst_block.var(input_name_mapping['Ids'][0])
            Weight_var = dst_block.var(input_name_mapping['W'][0])
            Out_var = dst_block.var(output_name_mapping['Out'][0])

            # got dist attribute info
            embedding_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name)[0]
            process_mesh_shape = op_dist_attr.get_process_mesh().topology
            process_mesh_group = op_dist_attr.get_process_mesh().process_group

            # caculate embedding offset
            # TODO generalize here, using cartisian product to allow any dimensional mesh shape
            mesh_shape = len(process_mesh_shape)
            assert mesh_shape <= 2, "row_parallel_embedding only support 1 or 2 dimensional process mesh, but got {}".format(
                process_mesh_shape)
            num_partition = process_mesh_shape[embedding_row_dim_mapping]
            # TODO generalize here, support any mesh group 
            if mesh_shape == 1:
                relative_idx = process_mesh_group.index(rank_id)
            else:
                relative_idx = rank_id % num_partition

            per_part_size = Weight_var.shape[0]
            relative_idx = relative_idx * per_part_size

            # TODO caculate ring id 
            model_parallel_axis, process_mesh = op_dist_attr.get_owner_context(
            )._get_model_parallel_info()
            group_ranks = _get_comm_group(process_mesh.process_group,
                                          process_mesh.topology,
                                          model_parallel_axis, rank_id)
            group = new_process_group(group_ranks)

            # append op
            check_variable_and_dtype(Ids_var, 'input', ['int32', 'int64'],
                                     'c_embedding')

            intermediate_var_0 = dst_block.create_var(
                name=unique_name.generate_with_ignorable_key(".".join(
                    ["c_embedding", 'tmp'])),
                dtype=Weight_var.dtype,
                shape=Out_var.shape,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=Out_var.stop_gradient)
            # copy Out_var's dist_attr to intermediate_var_0's dist_attr
            copy_distributed_attr_for_var(op_dist_attr, intermediate_var_0,
                                          Out_var)

            check_variable_and_dtype(
                Out_var, 'tensor',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                'c_allreduce_sum')

            c_embedding_op = dst_block.append_op(
                type='c_embedding',
                inputs={'Ids': [Ids_var],
                        'W': [Weight_var]},
                outputs={'Out': [intermediate_var_0]},
                attrs={"start_index": relative_idx})

            # use_model_parallel
            c_allreduce_sum_op = dst_block.append_op(
                type='c_allreduce_sum',
                inputs={'X': [intermediate_var_0]},
                outputs={'Out': [Out_var]},
                attrs={
                    'ring_id': group.id,
                    'use_calc_stream': True,
                    'use_model_parallel': True,
                })

            # copy serial op's dist_attr to dist op's dist_attr
            copy_distributed_attr_for_dist_op(c_embedding_op, dst_block,
                                              op_dist_attr)
            copy_distributed_attr_for_dist_op(c_allreduce_sum_op, dst_block,
                                              op_dist_attr)

        if in_dygraph_mode():
            raise NotImplementedError(
                "Dist op for [{}] with idx [{}] is NOT implemented yet.".format(
                    "matmul", 0))
        else:
            return static_handle


register_distributed_operator_impl("lookup_table_v2",
                                   DistributedEmbeddingImpl("row_parallel"))
