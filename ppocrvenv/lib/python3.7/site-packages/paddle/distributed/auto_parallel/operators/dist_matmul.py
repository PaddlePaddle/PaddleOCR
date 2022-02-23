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


def _update_dims_mapping_for_matmul(op_dist_attr):
    changed = False
    op_desc = op_dist_attr.get_owner_op().desc
    x_name = op_desc.input('X')[0]
    y_name = op_desc.input('Y')[0]
    out_name = op_desc.output('Out')[0]
    x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
    y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
    out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
    x_dims_mapping_len = len(x_dims_mapping)
    y_dims_mapping_len = len(y_dims_mapping)
    out_dims_mapping_len = len(out_dims_mapping)

    # Add dim mapping to Make sure the length dims_mapping be at least 2
    if x_dims_mapping_len == 1:
        x_dims_mapping.insert(0, -1)
    if y_dims_mapping_len == 1:
        y_dims_mapping.insert(1, -1)

    # Deal with dim > 2 and take care of broadcasting 
    if out_dims_mapping_len > 2:
        broadcast_x_dims_mapping = []
        broadcast_y_dims_mapping = []
        broadcast_out_dims_mapping = []

        for i in range(out_dims_mapping_len - x_dims_mapping_len):
            broadcast_x_dims_mapping.append(out_dims_mapping[i])
        for i in range(x_dims_mapping_len - 2):
            broadcast_x_dims_mapping.append(x_dims_mapping[i])

        for i in range(out_dims_mapping_len - y_dims_mapping_len):
            broadcast_y_dims_mapping.append(out_dims_mapping[i])
        for i in range(y_dims_mapping_len - 2):
            broadcast_y_dims_mapping.append(y_dims_mapping[i])

        for i in range(out_dims_mapping_len - 2):
            broadcast_out_dims_mapping.append(out_dims_mapping[i])

        compatible_dims_mapping = compute_compatible_dims_mapping([
            broadcast_x_dims_mapping, broadcast_y_dims_mapping,
            broadcast_out_dims_mapping
        ])
        assert compatible_dims_mapping is not None, "There is no compatible dim mapping."

        for i in range(x_dims_mapping_len - 2):
            new_idx = i + (out_dims_mapping_len - x_dims_mapping_len)
            if x_dims_mapping[i] != compatible_dims_mapping[new_idx]:
                x_dims_mapping[i] = compatible_dims_mapping[new_idx]
                changed = True

        for i in range(y_dims_mapping_len - 2):
            new_idx = i + (out_dims_mapping_len - y_dims_mapping_len)
            if y_dims_mapping[i] != compatible_dims_mapping[new_idx]:
                y_dims_mapping[i] = compatible_dims_mapping[new_idx]
                changed = True

        for i in range(out_dims_mapping_len - 2):
            if out_dims_mapping[i] != compatible_dims_mapping[i]:
                out_dims_mapping[i] = compatible_dims_mapping[i]
                changed = True

    # The following which uses negative index can be work 
    # when len(out_dims_mapping) > 2 and len(out_dims_mapping) <=2
    dim_changed = compute_compatible_and_update_dim_mapping(
        [x_dims_mapping, y_dims_mapping], [-1, -2])
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
        [x_dims_mapping, out_dims_mapping], [-2, -2])
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
        [y_dims_mapping, out_dims_mapping], [-1, -1])
    if dim_changed:
        changed = True

    # Remove unnecessary dim mapping to make sure the lenght of dims_mapping is same as its tensor
    if x_dims_mapping_len == 1:
        x_dims_mapping.pop(0)
    if y_dims_mapping_len == 1:
        y_dims_mapping.pop(1)

    assert len(x_dims_mapping) == x_dims_mapping_len
    assert len(y_dims_mapping) == y_dims_mapping_len
    assert len(out_dims_mapping) == out_dims_mapping_len

    return changed


class DistributedMatmul(DistributedOperator):
    def __init__(self, name):
        super(DistributedMatmul, self).__init__()
        self._name = name


register_distributed_operator("matmul", DistributedMatmul("matmul"))


# ColumnParallel
class DistributedMatmulImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl0, self).__init__()
        self._name = name
        self._forward_implemented = True
        self._backward_implemented = False

    def is_process_mesh_compatible(self, op_dist_attr):
        """ No restriction for now. """
        return True

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_dim_shard(y_dims_mapping[0]) or is_dim_replicate(y_dims_mapping[
                1]):
            return False
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_replicate(out_dims_mapping[-1]):
            return False
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
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
            ) == 2, "col_parallel_linear take 2 inputs variable but got {}".format(
                input_name_mapping)
            assert len(
                output_name_mapping
            ) == 1, "col_parallel_linear take 2 inputs variable but got {}".format(
                output_name_mapping)
            assert len(
                input_name_mapping['X']
            ) == 1, "col_parallel_linear input X take 1 variable but got {}".format(
                input_name_mapping['X'])
            assert len(
                input_name_mapping['Y']
            ) == 1, "col_parallel_linear input Y take 1 variable but got {}".format(
                input_name_mapping['Y'])
            assert len(
                output_name_mapping['Out']
            ) == 1, "col_parallel_linear input Out take 1 variable but got {}".format(
                input_name_mapping['Out'])
            X_var = dst_block.var(input_name_mapping['X'][0])
            Weight_var = dst_block.var(input_name_mapping['Y'][0])
            Out_var = dst_block.var(output_name_mapping['Out'][0])

            # TODO infer logic comm presentation
            model_parallel_axis, process_mesh = op_dist_attr.get_owner_context(
            )._get_model_parallel_info()
            group_ranks = _get_comm_group(process_mesh.process_group,
                                          process_mesh.topology,
                                          model_parallel_axis, rank_id)
            group = new_process_group(group_ranks)

            intermediate_var_0 = dst_block.create_var(
                name=unique_name.generate_with_ignorable_key(".".join(
                    ["c_identity", 'tmp'])),
                dtype=X_var.dtype,
                shape=X_var.shape,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=X_var.stop_gradient)
            # copy X_var's dist_attr to intermediate_var_0's dist_attr
            copy_distributed_attr_for_var(op_dist_attr, intermediate_var_0,
                                          X_var)

            check_variable_and_dtype(
                X_var, 'tensor',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                '_c_identity')

            c_identity_op = dst_block.append_op(
                type='c_identity',
                inputs={'X': [X_var]},
                outputs={'Out': intermediate_var_0},
                attrs={
                    'ring_id': group.id,
                    'use_calc_stream': True,
                    'use_model_parallel': True,
                })

            check_variable_and_dtype(intermediate_var_0, 'x',
                                     ['float16', 'float32', 'float64'],
                                     'linear')
            check_dtype(intermediate_var_0.dtype, 'dtype',
                        ['float16', 'float32', 'float64'], 'linear')
            attrs = {
                'transpose_X': False,
                'transpose_Y': False,
                'alpha': 1,
            }
            inputs = {'X': [intermediate_var_0], 'Y': [Weight_var]}
            matmul_op = dst_block.append_op(
                type='matmul',
                inputs=inputs,
                outputs={'Out': Out_var},
                attrs=attrs)

            # copy serial op's dist_attr to dist op's dist_attr
            copy_distributed_attr_for_dist_op(c_identity_op, dst_block,
                                              op_dist_attr)
            copy_distributed_attr_for_dist_op(matmul_op, dst_block,
                                              op_dist_attr)

        if in_dygraph_mode():
            raise NotImplementedError(
                "Dist op for [{}] with idx [{}] is NOT implemented yet.".format(
                    "matmul", 0))
        else:
            return static_handle


# RowParallel
class DistributedMatmulImpl1(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl1, self).__init__()
        self._name = name
        self._forward_implemented = True
        self._backward_implemented = False

    def is_process_mesh_compatible(self, op_dist_attr):
        """ No restriction for now. """
        return True

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_replicate(x_dims_mapping[-1]):
            return False
        if is_dim_replicate(y_dims_mapping[-2]) or is_dim_shard(y_dims_mapping[
                -1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_shard(out_dims_mapping[-1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
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
            ) == 2, "col_parallel_linear take 2 inputs variable but got {}".format(
                input_name_mapping)
            assert len(
                output_name_mapping
            ) == 1, "col_parallel_linear take 2 inputs variable but got {}".format(
                output_name_mapping)
            assert len(
                input_name_mapping['X']
            ) == 1, "col_parallel_linear input X take 1 variable but got {}".format(
                input_name_mapping['X'])
            assert len(
                input_name_mapping['Y']
            ) == 1, "col_parallel_linear input Y take 1 variable but got {}".format(
                input_name_mapping['Y'])
            assert len(
                output_name_mapping['Out']
            ) == 1, "col_parallel_linear input Out take 1 variable but got {}".format(
                input_name_mapping['Out'])
            X_var = dst_block.var(input_name_mapping['X'][0])
            Weight_var = dst_block.var(input_name_mapping['Y'][0])
            Out_var = dst_block.var(output_name_mapping['Out'][0])

            # TODO infer logic comm presentation
            model_parallel_axis, process_mesh = op_dist_attr.get_owner_context(
            )._get_model_parallel_info()
            group_ranks = _get_comm_group(process_mesh.process_group,
                                          process_mesh.topology,
                                          model_parallel_axis, rank_id)
            group = new_process_group(group_ranks)

            check_variable_and_dtype(
                X_var, 'x', ['float16', 'float32', 'float64'], 'linear')
            check_dtype(X_var.dtype, 'dtype',
                        ['float16', 'float32', 'float64'], 'linear')
            attrs = {
                'transpose_X': False,
                'transpose_Y': False,
                'alpha': 1,
            }
            inputs = {'X': X_var, 'Y': Weight_var}
            intermediate_var_0 = dst_block.create_var(
                shape=Out_var.shape,
                dtype=Out_var.dtype,
                type=Out_var.type,
                lod_level=Out_var.lod_level,
                persistable=False,
                is_data=False,
                need_check_feed=Out_var.desc.need_check_feed())
            # copy Out_var's dist_attr to intermediate_var_0's dist_attr
            copy_distributed_attr_for_var(op_dist_attr, intermediate_var_0,
                                          Out_var)

            matmul_op = dst_block.append_op(
                type='matmul',
                inputs=inputs,
                outputs={'Out': intermediate_var_0},
                attrs=attrs)

            c_allreduce_sum_op = dst_block.append_op(
                type='c_allreduce_sum',
                inputs={'X': intermediate_var_0},
                outputs={'Out': Out_var},
                attrs={
                    'ring_id': group.id,
                    'use_calc_stream': True,
                    'use_model_parallel': True
                })

            # copy serial op's dist_attr to dist op's dist_attr
            copy_distributed_attr_for_dist_op(matmul_op, dst_block,
                                              op_dist_attr)
            copy_distributed_attr_for_dist_op(c_allreduce_sum_op, dst_block,
                                              op_dist_attr)

        if in_dygraph_mode():
            raise NotImplementedError(
                "Dist op for [{}] with idx [{}] is NOT implemented yet.".format(
                    "matmul", 0))
        else:
            return static_handle


# ReplicateParallel 
class DistributedMatmulImpl2(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl2, self).__init__()
        self._name = name

    def is_process_mesh_compatible(self, op_dist_attr):
        """ No restriction for now. """
        return True

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)

        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_valid_list_index(x_dims_mapping,
                               -2) and is_dim_shard(x_dims_mapping[-2]):
            return False

        if is_dim_shard(y_dims_mapping[-1]):
            return False
        if is_valid_list_index(y_dims_mapping,
                               -2) and is_dim_shard(y_dims_mapping[-2]):
            return False

        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if is_dim_shard(out_dims_mapping[-1]):
            return False
        if is_valid_list_index(out_dims_mapping,
                               -2) and is_dim_shard(out_dims_mapping[-2]):
            return False

        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
        if dim_changed:
            changed = True
        return changed


register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl0("column_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl1("row_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl2("replicate_parallel"))


class DistributedMatmulV2(DistributedOperator):
    def __init__(self, name):
        super(DistributedMatmulV2, self).__init__()
        self._name = name


register_distributed_operator("matmul_v2", DistributedMatmulV2("matmul_v2"))


# ColumnParallel
class DistributedMatmulV2Impl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulV2Impl0, self).__init__()
        self._name = name
        self._forward_implemented = True
        self._backward_implemented = False

    def is_process_mesh_compatible(self, op_dist_attr):
        """ No restriction for now. """
        return True

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_dim_shard(y_dims_mapping[0]) or is_dim_replicate(y_dims_mapping[
                1]):
            return False
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_replicate(out_dims_mapping[-1]):
            return False
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
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
            ) == 2, "col_parallel_linear take 2 inputs variable but got {}".format(
                input_name_mapping)
            assert len(
                output_name_mapping
            ) == 1, "col_parallel_linear take 2 inputs variable but got {}".format(
                output_name_mapping)
            assert len(
                input_name_mapping['X']
            ) == 1, "col_parallel_linear input X take 1 variable but got {}".format(
                input_name_mapping['X'])
            assert len(
                input_name_mapping['Y']
            ) == 1, "col_parallel_linear input Y take 1 variable but got {}".format(
                input_name_mapping['Y'])
            assert len(
                output_name_mapping['Out']
            ) == 1, "col_parallel_linear input Out take 1 variable but got {}".format(
                input_name_mapping['Out'])
            X_var = dst_block.var(input_name_mapping['X'][0])
            Weight_var = dst_block.var(input_name_mapping['Y'][0])
            Out_var = dst_block.var(output_name_mapping['Out'][0])

            # TODO infer logic comm presentation
            model_parallel_axis, process_mesh = op_dist_attr.get_owner_context(
            )._get_model_parallel_info()
            group_ranks = _get_comm_group(process_mesh.process_group,
                                          process_mesh.topology,
                                          model_parallel_axis, rank_id)
            group = new_process_group(group_ranks)

            intermediate_var_0 = dst_block.create_var(
                name=unique_name.generate_with_ignorable_key(".".join(
                    ["c_identity", 'tmp'])),
                dtype=X_var.dtype,
                shape=X_var.shape,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=X_var.stop_gradient)
            # copy X_var's dist_attr to intermediate_var_0's dist_attr
            copy_distributed_attr_for_var(op_dist_attr, intermediate_var_0,
                                          X_var)

            check_variable_and_dtype(
                X_var, 'tensor',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                '_c_identity')

            c_identity_op = dst_block.append_op(
                type='c_identity',
                inputs={'X': [X_var]},
                outputs={'Out': intermediate_var_0},
                attrs={
                    'ring_id': group.id,
                    'use_calc_stream': True,
                    'use_model_parallel': True,
                })

            check_variable_and_dtype(intermediate_var_0, 'x',
                                     ['float16', 'float32', 'float64'],
                                     'linear')
            check_dtype(intermediate_var_0.dtype, 'dtype',
                        ['float16', 'float32', 'float64'], 'linear')
            attrs = {'trans_x': False, 'trans_y': False}
            inputs = {'X': [intermediate_var_0], 'Y': [Weight_var]}
            matmul_v2_op = dst_block.append_op(
                type='matmul_v2',
                inputs=inputs,
                outputs={'Out': Out_var},
                attrs=attrs)

            # copy serial op's dist_attr to dist op's dist_attr
            copy_distributed_attr_for_dist_op(c_identity_op, dst_block,
                                              op_dist_attr)
            copy_distributed_attr_for_dist_op(matmul_v2_op, dst_block,
                                              op_dist_attr)

        if in_dygraph_mode():
            raise NotImplementedError(
                "Dist op for [{}] with idx [{}] is NOT implemented yet.".format(
                    "matmul", 0))
        else:
            return static_handle


# RowParallel
class DistributedMatmulV2Impl1(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulV2Impl1, self).__init__()
        self._name = name
        self._forward_implemented = True
        self._backward_implemented = False

    def is_process_mesh_compatible(self, op_dist_attr):
        """ No restriction for now. """
        return True

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_replicate(x_dims_mapping[-1]):
            return False
        if is_dim_replicate(y_dims_mapping[-2]) or is_dim_shard(y_dims_mapping[
                -1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_shard(out_dims_mapping[-1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
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
            ) == 2, "col_parallel_linear take 2 inputs variable but got {}".format(
                input_name_mapping)
            assert len(
                output_name_mapping
            ) == 1, "col_parallel_linear take 2 inputs variable but got {}".format(
                output_name_mapping)
            assert len(
                input_name_mapping['X']
            ) == 1, "col_parallel_linear input X take 1 variable but got {}".format(
                input_name_mapping['X'])
            assert len(
                input_name_mapping['Y']
            ) == 1, "col_parallel_linear input Y take 1 variable but got {}".format(
                input_name_mapping['Y'])
            assert len(
                output_name_mapping['Out']
            ) == 1, "col_parallel_linear input Out take 1 variable but got {}".format(
                input_name_mapping['Out'])
            X_var = dst_block.var(input_name_mapping['X'][0])
            Weight_var = dst_block.var(input_name_mapping['Y'][0])
            Out_var = dst_block.var(output_name_mapping['Out'][0])

            # TODO infer logic comm presentation
            model_parallel_axis, process_mesh = op_dist_attr.get_owner_context(
            )._get_model_parallel_info()
            group_ranks = _get_comm_group(process_mesh.process_group,
                                          process_mesh.topology,
                                          model_parallel_axis, rank_id)
            group = new_process_group(group_ranks)

            check_variable_and_dtype(
                X_var, 'x', ['float16', 'float32', 'float64'], 'linear')
            check_dtype(X_var.dtype, 'dtype',
                        ['float16', 'float32', 'float64'], 'linear')
            attrs = {'trans_x': False, 'trans_y': False}
            inputs = {'X': X_var, 'Y': Weight_var}
            intermediate_var_0 = dst_block.create_var(
                shape=Out_var.shape,
                dtype=Out_var.dtype,
                type=Out_var.type,
                lod_level=Out_var.lod_level,
                persistable=False,
                is_data=False,
                need_check_feed=Out_var.desc.need_check_feed())
            # copy Out_var's dist_attr to intermediate_var_0's dist_attr
            copy_distributed_attr_for_var(op_dist_attr, intermediate_var_0,
                                          Out_var)

            matmul_v2_op = dst_block.append_op(
                type='matmul_v2',
                inputs=inputs,
                outputs={'Out': intermediate_var_0},
                attrs=attrs)

            c_allreduce_sum_op = dst_block.append_op(
                type='c_allreduce_sum',
                inputs={'X': intermediate_var_0},
                outputs={'Out': Out_var},
                attrs={
                    'ring_id': group.id,
                    'use_calc_stream': True,
                    'use_model_parallel': True
                })

            # copy serial op's dist_attr to dist op's dist_attr
            copy_distributed_attr_for_dist_op(matmul_v2_op, dst_block,
                                              op_dist_attr)
            copy_distributed_attr_for_dist_op(c_allreduce_sum_op, dst_block,
                                              op_dist_attr)

        if in_dygraph_mode():
            raise NotImplementedError(
                "Dist op for [{}] with idx [{}] is NOT implemented yet.".format(
                    "matmul", 0))
        else:
            return static_handle


# ReplicateParallel 
class DistributedMatmulV2Impl2(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulV2Impl2, self).__init__()
        self._name = name

    def is_process_mesh_compatible(self, op_dist_attr):
        """ No restriction for now. """
        return True

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)

        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_valid_list_index(x_dims_mapping,
                               -2) and is_dim_shard(x_dims_mapping[-2]):
            return False

        if is_dim_shard(y_dims_mapping[-1]):
            return False
        if is_valid_list_index(y_dims_mapping,
                               -2) and is_dim_shard(y_dims_mapping[-2]):
            return False

        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if is_dim_shard(out_dims_mapping[-1]):
            return False
        if is_valid_list_index(out_dims_mapping,
                               -2) and is_dim_shard(out_dims_mapping[-2]):
            return False

        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
        if dim_changed:
            changed = True
        return changed


register_distributed_operator_impl("matmul_v2",
                                   DistributedMatmulV2Impl0("column_parallel"))
register_distributed_operator_impl("matmul_v2",
                                   DistributedMatmulV2Impl1("row_parallel"))
register_distributed_operator_impl(
    "matmul_v2", DistributedMatmulV2Impl2("replicate_parallel"))
