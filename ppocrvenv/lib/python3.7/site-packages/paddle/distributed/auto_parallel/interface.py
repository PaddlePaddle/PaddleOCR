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
# limitations under the License.

import numpy
import copy
import paddle
import paddle.fluid.core as core
from paddle.fluid.framework import Variable
from paddle.fluid.framework import in_dygraph_mode

__all__ = []

# a map from ProcessMesh ids to the ProcessMesh instances
_g_process_mesh_map = dict()

# user defined map from logical process ids to physical ones
_user_defined_physical_map = None


def _append_attr_suffix(name):
    """
    Append auto parallel suffix for distributed attribute name.
    """
    return name + core.kAutoParallelSuffix()


def _remove_attr_suffix(name):
    """
    Remove auto parallel suffix from distributed attribute name.
    """
    return name.strip(core.kAutoParallelSuffix())


def _static_mode_check():
    if in_dygraph_mode():
        raise RuntimeError("Auto-parallel only supports static mode, "
                           "please use paddle.enable_static().")


def _get_nested_list_shape(nested_list):
    """
    Get the shape of a nested_list.
    """
    result = []
    while isinstance(nested_list, list):
        result.append(len(nested_list))
        nested_list = nested_list[0]
    return result


def _flatten_nested_list(nested_list):
    """
    Get a list of all items in a nested_list.
    Ref: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    """
    result = numpy.array(nested_list).flatten().tolist()
    return result


class ProcessMesh(object):
    r"""
    The class `Processmesh` describes the topology of logical processes. 
    A mesh is an N-dimensional array. The shape of the N-dimensional
    array represents the topology of logical processes and every
    element of the N-dimensional array represent a logical process. For
    example, the 2-dimensional array [[2, 4, 5], [0, 1, 3]]
    illustrates six logical processes organized as the topology [2, 3],
    i.e., the shape of the 2-dimensional array. With the above topology,
    there are two parallel groups, where the first parallel group has a
    parallel degree of 2 and the second one has a parallel degree of 3.
    And the first logical process is the one with id=2.

    Args:
        mesh (list): an N-dimensional array (nested list) describes the toplogy
            of logical processes. The shape of the N-dimensional array
            represents the topology of logical processes and every 
            element of the N-dimensional array represents a logical process.
        parent (ProcessMesh, optional): the parent ProcessMesh. None means
            the ProcessMesh is the root one without parent ProcessMesh.
            Default: None.
    
    Returns:
        None

    Raises:
        ValueError: If `mesh` is not an instance of list.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist
            
            paddle.enable_static()
            
            mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
            assert mesh.parent is None
            assert mesh.topology == [2, 3]
            assert mesh.process_group == [2, 4, 5, 0, 1, 3]
            mesh.set_placement([0, 1, 2, 3, 4, 5])

    """

    def __init__(self, mesh, parent=None):
        _static_mode_check()
        if mesh is None or not isinstance(mesh, list):
            raise ValueError('mesh must be an instance of list.')

        self._topology = _get_nested_list_shape(mesh)
        self._processes = _flatten_nested_list(mesh)

        # Every element of mesh must be >= 0.
        assert min(self._processes) >= 0, ('All elements of mesh must be >= 0.')

        unique_ids = set(self._processes)
        assert len(unique_ids) == len(self._processes), (
            'All elements of mesh must be unique.')

        if parent is None:
            # For root ProcessMesh, the ids of logical processes must be range
            # from 0 to N-1, where N is the number of logical processes. 
            assert max(self._processes) == len(self._processes) - 1, (
                'For root ProcessMesh, ids of logical processes must be range '
                'from 0 to N-1, where N is the number of logical processes.')

            parent_id = core.kNoneProcessMeshIndex()
            assert len(_g_process_mesh_map.keys()) == 0, (
                'The first ProcessMesh must be the root, which has no parent.')
        else:
            assert len(_g_process_mesh_map.keys()) > 0, (
                'All ProcessMesh must have a parent except the root one.')

            assert isinstance(parent, ProcessMesh), (
                'parent must be an instance of ProcessMesh.')
            parent_id = parent._desc.id

            # All elements in mesh must belong to its parent
            parent_ids = set(parent.process_group)
            assert unique_ids <= parent_ids, (
                'All elements in mesh must belong to its parent.')

        self._desc = core.ProcessMeshDesc(self._topology, self._processes,
                                          parent_id)

        self._id = self._desc.id
        self._parent_id = parent_id
        assert self._id not in _g_process_mesh_map, (
            "The ProcessMesh with id %d already exists." % self._id)
        _g_process_mesh_map[self._id] = self

    @property
    def topology(self):
        r"""
        Get the topology of logical processes belonging to this ProcessMesh.
        This is the shape of `mesh` used to initialized this ProcessMesh.
        """
        return self._topology

    @property
    def process_group(self):
        r"""
        Get a list of all processes belonging to this ProcessMesh.
        """
        return self._processes

    @property
    def parent(self):
        r"""
        Get the parent ProcessMesh.
        """
        if self._parent_id == core.kNoneProcessMeshIndex(): return None
        assert self._parent_id in _g_process_mesh_map, (
            "parent with id %d does not exist." % self._parent_id)
        return _g_process_mesh_map[self._parent_id]

    @property
    def ndim(self):
        r"""
        Get the number of dimension of ProcessMesh.
        """
        return len(self._topology)

    def set_placement(self, order):
        """
        Set the map from logical processes to physical ones using the
        user defined order.

        Args:
            order (list): order of the physical process ids.
        
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle
                import paddle.distributed as dist
                
                paddle.enable_static()
                
                mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
                mesh.set_placement([0, 1, 2, 3, 4, 5])

        """
        assert self.parent is None, (
            "This function can only be called by the root ProcessMesh.")
        unique_ids = set(order)
        assert isinstance(order, list)

        assert len(unique_ids) == len(order), (
            "All elements in order must be unique.")
        assert min(order) == 0
        assert max(order) == len(order) - 1, (
            "All elements in order must be from 0 to N - 1, where N "
            "is the number of physical processes.")

        logical_order = self.process_group
        global _user_defined_physical_map
        assert _user_defined_physical_map is None, (
            "This function can only be called once.")
        _user_defined_physical_map = dict()

        assert len(logical_order) == len(order)
        for idx, l_id in enumerate(logical_order):
            _user_defined_physical_map[l_id] = order[idx]

    def _reset_global_process_mesh_map(self):
        """
        Remove all process mesh in _g_process_mesh_map, make it empty.
        """

        _g_process_mesh_map = dict()

    def __eq__(self, other):
        assert other and isinstance(other, ProcessMesh)
        if self.topology != other.topology or self.process_group != other.process_group:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        str = "shape {} and process group {}".format(self.topology,
                                                     self.process_group)
        return str

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # No need to copy the owner tensor and context
            if k == "_desc":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


def _dim_mapping_checker(tensor, mesh, dim_mapping):
    assert isinstance(mesh,
                      ProcessMesh), 'The type of mesh must be ProcessMesh.'
    assert isinstance(dim_mapping,
                      list), 'The type of dim_mapping must be list.'
    assert len(tensor.shape) == len(dim_mapping), (
        'The number of dimensions '
        'of tensor must be the same as the length of its corresponding '
        'dim_mapping.')
    mesh_dim = len(mesh.topology)
    dim_set = set()
    for i in range(len(dim_mapping)):
        assert dim_mapping[i] == -1 or (
            dim_mapping[i] < mesh_dim and dim_mapping[i] >= 0), (
                'Each element '
                'in dim_mapping must be greater than zero and less than the '
                'length of its corresponding topology, or it must be -1.')
        if dim_mapping[i] >= 0:
            assert dim_mapping[i] not in dim_set
            dim_set.add(dim_mapping[i])


def shard_tensor(x, mesh, dim_mapping):
    """
    Add distributed attributes for a tensors.

    Args:
        x (Tensor): the tensor to process.
        mesh (ProcessMesh): an instance of ProcessMesh to describe the topology of logical processes.
        dim_mapping (list): a list to describe the mapping between `x` and `mesh`,
            the dimension `i` of `x` is split across the dimension `dims_mapping[i]`, where -1 means
            without parition along the corresponding dimension.

    Returns:
        Tensor: the tensor `x` itself.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist
            
            paddle.enable_static()

            mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
            x = paddle.ones([4, 6])
            dist.shard_tensor(x, mesh, [0, -1])

    """
    _static_mode_check()
    _dim_mapping_checker(x, mesh, dim_mapping)
    attr_name = _append_attr_suffix('mesh_id')
    x._set_attr(attr_name, mesh._id)
    attr_name = _append_attr_suffix('dim_mapping')
    x._set_attr(attr_name, dim_mapping)
    return x


def set_shard_mask(x, mask):
    """
    Set the mask for a tensor which mask out the tensor from some processes in its mesh.

    Args:
        x (Tensor): the tensor to process.
        mask (list): a nested list. The shape of `mask` must be the same as the ProcessMesh belonging to
            the tensor `x`. Every value of `mask` must be one or zero, where one means 
            the tenor `x` will be put on the corresponding logical process and zero means the tensor `x`
            will not be put on the corresponding logical process.
            For example, for a ProcessMesh represented by the 2-dimensional
            array [[2, 4, 5], [0, 1, 3]], and a `mask` given by the
            2-dimensional [[1, 0, 1], [0, 1, 0]],
            then the tensor `x` will only be put on logical processes 2, 5 and 1.

    Returns:
        Tensor: the tensor `x` itself.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
            mask = [[1, 0, 1], [0, 1, 0]]
            x = paddle.ones([4, 6])
            dist.shard_tensor(x, mesh, [-1, 1])
            dist.set_shard_mask(x, mask)

    """
    _static_mode_check()
    assert isinstance(mask, list)
    np_mask = numpy.array(mask)
    min_ele = numpy.min(np_mask)
    max_ele = numpy.max(np_mask)
    mesh_attr_name = _append_attr_suffix('mesh_id')
    assert x._has_attr(mesh_attr_name), \
        "Please set process mesh for the variable firstly."
    assert min_ele >= 0 and max_ele <= 1, "Elements in mask must be 0 or 1."
    x_mesh = x.process_mesh
    assert x_mesh, "Please set process mesh for the variable firstly."
    assert x_mesh.topology == list(np_mask.shape), (
        "The shape of mask "
        "must be the same as the shape of its Process Mesh.")
    attr_name = _append_attr_suffix('mask')
    x._set_attr(attr_name, _flatten_nested_list(mask))
    return x


def shard_op(op_fn, mesh, dim_mapping_dict, **kwargs):
    """
    Call a functioin and add distributed attributes for ops added by the function.

    Args:
        op_fn (callable): a callable object of an API.
        mesh (ProcessMesh): an instance of ProcessMesh specifies the topology of logical processes.
        dim_mapping_dict (dict): a mapping from tensor's name to its dims_mapping.
            The dim_mapping is a list to describe the mapping between a tensor and `mesh`,
            the dimension `i` of the tensor is split across the dimension `dim_mapping[i]`,
            where -1 means without parition along the corresponding dimension.
        kwargs (dict): a dict of parameter passed to the function `op_fn`.

    Returns:
        list: the outputs of the function `op_fn`.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
            x = paddle.ones([4, 6])
            y = paddle.zeros([4, 6])
            kwargs = {'x': x, 'y': y}
            dist.shard_op(paddle.add, mesh, None, **kwargs)

    """
    _static_mode_check()
    main_prog = paddle.fluid.default_main_program()
    main_block = main_prog.global_block()
    op_size = len(main_block.ops)
    output = op_fn(**kwargs)
    new_op_size = len(main_block.ops)
    if dim_mapping_dict is None:
        dim_mapping_dict = dict()
    else:
        assert isinstance(dim_mapping_dict,
                          dict), 'The type of dim_mapping_dict must be dict.'
        for var_name in dim_mapping_dict.keys():
            dim_mapping = dim_mapping_dict[var_name]
            tensor = main_block.var(var_name)
            _dim_mapping_checker(tensor, mesh, dim_mapping)
    for idx in range(op_size, new_op_size):
        op = main_block.ops[idx]
        attr_name = _append_attr_suffix('mesh_id')
        op._set_attr(attr_name, mesh._id)
        for var_name in dim_mapping_dict.keys():
            assert var_name in op.output_arg_names + op.input_arg_names
            attr_name = _append_attr_suffix(var_name)
            if var_name in op.input_arg_names:
                # we use the prefix "IN_" to indicates an input argument name
                attr_name = "IN_" + attr_name
            else:
                # we use the prefix "OUT_" to indicates an input argument name
                attr_name = "OUT_" + attr_name
            op._set_attr(attr_name, dim_mapping_dict[var_name])

    if isinstance(output, Variable):
        output = [output]
    return list(output)


def set_offload_device(x, device):
    """
    Set the device that the tensor `x` will be put on.

    Args:
        x (tensor): the tensor to process.
        device (str): the device that the tensor `x` will be put on, e.g., 'cpu'.

    Returns:
        Tensor: the tensor `x` itself.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            x = paddle.ones([4, 6])
            dist.set_offload_device(x, 'cpu')

    """
    _static_mode_check()
    assert device == "cpu", "Only 'cpu' is supported for destination device."
    attr_name = _append_attr_suffix("offload_device")
    x._set_attr(attr_name, device)
    return x


def set_pipeline_stage(stage):
    """
    Set the pipeline stage of the following ops.

    Args:
        stage (int): the pipeline stage the following ops belonging to.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            dist.set_pipeline_stage(0)

    """
    from paddle.fluid.framework import _set_pipeline_stage
    _static_mode_check()
    assert isinstance(stage, int), 'The type of stage must be int.'
    _set_pipeline_stage(stage)
