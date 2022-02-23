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

from __future__ import print_function
import collections
import copy
import six
import numpy as np
from ..framework import Block, Variable, in_dygraph_mode
from ..data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..layer_helper import LayerHelper
from sys import version_info


def convert_to_list(value, n, name, dtype=int):
    """
    Converts a single numerical type or iterable of numerical
    types into an numerical type list.

    Arguments:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the list to be returned.
      name: The name of the argument being validated, e.g. "stride" or
        "filter_size". This is only used to format error messages.
      dtype: the numerical type of the element of the list to be returned.

    Returns:
      A list of n dtypes.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, dtype):
        return [value, ] * n
    else:
        try:
            value_list = list(value)
        except TypeError:
            raise ValueError("The " + name +
                             "'s type must be list or tuple. Received: " + str(
                                 value))
        if len(value_list) != n:
            raise ValueError("The " + name + "'s length must be " + str(n) +
                             ". Received: " + str(value))
        for single_value in value_list:
            try:
                dtype(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The " + name + "'s type must be a list or tuple of " + str(
                        n) + " " + str(dtype) + " . Received: " + str(
                            value) + " "
                    "including element " + str(single_value) + " of type" + " "
                    + str(type(single_value)))
        return value_list


def is_sequence(seq):
    """
    Whether `seq` is an entry or nested structure
    """
    if isinstance(seq, dict):
        return True
    return (isinstance(seq, collections.Sequence) and
            not isinstance(seq, six.string_types))


def _hash_with_id(*args):
    """
    Return int hash value calculated by id(arg) or tuple(id1,id2, ...).
    """
    assert len(args) > 0
    info = tuple([id(v) for v in args])
    return hash(info) & 0xfffffff


def _sorted(dict_):
    """
    Returns a sorted list of the dict keys, with error if keys not sortable.
    """
    try:
        return sorted(six.iterkeys(dict_))
    except TypeError:
        raise TypeError("nest only supports dicts with sortable keys.")


def _yield_value(iterable):
    if isinstance(iterable, dict):
        # Iterate through dictionaries in a deterministic order by sorting the
        # keys. Notice this means that we ignore the original order of `OrderedDict`
        # instances. This is intentional, to avoid potential bugs caused by mixing
        # ordered and plain dicts (e.g., flattening a dict but using a
        # corresponding `OrderedDict` to pack it back).
        for key in _sorted(iterable):
            yield iterable[key]
    else:
        for value in iterable:
            yield value


def _yield_flat_nest(nest):
    for n in _yield_value(nest):
        if is_sequence(n):
            for ni in _yield_flat_nest(n):
                yield ni
        else:
            yield n


def flatten(nest):
    """
	:alias_main: paddle.flatten
	:alias: paddle.flatten,paddle.tensor.flatten,paddle.tensor.manipulation.flatten
	:old_api: paddle.fluid.layers.flatten

    Traverse all entries in the nested structure and put them into an list.
    """
    if is_sequence(nest):
        return list(_yield_flat_nest(nest))
    else:
        return [nest]


def _sequence_like(instance, args):
    """
    Convert the sequence `args` to the same type as `instance`.
    """
    if isinstance(instance, dict):
        # Pack dictionaries in a deterministic order by sorting the keys.
        # Notice this means that we ignore the original order of `OrderedDict`
        # instances. This is intentional, to avoid potential bugs caused by mixing
        # ordered and plain dicts (e.g., flattening a dict but using a
        # corresponding `OrderedDict` to pack it back).
        result = dict(zip(_sorted(instance), args))
        return type(instance)((key, result[key])
                              for key in six.iterkeys(instance))
    elif (isinstance(instance, tuple) and hasattr(instance, "_fields") and
          isinstance(instance._fields, collections.Sequence) and
          all(isinstance(f, six.string_types) for f in instance._fields)):
        # This is a namedtuple
        return type(instance)(*args)
    else:
        # Not a namedtuple
        return type(instance)(args)


def _packed_nest_with_indices(structure, flat, index):
    """
    Helper function for pack_sequence_as.
    """
    packed = []
    for s in _yield_value(structure):
        if is_sequence(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(_sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def pack_sequence_as(structure, flat_sequence):
    """
    Pack a given flattened sequence into a given structure.
    """
    if not is_sequence(flat_sequence):
        raise TypeError("flat_sequence must be a sequence")
    if not is_sequence(structure):
        if len(flat_sequence) != 1:
            raise ValueError(
                "Structure is a scalar but len(flat_sequence) == %d > 1" %
                len(flat_sequence))
        return flat_sequence[0]
    flat_structure = flatten(structure)
    if len(flat_structure) != len(flat_sequence):
        raise ValueError(
            "Could not pack sequence. Structure had %d elements, but flat_sequence "
            "had %d elements.  Structure: %s, flat_sequence: %s." %
            (len(flat_structure), len(flat_sequence), structure, flat_sequence))
    _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
    return _sequence_like(structure, packed)


def map_structure(func, *structure):
    """
    Apply `func` to each entry in `structure` and return a new structure.
    """
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    return pack_sequence_as(structure[0], [func(*x) for x in entries])


def hold_mutable_vars(structure):
    """
    Returns whether structure holds sequence like `list/dict`.
    """
    for s in structure:
        if is_sequence(s):
            return True
    return False


def copy_mutable_vars(structure):
    """
    Returns vars copied from sequence without mutable property.
    """
    flat_structure = copy.copy(flatten(structure))
    return pack_sequence_as(structure, flat_structure)


def _recursive_assert_same_structure(nest1, nest2, check_types):
    """
    Helper function for `assert_same_structure`.
    """
    is_sequence_nest1 = is_sequence(nest1)
    if is_sequence_nest1 != is_sequence(nest2):
        raise ValueError(
            "The two structures don't have the same nested structure.\n\n"
            "First structure: %s\n\nSecond structure: %s." % (nest1, nest2))
    if not is_sequence_nest1:
        return  # finished checking
    if check_types:
        type_nest1 = type(nest1)
        type_nest2 = type(nest2)
        if type_nest1 != type_nest2:
            raise TypeError(
                "The two structures don't have the same sequence type. First "
                "structure has type %s, while second structure has type %s." %
                (type_nest1, type_nest2))
        if isinstance(nest1, dict):
            keys1 = set(six.iterkeys(nest1))
            keys2 = set(six.iterkeys(nest2))
            if keys1 != keys2:
                raise ValueError(
                    "The two dictionaries don't have the same set of keys. First "
                    "structure has keys {}, while second structure has keys {}."
                    .format(keys1, keys2))
    nest1_as_sequence = [n for n in _yield_value(nest1)]
    nest2_as_sequence = [n for n in _yield_value(nest2)]
    for n1, n2 in zip(nest1_as_sequence, nest2_as_sequence):
        _recursive_assert_same_structure(n1, n2, check_types)


def assert_same_structure(nest1, nest2, check_types=True):
    """
    Confirm two nested structures with the same structure.
    """
    len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
    len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
    if len_nest1 != len_nest2:
        raise ValueError("The two structures don't have the same number of "
                         "elements.\n\nFirst structure (%i elements): %s\n\n"
                         "Second structure (%i elements): %s" %
                         (len_nest1, nest1, len_nest2, nest2))
    _recursive_assert_same_structure(nest1, nest2, check_types)


def _is_symmetric_padding(padding, data_dim):
    """
    Check whether padding is symmetrical.
    """
    assert len(padding) == data_dim * 2 or len(padding) == data_dim
    is_sys = True
    if len(padding) == data_dim * 2:
        for i in range(data_dim):
            if padding[i * 2] != padding[i * 2 + 1]:
                is_sys = False
    return is_sys


def _contain_var(list_or_tuple):
    """
    Check whether list or tuple contains variable.
    """
    for item in list_or_tuple:
        if isinstance(item, Variable):
            return True
    return False


def get_shape_tensor_inputs(inputs, attrs, shape, op_type):
    from .tensor import fill_constant, cast

    def _get_attr_shape(list_shape):
        attr_shape = []
        for idx, dim in enumerate(list_shape):
            if isinstance(dim, Variable):
                attr_shape.append(-1)
            else:
                attr_shape.append(dim)
        return attr_shape

    def _get_shape_tensor(list_shape):
        shape_tensor_list = []
        for idx, dim in enumerate(list_shape):
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                check_dtype(
                    dim.dtype, 'shape[' + str(idx) + ']', ['int32', 'int64'],
                    op_type,
                    '(When type of shape in' + op_type + 'is list or tuple.)')
                if convert_dtype(dim.dtype) == 'int64':
                    dim = cast(x=dim, dtype='int32')
                shape_tensor_list.append(dim)
            else:
                temp_out = fill_constant([1], 'int32', dim, force_cpu=True)
                shape_tensor_list.append(temp_out)
        return shape_tensor_list

    if isinstance(shape, Variable):
        shape.stop_gradient = True
        check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'fill_constant',
                    '(When type of shape in' + op_type + ' is Variable.)')
        if (convert_dtype(shape.dtype) == 'int64'):
            shape = cast(shape, 'int32')
        inputs["ShapeTensor"] = shape
    elif isinstance(shape, (list, tuple)):
        assert len(shape) > 0, (
            "The size of 'shape' in" + op_type + " can't be zero, "
            "but received %s." % len(shape))
        attrs["shape"] = _get_attr_shape(shape)
        if _contain_var(shape):
            inputs['ShapeTensorList'] = _get_shape_tensor(shape)
    else:
        raise TypeError("Shape only supports Variable, or list, or tuple.")


def _convert_to_tensor_list(old_list, dtype="int32"):
    """
    Converts all elements of a list to Variable.
    """
    from .tensor import fill_constant
    new_list_tensor = []
    for ele in old_list:

        if isinstance(ele, Variable):
            ele.stop_gradient = True
            new_list_tensor.append(ele)
        else:
            assert isinstance(ele, six.integer_types)
            temp_out = fill_constant([1], dtype, ele, force_cpu=True)
            new_list_tensor.append(temp_out)
    return new_list_tensor


def convert_shape_to_list(shape):
    """
    Convert shape(list, tuple, variable) to list in imperative mode
    """
    if isinstance(shape, (list, tuple)):
        shape = list(
            map(lambda x: x.numpy()[0] if isinstance(x, Variable) else x,
                shape))
    else:
        shape = shape.numpy().astype(int).tolist()
    return shape


def check_shape(shape):
    """
    Check shape type and shape elements type before passing it to fill_constant
    """
    if isinstance(shape, Variable):
        check_dtype(shape.dtype, 'shape', ['int32', 'int64'], 'fill_constant')
    else:
        for ele in shape:
            if not isinstance(ele, Variable):
                if ele < 0:
                    raise ValueError(
                        "All elements in ``shape`` must be positive when it's a list or tuple"
                    )
                if not isinstance(ele, six.integer_types):
                    raise TypeError(
                        "All elements in ``shape`` must be integers when it's a list or tuple"
                    )


def try_set_static_shape_tensor(tensor, shape):
    """Try to set static shape of tensor from a shape tensor.
    
    For example,

    import paddle
    paddle.enable_static()
    data = paddle.static.data(name="x", shape=[-1, 2], dtype='float32')
    shape = paddle.shape(data)  # shape should be [-1, 2] instead of [-1, -1]
    x = paddle.uniform(shape) 
    print(x.shape) 
    # (-1, 2)
    
    """
    if not in_dygraph_mode():
        # static mode, and shape is not all inferred (contains -1)
        if -1 in tensor.shape:
            if isinstance(shape, Variable):
                shape = try_get_constant_shape_from_tensor(shape)
                if shape:
                    tensor.desc.set_shape(shape)


def try_get_constant_shape_from_tensor(shape_tensor):
    """Try to get shape from a tensor with constant value.

    For example,
    
    import paddle
    paddle.enable_static()
    data = paddle.static.data(name="x", shape=[-1, 2], dtype='float32')
    shape = paddle.shape(data)  # shape should be [-1, 2] instead of [-1, -1]
    x = paddle.uniform(shape) 
    print(x.shape) 
    # (-1, 2)
    
    """
    if not in_dygraph_mode():
        try:
            if shape_tensor.op is not None:
                generate_op = shape_tensor.op
                if generate_op.type == 'shape':
                    var = shape_tensor.block.vars[generate_op.input_arg_names[
                        0]]
                    return var.shape
        except:
            return None

        return None


def get_inputs_outputs_in_block(block):
    """
    Returns the inputs and outputs variable used in this block but not
    created in this block.
    """
    assert isinstance(
        block,
        Block), "input non-Block argument for get_inputs_outputs_in_block."
    assert block.parent_idx != -1, "input block should be a sub-block, not main block."

    # Find input/output var names of all ops in block
    inner_inputs = set()
    inner_outputs = set()
    for op in block.ops:
        for iname in op.input_names:
            for in_var_name in op.input(iname):
                if not block.has_var(in_var_name):
                    # variable not created in this block
                    inner_inputs.add(in_var_name)
        for oname in op.output_names:
            for out_var_name in op.output(oname):
                if not block.has_var(out_var_name):
                    # variable not created in this block
                    inner_outputs.add(out_var_name)

    return inner_inputs, inner_outputs
