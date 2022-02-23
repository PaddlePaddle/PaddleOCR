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

import sys
import numpy as np
from . import unique_name
from . import core
import paddle

MAX_INTEGER = 2**31 - 1


def is_list_tuple(index, contain_type):
    def _is_list_tuple(item):
        if not (isinstance(item, (list, tuple)) or type(item) == contain_type):
            return False
        if isinstance(item, (tuple, list)):
            for s in item:
                if not _is_list_tuple(s):
                    return False
        return True

    if not isinstance(index, (tuple, list)):
        return False
    for s in index:
        if not _is_list_tuple(s):
            return False
    return True


def is_one_dim_list(index, contain_type):
    if isinstance(index, list):
        for i in index:
            if not isinstance(i, contain_type):
                return False
    else:
        return False
    return True


def get_list_index_shape(var_dims, index_dims):
    var_dims_size = len(var_dims)
    index_dims_size = len(index_dims)

    out_dims_size = var_dims_size - index_dims[0] + index_dims_size - 1

    out_dims_shape = [1] * out_dims_size

    out_dims_shape[:index_dims_size - 1] = index_dims[1:]

    out_dims_shape[index_dims_size - 1:] = var_dims[index_dims[0]:]
    return out_dims_shape


class SliceInfo:
    def __init__(self):
        self.pre_shape = None
        self.indexes = []
        self.dtype = None

    def update(self, index):
        if is_list_tuple(index, int) or isinstance(index, (
                paddle.fluid.Variable, np.ndarray)):
            # convert index to Tensor
            if not isinstance(index, paddle.fluid.Variable):
                index = paddle.assign(index)

            if self.dtype is None:
                self.dtype = index.dtype
            else:
                if index.dtype != self.dtype:
                    raise IndexError(
                        "Data type of Tensor/List index should be same. The current data type is {}, but the previous data type is {}.".
                        format(index.dtype, self.dtype))

            self.indexes.append(index)

            if self.pre_shape is None:
                self.pre_shape = index.shape
            else:
                if self.pre_shape != index.shape:
                    # broadcast 
                    cur_shape = paddle.broadcast_shape(self.pre_shape,
                                                       index.shape)
                    for i in range(len(self.indexes)):
                        self.indexes[i] = paddle.broadcast_to(self.indexes[i],
                                                              cur_shape)
                self.pre_shape = self.indexes[-1].shape
        else:
            raise ValueError(
                "Index should be list/tuple of int or Tensor, but received {}.".
                format(index))

    def shape_stride(self, shape):
        s = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            s[i] = shape[i + 1] * s[i + 1]

        return s

    def numel(self, shape):
        return reduce(lambda x, y: x * y, shape)

    def get_offset_stride(self, tensor_shape):
        for index in self.indexes:
            if not isinstance(index, paddle.fluid.Variable):
                raise ValueError(
                    "only support list/tensor index, but received {}.".format(
                        type(index)))

        if len(self.indexes) <= len(tensor_shape) or len(self.indexes) == 1:
            shape = paddle.stack(self.indexes)
            axes = list(range(1, len(self.pre_shape) + 1)) + [0, ]

        else:
            raise ValueError(
                "too many indices for tensor: tensor is {}-dimensional, but {} were indexed".
                format(len(tensor_shape), self.pre_shape[0]))

        shape_transpose = paddle.transpose(shape, axes)
        return shape_transpose

    def get_item(self, tensor):
        shape_transpose = self.get_offset_stride(tensor.shape)
        index = paddle.assign(shape_transpose)
        return paddle.gather_nd(tensor, index)

    def set_item(self, tensor_origin, value):

        if not isinstance(value, paddle.fluid.Variable):
            value = paddle.assign(value)
        tensor_type = None

        if tensor_origin.dtype in [
                core.VarDesc.VarType.FP32, core.VarDesc.VarType.FP64
        ]:
            tensor = tensor_origin
        else:
            tensor_type = tensor_origin.dtype
            tensor = tensor_origin.astype(core.VarDesc.VarType.FP32)

        if value.dtype != tensor.dtype:
            value = value.astype(tensor.dtype)

        shape_transpose = self.get_offset_stride(tensor_origin.shape)
        index = paddle.assign(shape_transpose)

        gather_tensor_shape = get_list_index_shape(
            tensor.shape, [len(self.indexes), ] + list(self.indexes[-1].shape))

        value_dims_bd = [1, ] * len(gather_tensor_shape)
        value_dims_bd[-len(value.shape):] = list(value.shape)

        for i in range(len(gather_tensor_shape)):
            if not (value_dims_bd[i] == gather_tensor_shape[i] or
                    value_dims_bd[i] == 1):
                raise ValueError("{} can not broadcast into {}".format(
                    value.shape, gather_tensor_shape))

        value_broadcast = paddle.broadcast_to(value, gather_tensor_shape)

        value_1d = value_broadcast.reshape([-1] + gather_tensor_shape[len(
            index.shape) - 1:])

        index_1d = index.reshape([-1, index.shape[-1]])

        tensor_stride = paddle.assign(
            self.shape_stride(tensor.shape[:index.shape[-1]]))
        inds = []
        for i in range(index_1d.shape[0]):
            temp = (index_1d[i] * tensor_stride).sum()
            inds.append(temp)
        index_1d = paddle.stack(inds).reshape([-1])
        t_reshape = tensor.reshape([-1] + list(tensor.shape[index.shape[-1]:]))
        out = paddle.scatter(t_reshape, index_1d, value_1d)
        if tensor_type is not None:
            out = out.astype(tensor_type)
        tensor_origin[:] = out.reshape(tensor_origin.shape)

        return tensor_origin


def replace_ellipsis(var, item):
    from .framework import Variable
    # Use slice(None) to replace Ellipsis.
    # For var, var.shape = [3,4,5,6]
    #
    #   var[..., 1:2] -> var[:, :, :, 1:2]
    #   var[0, ...] -> var[0]
    #   var[0, ..., 1:2] -> var[0, :, :, 1:2]

    item = list(item)

    # Remove Variable to skip bug when counting Ellipsis
    item_remove_var = [
        ele for ele in item
        if not isinstance(ele, (Variable, np.ndarray)) and ele is not None
    ]
    ell_count = item_remove_var.count(Ellipsis)
    if ell_count == 0:
        return item
    elif ell_count > 1:
        raise IndexError("An index can only have a single ellipsis ('...')")

    ell_idx = item.index(Ellipsis)

    if ell_idx == len(item) - 1:
        return item[:-1]
    else:
        item[ell_idx:ell_idx + 1] = [slice(None)] * (
            len(var.shape) - len(item) + item.count(None) + 1)

    return item


def replace_ndarray(item):
    new_item = []
    for slice_item in item:
        if isinstance(slice_item, np.ndarray):
            new_item.append(paddle.assign(slice_item))
        else:
            new_item.append(slice_item)
    return new_item


def replace_none(item):
    new_item = []
    none_axes = []
    for i, slice_item in enumerate(item):
        if slice_item is None:
            none_axes.append(i)
        else:
            new_item.append(slice_item)
    return new_item, none_axes


def is_integer_or_scalar_tensor(ele):
    from .framework import Variable
    if isinstance(ele, int):
        return True
    elif isinstance(ele, Variable):
        if len(ele.shape) == 1 and ele.shape[0] == 1:
            return True
    return False


def deal_attrs(attrs, attr, attr_name, tensor_attr_name, inputs, infer_flags):
    from .framework import Variable
    from .layers import utils

    if utils._contain_var(attr):
        inputs[tensor_attr_name] = utils._convert_to_tensor_list(
            attr, dtype="int64")
        for i, dim in enumerate(attr):
            if isinstance(dim, Variable):
                attrs[attr_name].append(-1)
                infer_flags[i] = -1
            else:
                attrs[attr_name].append(dim)
    else:
        attrs[attr_name] = attr


def _getitem_impl_(var, item):
    """
    Slice the variable.

    Args:
        item(int/slice/tuple) : the index.

    Returns:
        Sliced variable
    """
    from .framework import default_main_program, Variable
    if isinstance(item, list):
        if not is_one_dim_list(item, int):
            item = tuple(item)

    if not isinstance(item, tuple):
        item = (item, )

    decrease_axes = []
    axes = []
    starts = []
    ends = []
    steps = []
    reverse_axes = []

    use_strided_slice = False
    item = replace_ndarray(item)
    item = replace_ellipsis(var, item)
    item, none_axes = replace_none(item)
    slice_info = SliceInfo()

    for dim, slice_item in enumerate(item):
        if is_integer_or_scalar_tensor(slice_item):
            if isinstance(slice_item,
                          int) and var.shape[dim] is not None and var.shape[
                              dim] >= 0 and slice_item >= var.shape[dim]:
                # For python, if users write a, b = var, the __getitem__
                # method will iterate through 0, 1, 2 ... until __getitem__
                # throws an IndexError, then stop. The var[0], var[1] will
                # be given to a, b respectively. If more values are given,
                # the unpack size would cause error.
                #
                # We raises IndexError here to support grammar like `a, b = var`
                raise IndexError(
                    "slice_item %d at dim %d should be >= 0 and < var.shape[%d]: %d"
                    % (slice_item, dim, dim, var.shape[dim]))
            decrease_axes.append(dim)
            start = slice_item
            step = 1
            end = slice_item + 1 if slice_item != -1 else MAX_INTEGER

        elif isinstance(slice_item, slice):
            start = slice_item.start
            end = slice_item.stop
            step = slice_item.step

            if start is None and end is None and step is None:
                continue

            step = 1 if step is None else step

            if start is None:
                start = 0 if step > 0 else MAX_INTEGER
            if end is None:
                end = MAX_INTEGER if step > 0 else -1

        elif isinstance(slice_item, list):
            all_bool = True

            if is_list_tuple(slice_item, int):
                slice_info.update(slice_item)
                continue

            for i in slice_item:
                if type(i) is int:
                    all_bool = False
                elif not isinstance(i, bool):
                    raise TypeError("Only support int or bool in index list.")

            if len(item) != 1:
                raise IndexError(
                    "When index contains a list, its length must be 1, but received {}.".
                    format(len(item)))
            new_slice_item = []
            if all_bool:
                if len(slice_item) != var.shape[0]:
                    raise IndexError(
                        "The dimension of bool index doesn't match indexed array along "\
                        "dimension 0, the target dimension is {}, but received {}.".
                        format(var.shape[0], len(slice_item)))
                for idx, ele in enumerate(slice_item):
                    if ele is True:
                        new_slice_item.append(idx)
                slice_item = new_slice_item
            else:
                for idx, ele in enumerate(slice_item):
                    if type(ele) is int:
                        new_slice_item.append(ele)
                    elif ele is True:
                        new_slice_item.append(1)
                    else:
                        new_slice_item.append(0)
                slice_item = new_slice_item

            from .layers import assign
            from ..tensor import index_select

            idx = assign(np.array(slice_item).astype("int32"))
            return index_select(var, index=idx, axis=0)

        elif isinstance(slice_item, (Variable)):
            if len(item) == 1:

                from ..tensor import index_select, gather_nd
                from .layers.nn import where

                if slice_item.dtype == paddle.bool:
                    if len(slice_item.shape) > len(var.shape):
                        raise IndexError(
                            "The dims of bool index doesn't match indexed array, "
                            "the dims of bool index except to be equal or less "
                            "than {}, but received {}.".format(
                                len(var.shape), len(slice_item.shape)))
                    for i, dim_len in enumerate(slice_item.shape):
                        if dim_len != var.shape[i]:
                            raise IndexError(
                                "The dimension of bool index doesn't match indexed array along "\
                                "dimension {}, the target dimension is {}, but received {}.".
                                format(i, var.shape[i], dim_len))
                    bool_2_idx = where(slice_item == True)
                    return gather_nd(var, bool_2_idx)
                else:
                    if len(slice_item.shape) == 1:
                        return index_select(var, index=slice_item, axis=0)
                    else:
                        slice_info.update(slice_item)
                        continue
            else:
                slice_info.update(slice_item)
                continue

        else:
            raise IndexError(
                "Valid index accept int or slice or ellipsis or list, but received {}.".
                format(slice_item))

        axes.append(dim)
        starts.append(start)
        ends.append(end)
        steps.append(step)
        use_strided_slice = True if step != 1 else use_strided_slice

    if slice_info.indexes:
        if len(slice_info.indexes) != len(item):
            raise IndexError(
                "Valid index accept int or slice or ellipsis or list, but received {}.".
                format(item))
        return slice_info.get_item(var)

    inputs = {'Input': [var]}
    attrs = {
        'axes': axes,
        'starts': [],
        'ends': [],
        'decrease_axis': decrease_axes
    }
    if use_strided_slice:
        attrs['strides'] = []

    infer_flags = [1] * len(axes)
    deal_attrs(attrs, starts, "starts", "StartsTensorList", inputs, infer_flags)
    deal_attrs(attrs, ends, "ends", "EndsTensorList", inputs, infer_flags)
    deal_attrs(attrs, steps, "strides", "StridesTensorList", inputs,
               infer_flags)
    attrs['infer_flags'] = infer_flags

    out = var
    if len(axes) > 0:
        target_block = default_main_program().current_block()
        op_type = "strided_slice" if use_strided_slice else "slice"

        slice_out_var = target_block.create_var(
            name=unique_name.generate_with_ignorable_key(var.name + "_" +
                                                         op_type),
            dtype=var.dtype)
        target_block.append_op(
            type=op_type,
            inputs=inputs,
            outputs={'Out': [slice_out_var]},
            attrs=attrs)
        out = slice_out_var

    if len(reverse_axes) > 0:
        from .layers.tensor import reverse
        out = reverse(out, axis=reverse_axes)

    # Deal with cases when all axes are decreased.
    # After slice, the shape of out is [1], which should have been [], but Paddle doesn't support scalar.
    # In order to ensure the correctness of the final shape of out, one dimension of out needs to be decreased.
    # For example:
    # # x.shape: (2,3,4)
    # out = x[0, 1, 1, None] # out.shape : (1)
    if len(decrease_axes) == len(var.shape):
        none_axes = none_axes[1:]

    if len(none_axes) > 0:
        # Deal with cases that decrease_axes is not empty
        # For example:
        # # x.shape: (2,3,4)
        # out = x[0, 0:2, None] # out.shape : (2, 1, 4)
        for idx, axis in enumerate(none_axes):
            l = len([i for i in decrease_axes if i < axis])
            new_axis = axis - l
            none_axes[idx] = new_axis

        # Deal with cases when all axes are decreased.
        # After slice, the shape of out is [1], which should have been [], but Paddle doesn't support scalar.
        # In order to ensure the correctness of the final shape of out, one dimension of out needs to be decreased.
        # For example:
        # # x.shape: (2,3,4)
        # out = x[0, 1, 1, None] # out.shape : (1)

        from ..tensor import unsqueeze
        out = unsqueeze(out, axis=none_axes)

    return out


def _setitem_impl_(var, item, value):
    from .framework import default_main_program, Variable

    inputs = {'Input': var}
    if isinstance(item, list):
        if not is_one_dim_list(item, int):
            item = tuple(item)
    # 1. Parse item
    if not isinstance(item, tuple):
        item = (item, )

    decrease_axes = []
    axes = []
    starts = []
    ends = []
    steps = []

    item = replace_ndarray(item)
    item = replace_ellipsis(var, item)
    item, none_axes = replace_none(item)
    slice_info = SliceInfo()
    dim = 0
    for _, slice_item in enumerate(item):
        if is_integer_or_scalar_tensor(slice_item):
            decrease_axes.append(dim)
            start = slice_item
            end = slice_item + 1 if slice_item != -1 else MAX_INTEGER
            step = 1

        elif isinstance(slice_item, slice):
            start = slice_item.start
            end = slice_item.stop
            step = slice_item.step

            if start is None and end is None and step is None:
                dim += 1
                continue

            step = 1 if step is None else step

            if not isinstance(step, Variable) and step == 0:
                raise ValueError(
                    "When assign a value to a paddle.Tensor, step can not be 0, "
                    "but received step is {}.".format(step))

            if isinstance(step, Variable) and (start is None or end is None):
                raise ValueError(
                    "When assign a value to a paddle.Tensor, it's not supported that "
                    "the start or end is None when the type of step is paddle.Tensor."
                )

            if start is None:
                start = 0 if step > 0 else MAX_INTEGER

            if end is None:
                end = MAX_INTEGER if step > 0 else (0 - MAX_INTEGER)
        elif isinstance(slice_item, list):
            if is_list_tuple(slice_item, int):
                slice_info.update(slice_item)
                continue

            for i in slice_item:
                if not isinstance(i, bool):
                    raise TypeError("Doesn't support {} in index list.".format(
                        type(i)))

            if len(item) != 1:
                raise IndexError(
                    "When index contains a bool list, its length must be 1, but received {}.".
                    format(len(item)))

            from .layers import assign
            idx_tensor = assign(slice_item)
            return set_value_for_bool_tensor(var, idx_tensor, value)

        elif isinstance(slice_item, Variable):
            if slice_item.dtype == core.VarDesc.VarType.BOOL:
                if len(item) != 1:
                    raise IndexError(
                        "When index contains a bool tensor, its length must be 1, but received {}.".
                        format(len(item)))
                return set_value_for_bool_tensor(var, slice_item, value)
            else:
                slice_info.update(slice_item)
                continue
        else:
            raise IndexError(
                "Valid index accept int, slice, ellipsis, None, list of bool, Variable, "
                "but received {}.".format(slice_item))

        axes.append(dim)
        starts.append(start)
        ends.append(end)
        steps.append(step)

        dim += 1
    if slice_info.indexes:
        if len(slice_info.indexes) != len(item):
            raise IndexError(
                "Valid index accept int or slice or ellipsis or list, but received {}.".
                format(item))
        return slice_info.set_item(var, value)
    attrs = {
        'axes': axes,
        'starts': starts,
        'ends': ends,
        'steps': steps,
        'decrease_axes': decrease_axes,
        'none_axes': none_axes
    }

    from .layers import utils
    if utils._contain_var(starts):
        inputs['StartsTensorList'] = utils._convert_to_tensor_list(starts)
        del attrs['starts']
    if utils._contain_var(ends):
        inputs['EndsTensorList'] = utils._convert_to_tensor_list(ends)
        del attrs['ends']
    if utils._contain_var(steps):
        inputs['StepsTensorList'] = utils._convert_to_tensor_list(steps)
        del attrs['steps']

    # 2. Parse value
    dtype = var.dtype
    attrs['dtype'] = dtype

    from .data_feeder import convert_dtype
    #  2.1 value is an integer of float
    if isinstance(value, (int, float)):
        value = np.array([value]).astype(convert_dtype(dtype))

    #  2.2 value is a np.ndarray
    if isinstance(value, np.ndarray):
        shape = list(value.shape)
        if dtype == core.VarDesc.VarType.BOOL:
            value_name = "bool_values"
            values = [bool(v) for v in value.flat]
        elif dtype == core.VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in value.flat]
        elif dtype == core.VarDesc.VarType.FP64:
            value_name = "fp64_values"
            values = [float(v) for v in value.flat]
        elif dtype == core.VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in value.flat]
        elif dtype == core.VarDesc.VarType.INT64:
            value_name = "int64_values"
            values = [int(v) for v in value.flat]
        else:
            raise TypeError(
                "When assign a numpy.ndarray, integer or float to a paddle.Tensor, "
                "the data type of the paddle.Tensor must be bool, float32, int32 or int64, but "
                "received %s." % convert_dtype(dtype))
        attrs[value_name] = values
        attrs["shape"] = shape

    elif isinstance(value, Variable):
        inputs["ValueTensor"] = value
    else:
        raise TypeError(
            "Only support to assign an integer, float, numpy.ndarray or "
            "paddle.Tensor to a paddle.Tensor, but received {}".format(
                type(value)))

    cur_block = default_main_program().current_block()
    cur_block.append_op(
        type="set_value", inputs=inputs, outputs={'Out': var}, attrs=attrs)

    return var


# the item is a tensor of bool 
def set_value_for_bool_tensor(var, item, value):

    # TODO(zyfncg): Now scatter_nd_add only support float32 and float64 tensor, 
    # so in the current version we also only support float32 and float64 tensor, 
    # this problem will be fixed in the future.
    if var.dtype != core.VarDesc.VarType.FP32 and var.dtype != core.VarDesc.VarType.FP64:
        raise TypeError("Only support float and double tensor for bool index, "
                        "but received {}.".format(var.dtype))

    if len(item.shape) > len(var.shape):
        raise IndexError("The dims of bool index doesn't match indexed array, "
                         "the dims of bool index except to be equal or less "
                         "than {}, but received {}.".format(
                             len(var.shape), len(item.shape)))
    for i, dim_len in enumerate(item.shape):
        if dim_len != var.shape[i]:
            raise IndexError(
                "The dimension of bool index doesn't match indexed array along "
                "dimension {}, the target dimension is {}, but received {}.".
                format(i, var.shape[i], dim_len))

    def idx_not_empty(var, item, value):
        from .framework import Variable
        from .layers import assign
        from .layers.nn import where
        from ..tensor import gather_nd, scatter_nd_add

        if not isinstance(value, Variable):
            value = assign(value).cast(var.dtype)

        idx = where(item)
        gather_val = gather_nd(var, idx)
        gather_val_new = value - gather_val
        out = scatter_nd_add(var, idx, gather_val_new)
        var[:] = out

    from .layers.control_flow import cond
    # If all the bool index is False, just do nothing
    cond(item.any(), lambda: idx_not_empty(var, item, value))

    return var
