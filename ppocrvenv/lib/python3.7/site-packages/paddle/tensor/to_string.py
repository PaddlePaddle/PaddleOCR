# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
from paddle.fluid.layers import core
from paddle.fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype

__all__ = []


class PrintOptions(object):
    precision = 8
    threshold = 1000
    edgeitems = 3
    linewidth = 80
    sci_mode = False


DEFAULT_PRINT_OPTIONS = PrintOptions()


def set_printoptions(precision=None,
                     threshold=None,
                     edgeitems=None,
                     sci_mode=None,
                     linewidth=None):
    """Set the printing options for Tensor.
    NOTE: The function is similar with numpy.set_printoptions()

    Args:
        precision (int, optional): Number of digits of the floating number, default 8.
        threshold (int, optional): Total number of elements printed, default 1000.
        edgeitems (int, optional): Number of elements in summary at the begining and ending of each dimension, default 3.
        sci_mode (bool, optional): Format the floating number with scientific notation or not, default False.
        linewidth (int, optional): Number of characters each line, default 80.
       
    
    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle

            paddle.seed(10)
            a = paddle.rand([10, 20])
            paddle.set_printoptions(4, 100, 3)
            print(a)
            
            '''
            Tensor(shape=[10, 20], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                   [[0.0002, 0.8503, 0.0135, ..., 0.9508, 0.2621, 0.6661],
                    [0.9710, 0.2605, 0.9950, ..., 0.4427, 0.9241, 0.9363],
                    [0.0948, 0.3226, 0.9955, ..., 0.1198, 0.0889, 0.9231],
                    ...,
                    [0.7206, 0.0941, 0.5292, ..., 0.4856, 0.1379, 0.0351],
                    [0.1745, 0.5621, 0.3602, ..., 0.2998, 0.4011, 0.1764],
                    [0.0728, 0.7786, 0.0314, ..., 0.2583, 0.1654, 0.0637]])
            '''
    """
    kwargs = {}

    if precision is not None:
        check_type(precision, 'precision', (int), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.precision = precision
        kwargs['precision'] = precision
    if threshold is not None:
        check_type(threshold, 'threshold', (int), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.threshold = threshold
        kwargs['threshold'] = threshold
    if edgeitems is not None:
        check_type(edgeitems, 'edgeitems', (int), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.edgeitems = edgeitems
        kwargs['edgeitems'] = edgeitems
    if linewidth is not None:
        check_type(linewidth, 'linewidth', (int), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.linewidth = linewidth
        kwargs['linewidth'] = linewidth
    if sci_mode is not None:
        check_type(sci_mode, 'sci_mode', (bool), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.sci_mode = sci_mode
        kwargs['sci_mode'] = sci_mode
    core.set_printoptions(**kwargs)


def _to_summary(var):
    edgeitems = DEFAULT_PRINT_OPTIONS.edgeitems

    # Handle tensor of shape contains 0, like [0, 2], [3, 0, 3]
    if np.prod(var.shape) == 0:
        return np.array([])

    if len(var.shape) == 0:
        return var
    elif len(var.shape) == 1:
        if var.shape[0] > 2 * edgeitems:
            return np.concatenate([var[:edgeitems], var[(-1 * edgeitems):]])
        else:
            return var
    else:
        # recursively handle all dimensions
        if var.shape[0] > 2 * edgeitems:
            begin = [x for x in var[:edgeitems]]
            end = [x for x in var[(-1 * edgeitems):]]
            return np.stack([_to_summary(x) for x in (begin + end)])
        else:
            return np.stack([_to_summary(x) for x in var])


def _format_item(np_var, max_width=0, signed=False):
    if np_var.dtype == np.float32 or np_var.dtype == np.float64 or np_var.dtype == np.float16:
        if DEFAULT_PRINT_OPTIONS.sci_mode:
            item_str = '{{:.{}e}}'.format(
                DEFAULT_PRINT_OPTIONS.precision).format(np_var)
        elif np.ceil(np_var) == np_var:
            item_str = '{:.0f}.'.format(np_var)
        else:
            item_str = '{{:.{}f}}'.format(
                DEFAULT_PRINT_OPTIONS.precision).format(np_var)
    else:
        item_str = '{}'.format(np_var)

    if max_width > len(item_str):
        if signed:  # handle sign character for tenosr with negative item
            if np_var < 0:
                return item_str.ljust(max_width)
            else:
                return ' ' + item_str.ljust(max_width - 1)
        else:
            return item_str.ljust(max_width)
    else:  # used for _get_max_width
        return item_str


def _get_max_width(var):
    # return max_width for a scalar
    max_width = 0
    signed = False
    for item in list(var.flatten()):
        if (not signed) and (item < 0):
            signed = True
        item_str = _format_item(item)
        max_width = max(max_width, len(item_str))

    return max_width, signed


def _format_tensor(var, summary, indent=0, max_width=0, signed=False):
    """
    Format a tensor

    Args:
        var(Tensor): The tensor to be formatted.
        summary(bool): Do summary or not. If true, some elements will not be printed, and be replaced with "...".
        indent(int): The indent of each line.
        max_width(int): The max width of each elements in var.
        signed(bool): Print +/- or not.
    """
    edgeitems = DEFAULT_PRINT_OPTIONS.edgeitems
    linewidth = DEFAULT_PRINT_OPTIONS.linewidth

    if len(var.shape) == 0:
        # currently, shape = [], i.e., scaler tensor is not supported.
        # If it is supported, it should be formatted like this.
        return _format_item(var, max_width, signed)
    elif len(var.shape) == 1:
        item_length = max_width + 2
        items_per_line = (linewidth - indent) // item_length
        items_per_line = max(1, items_per_line)

        if summary and var.shape[0] > 2 * edgeitems:
            items = [
                _format_item(item, max_width, signed)
                for item in list(var)[:edgeitems]
            ] + ['...'] + [
                _format_item(item, max_width, signed)
                for item in list(var)[(-1 * edgeitems):]
            ]
        else:
            items = [
                _format_item(item, max_width, signed) for item in list(var)
            ]
        lines = [
            items[i:i + items_per_line]
            for i in range(0, len(items), items_per_line)
        ]
        s = (',\n' + ' ' *
             (indent + 1)).join([', '.join(line) for line in lines])
        return '[' + s + ']'
    else:
        # recursively handle all dimensions
        if summary and var.shape[0] > 2 * edgeitems:
            vars = [
                _format_tensor(x, summary, indent + 1, max_width, signed)
                for x in var[:edgeitems]
            ] + ['...'] + [
                _format_tensor(x, summary, indent + 1, max_width, signed)
                for x in var[(-1 * edgeitems):]
            ]
        else:
            vars = [
                _format_tensor(x, summary, indent + 1, max_width, signed)
                for x in var
            ]

        return '[' + (',' + '\n' * (len(var.shape) - 1) + ' ' *
                      (indent + 1)).join(vars) + ']'


def to_string(var, prefix='Tensor'):
    indent = len(prefix) + 1

    _template = "{prefix}(shape={shape}, dtype={dtype}, place={place}, stop_gradient={stop_gradient},\n{indent}{data})"

    tensor = var.value().get_tensor()
    if not tensor._is_initialized():
        return "Tensor(Not initialized)"

    np_var = var.numpy()

    if len(var.shape) == 0:
        size = 0
    else:
        size = 1
        for dim in var.shape:
            size *= dim

    summary = False
    if size > DEFAULT_PRINT_OPTIONS.threshold:
        summary = True

    max_width, signed = _get_max_width(_to_summary(np_var))

    data = _format_tensor(
        np_var, summary, indent=indent, max_width=max_width, signed=signed)

    return _template.format(
        prefix=prefix,
        shape=var.shape,
        dtype=convert_dtype(var.dtype),
        place=var._place_str,
        stop_gradient=var.stop_gradient,
        indent=' ' * indent,
        data=data)
