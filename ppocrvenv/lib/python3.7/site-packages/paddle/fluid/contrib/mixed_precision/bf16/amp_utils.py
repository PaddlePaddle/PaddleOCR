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

from __future__ import print_function

from .... import core
from .... import framework
from .... import global_scope
from ....log_helper import get_logger
from ....wrapped_decorator import signature_safe_contextmanager
from .amp_lists import AutoMixedPrecisionListsBF16
from ..fp16_utils import find_true_prev_op, find_true_post_op, _rename_arg, \
    find_op_index, _rename_op_input

import collections
import struct
import logging
import numpy as np

__all__ = [
    "bf16_guard", "rewrite_program_bf16", "cast_model_to_bf16",
    "cast_parameters_to_bf16", "convert_float_to_uint16"
]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

_valid_types = [
    core.VarDesc.VarType.LOD_TENSOR, core.VarDesc.VarType.SELECTED_ROWS,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY
]

_bf16_guard_pattern = "__use_bf16__"


def convert_float_to_uint16(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack('<I', struct.pack('<f', x))[0] >> 16,
        otypes=[np.uint16])(in_list.flat)
    return np.reshape(out, in_list.shape)


def _dtype_to_str(dtype):
    """
    Convert specific variable type to its corresponding string.

    Args:
        dtype (VarType): Variable type.
    """
    if dtype == core.VarDesc.VarType.BF16:
        return 'bf16'
    else:
        return 'fp32'


def _insert_cast_op(block, op, idx, src_dtype, dest_dtype):
    """
    Insert cast op and rename args of input and output.

    Args:
        block (Program): The block in which the operator is.
        op (Operator): The operator to insert cast op.
        idx (int): The index of current operator.
        src_dtype (VarType): The input variable dtype of cast op.
        dest_dtype (VarType): The output variable dtype of cast op.

    Returns:
        num_cast_op (int): The number of cast ops that have been inserted.
    """
    num_cast_ops = 0

    for in_name in op.input_names:
        if src_dtype == core.VarDesc.VarType.FP32 and op.type in [
                'batch_norm', 'fused_bn_add_activation', 'layer_norm'
        ]:
            if in_name not in {'X', 'Z'}:
                continue
        for in_var_name in op.input(in_name):
            in_var = block.var(in_var_name)
            if in_var.type not in _valid_types or in_var.dtype == dest_dtype:
                continue
            if in_var.dtype == src_dtype:
                cast_name = in_var.name + '.cast_' + _dtype_to_str(dest_dtype)
                out_var = block.vars.get(cast_name)
                if out_var is None or out_var.dtype != dest_dtype:
                    out_var = block.create_var(
                        name=cast_name,
                        dtype=dest_dtype,
                        persistable=False,
                        stop_gradient=in_var.stop_gradient)

                    block._insert_op(
                        idx,
                        type="cast",
                        inputs={"X": in_var},
                        outputs={"Out": out_var},
                        attrs={
                            "in_dtype": in_var.dtype,
                            "out_dtype": out_var.dtype
                        })
                    num_cast_ops += 1
                _rename_arg(op, in_var.name, out_var.name)
            else:
                if op.has_attr('in_dtype'):
                    op._set_attr('in_dtype', dest_dtype)
    if src_dtype == core.VarDesc.VarType.FP32 and dest_dtype == core.VarDesc.VarType.BF16:
        for out_name in op.output_names:
            if op.type in [
                    'batch_norm', 'fused_bn_add_activation', 'layer_norm'
            ] and out_name != 'Y':
                continue
            for out_var_name in op.output(out_name):
                out_var = block.var(out_var_name)
                if out_var.type not in _valid_types:
                    continue
                if out_var.dtype == core.VarDesc.VarType.FP32:
                    out_var.desc.set_dtype(core.VarDesc.VarType.BF16)
                    if op.has_attr('out_dtype'):
                        op._set_attr('out_dtype', core.VarDesc.VarType.BF16)
    return num_cast_ops


def _insert_cast_post_op(block, op, idx, src_dtype, dest_dtype, target_name,
                         op_var_rename_map):
    num_cast_ops = 0
    target_var = block.var(target_name)
    if target_var.type not in _valid_types or target_var.dtype == dest_dtype:
        return num_cast_ops

    assert target_var.dtype == src_dtype, \
        "The real dtype({}) is not equal to the src dtype({})".format(_dtype_to_str(target_var.dtype), _dtype_to_str(src_dtype))

    cast_name = target_var.name + '.cast_' + _dtype_to_str(dest_dtype)
    cast_var = block.vars.get(cast_name)
    if cast_var is None or cast_var.dtype != dest_dtype:
        cast_var = block.create_var(
            name=cast_name,
            dtype=dest_dtype,
            persistable=False,
            stop_gradient=target_var.stop_gradient)
        block._insert_op(
            idx,
            type="cast",
            inputs={"X": target_var},
            outputs={"Out": cast_var},
            attrs={"in_dtype": target_var.dtype,
                   "out_dtype": cast_var.dtype})
        num_cast_ops += 1
        op_var_rename_map[block.idx][target_var.name] = cast_var.name

    return num_cast_ops


def _is_in_fp32_varnames(op, amp_lists):
    if not amp_lists.fp32_varnames:
        return False

    for in_name in op.input_arg_names:
        if in_name in amp_lists.fp32_varnames:
            return True

    for out_name in op.output_arg_names:
        if out_name in amp_lists.fp32_varnames:
            return True

    return False


def _need_keep_fp32(op, unsupported_op_list, use_bf16_guard):
    if op.type in unsupported_op_list:
        # the highest priority condition: If ops don't have bf16 computing kernels,
        # they must be executed in fp32 calculation pattern.
        return True

    # process ops about learning rate
    in_out_arg_names = []
    in_out_arg_names.extend(list(op.input_arg_names))
    in_out_arg_names.extend(list(op.output_arg_names))
    for name in in_out_arg_names:
        if "learning_rate" in name:
            return True

    if use_bf16_guard:
        if op.has_attr("op_namescope") and \
                (_bf16_guard_pattern in op.attr("op_namescope")):
            # op in bf16 guard
            return False
        else:
            # op not in bf16 guard
            return True
    else:
        return False


@signature_safe_contextmanager
def bf16_guard():
    """
    As for the pure bf16 training, if users set `use_bf16_guard` to True,
    only those ops created in the context manager `bf16_guard` will be
    transformed as float16 type.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F
            paddle.enable_static()
            data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
            conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)

            with paddle.static.amp.bf16_guard():
                bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
                pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                hidden = paddle.static.nn.fc(pool, size=10)
                loss = paddle.mean(hidden)
    """
    with framework.name_scope(prefix=_bf16_guard_pattern):
        yield


def are_post_ops_bf16(post_ops, keep_fp32_ops):
    for post_op in post_ops:
        for op in post_op:
            if op in keep_fp32_ops:
                return False
    return True


def cast_initializers_to_bf16(startup_prog,
                              amp_lists,
                              block,
                              all_ops,
                              keep_fp32_ops,
                              to_bf16_var_names=None):
    prepend_ops = startup_prog.global_block().ops
    for op in prepend_ops:
        if str(op.type) in amp_lists.bf16_initializer_list:
            change_op = True
            op_post_ops = []
            op_out_vars = []
            for out_name in op.output_names:
                for out_var_name in op.output(out_name):
                    out_var = block.var(out_var_name)
                    post_op = find_true_post_op(all_ops, op, out_var_name, True)

                    if out_var is None or out_var.type not in _valid_types:
                        change_op = False
                        break
                    op_post_ops.append(post_op)
                    op_out_vars.append(out_var)

            if change_op and are_post_ops_bf16(op_post_ops, keep_fp32_ops):
                for out_var in op_out_vars:
                    if out_var.dtype == core.VarDesc.VarType.FP32:
                        out_var.desc.set_dtype(core.VarDesc.VarType.BF16)
                    if to_bf16_var_names is not None and out_var.name in to_bf16_var_names:
                        to_bf16_var_names.remove(out_var.name)
                if op.has_attr('dtype') and op.attr(
                        'dtype') == core.VarDesc.VarType.FP32:
                    op._set_attr('dtype', core.VarDesc.VarType.BF16)


def cast_model_to_bf16(program,
                       startup_prog=None,
                       amp_lists=None,
                       use_bf16_guard=True):
    """
    Traverse all ops in the whole model and set their inputs and outputs
    to the bf16 data type. This function will do some special processing for
    the batch normalization, which will keep the batchnorm's computations in FP32.
    Args:
        program (Program): The used program.
        amp_lists (AutoMixedPrecisionListsBF16): An AutoMixedPrecisionListsBF16 object.
        use_bf16_guard(bool): Determine whether to use `bf16_guard` when
                              constructing the program. Default True.
    """

    if amp_lists is None:
        amp_lists = AutoMixedPrecisionListsBF16()
    global_block = program.global_block()
    keep_fp32_ops = set()
    to_bf16_var_names = set()
    to_bf16_pre_cast_ops = set()
    origin_ops = []
    for block in program.blocks:
        origin_ops.extend(block.ops)

    for block in program.blocks:
        ops = block.ops
        for op in ops:
            if op.type == 'create_py_reader' or op.type == 'read':
                continue
            if _need_keep_fp32(op, amp_lists.unsupported_list, use_bf16_guard):
                keep_fp32_ops.add(op)
                continue  # processed below
            for in_name in op.input_names:
                if op.type in {
                        'batch_norm', 'fused_bn_add_activation', 'layer_norm'
                } and in_name not in {'X', 'Z'}:
                    continue
                for in_var_name in op.input(in_name):
                    in_var = None
                    try:
                        in_var = block.var(in_var_name)
                    except ValueError as e:
                        _logger.debug(
                            "-- {}, try to get it in the global block --".
                            format(e))
                        in_var = global_block.var(in_var_name)
                        if in_var is not None:
                            _logger.debug(
                                "-- var {} is got in the global block --".
                                format(in_var_name))

                    if in_var is None or in_var.type not in _valid_types:
                        continue

                    if in_var.dtype == core.VarDesc.VarType.FP32:
                        in_var.desc.set_dtype(core.VarDesc.VarType.BF16)
                        to_bf16_var_names.add(in_var_name)

                    _logger.debug(
                        "-- op type: {}, in var name: {}, in var dtype: {} --".
                        format(op.type, in_var_name, in_var.dtype))

            for out_name in op.output_names:
                if op.type in {
                        'batch_norm', 'fused_bn_add_activation', 'layer_norm'
                } and out_name != 'Y':
                    continue
                for out_var_name in op.output(out_name):
                    out_var = None
                    try:
                        out_var = block.var(out_var_name)
                    except ValueError as e:
                        _logger.debug(
                            "-- {}, try to get it in the global block --".
                            format(e))
                        out_var = global_block.var(out_var_name)
                        if out_var is not None:
                            _logger.debug(
                                "-- var {} is got in the global block --".
                                format(out_var_name))

                    if out_var is None or out_var.type not in _valid_types:
                        continue

                    if out_var.dtype == core.VarDesc.VarType.FP32:
                        out_var.desc.set_dtype(core.VarDesc.VarType.BF16)

                    _logger.debug(
                        "-- op type: {}, out var name: {}, out var dtype: {} --".
                        format(op.type, out_var_name, out_var.dtype))
            for attr_name in ['in_dtype', 'out_dtype', 'dtype']:
                if op.has_attr(attr_name) and op.attr(
                        attr_name) == core.VarDesc.VarType.FP32:
                    op._set_attr(attr_name, core.VarDesc.VarType.BF16)
            if op.has_attr('use_mkldnn'):
                op._set_attr('use_mkldnn', True)
            if op.has_attr('mkldnn_data_type'):
                op._set_attr('mkldnn_data_type', 'bfloat16')

        if startup_prog is not None:
            cast_initializers_to_bf16(startup_prog, amp_lists, global_block,
                                      ops, keep_fp32_ops, to_bf16_var_names)

    # process ops in keep_fp32_ops
    op_var_rename_map = [
        collections.OrderedDict() for _ in range(len(program.blocks))
    ]
    for block in program.blocks:
        ops = block.ops
        idx = 0
        while idx < len(ops):
            op = ops[idx]
            num_cast_ops = 0
            if op not in keep_fp32_ops:
                if op in to_bf16_pre_cast_ops:
                    in_var_cast_num = _insert_cast_op(block, op, idx,
                                                      core.VarDesc.VarType.FP32,
                                                      core.VarDesc.VarType.BF16)
                    num_cast_ops += in_var_cast_num
            else:
                pre_cast_num = _insert_cast_op(block, op, idx,
                                               core.VarDesc.VarType.BF16,
                                               core.VarDesc.VarType.FP32)
                num_cast_ops += pre_cast_num
                for out_var_name in op.output_arg_names:
                    out_var = block.vars.get(out_var_name)
                    if out_var is None or out_var.type not in _valid_types:
                        continue
                    if out_var.dtype == core.VarDesc.VarType.BF16:
                        out_var.desc.set_dtype(core.VarDesc.VarType.FP32)
                        post_ops = find_true_post_op(ops, op, out_var_name)
                        for post_op in post_ops:
                            if post_op in keep_fp32_ops:
                                continue
                            post_cast_num = _insert_cast_post_op(
                                block, op, idx + pre_cast_num + 1,
                                core.VarDesc.VarType.FP32,
                                core.VarDesc.VarType.BF16, out_var_name,
                                op_var_rename_map)
                            num_cast_ops += post_cast_num
            idx += num_cast_ops + 1

    _rename_op_input(program, op_var_rename_map, origin_ops, keep_fp32_ops)
    return to_bf16_var_names


def cast_parameters_to_bf16(place, program, scope=None, to_bf16_var_names=None):
    """
    Traverse all parameters in the whole model and set them to the BF16 data type.
    Whereas, this function will keep parameters of batchnorms in FP32.
    Args:
        place(fluid.CPUPlace|fluid.CUDAPlace): `place` is used to restore the BF16 weight tensors.
        program (Program): The used program.
        scope(fluid.Scope, optional): `scope` is used to get the FP32 weight tensor values.
                                      Default is None.
        to_bf16_var_names(set|list, optional): The data types of vars in `to_bf16_var_names`
                                               will be set to BF16. Usually, it is the returned
                                               value of `cast_model_to_bf16` API.
    """
    all_parameters = []
    for block in program.blocks:
        all_parameters.extend(block.all_parameters())

    bf16_var_names = to_bf16_var_names if to_bf16_var_names else set()
    var_scope = scope if scope else global_scope()
    for param in all_parameters:
        if param.name in bf16_var_names:
            _logger.debug("---- cast {} to bf16 dtype ----".format(param.name))
            param_t = var_scope.find_var(param.name).get_tensor()
            data = np.array(param_t)
            param_t.set(convert_float_to_uint16(data), place)


def rewrite_program_bf16(main_prog, amp_lists=None):
    """
    Traverse all ops in current block and insert cast op according to
    which set current op belongs to.

    1. When an op belongs to the fp32 list, add it to fp32 set
    2. When an op belongs to the bf16 list, add it to bf16 set
    3. When an op belongs to the gray list. If one
       of its inputs is the output of fp32 set op or fp32 list op,
       add it to fp32 set. If all of its previous ops are not fp32
       op and one of its inputs is the output of bf16 set op or
       bf16 list op, add it to bf16 set.
    4. When an op isn't in the lists, add it to fp32 op set.
    5. Add necessary cast ops to make sure that fp32 set op will be
       computed in fp32 mode, while bf16 set op will be computed in
       bf16 mode.

    Args:
        main_prog (Program): The main program for training.
    """
    if amp_lists is None:
        amp_lists = AutoMixedPrecisionListsBF16()
    block = main_prog.global_block()
    ops = block.ops
    bf16_op_set = set()
    fp32_op_set = set()
    for op in ops:

        # NOTE(zhiqiu): 'create_py_reader' and 'read' is used in non-iterable DataLoder,
        # we don't need to handle reader op and the input of 'create_py_reader' is not
        # in block, which may result in errors.
        # See GeneratorLoader._init_non_iterable() for details.
        if op.type == 'create_py_reader' or op.type == 'read':
            continue

        if amp_lists.fp32_varnames is not None and _is_in_fp32_varnames(
                op, amp_lists):
            fp32_op_set.add(op)
            continue

        if op.type in amp_lists.fp32_list:
            fp32_op_set.add(op)
        elif op.type in amp_lists.bf16_list:
            bf16_op_set.add(op)
        elif op.type in amp_lists.gray_list:
            is_fp32_op = False
            is_bf16_op = False
            for in_name in op.input_names:
                # if this op has inputs
                if in_name:
                    for in_var_name in op.input(in_name):
                        in_var = block.var(in_var_name)
                        # this in_var isn't the output of other op
                        if in_var.op is None:
                            continue
                        elif in_var.op is op:
                            prev_op = find_true_prev_op(ops, op, in_var_name)
                            if prev_op is None:
                                continue
                        else:
                            prev_op = in_var.op
                        # if it's one of inputs
                        if prev_op in fp32_op_set or \
                                prev_op.type in amp_lists.fp32_list:
                            is_fp32_op = True
                        elif prev_op in bf16_op_set or \
                                prev_op.type in amp_lists.bf16_list:
                            is_bf16_op = True
            if is_fp32_op:
                fp32_op_set.add(op)
            elif is_bf16_op:
                bf16_op_set.add(op)
            else:
                pass
        else:
            # For numerical safe, we apply fp32 computation on ops that
            # are not determined which list they should stay.
            fp32_op_set.add(op)

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        num_cast_ops = 0
        if op in fp32_op_set:
            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.BF16,
                                           core.VarDesc.VarType.FP32)
        elif op in bf16_op_set:
            if op.has_attr('use_mkldnn'):
                op._set_attr('use_mkldnn', True)
                op._set_attr('mkldnn_data_type', 'bfloat16')
            elif op.has_attr('dtype') and op.attr(
                    'dtype') == core.VarDesc.VarType.FP32:
                op._set_attr('dtype', core.VarDesc.VarType.BF16)

            num_cast_ops = _insert_cast_op(block, op, idx,
                                           core.VarDesc.VarType.FP32,
                                           core.VarDesc.VarType.BF16)
        else:
            pass

        idx += num_cast_ops + 1
