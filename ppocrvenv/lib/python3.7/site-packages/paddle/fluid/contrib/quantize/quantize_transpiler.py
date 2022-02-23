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

import collections
import numpy as np

from paddle.fluid.framework import default_main_program, default_startup_program, program_guard
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import unique_name
from paddle.fluid import core
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.nn import autoincreased_step_counter
from paddle.fluid.framework import Variable
from paddle.fluid.executor import global_scope

__all__ = ['QuantizeTranspiler']

_QUANTIZABLE_OP_TYPES = ['conv2d', 'depthwise_conv2d', 'mul']


def _quantized_var_name(var_name):
    """
    Return quantized variable name for the input `var_name`.
    """
    return "%s.quantized" % (var_name)


def _dequantized_var_name(var_name):
    """
    Return dequantized variable name for the input `var_name`.
    """
    return "%s.dequantized" % (var_name)


def _quantized_scale_name(var_name):
    """
    Return quantized variable name for the input `var_name`.
    """
    return "%s.scale" % (var_name)


def _original_var_name(var_name):
    """
    Return the original variable name.
    """
    if var_name.endswith('.quantized.dequantized'):
        return var_name[:-len('.quantized.dequantized')]
    if var_name.endswith('.quantized'):
        return var_name[:-len('.quantized')]
    if var_name.endswith('.dequantized'):
        return var_name[:-len('.dequantized')]
    if var_name.endswith('.scale'):
        return var_name[:-len('.scale')]
    else:
        return var_name


def _is_float(v):
    return isinstance(v, float) or isinstance(v, np.float32)


def quant(x, scale, num_bits):
    y = np.round(x / scale * ((1 << (num_bits - 1)) - 1))
    return y


class QuantizeTranspiler(object):
    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 activation_quantize_type='abs_max',
                 weight_quantize_type='abs_max',
                 window_size=10000,
                 moving_rate=0.9):
        """
        Convert and rewrite the fluid Program according to weight and
        activation quantization type.

        Args:
            weight_bits (int): quantization bit number for weights,
                the bias is not quantized.
            activation_bits (int): quantization bit number for activation.
            activation_quantize_type (str): quantization type for activation,
                now support 'abs_max', 'range_abs_max'. If use 'abs_max' mode,
                the quantization scale will be calculated dynamically each step
                in both training and testing period. If use 'range_abs_max',
                a static quantization scale will be calculated during training
                and used in inference.
            weight_quantize_type (str): quantization type for weights,
                support 'abs_max'. The 'range_abs_max' usually is not used for
                weight, since weights are fixed once the model is well trained.
            window_size (int): the window size for 'range_abs_max' quantization.

        Examples:

        .. code-block:: python

            # the original program will be rewrite, if you don't want to
            # change it, please clone at first.
            # quantize_program = program.clone()
            t = fluid.QuantizeTranspiler()
            t.transpile(quantize_program)

        """
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        quant_type = ['abs_max', 'range_abs_max', 'moving_average_abs_max']
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be ",
                "'abs_max' or 'range_abs_max' or 'moving_average_abs_max'.",
                str(weight_quantize_type))
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be ",
                "'abs_max' or 'range_abs_max' or 'moving_average_abs_max'.",
                str(activation_quantize_type))

        self.weight_quantize_type = weight_quantize_type
        self.activation_quantize_type = activation_quantize_type

        self.window_size = window_size
        self.moving_rate = moving_rate
        self.helper = LayerHelper(self.__class__.__name__)
        self.fake_quant_op_types = [
            'fake_quantize_abs_max', 'fake_quantize_range_abs_max',
            'fake_quantize_moving_average_abs_max'
        ]
        self.fake_dequant_op_types = ['fake_dequantize_max_abs']
        self.is_test = None
        self.global_step = None

    def training_transpile(self, program=None, startup_program=None):
        """Rewrites a training input program in place for simulated
        quantization. Insert fake quantization and de-quantization ops into
        program to simulate the error introduced by quantization. And change
        the gradient ops' input by using the faked quantization weights and
        activation. Since the program is transformed in place, the graph
        connection will change.

        Args:
            program (Program): the input program to be transpile.
        """
        self.is_test = False
        program = default_main_program() if program is None else program
        startup_program = default_startup_program() if startup_program is \
            None else startup_program

        # marked the variable which has been quantized and dequantized.
        dequanted_vars = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        grad_op_types = ['%s_grad' % (type) for type in _QUANTIZABLE_OP_TYPES]

        params = [p.name for p in program.global_block().iter_parameters()]

        def _transpile_forward(block, op):
            idx = block.ops.index(op)
            block_id = block.idx
            # insert quant op and dequant op
            for name in op.input_arg_names:
                #if share input between ops
                if name in dequanted_vars[block_id]:
                    dequant_var = dequanted_vars[block_id][name]
                else:
                    var = block.var(name)
                    quant_bits = self.weight_bits if var.name in params \
                                 else self.activation_bits
                    quant_type = self.weight_quantize_type if var.name \
                        in params else self.activation_quantize_type

                    quant_var, scale_var = self._insert_quant_op(
                        block, idx, var, quant_bits, quant_type)
                    dequant_var = self._insert_dequant_op(
                        block, idx + 1, quant_var, scale_var, quant_bits)
                    dequanted_vars[block_id][name] = dequant_var
                # rename the forward op inputs
                op._rename_input(name, dequant_var.name)

        def _transpile_backward(block, op):
            block_id = block.idx
            no_dequanted_input_vars = True
            for name in op.input_arg_names:
                if name in dequanted_vars[block_id]:
                    dequant_var = dequanted_vars[block_id][name]
                    op._rename_input(name, dequant_var.name)
                    no_dequanted_input_vars = False
            if no_dequanted_input_vars:
                raise ValueError("There is no dequanted inputs for op %s." %
                                 (op.type))

        with program_guard(program, startup_program):
            self._create_global_step()
            for block in program.blocks:
                ops = list(block.ops)
                block_id = block.idx
                for op in ops:
                    # rewrite the forward ProgramDes
                    if op.type in _QUANTIZABLE_OP_TYPES:
                        _transpile_forward(block, op)
                    # rename the backward op inputs
                    if op.type in grad_op_types:
                        _transpile_backward(block, op)

    def _create_global_step(self):
        if self.weight_quantize_type == 'range_abs_max' or \
            self.activation_quantize_type == 'range_abs_max':
            self.global_step = autoincreased_step_counter()

    def freeze_program(self, program, place, scope=None):
        """Freeze input training program for inference.

        Args:
            program (Program): the input program to be transpile.
        """

        self.is_test = True
        scope = global_scope() if scope is None else scope
        program = default_main_program() if program is None else program

        persistable_vars = [
            v.name
            for v in filter(lambda var: var.persistable, program.list_vars())
        ]
        op_in_rename_map = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        op_out_rename_map = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        var_scale_map = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]

        def _remove_fake_quant_and_dequant_op(block, op):
            idx = block.ops.index(op)
            block_id = block.idx
            k = op.output('Out')[0]
            v = op.input('X')[0]
            if v not in op_in_rename_map[block_id]:
                op_in_rename_map[block_id][k] = v
            else:
                op_in_rename_map[block_id][k] = op_in_rename_map[block_id][v]
            block._remove_op(idx)

        def _insert_post_dequant_op(block, op):
            idx = block.ops.index(op)
            block_id = block.idx
            max_range = None
            scale_var = None
            for name in op.input_arg_names:
                #rename input name of the op to the input name of last op which has be removed
                if name in op_in_rename_map[block_id]:
                    op._rename_input(name, op_in_rename_map[block_id][name])

                scale_v = var_scale_map[block_id][_original_var_name(name)]
                if _original_var_name(name) in persistable_vars:
                    param_range = (1 << (self.weight_bits - 1)) - 1
                    act_range = (1 << (self.activation_bits - 1)) - 1
                    assert _is_float(scale_v)
                    max_range = param_range * act_range / scale_v
                else:
                    assert isinstance(scale_v, Variable)
                    scale_var = scale_v

            if len(op.output_arg_names) != 1:
                raise ValueError("Only support one output, but op %s has"
                                 " more than one output." % (op.type))
            out_var = block.var(op.output_arg_names[0])
            dequant_var = block.create_var(
                name=_dequantized_var_name(out_var.name),
                type=out_var.type,
                shape=out_var.shape,
                dtype=out_var.dtype)
            # insert fake_dequantize_op
            dequant_op = block._insert_op(
                idx + 1,
                type="fake_dequantize_max_abs",
                attrs={'max_range': float(max_range)},
                inputs={"X": out_var,
                        'Scale': scale_var},
                outputs={"Out": dequant_var})
            op_out_rename_map[block_id][out_var.name] = dequant_var.name
            return dequant_var

        def _load_var(name):
            return np.array(scope.find_var(name).get_tensor())

        def _restore_var(name, arr):
            t = scope.find_var(name).get_tensor()
            t.set(arr, place)

        for block in program.blocks:
            ops = list(block.ops)
            block_id = block.idx
            for op in ops:
                op_type = op.type

                # insert dequant_op after fc/conv, need to rename
                # input of the followed ops(of fc/conv) to the dquant_op
                for name in op.input_arg_names:
                    if name in op_out_rename_map[block_id]:
                        op._rename_input(name,
                                         op_out_rename_map[block_id][name])

                if op_type in self.fake_quant_op_types:
                    in_arg_name = op.input('X')[0]
                    if in_arg_name in persistable_vars:
                        if self.weight_quantize_type == 'abs_max':
                            param = _load_var(in_arg_name)
                            scale_v = np.max(np.abs(param))
                        else:
                            scale_v = _load_var(op.output('OutScale')[0])
                        var_scale_map[block_id][in_arg_name] = scale_v
                    else:
                        scale_v = block.var(op.output('OutScale')[0])
                        var_scale_map[block_id][in_arg_name] = scale_v

                    if in_arg_name in persistable_vars:
                        _remove_fake_quant_and_dequant_op(block, op)
                        # quantize weight and restore
                        param_t = _load_var(in_arg_name)
                        param_q_t = quant(param_t, scale_v, self.weight_bits)
                        _restore_var(in_arg_name, param_q_t)

                if op_type in self.fake_dequant_op_types:
                    _remove_fake_quant_and_dequant_op(block, op)

                if op_type in _QUANTIZABLE_OP_TYPES:
                    dequant_var = _insert_post_dequant_op(block, op)

        # remove the unused var in ProgramDesc
        self._remove_unused_var(program)
        #program = program.clone()

    def convert_to_int8(self, program, place, scope=None):
        scope = global_scope() if scope is None else scope
        program = default_main_program() if program is None else program

        def _load_var(name):
            return np.array(scope.find_var(name).get_tensor())

        global_block = program.global_block()

        def convert_to_int8(var):
            int8_var_name = var.name + ".int8"
            int8_var = global_block.create_parameter(
                name=int8_var_name.encode('ascii'),
                type=var.type,
                dtype=core.VarDesc.VarType.INT8,
                shape=var.shape)

            tensor = _load_var(var.name)

            scope.var(int8_var_name)
            int8_tensor = scope.find_var(int8_var_name).get_tensor()
            int8_tensor.set(tensor.astype(np.int8), place)
            return int8_var

        input_map = {}
        for block in program.blocks:
            for op in list(block.ops):
                if op.type in _QUANTIZABLE_OP_TYPES:
                    for name in op.input_arg_names:
                        var = block.var(name)
                        if var.persistable:
                            if name not in input_map:
                                int8_var = convert_to_int8(var)
                                input_map[name] = int8_var.name
                            op._rename_input(name, input_map[name])
        self._remove_unused_var(program)

    def _remove_unused_var(self, program):
        all_remove_vars = []
        for block in program.blocks:
            args = []
            for op in block.ops:
                args += op.input_arg_names
                args += op.output_arg_names
            args = list(set(args))  #vals of all left ops
            var_names = block.vars.keys()  # all vals
            sub_block_remove_vars = []
            for var in var_names:
                if var not in args:
                    sub_block_remove_vars.append(var)
            all_remove_vars.append(sub_block_remove_vars)

        remove_vars = [list(set(v)) for v in all_remove_vars]
        for i, block in enumerate(program.blocks):
            for v in remove_vars[i]:
                block._remove_var(v)

    def _insert_quant_abs_max_op(self, block, idx, var, quant_bits):
        """Insert fake_quantize_abs_max op.
        """
        quant_var = block.create_var(
            name=_quantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        scale = block.create_var(
            name=_quantized_scale_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        quant_op = block._insert_op(
            idx,
            type='fake_quantize_abs_max',
            attrs={'bit_length': quant_bits},
            inputs={'X': var},
            outputs={'Out': quant_var,
                     'OutScale': scale})
        return quant_var, scale

    def _insert_quant_range_abs_max_op(self, block, idx, var, quant_bits):
        """Insert fake_quantize_range_abs_max
        """
        quant_var = block.create_var(
            name=_quantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        scale = self.helper.create_parameter(
            attr=ParamAttr(
                name=_quantized_scale_name(var.name),
                initializer=Constant(0.001),
                trainable=False),
            shape=[1],
            dtype=var.dtype)
        scale.stop_gradient = True

        ins = {'X': var, 'InScale': scale}
        outs = {'Out': quant_var, 'OutScale': scale}
        if not self.is_test:
            # A global step counter variable with type int64
            scales = self.helper.create_global_variable(
                name=unique_name.generate('scales'),
                persistable=True,
                dtype=var.dtype,
                shape=[self.window_size])
            self.helper.set_variable_initializer(
                scales, initializer=Constant(value=0))

            ins['Iter'] = self.global_step
            outs['OutScales'] = scales

        attrs = {
            'window_size': self.window_size,
            'bit_length': quant_bits,
            'is_test': self.is_test
        }

        quant_op = block._insert_op(
            idx,
            type='fake_quantize_range_abs_max',
            attrs=attrs,
            inputs=ins,
            outputs=outs)

        return quant_var, scale

    def _insert_quant_moving_average_abs_max_op(self, block, idx, var,
                                                quant_bits):
        """Insert fake_quantize_moving_average_abs_max
        """
        quant_var = block.create_var(
            name=_quantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        state = self.helper.create_global_variable(
            name=unique_name.generate('state'),
            persistable=True,
            dtype=var.dtype,
            shape=[1])
        self.helper.set_variable_initializer(
            state, initializer=Constant(value=1))
        accum = self.helper.create_global_variable(
            name=unique_name.generate('accum'),
            persistable=True,
            dtype=var.dtype,
            shape=[1])
        self.helper.set_variable_initializer(
            accum, initializer=Constant(value=1))
        scale = self.helper.create_parameter(
            attr=ParamAttr(
                name=_quantized_scale_name(var.name),
                initializer=Constant(0.001),
                trainable=False),
            shape=[1],
            dtype=var.dtype)
        scale.stop_gradient = True

        ins = {'X': var, 'InScale': scale}
        outs = {'Out': quant_var, 'OutScale': scale}
        if not self.is_test:
            ins['InState'] = state
            ins['InAccum'] = accum
            outs['OutState'] = state
            outs['OutAccum'] = accum

        attrs = {
            'bit_length': quant_bits,
            'moving_rate': self.moving_rate,
            'is_test': self.is_test
        }

        quant_op = block._insert_op(
            idx,
            type='fake_quantize_moving_average_abs_max',
            attrs=attrs,
            inputs=ins,
            outputs=outs)

        return quant_var, scale

    def _insert_quant_op(self, block, idx, var, quant_bits, quant_type):
        """
        Insert fake_quantize_op
        """
        if quant_type == 'abs_max':
            return self._insert_quant_abs_max_op(block, idx, var, quant_bits)
        elif quant_type == 'range_abs_max':
            return self._insert_quant_range_abs_max_op(block, idx, var,
                                                       quant_bits)
        elif quant_type == 'moving_average_abs_max':
            return self._insert_quant_moving_average_abs_max_op(block, idx, var,
                                                                quant_bits)

    def _insert_dequant_op(self, block, idx, var, scale, quant_bits):
        """
        Insert fake_quantize_op
        """
        dequant_var = block.create_var(
            name=_dequantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        # insert fake_dequantize_op
        max_range = (1 << (quant_bits - 1)) - 1
        dequant_op = block._insert_op(
            idx,
            type="fake_dequantize_max_abs",
            attrs={'max_range': float(max_range)},
            inputs={"X": var,
                    'Scale': scale},
            outputs={"Out": dequant_var})
        return dequant_var
