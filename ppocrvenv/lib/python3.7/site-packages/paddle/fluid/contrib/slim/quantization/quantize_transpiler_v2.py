#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import numpy as np
from .... import core
from ....framework import Program, Operator, Variable, program_guard
from ....executor import global_scope
from .... import unique_name
from ....layer_helper import LayerHelper
from ....param_attr import ParamAttr
from ....initializer import Constant
from ....log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def find_next_ops(block, var_name):
    """
    Find all followed ops for the input variable.
    """
    res_ops = []
    for op in block.ops:
        if var_name in op.input_arg_names:
            res_ops.append(op)
    return res_ops


def load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Cannot find " + var_name + " in scope."
    return np.array(var_node.get_tensor())


class QuantizeTranspilerV2(object):
    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='moving_average_abs_max',
                 quantizable_op_type=[
                     'conv2d',
                     'depthwise_conv2d',
                     'mul',
                 ],
                 skip_pattern=['skip_quant']):
        """
        Apply fake quant for the quantized ops. 

        Args:
            weight_bits(int): the bit of quantized weight.
            activation_bits(int): the bit of quantized activation.
            weight_quantize_type(str): the quantization type for weight.
                Only support to be 'abs_max' and 'channel_wise_abs_max'.
            activation_quantize_type(str): the quantization type for activation.
                Only support to be 'abs_max' and 'moving_average_abs_max'.
            quantizable_op_type(str): set the op type for quantization.
            skip_pattern(str|list): The user-defined quantization skip pattern, which
                will be presented in the name scope of an op. When the skip pattern is
                detected in an op's name scope, the corresponding op will not be quantized.
        """
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits

        assert activation_quantize_type in \
            ["abs_max", "moving_average_abs_max"], \
            "activation_quantize_type should be abs_max " \
            "or moving_average_abs_max for now."
        assert weight_quantize_type in ["abs_max", "channel_wise_abs_max"], \
            "weight_quantize_type should be abs_max or channel_wise_abs_max."
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type

        for op_type in quantizable_op_type:
            assert op_type in ['conv2d', 'depthwise_conv2d', 'mul'], \
                "Quantize op should be ['conv2d', 'depthwise_conv2d', 'mul']"
        self._quantizable_ops = quantizable_op_type
        self._quantizable_grad_ops = [
            '%s_grad' % (op) for op in self._quantizable_ops
        ]

        self._skip_pattern = skip_pattern
        self._helper = LayerHelper(self.__class__.__name__)

        self._moving_rate = 0.9
        self._out_ch_axis1_ops = ['conv2d_transpose', 'mul', 'matmul']

    def apply(self, program, startup_program, is_test=False):
        """
        Apply quantization to fluid Program.

        Args:
            program(Program): the train or test program to be quantized.
            startup_program(Program): the corresponding startup_program.
            is_test(bool): Whethe the program is used for test.
        Returns:
            None
        """
        assert isinstance(program, Program), \
            "program must be the instance of Program"
        assert isinstance(startup_program, Program), \
            "startup_program must be the instance of Program"

        var_rename_map = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        with program_guard(program, startup_program):
            for block in program.blocks:
                ops = list(block.ops)
                for op in ops:
                    if op.type in self._quantizable_ops and \
                        (not self._is_skip_quant(op)):
                        self._transform_forward(block, op, var_rename_map,
                                                is_test)

            for block in program.blocks:
                ops = list(block.ops)
                for op in ops:
                    if op.type in self._quantizable_grad_ops and \
                        (not self._is_skip_quant(op)):
                        self._transform_backward(block, op, var_rename_map)

    def convert(self, test_program, scope=None):
        """
        Convert the test program. 
        Get the out scale from the moving_average_abs_max_scale op and save the
        out scale into the quantized op. 
        Args:
            test_program(Program): the test program to be converted.
            scope(fluid.Scope, optional): The scope of the program, use it to load 
                and save variables. If scope=None, get scope by global_scope(). 
        """
        scope = global_scope() if scope == None else scope

        for block in test_program.blocks:
            for op in block.ops:
                if op.has_attr("quantization_type") \
                    and op.attr("quantization_type") == "qat_with_weight":
                    # quant op -> var1 -> fake op -> var2
                    assert len(op.output_arg_names) == 1
                    var1_name = op.output_arg_names[0]

                    fake_ops = find_next_ops(block, var1_name)
                    assert len(fake_ops) == 1
                    fake_op = fake_ops[0]
                    assert fake_op.type == "moving_average_abs_max_scale"

                    out_scale_name = fake_op.output("OutScale")
                    out_threshold = load_variable_data(scope, out_scale_name[0])
                    op._set_attr("out_threshold", float(out_threshold))

                    var2_name = fake_op.output("Out")[0]
                    op._rename_output(var1_name, var2_name)
                    fake_op._rename_output(var2_name, var1_name)

    def _transform_forward(self, block, op, var_rename_map, is_test):
        """
        Insert fake quant op before the target ops.
        """
        op._set_attr("quantization_type", "qat_with_weight")

        # insert fake quant op before the quantized op
        for in_name in op.input_arg_names:
            block_id = block.idx
            idx = block.ops.index(op)

            if in_name in var_rename_map[block_id]:
                new_in_name = var_rename_map[block_id][in_name]
            else:
                in_var = block.var(in_name)
                target_dtype = [
                    core.VarDesc.VarType.FP32, core.VarDesc.VarType.FP16
                ]
                if in_var.dtype not in target_dtype:
                    continue

                quant_bits = self._weight_bits if in_var.persistable \
                        else self._activation_bits
                quant_type = self._weight_quantize_type if in_var.persistable \
                        else self._activation_quantize_type

                if quant_type == "abs_max":
                    new_var = self._insert_abs_max_fq_op(block, idx, in_var,
                                                         quant_bits)
                elif quant_type == "moving_average_abs_max":
                    new_var = self._insert_ma_abs_max_fq_op(block, idx, in_var,
                                                            quant_bits, is_test)
                elif quant_type == "channel_wise_abs_max":
                    ch_axis = 1 if op.type in self._out_ch_axis1_ops else 0
                    new_var = self._insert_pc_abs_max_fq_op(block, idx, in_var,
                                                            quant_bits, ch_axis)
                else:
                    _logger.error("Don't support the quant_type: %s" %
                                  quant_type)
                    continue

                new_in_name = new_var.name
                var_rename_map[block_id][in_name] = new_in_name

            op._rename_input(in_name, new_in_name)

        # insert out scale op followed the quantized op
        for out_name in op.output_arg_names:
            next_ops = find_next_ops(block, out_name)

            idx = block.ops.index(op)
            out_var = block.var(out_name)
            new_out_var = self._insert_ma_abs_max_scale_op(
                block, idx + 1, out_var, is_test, True)

            for next_op in next_ops:
                if "_grad" not in next_op.type:
                    next_op._rename_input(out_name, new_out_var.name)

    def _is_skip_quant(self, op):
        """
        Analyse whether the op should skip quantization or not.
        """
        user_skipped = False
        if isinstance(self._skip_pattern, list):
            user_skipped = op.has_attr("op_namescope") and \
                            any(pattern in op.attr("op_namescope") \
                                for pattern in self._skip_pattern)
        elif isinstance(self._skip_pattern, str):
            user_skipped = op.has_attr("op_namescope") and \
                            op.attr("op_namescope").find(
                                self._skip_pattern) != -1
        return user_skipped

    def _transform_backward(self, block, op, var_rename_map):
        """
        Update the backword of the target ops.
        Note: for the grad ops, only rename the input, skip rename the output.
        """
        block_id = block.idx
        no_dequanted_input_vars = True
        for name in op.input_arg_names:
            if name in var_rename_map[block_id]:
                new_var_name = var_rename_map[block_id][name]
                op._rename_input(name, new_var_name)
                no_dequanted_input_vars = False
        if no_dequanted_input_vars:
            raise ValueError("There is no dequanted inputs for op %s." %
                             (op.type))

    def _insert_abs_max_fq_op(self, block, idx, in_var, quant_bits):
        """
        Inset abs max fake quant op.
        """
        quant_dequant_var = block.create_var(
            type=in_var.type,
            name="{}.quant_dequant".format(in_var.name),
            shape=in_var.shape,
            dtype=in_var.dtype)
        scale_var = self._helper.create_parameter(
            attr=ParamAttr(
                name="{}.quant_dequant.scale".format(in_var.name),
                initializer=Constant(0.),
                trainable=False),
            shape=[1],
            dtype=in_var.dtype)
        scale_var.stop_gradient = True

        inputs = {'X': in_var}
        outputs = {'Out': quant_dequant_var, 'OutScale': scale_var}
        attrs = {'bit_length': quant_bits}
        block._insert_op(
            idx,
            type='fake_quantize_dequantize_abs_max',
            attrs=attrs,
            inputs=inputs,
            outputs=outputs)
        return quant_dequant_var

    def _insert_ma_abs_max_fq_op(self, block, idx, in_var, quant_bits, is_test):
        """
        Insert moving average abs max fake quant op.
        """
        quant_dequant_var = block.create_var(
            type=in_var.type,
            name="{}.quant_dequant".format(in_var.name),
            shape=in_var.shape,
            dtype=in_var.dtype)

        scale_var = self._helper.create_parameter(
            attr=ParamAttr(
                name="{}.quant_dequant.scale".format(in_var.name),
                initializer=Constant(0.),
                trainable=False),
            shape=[1],
            dtype=in_var.dtype)
        scale_var.stop_gradient = True

        if not is_test:
            state_var = self._helper.create_parameter(
                attr=ParamAttr(
                    name="{}.quant_dequant.state".format(in_var.name),
                    initializer=Constant(0),
                    trainable=False),
                shape=[1],
                dtype=in_var.dtype)
            state_var.stop_gradient = True

            accum_var = self._helper.create_parameter(
                attr=ParamAttr(
                    name="{}.quant_dequant.accum".format(in_var.name),
                    initializer=Constant(0),
                    trainable=False),
                shape=[1],
                dtype=in_var.dtype)
            accum_var.stop_gradient = True

        attrs = {
            'moving_rate': self._moving_rate,
            'bit_length': quant_bits,
            'is_test': is_test
        }
        inputs = {'X': in_var, 'InScale': scale_var}
        outputs = {'Out': quant_dequant_var, 'OutScale': scale_var}
        if not is_test:
            inputs['InState'] = state_var
            inputs['InAccum'] = accum_var
            outputs['OutState'] = state_var
            outputs['OutAccum'] = accum_var

        block._insert_op(
            idx,
            type='fake_quantize_dequantize_moving_average_abs_max',
            attrs=attrs,
            inputs=inputs,
            outputs=outputs)
        return quant_dequant_var

    def _insert_pc_abs_max_fq_op(self, block, idx, in_var, quant_bits, ch_axis):
        """
        Insert per channel abs max fake quant op.
        """
        quant_dequant_var = block.create_var(
            type=in_var.type,
            name="{}.quant_dequant".format(in_var.name),
            shape=in_var.shape,
            dtype=in_var.dtype)

        scale_var = self._helper.create_parameter(
            attr=ParamAttr(
                name="{}.quant_dequant.scale".format(in_var.name),
                initializer=Constant(0.),
                trainable=False),
            shape=[in_var.shape[ch_axis]],
            dtype=in_var.dtype)
        scale_var.stop_gradient = True

        inputs = {'X': in_var}
        outputs = {'Out': quant_dequant_var, 'OutScale': scale_var}
        attrs = {'bit_length': quant_bits, 'quant_axis': ch_axis}
        block._insert_op(
            idx,
            type='fake_channel_wise_quantize_dequantize_abs_max',
            attrs=attrs,
            inputs=inputs,
            outputs=outputs)
        return quant_dequant_var

    def _insert_ma_abs_max_scale_op(self,
                                    block,
                                    idx,
                                    in_var,
                                    is_test,
                                    has_out_var=False):
        """
        Insert moving average abs max scale op.
        """
        scale_var = self._helper.create_parameter(
            attr=ParamAttr(
                name="{}.outscale.scale".format(in_var.name),
                initializer=Constant(0.),
                trainable=False),
            shape=[1],
            dtype=in_var.dtype)
        scale_var.stop_gradient = True

        attrs = {'moving_rate': self._moving_rate, 'is_test': is_test}
        inputs = {'X': in_var}
        outputs = {'OutScale': scale_var}

        if not is_test:
            state_var = self._helper.create_parameter(
                attr=ParamAttr(
                    name="{}.outscale.state".format(in_var.name),
                    initializer=Constant(0),
                    trainable=False),
                shape=[1],
                dtype=in_var.dtype)
            state_var.stop_gradient = True

            accum_var = self._helper.create_parameter(
                attr=ParamAttr(
                    name="{}.outscale.accum".format(in_var.name),
                    initializer=Constant(0),
                    trainable=False),
                shape=[1],
                dtype=in_var.dtype)
            accum_var.stop_gradient = True

            inputs['InState'] = state_var
            inputs['InAccum'] = accum_var
            outputs['OutState'] = state_var
            outputs['OutAccum'] = accum_var

        if has_out_var:
            out_var = block.create_var(
                type=in_var.type,
                name="{}.tmp".format(in_var.name),
                shape=in_var.shape,
                dtype=in_var.dtype)

            outputs['Out'] = out_var

        block._insert_op(
            idx,
            type='moving_average_abs_max_scale',
            attrs=attrs,
            inputs=inputs,
            outputs=outputs)

        if has_out_var:
            return out_var
