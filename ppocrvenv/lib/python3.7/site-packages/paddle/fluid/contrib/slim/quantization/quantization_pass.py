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
from ..... import compat as cpt
from .... import core
from ....framework import IrGraph
from ....framework import IrNode
from ....framework import Operator
from .... import unique_name

from ....framework import Program, program_guard, default_startup_program
from ....data import data
from ....layers import mean
from ....executor import scope_guard
from ....framework import _get_paddle_place

__all__ = [
    'QuantizationTransformPass', 'QuantizationFreezePass', 'ConvertToInt8Pass',
    'TransformForMobilePass', 'OutScaleForTrainingPass',
    'OutScaleForInferencePass', 'AddQuantDequantPass'
]

_fake_quant_op_list = [
    'fake_quantize_abs_max', 'fake_quantize_range_abs_max',
    'fake_quantize_moving_average_abs_max', 'fake_channel_wise_quantize_abs_max'
]

_fake_dequant_op_list = [
    'fake_dequantize_max_abs', 'fake_channel_wise_dequantize_max_abs'
]

_fake_quant_dequant_op_list = [
    'fake_quantize_dequantize_moving_average_abs_max'
]

_out_scale_op_list = [
    "conv2d",
    "depthwise_conv2d",
    "mul",
    "matmul",
    "matmul_v2",
    "relu",
    "leaky_relu",
    "relu6",
    "sigmoid",
    "tanh",
    "prelu",
    "swish",
    "softmax",
    "batch_norm",
    "layer_norm",
    "elementwise_add",
    "pool2d",
    "reshape2",
    "transpose2",
    "concat",
    "elementwise_mul",
    "scale",
    "slice",
    "hard_swish",
    "hard_sigmoid",
    "conv2d_transpose",
    "gru",
    "bilinear_interp",
    "nearest_interp",
    "trilinear_interp",
    "flatten",
    "flatten2",
    "transpose",
    "pad2d",
    "reshape",
    "layer_norm",
]

# list op real input and output names, to avoid processing input such as AxisTensor.
_op_real_in_out_name = {
    "conv2d": [["Input", "Filter"], ["Output"]],
    "depthwise_conv2d": [["Input", "Filter"], ["Output"]],
    "conv2d_transpose": [["Input", "Filter"], ["Output"]],
    "mul": [["X", "Y"], ["Out"]],
    "matmul": [["X", "Y"], ["Out"]],
    "matmul_v2": [["X", "Y"], ["Out"]],
    "pool2d": [["X"], ["Out"]],
    "elementwise_add": [["X", "Y"], ["Out"]],
    "concat": [["X"], ["Out"]],
    "softmax": [["X"], ["Out"]],
    "argmax": [["X"], ["Out"]],
    "transpose": [["X"], ["Out"]],
    "equal": [["X", "Y"], ["Out"]],
    "gather": [["X"], ["Out"]],
    "greater_equal": [["X", "Y"], ["Out"]],
    "greater_than": [["X", "Y"], ["Out"]],
    "less_equal": [["X", "Y"], ["Out"]],
    "less_than": [["X", "Y"], ["Out"]],
    "mean": [["X"], ["Out"]],
    "not_equal": [["X", "Y"], ["Out"]],
    "reshape": [["X"], ["Out"]],
    "reshape2": [["X"], ["Out"]],
    "transpose2": [["X"], ["Out"]],
    "bilinear_interp": [["X"], ["Out"]],
    "nearest_interp": [["X"], ["Out"]],
    "trilinear_interp": [["X"], ["Out"]],
    "slice": [["Input"], ["Out"]],
    "squeeze": [["X"], ["Out"]],
    "elementwise_sub": [["X", "Y"], ["Out"]],
    "relu": [["X"], ["Out"]],
    "relu6": [["X"], ["Out"]],
    "leaky_relu": [["X"], ["Out"]],
    "prelu": [["X"], ["Out"]],
    "tanh": [["X"], ["Out"]],
    "swish": [["X"], ["Out"]],
    "dropout": [["X"], ["Out"]],
    "batch_norm": [["X"], ["Y"]],
    "layer_norm": [["X"], ["Y"]],
    "sigmoid": [["X"], ["Out"]],
    "elementwise_mul": [["X", "Y"], ["Out"]],
    "scale": [["X"], ["Out"]],
    "hard_swish": [["X"], ["Out"]],
    "hard_sigmoid": [["X"], ["Out"]],
    "gru": [["Input", "Weight"], ["Hidden"]],
    "lstm": [["Input", "Weight"], ["Hidden"]],
    "pad2d": [["X"], ["Out"]],
    "flatten": [["X"], ["Out"]],
    "flatten2": [["X"], ["Out"]],
    "unsqueeze2": [["X"], ["Out"]],
    "flatten_contiguous_range": [['X'], ["Out"]],
}

_conv_ops = ['conv2d', 'depthwise_conv2d', 'conv2d_transpose']

_channelwise_quant_axis1_ops = [
    'conv2d_transpose', 'mul', 'matmul', 'matmul_v2'
]


def _get_op_input_var_names(op):
    """
    Get the input var names of the op.
    Args:
        op(IrNode, Operator): the input op.
    Returns:
        input_var_names or None.
    """
    assert isinstance(op, (IrNode, Operator)), \
        "The input op should be IrNode or Operator."
    var_names = []
    op_name = op.name() if isinstance(op, IrNode) \
        else op.type
    if op_name not in _op_real_in_out_name:
        return []

    name_list = _op_real_in_out_name[op_name][0]
    for name in name_list:
        var_name = op.input(name)
        if isinstance(var_name, list):
            var_names.extend(var_name)
        else:
            var_names.append(var_name)
    return var_names


def _get_input_name_index(op, input_var_name):
    """Get the input name and index of the var_name in the op"""
    assert isinstance(op, (IrNode, Operator)), \
        "The input op should be IrNode or Operator."
    op_name = op.name() if isinstance(op, IrNode) \
        else op.type
    if op_name not in _op_real_in_out_name:
        return None

    res = None
    for argname in _op_real_in_out_name[op_name][0]:
        var_names = op.input(argname)
        for index, name in enumerate(var_names):
            if name == input_var_name:
                res = (argname, index)
    return res


def _get_op_output_var_names(op):
    """ """
    assert isinstance(op, (IrNode, Operator)), \
        "The input op should be IrNode or Operator."
    var_names = []
    op_name = op.name() if isinstance(op, IrNode) \
        else op.type
    if op_name not in _op_real_in_out_name:
        return []

    name_list = _op_real_in_out_name[op_name][1]
    for name in name_list:
        var_name = op.output(name)
        if isinstance(var_name, list):
            var_names.extend(var_name)
        else:
            var_names.append(var_name)
    return var_names


def _get_output_name_index(op, output_var_name):
    """Get the output name and index of the var_name in the op"""
    assert isinstance(op, (IrNode, Operator)), \
        "The input op should be IrNode or Operator."
    op_name = op.name() if isinstance(op, IrNode) \
        else op.type
    if op_name not in _op_real_in_out_name:
        return None

    name_list = _op_real_in_out_name[op_name][1]
    res = None
    for name in name_list:
        var_name = op.output(name)
        for index, val in enumerate(var_name):
            if val == output_var_name:
                res = (name, index)
    return res


def _init_var_node(var_node, value, scope, place):
    assert isinstance(value,
                      np.ndarray), 'The type of value should be numpy array.'
    assert scope is not None, \
        'The scope cannot be set None.'
    assert place is not None, \
        'The place cannot be set None.'
    tensor = scope.var(var_node.name()).get_tensor()
    tensor.set(value, place)


def _is_input_all_not_persistable(graph, op_node):
    '''
    Analyse the real inputs of the op node are all not persistable.
    '''
    is_input_all_not_persistable = True
    for var_name in _get_op_input_var_names(op_node):
        in_node = graph._find_node_by_name(op_node.inputs, var_name)
        is_input_all_not_persistable = (is_input_all_not_persistable and \
            (not in_node.persistable()))
    return is_input_all_not_persistable


def _check_grandchild_op_node(op_node, grandchild_op_name):
    '''
    Check whether the fake_quant node has a grandchild op node named
    grandchild_op_name.
    '''
    for out1_var_node in op_node.outputs:
        for out1_op_node in out1_var_node.outputs:
            for out2_var_node in out1_op_node.outputs:
                for out2_op_node in out2_var_node.outputs:
                    if out2_op_node.name() == grandchild_op_name:
                        return True
    return False


class QuantizationTransformPass(object):
    """
    Quantize the ops that have weights. Add quant and dequant ops for
    the quantized ops's inputs.
    """
    _supported_quantizable_op_type = [
        'conv2d', 'depthwise_conv2d', 'conv2d_transpose', 'mul', 'matmul',
        'matmul_v2'
    ]

    def __init__(self,
                 scope=None,
                 place=None,
                 weight_bits=8,
                 activation_bits=8,
                 activation_quantize_type='abs_max',
                 weight_quantize_type='abs_max',
                 window_size=10000,
                 moving_rate=0.9,
                 skip_pattern=['skip_quant'],
                 quantizable_op_type=['conv2d', 'depthwise_conv2d', 'mul'],
                 weight_quantize_func=None,
                 act_quantize_func=None,
                 weight_preprocess_func=None,
                 act_preprocess_func=None,
                 optimizer_func=None,
                 executor=None):
        r"""
        Constructor.

        Args:
            scope(fluid.Scope): When activation use 'range_abs_max' as the quantize
                type, this pass will create some new parameters. The scope is used to
                initialize these new parameters.
            place(fluid.CPUPlace|fluid.CUDAPlace|str): place is used to initialize new
                parameters described above. If it's string, It can be ``cpu``, and ``gpu:x``,
                where ``x`` is the index of the GPUs. 
            weight_bits(int): quantization bit number for weights,
                the bias is not quantized.
            activation_bits(int): quantization bit number for activation.
            activation_quantize_type(str): quantization type for activation,
                now support 'abs_max', 'range_abs_max' and 'moving_average_abs_max'.
                If use 'abs_max' mode, the quantization scale will be calculated
                dynamically each step in both training and testing period. If use
                'range_abs_max', a static quantization scale will be calculated
                during training and used in inference.
            weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. The 'range_abs_max'
                usually is not used for weight, since weights are fixed once the
                model is well trained.
            window_size(int): the window size for 'range_abs_max' quantization.
            moving_rate(float): the param for 'moving_average_abs_max' quantization.
            skip_pattern(str or str list): The user-defined quantization skip pattern, which
                will be presented in the name scope of an op. When the skip pattern is
                detected in an op's name scope, the corresponding op will not be quantized. 
            quantizable_op_type(list[str]): List the type of ops that will be quantized. 
                Default is ["conv2d", "depthwise_conv2d", "mul"]. The quantizable_op_type in
                QuantizationFreezePass and ConvertToInt8Pass must be the same as this.
            weight_quantize_func(function): Function that defines how to quantize weight.
                Using this can quickly test if user's quantization method works or not.
                In this function, user should both define quantization function and
                dequantization function, that is, the function's input is non-quantized
                weight and function returns dequantized weight. If None, will use
                quantization op defined by 'weight_quantize_type'. Default is None.
            act_quantize_func(function): Function that defines how to quantize activation.
                Using this can quickly test if user's quantization method works or not.
                In this function, user should both define quantization and dequantization
                process, that is, the function's input is non-quantized activation and
                function returns dequantized activation. If None, will use quantization
                op defined by 'activation_quantize_type'. Default is None.
            weight_preprocess_func(function): Function that defines how to preprocess
                weight before quantization. Using this can quickly test if user's preprocess
                method works or not. The function's input is non-quantized weight and
                function returns processed weight to be quantized. If None, the weight will
                be quantized directly. Default is None.
            act_preprocess_func(function): Function that defines how to preprocess
                activation before quantization. Using this can quickly test if user's
                preprocess method works or not. The function's input is non-quantized
                activation and function returns processed activation to be quantized.
                If None, the activation will be quantized directly. Default is None.
            optimizer_func(function): Fuction return a optimizer. When 'is_test' is
                False and user want to use self-defined quantization function and
                preprocess function, this function must be set. Default is None.
            executor(Fluid.Executor): If user want to use self-defined quantization
                function and preprocess function, executor must be set for initialization.
                Default is None.


        Examples:
        .. code-block:: python
            # The original graph will be rewrite.
            import paddle.fluid as fluid
            from paddle.fluid.contrib.slim.quantization \
                import QuantizationTransformPass
            from paddle.fluid.contrib.slim.graph import IrGraph
            from paddle.fluid import core

            graph = IrGraph(core.Graph(program.desc), for_test=False)
            place = fluid.CPUPlace()
            transform_pass = QuantizationTransformPass(fluid.global_scope(),
            place)
            transform_pass.apply(graph)
        """
        self._scope = scope
        self._place = _get_paddle_place(place)
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._skip_pattern = skip_pattern
        self._weight_quantize_func = weight_quantize_func
        self._act_quantize_func = act_quantize_func
        self._weight_preprocess_func = weight_preprocess_func
        self._act_preprocess_func = act_preprocess_func
        self._optimizer = optimizer_func
        self._exe = executor
        quant_type = [
            'abs_max', 'channel_wise_abs_max', 'range_abs_max',
            'moving_average_abs_max'
        ]
        assert activation_quantize_type != 'channel_wise_abs_max', \
            "The activation quantization type does not support 'channel_wise_abs_max'."
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be "
                "'abs_max' or 'range_abs_max' or 'moving_average_abs_max'." %
                (str(activation_quantize_type)))
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be "
                "'abs_max' or 'channel_wise_abs_max' or 'range_abs_max' "
                "or 'moving_average_abs_max'." % (str(weight_quantize_type)))

        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type
        self._window_size = window_size
        self._moving_rate = moving_rate

        self._quantizable_ops = quantizable_op_type
        for op in self._quantizable_ops:
            assert op in QuantizationTransformPass._supported_quantizable_op_type, \
                op + " is not supported for quantization."
        self._quantizable_grad_ops = [
            '%s_grad' % (op) for op in self._quantizable_ops
        ]
        self._is_test = None
        self._global_step = None

        self.create_var_map = {}
        self.create_op_map = {}

    def apply(self, graph):
        """
        Quantize the graph for training process. According to weight and
        activation quantization type, the graph will be added some fake
        quantize operators and fake dequantize operators.

        Args:
            graph(IrGraph): the applied graph.
        Returns:
            None
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        self._is_test = graph.is_test()
        # marked the variable which has been dequantized.
        dequantized_vars = collections.OrderedDict()
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        processed_vars = []

        def _quant_preprocess(op_node):
            user_skipped = False
            if isinstance(self._skip_pattern, list):
                user_skipped = op_node.op().has_attr("op_namescope") and \
                               any(pattern in op_node.op().attr("op_namescope") \
                                   for pattern in self._skip_pattern)
            elif isinstance(self._skip_pattern, str):
                user_skipped = op_node.op().has_attr("op_namescope") and \
                               op_node.op().attr("op_namescope").find(
                                   self._skip_pattern) != -1

            if user_skipped:
                op_node.op()._set_attr("skip_quant", True)
                op_node.op()._set_attr("with_quant_attr", True)

        def _transform_forward(graph, op):
            op.op()._set_attr("quantization_type", "qat_with_weight")
            op.op()._set_attr("with_quant_attr", True)
            inputs = op.inputs
            for var_node in inputs:
                if var_node.name() not in op.input_arg_names():
                    continue
                if var_node.name() in dequantized_vars:
                    dequant_var_node = dequantized_vars[var_node.name()]
                else:
                    name = var_node.name()
                    if name in processed_vars:
                        continue
                    is_weight = True if var_node.name() in persistable_vars \
                        else False

                    # if var node is weight and weight_preprocess_func is not None,
                    # will insert weight preprocess func 
                    # to preorocess weight before quantization
                    # if var node is activation and act_preprocess_func is not None, 
                    # will insert activation preprocess func 
                    # to preorocess activation before quantization
                    if is_weight and self._weight_preprocess_func is not None:
                        var_node = self._insert_func(
                            graph, self._weight_preprocess_func, var_node, op)
                    elif not is_weight and self._act_preprocess_func is not None:
                        var_node = self._insert_func(
                            graph, self._act_preprocess_func, var_node, op)

                    # if var node is weight and weight_quantize_func is not None,
                    # will insert weight quantize func to quantize and dequantize weight
                    # if var node is activation and act_quantize_func is not None,
                    # will insert act quantize func to quantize and dequantize activation
                    if is_weight and self._weight_quantize_func is not None:
                        target_out_node = self._insert_func(
                            graph, self._weight_quantize_func, var_node, op)
                        processed_vars.append(name)
                        continue
                    elif not is_weight and self._act_quantize_func is not None:
                        target_out_node = self._insert_func(
                            graph, self._act_quantize_func, var_node, op)
                        processed_vars.append(name)
                        continue

                    quant_bits = self._weight_bits if var_node.name() in persistable_vars \
                        else self._activation_bits
                    quant_type = self._weight_quantize_type if is_weight \
                        else self._activation_quantize_type
                    if quant_type == 'channel_wise_abs_max':  # Weight quantization
                        quant_axis = 1 if op.name() in \
                            _channelwise_quant_axis1_ops else 0
                        quant_var_node, scale_var_node = self._insert_channel_quant_op(
                            graph, var_node, name, quant_bits, quant_axis)
                        dequant_var_node = self._insert_channel_dequant_op(
                            graph, quant_var_node, [scale_var_node],
                            [quant_bits], quant_axis)
                    else:
                        quant_var_node, scale_var_node = self._insert_quant_op(
                            graph, var_node, name, quant_bits, quant_type)
                        dequant_var_node = self._insert_dequant_op(
                            graph, quant_var_node, scale_var_node, quant_bits)
                    dequantized_vars[name] = dequant_var_node
                graph.update_input_link(var_node, dequant_var_node, op)

        def _transform_backward(graph, op):
            for var_node in op.inputs:
                if var_node.name() not in op.input_arg_names():
                    continue
                if var_node.name() in dequantized_vars:
                    dequant_var_node = dequantized_vars[var_node.name()]
                    graph.update_input_link(var_node, dequant_var_node, op)

        def _has_weight(op):
            has_weight = False
            for var_node in op.inputs:
                if var_node.name() not in op.input_arg_names():
                    continue
                name = var_node.name()
                if var_node.name() in persistable_vars:
                    has_weight = True
            return has_weight

        if not self._is_test:
            self._create_global_step(graph)
        ops = graph.all_op_nodes()
        # Do the preproccess of quantization, such as skipping some ops
        # for not being quantized.
        for op in ops:
            if op.name() in self._quantizable_ops or \
                    op.name() in self._quantizable_grad_ops:
                _quant_preprocess(op)
        # Insert mapping table to solve the problem in saving inference model.
        graph.out_node_mapping_table = dict()
        # The process of _transform_forward and _transform_backward is needed in two for loops.
        # The loop for transforming the forward graph:
        for op in ops:
            if op.name() in self._quantizable_ops:
                if not self._is_skip_quant(graph, op) and _has_weight(op):
                    _transform_forward(graph, op)
        # The loop for renaming the inputs of backward op.
        for op in ops:
            if op.name() in self._quantizable_grad_ops and _has_weight(op):
                _transform_backward(graph, op)
        graph.resolve_hazard()
        return graph

    def _create_global_step(self, graph):
        if self._weight_quantize_type == 'range_abs_max' or \
                self._activation_quantize_type == 'range_abs_max':
            counter_name = cpt.to_text('@STEP_COUNTER@')
            for node in graph.all_var_nodes():
                if node.name() == counter_name:
                    self._global_step = node
            if self._global_step is None:
                global_step_in = graph.create_persistable_node(
                    name=counter_name,
                    var_type=core.VarDesc.VarType.LOD_TENSOR,
                    shape=[1],
                    var_dtype=core.VarDesc.VarType.INT64)
                _init_var_node(
                    global_step_in,
                    np.zeros(
                        [1], dtype='int64'),
                    self._scope,
                    self._place)
                global_step_out = graph.create_var_node_from_desc(
                    global_step_in.var())
                # The attribute of `op_role` is needed by ParallelExecutor.
                increment_op = graph.create_op_node(
                    op_type='increment',
                    attrs={
                        'step': 1.0,
                        'op_role':
                        core.op_proto_and_checker_maker.OpRole.Forward
                    },
                    inputs={'X': global_step_in},
                    outputs={'Out': global_step_out})
                graph.link_to(global_step_in, increment_op)
                graph.link_to(increment_op, global_step_out)
                self._global_step = global_step_out

    def _insert_quant_op(self, graph, var_node, name, quant_bits, quant_type):
        """
        Insert fake_quantize_op in the graph.
        """
        if quant_type == 'abs_max':
            return self._insert_quant_abs_max_op(graph, var_node, name,
                                                 quant_bits)
        elif quant_type == 'range_abs_max':
            return self._insert_quant_range_abs_max_op(graph, var_node, name,
                                                       quant_bits)
        elif quant_type == 'moving_average_abs_max':
            return self._insert_quant_moving_average_abs_max_op(
                graph, var_node, name, quant_bits)

    def _insert_quant_abs_max_op(self, graph, var_node, name, quant_bits):
        """
        Insert fake_quantize_abs_max op in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(name),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        scale_var_node = graph.create_persistable_node(
            name=self._quantized_scale_name(name),
            var_type=var_node.type(),
            shape=[1],
            var_dtype=var_node.dtype())
        data_type = 'float64' if var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(
            scale_var_node,
            np.zeros(
                scale_var_node.shape(), dtype=data_type),
            self._scope,
            self._place)
        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_abs_max',
            attrs={
                'bit_length': quant_bits,
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': var_node},
            outputs={'Out': quant_var_node,
                     'OutScale': scale_var_node})
        graph.link_to(var_node, quant_op_node)
        graph.link_to(quant_op_node, quant_var_node)
        graph.link_to(quant_op_node, scale_var_node)
        return quant_var_node, scale_var_node

    def _insert_quant_range_abs_max_op(self, graph, var_node, name, quant_bits):
        """
        Insert fake_quantize_range_abs_max on the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(name),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())

        scale_in_node = graph.create_persistable_node(
            name=self._quantized_scale_name(name),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[1],
            var_dtype=var_node.dtype())
        data_type = 'float64' if var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(
            scale_in_node,
            np.array(
                [0.001], dtype=data_type),
            self._scope,
            self._place)

        scale_out_node = graph.create_var_node_from_desc(scale_in_node.var())
        inputs = {'X': var_node, 'InScale': scale_in_node}
        outputs = {'Out': quant_var_node, 'OutScale': scale_out_node}

        if not self._is_test:
            # The name of scales_var_node maybe 'scales_0', 'scales_1', etc.
            scales_node = graph.create_persistable_node(
                name=unique_name.generate('scales'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                shape=[self._window_size],
                var_dtype=var_node.dtype())
            data_type = 'float64' if var_node.dtype(
            ) == core.VarDesc.VarType.FP64 else 'float32'
            _init_var_node(
                scales_node,
                np.zeros(
                    [self._window_size], dtype=data_type),
                self._scope,
                self._place)

            inputs['Iter'] = self._global_step
            outputs['OutScales'] = scales_node
        attrs = {
            'window_size': self._window_size,
            'bit_length': quant_bits,
            'is_test': self._is_test,
            'op_role': core.op_proto_and_checker_maker.OpRole.Forward
        }
        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_range_abs_max',
            attrs=attrs,
            inputs=inputs,
            outputs=outputs)

        graph.link_to(var_node, quant_op_node)
        graph.link_to(scale_in_node, quant_op_node)
        graph.link_to(quant_op_node, quant_var_node)
        graph.link_to(quant_op_node, scale_out_node)

        if not self._is_test:
            graph.link_to(self._global_step, quant_op_node)
            graph.link_to(quant_op_node, scales_node)

        return quant_var_node, scale_out_node

    def _insert_quant_moving_average_abs_max_op(self, graph, var_node, name,
                                                quant_bits):
        """Insert fake_quantize_moving_average_abs_max
        """
        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(name),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        scale_in_node = graph.create_persistable_node(
            name=self._quantized_scale_name(name),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[1],
            var_dtype=var_node.dtype())
        data_type = 'float64' if var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(
            scale_in_node,
            np.array(
                [0.001], dtype=data_type),
            self._scope,
            self._place)

        scale_out_node = graph.create_var_node_from_desc(scale_in_node.var())
        ins = {'X': var_node, 'InScale': scale_in_node}
        outs = {'Out': quant_var_node, 'OutScale': scale_out_node}
        if not self._is_test:
            state_in_node = graph.create_persistable_node(
                name=unique_name.generate('state'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            data_type = 'float64' if var_node.dtype(
            ) == core.VarDesc.VarType.FP64 else 'float32'
            _init_var_node(
                state_in_node,
                np.ones(
                    [1], dtype=data_type),
                self._scope,
                self._place)
            accum_in_node = graph.create_persistable_node(
                name=unique_name.generate('accum'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            _init_var_node(
                accum_in_node,
                np.ones(
                    [1], dtype=data_type),
                self._scope,
                self._place)
            state_out_node = graph.create_var_node_from_desc(state_in_node.var(
            ))
            accum_out_node = graph.create_var_node_from_desc(accum_in_node.var(
            ))

            ins['InState'] = state_in_node
            ins['InAccum'] = accum_in_node
            outs['OutState'] = state_out_node
            outs['OutAccum'] = accum_out_node

        attrs = {
            'bit_length': quant_bits,
            'moving_rate': self._moving_rate,
            'is_test': self._is_test,
            'op_role': core.op_proto_and_checker_maker.OpRole.Forward
        }

        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_moving_average_abs_max',
            attrs=attrs,
            inputs=ins,
            outputs=outs)

        graph.link_to(var_node, quant_op_node)
        graph.link_to(scale_in_node, quant_op_node)
        graph.link_to(quant_op_node, quant_var_node)
        graph.link_to(quant_op_node, scale_out_node)

        if not self._is_test:
            graph.link_to(state_in_node, quant_op_node)
            graph.link_to(accum_in_node, quant_op_node)
            graph.link_to(quant_op_node, state_out_node)
            graph.link_to(quant_op_node, accum_out_node)

        return quant_var_node, scale_out_node

    def _insert_channel_quant_op(self, graph, var_node, name, quant_bits,
                                 quant_axis):
        """
        Insert fake_channel_wise_quantize_abs_max op in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        quant_var_node = graph.create_var_node(
            name=self._quantized_var_name(name),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        scale_var_node = graph.create_persistable_node(
            name=self._quantized_scale_name(name),
            var_type=var_node.type(),
            shape=[var_node.shape()[quant_axis]],
            var_dtype=var_node.dtype())
        data_type = 'float64' if var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(
            scale_var_node,
            np.zeros(
                scale_var_node.shape(), dtype=data_type),
            self._scope,
            self._place)
        quant_op_node = graph.create_op_node(
            op_type='fake_channel_wise_quantize_abs_max',
            attrs={
                'bit_length': quant_bits,
                'quant_axis': quant_axis,
                'is_test': self._is_test,
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': var_node},
            outputs={'Out': quant_var_node,
                     'OutScale': scale_var_node})
        graph.link_to(var_node, quant_op_node)
        graph.link_to(quant_op_node, quant_var_node)
        graph.link_to(quant_op_node, scale_var_node)
        return quant_var_node, scale_var_node

    def _insert_dequant_op(self, graph, var_node, scale_var_node, quant_bits):
        """
        Insert fake_dequantize_op in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        max_range = (1 << (quant_bits - 1)) - 1
        dequant_op_node = graph.create_op_node(
            op_type='fake_dequantize_max_abs',
            attrs={
                'max_range': float(max_range),
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': var_node,
                    'Scale': scale_var_node},
            outputs={'Out': dequant_var_node})
        graph.link_to(var_node, dequant_op_node)
        graph.link_to(scale_var_node, dequant_op_node)
        graph.link_to(dequant_op_node, dequant_var_node)
        return dequant_var_node

    def _insert_channel_dequant_op(self, graph, var_node, scale_var_nodes,
                                   quant_bits, quant_axis):
        """
        Insert fake_channel_wise_dequantize_max_abs in the graph.
        """
        assert var_node.is_var(), '{} is not a var'.format(var_node.name())

        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        dequant_op_node = graph.create_op_node(
            op_type='fake_channel_wise_dequantize_max_abs',
            attrs={
                'quant_bits': quant_bits,
                'quant_axis': quant_axis,
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': var_node,
                    'Scales': scale_var_nodes},
            outputs={'Out': dequant_var_node})
        graph.link_to(var_node, dequant_op_node)
        for scale_n in scale_var_nodes:
            graph.link_to(scale_n, dequant_op_node)
        graph.link_to(dequant_op_node, dequant_var_node)
        return dequant_var_node

    def _create_new_node(self, graph, in_node):
        """
        create a node that same with in_node in graph
        Args:
            graph(IrGraph): create node in graph.
            in_node(IrVarNode): create node that same with in_node.
        Returns:
            created new node
        """
        key = ''
        for inp in in_node.inputs:
            key = key + inp.name()
        key = key + in_node.name()
        for inp in in_node.outputs:
            key = key + inp.name()

        if key in self.create_var_map.keys():
            new_node = self.create_var_map[key]
        elif in_node.is_ctrl_var():
            new_node = graph.create_control_dep_var()
            self.create_var_map[key] = new_node
        else:
            new_node = graph.create_var_node_from_desc(in_node.node.var())
            self.create_var_map[key] = new_node
        return new_node

    def _copy_graph(self, graph, source_graph, op_node):
        """
        copy op_node in source_graph to graph. And will run recursively 
        for next ops that link to op_node's outputs.
        Args:
            graph(IrGraph): target graph to copy.
            source_graph(IrGraph): source graph to copy.
            op_node(IrOpNode): op node in source_graph.
        Returns:
            None

        """
        key = ''
        for inp in op_node.inputs:
            key = key + inp.name()
        key = key + op_node.name()
        for inp in op_node.outputs:
            key = key + inp.name()
        has_created = False
        if key in self.create_op_map.keys():
            new_op_node = self.create_op_map[key]
            has_created = True
        else:
            new_op_node = graph.create_op_node_from_desc(op_node.node.op())
            self.create_op_map[key] = new_op_node
        if has_created:
            return
        for in_node in op_node.inputs:
            new_node = self._create_new_node(graph, in_node)
            graph.link_to(new_node, new_op_node)
        for in_node in op_node.outputs:
            new_node = self._create_new_node(graph, in_node)
            graph.link_to(new_op_node, new_node)
        for var_node in op_node.outputs:
            for next_op_node in var_node.outputs:
                self._copy_graph(graph, source_graph, next_op_node)
        return

    def _insert_func(self, graph, func, var_node, op):
        """
        Insert a tmp program that returned by func between var_node and op.

        Args:
            graph(IrGraph): target graph to insert tmp program.
            func(Function): function to define a tmp program
            var_node(IrVarNode): node in target graph.
            op(IrOpNode): op in target graph.
        Returns:
            op's new input that replaces var_node
        """
        tmp_program = Program()
        startup_program = Program()
        with program_guard(tmp_program, startup_program):
            with unique_name.guard(var_node.name() + "_"):
                in_node = data(
                    var_node.name() + '_tmp_input',
                    shape=var_node.shape(),
                    dtype='float32')
                out_node = func(in_node)
                graph.out_node_mapping_table[out_node.name] = var_node.name()
                # loss shape must be 1 when minimize
                loss = mean(out_node)
                if not graph._for_test:
                    assert self._optimizer, "optimizer_func must be set when graph is test graph"
                    in_node.stop_gradient = False
                    optimizer = self._optimizer()
                    optimizer.minimize(loss)
        with scope_guard(self._scope):
            self._exe.run(startup_program)

        tmp_graph = IrGraph(
            core.Graph(tmp_program.desc), for_test=graph._for_test)
        in_node = tmp_graph._find_node_by_name(tmp_graph.all_var_nodes(),
                                               in_node.name)
        out_node = tmp_graph._find_node_by_name(tmp_graph.all_var_nodes(),
                                                out_node.name)

        in_node_params = []
        in_op_node = []
        # copy tmp graph to graph, after that, we can insert tmp graph's copy to graph.
        for node in tmp_graph.all_var_nodes():
            if node.inputs == [] and node.persistable():
                in_node_params.append(node)
        for node in tmp_graph.all_op_nodes():
            if node.inputs == []:
                in_op_node.append(node)
        for node in in_node.outputs:
            self._copy_graph(graph, tmp_graph, node)
        for node in in_node_params:
            for op_node in node.outputs:
                self._copy_graph(graph, tmp_graph, op_node)
        for node in in_op_node:
            self._copy_graph(graph, tmp_graph, node)

        target_in_node = graph._find_node_by_name(graph.all_var_nodes(),
                                                  in_node.name())
        target_out_node = graph._find_node_by_name(graph.all_var_nodes(),
                                                   out_node.name())
        loss_node = graph._find_node_by_name(graph.all_var_nodes(), loss.name)
        outputs = target_in_node.outputs
        for node in outputs:
            graph.update_input_link(target_in_node, var_node, node)
        graph.update_input_link(var_node, target_out_node, op)

        # update grad
        if not graph._for_test:
            op_out = op.outputs[0]
            op_out_grad = graph._find_node_by_name(graph.all_var_nodes(),
                                                   op_out.name() + "@GRAD")
            # find op's gradient op, such as conv2d_grad
            op_grad = op_out_grad.outputs[0]
            target_out_grad_node = graph._find_node_by_name(
                graph.all_var_nodes(), target_out_node.name() + "@GRAD")
            in_node_grad = graph._find_node_by_name(
                graph.all_var_nodes(), target_in_node.name() + "@GRAD")
            in_node_grad_op = in_node_grad.inputs
            # update op_grad's input
            graph.update_input_link(var_node, target_out_node, op_grad)

            op_grad_out = None
            # find var_node's corresponding grad node
            for node in op_grad.outputs:
                if var_node.name() + "@GRAD" in node.name():
                    op_grad_out = node
            # update op_grad's output
            if op_grad_out is not None:
                graph.update_output_link(op_grad_out, target_out_grad_node,
                                         op_grad)
            else:
                graph.link_to(op_grad, target_out_grad_node)

            for node in in_node_grad_op:
                graph.update_input_link(target_in_node, var_node, node)
                if op_grad_out:
                    graph.update_output_link(in_node_grad, op_grad_out, node)
            # remove useless nodes
            mean_grad = target_out_grad_node.inputs[0]
            mean_out_grad = mean_grad.inputs[0]
            fill_constant_node = mean_out_grad.inputs[0]
            graph.safe_remove_nodes(mean_grad)
            graph.safe_remove_nodes(mean_out_grad)
            graph.safe_remove_nodes(fill_constant_node)
            graph.safe_remove_nodes(in_node_grad)

        graph.safe_remove_nodes(loss_node.inputs[0])
        graph.safe_remove_nodes(loss_node)
        graph.safe_remove_nodes(target_in_node)
        return target_out_node

    def _quantized_var_name(self, var_name):
        """
        Return quantized variable name for the input `var_name`.
        """
        return "%s.quantized" % (var_name)

    def _dequantized_var_name(self, var_name):
        """
        Return dequantized variable name for the input `var_name`.
        """
        return "%s.dequantized" % (var_name)

    def _quantized_scale_name(self, var_name):
        """
        Return the scale name of quantized variable for the input `var_name`.
        """
        return "%s.scale" % (var_name)

    def _is_skip_quant(self, graph, op_node):
        """
        Analyse whether the op node skips quantization.
        """
        is_skip = False
        if op_node.op().has_attr("skip_quant") and \
            op_node.op().attr("skip_quant"):
            is_skip = True
        # if the inputs of mul and matmul are not all persistable, use
        # AddQuantDequantPass to quantize them.
        if op_node.name() in ["mul", "matmul"] and \
            _is_input_all_not_persistable(graph, op_node):
            is_skip = True
        if op_node.op().has_attr("quantization_type") and \
            op_node.op().attr("quantization_type") == "qat_without_weight":
            is_skip = True
        return is_skip


class QuantizationFreezePass(object):
    def __init__(self,
                 scope,
                 place,
                 bias_correction=False,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max',
                 quantizable_op_type=None):
        """
        The freeze pass is used to adjust the quantize operator order, for example:
            1) `activation -> quant -> dequant -> conv2d` will be frozen into
            `activation -> quant -> conv2d -> dequant`
            2) `weight -> quant -> dequant -> conv2d` will be frozen into `weight -> conv2d`,
            and weight will be scaled offline.

        Args:
            scope(fluid.Scope): scope is used to get the weight tensor values.
            place(fluid.CPUPlace|fluid.CUDAPlace|str): place is used to restore the weight tensors.
                If it's string, It can be ``cpu``, and ``gpu:x``, where ``x`` is the index of the GPUs.
            bias_correction(bool): whether use bias correction for post-training quantization.
                 https://arxiv.org/abs/1810.05723.
            weight_bits(int): quantization bit number for weights.
            activation_bits(int): quantization bit number for activation.
            weight_quantize_type(str): quantization type for weights, support 'abs_max' and 
                'channel_wise_abs_max'. The 'range_abs_max' usually is not used for weight, 
                since weights are fixed once the model is well trained.
            quantizable_op_type(list[str]): This input param will be removed latter. The pass
                will process all quantized op, so it is not necessary to set the input param.
        """
        assert scope is not None, \
            'The scope cannot be set None.'
        assert place is not None, \
            'The place cannot be set None.'
        self._scope = scope
        self._bias_correction = bias_correction
        self._place = _get_paddle_place(place)
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._weight_quantize_type = weight_quantize_type
        self._fake_quant_op_names = _fake_quant_op_list
        self._fake_dequant_op_names = _fake_dequant_op_list
        self._op_input_rename_map = collections.OrderedDict()
        self._op_output_rename_map = collections.OrderedDict()
        self._quant_var_scale_map = collections.OrderedDict()

    def apply(self, graph):
        """
        Adjust quantize/dequantize operators order for the inference process.

        Args:
            graph(IrGraph): the applied graph.
        Returns:
            None
        """
        # Get input scales in fake quant op and process weights
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        ops = graph.all_op_nodes()
        for op_node in ops:
            op_name = op_node.name()
            if op_name in self._fake_quant_op_names:
                input_arg_name = op_node.input('X')[0]
                if hasattr(graph, 'out_node_mapping_table'):
                    if input_arg_name in graph.out_node_mapping_table.keys():
                        input_arg_name = graph.out_node_mapping_table[
                            input_arg_name]
                if input_arg_name not in persistable_vars:
                    scale_v = graph._find_node_by_name(
                        op_node.outputs, op_node.output('OutScale')[0])
                    self._quant_var_scale_map[input_arg_name] = scale_v
                else:
                    # Obtain scale from OutScale var node
                    scale_v = self._load_var(op_node.output('OutScale')[0])
                    assert scale_v.ndim in [
                        1, 2
                    ], "the dim of scale_v should be 1 or 2"
                    if scale_v.ndim == 2:
                        scale_v = scale_v[0]
                    if scale_v.size == 1 and self._weight_quantize_type == 'abs_max':
                        scale_v = scale_v[0]
                    else:
                        scale_v = scale_v.tolist()
                    self._quant_var_scale_map[input_arg_name] = scale_v
                    # Quantize weight and restore
                    param_v = self._load_var(input_arg_name)
                    if isinstance(scale_v, list) and \
                        any(_check_grandchild_op_node(op_node, op)
                        for op in _channelwise_quant_axis1_ops):
                        quant_axis = 1
                    else:
                        quant_axis = 0
                    quantized_param_v = self._quant(
                        param_v.copy(), scale_v, self._weight_bits, quant_axis)
                    if self._bias_correction == True:
                        quantized_param_v = self._bias_correction_w(
                            param_v, quantized_param_v, scale_v, quant_axis)
                    self._restore_var(input_arg_name, quantized_param_v)
                    self._remove_fake_quant_and_dequant_op(graph, op_node)

        # Remove all fake dequant op
        ops = graph.all_op_nodes()
        for op_node in ops:
            op_name = op_node.name()
            if op_name in self._fake_dequant_op_names:
                self._remove_fake_quant_and_dequant_op(graph, op_node)

        # Insert post dequant op
        ops = graph.all_op_nodes()
        for op_node in ops:
            op_node_desc = op_node.op()
            if op_node_desc.has_attr("quantization_type") and \
                op_node_desc.attr("quantization_type") == "qat_with_weight":
                if self._weight_quantize_type == 'channel_wise_abs_max':
                    self._insert_post_channel_dequant_op(graph, op_node,
                                                         quant_axis)
                else:
                    self._insert_post_dequant_op(graph, op_node)

        # Rename inputs of the followed ops after inserting dequant_op after fc/conv
        for op_node in ops:
            for var_node in op_node.inputs:
                if var_node.node in self._op_output_rename_map:
                    old_in = var_node
                    new_in = self._op_output_rename_map[var_node.node]
                    graph.update_input_link(old_in, new_in, op_node)

        # remove the unused var node in the graph
        self._remove_unused_var_nodes(graph)
        graph.resolve_hazard()
        return graph

    def _remove_fake_quant_and_dequant_op(self, graph, op_node):
        k = graph._find_node_by_name(op_node.outputs, op_node.output('Out')[0])
        v = graph._find_node_by_name(op_node.inputs, op_node.input('X')[0])
        if v.node not in self._op_input_rename_map:
            self._op_input_rename_map[k.node] = v
        else:
            self._op_input_rename_map[k.node] = self._op_input_rename_map[
                v.node]
        graph.safe_remove_nodes(op_node)

    def _insert_post_channel_dequant_op(self, graph, op_node, quant_axis):
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        for var_node in op_node.inputs:
            name = var_node.name()
            if name not in op_node.input_arg_names():
                continue
            if var_node.node in self._op_input_rename_map:
                old_in = var_node
                new_in = self._op_input_rename_map[var_node.node]
                new_in.clear_outputs()
                graph.update_input_link(old_in, new_in, op_node)
            original_var_name = self._original_var_name(name)
            scale_v = self._quant_var_scale_map[original_var_name]
            if original_var_name in persistable_vars:
                assert isinstance(
                    scale_v,
                    list), 'The scale of parameter %s is not a list.' % (
                        original_var_name)
                channel_scale = np.array(scale_v)
            else:
                assert isinstance(scale_v, IrNode)
                scale_var_node = self._quant_var_scale_map[original_var_name]

        if len(op_node.output_arg_names()) != 1:
            raise ValueError("Only support one output, but op %s has"
                             " more than one output." % (op_node.name()))

        output_var_node = graph._find_node_by_name(
            op_node.outputs, op_node.output_arg_names()[0])
        weight_scale_node = graph.create_persistable_node(
            name=unique_name.generate('channel_scale'),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[channel_scale.shape[0]],
            var_dtype=output_var_node.dtype())
        data_type = 'float64' if output_var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(weight_scale_node,
                       channel_scale.astype(data_type), self._scope,
                       self._place)
        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(output_var_node.name()),
            var_type=output_var_node.type(),
            shape=output_var_node.shape(),
            var_dtype=output_var_node.dtype())
        x_num_col_dims = 1
        if op_node.name() in ['matmul', 'matmul_v2', 'mul']:
            x_num_col_dims = len(op_node.outputs[0].shape()) - 1
        if op_node.op().has_attr("x_num_col_dims"):
            x_num_col_dims = op_node.op().attr("x_num_col_dims")
        dequant_op_node = graph.create_op_node(
            op_type='fake_channel_wise_dequantize_max_abs',
            attrs={
                'quant_bits': [self._weight_bits, self._activation_bits],
                'quant_axis': quant_axis,
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward,
                'x_num_col_dims': x_num_col_dims
            },
            inputs={
                'X': output_var_node,
                'Scales': [weight_scale_node, scale_var_node]
            },
            outputs={'Out': dequant_var_node})
        graph.link_to(output_var_node, dequant_op_node)
        graph.link_to(scale_var_node, dequant_op_node)
        graph.link_to(weight_scale_node, dequant_op_node)
        graph.link_to(dequant_op_node, dequant_var_node)
        self._op_output_rename_map[output_var_node.node] = dequant_var_node
        return dequant_var_node

    def _insert_post_dequant_op(self, graph, op_node):
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        max_range = 1
        param_range = (1 << (self._weight_bits - 1)) - 1
        act_range = (1 << (self._activation_bits - 1)) - 1
        for var_node in op_node.inputs:
            name = var_node.name()
            if name not in op_node.input_arg_names():
                continue
            if var_node.node in self._op_input_rename_map:
                old_in = var_node
                new_in = self._op_input_rename_map[var_node.node]
                new_in.clear_outputs()
                graph.update_input_link(old_in, new_in, op_node)
            original_var_name = self._original_var_name(name)
            scale_v = self._quant_var_scale_map[original_var_name]
            if original_var_name in persistable_vars:
                assert self._is_float(
                    scale_v), 'The scale of parameter %s is not a float.' % (
                        original_var_name)
                scale_v = 1e-8 if scale_v == 0.0 else scale_v
                max_range *= param_range / scale_v
            else:
                max_range *= act_range
                assert isinstance(scale_v, IrNode)
                scale_var_node = self._quant_var_scale_map[original_var_name]

        if len(op_node.output_arg_names()) != 1:
            raise ValueError("Only support one output, but op %s has"
                             " more than one output." % (op_node.name()))

        output_var_node = graph._find_node_by_name(
            op_node.outputs, op_node.output_arg_names()[0])
        dequant_var_node = graph.create_var_node(
            name=self._dequantized_var_name(output_var_node.name()),
            var_type=output_var_node.type(),
            shape=output_var_node.shape(),
            var_dtype=output_var_node.dtype())
        dequant_op_node = graph.create_op_node(
            op_type='fake_dequantize_max_abs',
            attrs={
                'max_range': float(max_range),
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': output_var_node,
                    'Scale': scale_var_node},
            outputs={'Out': dequant_var_node})
        graph.link_to(output_var_node, dequant_op_node)
        graph.link_to(scale_var_node, dequant_op_node)
        graph.link_to(dequant_op_node, dequant_var_node)
        self._op_output_rename_map[output_var_node.node] = dequant_var_node
        return dequant_var_node

    def _load_var(self, name):
        return np.array(self._scope.find_var(name).get_tensor())

    def _restore_var(self, name, array):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array, self._place)

    def _remove_unused_var_nodes(self, graph):
        all_used_vars = set()
        ops = graph.all_op_nodes()
        for op_node in ops:
            for input_node in op_node.inputs:
                all_used_vars.add(input_node)
            for output_node in op_node.outputs:
                all_used_vars.add(output_node)

        all_used_vars = {n.node for n in all_used_vars}
        all_unused_vars = {
            n
            for n in filter(lambda node: node.node not in all_used_vars,
                            graph.all_var_nodes())
        }
        graph.safe_remove_nodes(all_unused_vars)

    def _original_var_name(self, var_name):
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

    def _dequantized_var_name(self, var_name):
        """
        Return dequantized variable name for the input `var_name`.
        """
        return "%s.dequantized" % (var_name)

    def _is_float(self, v):
        return isinstance(v, float) or isinstance(v, np.float32) \
            or isinstance(v, np.float64)

    def _quant(self, x, scale, num_bits, quant_axis):
        assert quant_axis in [0, 1], 'quant_axis should be 0 or 1 for now.'
        bnt = (1 << (num_bits - 1)) - 1

        def _clip(x, scale):
            x[x > scale] = scale
            x[x < -scale] = -scale
            return x

        if isinstance(scale, list):
            for i, s in enumerate(scale):
                if s == 0.0:
                    s = 1e-8
                if quant_axis == 0:
                    x[i] = _clip(x[i], s)
                    x[i] = np.round(x[i] / s * bnt)
                else:
                    x[:, i] = _clip(x[:, i], s)
                    x[:, i] = np.round(x[:, i] / s * bnt)
        else:
            scale = 1e-8 if scale == 0.0 else scale
            x = _clip(x, scale)
            x = np.round(x / scale * bnt)
        return x

    def _bias_correction_w(self, x, x_quant, scale_v, quant_axis):
        '''
        Bias correction for weight
        '''
        eps = 1e-8
        bnt = (1 << (self._weight_bits - 1)) - 1
        x_dequant = x_quant.copy()
        if isinstance(scale_v, list):
            if quant_axis == 0:
                for i, s in enumerate(scale_v):
                    x_dequant[i] = x_dequant[i] * s / bnt
                quant_bias = x - x_dequant
                mean_bias = quant_bias.reshape(quant_bias.shape[0], -1).mean(-1)
                std_orig = x.reshape(x.shape[0], -1).std(-1)
                std_quant = x_dequant.reshape(x_dequant.shape[0], -1).std(-1)
                std_bias = std_orig / (std_quant + eps)
            else:
                for i, s in enumerate(scale_v):
                    x_dequant[:, i] = x_quant[:, i] * s / bnt
                quant_bias = x - x_dequant
                mean_bias = np.array([
                    quant_bias[:, i].mean() for i in range(quant_bias.shape[1])
                ])
                std_orig = np.array([x[:, i].std() for i in range(x.shape[1])])
                std_quant = np.array(
                    [x_dequant[:, i].std() for i in range(x_dequant.shape[1])])
                std_bias = std_orig / (std_quant + eps)
        else:
            x_dequant = x_quant * scale_v / bnt
            mean_bias = (x - x_dequant).mean()
            std_bias = x.std() / (x_dequant.std() + eps)
        if mean_bias.ndim == 1:
            std_bias = np.resize(std_bias, x.shape)
            mean_bias = np.resize(mean_bias, x.shape)

        x_dequant = (mean_bias + x_dequant) * std_bias
        quantized_param_v = self._quant(x_dequant, scale_v, self._weight_bits,
                                        quant_axis)
        return quantized_param_v


class ConvertToInt8Pass(object):
    def __init__(self, scope, place, quantizable_op_type=None):
        """
        Convert the weights into int8_t type.

        Args:
            scope(fluid.Scope): scope is used to get the weight tensor values.
            place(fluid.CPUPlace|fluid.CUDAPlace|str): place is used to restore the
                8bits weight tensors. If it's string, It can be ``cpu``, and ``gpu:x``,
                where ``x`` is the index of the GPUs.
            quantizable_op_type(list[str]): This input param will be removed latter. The pass
                will process all quantized op, so it is not necessary to set the input param.
        """
        assert scope is not None, \
            'The scope cannot be set None.'
        assert place is not None, \
            'The place cannot be set None.'
        self._scope = scope
        self._place = _get_paddle_place(place)

    def apply(self, graph):
        """
        Convert weights' type of the graph. After that, the data type of the
        graph weights is int8_t.

        Args:
            graph(IrGraph): the applied graph.
        Returns:
            None
        """
        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        ops = graph.all_op_nodes()
        input_map = {}
        for op_node in ops:
            if op_node.op().has_attr("quantization_type") and \
                op_node.op().attr("quantization_type") == "qat_with_weight":
                for var_node in op_node.inputs:
                    name = var_node.name()
                    if name in persistable_vars:
                        if name not in input_map:
                            int8_var_node = self._convert_to_int8(graph,
                                                                  var_node)
                            input_map[name] = int8_var_node
                        graph.update_input_link(var_node, input_map[name],
                                                op_node)

        # remove the unused var node in the graph
        self._remove_unused_var_nodes(graph)
        graph.resolve_hazard()
        return graph

    def _convert_to_int8(self, graph, var_node):
        int8_var_node_name = var_node.name() + ".int8"
        int8_var_node = graph.create_persistable_node(
            name=cpt.to_text(int8_var_node_name),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=core.VarDesc.VarType.INT8)
        array = self._load_var(var_node.name())
        self._scope.var(int8_var_node_name)
        self._store_var(int8_var_node_name, array, np.int8)
        return int8_var_node

    def _load_var(self, name):
        return np.array(self._scope.find_var(name).get_tensor())

    def _store_var(self, name, array, dtype):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array.astype(dtype), self._place)

    def _remove_unused_var_nodes(self, graph):
        all_used_vars = set()
        ops = graph.all_op_nodes()
        for op_node in ops:
            for input_node in op_node.inputs:
                all_used_vars.add(input_node)
            for output_node in op_node.outputs:
                all_used_vars.add(output_node)

        all_used_vars = {n.node for n in all_used_vars}
        all_unused_vars = {
            n
            for n in filter(lambda node: node.node not in all_used_vars,
                            graph.all_var_nodes())
        }
        graph.safe_remove_nodes(all_unused_vars)


class TransformForMobilePass(object):
    def __init__(self):
        """
        This pass is used to convert the frozen graph for paddle-mobile execution.
        """
        self._fake_quant_op_names = _fake_quant_op_list
        self._fake_dequant_op_names = _fake_dequant_op_list

    def apply(self, graph):
        """
        Because paddle-mobile use `quantize` an `dequantize` as the names of
        quantize operator and dequantize operator, the `apply` function just
        realize this logic.

        Args:
            graph(IrGraph): the graph will be transformed.
        Returns:
            None
        """
        ops = graph.all_op_nodes()
        for op_node in ops:
            name = op_node.name()
            if name in self._fake_quant_op_names:
                op_node.set_type('quantize')
                quant_node = graph.create_op_node_from_desc(op_node.op())
                for input_node in op_node.inputs:
                    graph.link_to(input_node, quant_node)
                for output_node in op_node.outputs:
                    graph.link_to(quant_node, output_node)
                graph.safe_remove_nodes(op_node)
            if name in self._fake_dequant_op_names:
                op_node.set_type('dequantize')
                dequant_node = graph.create_op_node_from_desc(op_node.op())
                for input_node in op_node.inputs:
                    graph.link_to(input_node, dequant_node)
                for output_node in op_node.outputs:
                    graph.link_to(dequant_node, output_node)
                graph.safe_remove_nodes(op_node)
        graph.resolve_hazard()
        return graph


class OutScaleForTrainingPass(object):
    def __init__(self, scope=None, place=None, moving_rate=0.9):
        """
        This pass is used for calculating output scales of some operators.
        These output scales may be used by tensorRT or some other inference engines.

        Args:
            scope(fluid.Scope): The scope is used to initialize these new parameters.
            place(fluid.CPUPlace|fluid.CUDAPlace|str): The place is used to initialize new parameters.
                If it's string, It can be ``cpu``, and ``gpu:x``, where ``x`` is the
                index of the GPUs.
            moving_rate(float): The decay coefficient of moving average. The default value is 0.9.
        """
        self._scope = scope
        self._place = _get_paddle_place(place)
        self._moving_rate = moving_rate
        self._is_test = None
        self._teller_set = _out_scale_op_list

    def apply(self, graph):
        """
        Insert the `moving_average_abs_max_scale` op in order to calculate output scales
        of operators in the teller_set.

        Args:
            graph(IrGraph): the target graph.
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        self._is_test = graph.is_test()
        target_ops = []
        for op in graph.all_op_nodes():
            if op.name() in self._teller_set:
                target_ops.append(op)
        for op in target_ops:
            for output_var_name in _get_op_output_var_names(op):
                in_node = graph._find_node_by_name(op.outputs, output_var_name)
                if in_node.dtype() not in \
                    [core.VarDesc.VarType.FP64, core.VarDesc.VarType.FP32]:
                    continue

                scale_node = graph.create_persistable_node(
                    name=self._scale_name(in_node.name()),
                    var_type=core.VarDesc.VarType.LOD_TENSOR,
                    shape=[1],
                    var_dtype=in_node.dtype())
                data_type = 'float64' if in_node.dtype() \
                    == core.VarDesc.VarType.FP64 else 'float32'
                _init_var_node(
                    scale_node,
                    np.ones(
                        [1], dtype=data_type),
                    self._scope,
                    self._place)
                ins = {'X': in_node}
                outs = {'OutScale': scale_node}
                if not self._is_test:
                    state_in_node = graph.create_persistable_node(
                        name=unique_name.generate('scale_state@'),
                        var_type=core.VarDesc.VarType.LOD_TENSOR,
                        var_dtype=in_node.dtype(),
                        shape=[1])
                    _init_var_node(
                        state_in_node,
                        np.ones(
                            [1], dtype=data_type),
                        self._scope,
                        self._place)
                    accum_in_node = graph.create_persistable_node(
                        name=unique_name.generate('scale_accum@'),
                        var_type=core.VarDesc.VarType.LOD_TENSOR,
                        var_dtype=in_node.dtype(),
                        shape=[1])
                    _init_var_node(
                        accum_in_node,
                        np.ones(
                            [1], dtype=data_type),
                        self._scope,
                        self._place)
                    state_out_node = graph.create_var_node_from_desc(
                        state_in_node.var())
                    accum_out_node = graph.create_var_node_from_desc(
                        accum_in_node.var())

                    ins['InState'] = state_in_node
                    ins['InAccum'] = accum_in_node
                    outs['OutState'] = state_out_node
                    outs['OutAccum'] = accum_out_node

                attrs = {
                    'moving_rate': self._moving_rate,
                    'is_test': self._is_test,
                    'op_role': core.op_proto_and_checker_maker.OpRole.Forward
                }
                scale_op_node = graph.create_op_node(
                    op_type='moving_average_abs_max_scale',
                    attrs=attrs,
                    inputs=ins,
                    outputs=outs)
                graph.link_to(in_node, scale_op_node)
                graph.link_to(scale_op_node, scale_node)
                if not self._is_test:
                    graph.link_to(state_in_node, scale_op_node)
                    graph.link_to(accum_in_node, scale_op_node)
                    graph.link_to(scale_op_node, state_out_node)
                    graph.link_to(scale_op_node, accum_out_node)
        graph.resolve_hazard()
        return graph

    def _scale_name(self, var_name):
        """
        Return the scale name for the var named `var_name`.
        """
        return "%s@scale" % (var_name)


class OutScaleForInferencePass(object):
    def __init__(self, scope=None):
        """
        This pass is used for setting output scales of some operators.
        These output scales may be used by tensorRT or some other inference engines.

        Args:
            scope(fluid.Scope): The scope is used to initialize these new parameters.
        """
        self._scope = scope
        self._teller_set = _out_scale_op_list

    def apply(self, graph):
        """
        Get output scales from the scope and set these scales in op_descs
        of operators in the teller_set.

        Args:
            graph(IrGraph): the target graph.
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        op_nodes = graph.all_op_nodes()
        for op_node in op_nodes:
            if op_node.name() in self._teller_set:
                var_names = _get_op_output_var_names(op_node)
                for var_name in var_names:
                    in_node = graph._find_node_by_name(op_node.outputs,
                                                       var_name)
                    if in_node.dtype() not in \
                        [core.VarDesc.VarType.FP64, core.VarDesc.VarType.FP32]:
                        continue

                    scale_name = self._scale_name(var_name)
                    scale_var = self._scope.find_var(scale_name)
                    assert scale_var is not None, \
                        "Can not find {} variable in the scope".format(scale_name)
                    scale_value = np.array(scale_var.get_tensor())[0]

                    # For compatibility, we save output threshold by two methods.
                    op_node.op()._set_attr("out_threshold", float(scale_value))

                    argname_index = _get_output_name_index(op_node, var_name)
                    assert argname_index is not None, \
                        var_name + " is not the output of the op"
                    op_node.op()._set_attr(argname_index[0] + str(argname_index[1]) \
                        + "_threshold", float(scale_value))
                    op_node.op()._set_attr("with_quant_attr", True)
        graph.resolve_hazard()
        return graph

    def _scale_name(self, var_name):
        """
        Return the scale name for the var named `var_name`.
        """
        return "%s@scale" % (var_name)


class AddQuantDequantPass(object):
    """
    Quantize the ops that do not have weights, and add quant_dequant op for the 
    quantized ops's inputs.
    """
    _supported_quantizable_op_type = [
        "pool2d", "elementwise_add", "concat", "softmax", "argmax", "transpose",
        "equal", "gather", "greater_equal", "greater_than", "less_equal",
        "less_than", "mean", "not_equal", "reshape", "reshape2",
        "bilinear_interp", "nearest_interp", "trilinear_interp", "slice",
        "squeeze", "elementwise_sub", "mul", "matmul", "relu", "relu6",
        "leaky_relu", "tanh", "swish", "scale", "transpose", "transpose2",
        "sigmoid", "pad2d", "flatten", "flatten2", "batch_norm", "layer_norm",
        "matmul_v2"
    ]

    # To be compatible with PaddleSlim, not remove _activation_type for now
    _activation_type = ["relu", "relu6", "leaky_relu", "tanh", "swish"]

    def __init__(self,
                 scope=None,
                 place=None,
                 moving_rate=0.9,
                 quant_bits=8,
                 skip_pattern=["skip_quant"],
                 quantizable_op_type=["elementwise_add", "pool2d"],
                 is_full_quantized=False):
        """
        Constructor.

        Args:
            scope(fluid.Scope): The scope is used to initialize these new parameters.
            place(fluid.CPUPlace|fluid.CUDAPlace|str): place is used to initialize new
                parameters described above. If ``place`` is string, it can be It can be ``cpu``
                or ``gpu:x``, where ``x`` is the index of the GPUs.
            moving_rate(float, optional): the param for 'quant_dequant_moving_average_abs_max' 
                quantization. Default is 0.9.
            quant_bits(int, optional): quantization bit number for activation. Default is 8.
            skip_pattern(str, optional): The user-defined quantization skip pattern, which
                will be presented in the name scope of an op. When the skip pattern is
                detected in an op's name scope, the corresponding op will not be quantized.
                Default is 'skip_quant'.
            quantizable_op_type(list[str], optional): List the type of ops that will be 
                quantized. Default is ["elementwise_add", "pool2d"]. 
            is_full_quantized(bool, optional): If set is_full_quantized as True, apply 
                quantization to all supported quantizable op type. If set is_full_quantized
                as False, only apply quantization to the op type according to the input 
                quantizable_op_type.
        """
        self._scope = scope
        self._place = _get_paddle_place(place)
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits
        self._is_test = None
        self._skip_pattern = skip_pattern

        if is_full_quantized:
            self._quantizable_op_type = \
                AddQuantDequantPass._supported_quantizable_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in quantizable_op_type:
                assert op_type in AddQuantDequantPass._supported_quantizable_op_type, \
                    op_type + " is not supported for quantization."
        self._quantizable_grad_op_type = [
            '%s_grad' % (op) for op in self._quantizable_op_type
        ]

        assert self._scope != None, "scope must not be None."
        assert self._place != None, "place must not be None."

    def apply(self, graph):
        """
        Add quant_dequant before some ops, such as the 'elementwise_add' and
        'pool2d' op.

        Args:
            graph(IrGraph): the target graph.
        Returns:
            None
        """
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        self._is_test = graph.is_test()
        dequantized_vars_map = collections.OrderedDict()

        # Forward stage, insert quant_dequant op
        all_op_nodes = graph.all_op_nodes()
        for op_node in all_op_nodes:
            if op_node.name() in self._quantizable_op_type:
                is_skip = False
                if isinstance(self._skip_pattern, list):
                    is_skip = op_node.op().has_attr("op_namescope") and \
                                   any(pattern in op_node.op().attr("op_namescope") for pattern in self._skip_pattern)
                elif isinstance(self._skip_pattern, str):
                    is_skip = op_node.op().has_attr("op_namescope") and \
                                   op_node.op().attr("op_namescope").find(self._skip_pattern) != -1
                is_quantized = op_node.op().has_attr("quantization_type") and \
                    op_node.op().attr("quantization_type") == "qat_with_weight"
                if is_skip or is_quantized or \
                    (not _is_input_all_not_persistable(graph, op_node)):
                    continue

                op_node.op()._set_attr("quantization_type",
                                       "qat_without_weight")
                op_node.op()._set_attr("activation_bits", self._quant_bits)
                op_node.op()._set_attr("with_quant_attr", True)
                arg_names = _get_op_input_var_names(op_node)
                for arg_name in arg_names:
                    in_node = graph._find_node_by_name(op_node.inputs, arg_name)
                    if arg_name in dequantized_vars_map:
                        quant_var_node = dequantized_vars_map[arg_name]
                    else:
                        quant_var_node, _ = \
                            self._inser_quant_dequant_moving_average_abs_max_op(
                            graph, in_node, self._quant_bits)
                        dequantized_vars_map[arg_name] = quant_var_node
                    graph.update_input_link(in_node, quant_var_node, op_node)

        # Backward stage, update input link
        for op_node in all_op_nodes:
            if op_node.name() in self._quantizable_grad_op_type:
                for input_name in op_node.input_arg_names():
                    if input_name in dequantized_vars_map:
                        in_node = graph._find_node_by_name(op_node.inputs,
                                                           input_name)
                        dequant_var_node = dequantized_vars_map[input_name]
                        graph.update_input_link(in_node, dequant_var_node,
                                                op_node)

        graph.resolve_hazard()
        return graph

    def _inser_quant_dequant_moving_average_abs_max_op(self, graph, var_node,
                                                       quant_bits):
        """Insert fake_quantize_dequantize_moving_average_abs_max op.
        """
        quant_var_node = graph.create_var_node(
            name="{}.quant_dequant".format(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=var_node.dtype())
        scale_in_node = graph.create_persistable_node(
            name="{}.quant_dequant.scale".format(var_node.name()),
            var_type=core.VarDesc.VarType.LOD_TENSOR,
            shape=[1],
            var_dtype=var_node.dtype())
        data_type = 'float64' if var_node.dtype(
        ) == core.VarDesc.VarType.FP64 else 'float32'
        _init_var_node(
            scale_in_node,
            np.array(
                [0.001], dtype=data_type),
            self._scope,
            self._place)

        scale_out_node = graph.create_var_node_from_desc(scale_in_node.var())
        ins = {'X': var_node, 'InScale': scale_in_node}
        outs = {'Out': quant_var_node, 'OutScale': scale_out_node}
        if not self._is_test:
            state_in_node = graph.create_persistable_node(
                name=unique_name.generate('quant_dequant.state'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            data_type = 'float64' if var_node.dtype(
            ) == core.VarDesc.VarType.FP64 else 'float32'
            _init_var_node(
                state_in_node,
                np.ones(
                    [1], dtype=data_type),
                self._scope,
                self._place)
            accum_in_node = graph.create_persistable_node(
                name=unique_name.generate('quant_dequant.accum'),
                var_type=core.VarDesc.VarType.LOD_TENSOR,
                var_dtype=var_node.dtype(),
                shape=[1])
            _init_var_node(
                accum_in_node,
                np.ones(
                    [1], dtype=data_type),
                self._scope,
                self._place)
            state_out_node = graph.create_var_node_from_desc(state_in_node.var(
            ))
            accum_out_node = graph.create_var_node_from_desc(accum_in_node.var(
            ))

            ins['InState'] = state_in_node
            ins['InAccum'] = accum_in_node
            outs['OutState'] = state_out_node
            outs['OutAccum'] = accum_out_node

        attrs = {
            'bit_length': quant_bits,
            'moving_rate': self._moving_rate,
            'is_test': self._is_test,
            'op_role': core.op_proto_and_checker_maker.OpRole.Forward
        }

        quant_op_node = graph.create_op_node(
            op_type='fake_quantize_dequantize_moving_average_abs_max',
            attrs=attrs,
            inputs=ins,
            outputs=outs)

        graph.link_to(var_node, quant_op_node)
        graph.link_to(scale_in_node, quant_op_node)
        graph.link_to(quant_op_node, quant_var_node)
        graph.link_to(quant_op_node, scale_out_node)

        if not self._is_test:
            graph.link_to(state_in_node, quant_op_node)
            graph.link_to(accum_in_node, quant_op_node)
            graph.link_to(quant_op_node, state_out_node)
            graph.link_to(quant_op_node, accum_out_node)

        return quant_var_node, scale_out_node
