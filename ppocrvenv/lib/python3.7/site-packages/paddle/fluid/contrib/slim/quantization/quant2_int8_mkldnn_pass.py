#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from .... import core
from ....framework import IrGraph
from ....framework import _get_paddle_place

__all__ = ['Quant2Int8MkldnnPass']

OpRole = core.op_proto_and_checker_maker.OpRole


class Quant2Int8MkldnnPass(object):
    """
    Transform a quant model IrGraph into MKL-DNN supported INT8 IrGraph.
    The pass consists of the following transformations:
        1. gather scale values from fake quantize/dequantize operators,
        2. extract FP32 inference model graph from the quant graph, i.e.
            a.  remove fake quantize/dequantize operators,
            b.  dequantize conv2d and mul's weights,
        3. optimize the FP32 graph using standard FP32 optimization fuses
            (e.g. `conv2d`+`bn` -> `conv2d`),
        4. quantize the optimized FP32 graph using standard INT8v2 quantization
            passes (`cpu_quantize_pass`, `cpu_quantize_squash_pass`).
    """

    def __init__(self,
                 _ops_to_quantize,
                 _op_ids_to_skip=None,
                 _scope=None,
                 _place=None,
                 _core=None,
                 _debug=False):
        self._scope = _scope
        self._place = _get_paddle_place(_place)
        self._core = _core
        self._debug = _debug
        self._fake_quantize_types = [
            'fake_quantize_moving_average_abs_max',
            'fake_quantize_range_abs_max',
        ]
        self._fake_dequantize_types = [
            'fake_dequantize_max_abs', 'fake_channel_wise_dequantize_max_abs'
        ]
        self._fake_quantize_dequantize_types = [
            'fake_quantize_dequantize_abs_max',
            'fake_quantize_dequantize_moving_average_abs_max',
            'fake_channel_wise_quantize_dequantize_abs_max'
        ]
        self._ops_to_quantize = _ops_to_quantize
        self._op_ids_to_skip = _op_ids_to_skip if _op_ids_to_skip is not None else set(
            [-1])
        self._scale_immutable_ops = ['transpose2', 'reshape2', 'pool2d']
        self._scale_ops = ['scale']
        self._conv_ops = ['conv2d', 'depthwise_conv2d']
        self._pool_ops = ['pool2d']
        self._mul_ops = ['mul']
        self._fc_ops = ['fc']
        self._relu_ops = ['relu', 'relu6']
        self._matmul_ops = ['matmul']
        self._gru_ops = ['fusion_gru', 'multi_gru']
        self._lstm_ops = ['fusion_lstm']
        self._weight_thresholds = {}
        # Collect the Input and Output sclaes from Fake quant models
        self._var_quant_scales = {}
        self._max_range = {}
        self._s8_max = 127
        self._pass_idx = 0
        self._pass_group = 'int8'

    def apply(self, graph):
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'

        self._reset_pass_idx_and_group('int8')
        graph = self._label_skip_quantized_op(graph)
        graph = self._gather_weight_thresholds_from_fake(graph)
        graph = self._gather_input_scales_from_fake(graph)
        graph = self._gather_output_scales_from_attr(graph)
        graph = self._remove_fake_ops(graph)
        graph = self._dequantize_weights(graph)
        graph = self._optimize_fp32_graph(graph)
        graph = self._compute_weight_scales(graph)
        # This function causes nondeterministic quantization behavior
        # graph = self._update_relu_output_scales(graph)
        graph = self._propagate_scales(graph)
        graph = self._quantize_fp32_graph(graph)
        graph = self._final_optimizations(graph)
        graph = self._cleanup(graph)
        return graph

    def prepare_and_optimize_fp32(self, graph):
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'

        self._reset_pass_idx_and_group('fp32')
        graph = self._optimize_fp32_graph(graph)
        graph = self._final_optimizations(graph)
        graph = self._cleanup(graph)
        return graph

    def _reset_pass_idx_and_group(self, group):
        self._pass_idx = 0
        self._pass_group = group

    def _convert_scale2tensor(self, scale):
        tensor = core.LoDTensor()
        tensor.set(scale, core.CPUPlace())
        return tensor

    def _is_quantizing_all_ops(self):
        return len(self._ops_to_quantize) == 0

    def _is_any_of_op_types_in_graph(self, op_types, graph):
        return any(op.name() in op_types for op in graph.all_op_nodes())

    def _is_any_of_op_types_quantized(self, op_types, graph):
        return self._is_any_of_op_types_in_graph(
            op_types, graph) and (self._is_quantizing_all_ops() or
                                  any(op_type in self._ops_to_quantize
                                      for op_type in op_types))

    def _is_conv_quantized(self, graph):
        return self._is_any_of_op_types_quantized(self._conv_ops, graph)

    def _is_fc_quantized(self, graph):
        return self._is_any_of_op_types_quantized(self._fc_ops, graph)

    def _label_skip_quantized_op(self, graph):
        """
        For some ops(conv2d, depthwise_conv2d, mul, matml), find and label
        the skip quantized ops. cpu_quantize_placement_pass will use the
        label to identify it.
        For static models, the skip quantized ops have `skip_quant` attr.
        Therefore, it only needs to find and label the skip quantized ops for
        dygraph models, in which the quantized ops don't have `quantization_type`
        attr.
        """
        target_ops = self._conv_ops + self._mul_ops + self._matmul_ops
        for op_node in graph.all_op_nodes():
            if op_node.name() in target_ops and \
               not op_node.op().has_attr("quantization_type"):
                is_quantized_op = True
                for var_node in op_node.inputs:
                    for front_op_node in var_node.inputs:
                        if "quantize_dequantize" not in front_op_node.name():
                            is_quantized_op = False
                if not is_quantized_op:
                    op_node.op()._set_attr("skip_quant", True)
        return graph

    def _add_scale_for_vars(self, var_names, use_unsigned_int, lod_tensor):
        """
        Save quantization scales for variables. Do not overwrite.
        """
        scales = self._var_quant_scales
        for var_name in var_names:
            if var_name not in scales:
                scales[var_name] = (use_unsigned_int, lod_tensor)

    def _gather_input_scales_from_fake(self, graph):
        # fake_quantize_dequantize_abs_max doesn't have scale value
        fake_ops = ['fake_quantize_dequantize_moving_average_abs_max']
        fake_ops.extend(self._fake_quantize_types)

        for op in graph.all_op_nodes():
            if op.name() in fake_ops:
                bit_length = op.op().attr("bit_length")
                assert bit_length == 8, 'Unsupported number quantization bits ({}). Only 8 is supported now.'.format(
                    bit_length)

                input_name = op.input("X")[0]
                scale_name = op.input("InScale")[0]
                output_name = op.output("Out")[0]
                # Gather new weight scales after folding batchnorm in convolution
                scale = np.array(1.0 / self._load_param(
                    self._scope, scale_name)[0]).astype(np.float64)
                scale[scale == np.Inf] = 0.0
                lod_tensor = self._convert_scale2tensor(scale)
                use_unsigned_int = False
                self._add_scale_for_vars([input_name, output_name],
                                         use_unsigned_int, lod_tensor)

        return graph

    def _gather_weight_thresholds_from_fake(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._fake_dequantize_types:
                input_name = op.input("X")[0]
                if op.op().has_attr("max_range"):
                    _max_range = np.array(op.op().attr("max_range")).astype(
                        np.float64)
                    self._weight_thresholds[input_name] = np.array(
                        self._s8_max * self._s8_max /
                        _max_range).astype(np.float64)
                else:
                    scale_name = op.input("Scales")[0]
                    self._weight_thresholds[input_name] = np.array(
                        self._load_param(self._scope, scale_name)).astype(
                            np.float64)

        return graph

    def _gather_output_scales_from_attr(self, graph):
        for op in graph.all_op_nodes():
            if op.op().has_attr("out_threshold"):
                attr_scale = op.op().attr("out_threshold")
                if attr_scale == 0.0:
                    continue
                scale = np.array(1.0 / attr_scale).astype(np.float64)
                scale[scale == np.Inf] = 0.0
                scale_lod_tensor = self._convert_scale2tensor(scale)
                use_unsigned_int = False
                for output_name in op.op().outputs():
                    for out_var_name in op.op().output(output_name):
                        self._add_scale_for_vars(
                            [out_var_name], use_unsigned_int, scale_lod_tensor)

        return graph

    def _propagate_scales(self, graph):
        def _update_scale_op_in_scale(op, input, output):
            unsigned, tensor = self._var_quant_scales[output]
            scale = np.array(tensor) * op.op().attr("scale")
            new_tensor = self._convert_scale2tensor(scale.astype(np.float64))
            self._var_quant_scales[input] = (unsigned, new_tensor)

        def _update_scales(graph):
            waiting_for_scale = set()
            for op in graph.all_op_nodes():
                if op.name() in self._scale_immutable_ops:
                    input_name = op.input("X")[0]
                    output_name = op.output("Out")[0]
                    tensor_names = [input_name, output_name]

                    if all(name not in self._var_quant_scales
                           for name in tensor_names):
                        waiting_for_scale.update(tensor_names)
                        continue
                    elif input_name in self._var_quant_scales:
                        self._var_quant_scales[
                            output_name] = self._var_quant_scales[input_name]
                    elif output_name in self._var_quant_scales:
                        self._var_quant_scales[
                            input_name] = self._var_quant_scales[output_name]
                elif op.name() in self._scale_ops:
                    input_name = op.input("X")[0]
                    output_name = op.output("Out")[0]
                    if output_name in self._var_quant_scales:
                        _update_scale_op_in_scale(op, input_name, output_name)
            return waiting_for_scale

        waiting_for_scale = _update_scales(graph)
        waiting_for_scale_prev = set()

        while len(waiting_for_scale
                  ) != 0 and waiting_for_scale != waiting_for_scale_prev:
            waiting_for_scale_prev = waiting_for_scale
            waiting_for_scale = _update_scales(graph)

        return graph

    def _load_param(self, scope, param_name):
        return np.array(scope.find_var(param_name).get_tensor())

    def _remove_fake_ops(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._fake_quantize_types:
                self._remove_fake_quantize(graph, op)
            elif op.name() in self._fake_dequantize_types:
                self._remove_fake_dequantize(graph, op)
            elif op.name() in self._fake_quantize_dequantize_types:
                self._remove_fake_dequantize(graph, op)

        return graph

    def _remove_fake_quantize(self, graph, op):
        fake_quant_in = graph._find_node_by_name(op.inputs, op.input("X")[0])
        fake_quant_in_scale = graph._find_node_by_name(op.inputs,
                                                       op.input("InScale")[0])
        fake_quant_out = graph._find_node_by_name(op.outputs,
                                                  op.output("Out")[0])
        fake_quant_out_scale = graph._find_node_by_name(
            op.outputs, op.output("OutScale")[0])

        next_ops = fake_quant_out.outputs
        for next_op in next_ops:
            self._swap_inputs(next_op, fake_quant_out, fake_quant_in)
            graph.link_to(fake_quant_in, next_op)
        graph.safe_remove_nodes(
            {op, fake_quant_in_scale, fake_quant_out, fake_quant_out_scale})

        return graph

    def _remove_fake_dequantize(self, graph, op):
        fake_dequant_in = graph._find_node_by_name(op.inputs, op.input("X")[0])
        fake_dequant_out = graph._find_node_by_name(op.outputs,
                                                    op.output("Out")[0])

        next_ops = fake_dequant_out.outputs
        for next_op in next_ops:
            self._swap_inputs(next_op, fake_dequant_out, fake_dequant_in)
            graph.link_to(fake_dequant_in, next_op)
        graph.safe_remove_nodes({op, fake_dequant_out})

        return graph

    def _swap_inputs(self, op, old_input, new_input):
        for input_name in op.op().input_names():
            if old_input.name() in op.input(input_name):
                op.op().set_input(input_name, [
                    new_input.name() if x == old_input.name() else x
                    for x in op.input(input_name)
                ])

    def _dequantize_weights(self, graph):
        def _is_int8_weights(op_node, weight_name):
            weight_var_name = op_node.input(weight_name)[0]
            weight = self._load_param(self._scope, weight_var_name)
            return np.all(np.mod(weight, 1) == 0)

        for op in graph.all_op_nodes():
            if op.name() in self._conv_ops and _is_int8_weights(op, "Filter"):
                self._dequantize_op_weights(graph, op, "Filter", "Output")
            elif op.name() in self._mul_ops and _is_int8_weights(op, "Y"):
                self._dequantize_op_weights(graph, op, "Y", "Out")
        return graph

    def _dequantize_op_weights(self, graph, op_node, weight_name, output_name):
        weight_var_name = op_node.input(weight_name)[0]
        output_var_name = op_node.output(output_name)[0]
        # Convert int8 range weights to fp32 range weights
        scales = self._weight_thresholds[output_var_name]
        weight = self._load_param(self._scope, weight_var_name)
        if scales.size == 1 or scales.size == weight.shape[0]:
            w_fp32 = np.multiply(np.divide(weight, self._s8_max).T, scales.T).T
        elif len(weight.shape) > 1 and scales.size == weight.shape[1]:
            w_fp32 = np.multiply(np.divide(weight, self._s8_max), scales)
        else:
            raise ValueError(
                "The size of weight scales vector ({}) does not match the dimensions ({}) of the weights tensor {}."
                .format(scales.size, weight.shape, weight_var_name))
        w_fp32 = w_fp32.reshape(weight.shape).astype(np.float32)
        self._restore_var(weight_var_name, w_fp32)

    def _restore_var(self, name, array):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array, self._place)

    def _update_activations(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._conv_ops and not op.op().has_attr(
                    "fuse_activation"):
                activation = ""
                if op.op().has_attr("fuse_relu") and op.op().attr("fuse_relu"):
                    activation = "relu"
                elif op.op().has_attr("fuse_brelu") and op.op().attr(
                        "fuse_brelu"):
                    activation = "relu6"
                    alpha = 6.0
                    if op.op().has_attr("fuse_brelu_threshold"):
                        alpha = op.op().attr("fuse_brelu_threshold")
                    op.set_attr("fuse_alpha", alpha)
                op.set_attr("fuse_activation", activation)
        return graph

    def _remove_ctrl_vars(self, graph):
        remove_ctr_vars = set()
        for node in graph.all_var_nodes():
            if node.is_ctrl_var():
                remove_ctr_vars.add(node)
        graph.safe_remove_nodes(remove_ctr_vars)
        return graph

    def _optimize_fp32_graph(self, graph):
        graph = self._update_activations(graph)
        graph = self._remove_ctrl_vars(graph)
        graph = self._apply_pass(graph, 'attention_lstm_fuse_pass')
        graph = self._apply_pass(graph, 'seqconv_eltadd_relu_fuse_pass')
        #  graph = self._apply_pass(graph, 'seqpool_concat_fuse_pass')
        graph = self._apply_pass(graph, 'seqpool_cvm_concat_fuse_pass')
        #  graph = self._apply_pass(graph, 'embedding_fc_lstm_fuse_pass')
        graph = self._apply_pass(graph, 'fc_lstm_fuse_pass')
        graph = self._apply_pass(graph, 'mul_lstm_fuse_pass')
        graph = self._apply_pass(graph, 'fc_gru_fuse_pass')
        graph = self._apply_pass(graph, 'mul_gru_fuse_pass')
        graph = self._apply_pass(graph, 'multi_gru_fuse_pass')
        graph = self._apply_pass(graph, 'multi_gru_seq_fuse_pass')
        graph = self._apply_pass(graph, 'seq_concat_fc_fuse_pass')
        graph = self._apply_pass(graph, 'squared_mat_sub_fuse_pass')
        graph = self._apply_pass(graph, 'is_test_pass')
        graph = self._apply_pass(graph, 'mkldnn_placement_pass',
                                 ['mkldnn_enabled_op_types'], [set()])
        graph = self._apply_pass(graph, 'depthwise_conv_mkldnn_pass')
        graph = self._apply_pass(graph, 'conv_bn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_eltwiseadd_bn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_transpose_bn_fuse_pass')
        graph = self._apply_pass(graph,
                                 'conv_transpose_eltwiseadd_bn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_bias_mkldnn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_elementwise_add_mkldnn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_relu_mkldnn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_relu6_mkldnn_fuse_pass')
        graph = self._apply_pass(graph, 'fc_fuse_pass',
                                 ['use_gpu', 'use_fc_padding'], [False, False])
        graph = self._apply_pass(graph, 'repeated_fc_relu_fuse_pass')
        if self._is_fc_quantized(graph):
            graph = self._apply_pass(graph, 'fc_mkldnn_pass')
        graph = self._apply_pass(graph, 'matmul_transpose_reshape_fuse_pass')
        # the following pass should be the last one since it will work on all fused ops.
        graph = self._apply_pass(graph, 'runtime_context_cache_pass')
        return graph

    def _apply_pass(self, graph, pass_name, attrs=None, attr_values=None):
        ir_pass = core.get_pass(pass_name)
        cpp_graph = graph.graph
        if not cpp_graph.has('__param_scope__'):
            cpp_graph.set_not_owned('__param_scope__', self._scope)
        if attrs:
            assert attr_values and len(attrs) == len(
                attr_values
            ), "Different number of pass attributes and their values."
            for attr, value in zip(attrs, attr_values):
                ir_pass.set(attr, value)
        ir_pass.apply(cpp_graph)
        if self._debug:
            graph.draw('.', '{}_{}_{}'.format(self._pass_group, self._pass_idx,
                                              pass_name), graph.all_op_nodes())
        self._remove_unused_var_nodes(graph)
        self._pass_idx += 1
        return graph

    def _final_optimizations(self, graph):
        # remove dropout ops
        graph = self._apply_pass(graph, 'simplify_with_basic_ops_pass')
        # make some MKL-DNN ops working inplace
        graph = self._apply_pass(graph, 'mkldnn_inplace_pass')
        return graph

    def _cleanup(self, graph):
        graph = self._remove_unused_var_nodes(graph)
        graph = self._set_op_role_forward(graph)
        return graph

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
        return graph

    def _set_op_role_forward(self, graph):
        ops = graph.all_op_nodes()
        for op in ops:
            op.set_attr("op_role", OpRole.Forward)
        return graph

    def _compute_weight_scales(self, graph):
        def _compute_var_scales(ops, w_name, axis):
            for op in graph.all_op_nodes():
                if op.op().type() in ops:
                    weight_var_name = op.input(w_name)[0]
                    weights = np.array(
                        self._load_param(self._scope, weight_var_name))
                    scales = 1.0 / np.amax(
                        np.abs(weights.reshape(weights.shape[0], -1)).astype(
                            np.float64),
                        axis=axis)
                    scales[scales == np.Inf] = 0.0

                    lod_tensor = self._convert_scale2tensor(scales)
                    use_unsigned_int = False
                    self._var_quant_scales[weight_var_name] = (use_unsigned_int,
                                                               lod_tensor)

        def _compute_single_gru_weight_scales(wx_var_name, wh_var_name):
            wx = np.array(self._load_param(self._scope, wx_var_name))
            wh = np.array(self._load_param(self._scope, wh_var_name))
            OC = wh.shape[0]
            scale_ur = 1.0 / np.max(np.abs(
                np.concatenate(
                    [
                        wx[:, :2 * OC], wh.flatten()[:2 * OC * OC].reshape(OC, 2
                                                                           * OC)
                    ],
                    axis=0)),
                                    axis=0)
            scale_o = 1.0 / np.max(np.abs(
                np.concatenate(
                    [
                        wx[:, 2 * OC:], wh.flatten()[2 * OC * OC:].reshape(OC,
                                                                           OC)
                    ],
                    axis=0)),
                                   axis=0)

            gru_weights_scale = np.concatenate([scale_ur,
                                                scale_o]).astype('float')

            return self._convert_scale2tensor(gru_weights_scale)

        def _compute_gru_weight_scales(wx_name, wh_name):
            for op in graph.all_op_nodes():
                if op.op().type() in self._gru_ops:
                    assert len(op.input(wx_name)) == len(
                        op.input(wh_name)
                    ), 'Mismatch in number of weights inputs ({} for WeightX vs. {} for WeightH).'.format(
                        len(op.input(wx_name)), len(op.input(wh_name)))
                    for i, wx_var_name in enumerate(op.input(wx_name)):
                        wh_var_name = op.input(wh_name)[i]
                        use_unsigned_int = False
                        lod_tensor = _compute_single_gru_weight_scales(
                            wx_var_name, wh_var_name)
                        self._var_quant_scales[wx_var_name] = (use_unsigned_int,
                                                               lod_tensor)

        def _compute_single_lstm_weight_scales(wx_var_name, wh_var_name):
            wx = np.array(self._load_param(self._scope, wx_var_name))
            wh = np.array(self._load_param(self._scope, wh_var_name))

            lstm_weights_scale = 1.0 / np.max(
                np.abs(np.concatenate(
                    [wx[:, :], wh[:, :]], axis=0)), axis=0)
            lstm_weights_scale = lstm_weights_scale.astype('float')

            return self._convert_scale2tensor(lstm_weights_scale)

        def _compute_lstm_weight_scales(wx_name, wh_name):
            for op in graph.all_op_nodes():
                if op.op().type() in self._lstm_ops:
                    assert len(op.input(wx_name)) == len(
                        op.input(wh_name)
                    ), 'Mismatch in number of weights inputs ({} for WeightX vs. {} for WeightH).'.format(
                        len(op.input(wx_name)), len(op.input(wh_name)))
                    for i, wx_var_name in enumerate(op.input(wx_name)):
                        wh_var_name = op.input(wh_name)[i]
                        use_unsigned_int = False
                        lod_tensor = _compute_single_lstm_weight_scales(
                            wx_var_name, wh_var_name)
                        self._var_quant_scales[wx_var_name] = (use_unsigned_int,
                                                               lod_tensor)

        _compute_var_scales(self._conv_ops, "Filter", axis=1)
        _compute_var_scales(self._fc_ops, "W", axis=0)
        _compute_var_scales(self._gru_ops, "WeightH", axis=0)
        _compute_var_scales(self._lstm_ops, "WeightH", axis=0)
        _compute_gru_weight_scales("WeightX", "WeightH")
        _compute_lstm_weight_scales("WeightX", "WeightH")
        return graph

    def _find_avg_pooling_ids(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._pool_ops:
                if op.op().attr("pooling_type") == "avg":
                    self._op_ids_to_skip.add(op.id())
        return self._op_ids_to_skip

    def _update_relu_output_scales(self, graph):
        def _set_unsigned_scale(graph, ops, op_out_name, predicate):
            '''
            Sets the type of an output scale of a passed op type(s) to 'unsigned int8' if the
            predicate applied on op passes. Typically, the predicate checks if op's
            activation is set to relu.
            '''
            for op in graph.all_op_nodes():
                if op.name() in ops:
                    out_name = op.output(op_out_name)[0]
                    if out_name in self._var_quant_scales and predicate(op.op(
                    )):
                        is_unsigned, tensor = self._var_quant_scales[out_name]
                        if is_unsigned is False:
                            # If the variable is signed, it means that the scales for this var
                            # were computed for signed data, so the scale must be multiplied by 2
                            # to fill the entire range of uint8
                            scale = np.array(tensor) * 2
                            tensor = self._convert_scale2tensor(
                                scale.astype(np.float64))
                        self._var_quant_scales[out_name] = (True, tensor)
            return graph

        def conv_predicate(op):
            return op.attr("fuse_activation") in self._relu_ops

        graph = _set_unsigned_scale(graph, self._conv_ops, "Output",
                                    conv_predicate)

        def fc_predicate(op):
            return op.attr("activation_type") in self._relu_ops

        graph = _set_unsigned_scale(graph, self._fc_ops, "Out", fc_predicate)

        graph = _set_unsigned_scale(graph, self._relu_ops, 'Out',
                                    lambda op: True)

        return graph

    def _get_data_layout(self, graph):
        return 'NHWC' if self._is_conv_quantized(graph) else 'NCHW'

    def _quantize_fp32_graph(self, graph):
        graph = self._apply_pass(
            graph, 'cpu_quantize_placement_pass',
            ['quantize_enabled_op_types', 'quantize_excluded_op_ids'],
            [self._ops_to_quantize, self._find_avg_pooling_ids(graph)])
        graph = self._apply_pass(graph, 'scale_matmul_fuse_pass')
        graph = self._apply_pass(graph,
                                 'reshape_transpose_matmul_mkldnn_fuse_pass')
        graph = self._apply_pass(
            graph, 'cpu_quantize_pass', ['quant_var_scales', 'data_layout'],
            [self._var_quant_scales, self._get_data_layout(graph)])
        graph = self._apply_pass(graph, 'cpu_quantize_squash_pass')
        return graph
