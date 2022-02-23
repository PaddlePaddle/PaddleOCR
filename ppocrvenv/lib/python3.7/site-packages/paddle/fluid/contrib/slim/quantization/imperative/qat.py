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
import sys
import os
import warnings

import paddle
import paddle.nn.quant.quant_layers as quant_layers
from paddle.fluid import dygraph, core, framework, unique_name
from paddle.fluid.framework import IrGraph
from paddle.fluid.executor import Executor, global_scope
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.fluid.io import load_inference_model, save_inference_model
from paddle.fluid.log_helper import get_logger
from .. import quantization_pass
from . import utils

__all__ = ['ImperativeQuantAware']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativeQuantAware(object):
    """
    Applying quantization aware training (QAT) to the dgraph model.
    """

    def __init__(
            self,
            quantizable_layer_type=['Conv2D', 'Linear', 'Conv2DTranspose'],
            weight_quantize_type='abs_max',
            activation_quantize_type='moving_average_abs_max',
            weight_bits=8,
            activation_bits=8,
            moving_rate=0.9,
            weight_preprocess_layer=None,
            act_preprocess_layer=None,
            weight_quantize_layer=None,
            act_quantize_layer=None):
        """
        The constructor for ImperativeQuantAware.

        Args:
            quantizable_layer_type(list[str | layer]): List the type of
                layers that will be quantized. Default is ['Conv2D', 'Linear'].
            weight_quantize_type(str): quantization type for weights,
                which supports 'abs_max' and 'channel_wise_abs_max'.
            activation_quantize_type(str): quantization type for activations,
                which supports 'abs_max' and 'moving_average_abs_max' now.
                If using 'abs_max' mode, the quantization scale will be
                calculated dynamically each step in both training and testing
                period. If using 'moving_average_abs_max', the static
                quantization scale will be calculated during training and
                used in inference.
            weight_bits(int): quantization bit number for weights, whereas
                the bias is not quantized.
            activation_bits(int): quantization bit number for activations.
            moving_rate(float): the parameter for 'moving_average_abs_max'
                quantization.
            weight_preprocess_layer(paddle.nn.Layer, optional): A paddle
                Layer that defines how to preprocess weight before quantization.
                Using this can quickly test if user's preprocess method works
                or not. The input is non-quantized weight and function returns
                processed weight to be quantized.
                If None, the weight will be quantized directly.
                Default is None.
            act_preprocess_layer(paddle.nn.Layer, optional): A paddle Layer
                that defines how to preprocess activation before quantization.
                Using this can quickly test if user's preprocess method works
                or not. The input is non-quantized activation and function returns
                processed activation to be quantized.
                If None, the activation will be quantized directly.
                Default is None.
            weight_quantize_layer(paddle.nn.Layer, optional): A paddle Layer that
                defines how to quantize weight.
                Using this can quickly test if user's quantization method works or not.
                In this layer, user should both define quantization method and
                dequantization method, that is, the function's input is non-quantized
                weight and returns dequantized weight.
                If None, will use uantization op defined by 'weight_quantize_type'.
                Default is None.
            act_quantize_layer(paddle.nn.Layer, optional): A paddle Layer that defines
                how to quantize activation.
                Using this can quickly test if user's quantization method works or not.
                In this layer, user should both define quantization method and
                dequantization method, that is, the function's input is non-quantized
                activation and returns dequantized activation. 
                If None, will use quantization op defined by 'activation_quantize_type'.
                Default is None.

        Note:
            If user sets attribute 'skip_quant' to a Layer that support dynamic
            quantization and sets it to true, the layer would not be quantized
            during training. If this attribute is not sets or the attribute is
            false, the Layer would be qunatized in training.

        Examples 1:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware
            from paddle.vision.models \
                import resnet
            
            model = resnet.resnet50(pretrained=True)

            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')
            
            # Add the fake quant logical.
            # The original model will be rewrite.
            # The outscale of outputs in supportted layers would be calculated.
            imperative_qat.quantize(model)

            # Fine-tune the quantized model
            # ...
            
            # Save quant model for the inference.
            imperative_qat.save_quantized_model(
                layer=model,
                model_path="./resnet50_qat",
                input_spec=[
                    paddle.static.InputSpec(
                    shape=[None, 3, 224, 224], dtype='float32')])

        Examples 2:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware

            class ImperativeModel(paddle.nn.Layer):
                def __init__(self):
                    super(ImperativeModel, self).__init__()
                    # self.linear_0 would skip the quantization.
                    self.linear_0 = paddle.nn.Linear(784, 400)
                    self.linear_0.skip_quant = True

                    # self.linear_1 would not skip the quantization.
                    self.linear_1 = paddle.nn.Linear(400, 10)
                    self.linear_1.skip_quant = False

                def forward(self, inputs):
                    x = self.linear_0(inputs)
                    x = self.linear_1(inputs)
                    return x

            model = ImperativeModel()
            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')

            # Add the fake quant logical.
            # The original model will be rewrite.
            #
            # There is only one Layer(self.linear1) would be added the
            # fake quant logical.
            imperative_qat.quantize(model)

            # Fine-tune the quantized model
            # ...

            # Save quant model for the inference.
            imperative_qat.save_quantized_model(
                layer=model,
                model_path="./imperative_model_qat")
        """
        super(ImperativeQuantAware, self).__init__()

        kwargs = {
            "quantizable_layer_type": quantizable_layer_type,
            "weight_quantize_type": weight_quantize_type,
            "activation_quantize_type": activation_quantize_type,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "moving_rate": moving_rate,
            "weight_preprocess_layer": weight_preprocess_layer,
            "act_preprocess_layer": act_preprocess_layer,
            "weight_quantize_layer": weight_quantize_layer,
            "act_quantize_layer": act_quantize_layer
        }

        self._quantize_inputs = ImperativeQuantizeInputs(**kwargs)

        self._quantize_outputs = ImperativeQuantizeOutputs(moving_rate)

    def quantize(self, model):
        """
        According to weights' and activations' quantization types,
        the model will be added some fake quant ops, such as
        fake_quantize_dequantize_moving_average_abs_max,
        fake_quantize_dequantize_abs_max and so on. At the same time,
        the out_scale value of outputs would be calculated.

        Args:
            model(paddle.nn.Layer): the model to be quantized.
        Returns:
            None

        Examples:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware

            class ImperativeModel(paddle.nn.Layer):
                def __init__(self):
                    super(ImperativeModel, self).__init__()
                    # self.linear_0 would skip the quantization.
                    self.linear_0 = paddle.nn.Linear(784, 400)
                    self.linear_0.skip_quant = True

                    # self.linear_1 would not skip the quantization.
                    self.linear_1 = paddle.nn.Linear(400, 10)
                    self.linear_1.skip_quant = False

                def forward(self, inputs):
                    x = self.linear_0(inputs)
                    x = self.linear_1(inputs)
                    return x

            model = ImperativeModel()
            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')

            # Add the fake quant logical.
            # The original model will be rewrite.
            #
            # There is only one Layer(self.linear1) would be added the
            # fake quant logical.
            imperative_qat.quantize(model)
        """
        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."
        self._quantize_inputs.apply(model)
        self._quantize_outputs.apply(model)

    def save_quantized_model(self, layer, path, input_spec=None, **config):
        self._quantize_outputs.save_quantized_model(layer, path, input_spec,
                                                    **config)


class ImperativeQuantizeInputs(object):
    """
    Based on the input params, add the quant_dequant computational
    logic both for activation inputs and weight inputs.
    """

    def __init__(
            self,
            quantizable_layer_type=['Conv2D', 'Linear', 'Conv2DTranspose'],
            weight_quantize_type='abs_max',
            activation_quantize_type='moving_average_abs_max',
            weight_bits=8,
            activation_bits=8,
            moving_rate=0.9,
            weight_preprocess_layer=None,
            act_preprocess_layer=None,
            weight_quantize_layer=None,
            act_quantize_layer=None):
        """
        The constructor for ImperativeQuantizeInputs. 

        Please refer to the args of ImperativeQuantAware.
        """
        super(ImperativeQuantizeInputs, self).__init__()

        self._quantizable_layer_type = tuple(
            utils.layer_name_map[layer]
            if layer in utils.layer_name_map else layer
            for layer in quantizable_layer_type)
        for layer in self._quantizable_layer_type:
            assert not isinstance(layer, str) \
                and layer in utils.fake_quant_input_layers, \
                "%s is unspported to be quantized." % layer

        quantize_type = {
            'abs_max', 'moving_average_abs_max', 'channel_wise_abs_max'
        }
        assert weight_quantize_type != 'moving_average_abs_max' \
            and weight_quantize_type in quantize_type, \
            "Unsupported weight_quantize_type: %s. It can only " \
            "be abs_max or channel_wise_abs_max." % weight_quantize_type
        # TODO (jc): activation_quantize_type supports range_abs_max
        assert activation_quantize_type == 'moving_average_abs_max', \
            "Unsupported activation_quantize_type: %s. It can " \
            "only be moving_average_abs_max now." \
            % activation_quantize_type

        bits_check = lambda bits: isinstance(bits, int) \
            and bits >= 0 and bits <= 16
        assert bits_check(weight_bits), \
            "weight_bits should be 1, 2,... or 16."
        assert bits_check(activation_bits), \
            "activation_bits should be 1, 2,... or 16."

        layer_check = lambda method: method is None or \
            issubclass(method, dygraph.layers.Layer)
        assert layer_check(weight_preprocess_layer), \
            "weight_preprocess should be nn.Layer."
        assert layer_check(act_preprocess_layer), \
            "act_preprocess should be nn.Layer."
        assert layer_check(weight_quantize_layer), \
            "weight_quantize should be nn.Layer."
        assert layer_check(act_quantize_layer), \
            "act_quantize should be nn.Layer."

        self._kwargs = {
            "weight_quantize_type": weight_quantize_type,
            "activation_quantize_type": activation_quantize_type,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "moving_rate": moving_rate,
            "weight_pre_layer": weight_preprocess_layer,
            "act_pre_layer": act_preprocess_layer,
            "weight_quant_layer": weight_quantize_layer,
            "act_quant_layer": act_quantize_layer
        }

    def apply(self, model):
        """
        Quantize the weights and activations to calculate for specific 
        layers.

        Args:
            model(paddle.nn.Layer): The target model which would
                calculate the input quantization scale.

        Returns:
            None
        """

        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."

        for name, cur_layer in model.named_sublayers():
            if not isinstance(cur_layer, self._quantizable_layer_type) \
                or (hasattr(cur_layer, "skip_quant") \
                    and cur_layer.skip_quant == True):
                continue

            parent_layer, sub_name = \
                utils.find_parent_layer_and_sub_name(model, name)

            cur_quant_layer = self._get_input_quantized_layer(cur_layer)
            setattr(parent_layer, sub_name, cur_quant_layer)

    def _get_input_quantized_layer(self, layer):
        quant_layer_name = None

        for key, value in utils.layer_name_map.items():
            if isinstance(layer, value):
                quant_layer_name = 'Quantized' + key
                break
        assert quant_layer_name is not None, \
            "The layer %s is unsupported to be quantized." \
            % layer.full_name()

        return quant_layers.__dict__[quant_layer_name](layer, **self._kwargs)


class ImperativeQuantizeOutputs(object):
    """
    Calculate the output scales for target layers.
    """

    def __init__(self, moving_rate=0.9):
        """
        The constructor for ImperativeQuantizeOutputs.

        Args:
            moving_rate(float): The decay coefficient of moving average.
                                The default value is 0.9.
        """
        super(ImperativeQuantizeOutputs, self).__init__()
        self._moving_rate = moving_rate

    def apply(self, model):
        """
        Insert the `moving_average_abs_max_scale` layers to calculate the
        output scales for specific layers in the dygraph model.

        Args:
            model(paddle.nn.Layer): The target model which would be
                calculate the output quantization scale.

        Returns:
            None
        """
        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."

        for cur_name, cur_layer in model.named_sublayers():
            if '_act_preprocess' in cur_name:
                continue
            if not self._is_target_layer(cur_layer):
                continue

            parent_layer, sub_name = \
                utils.find_parent_layer_and_sub_name(model, cur_name)

            if isinstance(cur_layer, tuple(utils.fake_quant_output_layers)):
                cur_quant_layer = quant_layers.FakeQuantMAOutputScaleLayer(
                    cur_layer, self._moving_rate)
            else:
                cur_quant_layer = quant_layers.MAOutputScaleLayer(
                    cur_layer, self._moving_rate)

            setattr(parent_layer, sub_name, cur_quant_layer)

    def save_quantized_model(self, model, path, input_spec=None, **config):
        """
        Save the quantized model for the inference.

        Args:
            model (Layer): The model to be saved.
            path (str): The path prefix to save model. The format is 
                ``dirname/file_prefix`` or ``file_prefix``.
            input_spec (list[InputSpec|Tensor], optional): Describes the input
                of the saved model's forward method, which can be described by
                InputSpec or example Tensor. If None, all input variables of 
                the original Layer's forward method would be the inputs of
                the saved model. Default None.
            **configs (dict, optional): Other save configuration options for
                compatibility. We do not recommend using these configurations,
                they may be removed in the future. If not necessary, DO NOT use
                them. Default None.
                The following options are currently supported:
                (1) output_spec (list[Tensor]): Selects the output targets of
                the saved model. By default, all return variables of original
                Layer's forward method are kept as the output of the saved model.
                If the provided ``output_spec`` list is not all output variables, 
                the saved model will be pruned according to the given
                ``output_spec`` list. 

        Returns:
            None
        """
        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."

        paddle.jit.save(layer=model, path=path, input_spec=input_spec, **config)

        is_dynamic_mode = False
        if paddle.in_dynamic_mode():
            is_dynamic_mode = True
            paddle.enable_static()

        place = core.CPUPlace()
        scope = global_scope()
        exe = Executor(place)

        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        model_filename = basename + INFER_MODEL_SUFFIX
        params_filename = basename + INFER_PARAMS_SUFFIX

        [infer_program, feed_target_names, fetch_targets] = (
            load_inference_model(
                dirname=dirname,
                executor=exe,
                model_filename=model_filename,
                params_filename=params_filename))

        self._gather_scales(infer_program, scope, fetch_targets)

        # Remove `moving_average_abs_max_scale` node in sub graphs.
        graph = IrGraph(core.Graph(infer_program.desc), for_test=False)
        for sub_graph in graph.all_sub_graphs():
            for _op in sub_graph.all_op_nodes():
                if _op.name() == "moving_average_abs_max_scale":
                    sub_graph.safe_remove_nodes(_op)
            sub_graph.resolve_hazard()
        infer_program = graph.to_program()

        self._set_skip_quant_attr(infer_program)

        save_inference_model(
            dirname=dirname,
            feeded_var_names=feed_target_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=infer_program.clone(),
            model_filename=model_filename,
            params_filename=params_filename,
            clip_extra=False)

        if is_dynamic_mode:
            paddle.disable_static()

    def _is_target_layer(self, layer):
        """
        Whether the layer needs to calculate output scales.
        """
        flag = False
        if isinstance(layer, dygraph.Layer):
            # exclude fake_quant ops in quant_layers file
            if utils.is_leaf_layer(layer) and \
                not isinstance(layer, tuple(utils.fake_quant_leaf_layers)):
                flag = True

            if isinstance(layer, tuple(utils.fake_quant_wrap_layers)):
                flag = True

            if isinstance(layer, paddle.nn.quant.FloatFunctionalLayer):
                flag = True

        return flag

    def _gather_scales(self, program, scope, fetch_targets):
        """
        Get all scales from fake ops, save them into the corresponding ops
        and delete all moving_average_abs_max_scale ops.
        """

        def _gather_input_scale():
            target_ops = []
            skip_ops = utils.fake_quantize_dequantize_op_types + \
                ["moving_average_abs_max_scale"]
            for block in program.blocks:
                for op in block.ops:
                    if op.type not in skip_ops:
                        target_ops.append(op)

            for op in target_ops:
                for in_var_name in utils._get_op_input_var_names(op):
                    previous_op = utils.find_previous_op(op.block, in_var_name)

                    if previous_op is not None and \
                        ("quantize_dequantize" in previous_op.type or \
                        previous_op.type == "moving_average_abs_max_scale"):
                        scale_name = previous_op.output('OutScale')[0]
                        in_scale = utils.load_variable_data(scope, scale_name)
                        in_scale = utils.fp_numpy_to_naive(in_scale)
                        argname, index = utils._get_input_name_index(
                            op, in_var_name)
                        op._set_attr(argname + str(index) + "_threshold",
                                     in_scale)
                        op._set_attr("with_quant_attr", True)

        def _gather_output_scale():
            target_ops = []
            for block in program.blocks:
                for op in block.ops:
                    if op.type == "moving_average_abs_max_scale":
                        target_ops.append(op)

            for op in target_ops:
                in_var_name = op.input('X')[0]
                out_var_name = op.output('Out')[0]
                block = op.block
                previous_op = utils.find_previous_op(block, in_var_name)
                next_ops = utils.find_next_ops(block, out_var_name)

                out_scale_name = op.output('OutScale')[0]
                out_scale = utils.load_variable_data(scope, out_scale_name)
                out_scale = utils.fp_numpy_to_naive(out_scale)

                if previous_op.type != "feed":
                    res = utils._get_output_name_index(previous_op, in_var_name)
                    if res is not None:
                        argname, index = res
                        previous_op._set_attr(
                            argname + str(index) + "_threshold", out_scale)
                        previous_op._set_attr("out_threshold", out_scale)
                        previous_op._set_attr("with_quant_attr", True)

                for next_op in next_ops:
                    next_op._rename_input(out_var_name, in_var_name)
                    # If next_op is `fetch` and out_var_name in fetch_targets,
                    # fetch_targets must update to in_var_name when rename input.
                    for i in range(len(fetch_targets)):
                        if fetch_targets[i].name == out_var_name:
                            fetch_targets[i] = block.var(in_var_name)

        _gather_input_scale()
        _gather_output_scale()

    def _set_skip_quant_attr(self, program):
        """
        Label the skip quantized ops.
        """
        for block in program.blocks:
            for op in block.ops:
                if self._is_skip_quant_op(block, op):
                    op._set_attr("skip_quant", True)
                    op._set_attr("with_quant_attr", True)

    def _is_skip_quant_op(self, block, in_op):
        """
        The input op should be skipped quantization.
        1. the type of input op should be conv2d, depthwise_conv2d or matmul
        2. the previous ops of the input op are not fake_quantize_dequantize ops
        """
        target_op_types = [
            "conv2d", "depthwise_conv2d", "matmul", "conv2d_transpose"
        ]
        if in_op.type not in target_op_types:
            return False

        previous_ops = [utils.find_previous_op(block, arg_name) \
            for arg_name in in_op.input_arg_names]
        return any(op is not None and op.type not in \
            utils.fake_quantize_dequantize_op_types for op in previous_ops)
