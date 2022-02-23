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

import logging
import copy
import os
import numpy as np

import paddle
import paddle.nn.quant.quant_layers as quant_layers
from paddle.fluid.log_helper import get_logger
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX

from . import fuse_utils
from . import utils
from . import ptq_hooks
from . import ptq_config
from . import ptq_quantizer
from .ptq_registry import PTQRegistry

__all__ = ['ImperativePTQ']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativePTQ(object):
    """
    Static post training quantization.
    """

    def __init__(self, quant_config=ptq_config.default_ptq_config):
        """
        Constructor.

        Args:
            quant_config(PTQConfig): the config of post training quantization.
                The config has weight_quantizer and activation_quantizer.
                In default, the weight_quantizer is PerChannelAbsmaxQuantizer
                and the activation_quantizer is KLQuantizer.
        """
        super(ImperativePTQ, self).__init__()

        assert isinstance(quant_config, ptq_config.PTQConfig)

        self._quant_config = quant_config

    def quantize(self, model, inplace=False, fuse=False, fuse_list=None):
        """
        Add quant config and hook to the target layer.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
            inplace(bool): Whether apply quantization to the input model.
                           Default: False.
            fuse(bool): Whether to fuse layers.
                        Default: False.
            fuse_list(list): The layers' names to be fused. For example,
                "fuse_list = [["conv1", "bn1"], ["conv2", "bn2"]]".
                A TypeError would be raised if "fuse" was set as
                True but "fuse_list" was None.
                Default: None.
        Return
            quantized_model(paddle.nn.Layer): The quantized model.
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."
        if not inplace:
            model = copy.deepcopy(model)
        if fuse:
            model.eval()
            model = fuse_utils.fuse_layers(model, fuse_list)
        for name, layer in model.named_sublayers():
            if PTQRegistry.is_supported_layer(layer) \
                and utils.is_leaf_layer(layer) \
                and not self._is_skip_layer(layer):

                # Add quant config
                quant_config = copy.deepcopy(self._quant_config)
                if PTQRegistry.is_simulated_quant_layer(layer):
                    quant_config.enable_in_act_quantizer = True
                layer._quant_config = quant_config

                # register hook
                hook = ptq_hooks.quant_forward_post_hook
                quant_hook_handle = layer.register_forward_post_hook(hook)
                quant_config.quant_hook_handle = quant_hook_handle
                layer._forward_post_hooks.move_to_end(
                    quant_hook_handle._hook_id, last=False)

        return model

    def save_quantized_model(self, model, path, input_spec=None, **config):
        """
        1. Convert the quantized model
        2. Call jit.save to save the inference model
        3. Post process the inference model.

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

        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."

        # Convert and save dygraph quantized model
        self._convert(model)

        paddle.jit.save(layer=model, path=path, input_spec=input_spec, **config)

        # Load inference program
        is_dynamic_mode = False
        if paddle.in_dynamic_mode():
            is_dynamic_mode = True
            paddle.enable_static()

        place = paddle.CPUPlace()
        scope = paddle.static.global_scope()
        exe = paddle.static.Executor(place)

        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        model_filename = basename + INFER_MODEL_SUFFIX
        params_filename = basename + INFER_PARAMS_SUFFIX

        [infer_program, feed_target_names, fetch_targets] = (
            paddle.fluid.io.load_inference_model(
                dirname=dirname,
                executor=exe,
                model_filename=model_filename,
                params_filename=params_filename))

        # Process inference program
        self._clean_up(infer_program)
        self._gather_input_thresholds(infer_program, scope)
        self._remove_scale_op(infer_program)

        # Save final program
        paddle.fluid.io.save_inference_model(
            dirname=dirname,
            feeded_var_names=feed_target_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=infer_program.clone(),
            model_filename=model_filename,
            params_filename=params_filename)

        if is_dynamic_mode:
            paddle.disable_static()

    def _convert(self, model):
        """
        Convert the quantized model.

        Args:
            model(paddle.nn.Layer): The quantized model.
            inplace(bool): Whether apply conversion to the input model.
                           Default: False.
        Returns:
            None
        """

        for name, sub_layer in model.named_sublayers():
            if self._is_quant_layer(sub_layer):
                sub_layer._quant_config.quant_hook_handle.remove()

        self._cal_thresholds(model)

        for name, sub_layer in model.named_sublayers():
            if self._is_quant_layer(sub_layer):
                self._save_output_thresholds(sub_layer, sub_layer._quant_config)

        self._wrap_simulated_layers(model)

    def _cal_thresholds(self, model):
        """
        Calculate the thresholds of inputs and outputs.

        Args:
            model(paddle.nn.Layer): The quantized model.
        Returns:
            None
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The input model must be the instance of paddle.nn.Layer."

        total_num = 0
        cur_num = 0
        for name, sub_layer in model.named_sublayers():
            if self._is_quant_layer(sub_layer):
                total_num += 1

        for name, sub_layer in model.named_sublayers():
            if self._is_quant_layer(sub_layer):
                cur_num += 1
                if cur_num % 5 == 0:
                    _logger.info("Process the %s / %s layer" %
                                 (cur_num, total_num))

                quant_config = sub_layer._quant_config

                if quant_config.enable_in_act_quantizer:
                    quant_config.in_act_quantizer.cal_thresholds()
                quant_config.out_act_quantizer.cal_thresholds()

                if PTQRegistry.is_simulated_quant_layer(sub_layer):
                    weights = (sub_layer.weight, )
                    quant_config.wt_quantizer.sample_data(sub_layer, weights)
                    quant_config.wt_quantizer.cal_thresholds()

    def _save_output_thresholds(self, sub_layer, quant_config):
        """
        Save the output thresholds to the layer.

        Args:
            sub_layer(paddle.nn.Layer): The quantized layer.
            quant_config(PTQConfig): the quant config for the layer.
        Returns:
            None
        """
        assert isinstance(sub_layer, paddle.nn.Layer), \
            "The input model must be the instance of paddle.nn.Layer."

        layer_info = PTQRegistry.layer_info(sub_layer)

        output_names = layer_info.output_names
        output_thresholds = quant_config.out_act_quantizer.thresholds
        assert len(output_names) == 1
        assert len(output_thresholds) == 1
        save_name = output_names[0] + str(0) + "_threshold"
        sub_layer._set_op_attrs({save_name: output_thresholds[0]})
        sub_layer._set_op_attrs({"out_threshold": output_thresholds[0]})

    def _wrap_simulated_layers(self, model):
        """
        Replace conv2d and linear with the quantized layers, and save
        thresholds into the fake layers.
        Args:
            model(paddle.nn.Layer): The model to be quantized.
        Returns:
            None
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The input model must be the instance of paddle.nn.Layer."

        for name, sub_layer in model.named_sublayers():
            if self._is_quant_layer(sub_layer) \
                and PTQRegistry.is_simulated_quant_layer(sub_layer):

                quant_config = sub_layer._quant_config
                assert quant_config.enable_in_act_quantizer == True
                wt_quantizer = quant_config.wt_quantizer
                in_act_quantizer = quant_config.in_act_quantizer

                # create layer
                quant_layer_name = None
                for key, value in utils.layer_name_map.items():
                    if isinstance(sub_layer, value):
                        quant_layer_name = 'Quantized' + key
                        break
                assert quant_layer_name is not None

                if isinstance(wt_quantizer, ptq_quantizer.AbsmaxQuantizer):
                    weight_quantize_type = "abs_max"
                else:
                    weight_quantize_type = "channel_wise_abs_max"
                kwargs = {
                    "weight_quantize_type": weight_quantize_type,
                    "activation_quantize_type": "moving_average_abs_max",
                    "weight_bits": wt_quantizer.quant_bits,
                    "activation_bits": in_act_quantizer.quant_bits,
                }

                quant_layer = quant_layers.__dict__[quant_layer_name](sub_layer,
                                                                      **kwargs)

                # save the input thresholds
                assert hasattr(quant_layer, "_fake_quant_input")
                assert hasattr(quant_layer._fake_quant_input, "_scale")
                assert len(in_act_quantizer.thresholds) == 1
                input_threshold = np.array(
                    [in_act_quantizer.thresholds[0]], dtype=np.float32)
                quant_layer._fake_quant_input._scale.set_value(input_threshold)

                assert hasattr(quant_layer, "_fake_quant_weight")
                assert hasattr(quant_layer._fake_quant_weight, "_scale")
                assert len(wt_quantizer.thresholds) == 1
                weight_threshold = wt_quantizer.thresholds[0]
                if isinstance(weight_threshold, list):
                    weight_threshold = np.array(
                        weight_threshold, dtype=np.float32)
                else:
                    weight_threshold = np.array(
                        [weight_threshold], dtype=np.float32)
                quant_layer._fake_quant_weight._scale.set_value(
                    weight_threshold)

                # save the output thresholds
                self._save_output_thresholds(quant_layer, quant_config)

                # replace the layer
                parent_layer, sub_name = \
                    utils.find_parent_layer_and_sub_name(model, name)
                setattr(parent_layer, sub_name, quant_layer)

    def _gather_input_thresholds(self, program, scope):
        """
        Get and save input thresholds from the front ops.

        Args:
            program(Program): the input infer program.
            scope(Scope): the corresponding scope for the program.
        Returns:
            None
        """
        for op in utils.program_all_ops(program):
            for in_var_name in utils._get_op_input_var_names(op):
                previous_op = utils.find_previous_op(op.block, in_var_name)
                if previous_op is None:
                    continue

                if "quantize_dequantize" in previous_op.type or \
                    previous_op.type == "moving_average_abs_max_scale":
                    attr_name = previous_op.output('OutScale')[0]
                    in_threshold = utils.load_variable_data(scope, attr_name)
                    in_threshold = utils.fp_numpy_to_naive(in_threshold)
                    argname, index = utils._get_input_name_index(op,
                                                                 in_var_name)
                    op._set_attr(argname + str(index) + "_threshold",
                                 in_threshold)
                    op._set_attr("with_quant_attr", True)
                else:
                    for out_var_name in utils._get_op_output_var_names(
                            previous_op):
                        if out_var_name != in_var_name:
                            continue
                        argname, index = utils._get_output_name_index(
                            previous_op, out_var_name)
                        attr_name = argname + str(index) + "_threshold"
                        if not previous_op.has_attr(attr_name):
                            continue
                        threshold = previous_op.attr(attr_name)

                        argname, index = utils._get_input_name_index(
                            op, in_var_name)
                        attr_name = argname + str(index) + "_threshold"
                        op._set_attr(attr_name, threshold)
                        op._set_attr("with_quant_attr", True)

    def _clean_up(self, program):
        """
        Remove useless thresholds which are added in jit.save.

        Args:
            program(Program): the input infer program.
        Returns:
            None
        """

        def _helper(op, next_op, old_attr_name, new_attr_name):
            if op.has_attr(old_attr_name) and next_op.has_attr(old_attr_name) \
                and op.attr(old_attr_name) == next_op.attr(old_attr_name):
                threshold = op.attr(old_attr_name)
                op._remove_attr(old_attr_name)
                next_op._remove_attr(old_attr_name)
                next_op._set_attr(new_attr_name, threshold)
                next_op._set_attr("with_quant_attr", True)

        for op in utils.program_all_ops(program):
            if "quantize_dequantize" in op.type:
                # remove the thresholds in fake ops
                for attr_name in op.attr_names:
                    if "_threshold" in attr_name:
                        op._remove_attr(attr_name)
            elif op.type in ["conv2d", "matmul"]:
                # change the thresholds in conv2d/matmul + eleadd
                arg_name = "Output" if op.type == "conv2d" else "Out"
                out_var_name = op.output(arg_name)[0]
                next_ops = utils.find_next_ops(op.block, out_var_name)
                if len(next_ops) > 1 or next_ops[0].type != "elementwise_add":
                    continue
                next_op = next_ops[0]

                argname, index = utils._get_output_name_index(op, out_var_name)
                old_attr_name = argname + str(index) + "_threshold"

                argname, index = utils._get_output_name_index(
                    next_op, next_op.output("Out")[0])
                new_attr_name = argname + str(index) + "_threshold"

                _helper(op, next_op, old_attr_name, new_attr_name)
                _helper(op, next_op, "out_threshold", "out_threshold")

    def _remove_scale_op(self, program):
        """
        Remove the moving_average_abs_max_scale op.
        """
        for op in utils.program_all_ops(program):
            if op.type == "moving_average_abs_max_scale":
                in_var_name = op.input("X")[0]
                out_var_name = op.output("Out")[0]
                next_ops = utils.find_next_ops(op.block, out_var_name)
                for next_op in next_ops:
                    next_op._rename_input(out_var_name, in_var_name)

    @staticmethod
    def _is_skip_layer(layer):
        return hasattr(layer, "skip_quant") and layer.skip_quant == True

    @staticmethod
    def _is_quant_layer(layer):
        return hasattr(layer, "_quant_config")
