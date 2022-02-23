# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import six
import warnings

from paddle.fluid.framework import default_main_program, in_dygraph_mode


class LayerOpsRecoder:
    """
    Record generated operators information in nn.Layer.
    """

    def __init__(self, start=-1, end=-1, ops=None, is_valid=False, hooks=None):
        self.start = start
        self.end = end
        self.ops = ops
        self.is_valid = is_valid
        self.hooks = hooks


def record_program_ops_pre_hook(layer, inputs):
    """
    A pre-hook to mark op numbers before enter layer.forward.
    """
    if not in_dygraph_mode():
        if layer._op_recorder.start < 0:
            layer._op_recorder.start = len(default_main_program().current_block(
            ).ops)
            layer._op_recorder.is_valid = True
        else:
            layer._op_recorder.is_valid = False
            warnings.warn(
                "{} has recorded the op information before. Please check whether you call this layer twice.".
                format(layer._full_name))

    return None


def set_op_customized_attrs_post_hook(layer, inputs, outputs):
    """
    A post-hook to append customized attributes into all operators generated in current layer.
    """
    if not in_dygraph_mode() and layer._op_recorder.is_valid:

        start = layer._op_recorder.start
        end = len(default_main_program().current_block().ops)
        assert (start >= 0 and end >= start)
        ops = default_main_program().current_block().ops[start:end]

        layer._op_recorder.end = end
        layer._op_recorder.ops = ops

        for op in ops:
            for attr_name, val in six.iteritems(layer._customized_attrs):
                op._set_attr(attr_name, val)

        # remove pre-hook and post-hook
        for hook_helper in layer._op_recorder.hooks:
            hook_helper.remove()

    return None
