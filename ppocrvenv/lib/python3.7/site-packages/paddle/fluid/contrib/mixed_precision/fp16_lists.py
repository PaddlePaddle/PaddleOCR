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

import copy
from ... import core

__all__ = ["CustomOpLists", "AutoMixedPrecisionLists"]


class AutoMixedPrecisionLists(object):
    """
    AutoMixedPrecisionLists is a class for black/white list. It can update
    pre-defined black list and white list according to users' custom black
    white lists. The lists are used for an algorithm which determines op's
    execution mode (fp32 or fp16).

    Args:
        custom_white_list (set): Users' custom white list.
        custom_black_list (set): Users' custom black list.
        custom_black_varnames (set): Users' custom black varibles' names.
    """

    def __init__(self,
                 custom_white_list=None,
                 custom_black_list=None,
                 custom_black_varnames=None):
        self._custom_white_list = custom_white_list
        self._custom_black_list = custom_black_list
        self.white_list = copy.copy(white_list)
        self.black_list = copy.copy(black_list)
        self.gray_list = copy.copy(gray_list)
        self.unsupported_list = copy.copy(unsupported_fp16_list)
        self.black_varnames = copy.copy(custom_black_varnames)
        self._update_list()

    def _update_list(self):
        """
        Update black and white list according to users' custom list.
        """
        if self._custom_white_list and self._custom_black_list:
            for op_name in self._custom_white_list:
                if op_name in self._custom_black_list:
                    raise ValueError("Custom white list overlap "
                                     "custom black list")
        if self._custom_white_list:
            for op_name in self._custom_white_list:
                if op_name in self.black_list:
                    self.black_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.white_list.add(op_name)
        if self._custom_black_list:
            for op_name in self._custom_black_list:
                if op_name in self.white_list:
                    self.white_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.black_list.add(op_name)
                self.unsupported_list.add(op_name)


# The three sets listed below are changed dynamiclly. They don't contain all
# paddle ops currently.

# The set of ops that support fp16 calculation and are considered numerically-
# safe and performance-critical. These ops are always converted to fp16.
white_list = {
    'conv2d',
    'matmul',
    'matmul_v2',
    'mul',
}

# The set of ops that support fp16 calculation and are considered numerically-
# dangerous and whose effects may also be observed in downstream ops.
black_list = {
    'exp',
    'square',
    'log',
    'mean',
    'sum',
    'cos_sim',
    'softmax',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'c_softmax_with_cross_entropy',
    'cross_entropy',
    'cross_entropy2',
    # fp16 is slower than fp32, though fp16 is supported.
    'lookup_table',
    'lookup_table_v2',
    # default fp32 can avoid return inf when the sum value large than 65504
    'reduce_sum',
}

# This set contains two types of ops. All ops supported fp16 calculation. One
# of two types is considered numerically-safe, but may be made unsafe by an
# upstream blacklist op. Another type do not have numerically-significant
# effects, like stack, flatten2.
gray_list = {
    'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'elementwise_mod',
    'elementwise_floordiv',
    'batch_norm',
    'layer_norm',
    'tanh',
    'sigmoid',
    'top_k',
    'pool2d',
    'pool3d',
    'dropout',
    'relu',
    'relu6',
    'leaky_relu',
    'soft_relu',
    'flatten2',
    'stack',
    'unstack',
    'uniform_random',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'gaussian_random_batch_size_like',
    'slice',
    'rank',
    'scale',
    'transpose2',
    'reshape2',
    'gather',
    'fill_constant',
    'get_tensor_from_selected_rows',
    'sign',
    'cast',
    'fused_bn_add_activation',
    'c_identity',
    'c_concat',
    'c_allreduce_sum',
    'concat',
    'split',
    'fused_feedforward',
    'fused_attention',
}

# The set of ops that don't support fp16 calculation
# lookup_table fp16 is slower than fp32, though fp16 is supported.
_sys_unsupported_fp16_list = []
if core.is_compiled_with_xpu():
    _, _, _sys_unsupported_fp16_list = core.op_supported_infos(
        'XPU', core.VarDesc.VarType.FP16)
elif core.is_compiled_with_npu():
    _, _, _sys_unsupported_fp16_list = core.op_supported_infos(
        'NPU', core.VarDesc.VarType.FP16)
else:
    _, _, _sys_unsupported_fp16_list = core.op_supported_infos(
        'GPU', core.VarDesc.VarType.FP16)

unsupported_fp16_list = {'lookup_table',
                         'lookup_table_v2'} | _sys_unsupported_fp16_list

CustomOpLists = AutoMixedPrecisionLists
