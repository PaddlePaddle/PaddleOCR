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

import copy
from paddle.fluid import core

from ..fp16_lists import white_list as white_list_fp16, black_list as black_list_fp16,\
    gray_list as gray_list_fp16

__all__ = ["AutoMixedPrecisionListsBF16"]


class AutoMixedPrecisionListsBF16(object):
    """
    AutoMixedPrecisionListsBF16 is a class for fp32/bf16 op types list. The lists are used for an
    algorithm which determines op's execution mode (fp32 or bf16).It can update pre-defined
    fp32 list and bf16 list according to users' custom fp32 bf16 lists.

    Args:
        custom_bf16_list (set): Users' custom bf16 list.
        custom_fp32_list (set): Users' custom fp32 list.
        custom_fp32_varnames (set): Users' custom fp32 variables' names.

    Examples:
        .. code-block:: python
        import paddle
        paddle.enable_static()
        with paddle.static.amp.bf16_guard():
            paddle.static.amp.AutoMixedPrecisionListsBF16(custom_fp32_list={'lstm'})
    """

    def __init__(self,
                 custom_bf16_list=None,
                 custom_fp32_list=None,
                 custom_fp32_varnames=None):
        self._custom_bf16_list = custom_bf16_list
        self._custom_fp32_list = custom_fp32_list
        self.bf16_list = copy.copy(bf16_list)
        self.fp32_list = copy.copy(fp32_list)
        self.gray_list = copy.copy(gray_list)
        self.bf16_initializer_list = copy.copy(bf16_initializer_list)
        self.unsupported_list = copy.copy(unsupported_list)
        self.fp32_varnames = copy.copy(custom_fp32_varnames)
        self._update_list()

    def _update_list(self):
        """
        Update fp32 and bf16 list according to users' custom list.
        """
        if self._custom_bf16_list and self._custom_fp32_list:
            for op_name in self._custom_bf16_list:
                if op_name in self._custom_fp32_list:
                    raise ValueError("Custom bf16 list overlap "
                                     "custom fp32 list")
        if self._custom_bf16_list:
            for op_name in self._custom_bf16_list:
                if op_name in self.fp32_list:
                    self.fp32_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.bf16_list.add(op_name)
        if self._custom_fp32_list:
            for op_name in self._custom_fp32_list:
                if op_name in self.bf16_list:
                    self.bf16_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.fp32_list.add(op_name)
                self.unsupported_list.add(op_name)


bf16_initializer_list = {'fill_constant', 'uniform_random'}

# always bf16
bf16_list = {'elementwise_add', }

# depends on the prev_op type
gray_list = {
    'cast',
    'fill_constant',
    'reduce_mean',
    'reshape2',
    'scale',
}

_, _, _sys_unsupported_bf16_list = core.op_supported_infos(
    'CPU', core.VarDesc.VarType.BF16)
unsupported_list = _sys_unsupported_bf16_list

fp32_list = black_list_fp16.copy().copy()
fp32_list |= white_list_fp16
fp32_list |= gray_list_fp16

fp32_list -= bf16_list
fp32_list -= gray_list
unsupported_list -= bf16_list
unsupported_list -= gray_list
