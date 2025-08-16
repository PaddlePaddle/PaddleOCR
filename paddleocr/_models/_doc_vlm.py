# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import abc

from .._utils.cli import (
    get_subcommand_args,
    perform_simple_inference,
)
from .base import PaddleXPredictorWrapper, PredictorCLISubcommandExecutor
from paddlex.utils.pipeline_arguments import custom_type


class BaseDocVLM(PaddleXPredictorWrapper):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._extra_init_args = {}
        super().__init__(*args, **kwargs)

    def _get_extra_paddlex_predictor_init_args(self):
        return self._extra_init_args


class BaseDocVLMSubcommandExecutor(PredictorCLISubcommandExecutor):
    input_validator = staticmethod(custom_type(dict))

    @property
    @abc.abstractmethod
    def wrapper_cls(self):
        raise NotImplementedError

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        params["input"] = self.input_validator(params["input"])
        perform_simple_inference(self.wrapper_cls, params)
