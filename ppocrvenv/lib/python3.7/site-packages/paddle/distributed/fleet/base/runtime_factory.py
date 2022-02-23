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
from ..runtime.collective_runtime import CollectiveRuntime
from ..runtime.parameter_server_runtime import ParameterServerRuntime
from ..runtime.the_one_ps import TheOnePSRuntime

__all__ = []


class RuntimeFactory(object):
    def __init__(self):
        pass

    def _create_runtime(self, context):
        if context["role_maker"]._is_collective:
            collective_runtime = CollectiveRuntime()
            collective_runtime._set_basic_info(context)
            return collective_runtime

        k_steps = context["valid_strategy"].a_sync_configs["k_steps"]

        if not context["role_maker"]._is_collective and k_steps >= 0:
            ps_runtime = TheOnePSRuntime()
            ps_runtime._set_basic_info(context)
            return ps_runtime
