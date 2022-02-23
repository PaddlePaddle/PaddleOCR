# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = []


class RuntimeBase(object):
    def __init__(self):
        pass

    def _set_basic_info(self, context):
        self.context = context

    def _run_worker(self):
        pass

    def _init_server(self, *args, **kwargs):
        pass

    def _run_server(self):
        pass

    def _stop_worker(self):
        pass

    def _save_inference_model(self, *args, **kwargs):
        pass

    def _save_persistables(self, *args, **kwargs):
        pass
