# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

import yaml
import collections

from tools.program import load_config, merge_config

from ..text_rec_task import TextRecConfig


class TextDetConfig(TextRecConfig):
    def update_batch_size(self, batch_size):
        _cfg = {'Train.loader.batch_size_per_card': batch_size}
        self.update(_cfg)
