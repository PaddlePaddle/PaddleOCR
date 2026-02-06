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


def create_config_from_structure(structure, *, unset=None, config=None):
    if config is None:
        config = {}
    for k, v in structure.items():
        if v is unset:
            continue
        idx = k.find(".")
        if idx == -1:
            config[k] = v
        else:
            sk = k[:idx]
            if sk not in config:
                config[sk] = {}
            create_config_from_structure({k[idx + 1 :]: v}, config=config[sk])
    return config
