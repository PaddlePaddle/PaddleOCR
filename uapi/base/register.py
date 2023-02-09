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

from collections import OrderedDict

# TODO: We need a lightweight RDBMS to handle these tables.

arch_zoo = OrderedDict()
model_zoo = OrderedDict()

ARCH_INFO_REQUIRED_KEYS = ('arch_name', 'model', 'config_path',
                           'auto_compression_config_path')
ARCH_INFO_PRIMARY_KEY = 'arch_name'
assert ARCH_INFO_PRIMARY_KEY in ARCH_INFO_REQUIRED_KEYS
MODEL_INFO_REQUIRED_KEYS = ('model_name', 'model_cls', 'runner_cls',
                            'repo_root_path')
MODEL_INFO_PRIMARY_KEY = 'model_name'
assert MODEL_INFO_PRIMARY_KEY in MODEL_INFO_REQUIRED_KEYS

# Relations:
# 'model' in arch info <-> 'model_name' in model info


class PaddleModel(object):
    # We constrain function params here
    def __new__(cls, model_name):
        arch_info = get_registered_arch_info(model_name)
        return build_model_from_arch_info(arch_info)


def _validate_arch_info(arch_info):
    for key in ARCH_INFO_REQUIRED_KEYS:
        if key not in arch_info:
            raise KeyError(f"Key '{key}' is required, but not found.")


def _validate_model_info(model_info):
    for key in MODEL_INFO_REQUIRED_KEYS:
        if key not in model_info:
            raise KeyError(f"Key '{key}' is required, but not found.")


def register_arch_info(data):
    global arch_zoo
    _validate_arch_info(data)
    prim_key = data[ARCH_INFO_PRIMARY_KEY]
    arch_zoo[prim_key] = data


def register_model_info(data):
    global model_zoo
    _validate_model_info(data)
    prim_key = data[MODEL_INFO_PRIMARY_KEY]
    model_zoo[prim_key] = data


def get_registered_arch_info(prim_key):
    return arch_zoo[prim_key]


def get_registered_model_info(prim_key):
    return model_zoo[prim_key]


def build_runner_from_arch_info(arch_info):
    model_name = arch_info['model']
    # `model_name` being the primary key of model info
    model_info = get_registered_model_info(model_name)
    runner_cls = model_info['runner_cls']
    repo_root_path = model_info['repo_root_path']
    return runner_cls(repo_root_path=repo_root_path)


def build_model_from_arch_info(arch_info):
    arch_name = arch_info['arch_name']
    model_name = arch_info['model']
    # `model_name` being the primary key of model info
    model_info = get_registered_model_info(model_name)
    model_cls = model_info['model_cls']
    return model_cls(model_name=arch_name)
