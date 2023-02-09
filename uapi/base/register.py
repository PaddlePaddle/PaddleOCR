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

model_zoo = OrderedDict()
suite_zoo = OrderedDict()

MODEL_INFO_REQUIRED_KEYS = ('model_name', 'suite', 'config_path',
                            'auto_compression_config_path')
MODEL_INFO_PRIMARY_KEY = 'model_name'
assert MODEL_INFO_PRIMARY_KEY in MODEL_INFO_REQUIRED_KEYS
SUITE_INFO_REQUIRED_KEYS = ('suite_name', 'model', 'runner', 'runner_root_path')
SUITE_INFO_PRIMARY_KEY = 'suite_name'
assert SUITE_INFO_PRIMARY_KEY in SUITE_INFO_REQUIRED_KEYS

# Relations:
# 'suite' in model info <-> 'suite_name' in suite info


class PaddleModel(object):
    # We constrain function params here
    def __new__(cls, model_name):
        model_info = get_registered_model_info(model_name)
        return build_model_from_model_info(model_info)


def _validate_model_info(model_info):
    for key in MODEL_INFO_REQUIRED_KEYS:
        if key not in model_info:
            raise KeyError(f"Key '{key}' is required, but not found.")


def _validate_suite_info(suite_info):
    for key in SUITE_INFO_REQUIRED_KEYS:
        if key not in suite_info:
            raise KeyError(f"Key '{key}' is required, but not found.")


def register_model_info(data):
    global model_zoo
    _validate_model_info(data)
    prim_key = data[MODEL_INFO_PRIMARY_KEY]
    model_zoo[prim_key] = data


def register_suite_info(data):
    global suite_zoo
    _validate_suite_info(data)
    prim_key = data[SUITE_INFO_PRIMARY_KEY]
    suite_zoo[prim_key] = data


def get_registered_model_info(prim_key):
    return model_zoo[prim_key]


def get_registered_suite_info(prim_key):
    return suite_zoo[prim_key]


def build_runner_from_model_info(model_info):
    suite_name = model_info['suite']
    # `suite_name` being the primary key of suite info
    suite_info = get_registered_suite_info(suite_name)
    runner_cls = suite_info['runner']
    runner_root_path = suite_info['runner_root_path']
    return runner_cls(runner_root_path=runner_root_path)


def build_model_from_model_info(model_info):
    model_name = model_info['model_name']
    suite_name = model_info['suite']
    # `suite_name` being the primary key of suite info
    suite_info = get_registered_suite_info(suite_name)
    model_cls = suite_info['model']
    return model_cls(model_name=model_name)
