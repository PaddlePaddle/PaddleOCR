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
import os.path as osp

# TODO: Set cache directory in a global config module
CACHE_DIR = osp.abspath(osp.join('.cache', 'paddle_uapi'))

if not osp.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_cache_dir(*args, **kwargs):
    # `args` and `kwargs` reserved for extension
    return CACHE_DIR


def create_yaml_config_file(tag, noclobber=True):
    cache_dir = get_cache_dir()
    file_path = osp.join(cache_dir, f"{tag}.yml")
    if noclobber and osp.exists(file_path):
        raise FileExistsError
    # Overwrite content
    with open(file_path, 'w') as f:
        f.write("")
    return file_path
