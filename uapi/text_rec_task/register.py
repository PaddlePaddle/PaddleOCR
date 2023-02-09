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

import os.path as osp

from ..base.register import register_arch_info, register_model_info
from .model import TextRecModel
from .runner import TextRecRunner

# XXX: Hard-code relative path of repo root dir
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
register_model_info({
    'model_name': 'TextRecModel',
    'model_cls': TextRecModel,
    'runner_cls': TextRecRunner,
    'repo_root_path': REPO_ROOT_PATH
})

PPOCRv3_REC_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml')

register_arch_info({
    'arch_name': 'ch_PP-OCRv3_rec',
    'model': 'TextRecModel',
    'config_path': PPOCRv3_REC_CFG_PATH,
    'auto_compression_config_path': PPOCRv3_REC_CFG_PATH
})
