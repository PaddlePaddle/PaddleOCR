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

from ..base.register import register_suite_info, register_model_info
from .model import TextDetModel
from .runner import TextDetRunner

# XXX: Hard-code relative path of repo root dir
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
register_suite_info({
    'suite_name': 'OCRDet',
    'model': TextDetModel,
    'runner': TextDetRunner,
    'runner_root_path': REPO_ROOT_PATH
})

PPOCRv3_DET_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml')

register_model_info({
    'model_name': 'ch_PP-OCRv3_det',
    'suite': 'OCRDet',
    'config_path': PPOCRv3_DET_CFG_PATH,
    'auto_compression_config_path': PPOCRv3_DET_CFG_PATH
})
