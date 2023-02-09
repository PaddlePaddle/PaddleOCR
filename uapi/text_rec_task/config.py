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

from ..base import BaseConfig


class TextRecConfig(BaseConfig):
    def update(self, dict_like_obj):
        dict_ = merge_config(self.dict, dict_like_obj)
        self.reset_from_dict(dict_)

    def load(self, config_file_path):
        dict_ = load_config(config_file_path)
        if not isinstance(dict_, dict):
            raise TypeError
        self.reset_from_dict(dict_)

    def dump(self, config_file_path):
        with open(config_file_path, 'w') as f:
            yaml.dump(self.dict, f, default_flow_style=False, sort_keys=False)

    def _update_dataset_config(self, dataset_root_path):
        _cfg = {
            'Train.dataset.name': 'SimpleDataSet',
            'Train.dataset.data_dir': dataset_root_path,
            'Train.dataset.label_file_list':
            [os.path.join(dataset_root_path, 'train.txt')],
            'Eval.dataset.name': 'SimpleDataSet',
            'Eval.dataset.data_dir': dataset_root_path,
            'Eval.dataset.label_file_list':
            [os.path.join(dataset_root_path, 'val.txt')],
        }
        self.update(_cfg)

    def _update_batch_size_config(self, batch_size):
        _cfg = {
            'Train.loader.batch_size_per_card': batch_size,
            'Eval.loader.batch_size_per_card': batch_size
        }
        self.update(_cfg)

    def _update_amp_config(self, amp):
        _cfg = {
            'Global.use_amp': amp is not None,
            'Global.amp_level': amp,
        }
        self.update(_cfg)

    def _update_lr_config(self, lr):
        _cfg = {
            'Optimizer.lr.learning_rate': lr,
            # 'Optimizer.lr.warmup_epoch': 0,
            # 'Optimizer.lr.name': 'Const',
        }
        self.update(_cfg)

    def _update_device_config(self, device):
        device = device.split(':')[0]
        default_cfg = {
            'Global.use_gpu': False,
            'Global.use_xpu': False,
            'Global.use_npu': False,
            'Global.use_mlu': False,
        }

        device_cfg = {
            'cpu': {},
            'gpu': {
                'Global.use_gpu': True
            },
            'xpu': {
                'Global.use_xpu': True
            },
            'mlu': {
                'Global.use_mlu': True
            }
        }
        default_cfg.update(device_cfg[device])
        self.update(default_cfg)

    def model_type(self, model_name):
        if model_name == 'ch_PP-OCRv3_rec':
            return 'SVTR_LCNet'
        if 'Models' in self.dict['Architecture']:
            return self.dict['Architecture']['Models']['Student']['algorithm']

        return self.dict['Architecture']['algorithm']
