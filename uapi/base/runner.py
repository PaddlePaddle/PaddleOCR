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
import sys
import abc

from .utils.misc import run_cmd as _run_cmd, abspath


class BaseRunner(metaclass=abc.ABCMeta):
    def __init__(self, runner_root_path):
        self.runner_root_path = abspath(runner_root_path)

        self.python = sys.executable

    @abc.abstractmethod
    def train(self, config_file_path, cli_args, device):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, config_file_path, cli_args, device):
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, config_file_path, cli_args, device):
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, config_file_path, cli_args, device):
        raise NotImplementedError

    @abc.abstractmethod
    def compression(self, config_file_path, cli_args, device):
        raise NotImplementedError

    def distributed(self, device):
        # TODO: docstring
        python = self.python
        if device is None:
            # By default use a GPU device
            return python, 'gpu'
        # According to https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html
        if ':' not in device:
            return python, device
        else:
            device, dev_ids = device.split(':')
            num_devices = len(dev_ids.split(','))
        if num_devices > 1:
            python += " -m paddle.distributed.launch"
            python += f" --gpus {dev_ids}"
        elif num_devices == 1:
            # TODO: Accommodate Windows system
            python = f"CUDA_VISIBLE_DEVICES={dev_ids} {python}"
        return python, device

    def run_cmd(self, cmd, switch_wdir=False, **kwargs):
        if switch_wdir:
            if 'wd' in kwargs:
                raise KeyError
            if isinstance(switch_wdir, str):
                # In this case `switch_wdir` specifies a relative path
                kwargs['wd'] = os.path.join(self.runner_root_path, switch_wdir)
            else:
                kwargs['wd'] = self.runner_root_path
        return _run_cmd(cmd, **kwargs)
