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
    """
    Abstract base class of Runner.

    Runner is responsible for executing training/inference/compression commands.
    """

    def __init__(self, runner_root_path):
        # The path to the directory where the scripts reside, 
        # e.g. the root directory of the repository.
        self.runner_root_path = abspath(runner_root_path)
        # Path to python interpreter
        self.python = sys.executable

    def prepare(self):
        """
        Make preparations for the execution of commands.

        For example, download prerequisites and install dependencies.
        """
        # By default we do nothing
        pass

    @abc.abstractmethod
    def train(self, config_path, cli_args, device):
        """
        Execute model training command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, config_path, cli_args, device):
        """
        Execute prediction command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, config_path, cli_args, device):
        """
        Execute model export command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, config_path, cli_args, device):
        """
        Execute model inference command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compression(self, config_path, train_cli_args, export_cli_args, device,
                    train_save_dir):
        """
        Execute model compression (quantization aware training and model export) commands.

        Args:
            config_path (str): Path of the configuration file.
            train_cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments used for model 
                training.
            train_cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments used for model 
                export.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.
            train_save_dir (str): Directory to store model snapshots and the exported model.
        """
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
