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

from ..base import BaseModel
from ..base.utils.path import get_cache_dir
from ..base.utils.arg import CLIArgument
from ..base.utils.misc import abspath
from .config import TextRecConfig


class TextRecModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.update_config_cls()

    def update_config_cls(self):
        self.config_cls = TextRecConfig

    def train(self,
              dataset,
              batch_size=None,
              learning_rate=None,
              epochs_iters=None,
              device=None,
              resume_path=None,
              dy2st=None,
              amp=None,
              save_dir=None):
        # NOTE: We must use an absolute path here, 
        # so we can run the scripts either inside or outside the repo dir.
        dataset = abspath(dataset)
        if dy2st is None:
            dy2st = False
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.arch_info['config_path']
        config = self.config_cls.build_from_file(config_file_path)
        config._update_dataset_config(dataset)
        config._update_amp_config(amp)
        config._update_device_config(device)

        if batch_size is not None:
            config._update_batch_size_config(batch_size)
        if learning_rate is not None:
            config._update_lr_config(learning_rate)
        if epochs_iters is not None:
            config.update({'Global.epoch_num': epochs_iters})
        if resume_path is not None:
            config.update({'Global.checkpoints': resume_path})
        if dy2st:
            config.update({'Global.to_static': True})
        if save_dir is not None:
            config.update({'Global.save_model_dir': save_dir})

        config_file_path = self.config_file_path
        config.dump(config_file_path)
        self.runner.train(config_file_path, device)

    def predict(self,
                weight_path=None,
                device=None,
                input_path=None,
                save_dir=None):
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.arch_info['config_path']
        config = self.config_cls.build_from_file(config_file_path)
        if weight_path is not None:
            config.update({'Global.pretrained_model': weight_path})
        if input_path is not None:
            config.update({'Global.infer_img': input_path})
        if save_dir is not None:
            config.update({
                'Global.save_res_path': os.path.join(save_dir, 'res.txt')
            })
        config_file_path = self.config_file_path
        config.dump(config_file_path)

        self.runner.predict(config_file_path, device)

    def export(self, weight_path=None, save_dir=None, input_shape=None):
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.arch_info['config_path']
        config = self.config_cls.build_from_file(config_file_path)
        if weight_path is not None:
            config.update({'Global.pretrained_model': weight_path})
        if save_dir is not None:
            config.update({'Global.save_inference_dir': save_dir})
        config_file_path = self.config_file_path
        config.dump(config_file_path)

        self.runner.export(config_file_path, None)

    def infer(self, model_dir, device=None, input_path=None, save_dir=None):
        model_dir = abspath(model_dir)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        config_file_path = self.arch_info['config_path']
        config = self.config_cls.build_from_file(config_file_path)
        model_type = config.model_type(self.name)
        # Parse CLI arguments
        cli_args = []
        cli_args.append(CLIArgument('--rec_model_dir', model_dir))
        cli_args.append(CLIArgument('--rec_algorithm', model_type))
        if input_path is not None:
            cli_args.append(CLIArgument('--image_dir', input_path))

        self.runner.infer(cli_args, device)

    def compression(self,
                    dataset,
                    batch_size=None,
                    learning_rate=None,
                    epochs_iters=None,
                    device=None,
                    weight_path=None,
                    save_dir=None):
        dataset = abspath(dataset)
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.arch_info['config_path']
        config = self.config_cls.build_from_file(config_file_path)
        config._update_dataset_config(dataset)
        config._update_device_config(device)

        if batch_size is not None:
            config._update_batch_size_config(batch_size)
        if learning_rate is not None:
            config._update_lr_config(learning_rate)
        if epochs_iters is not None:
            config.update({'Global.epoch_num': epochs_iters})
        if weight_path is not None:
            config.update({'Global.pretrained_model': weight_path})
        if save_dir is not None:
            config.update({'Global.save_model_dir': save_dir})

        config_file_path = self.config_file_path
        config.dump(config_file_path)

        self.runner.compression(config_file_path, device)
