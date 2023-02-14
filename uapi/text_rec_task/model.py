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


class TextRecModel(BaseModel):
    def train(self,
              dataset=None,
              batch_size=None,
              learning_rate=None,
              epochs_iters=None,
              device='gpu',
              resume_path=None,
              dy2st=False,
              amp=None,
              use_vdl=True,
              save_dir=None):
        # NOTE: We must use an absolute path here, 
        # so we can run the scripts either inside or outside the repo dir.
        if dataset is not None:
            # NOTE: We must use an absolute path here, 
            # so we can run the scripts either inside or outside the repo dir.
            dataset = abspath(dataset)
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'train'))

        # Update YAML config file
        config = self.config.copy()
        config.update_dataset(dataset)
        config.update_amp(amp)
        config.update_device(device)

        if batch_size is not None:
            config.update_batch_size(batch_size)
        if learning_rate is not None:
            config.update_lr_scheduler(learning_rate)
        if epochs_iters is not None:
            config.update({'Global.epoch_num': epochs_iters})
        if resume_path is not None:
            config.update({'Global.checkpoints': resume_path})
        if dy2st:
            config.update({'Global.to_static': True})
        if use_vdl:
            config.update({'Global.use_visualdl': use_vdl})
        if save_dir is not None:
            config.update({'Global.save_model_dir': save_dir})

        config_path = self._config_path
        config.dump(config_path)
        self.runner.train(config_path, [], device)

    def predict(self, weight_path, input_path, device='gpu', save_dir=None):
        weight_path = abspath(weight_path)
        input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'predict'))

        # Update YAML config file
        config = self.config.copy()
        if weight_path is not None:
            config.update({'Global.pretrained_model': weight_path})
        if input_path is not None:
            config.update({'Global.infer_img': input_path})
        if save_dir is not None:
            config.update({
                'Global.save_res_path': os.path.join(save_dir, 'res.txt')
            })
        config_path = self._config_path
        config.dump(config_path)

        self.runner.predict(config_path, [], device)

    def export(self, weight_path, save_dir=None, input_shape=None):
        weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'export'))

        # Update YAML config file
        config = self.config.copy()
        if weight_path is not None:
            config.update({'Global.pretrained_model': weight_path})
        if save_dir is not None:
            config.update({'Global.save_inference_dir': save_dir})
        config_path = self._config_path
        config.dump(config_path)

        self.runner.export(config_path, [], None)

    def infer(self, model_dir, input_path, device='gpu', save_dir=None):
        model_dir = abspath(model_dir)
        input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'infer'))

        # Update YAML config file
        config = self.config.copy()
        model_type = config.model_type(self.name)
        config_path = self._config_path
        config.dump(config_path)
        # Parse CLI arguments
        cli_args = []
        cli_args.append(CLIArgument('--rec_model_dir', model_dir))
        cli_args.append(CLIArgument('--rec_algorithm', model_type))
        if input_path is not None:
            cli_args.append(CLIArgument('--image_dir', input_path))

        self.runner.infer(config_path, cli_args, device)

    def compression(self,
                    weight_path,
                    dataset=None,
                    batch_size=None,
                    learning_rate=None,
                    epochs_iters=None,
                    device='gpu',
                    use_vdl=True,
                    save_dir=None,
                    input_shape=None):
        weight_path = abspath(weight_path)
        if dataset is not None:
            dataset = abspath(dataset)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'compress'))

        # Update YAML config file
        config = self.config.copy()
        config.update_dataset(dataset)
        config.update_device(device)

        if batch_size is not None:
            config.update_batch_size(batch_size)
        if learning_rate is not None:
            config.update_lr_scheduler(learning_rate)
        if epochs_iters is not None:
            config.update({'Global.epoch_num': epochs_iters})
        if weight_path is not None:
            config.update({'Global.pretrained_model': weight_path})
        if use_vdl:
            config.update({'Global.use_visualdl': use_vdl})
        if save_dir is not None:
            config.update({'Global.save_model_dir': save_dir})

        config_path = self._config_path
        config.dump(config_path)

        self.runner.compression(config_path, [], device, save_dir)
