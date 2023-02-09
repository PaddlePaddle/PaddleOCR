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

from ..base.utils.path import get_cache_dir
from ..base.utils.arg import CLIArgument
from ..base.utils.misc import abspath

from ..text_rec_task import TextRecModel
from .config import TextDetConfig


class TextDetModel(TextRecModel):
    def update_config_cls(self):
        self.config_cls = TextDetConfig

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
        cli_args.append(CLIArgument('--det_model_dir', model_dir))
        cli_args.append(CLIArgument('--det_algorithm', model_type))
        if input_path is not None:
            cli_args.append(CLIArgument('--image_dir', input_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--draw_img_save_dir', save_dir))

        self.runner.infer(cli_args, device)