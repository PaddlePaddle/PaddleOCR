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

from ..base import BaseRunner


class TextRecRunner(BaseRunner):
    def train(self, config_path, cli_args, device):
        python, _ = self.distributed(device)
        cmd = f"{python} tools/train.py -c {config_path}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def predict(self, config_path, cli_args, device):
        self.distributed(device)
        cmd = f"{self.python} tools/infer_rec.py -c {config_path}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(self, config_path, cli_args, device):
        # `device` unused
        cmd = f"{self.python} tools/export_model.py -c {config_path}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_path, cli_args, device):
        _, device_type = self.distributed(device)
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{self.python} tools/infer/predict_rec.py --use_gpu {device_type=='gpu'} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def compression(self, config_path, cli_args, device, save_dir):
        # Step 1: Train model
        python, _ = self.distributed(device)
        cmd = f"{python} deploy/slim/quantization/quant.py -c {config_path}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

        # Step 2: Export model
        cli_args = [
            ' -o',
            f'Global.checkpoints={save_dir}/latest',
            f'Global.save_inference_dir={save_dir}',
        ]
        cli_args = ' '.join(cli_args)
        cmd = f"{python} deploy/slim/quantization/export_model.py -c {config_path} {cli_args}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)
