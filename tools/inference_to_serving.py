# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
from paddle_serving_client.io import inference_model_to_serving
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--server_dir", type=str, default="serving_server_dir")
    parser.add_argument("--client_dir", type=str, default="serving_client_dir")
    return parser.parse_args()

args = parse_args()
inference_model_dir = args.model_dir
serving_client_dir = args.server_dir
serving_server_dir = args.client_dir
feed_var_names, fetch_var_names = inference_model_to_serving(
        inference_model_dir, serving_client_dir, serving_server_dir, model_filename="model", params_filename="params")
