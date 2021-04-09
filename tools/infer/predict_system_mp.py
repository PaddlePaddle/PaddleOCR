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
import os
import sys
import paddle
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import tools.infer.utility as utility

if __name__ == "__main__":
    args = utility.parse_args()

    p_list = []

    inference_dir = "inference_results"
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    total_process_num = args.total_process_num
    for process_id in range(total_process_num):
        cmd = [sys.executable, "-u", "tools/infer/predict_system.py"
               ] + sys.argv[1:] + ["--process_id={}".format(process_id)]

        with open("{}/results.{}".format(inference_dir, process_id),
                  "w") as fin:
            p = subprocess.Popen(cmd, stdout=fin, stderr=fin)
            # if you want to print results in the screen, you can use the following command
            # p = subprocess.Popen(cmd, stdout=fin, stderr=sys.stdout)
            p_list.append(p)
    for p in p_list:
        p.wait()
