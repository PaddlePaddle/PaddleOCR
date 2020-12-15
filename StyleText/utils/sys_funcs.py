# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import errno
import paddle


def get_check_global_params(mode):
    check_params = [
        'use_gpu', 'max_text_length', 'image_shape', 'image_shape',
        'character_type', 'loss_type'
    ]
    if mode == "train_eval":
        check_params = check_params + [
            'train_batch_size_per_card', 'test_batch_size_per_card'
        ]
    elif mode == "test":
        check_params = check_params + ['test_batch_size_per_card']
    return check_params


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"
    if use_gpu:
        try:
            if not paddle.is_compiled_with_cuda():
                print(err)
                sys.exit(1)
        except:
            print("Fail to check gpu state.")
            sys.exit(1)


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))
