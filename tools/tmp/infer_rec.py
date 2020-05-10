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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing
import numpy as np


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid

from ppocr.utils.utility import load_config, merge_config
from ppocr.data.rec.reader_main import test_reader

from ppocr.utils.utility import ArgsParser
from ppocr.utils.character import CharacterOps, cal_predicts_accuracy
from ppocr.utils.check import check_gpu
from ppocr.utils.utility import create_module

from ppocr.utils.utility import initial_logger
logger = initial_logger()


def main():
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    char_ops = CharacterOps(config['Global'])
    config['Global']['char_num'] = char_ops.get_char_num()

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    rec_model = create_module(config['Architecture']['function'])(params=config)

    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            eval_outputs = rec_model(mode="test")
            eval_fetch_list = [v.name for v in eval_outputs]
    eval_prog = eval_prog.clone(for_test=True)
    exe.run(startup_prog)

    pretrain_weights = config['Global']['pretrain_weights']
    if pretrain_weights is not None:
        fluid.load(eval_prog, pretrain_weights)

    test_img_path = config['test_img_path']
    image_shape = config['Global']['image_shape']
    blobs = test_reader(image_shape, test_img_path)
    predict = exe.run(program=eval_prog,
                      feed={"image": blobs},
                      fetch_list=eval_fetch_list,
                      return_numpy=False)
    preds = np.array(predict[0])
    if preds.shape[1] == 1:
        preds = preds.reshape(-1)
        preds_lod = predict[0].lod()[0]
        preds_text = char_ops.decode(preds)
    else:
        end_pos = np.where(preds[0, :] == 1)[0]
        if len(end_pos) <= 1:
            preds_text = preds[0, 1:]
        else:
            preds_text = preds[0, 1:end_pos[1]]
        preds_text = preds_text.reshape(-1)
        preds_text = char_ops.decode(preds_text)

    fluid.io.save_inference_model(
        "./output/",
        feeded_var_names=['image'],
        target_vars=eval_outputs,
        executor=exe,
        main_program=eval_prog,
        model_filename="model",
        params_filename="params")
    print(preds)
    print(preds_text)


if __name__ == '__main__':
    parser = ArgsParser()
    FLAGS = parser.parse_args()
    main()
