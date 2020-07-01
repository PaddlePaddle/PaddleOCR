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

import numpy as np
import os
import sys
__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.append(os.path.join(__dir__, '..'))


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

import tools.program as program
from paddle import fluid
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.data.reader_main import reader_main
from ppocr.utils.save_load import init_model
from ppocr.utils.character import CharacterOps
from ppocr.utils.utility import create_module
from ppocr.utils.utility import get_image_file_list


def main():
    config = program.load_config(FLAGS.config)
    program.merge_config(FLAGS.opt)
    logger.info(config)
    char_ops = CharacterOps(config['Global'])
    loss_type = config['Global']['loss_type']
    config['Global']['char_ops'] = char_ops

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    #     check_gpu(use_gpu)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    rec_model = create_module(config['Architecture']['function'])(params=config)

    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            _, outputs = rec_model(mode="test")
            fetch_name_list = list(outputs.keys())
            fetch_varname_list = [outputs[v].name for v in fetch_name_list]
    eval_prog = eval_prog.clone(for_test=True)
    exe.run(startup_prog)

    init_model(config, eval_prog, exe)

    blobs = reader_main(config, 'test')()
    infer_img = config['Global']['infer_img']
    infer_list = get_image_file_list(infer_img)
    max_img_num = len(infer_list)
    if len(infer_list) == 0:
        logger.info("Can not find img in infer_img dir.")
    for i in range(max_img_num):
        print("infer_img:%s" % infer_list[i])
        img = next(blobs)
        predict = exe.run(program=eval_prog,
                          feed={"image": img},
                          fetch_list=fetch_varname_list,
                          return_numpy=False)
        if loss_type == "ctc":
            preds = np.array(predict[0])
            preds = preds.reshape(-1)
            preds_lod = predict[0].lod()[0]
            preds_text = char_ops.decode(preds)
            probs = np.array(predict[1])
            ind = np.argmax(probs, axis=1)
            blank = probs.shape[1]
            valid_ind = np.where(ind != (blank - 1))[0]
            if not valid_ind:
                continue
            score = np.mean(probs[valid_ind, ind[valid_ind]])
        elif loss_type == "attention":
            preds = np.array(predict[0])
            probs = np.array(predict[1])
            end_pos = np.where(preds[0, :] == 1)[0]
            if len(end_pos) <= 1:
                preds = preds[0, 1:]
                score = np.mean(probs[0, 1:])
            else:
                preds = preds[0, 1:end_pos[1]]
                score = np.mean(probs[0, 1:end_pos[1]])
            preds = preds.reshape(-1)
            preds_text = char_ops.decode(preds)

        print("\t index:", preds)
        print("\t word :", preds_text)
        print("\t score :", score)

    # save for inference model
    target_var = []
    for key, values in outputs.items():
        target_var.append(values)

    fluid.io.save_inference_model(
        "./output/",
        feeded_var_names=['image'],
        target_vars=target_var,
        executor=exe,
        main_program=eval_prog,
        model_filename="model",
        params_filename="params")


if __name__ == '__main__':
    parser = program.ArgsParser()
    FLAGS = parser.parse_args()
    main()
