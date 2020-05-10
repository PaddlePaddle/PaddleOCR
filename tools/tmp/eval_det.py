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
import sys
import time
import numpy as np
from copy import deepcopy
import json

# from paddle.fluid.contrib.model_stat import summary


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
from ppocr.utils.utility import create_module
from ppocr.utils.utility import load_config, merge_config
import ppocr.data.det.reader_main as reader
from ppocr.utils.utility import ArgsParser
from ppocr.utils.check import check_gpu
from ppocr.utils.checkpoint import load_pretrain, load_checkpoint, save, save_model

from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.utils.eval_utils import eval_det_run


def draw_det_res(dt_boxes, config, img_name, ino):
    if len(dt_boxes) > 0:
        img_set_path = config['TestReader']['img_set_dir']
        img_path = img_set_path + img_name
        import cv2
        src_im = cv2.imread(img_path)
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.imwrite("tmp%d.jpg" % ino, src_im)


def main():
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    print(config)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    det_model = create_module(config['Architecture']['function'])(params=config)

    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            eval_loader, eval_outputs = det_model(mode="test")
            eval_fetch_list = [v.name for v in eval_outputs]
    eval_prog = eval_prog.clone(for_test=True)
    exe.run(startup_prog)

    pretrain_weights = config['Global']['pretrain_weights']
    if pretrain_weights is not None:
        load_pretrain(exe, eval_prog, pretrain_weights)
#         fluid.load(eval_prog, pretrain_weights)
#         def if_exist(var):
#             return os.path.exists(os.path.join(pretrain_weights, var.name))
#         fluid.io.load_vars(exe, pretrain_weights, predicate=if_exist, main_program=eval_prog)
    else:
        logger.info("Not find pretrain_weights:%s" % pretrain_weights)
        sys.exit(0)

#     fluid.io.save_inference_model("./output/", feeded_var_names=['image'],
#         target_vars=eval_outputs, executor=exe, main_program=eval_prog,
#         model_filename="model", params_filename="params")
#     sys.exit(-1)

    metrics = eval_det_run(exe, eval_prog, eval_fetch_list, config, "test")
    logger.info("metrics:{}".format(metrics))
    logger.info("success!")


def test_reader():
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    print(config)
    tmp_reader = reader.test_reader(config=config)
    count = 0
    print_count = 0
    import time
    starttime = time.time()
    for data in tmp_reader():
        count += len(data)
        print_count += 1
        if print_count % 10 == 0:
            batch_time = (time.time() - starttime) / print_count
            print("reader:", count, len(data), batch_time)
    print("finish reader:", count)
    print("success")


if __name__ == '__main__':
    parser = ArgsParser()
    FLAGS = parser.parse_args()
    main()
#     test_reader()
