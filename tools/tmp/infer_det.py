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
            eval_outputs = det_model(mode="test")
            eval_fetch_list = [v.name for v in eval_outputs]
    eval_prog = eval_prog.clone(for_test=True)
    exe.run(startup_prog)

    pretrain_weights = config['Global']['pretrain_weights']
    if pretrain_weights is not None:
        fluid.load(eval_prog, pretrain_weights)
    else:
        logger.info("Not find pretrain_weights:%s" % pretrain_weights)
        sys.exit(0)

    save_res_path = config['Global']['save_res_path']
    with open(save_res_path, "wb") as fout:
        test_reader = reader.test_reader(config=config)
        tackling_num = 0
        for data in test_reader():
            img_num = len(data)
            tackling_num = tackling_num + img_num
            logger.info("tackling_num:%d", tackling_num)
            img_list = []
            ratio_list = []
            img_name_list = []
            for ino in range(img_num):
                img_list.append(data[ino][0])
                ratio_list.append(data[ino][1])
                img_name_list.append(data[ino][2])
            img_list = np.concatenate(img_list, axis=0)
            outs = exe.run(eval_prog,\
                feed={'image': img_list},\
                fetch_list=eval_fetch_list)

            global_params = config['Global']
            postprocess_params = deepcopy(config["PostProcess"])
            postprocess_params.update(global_params)
            postprocess = create_module(postprocess_params['function'])\
                (params=postprocess_params)
            dt_boxes_list = postprocess(outs, ratio_list)
            for ino in range(img_num):
                dt_boxes = dt_boxes_list[ino]
                img_name = img_name_list[ino]
                dt_boxes_json = []
                for box in dt_boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json['points'] = box.tolist()
                    dt_boxes_json.append(tmp_json)
                otstr = img_name + "\t" + json.dumps(dt_boxes_json) + "\n"
                fout.write(otstr.encode())
                #draw_det_res(dt_boxes, config, img_name, ino)
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
