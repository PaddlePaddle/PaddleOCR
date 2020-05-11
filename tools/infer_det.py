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
import program
from ppocr.utils.save_load import init_model
from ppocr.data.reader_main import reader_main
import cv2

from ppocr.utils.utility import initial_logger
logger = initial_logger()


def draw_det_res(dt_boxes, config, img_name, ino):
    if len(dt_boxes) > 0:
        img_set_path = config['TestReader']['img_set_dir']
        img_path = img_set_path + img_name
        import cv2
        src_im = cv2.imread(img_path)
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        save_det_path = os.path.basename(config['Global'][
            'save_res_path']) + "/det_results/"
        if not os.path.exists(save_det_path):
            os.makedirs(save_det_path)
        save_path = os.path.join(save_det_path, "det_{}.jpg".format(img_name))
        cv2.imwrite(save_path, src_im)
        logger.info("The detected Image saved in {}".format(save_path))


def simple_reader(img_file, config):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']
    if os.path.isfile(img_file) and img_file.split('.')[-1] in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            if single_file.split('.')[-1] in img_end:
                imgs_lists.append(os.path.join(img_file, single_file))
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))

    batch_size = config['Global']['test_batch_size_per_card']
    global_params = config['Global']
    params = deepcopy(config['TestReader'])
    params.update(global_params)
    reader_function = params['process_function']
    process_function = create_module(reader_function)(params)

    def batch_iter_reader():
        batch_outs = []
        for img_path in imgs_lists:
            img = cv2.imread(img_path)
            if img.shape[-1] == 1 or len(list(img.shape)) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img is None:
                logger.info("load image error:" + img_path)
                continue
            outs = process_function(img)
            outs.append(os.path.basename(img_path))
            print(outs[0].shape, outs[2])
            batch_outs.append(outs)
            if len(batch_outs) == batch_size:
                yield batch_outs
                batch_outs = []
        if len(batch_outs) != 0:
            yield batch_outs

    return batch_iter_reader


def main():
    config = program.load_config(FLAGS.config)
    program.merge_config(FLAGS.opt)
    print(config)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    program.check_gpu(use_gpu)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    det_model = create_module(config['Architecture']['function'])(params=config)

    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            _, eval_outputs = det_model(mode="test")
            fetch_name_list = list(eval_outputs.keys())
            eval_fetch_list = [eval_outputs[v].name for v in fetch_name_list]

    eval_prog = eval_prog.clone(for_test=True)
    exe.run(startup_prog)

    # load checkpoints
    checkpoints = config['Global'].get('checkpoints')
    if checkpoints:
        path = checkpoints
        fluid.load(eval_prog, path, exe)
        logger.info("Finish initing model from {}".format(path))
    else:
        raise Exception("{} not exists!".format(checkpoints))

    save_res_path = config['Global']['save_res_path']
    with open(save_res_path, "wb") as fout:
        # test_reader = reader_main(config=config, mode='test')
        single_img_path = config['TestReader']['single_img_path']
        test_reader = simple_reader(img_file=single_img_path, config=config)
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
            dt_boxes_list = postprocess({"maps": outs[0]}, ratio_list)
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
                draw_det_res(dt_boxes, config, img_name, ino)

    logger.info("success!")


if __name__ == '__main__':
    parser = program.ArgsParser()
    FLAGS = parser.parse_args()
    main()
