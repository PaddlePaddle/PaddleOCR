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
import os, sys
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
import cv2
import numpy as np


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    #params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)

    #params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_max_side_len", type=float, default=960)

    #DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)

    #EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    #params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    return parser.parse_args()


def get_image_file_list(image_dir):
    image_file_list = []
    if image_dir is None:
        return image_file_list
    if os.path.isfile(image_dir):
        image_file_list = [image_dir]
    elif os.path.isdir(image_dir):
        for single_file in os.listdir(image_dir):
            image_file_list.append(os.path.join(image_dir, single_file))
    return image_file_list


def create_predictor(args, mode):
    if mode == "det":
        model_dir = args.det_model_dir
    else:
        model_dir = args.rec_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/model"
    params_file_path = model_dir + "/params"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        logger.info("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = AnalysisConfig(model_file_path, params_file_path)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()

    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)
    #     if args.use_tensorrt:
    #         config.enable_tensorrt_engine(
    #             precision_mode=AnalysisConfig.Precision.Half
    #             if args.use_fp16 else AnalysisConfig.Precision.Float32,
    #             max_batch_size=args.batch_size)

    # config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_paddle_predictor(config)
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_tensor(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    img_name_pure = img_path.split("/")[-1]
    cv2.imwrite("./output/%s" % img_name_pure, src_im)


if __name__ == '__main__':
    args = parse_args()
    args.use_gpu = False
    root_path = "/Users/liuweiwei06/Desktop/TEST_CODES/icode/baidu/personal-code/PaddleOCR/"
    args.det_model_dir = root_path + "test_models/public_v1/ch_det_mv3_db"

    predictor, input_tensor, output_tensors = create_predictor(args, mode='det')
    print("det input", predictor.get_input_names())
    print("det output", predictor.get_output_names())
    # print(predictor.program(), file=open("det_program.txt", 'w'))
    outputs = []
    for output_tensor in output_tensors:
        output = output_tensor.copy_to_cpu()
        outputs.append(output)

    args.rec_model_dir = root_path + "test_models/public_v1/ch_rec_mv3_crnn/"
    rec_predictor, input_tensor, output_tensors = create_predictor(
        args, mode='rec')
    print("rec input", rec_predictor.get_input_names())
    print("rec output", rec_predictor.get_output_names())
