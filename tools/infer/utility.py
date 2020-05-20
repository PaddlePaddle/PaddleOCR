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
import json
from PIL import Image, ImageDraw, ImageFont


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
    parser.add_argument("--rec_batch_num", type=int, default=30)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    return parser.parse_args()


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

    # use zero copy
    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
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


def draw_text_det_res(dt_boxes, img_path, return_img=True):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return im


def draw_ocr(image, boxes, txts, scores, draw_txt=True, drop_score=0.5):
    from PIL import Image, ImageDraw, ImageFont

    img = image.copy()
    draw = ImageDraw.Draw(img)
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        draw.line([(box[0][0], box[0][1]), (box[1][0], box[1][1])], fill='red')
        draw.line([(box[1][0], box[1][1]), (box[2][0], box[2][1])], fill='red')
        draw.line([(box[2][0], box[2][1]), (box[3][0], box[3][1])], fill='red')
        draw.line([(box[3][0], box[3][1]), (box[0][0], box[0][1])], fill='red')
        draw.line(
            [(box[0][0] - 1, box[0][1] + 1), (box[1][0] - 1, box[1][1] + 1)],
            fill='red')
        draw.line(
            [(box[1][0] - 1, box[1][1] + 1), (box[2][0] - 1, box[2][1] + 1)],
            fill='red')
        draw.line(
            [(box[2][0] - 1, box[2][1] + 1), (box[3][0] - 1, box[3][1] + 1)],
            fill='red')
        draw.line(
            [(box[3][0] - 1, box[3][1] + 1), (box[0][0] - 1, box[0][1] + 1)],
            fill='red')

    if draw_txt:
        txt_color = (0, 0, 0)
        img = np.array(resize_img(img))
        _h = img.shape[0]
        blank_img = np.ones(shape=[_h, 600], dtype=np.int8) * 255
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)

        font_size = 20
        gap = 20
        title = "index           text           score"
        font = ImageFont.truetype(
            "./doc/simfang.ttf", font_size, encoding="utf-8")

        draw_txt.text((20, 0), title, txt_color, font=font)
        count = 0
        for idx, txt in enumerate(txts):
            if scores[idx] < drop_score:
                continue
            font = ImageFont.truetype(
                "./doc/simfang.ttf", font_size, encoding="utf-8")
            new_txt = str(count) + ':  ' + txt + '    ' + '%.3f' % (
                scores[count])
            draw_txt.text(
                (20, gap * (count + 1)), new_txt, txt_color, font=font)
            count += 1
        img = np.concatenate([np.array(img), np.array(blank_img)], axis=1)
    return img


if __name__ == '__main__':
    test_img = "./doc/test_v2"
    predict_txt = "./doc/predict.txt"
    f = open(predict_txt, 'r')
    data = f.readlines()
    img_path, anno = data[0].strip().split('\t')
    img_name = os.path.basename(img_path)
    img_path = os.path.join(test_img, img_name)
    image = Image.open(img_path)

    data = json.loads(anno)
    boxes, txts, scores = [], [], []
    for dic in data:
        boxes.append(dic['points'])
        txts.append(dic['transcription'])
        scores.append(round(dic['scores'], 3))

    new_img = draw_ocr(image, boxes, txts, scores, draw_txt=True)

    cv2.imwrite(img_name, new_img)
