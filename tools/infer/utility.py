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
import os
import sys
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import math
from paddle import inference
import time
from ppocr.utils.logging import get_logger

logger = get_logger()


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")
    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_dir", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt")
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_polygon", type=bool, default=True)
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)

    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--show_log", type=str2bool, default=True)

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


class Times(object):
    def __init__(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += self.et - self.st
        else:
            self.time = self.et - self.st

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class Timer(Times):
    def __init__(self):
        super(Timer, self).__init__()
        self.total_time = Times()
        self.preprocess_time = Times()
        self.inference_time = Times()
        self.postprocess_time = Times()
        self.img_num = 0

    def info(self, average=False):
        logger.info("----------------------- Perf info -----------------------")
        logger.info("total_time: {}, img_num: {}".format(self.total_time.value(
        ), self.img_num))
        preprocess_time = round(self.preprocess_time.value() / self.img_num,
                                4) if average else self.preprocess_time.value()
        postprocess_time = round(
            self.postprocess_time.value() / self.img_num,
            4) if average else self.postprocess_time.value()
        inference_time = round(self.inference_time.value() / self.img_num,
                               4) if average else self.inference_time.value()

        average_latency = self.total_time.value() / self.img_num
        logger.info("average_latency(ms): {:.2f}, QPS: {:2f}".format(
            average_latency * 1000, 1 / average_latency))
        logger.info(
            "preprocess_latency(ms): {:.2f}, inference_latency(ms): {:.2f}, postprocess_latency(ms): {:.2f}".
            format(preprocess_time * 1000, inference_time * 1000,
                   postprocess_time * 1000))

    def report(self, average=False):
        dic = {}
        dic['preprocess_time'] = round(
            self.preprocess_time.value() / self.img_num,
            4) if average else self.preprocess_time.value()
        dic['postprocess_time'] = round(
            self.postprocess_time.value() / self.img_num,
            4) if average else self.postprocess_time.value()
        dic['inference_time'] = round(
            self.inference_time.value() / self.img_num,
            4) if average else self.inference_time.value()
        dic['img_num'] = self.img_num
        dic['total_time'] = round(self.total_time.value(), 4)
        return dic


def create_predictor(args, mode, logger):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == 'cls':
        model_dir = args.cls_model_dir
    elif mode == 'rec':
        model_dir = args.rec_model_dir
    elif mode == 'structure':
        model_dir = args.structure_model_dir
    else:
        model_dir = args.e2e_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/inference.pdmodel"
    params_file_path = model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        logger.info("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = inference.Config(model_file_path, params_file_path)

    if hasattr(args, 'precision'):
        if args.precision == "fp16" and args.use_tensorrt:
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32
    else:
        precision = inference.PrecisionType.Float32

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=inference.PrecisionType.Float32,
                max_batch_size=args.max_batch_size,
                min_subgraph_size=3)  # skip the minmum trt subgraph
        if mode == "det":
            min_input_shape = {
                "x": [1, 3, 50, 50],
                "conv2d_92.tmp_0": [1, 96, 20, 20],
                "conv2d_91.tmp_0": [1, 96, 10, 10],
                "conv2d_59.tmp_0": [1, 96, 20, 20],
                "nearest_interp_v2_1.tmp_0": [1, 96, 10, 10],
                "nearest_interp_v2_2.tmp_0": [1, 96, 20, 20],
                "nearest_interp_v2_3.tmp_0": [1, 24, 20, 20],
                "nearest_interp_v2_4.tmp_0": [1, 24, 20, 20],
                "nearest_interp_v2_5.tmp_0": [1, 24, 20, 20],
                "elementwise_add_7": [1, 56, 2, 2],
                "nearest_interp_v2_0.tmp_0": [1, 96, 2, 2]
            }
            max_input_shape = {
                "x": [1, 3, 2000, 2000],
                "conv2d_92.tmp_0": [1, 96, 400, 400],
                "conv2d_91.tmp_0": [1, 96, 200, 200],
                "conv2d_59.tmp_0": [1, 96, 400, 400],
                "nearest_interp_v2_1.tmp_0": [1, 96, 200, 200],
                "nearest_interp_v2_2.tmp_0": [1, 96, 400, 400],
                "nearest_interp_v2_3.tmp_0": [1, 24, 400, 400],
                "nearest_interp_v2_4.tmp_0": [1, 24, 400, 400],
                "nearest_interp_v2_5.tmp_0": [1, 24, 400, 400],
                "elementwise_add_7": [1, 56, 400, 400],
                "nearest_interp_v2_0.tmp_0": [1, 96, 400, 400]
            }
            opt_input_shape = {
                "x": [1, 3, 640, 640],
                "conv2d_92.tmp_0": [1, 96, 160, 160],
                "conv2d_91.tmp_0": [1, 96, 80, 80],
                "conv2d_59.tmp_0": [1, 96, 160, 160],
                "nearest_interp_v2_1.tmp_0": [1, 96, 80, 80],
                "nearest_interp_v2_2.tmp_0": [1, 96, 160, 160],
                "nearest_interp_v2_3.tmp_0": [1, 24, 160, 160],
                "nearest_interp_v2_4.tmp_0": [1, 24, 160, 160],
                "nearest_interp_v2_5.tmp_0": [1, 24, 160, 160],
                "elementwise_add_7": [1, 56, 40, 40],
                "nearest_interp_v2_0.tmp_0": [1, 96, 40, 40]
            }
        elif mode == "rec":
            min_input_shape = {"x": [args.rec_batch_num, 3, 32, 10]}
            max_input_shape = {"x": [args.rec_batch_num, 3, 32, 2000]}
            opt_input_shape = {"x": [args.rec_batch_num, 3, 32, 320]}
        elif mode == "cls":
            min_input_shape = {"x": [args.rec_batch_num, 3, 48, 10]}
            max_input_shape = {"x": [args.rec_batch_num, 3, 48, 2000]}
            opt_input_shape = {"x": [args.rec_batch_num, 3, 48, 320]}
        else:
            min_input_shape = {"x": [1, 3, 10, 10]}
            max_input_shape = {"x": [1, 3, 1000, 1000]}
            opt_input_shape = {"x": [1, 3, 500, 500]}
        config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                          opt_input_shape)

    else:
        config.disable_gpu()
        if hasattr(args, "cpu_threads"):
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        else:
            # default cpu threads as 10
            config.set_cpu_math_library_num_threads(10)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)
    if mode == 'structure':
        config.switch_ir_optim(False)
    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors, config


def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1)
    return src_im


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


def get_current_memory_mb(gpu_id=None):
    """
    It is used to Obtain the memory usage of the CPU and GPU during the running of the program.
    And this function Current program is time-consuming.
    """
    import pynvml
    import psutil
    import GPUtil
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    gpu_percent = 0
    if gpu_id is not None:
        GPUs = GPUtil.getGPUs()
        gpu_load = GPUs[gpu_id].load
        gpu_percent = gpu_load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return round(cpu_mem, 4), round(gpu_mem, 4), round(gpu_percent, 4)


if __name__ == '__main__':
    pass
