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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time

import tools.infer.utility as utility
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppstructure.utility import parse_args
from picodet_postprocess import PicoDetPostProcess

logger = get_logger()


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    from PIL import Image, ImageDraw, ImageFont
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)

    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        label, label_id = txt
        draw_left.polygon(box, fill=color)

        font_size = 10
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        tw, th = font.getsize(label)
        draw_left.rectangle(
            [(box[0][0] + 1, box[0][1] - th), (box[0][0] + tw + 1, box[0][1])],
            fill=color)
        draw_left.text(
            (box[0][0] + 1, box[0][1] - th), label, fill=(0, 0, 0), font=font)

    img_left = Image.blend(image, img_left, 0.5)
    return np.array(img_left)


class LayoutPredictor(object):
    def __init__(self, args):
        pre_process_list = [{
            'Resize': {
                'size': [800, 608]
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image']
            }
        }]
        postprocess_params = {
            'name': 'PicoDetPostProcess',
            "layout_dict_path": args.layout_dict_path,
            "score_threshold": args.score_threshold,
            "nms_threshold": args.nms_threshold,
        }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'layout', logger)

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]

        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        img = img.copy()

        preds, elapse = 0, 1
        starttime = time.time()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()

        np_score_list, np_boxes_list = [], []
        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        for out_idx in range(num_outs):
            np_score_list.append(
                self.predictor.get_output_handle(output_names[out_idx])
                .copy_to_cpu())
            np_boxes_list.append(
                self.predictor.get_output_handle(output_names[
                    out_idx + num_outs]).copy_to_cpu())
        preds = dict(boxes=np_score_list, boxes_num=np_boxes_list)

        post_preds = self.postprocess_op(ori_im, img, preds)
        elapse = time.time() - starttime
        return post_preds, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    layout_predictor = LayoutPredictor(args)
    count = 0
    total_time = 0

    repeats = 50
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        # warmup 
        for i in range(repeats):
            layout_res, _ = layout_predictor(img)

        layout_res, elapse = layout_predictor(img)

        logger.info("result: {}".format(layout_res))

        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))
    logger.info('Predicr {} images, average time: {}, count:{}'.format(
        len(image_file_list), total_time / len(image_file_list), count))


if __name__ == "__main__":
    main(parse_args())
