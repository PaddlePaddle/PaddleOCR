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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
from PIL import Image, ImageDraw, ImageFont
import math


def draw_e2e_res_for_chinese(image,
                             boxes,
                             txts,
                             config,
                             img_name,
                             font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        box = np.array(box)
        box = [tuple(x) for x in box]
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(box, outline=color)
        font = ImageFont.truetype(font_path, 15, encoding="utf-8")
        draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))

    save_e2e_path = os.path.dirname(config['Global'][
        'save_res_path']) + "/e2e_results/"
    if not os.path.exists(save_e2e_path):
        os.makedirs(save_e2e_path)
    save_path = os.path.join(save_e2e_path, os.path.basename(img_name))
    cv2.imwrite(save_path, np.array(img_show)[:, :, ::-1])
    logger.info("The e2e Image saved in {}".format(save_path))


def draw_e2e_res(dt_boxes, strs, config, img, img_name):
    if len(dt_boxes) > 0:
        src_im = img
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
        save_det_path = os.path.dirname(config['Global'][
            'save_res_path']) + "/e2e_results/"
        if not os.path.exists(save_det_path):
            os.makedirs(save_det_path)
        save_path = os.path.join(save_det_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        logger.info("The e2e Image saved in {}".format(save_path))


def main():
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])

    load_model(config, model)

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    with open(save_res_path, "wb") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds, shape_list)
            points, strs = post_result['points'], post_result['texts']
            # write result
            dt_boxes_json = []
            for poly, str in zip(points, strs):
                tmp_json = {"transcription": str}
                tmp_json['points'] = poly.tolist()
                dt_boxes_json.append(tmp_json)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())
            src_img = cv2.imread(file)
            if global_config['infer_visual_type'] == 'EN':
                draw_e2e_res(points, strs, config, src_img, file)
            elif global_config['infer_visual_type'] == 'CN':
                src_img = Image.fromarray(
                    cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
                draw_e2e_res_for_chinese(
                    src_img,
                    points,
                    strs,
                    config,
                    file,
                    font_path="./doc/fonts/simfang.ttf")

    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
