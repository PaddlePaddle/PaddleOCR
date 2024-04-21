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
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from ppocr.utils.logging import get_logger

logger = get_logger()

import cv2
import numpy as np
import time
from PIL import Image
from ppocr.utils.utility import get_image_file_list
from tools.infer.utility import draw_ocr, draw_boxes, str2bool
from ppstructure.utility import draw_structure_result
from ppstructure.predict_system import to_excel

import requests
import json
import base64


def cv2_to_base64(image):
    return base64.b64encode(image).decode("utf8")


def draw_server_result(image_file, res):
    img = cv2.imread(image_file)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(res) == 0:
        return np.array(image)
    keys = res[0].keys()
    if "text_region" not in keys:  # for ocr_rec, draw function is invalid
        logger.info("draw function is invalid for ocr_rec!")
        return None
    elif "text" not in keys:  # for ocr_det
        logger.info("draw text boxes only!")
        boxes = []
        for dno in range(len(res)):
            boxes.append(res[dno]["text_region"])
        boxes = np.array(boxes)
        draw_img = draw_boxes(image, boxes)
        return draw_img
    else:  # for ocr_system
        logger.info("draw boxes and texts!")
        boxes = []
        texts = []
        scores = []
        for dno in range(len(res)):
            boxes.append(res[dno]["text_region"])
            texts.append(res[dno]["text"])
            scores.append(res[dno]["confidence"])
        boxes = np.array(boxes)
        scores = np.array(scores)
        draw_img = draw_ocr(image, boxes, texts, scores, draw_txt=True, drop_score=0.5)
        return draw_img


def save_structure_res(res, save_folder, image_file):
    img = cv2.imread(image_file)
    excel_save_folder = os.path.join(save_folder, os.path.basename(image_file))
    os.makedirs(excel_save_folder, exist_ok=True)
    # save res
    with open(os.path.join(excel_save_folder, "res.txt"), "w", encoding="utf8") as f:
        for region in res:
            if region["type"] == "Table":
                excel_path = os.path.join(
                    excel_save_folder, "{}.xlsx".format(region["bbox"])
                )
                to_excel(region["res"], excel_path)
            elif region["type"] == "Figure":
                x1, y1, x2, y2 = region["bbox"]
                print(region["bbox"])
                roi_img = img[y1:y2, x1:x2, :]
                img_path = os.path.join(
                    excel_save_folder, "{}.jpg".format(region["bbox"])
                )
                cv2.imwrite(img_path, roi_img)
            else:
                for text_result in region["res"]:
                    f.write("{}\n".format(json.dumps(text_result)))


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    is_visualize = False
    headers = {"Content-type": "application/json"}
    cnt = 0
    total_time = 0
    for image_file in image_file_list:
        img = open(image_file, "rb").read()
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        img_name = os.path.basename(image_file)
        # seed http request
        starttime = time.time()
        data = {"images": [cv2_to_base64(img)]}
        r = requests.post(url=args.server_url, headers=headers, data=json.dumps(data))
        elapse = time.time() - starttime
        total_time += elapse
        logger.info("Predict time of %s: %.3fs" % (image_file, elapse))
        res = r.json()["results"][0]
        logger.info(res)

        if args.visualize:
            draw_img = None
            if "structure_table" in args.server_url:
                to_excel(res["html"], "./{}.xlsx".format(img_name))
            elif "structure_system" in args.server_url:
                save_structure_res(res["regions"], args.output, image_file)
            else:
                draw_img = draw_server_result(image_file, res)
            if draw_img is not None:
                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                cv2.imwrite(
                    os.path.join(args.output, os.path.basename(image_file)),
                    draw_img[:, :, ::-1],
                )
                logger.info(
                    "The visualized image saved in {}".format(
                        os.path.join(args.output, os.path.basename(image_file))
                    )
                )
        cnt += 1
        if cnt % 100 == 0:
            logger.info("{} processed".format(cnt))
    logger.info("avg time cost: {}".format(float(total_time) / cnt))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="args for hub serving")
    parser.add_argument("--server_url", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--output", type=str, default="./hubserving_result")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
