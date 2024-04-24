# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

import cv2
import pickle
import paddle
from tqdm import tqdm
from ppstructure.table.table_metric import TEDS
from ppstructure.table.predict_table import TableSystem
from ppstructure.utility import init_args
from ppocr.utils.logging import get_logger

logger = get_logger()


def parse_args():
    parser = init_args()
    parser.add_argument("--gt_path", type=str)
    return parser.parse_args()


def load_txt(txt_path):
    pred_html_dict = {}
    if not os.path.exists(txt_path):
        return pred_html_dict
    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            img_name, pred_html = line
            pred_html_dict[img_name] = pred_html
    return pred_html_dict


def load_result(path):
    data = {}
    if os.path.exists(path):
        data = pickle.load(open(path, "rb"))
    return data


def save_result(path, data):
    old_data = load_result(path)
    old_data.update(data)
    with open(path, "wb") as f:
        pickle.dump(old_data, f)


def main(gt_path, img_root, args):
    os.makedirs(args.output, exist_ok=True)
    # init TableSystem
    text_sys = TableSystem(args)
    # load gt and preds html result
    gt_html_dict = load_txt(gt_path)

    ocr_result = load_result(os.path.join(args.output, "ocr.pickle"))
    structure_result = load_result(os.path.join(args.output, "structure.pickle"))

    pred_htmls = []
    gt_htmls = []
    for img_name, gt_html in tqdm(gt_html_dict.items()):
        img = cv2.imread(os.path.join(img_root, img_name))
        # run ocr and save result
        if img_name not in ocr_result:
            dt_boxes, rec_res, _, _ = text_sys._ocr(img)
            ocr_result[img_name] = [dt_boxes, rec_res]
            save_result(os.path.join(args.output, "ocr.pickle"), ocr_result)
        # run structure and save result
        if img_name not in structure_result:
            structure_res, _ = text_sys._structure(img)
            structure_result[img_name] = structure_res
            save_result(os.path.join(args.output, "structure.pickle"), structure_result)
        dt_boxes, rec_res = ocr_result[img_name]
        structure_res = structure_result[img_name]
        # match ocr and structure
        pred_html = text_sys.match(structure_res, dt_boxes, rec_res)

        pred_htmls.append(pred_html)
        gt_htmls.append(gt_html)

    # compute teds
    teds = TEDS(n_jobs=16)
    scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
    logger.info("teds: {}".format(sum(scores) / len(scores)))


if __name__ == "__main__":
    args = parse_args()
    main(args.gt_path, args.image_dir, args)
