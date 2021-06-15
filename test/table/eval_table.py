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

import cv2
import json
from tqdm import tqdm
from test.table.table_metric import TEDS
from test.table.predict_table import TableSystem
from test.utility import init_args
from ppocr.utils.logging import get_logger

logger = get_logger()


def parse_args():
    parser = init_args()
    parser.add_argument("--gt_path", type=str)
    return parser.parse_args()

def main(gt_path, img_root, args):
    teds = TEDS(n_jobs=16)

    text_sys = TableSystem(args)
    jsons_gt = json.load(open(gt_path))  # gt
    pred_htmls = []
    gt_htmls = []
    for img_name in tqdm(jsons_gt):
        # read image
        img = cv2.imread(os.path.join(img_root,img_name))
        pred_html = text_sys(img)
        pred_htmls.append(pred_html)

        gt_structures, gt_bboxes, gt_contents, contents_with_block = jsons_gt[img_name]
        gt_html, gt = get_gt_html(gt_structures, contents_with_block)
        gt_htmls.append(gt_html)
    scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
    logger.info('teds:', sum(scores) / len(scores))


def get_gt_html(gt_structures, contents_with_block):
    end_html = []
    td_index = 0
    for tag in gt_structures:
        if '</td>' in tag:
            if contents_with_block[td_index] != []:
                end_html.extend(contents_with_block[td_index])
            end_html.append(tag)
            td_index += 1
        else:
            end_html.append(tag)
    return ''.join(end_html), end_html


if __name__ == '__main__':
    args = parse_args()
    main(args.gt_path,args.image_dir, args)
