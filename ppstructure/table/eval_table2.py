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
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import pickle
import paddle
from tqdm import tqdm
from ppstructure.table.table_metric import TEDS
from ppstructure.table.predict_table import TableSystem
from ppstructure.table.matcher import TableMatch
from ppstructure.utility import init_args
from ppocr.utils.logging import get_logger

logger = get_logger()


def parse_args():
    parser = init_args()
    parser.add_argument("--gt_path", type=str)
    return parser.parse_args()


def load_pred_txt(txt_path):
    pred_html_dict = {}
    if not os.path.exists(txt_path):
        return pred_html_dict
    with open(txt_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            img_name, pred_html = line
            pred_html_dict[img_name] = pred_html
    return pred_html_dict


def load_result(path):
    data = {}
    if os.path.exists(path):
        data = pickle.load(open(path, 'rb'))
    return data


def save_result(path, data):
    old_data = load_result(path)
    old_data.update(data)
    with open(path, 'wb') as f:
        pickle.dump(old_data, f)


def compute_diff(ocr_result, structure_result, ocr_result_old,
                 structure_result_old):
    from rich.console import Console
    from rich.table import Column, Table

    console = Console()

    table = Table(
        show_header=True,
        header_style="bold yellow")  #show_header是否显示表头，header_style表头样式
    table.add_column("img_name", style="dim")  #添加表头列
    table.add_column("ocr_det")
    table.add_column("ocr_rec", justify="right")
    table.add_column("structure_bbox", justify="right")
    table.add_column("structure_token", justify="right")

    for img_name in ocr_result:
        dt_boxes, rec_res = ocr_result[img_name]
        dt_boxes_old, rec_res_old = ocr_result_old[img_name]
        structure_res = structure_result[img_name]
        structure_res_old = structure_result_old[img_name]

        # ocr_det 
        assert len(dt_boxes) == len(dt_boxes_old)
        ocr_det_diff = dt_boxes - dt_boxes_old
        ocr_det_diff = ocr_det_diff.mean()
        # ocr_rec 
        assert len(rec_res) == len(rec_res)
        ocr_rec_diff = True
        for i in range(len(rec_res)):
            txt, score = rec_res[i]
            txt_old, score_old = rec_res_old[i]
            if txt != txt_old:
                ocr_rec_diff = False
                print(txt, txt_old)

        # table
        assert len(structure_res[1]) == len(structure_res_old[1])
        structure_bbox_diff = structure_res[1] - structure_res_old[1]
        structure_bbox_diff = structure_bbox_diff.mean()

        assert len(structure_res[0]) == len(structure_res_old[0])
        structure_token_diff = True
        for i in range(len(structure_res[0])):
            if structure_res[0][i] != structure_res_old[0][i]:
                structure_token_diff = False
        table.add_row(img_name,
                      str(ocr_det_diff),
                      str(ocr_rec_diff),
                      str(structure_bbox_diff), str(structure_token_diff))
    console.print(table)


def load_gt_token():
    import json
    p = '/home/zhoujun20/table/PubTabNe/pubtabnet/PubTabNet_2.0.0_val.jsonl'

    d = {}
    with open(p, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            data_line = line.decode('utf-8').strip("\n")
            info = json.loads(data_line)
            file_name = info['filename']
            cells = info['html']['cells'].copy()
            structure = info['html']['structure']['tokens'].copy()
            d[file_name] = len(structure)
    return d


def load_gt_token1():
    import json
    p = '/home/zhoujun20/table/PubTabNe/pubtabnet/PubTabNet_2.0.0_val.jsonl'

    d = {}
    with open(p, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            data_line = line.decode('utf-8').strip("\n")
            info = json.loads(data_line)
            file_name = info['filename']
            cells = info['html']['cells'].copy()
            structure = info['html']['structure']['tokens'].copy()
            structure = ['<html>', '<body>', '<table>'
                         ] + structure + ['</table>', '</body>', '</html>']
            bbox = [x['bbox'] for x in cells if 'bbox' in x]
            d[file_name] = {'structure': structure, 'bbox': bbox}
    return d


def main(gt_path, img_root, args):
    os.makedirs(args.output, exist_ok=True)

    chunk = args.chunk
    # init TableSystem
    text_sys = TableSystem(args)
    # load gt and preds html result
    gt_html_dict = load_pred_txt(gt_path, chunk)

    ocr_result = load_result(os.path.join(args.output, 'ocr.pickle'))
    structure_result = load_result(
        os.path.join(args.output, 'structure.pickle'))

    pred_htmls = []
    gt_htmls = []
    for img_name, gt_html in tqdm(gt_html_dict.items()):
        img = cv2.imread(os.path.join(img_root, img_name))
        if img_name not in ocr_result:
            print('ocr')
            dt_boxes, rec_res, _, _ = text_sys._ocr(img)
            ocr_result[img_name] = [dt_boxes, rec_res]
            save_result(os.path.join(args.output, 'ocr.pickle'), ocr_result)
        if img_name not in structure_result:
            structure_res, _ = text_sys._structure(img)
            structure_result[img_name] = structure_res
            save_result(
                os.path.join(args.output, 'structure.pickle'), structure_result)
        dt_boxes, rec_res = ocr_result[img_name]
        structure_res = structure_result[img_name]
        pred_html = text_sys.match(structure_res, dt_boxes, rec_res)

        pred_htmls.append(pred_html)
        gt_htmls.append(gt_html)

    # save_result(os.path.join(args.output, 'ocr.pickle'), ocr_result)
    # save_result(os.path.join(args.output, 'structure.pickle'), structure_result)
    # compute teds
    teds = TEDS(n_jobs=16)
    scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
    logger.info('teds: {}'.format(sum(scores) / len(scores)))


def main1(gt_path, img_root, args):
    os.makedirs(args.output, exist_ok=True)

    # init TableSystem
    text_sys = TableSystem(args)
    # load gt and preds html result
    gt_html_dict = load_pred_txt(gt_path)

    ocr_result = load_result('../output/en/result_new/ocr.pickle')
    structure_result = load_result(
        os.path.join(args.output, 'structure.pickle'))

    token_len_dict = load_gt_token()
    pred_htmls = []
    gt_htmls = []
    for img_name, gt_html in tqdm(gt_html_dict.items()):
        if token_len_dict[img_name] > 500:
            continue
        img = cv2.imread(os.path.join(img_root, img_name))
        if img_name not in structure_result:
            structure_res, _ = text_sys._structure(img)
            structure_result[img_name] = structure_res
            save_result(
                os.path.join(args.output, 'structure.pickle'), structure_result)
        dt_boxes, rec_res = ocr_result[img_name]
        structure_res = structure_result[img_name]
        pred_html = text_sys.match(structure_res, dt_boxes, rec_res)
        pred_htmls.append(pred_html)
        gt_htmls.append(gt_html)

    # save_result(os.path.join(args.output, 'ocr.pickle'), ocr_result)
    # save_result(os.path.join(args.output, 'structure.pickle'), structure_result)
    # compute teds
    teds = TEDS(n_jobs=16)
    scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
    logger.info('teds: {}'.format(sum(scores) / len(scores)))


def main2(gt_path, img_root, args):
    os.makedirs(args.output, exist_ok=True)

    # init TableSystem
    match = TableMatch()
    # load gt and preds html result
    gt_html_dict = load_pred_txt(gt_path)

    ocr_result = load_result('../output/en/result_new/ocr.pickle')

    gt_dict = load_gt_token1()
    pred_htmls = []
    gt_htmls = []
    for img_name, gt_html in tqdm(gt_html_dict.items()):
        img = cv2.imread(os.path.join(img_root, img_name))

        dt_boxes, rec_res = ocr_result[img_name]
        d = gt_dict[img_name]
        structure_res = (d['structure'], d['bbox'])
        pred_html = match(structure_res, dt_boxes, rec_res)
        pred_htmls.append(pred_html)
        gt_htmls.append(gt_html)

    # save_result(os.path.join(args.output, 'ocr.pickle'), ocr_result)
    # save_result(os.path.join(args.output, 'structure.pickle'), structure_result)
    # compute teds
    teds = TEDS(n_jobs=16)
    scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
    logger.info('teds: {}'.format(sum(scores) / len(scores)))


if __name__ == '__main__':
    args = parse_args()
    main2(args.gt_path, args.image_dir, args)
