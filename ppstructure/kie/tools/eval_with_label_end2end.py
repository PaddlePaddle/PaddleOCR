# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import re
import sys
import shapely
from shapely.geometry import Polygon
import numpy as np
from collections import defaultdict
import operator
import Levenshtein
import argparse
import json
import copy


def parse_ser_results_fp(fp, fp_type="gt", ignore_background=True):
    # img/zh_val_0.jpg        {
    #     "height": 3508,
    #     "width": 2480,
    #     "ocr_info": [
    #         {"text": "Maribyrnong", "label": "other", "bbox": [1958, 144, 2184, 198]},
    #         {"text": "CITYCOUNCIL", "label": "other", "bbox": [2052, 183, 2171, 214]},
    #     ]
    assert fp_type in ["gt", "pred"]
    key = "label" if fp_type == "gt" else "pred"
    res_dict = dict()
    with open(fp, "r", encoding='utf-8') as fin:
        lines = fin.readlines()

    for _, line in enumerate(lines):
        img_path, info = line.strip().split("\t")
        # get key
        image_name = os.path.basename(img_path)
        res_dict[image_name] = []
        # get infos
        json_info = json.loads(info)
        for single_ocr_info in json_info["ocr_info"]:
            label = single_ocr_info[key].upper()
            if label in ["O", "OTHERS", "OTHER"]:
                label = "O"
            if ignore_background and label == "O":
                continue
            single_ocr_info["label"] = label
            res_dict[image_name].append(copy.deepcopy(single_ocr_info))
    return res_dict


def polygon_from_str(polygon_points):
    """
    Create a shapely polygon object from gt or dt line.
    """
    polygon_points = np.array(polygon_points).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(poly1, poly2):
    """
    Intersection over union between two shapely polygons.
    """
    if not poly1.intersects(
            poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            # except Exception as e:
            #     print(e)
            print('shapely.geos.TopologicalError occurred, iou set to 0')
            iou = 0
    return iou


def ed(args, str1, str2):
    if args.ignore_space:
        str1 = str1.replace(" ", "")
        str2 = str2.replace(" ", "")
    if args.ignore_case:
        str1 = str1.lower()
        str2 = str2.lower()
    return Levenshtein.distance(str1, str2)


def convert_bbox_to_polygon(bbox):
    """
    bbox  : [x1, y1, x2, y2]
    output: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    xmin, ymin, xmax, ymax = bbox
    poly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    return poly


def eval_e2e(args):
    # gt
    gt_results = parse_ser_results_fp(args.gt_json_path, "gt",
                                      args.ignore_background)
    # pred
    dt_results = parse_ser_results_fp(args.pred_json_path, "pred",
                                      args.ignore_background)
    iou_thresh = args.iou_thres
    num_gt_chars = 0
    gt_count = 0
    dt_count = 0
    hit = 0
    ed_sum = 0

    for img_name in dt_results:
        gt_info = gt_results[img_name]
        gt_count += len(gt_info)

        dt_info = dt_results[img_name]
        dt_count += len(dt_info)

        dt_match = [False] * len(dt_info)
        gt_match = [False] * len(gt_info)

        all_ious = defaultdict(tuple)
        # gt: {text, label, bbox or poly}
        for index_gt, gt in enumerate(gt_info):
            if "poly" not in gt:
                gt["poly"] = convert_bbox_to_polygon(gt["bbox"])
            gt_poly = polygon_from_str(gt["poly"])
            for index_dt, dt in enumerate(dt_info):
                if "poly" not in dt:
                    dt["poly"] = convert_bbox_to_polygon(dt["bbox"])
                dt_poly = polygon_from_str(dt["poly"])
                iou = polygon_iou(dt_poly, gt_poly)
                if iou >= iou_thresh:
                    all_ious[(index_gt, index_dt)] = iou
        sorted_ious = sorted(
            all_ious.items(), key=operator.itemgetter(1), reverse=True)
        sorted_gt_dt_pairs = [item[0] for item in sorted_ious]

        # matched gt and dt
        for gt_dt_pair in sorted_gt_dt_pairs:
            index_gt, index_dt = gt_dt_pair
            if gt_match[index_gt] == False and dt_match[index_dt] == False:
                gt_match[index_gt] = True
                dt_match[index_dt] = True
                # ocr rec results
                gt_text = gt_info[index_gt]["text"]
                dt_text = dt_info[index_dt]["text"]

                # ser results
                gt_label = gt_info[index_gt]["label"]
                dt_label = dt_info[index_dt]["pred"]

                if True:  # ignore_masks[index_gt] == '0':
                    ed_sum += ed(args, gt_text, dt_text)
                    num_gt_chars += len(gt_text)
                    if gt_text == dt_text:
                        if args.ignore_ser_prediction or gt_label == dt_label:
                            hit += 1

# unmatched dt
        for tindex, dt_match_flag in enumerate(dt_match):
            if dt_match_flag == False:
                dt_text = dt_info[tindex]["text"]
                gt_text = ""
                ed_sum += ed(args, dt_text, gt_text)

# unmatched gt
        for tindex, gt_match_flag in enumerate(gt_match):
            if gt_match_flag == False:
                dt_text = ""
                gt_text = gt_info[tindex]["text"]
                ed_sum += ed(args, gt_text, dt_text)
                num_gt_chars += len(gt_text)

    eps = 1e-9
    print("config: ", args)
    print('hit, dt_count, gt_count', hit, dt_count, gt_count)
    precision = hit / (dt_count + eps)
    recall = hit / (gt_count + eps)
    fmeasure = 2.0 * precision * recall / (precision + recall + eps)
    avg_edit_dist_img = ed_sum / len(gt_results)
    avg_edit_dist_field = ed_sum / (gt_count + eps)
    character_acc = 1 - ed_sum / (num_gt_chars + eps)

    print('character_acc: %.2f' % (character_acc * 100) + "%")
    print('avg_edit_dist_field: %.2f' % (avg_edit_dist_field))
    print('avg_edit_dist_img: %.2f' % (avg_edit_dist_img))
    print('precision: %.2f' % (precision * 100) + "%")
    print('recall: %.2f' % (recall * 100) + "%")
    print('fmeasure: %.2f' % (fmeasure * 100) + "%")

    return


def parse_args():
    """
    """

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument(
        "--gt_json_path",
        default=None,
        type=str,
        required=True, )
    parser.add_argument(
        "--pred_json_path",
        default=None,
        type=str,
        required=True, )

    parser.add_argument("--iou_thres", default=0.5, type=float)

    parser.add_argument(
        "--ignore_case",
        default=False,
        type=str2bool,
        help="whether to do lower case for the strs")

    parser.add_argument(
        "--ignore_space",
        default=True,
        type=str2bool,
        help="whether to ignore space")

    parser.add_argument(
        "--ignore_background",
        default=True,
        type=str2bool,
        help="whether to ignore other label")

    parser.add_argument(
        "--ignore_ser_prediction",
        default=False,
        type=str2bool,
        help="whether to ignore ocr pred results")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    eval_e2e(args)
