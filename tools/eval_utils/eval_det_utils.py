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

import logging
import numpy as np

import paddle.fluid as fluid

__all__ = ['eval_det_run']

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

from ppocr.utils.utility import create_module
from .eval_det_iou import DetectionIoUEvaluator
import json
from copy import deepcopy
import cv2
from ppocr.data.reader_main import reader_main
import os


def cal_det_res(exe, config, eval_info_dict):
    global_params = config['Global']
    save_res_path = global_params['save_res_path']
    postprocess_params = deepcopy(config["PostProcess"])
    postprocess_params.update(global_params)
    postprocess = create_module(postprocess_params['function']) \
        (params=postprocess_params)
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))
    with open(save_res_path, "wb") as fout:
        tackling_num = 0
        for data in eval_info_dict['reader']():
            img_num = len(data)
            tackling_num = tackling_num + img_num
            logger.info("test tackling num:%d", tackling_num)
            img_list = []
            ratio_list = []
            img_name_list = []
            for ino in range(img_num):
                img_list.append(data[ino][0])
                ratio_list.append(data[ino][1])
                img_name_list.append(data[ino][2])
            try:
                img_list = np.concatenate(img_list, axis=0)
            except:
                err = "concatenate error usually caused by different input image shapes in evaluation or testing.\n \
                Please set \"test_batch_size_per_card\" in main yml as 1\n \
                or add \"test_image_shape: [h, w]\" in reader yml for EvalReader."
                raise Exception(err)
            outs = exe.run(eval_info_dict['program'], \
                           feed={'image': img_list}, \
                           fetch_list=eval_info_dict['fetch_varname_list'])
            outs_dict = {}
            for tno in range(len(outs)):
                fetch_name = eval_info_dict['fetch_name_list'][tno]
                fetch_value = np.array(outs[tno])
                outs_dict[fetch_name] = fetch_value
            dt_boxes_list = postprocess(outs_dict, ratio_list)
            for ino in range(img_num):
                dt_boxes = dt_boxes_list[ino]
                img_name = img_name_list[ino]
                dt_boxes_json = []
                for box in dt_boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json['points'] = box.tolist()
                    dt_boxes_json.append(tmp_json)
                otstr = img_name + "\t" + json.dumps(dt_boxes_json) + "\n"
                fout.write(otstr.encode())
    return


def load_label_infor(label_file_path, do_ignore=False):
    img_name_label_dict = {}
    with open(label_file_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            substr = line.decode().strip("\n").split("\t")
            bbox_infor = json.loads(substr[1])
            bbox_num = len(bbox_infor)
            for bno in range(bbox_num):
                text = bbox_infor[bno]['transcription']
                ignore = False
                if text == "###" and do_ignore:
                    ignore = True
                bbox_infor[bno]['ignore'] = ignore
            img_name_label_dict[os.path.basename(substr[0])] = bbox_infor
    return img_name_label_dict


def cal_det_metrics(gt_label_path, save_res_path):
    """
    calculate the detection metrics
    Args:
        gt_label_path(string): The groundtruth detection label file path
        save_res_path(string): The saved predicted detection label path
    return:
        claculated metrics including Hmean、precision and recall
    """
    evaluator = DetectionIoUEvaluator()
    gt_label_infor = load_label_infor(gt_label_path, do_ignore=True)
    dt_label_infor = load_label_infor(save_res_path)
    results = []
    for img_name in gt_label_infor:
        gt_label = gt_label_infor[img_name]
        if img_name not in dt_label_infor:
            dt_label = []
        else:
            dt_label = dt_label_infor[img_name]
        result = evaluator.evaluate_image(gt_label, dt_label)
        results.append(result)
    methodMetrics = evaluator.combine_results(results)
    return methodMetrics


def eval_det_run(exe, config, eval_info_dict, mode):
    cal_det_res(exe, config, eval_info_dict)

    save_res_path = config['Global']['save_res_path']
    if mode == "eval":
        gt_label_path = config['EvalReader']['label_file_path']
        metrics = cal_det_metrics(gt_label_path, save_res_path)
    else:
        gt_label_path = config['TestReader']['label_file_path']
        do_eval = config['TestReader']['do_eval']
        if do_eval:
            metrics = cal_det_metrics(gt_label_path, save_res_path)
        else:
            metrics = {}
    return metrics
