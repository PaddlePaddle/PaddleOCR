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

import utility
from ppocr.utils.utility import initial_logger
logger = initial_logger()
import cv2
import predict_system
import copy
import numpy as np
import math
import time
import json

if __name__ == "__main__":
    args = utility.parse_args()
    text_sys = predict_system.TextSystem(args)

    image_file_list = []
    label_file_path = "./eval_perform/gt_res/test_chinese_ic15_500_4pts.txt"
    img_set_path = "./eval_perform/"
    with open(label_file_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            substr = line.decode('utf-8').strip("\n").split("\t")
            if "lsvt" in substr[0]:
                continue
            image_file_list.append(substr[0])

    total_time_all = 0
    count = 0
    save_path = "./output/predict.txt"
    fout = open(save_path, "wb")
    for image_name in image_file_list:
        image_file = img_set_path + image_name
        img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        count += 1
        total_time = 0
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        total_time_all += elapse
        print("Predict time of %s(%d): %.3fs" % (image_file, count, elapse))
        dt_num = len(dt_boxes)
        bbox_list = []
        for dno in range(dt_num):
            box = dt_boxes[dno]
            text, score = rec_res[dno]
            points = []
            for tno in range(len(box)):
                points.append([box[tno][0] * 1.0, box[tno][1] * 1.0])
            bbox_list.append({
                "transcription": text,
                "points": points,
                "scores": score * 1.0
            })
        otstr = image_name + "\t" + json.dumps(bbox_list) + "\n"
        fout.write(otstr.encode('utf-8'))
    avg_time = total_time_all / count
    logger.info("avg_time: {0}".format(avg_time))
    logger.info("avg_fps: {0}".format(1.0 / avg_time))
    fout.close()
