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
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import copy
import numpy as np
import time
import tools.infer.utility as utility
from tools.infer.predict_system import TextSystem
from ppstructure.table.predict_table import TableSystem, to_excel
from ppstructure.layout.predict_layout import LayoutDetector
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger

logger = get_logger()


def parse_args():
    parser = utility.init_args()

    # params for output
    parser.add_argument("--table_output", type=str, default='output/table')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_max_text_length", type=int, default=100)
    parser.add_argument("--table_max_elem_length", type=int, default=800)
    parser.add_argument("--table_max_cell_num", type=int, default=500)
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument("--table_char_type", type=str, default='en')
    parser.add_argument("--table_char_dict_path", type=str, default="./ppocr/utils/dict/table_structure_dict.txt")

    # params for layout detector
    parser.add_argument("--layout_model_dir", type=str)
    return parser.parse_args()


class OCRSystem():
    def __init__(self, args):
        self.text_system = TextSystem(args)
        self.table_system = TableSystem(args)
        self.table_layout = LayoutDetector(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score

    def __call__(self, img):
        ori_im = img.copy()
        layout_res = self.table_layout(copy.deepcopy(img))
        for region in layout_res:
            x1, y1, x2, y2 = region['bbox']
            roi_img = ori_im[y1:y2, x1:x2, :]
            if region['label'] == 'table':
                res = self.text_system(roi_img)
            else:
                res = self.text_system(roi_img)
            region['res'] = res
        return layout_res


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    save_folder = args.table_output
    os.makedirs(save_folder, exist_ok=True)

    text_sys = OCRSystem(args)
    img_num = len(image_file_list)
    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag = check_and_read_gif(image_file)
        img_name = os.path.basename(image_file).split('.')[0]
        # excel_path = os.path.join(excel_save_folder, + '.xlsx')
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        res = text_sys(img)

        excel_save_folder = os.path.join(save_folder, img_name)
        os.makedirs(excel_save_folder, exist_ok=True)
        # save res
        for region in res:
            if region['label'] == 'table':
                excel_path = os.path.join(excel_save_folder, '{}.xlsx'.format(region['bbox']))
                to_excel(region['res'], excel_path)
            else:
                with open(os.path.join(excel_save_folder, 'res.txt'),'a',encoding='utf8') as f:
                    for box, rec_res in zip(*region['res']):
                        f.write('{}\t{}\n'.format(np.array(box).reshape(-1).tolist(), rec_res))
        logger.info(res)
        elapse = time.time() - starttime
        logger.info("Predict time : {:.3f}s".format(elapse))


if __name__ == "__main__":
    args = parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
