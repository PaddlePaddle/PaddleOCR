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
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import numpy as np
import time
import logging
from copy import deepcopy
from attrdict import AttrDict

from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.predict_system import TextSystem
from ppstructure.table.predict_table import TableSystem, to_excel
from ppstructure.utility import parse_args, draw_structure_result
from ppstructure.recovery.docx import convert_info_docx

logger = get_logger()


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode
        if self.mode == 'structure':
            if not args.show_log:
                logger.setLevel(logging.INFO)
            if args.layout == False and args.ocr == True:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            args.drop_score = 0
            # init layout and ocr model
            self.text_system = None
            if args.layout:
                import layoutparser as lp
                config_path = None
                model_path = None
                if os.path.isdir(args.layout_path_model):
                    model_path = args.layout_path_model
                else:
                    config_path = args.layout_path_model
                self.table_layout = lp.PaddleDetectionLayoutModel(
                    config_path=config_path,
                    model_path=model_path,
                    label_map=args.layout_label_map,
                    threshold=0.5,
                    enable_mkldnn=args.enable_mkldnn,
                    enforce_cpu=not args.use_gpu,
                    thread_num=args.cpu_threads)
                if args.ocr:
                    self.text_system = TextSystem(args)
            else:
                self.table_layout = None
            if args.table:
                if self.text_system is not None:
                    self.table_system = TableSystem(
                        args, self.text_system.text_detector,
                        self.text_system.text_recognizer)
                else:
                    self.table_system = TableSystem(args)
            else:
                self.table_system = None

        elif self.mode == 'vqa':
            raise NotImplementedError

    def __call__(self, img, return_ocr_result_in_table=False):
        if self.mode == 'structure':
            ori_im = img.copy()
            if self.table_layout is not None:
                layout_res = self.table_layout.detect(img[..., ::-1])
            else:
                h, w = ori_im.shape[:2]
                layout_res = [AttrDict(coordinates=[0, 0, w, h], type='Table')]
            res_list = []
            for region in layout_res:
                res = ''
                x1, y1, x2, y2 = region.coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi_img = ori_im[y1:y2, x1:x2, :]
                if region.type == 'Table':
                    if self.table_system is not None:
                        res = self.table_system(roi_img,
                                                return_ocr_result_in_table)
                else:
                    if self.text_system is not None:
                        if args.recovery:
                            wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)
                            wht_im[y1:y2, x1:x2, :] = roi_img
                            filter_boxes, filter_rec_res = self.text_system(wht_im)
                        else:
                            filter_boxes, filter_rec_res = self.text_system(roi_img)
                        # remove style char
                        style_token = [
                            '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                            '</b>', '<sub>', '</sup>', '<overline>',
                            '</overline>', '<underline>', '</underline>', '<i>',
                            '</i>'
                        ]
                        res = []
                        for box, rec_res in zip(filter_boxes, filter_rec_res):
                            rec_str, rec_conf = rec_res
                            for token in style_token:
                                if token in rec_str:
                                    rec_str = rec_str.replace(token, '')
                            if not args.recovery:
                                box += [x1, y1]
                            res.append({
                                'text': rec_str,
                                'confidence': float(rec_conf),
                                'text_region': box.tolist()
                            })
                res_list.append({
                    'type': region.type,
                    'bbox': [x1, y1, x2, y2],
                    'img': roi_img,
                    'res': res
                })
            return res_list
        elif self.mode == 'vqa':
            raise NotImplementedError
        return None


def save_structure_res(res, save_folder, img_name):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    # save res
    with open(
            os.path.join(excel_save_folder, 'res.txt'), 'w',
            encoding='utf8') as f:
        for region in res_cp:
            roi_img = region.pop('img')
            f.write('{}\n'.format(json.dumps(region)))

            if region['type'] == 'Table' and len(region[
                    'res']) > 0 and 'html' in region['res']:
                excel_path = os.path.join(excel_save_folder,
                                          '{}.xlsx'.format(region['bbox']))
                to_excel(region['res']['html'], excel_path)
            elif region['type'] == 'Figure':
                img_path = os.path.join(excel_save_folder,
                                        '{}.jpg'.format(region['bbox']))
                cv2.imwrite(img_path, roi_img)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    structure_sys = StructureSystem(args)
    img_num = len(image_file_list)
    save_folder = os.path.join(args.output, structure_sys.mode)
    os.makedirs(save_folder, exist_ok=True)

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag = check_and_read_gif(image_file)
        img_name = os.path.basename(image_file).split('.')[0]

        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        res = structure_sys(img)

        if structure_sys.mode == 'structure':
            save_structure_res(res, save_folder, img_name)
            draw_img = draw_structure_result(img, res, args.vis_font_path)
            img_save_path = os.path.join(save_folder, img_name, 'show.jpg')
        elif structure_sys.mode == 'vqa':
            raise NotImplementedError
            # draw_img = draw_ser_results(img, res, args.vis_font_path)
            # img_save_path = os.path.join(save_folder, img_name + '.jpg')
        cv2.imwrite(img_save_path, draw_img)
        logger.info('result save to {}'.format(img_save_path))
        if args.recovery:
            convert_info_docx(img, res, save_folder, img_name) 
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
