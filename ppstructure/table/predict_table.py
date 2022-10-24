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
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import copy
import logging
import numpy as np
import time
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.utility as utility
from tools.infer.predict_system import sorted_boxes
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from ppstructure.table.matcher import TableMatch
from ppstructure.table.table_master_match import TableMasterMatcher
from ppstructure.utility import parse_args
import ppstructure.table.predict_structure as predict_strture

logger = get_logger()


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    #     print(shape)
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


class TableSystem(object):
    def __init__(self, args, text_detector=None, text_recognizer=None):
        self.args = args
        if not args.show_log:
            logger.setLevel(logging.INFO)
        benchmark_tmp = False
        if args.benchmark:
            benchmark_tmp = args.benchmark
            args.benchmark = False
        self.text_detector = predict_det.TextDetector(copy.deepcopy(
            args)) if text_detector is None else text_detector
        self.text_recognizer = predict_rec.TextRecognizer(copy.deepcopy(
            args)) if text_recognizer is None else text_recognizer
        if benchmark_tmp:
            args.benchmark = True
        self.table_structurer = predict_strture.TableStructurer(args)
        if args.table_algorithm in ['TableMaster']:
            self.match = TableMasterMatcher()
        else:
            self.match = TableMatch(filter_ocr_result=True)

        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
            args, 'table', logger)

    def __call__(self, img, return_ocr_result_in_table=False):
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()
        structure_res, elapse = self._structure(copy.deepcopy(img))
        result['cell_bbox'] = structure_res[1].tolist()
        time_dict['table'] = elapse

        dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(
            copy.deepcopy(img))
        time_dict['det'] = det_elapse
        time_dict['rec'] = rec_elapse

        if return_ocr_result_in_table:
            result['boxes'] = dt_boxes  #[x.tolist() for x in dt_boxes]
            result['rec_res'] = rec_res

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()
        time_dict['match'] = toc - tic
        result['html'] = pred_html
        end = time.time()
        time_dict['all'] = end - start
        return result, time_dict

    def _structure(self, img):
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        return structure_res, elapse

    def _ocr(self, img):
        h, w = img.shape[:2]
        dt_boxes, det_elapse = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), det_elapse))
        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0):int(y1), int(x0):int(x1), :]
            img_crop_list.append(text_rect)
        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), rec_elapse))
        return dt_boxes, rec_res, det_elapse, rec_elapse


def to_excel(html_table, excel_path):
    from tablepyxl import tablepyxl
    tablepyxl.document_to_xl(html_table, excel_path)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    os.makedirs(args.output, exist_ok=True)

    table_sys = TableSystem(args)
    img_num = len(image_file_list)

    f_html = open(
        os.path.join(args.output, 'show.html'), mode='w', encoding='utf-8')
    f_html.write('<html>\n<body>\n')
    f_html.write('<table border="1">\n')
    f_html.write(
        "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />"
    )
    f_html.write("<tr>\n")
    f_html.write('<td>img name\n')
    f_html.write('<td>ori image</td>')
    f_html.write('<td>table html</td>')
    f_html.write('<td>cell box</td>')
    f_html.write("</tr>\n")

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag, _ = check_and_read(image_file)
        excel_path = os.path.join(
            args.output, os.path.basename(image_file).split('.')[0] + '.xlsx')
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        pred_res, _ = table_sys(img)
        pred_html = pred_res['html']
        logger.info(pred_html)
        to_excel(pred_html, excel_path)
        logger.info('excel saved to {}'.format(excel_path))
        elapse = time.time() - starttime
        logger.info("Predict time : {:.3f}s".format(elapse))

        if len(pred_res['cell_bbox']) > 0 and len(pred_res['cell_bbox'][
                0]) == 4:
            img = predict_strture.draw_rectangle(image_file,
                                                 pred_res['cell_bbox'])
        else:
            img = utility.draw_boxes(img, pred_res['cell_bbox'])
        img_save_path = os.path.join(args.output, os.path.basename(image_file))
        cv2.imwrite(img_save_path, img)

        f_html.write("<tr>\n")
        f_html.write(f'<td> {os.path.basename(image_file)} <br/>\n')
        f_html.write(f'<td><img src="{image_file}" width=640></td>\n')
        f_html.write('<td><table  border="1">' + pred_html.replace(
            '<html><body><table>', '').replace('</table></body></html>', '') +
                     '</table></td>\n')
        f_html.write(
            f'<td><img src="{os.path.basename(image_file)}" width=640></td>\n')
        f_html.write("</tr>\n")
    f_html.write("</table>\n")
    f_html.close()

    if args.benchmark:
        table_sys.table_structurer.autolog.report()


if __name__ == "__main__":
    args = parse_args()
    if args.use_mp:
        import subprocess
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
