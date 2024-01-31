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
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import numpy as np
import time
import logging
from copy import deepcopy

from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results, draw_re_results
from tools.infer.predict_system import TextSystem
from ppstructure.layout.predict_layout import LayoutPredictor
from ppstructure.table.predict_table import TableSystem, to_excel
from ppstructure.utility import parse_args, draw_structure_result

logger = get_logger()


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode
        self.recovery = args.recovery

        self.image_orientation_predictor = None
        if args.image_orientation:
            import paddleclas
            self.image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation")

        if self.mode == 'structure':
            if not args.show_log:
                logger.setLevel(logging.INFO)
            if args.layout == False and args.ocr == True:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            args.drop_score = 0
            # init model
            self.layout_predictor = None
            self.text_system = None
            self.table_system = None
            if args.layout:
                self.layout_predictor = LayoutPredictor(args)
                if args.ocr:
                    self.text_system = TextSystem(args)
            if args.table:
                if self.text_system is not None:
                    self.table_system = TableSystem(
                        args, self.text_system.text_detector,
                        self.text_system.text_recognizer)
                else:
                    self.table_system = TableSystem(args)

        elif self.mode == 'kie':
            from ppstructure.kie.predict_kie_token_ser_re import SerRePredictor
            self.kie_predictor = SerRePredictor(args)

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        time_dict = {
            'image_orientation': 0,
            'layout': 0,
            'table': 0,
            'table_match': 0,
            'det': 0,
            'rec': 0,
            'kie': 0,
            'all': 0
        }
        start = time.time()
        if self.image_orientation_predictor is not None:
            tic = time.time()
            cls_result = self.image_orientation_predictor.predict(
                input_data=img)
            cls_res = next(cls_result)
            angle = cls_res[0]['label_names'][0]
            cv_rotate_code = {
                '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv2.ROTATE_180,
                '270': cv2.ROTATE_90_CLOCKWISE
            }
            if angle in cv_rotate_code:
                img = cv2.rotate(img, cv_rotate_code[angle])
            toc = time.time()
            time_dict['image_orientation'] = toc - tic
        if self.mode == 'structure':
            ori_im = img.copy()
            if self.layout_predictor is not None:
                layout_res, elapse = self.layout_predictor(img)
                time_dict['layout'] += elapse
            else:
                h, w = ori_im.shape[:2]
                layout_res = [dict(bbox=None, label='table')]
            res_list = []
            for region in layout_res:
                res = ''
                if region['bbox'] is not None:
                    x1, y1, x2, y2 = region['bbox']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                if region['label'] == 'table':
                    if self.table_system is not None:
                        res, table_time_dict = self.table_system(
                            roi_img, return_ocr_result_in_table)
                        time_dict['table'] += table_time_dict['table']
                        time_dict['table_match'] += table_time_dict['match']
                        time_dict['det'] += table_time_dict['det']
                        time_dict['rec'] += table_time_dict['rec']
                else:
                    if self.text_system is not None:
                        if self.recovery:
                            wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)
                            wht_im[y1:y2, x1:x2, :] = roi_img
                            filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                                wht_im)
                        else:
                            filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                                roi_img)
                        time_dict['det'] += ocr_time_dict['det']
                        time_dict['rec'] += ocr_time_dict['rec']

                        # remove style char,
                        # when using the recognition model trained on the PubtabNet dataset,
                        # it will recognize the text format in the table, such as <b>
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
                            if not self.recovery:
                                box += [x1, y1]
                            res.append({
                                'text': rec_str,
                                'confidence': float(rec_conf),
                                'text_region': box.tolist()
                            })
                res_list.append({
                    'type': region['label'].lower(),
                    'bbox': [x1, y1, x2, y2],
                    'img': roi_img,
                    'res': res,
                    'img_idx': img_idx
                })
            end = time.time()
            time_dict['all'] = end - start
            return res_list, time_dict
        elif self.mode == 'kie':
            re_res, elapse = self.kie_predictor(img)
            time_dict['kie'] = elapse
            time_dict['all'] = elapse
            return re_res[0], time_dict
        return None, None

table_number = 0

def close_unclosed_brackets(value, symbols):
    if pd.isna(value) or value == "nan":
        return ""  # Replace nan values with an empty string or any other desired value
    
    # Define a regular expression pattern to identify unclosed brackets for numbers, strings, and symbols
    pattern = re.compile(rf'(\S+)\s*\(([^)]*(?:{"|".join(re.escape(s) for s in symbols)}\s*\d+)?[^)]*)\s*\)?')

    # Use the regular expression to find matches in the value
    match = re.search(pattern, value)

    # If an unclosed bracket is found, close it
    if match:
        content_before_bracket = match.group(1)
        content_inside_bracket = match.group(2)
        corrected_value = f"{content_before_bracket} ({content_inside_bracket})"
        return corrected_value
    else:
        return value
        
ef process_excel_file(input_file_path, symbols):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(input_file_path, index_col=0, header=None)

    # Replace "nan" values with an empty string in the entire DataFrame
    df = df.replace({"nan": ""})

    # Drop rows and columns that contain only NaN values
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # If the DataFrame has a multi-level index, flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns]

    # Apply the close_unclosed_brackets function to each cell in the data
    df = df.applymap(lambda x: close_unclosed_brackets(str(x), symbols))

    # Overwrite the original Excel file with the corrected DataFrame
    df.to_excel(input_file_path)
    print(f"Overwritten the original Excel file with the corrected DataFrame.")

def get_unique_filename(base_folder, base_name, img_idx, extension):
    counter = 0
    file_path = os.path.join(base_folder, '{}_{:02d}.{}'.format(base_name, counter + 1, extension))

    # Check if the file already exists, if yes, increment the counter
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(base_folder, '{}_{:02d}.{}'.format(base_name, counter + 1, extension))

    return file_path

def save_structure_res(res, save_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)

    # save res
    with open(
        os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
        'w',
        encoding='utf8') as f:
        for region in res_cp:
            roi_img = region.pop('img')
            f.write('{}\n'.format(json.dumps(region)))

            if region['type'].lower() == 'table' and len(region['res']) > 0 and 'html' in region['res']:
                excel_path = get_unique_filename(excel_save_folder, 'table', img_idx, 'xlsx')
                to_excel(region['res']['html'], excel_path)

                # Read the Excel file into a DataFrame
                df = pd.read_excel(excel_path, header=None)

                # Define symbols to close in the close_unclosed_brackets function
                symbols_to_close = ["%", "@", "$"]  # Add other symbols as needed

                # Apply the close_unclosed_brackets function to each cell in the data
                df = df.applymap(lambda x: close_unclosed_brackets(str(x), symbols_to_close))
                df.columns = [close_unclosed_brackets(str(col), symbols_to_close) for col in df.columns]

                # Overwrite the original Excel file with the corrected DataFrame
                df.to_excel(excel_path, index=False, header=None)
                print(f"Overwritten the original '{os.path.basename(excel_path)}' file with the corrected DataFrame.")
            elif region['type'].lower() == 'figure':
                img_path = get_unique_filename(excel_save_folder, 'figure', img_idx, 'jpg')
                cv2.imwrite(img_path, roi_img)
#def save_structure_res(res, save_folder, img_name, img_idx=0):
 #   excel_save_folder = os.path.join(save_folder, img_name)
  #  os.makedirs(excel_save_folder, exist_ok=True)
   # res_cp = deepcopy(res)

    # save res
    #with open(
     #   os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
      #  'w',
       # encoding='utf8') as f:
        #for region in res_cp:
         #   roi_img = region.pop('img')
          #  f.write('{}\n'.format(json.dumps(region)))

           # if region['type'].lower() == 'table' and len(region['res']) > 0 and 'html' in region['res']:
            #    excel_path = get_unique_filename(excel_save_folder, 'table', img_idx, 'xlsx')
             #   to_excel(region['res']['html'], excel_path)
            #elif region['type'].lower() == 'figure':
             #   img_path = get_unique_filename(excel_save_folder, 'figure', img_idx, 'jpg')
              #  cv2.imwrite(img_path, roi_img)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    if not args.use_pdf2docx_api:
        structure_sys = StructureSystem(args)
        save_folder = os.path.join(args.output, structure_sys.mode)
        os.makedirs(save_folder, exist_ok=True)
    img_num = len(image_file_list)

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag_gif, flag_pdf = check_and_read(image_file)
        img_name = os.path.basename(image_file).split('.')[0]

        if args.recovery and args.use_pdf2docx_api and flag_pdf:
            from pdf2docx.converter import Converter
            os.makedirs(args.output, exist_ok=True)
            docx_file = os.path.join(args.output,
                                     '{}_api.docx'.format(img_name))
            cv = Converter(image_file)
            cv.convert(docx_file)
            cv.close()
            logger.info('docx save to {}'.format(docx_file))
            continue

        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)

        if not flag_pdf:
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            imgs = img

        all_res = []
        for index, img in enumerate(imgs):
            res, time_dict = structure_sys(img, img_idx=index)
            img_save_path = os.path.join(save_folder, img_name,
                                         'show_{}.jpg'.format(index))
            os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
            if structure_sys.mode == 'structure' and res != []:
                draw_img = draw_structure_result(img, res, args.vis_font_path)
                save_structure_res(res, save_folder, img_name, index)
            elif structure_sys.mode == 'kie':
                if structure_sys.kie_predictor.predictor is not None:
                    draw_img = draw_re_results(
                        img, res, font_path=args.vis_font_path)
                else:
                    draw_img = draw_ser_results(
                        img, res, font_path=args.vis_font_path)

                with open(
                        os.path.join(save_folder, img_name,
                                     'res_{}_kie.txt'.format(index)),
                        'w',
                        encoding='utf8') as f:
                    res_str = '{}\t{}\n'.format(
                        image_file,
                        json.dumps(
                            {
                                "ocr_info": res
                            }, ensure_ascii=False))
                    f.write(res_str)
            if res != []:
                cv2.imwrite(img_save_path, draw_img)
                logger.info('result save to {}'.format(img_save_path))
            if args.recovery and res != []:
                from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
                h, w, _ = img.shape
                res = sorted_layout_boxes(res, w)
                all_res += res

        if args.recovery and all_res != []:
            try:
                convert_info_docx(img, all_res, save_folder, img_name)
            except Exception as ex:
                logger.error("error in layout recovery image:{}, err msg: {}".
                             format(image_file, ex))
                continue
        logger.info("Predict time : {:.3f}s".format(time_dict['all']))


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
