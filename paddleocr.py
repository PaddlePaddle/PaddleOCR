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

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))

import cv2
import logging
import numpy as np
from pathlib import Path

from tools.infer import predict_system
from ppocr.utils.logging import get_logger

logger = get_logger()
from ppocr.utils.utility import check_and_read_gif, get_image_file_list
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.infer.utility import draw_ocr, str2bool
from ppstructure.utility import init_args, draw_structure_result
from ppstructure.predict_system import OCRSystem, save_structure_res

__all__ = ['PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result', 'save_structure_res']

model_urls = {
    'det': {
        'ch':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar',
        'en':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar',
        'structure': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar'
    },
    'rec': {
        'ch': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
        },
        'en': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/en_dict.txt'
        },
        'french': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/french_dict.txt'
        },
        'german': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/german_dict.txt'
        },
        'korean': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/korean_dict.txt'
        },
        'japan': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/japan_dict.txt'
        },
        'chinese_cht': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
        },
        'ta': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/ta_dict.txt'
        },
        'te': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/te_dict.txt'
        },
        'ka': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/ka_dict.txt'
        },
        'latin': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/latin_dict.txt'
        },
        'arabic': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/arabic_dict.txt'
        },
        'cyrillic': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
        },
        'devanagari': {
            'url':
                'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
        },
        'structure': {
            'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar',
            'dict_path': 'ppocr/utils/dict/table_dict.txt'
        }
    },
    'cls': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
    'table': {
        'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar',
        'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
    }
}

SUPPORT_DET_MODEL = ['DB']
VERSION = '2.2'
SUPPORT_REC_MODEL = ['CRNN']
BASE_DIR = os.path.expanduser("~/.paddleocr/")


def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default='ch')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')

    for action in parser._actions:
        if action.dest in ['rec_char_dict_path', 'table_char_dict_path']:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga',
        'hr', 'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms',
        'mt', 'nl', 'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk',
        'sl', 'sq', 'sv', 'sw', 'tl', 'tr', 'uz', 'vi'
    ]
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd',
        'ava', 'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new',
        'gom', 'sa', 'bgc'
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert lang in model_urls[
        'rec'], 'param lang must in {}, but got {}'.format(
        model_urls['rec'].keys(), lang)
    if lang == "ch":
        det_lang = "ch"
    elif lang == 'structure':
        det_lang = 'structure'
    else:
        det_lang = "en"
    return lang, det_lang


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        params.det_model_dir, det_url = confirm_model_dir_url(params.det_model_dir,
                                                              os.path.join(BASE_DIR, VERSION, 'ocr', 'det', det_lang),
                                                              model_urls['det'][det_lang])
        params.rec_model_dir, rec_url = confirm_model_dir_url(params.rec_model_dir,
                                                              os.path.join(BASE_DIR, VERSION, 'ocr', 'rec', lang),
                                                              model_urls['rec'][lang]['url'])
        params.cls_model_dir, cls_url = confirm_model_dir_url(params.cls_model_dir,
                                                              os.path.join(BASE_DIR, VERSION, 'ocr', 'cls'),
                                                              model_urls['cls'])
        # download model
        maybe_download(params.det_model_dir, det_url)
        maybe_download(params.rec_model_dir, rec_url)
        maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(Path(__file__).parent / model_urls['rec'][lang]['dict_path'])

        print(params)
        # init det_model and rec_model
        super().__init__(params)

    def ocr(self, img, det=True, rec=True, cls=True):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not, if false, only rec will be exec. default is True
            rec: use text recognition or not, if false, only det will be exec. default is True
        """
        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )

        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if det and rec:
            dt_boxes, rec_res = self.__call__(img, cls)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res


class PPStructure(OCRSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        if not params.show_log:
            logger.setLevel(logging.INFO)
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        params.det_model_dir, det_url = confirm_model_dir_url(params.det_model_dir,
                                                              os.path.join(BASE_DIR, VERSION, 'ocr', 'det', det_lang),
                                                              model_urls['det'][det_lang])
        params.rec_model_dir, rec_url = confirm_model_dir_url(params.rec_model_dir,
                                                              os.path.join(BASE_DIR, VERSION, 'ocr', 'rec', lang),
                                                              model_urls['rec'][lang]['url'])
        params.table_model_dir, table_url = confirm_model_dir_url(params.table_model_dir,
                                                                  os.path.join(BASE_DIR, VERSION, 'ocr', 'table'),
                                                                  model_urls['table']['url'])
        # download model
        maybe_download(params.det_model_dir, det_url)
        maybe_download(params.rec_model_dir, rec_url)
        maybe_download(params.table_model_dir, table_url)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(Path(__file__).parent / model_urls['rec'][lang]['dict_path'])
        if params.table_char_dict_path is None:
            params.table_char_dict_path = str(Path(__file__).parent / model_urls['table']['dict_path'])

        print(params)
        super().__init__(params)

    def __call__(self, img):
        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        res = super().__call__(img)
        return res


def main():
    # for cmd
    args = parse_args(mMain=True)
    image_dir = args.image_dir
    if is_link(image_dir):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return
    if args.type == 'ocr':
        engine = PaddleOCR(**(args.__dict__))
    elif args.type == 'structure':
        engine = PPStructure(**(args.__dict__))
    else:
        raise NotImplementedError

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        if args.type == 'ocr':
            result = engine.ocr(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.use_angle_cls)
            if result is not None:
                for line in result:
                    logger.info(line)
        elif args.type == 'structure':
            result = engine(img_path)
            save_structure_res(result, args.output, img_name)

            for item in result:
                item.pop('img')
                logger.info(item)
                
