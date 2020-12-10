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
import numpy as np
from pathlib import Path
import tarfile
import requests
from tqdm import tqdm

from tools.infer import predict_system
from ppocr.utils.logging import get_logger

logger = get_logger()
from ppocr.utils.utility import check_and_read_gif, get_image_file_list

__all__ = ['PaddleOCR']

model_urls = {
    'det':
        'https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar',
    'rec': {
        'ch': {
            'url':
                'https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar',
            'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
        },
        'en': {
            'url':
                'https://paddleocr.bj.bcebos.com/20-09-22/mobile/en/en_ppocr_mobile_v1.1_rec_infer.tar',
            'dict_path': './ppocr/utils/ic15_dict.txt'
        },
        'french': {
            'url':
                'https://paddleocr.bj.bcebos.com/20-09-22/mobile/fr/french_ppocr_mobile_v1.1_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/french_dict.txt'
        },
        'german': {
            'url':
                'https://paddleocr.bj.bcebos.com/20-09-22/mobile/ge/german_ppocr_mobile_v1.1_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/german_dict.txt'
        },
        'korean': {
            'url':
                'https://paddleocr.bj.bcebos.com/20-09-22/mobile/kr/korean_ppocr_mobile_v1.1_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/korean_dict.txt'
        },
        'japan': {
            'url':
                'https://paddleocr.bj.bcebos.com/20-09-22/mobile/jp/japan_ppocr_mobile_v1.1_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/japan_dict.txt'
        }
    },
    'cls':
        'https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar'
}

SUPPORT_DET_MODEL = ['DB']
SUPPORT_REC_MODEL = ['CRNN']
BASE_DIR = os.path.expanduser("~/.paddleocr/")


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)


def maybe_download(model_storage_directory, url):
    # using custom model
    if not os.path.exists(os.path.join(
            model_storage_directory, 'model')) or not os.path.exists(
        os.path.join(model_storage_directory, 'params')):
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, 'r') as tarObj:
            for member in tarObj.getmembers():
                if "model" in member.name:
                    filename = 'model'
                elif "params" in member.name:
                    filename = 'params'
                else:
                    continue
                file = tarObj.extractfile(member)
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as f:
                    f.write(file.read())
        os.remove(tmp_path)


def parse_args(mMain=True, add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain:
        parser = argparse.ArgumentParser(add_help=add_help)
        # params for prediction engine
        parser.add_argument("--use_gpu", type=str2bool, default=True)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)

        # params for text detector
        parser.add_argument("--image_dir", type=str)
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_dir", type=str, default=None)
        parser.add_argument("--det_limit_side_len", type=float, default=960)
        parser.add_argument("--det_limit_type", type=str, default='max')

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=2.0)

        # EAST parmas
        parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
        parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
        parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

        # params for text recognizer
        parser.add_argument("--rec_algorithm", type=str, default='CRNN')
        parser.add_argument("--rec_model_dir", type=str, default=None)
        parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
        parser.add_argument("--rec_char_type", type=str, default='ch')
        parser.add_argument("--rec_batch_num", type=int, default=30)
        parser.add_argument("--max_text_length", type=int, default=25)
        parser.add_argument("--rec_char_dict_path", type=str, default=None)
        parser.add_argument("--use_space_char", type=bool, default=True)
        parser.add_argument("--drop_score", type=float, default=0.5)

        # params for text classifier
        parser.add_argument("--cls_model_dir", type=str, default=None)
        parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
        parser.add_argument("--label_list", type=list, default=['0', '180'])
        parser.add_argument("--cls_batch_num", type=int, default=30)
        parser.add_argument("--cls_thresh", type=float, default=0.9)

        parser.add_argument("--enable_mkldnn", type=bool, default=False)
        parser.add_argument("--use_zero_copy_run", type=bool, default=False)
        parser.add_argument("--use_pdserving", type=str2bool, default=False)

        parser.add_argument("--lang", type=str, default='ch')
        parser.add_argument("--det", type=str2bool, default=True)
        parser.add_argument("--rec", type=str2bool, default=True)
        parser.add_argument("--use_angle_cls", type=str2bool, default=False)
        return parser.parse_args()
    else:
        return argparse.Namespace(use_gpu=True,
                                  ir_optim=True,
                                  use_tensorrt=False,
                                  gpu_mem=8000,
                                  image_dir='',
                                  det_algorithm='DB',
                                  det_model_dir=None,
                                  det_limit_side_len=960,
                                  det_limit_type='max',
                                  det_db_thresh=0.3,
                                  det_db_box_thresh=0.5,
                                  det_db_unclip_ratio=2.0,
                                  det_east_score_thresh=0.8,
                                  det_east_cover_thresh=0.1,
                                  det_east_nms_thresh=0.2,
                                  rec_algorithm='CRNN',
                                  rec_model_dir=None,
                                  rec_image_shape="3, 32, 320",
                                  rec_char_type='ch',
                                  rec_batch_num=30,
                                  max_text_length=25,
                                  rec_char_dict_path=None,
                                  use_space_char=True,
                                  drop_score=0.5,
                                  cls_model_dir=None,
                                  cls_image_shape="3, 48, 192",
                                  label_list=['0', '180'],
                                  cls_batch_num=30,
                                  cls_thresh=0.9,
                                  enable_mkldnn=False,
                                  use_zero_copy_run=False,
                                  use_pdserving=False,
                                  lang='ch',
                                  det=True,
                                  rec=True,
                                  use_angle_cls=False
                                  )


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        postprocess_params = parse_args(mMain=False, add_help=False)
        postprocess_params.__dict__.update(**kwargs)
        self.use_angle_cls = postprocess_params.use_angle_cls
        lang = postprocess_params.lang
        assert lang in model_urls[
            'rec'], 'param lang must in {}, but got {}'.format(
            model_urls['rec'].keys(), lang)
        if postprocess_params.rec_char_dict_path is None:
            postprocess_params.rec_char_dict_path = model_urls['rec'][lang][
                'dict_path']

        # init model dir
        if postprocess_params.det_model_dir is None:
            postprocess_params.det_model_dir = os.path.join(BASE_DIR, 'det')
        if postprocess_params.rec_model_dir is None:
            postprocess_params.rec_model_dir = os.path.join(
                BASE_DIR, 'rec/{}'.format(lang))
        if postprocess_params.cls_model_dir is None:
            postprocess_params.cls_model_dir = os.path.join(BASE_DIR, 'cls')
        print(postprocess_params)
        # download model
        maybe_download(postprocess_params.det_model_dir, model_urls['det'])
        maybe_download(postprocess_params.rec_model_dir,
                       model_urls['rec'][lang]['url'])
        maybe_download(postprocess_params.cls_model_dir, model_urls['cls'])

        if postprocess_params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if postprocess_params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        postprocess_params.rec_char_dict_path = Path(
            __file__).parent / postprocess_params.rec_char_dict_path

        # init det_model and rec_model
        super().__init__(postprocess_params)

    def ocr(self, img, det=True, rec=True, cls=False):
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

        self.use_angle_cls = cls
        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if det and rec:
            dt_boxes, rec_res = self.__call__(img)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res


def main():
    # for cmd
    args = parse_args(mMain=True)
    image_dir = args.image_dir
    if image_dir.startswith('http'):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return

    ocr_engine = PaddleOCR(**(args.__dict__))
    for img_path in image_file_list:
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        result = ocr_engine.ocr(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.use_angle_cls)
        if result is not None:
            for line in result:
                logger.info(line)
