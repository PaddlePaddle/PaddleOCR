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
from ppocr.utils.utility import initial_logger

logger = initial_logger()
from ppocr.utils.utility import check_and_read_gif

__all__ = ['PaddleOCR']

model_params = {
    'ch_det_mv3_db': {
        'url':
        'https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar',
        'algorithm': 'DB',
    },
    'ch_rec_mv3_crnn_enhance': {
        'url':
        'https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar',
        'algorithm': 'CRNN'
    },
}

SUPPORT_DET_MODEL = ['DB']
SUPPORT_REC_MODEL = ['Rosetta', 'CRNN', 'STARNet', 'RARE']


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
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logger.error("ERROR, something went wrong")
        sys.exit(0)


def download_and_unzip(url, model_storage_directory):
    tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
    print('download {} to {}'.format(url, tmp_path))
    os.makedirs(model_storage_directory, exist_ok=True)
    download_with_progressbar(url, tmp_path)
    with tarfile.open(tmp_path, 'r') as tarObj:
        for filename in tarObj.getnames():
            tarObj.extract(filename, model_storage_directory)
    os.remove(tmp_path)


def maybe_download(model_storage_directory, model_name, mode='det'):
    algorithm = None
    # using custom model
    if os.path.exists(os.path.join(model_name, 'model')) and os.path.exists(
            os.path.join(model_name, 'params')):
        return model_name, algorithm
    # using the model of paddleocr
    model_path = os.path.join(model_storage_directory, model_name)
    if not os.path.exists(os.path.join(model_path,
                                       'model')) or not os.path.exists(
                                           os.path.join(model_path, 'params')):
        assert model_name in model_params, 'model must in {}'.format(
            model_params.keys())
        download_and_unzip(model_params[model_name]['url'],
                           model_storage_directory)
        algorithm = model_params[model_name]['algorithm']
    return model_path, algorithm


def parse_args():
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_name", type=str, default='ch_det_mv3_db')
    parser.add_argument("--det_max_side_len", type=float, default=960)

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
    parser.add_argument(
        "--rec_model_name", type=str, default='ch_rec_mv3_crnn_enhance')
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=30)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument("--enable_mkldnn", type=bool, default=False)

    parser.add_argument("--model_storage_directory", type=str, default=False)
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    return parser.parse_args()


class PaddleOCR(predict_system.TextSystem):
    def __init__(self,
                 det_model_name='ch_det_mv3_db',
                 rec_model_name='ch_rec_mv3_crnn_enhance',
                 model_storage_directory=None,
                 log_level=20,
                 **kwargs):
        """
        paddleocr package
        args:
            det_model_name: det_model name, keep same with filename in paddleocr. default is ch_det_mv3_db
            det_model_name: rec_model name, keep same with filename in paddleocr. default is ch_rec_mv3_crnn_enhance
            model_storage_directory: model save path. default is ~/.paddleocr
                                    det model will save to  model_storage_directory/det_model
                                    rec model will save to  model_storage_directory/rec_model
            log_level:
            **kwargs: other params show in paddleocr --help
        """
        logger.setLevel(log_level)
        postprocess_params = parse_args()
        # init model dir
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        else:
            self.model_storage_directory = os.path.expanduser(
                "~/.paddleocr/") + '/model'
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        # download model
        det_model_path, det_algorithm = maybe_download(
            self.model_storage_directory, det_model_name, 'det')
        rec_model_path, rec_algorithm = maybe_download(
            self.model_storage_directory, rec_model_name, 'rec')
        # update model and post_process params
        postprocess_params.__dict__.update(**kwargs)
        postprocess_params.det_model_dir = det_model_path
        postprocess_params.rec_model_dir = rec_model_path
        if det_algorithm is not None:
            postprocess_params.det_algorithm = det_algorithm
        if rec_algorithm is not None:
            postprocess_params.rec_algorithm = rec_algorithm

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

    def ocr(self, img, det=True, rec=True):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not, if false, only rec will be exec. default is True
            rec: use text recognition or not, if false, only det will be exec. default is True
        """
        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, str):
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
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
            rec_res, elapse = self.text_recognizer(img)
            return rec_res
