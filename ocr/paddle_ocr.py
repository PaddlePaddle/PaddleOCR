"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2023-03-09
"""
import os
import sys
import importlib

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))

from ocr.paddle_ocr_settings import SUPPORT_OCR_MODEL_VERSION, BASE_DIR, SUPPORT_DET_MODEL, SUPPORT_REC_MODEL
from ocr.get_utils import check_img, parse_lang, get_model_config, parse_args

import numpy as np
from pathlib import Path

tools = importlib.import_module('.', 'tools')
ppocr = importlib.import_module('.', 'ppocr')
ppstructure = importlib.import_module('.', 'ppstructure')

from tools.infer import predict_system
from ppocr.utils.logging import get_logger

logger = get_logger()
from ppocr.utils.network import maybe_download, confirm_model_dir_url
from tools.infer.utility import check_gpu


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, use_gpu=True, use_angle_cls=False, lang="en", ocr_version="PP-OCRv3", use_onnx=False,
                 det_model_dir=None,
                 rec_model_dir=None, det_algorithm="DB", rec_algorithm="SVTR_LCNet", page_num=0, cls_model_dir=None,
                 rec_char_dict_path=None):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(main=False)
        kwargs = params.__dict__
        params.__dict__.update(kwargs)
        params.det_algorithm = det_algorithm
        params.rec_algorithm = rec_algorithm
        params.rec_char_dict_path = rec_char_dict_path
        params.cls_model_dir = cls_model_dir
        params.ocr_version = ocr_version
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version)
        params.use_gpu = check_gpu(use_gpu)

        self.use_angle_cls = use_angle_cls
        lang, det_lang = parse_lang(lang)

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            det_model_dir,
            os.path.join(BASE_DIR, 'whl', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
        cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
                                            'ch')
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            cls_model_dir,
            os.path.join(BASE_DIR, 'whl', 'cls'), cls_model_config['url'])
        if params.ocr_version == 'PP-OCRv3':
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"
        # download model if using paddle infer
        params.use_onnx = use_onnx
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)
        if rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent.parent / rec_model_config['dict_path'])

        # logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = page_num

    # def __init__(self, **kwargs):
    #     """
    #     paddleocr package
    #     args:
    #         **kwargs: other params show in paddleocr --help
    #     """
    #     params = parse_args(main=False)
    #     params.__dict__.update(**kwargs)
    #     assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
    #         SUPPORT_OCR_MODEL_VERSION, params.ocr_version)
    #     params.use_gpu = check_gpu(params.use_gpu)
    #
    #     if not params.show_log:
    #         logger.setLevel(logging.INFO)
    #     self.use_angle_cls = params.use_angle_cls
    #     lang, det_lang = parse_lang(params.lang)
    #
    #     # init model dir
    #     det_model_config = get_model_config('OCR', params.ocr_version, 'det',
    #                                         det_lang)
    #     params.det_model_dir, det_url = confirm_model_dir_url(
    #         params.det_model_dir,
    #         os.path.join(BASE_DIR, 'whl', 'det', det_lang),
    #         det_model_config['url'])
    #     rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
    #                                         lang)
    #     params.rec_model_dir, rec_url = confirm_model_dir_url(
    #         params.rec_model_dir,
    #         os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
    #     cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
    #                                         'ch')
    #     params.cls_model_dir, cls_url = confirm_model_dir_url(
    #         params.cls_model_dir,
    #         os.path.join(BASE_DIR, 'whl', 'cls'), cls_model_config['url'])
    #     if params.ocr_version == 'PP-OCRv3':
    #         params.rec_image_shape = "3, 48, 320"
    #     else:
    #         params.rec_image_shape = "3, 32, 320"
    #     # download model if using paddle infer
    #     if not params.use_onnx:
    #         maybe_download(params.det_model_dir, det_url)
    #         maybe_download(params.rec_model_dir, rec_url)
    #         maybe_download(params.cls_model_dir, cls_url)
    #
    #     if params.det_algorithm not in SUPPORT_DET_MODEL:
    #         logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
    #         sys.exit(0)
    #     if params.rec_algorithm not in SUPPORT_REC_MODEL:
    #         logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
    #         sys.exit(0)
    #
    #     if params.rec_char_dict_path is None:
    #         params.rec_char_dict_path = str(
    #             Path(__file__).parent / rec_model_config['dict_path'])
    #
    #     logger.debug(params)
    #     # init det_model and rec_model
    #     super().__init__(params)
    #     self.page_num = params.page_num

    def ocr(self, img, det=True, rec=True, cls=False):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not. If false, only rec will be exec. Default is True
            rec: use text recognition or not. If false, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If true, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )

        img = check_img(img)
        # for infer pdf file
        if isinstance(img, list):
            if self.page_num > len(img) or self.page_num == 0:
                self.page_num = len(img)
            imgs = img[:self.page_num]
        else:
            imgs = [img]
        if det and rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                dt_boxes, rec_res, _ = self.__call__(img, cls)
                tmp_res = [[box.tolist(), res]
                           for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                dt_boxes, elapse = self.text_detector(img)
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res
