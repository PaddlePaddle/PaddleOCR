# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import copy
import math
import os
import time

from paddle.fluid.core import AnalysisConfig, create_paddle_predictor, PaddleTensor
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, runnable, serving
from PIL import Image
import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

from tools.infer.utility import base64_to_cv2
from tools.infer.predict_rec import TextRecognizer

class Config(object):
    pass

@moduleinfo(
    name="ocr_rec",
    version="1.0.0",
    summary="ocr recognition service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class OCRRec(hub.Module):
    def _initialize(self, 
                    rec_model_dir="",
                    rec_algorithm="CRNN",
                    rec_char_dict_path="./ppocr/utils/ppocr_keys_v1.txt",
                    rec_batch_num=30,
                    use_gpu=False
                    ):
        """
        initialize with the necessary elements
        """
        self.config = Config()
        self.config.use_gpu = use_gpu
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        self.config.ir_optim = True
        self.config.gpu_mem = 8000

        #params for text recognizer
        self.config.rec_algorithm = rec_algorithm
        self.config.rec_model_dir = rec_model_dir
        # self.config.rec_model_dir = "./inference/rec/"
        
        self.config.rec_image_shape = "3, 32, 320"
        self.config.rec_char_type = 'ch'
        self.config.rec_batch_num = rec_batch_num
        self.config.rec_char_dict_path = rec_char_dict_path
        self.config.use_space_char = True

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                logger.info("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images

    def rec_text(self,
                images=[],
                paths=[]):
        """
        Get the text box in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The result of text detection box and save path of images.
        """

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."
        
        text_recognizer = TextRecognizer(self.config)
        img_list = []
        for img in predicted_data:
            if img is None:
                continue
            img_list.append(img)
        try:
            rec_res, predict_time = text_recognizer(img_list)
        except Exception as e:
            print(e)
            return []
        return rec_res

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.det_text(images_decode, **kwargs)
        return results

   
if __name__ == '__main__':
    ocr = OCRRec()
    image_path = [
        './doc/imgs_words/ch/word_1.jpg',
        './doc/imgs_words/ch/word_2.jpg',
        './doc/imgs_words/ch/word_3.jpg',
    ]
    res = ocr.rec_text(paths=image_path)
    print(res)