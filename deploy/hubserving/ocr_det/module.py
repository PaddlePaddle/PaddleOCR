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

from tools.infer.utility import draw_boxes, base64_to_cv2
from tools.infer.predict_det import TextDetector


@moduleinfo(
    name="ocr_det",
    version="1.0.0",
    summary="ocr detection service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class OCRDet(hub.Module):
    def _initialize(self, use_gpu=False):
        """
        initialize with the necessary elements
        """
        from ocr_det.params import read_params
        cfg = read_params()

        cfg.use_gpu = use_gpu
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 8000
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        cfg.ir_optim = True

        self.text_detector = TextDetector(cfg)

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

    def predict(self,
                images=[],
                paths=[],
                draw_img_save='ocr_det_result',
                visualization=False):
        """
        Get the text box in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
            draw_img_save (str): The directory to store output images.
            visualization (bool): Whether to save image or not.
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
        
        all_results = []
        for img in predicted_data:
            result = {'save_path': ''}
            if img is None:
                logger.info("error in loading image")
                result['data'] = []
                all_results.append(result)
                continue
            dt_boxes, elapse = self.text_detector(img)
            print("Predict time : ", elapse)
            result['data'] = dt_boxes.astype(np.int).tolist()

            if visualization:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw_img = draw_boxes(image, dt_boxes)
                draw_img = np.array(draw_img)
                if not os.path.exists(draw_img_save):
                    os.makedirs(draw_img_save)
                saved_name = 'ndarray_{}.jpg'.format(time.time())
                save_file_path = os.path.join(draw_img_save, saved_name)
                cv2.imwrite(save_file_path, draw_img[:, :, ::-1])
                print("The visualized image saved in {}".format(save_file_path))
                result['save_path'] = save_file_path

            all_results.append(result)
        return all_results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(images_decode, **kwargs)
        return results

   
if __name__ == '__main__':
    ocr = OCRDet()
    image_path = [
        './doc/imgs/11.jpg',
        './doc/imgs/12.jpg',
    ]
    res = ocr.predict(paths=image_path, visualization=True)
    print(res)