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

from tools.infer.utility import draw_ocr, base64_to_cv2
from tools.infer.predict_system import TextSystem


class Config(object):
    pass

@moduleinfo(
    name="ocr_system",
    version="1.0.0",
    summary="ocr system service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class OCRSystem(hub.Module):
    def _initialize(self, 
                    det_model_dir="",
                    det_algorithm="DB",
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

        #params for text detector
        self.config.det_algorithm = det_algorithm
        self.config.det_model_dir = det_model_dir
        # self.config.det_model_dir = "./inference/det/"

        #DB parmas
        self.config.det_db_thresh =0.3
        self.config.det_db_box_thresh =0.5
        self.config.det_db_unclip_ratio =2.0

        #EAST parmas
        self.config.det_east_score_thresh = 0.8
        self.config.det_east_cover_thresh = 0.1
        self.config.det_east_nms_thresh = 0.2

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

    def recognize_text(self,
                       images=[],
                       paths=[],
                       det_max_side_len=960,
                       draw_img_save='ocr_result',
                       visualization=False,
                       text_thresh=0.5):
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
            use_gpu (bool): Whether to use gpu.
            batch_size(int): the program deals once with one
            output_dir (str): The directory to store output images.
            visualization (bool): Whether to save image or not.
            box_thresh(float): the threshold of the detected text box's confidence
            text_thresh(float): the threshold of the recognize chinese texts' confidence
        Returns:
            res (list): The result of chinese texts and save path of images.
        """

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        self.config.det_max_side_len = det_max_side_len
        text_sys = TextSystem(self.config)
        cnt = 0
        all_results = []
        for img in predicted_data:
            result = {'save_path': ''}
            if img is None:
                logger.info("error in loading image")
                result['data'] = []
                all_results.append(result)
                continue
            starttime = time.time()
            dt_boxes, rec_res = text_sys(img)
            elapse = time.time() - starttime
            cnt += 1
            print("Predict time of image %d: %.3fs" % (cnt, elapse))
            dt_num = len(dt_boxes)
            rec_res_final = []
            for dno in range(dt_num):
                text, score = rec_res[dno]
                # if the recognized text confidence score is lower than text_thresh, then drop it
                if score >= text_thresh:
                    # text_str = "%s, %.3f" % (text, score)
                    # print(text_str)
                    rec_res_final.append(
                        {
                            'text': text,
                            'confidence': float(score),
                            'text_box_position': dt_boxes[dno].astype(np.int).tolist()
                        }
                    )
            result['data'] = rec_res_final

            if visualization:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr(image, boxes, txts, scores, draw_txt=True, drop_score=0.5)
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
        results = self.recognize_text(images_decode, **kwargs)
        return results

   
if __name__ == '__main__':
    ocr = OCRSystem()
    image_path = [
        './doc/imgs/11.jpg',
        './doc/imgs/12.jpg',
    ]
    res = ocr.recognize_text(paths=image_path, visualization=True)
    print(res)