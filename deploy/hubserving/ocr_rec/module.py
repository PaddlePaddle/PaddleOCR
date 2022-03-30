# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0, ".")
import copy
import paddlehub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, runnable, serving
import cv2
import paddlehub as hub

from tools.infer.utility import base64_to_cv2
from tools.infer.predict_rec import TextRecognizer
from tools.infer.utility import parse_args
from deploy.hubserving.ocr_rec.params import read_params


@moduleinfo(
    name="ocr_rec",
    version="1.0.0",
    summary="ocr recognition service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class OCRRec(hub.Module):
    def _initialize(self, use_gpu=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        cfg = self.merge_configs()

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
        cfg.enable_mkldnn = enable_mkldnn

        self.text_recognizer = TextRecognizer(cfg)

    def merge_configs(self, ):
        # deafult cfg
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        cfg = parse_args()

        update_cfg_map = vars(read_params())

        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        sys.argv = copy.deepcopy(backup_argv)
        return cfg

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

    def predict(self, images=[], paths=[]):
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

        img_list = []
        for img in predicted_data:
            if img is None:
                continue
            img_list.append(img)

        rec_res_final = []
        try:
            rec_res, predict_time = self.text_recognizer(img_list)
            for dno in range(len(rec_res)):
                text, score = rec_res[dno]
                rec_res_final.append({
                    'text': text,
                    'confidence': float(score),
                })
        except Exception as e:
            print(e)
            return [[]]

        return [rec_res_final]

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(images_decode, **kwargs)
        return results


if __name__ == '__main__':
    ocr = OCRRec()
    ocr._initialize()
    image_path = [
        './doc/imgs_words/ch/word_1.jpg',
        './doc/imgs_words/ch/word_2.jpg',
        './doc/imgs_words/ch/word_3.jpg',
    ]
    res = ocr.predict(paths=image_path)
    print(res)
