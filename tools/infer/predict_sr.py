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
from PIL import Image
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import math
import time
import traceback
import paddle

import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read

logger = get_logger()


class TextSR(object):
    def __init__(self, args):
        self.sr_image_shape = [int(v) for v in args.sr_image_shape.split(",")]
        self.sr_batch_num = args.sr_batch_num

        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'sr', logger)
        self.benchmark = args.benchmark
        if args.benchmark:
            import auto_log
            pid = os.getpid()
            gpu_id = utility.get_infer_gpuid()
            self.autolog = auto_log.AutoLogger(
                model_name="sr",
                model_precision=args.precision,
                batch_size=args.sr_batch_num,
                data_shape="dynamic",
                save_path=None,  #args.save_log_path,
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.sr_image_shape
        img = img.resize((imgW // 2, imgH // 2), Image.BICUBIC)
        img_numpy = np.array(img).astype("float32")
        img_numpy = img_numpy.transpose((2, 0, 1)) / 255
        return img_numpy

    def __call__(self, img_list):
        img_num = len(img_list)
        batch_num = self.sr_batch_num
        st = time.time()
        st = time.time()
        all_result = [] * img_num
        if self.benchmark:
            self.autolog.times.start()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.sr_image_shape
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[ino])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.benchmark:
                self.autolog.times.stamp()
            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            if len(outputs) != 1:
                preds = outputs
            else:
                preds = outputs[0]
            all_result.append(outputs)
        if self.benchmark:
            self.autolog.times.end(stamp=True)
        return all_result, time.time() - st


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextSR(args)
    valid_image_file_list = []
    img_list = []

    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [16, 64, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.sr_batch_num))

    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = Image.open(image_file).convert("RGB")
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        preds, _ = text_recognizer(img_list)
        for beg_no in range(len(preds)):
            sr_img = preds[beg_no][1]
            lr_img = preds[beg_no][0]
            for i in (range(sr_img.shape[0])):
                fm_sr = (sr_img[i] * 255).transpose(1, 2, 0).astype(np.uint8)
                fm_lr = (lr_img[i] * 255).transpose(1, 2, 0).astype(np.uint8)
                img_name_pure = os.path.split(valid_image_file_list[
                    beg_no * args.sr_batch_num + i])[-1]
                cv2.imwrite("infer_result/sr_{}".format(img_name_pure),
                            fm_sr[:, :, ::-1])
                logger.info("The visualized image saved in infer_result/sr_{}".
                            format(img_name_pure))

    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    if args.benchmark:
        text_recognizer.autolog.report()


if __name__ == "__main__":
    main(utility.parse_args())
