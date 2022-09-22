# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import numpy as np
import time

import tools.infer.utility as utility
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args

from paddleocr import PaddleOCR

logger = get_logger()


class SerPredictor(object):
    def __init__(self, args):
        self.ocr_engine = PaddleOCR(
            use_angle_cls=args.use_angle_cls,
            det_model_dir=args.det_model_dir,
            rec_model_dir=args.rec_model_dir,
            show_log=False,
            use_gpu=args.use_gpu)

        pre_process_list = [{
            'VQATokenLabelEncode': {
                'algorithm': args.kie_algorithm,
                'class_path': args.ser_dict_path,
                'contains_re': False,
                'ocr_engine': self.ocr_engine,
                'order_method': args.ocr_order_method,
            }
        }, {
            'VQATokenPad': {
                'max_seq_len': 512,
                'return_attention_mask': True
            }
        }, {
            'VQASerTokenChunk': {
                'max_seq_len': 512,
                'return_attention_mask': True
            }
        }, {
            'Resize': {
                'size': [224, 224]
            }
        }, {
            'NormalizeImage': {
                'std': [58.395, 57.12, 57.375],
                'mean': [123.675, 116.28, 103.53],
                'scale': '1',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': [
                    'input_ids', 'bbox', 'attention_mask', 'token_type_ids',
                    'image', 'labels', 'segment_offset_id', 'ocr_info',
                    'entities'
                ]
            }
        }]
        postprocess_params = {
            'name': 'VQASerTokenLayoutLMPostProcess',
            "class_path": args.ser_dict_path,
        }

        self.preprocess_op = create_operators(pre_process_list,
                                              {'infer_mode': True})
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'ser', logger)

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        if data[0] is None:
            return None, 0
        starttime = time.time()

        for idx in range(len(data)):
            if isinstance(data[idx], np.ndarray):
                data[idx] = np.expand_dims(data[idx], axis=0)
            else:
                data[idx] = [data[idx]]

        for idx in range(len(self.input_tensor)):
            self.input_tensor[idx].copy_from_cpu(data[idx])

        self.predictor.run()

        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        preds = outputs[0]

        post_result = self.postprocess_op(
            preds, segment_offset_ids=data[6], ocr_infos=data[7])
        elapse = time.time() - starttime
        return post_result, data, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    ser_predictor = SerPredictor(args)
    count = 0
    total_time = 0

    os.makedirs(args.output, exist_ok=True)
    with open(
            os.path.join(args.output, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
                img = img[:, :, ::-1]
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            ser_res, _, elapse = ser_predictor(img)
            ser_res = ser_res[0]

            res_str = '{}\t{}\n'.format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": ser_res,
                    }, ensure_ascii=False))
            f_w.write(res_str)

            img_res = draw_ser_results(
                image_file,
                ser_res,
                font_path=args.vis_font_path, )

            img_save_path = os.path.join(args.output,
                                         os.path.basename(image_file))
            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    main(parse_args())
