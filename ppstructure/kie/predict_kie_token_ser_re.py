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
from tools.infer_kie_token_ser_re import make_input
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results, draw_re_results
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args
from ppstructure.kie.predict_kie_token_ser import SerPredictor

logger = get_logger()


class SerRePredictor(object):
    def __init__(self, args):
        self.use_visual_backbone = args.use_visual_backbone
        self.ser_engine = SerPredictor(args)
        if args.re_model_dir is not None:
            postprocess_params = {'name': 'VQAReTokenLayoutLMPostProcess'}
            self.postprocess_op = build_post_process(postprocess_params)
            self.predictor, self.input_tensor, self.output_tensors, self.config = \
                utility.create_predictor(args, 're', logger)
        else:
            self.predictor = None

    def __call__(self, img):
        starttime = time.time()
        ser_results, ser_inputs, ser_elapse = self.ser_engine(img)
        if self.predictor is None:
            return ser_results, ser_elapse

        re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)
        if self.use_visual_backbone == False:
            re_input.pop(4)
        for idx in range(len(self.input_tensor)):
            self.input_tensor[idx].copy_from_cpu(re_input[idx])

        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        preds = dict(
            loss=outputs[1],
            pred_relations=outputs[2],
            hidden_states=outputs[0], )

        post_result = self.postprocess_op(
            preds,
            ser_results=ser_results,
            entity_idx_dict_batch=entity_idx_dict_batch)

        elapse = time.time() - starttime
        return post_result, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    ser_re_predictor = SerRePredictor(args)
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
            re_res, elapse = ser_re_predictor(img)
            re_res = re_res[0]

            res_str = '{}\t{}\n'.format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": re_res,
                    }, ensure_ascii=False))
            f_w.write(res_str)
            if ser_re_predictor.predictor is not None:
                img_res = draw_re_results(
                    image_file, re_res, font_path=args.vis_font_path)
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] +
                    "_ser_re.jpg")
            else:
                img_res = draw_ser_results(
                    image_file, re_res, font_path=args.vis_font_path)
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] +
                    "_ser.jpg")

            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    main(parse_args())
