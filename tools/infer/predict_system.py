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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_current_memory_mb
import tools.infer.benchmark_utils as benchmark_utils
logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)

        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)

        rec_res, elapse = self.text_recognizer(img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    for idx, image_file in enumerate(image_file_list):
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        total_time += elapse
        if args.benchmark and idx % 20 == 0:
            cm, gm, gu = get_current_memory_mb(0)
            cpu_mem += cm
            gpu_mem += gm
            gpu_util += gu
            count += 1

        logger.info(
            str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse))
        for text, score in rec_res:
            logger.info("{}, {:.3f}".format(text, score))

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            if flag:
                image_file = image_file[:-3] + "png"
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            logger.info("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))

    logger.info("The predict total time is {}".format(time.time() - _st))
    logger.info("\nThe predict total time is {}".format(total_time))

    img_num = text_sys.text_detector.det_times.img_num
    if args.benchmark:
        mems = {
            'cpu_rss_mb': cpu_mem / count,
            'gpu_rss_mb': gpu_mem / count,
            'gpu_util': gpu_util * 100 / count
        }
    else:
        mems = None
    det_time_dict = text_sys.text_detector.det_times.report(average=True)
    rec_time_dict = text_sys.text_recognizer.rec_times.report(average=True)
    det_model_name = args.det_model_dir
    rec_model_name = args.rec_model_dir

    # construct det log information
    model_info = {
        'model_name': args.det_model_dir.split('/')[-1],
        'precision': args.precision
    }
    data_info = {
        'batch_size': 1,
        'shape': 'dynamic_shape',
        'data_num': det_time_dict['img_num']
    }
    perf_info = {
        'preprocess_time_s': det_time_dict['preprocess_time'],
        'inference_time_s': det_time_dict['inference_time'],
        'postprocess_time_s': det_time_dict['postprocess_time'],
        'total_time_s': det_time_dict['total_time']
    }

    benchmark_log = benchmark_utils.PaddleInferBenchmark(
        text_sys.text_detector.config, model_info, data_info, perf_info, mems,
        args.save_log_path)
    benchmark_log("Det")

    # construct rec log information
    model_info = {
        'model_name': args.rec_model_dir.split('/')[-1],
        'precision': args.precision
    }
    data_info = {
        'batch_size': args.rec_batch_num,
        'shape': 'dynamic_shape',
        'data_num': rec_time_dict['img_num']
    }
    perf_info = {
        'preprocess_time_s': rec_time_dict['preprocess_time'],
        'inference_time_s': rec_time_dict['inference_time'],
        'postprocess_time_s': rec_time_dict['postprocess_time'],
        'total_time_s': rec_time_dict['total_time']
    }
    benchmark_log = benchmark_utils.PaddleInferBenchmark(
        text_sys.text_recognizer.config, model_info, data_info, perf_info, mems,
        args.save_log_path)
    benchmark_log("Rec")


if __name__ == "__main__":
    main(utility.parse_args())
