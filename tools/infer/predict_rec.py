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
sys.path.append(__dir__)
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


class TextRecognizer(object):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }
        if self.rec_algorithm == "SRN":
            postprocess_params = {
                'name': 'SRNLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "RARE":
            postprocess_params = {
                'name': 'AttnLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == 'NRTR':
            postprocess_params = {
                'name': 'NRTRLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "SAR":
            postprocess_params = {
                'name': 'SARLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "VisionLAN":
            postprocess_params = {
                'name': 'VLLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == 'ViTSTR':
            postprocess_params = {
                'name': 'ViTSTRLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == 'ABINet':
            postprocess_params = {
                'name': 'ABINetLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "SPIN":
            postprocess_params = {
                'name': 'SPINLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "RobustScanner":
            postprocess_params = {
                'name': 'SARLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "rm_symbol": True
            }
        elif self.rec_algorithm == 'RFL':
            postprocess_params = {
                'name': 'RFLLabelDecode',
                "character_dict_path": None,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "SATRN":
            postprocess_params = {
                'name': 'SATRNLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "rm_symbol": True
            }
        elif self.rec_algorithm == "PREN":
            postprocess_params = {'name': 'PRENLabelDecode'}
        elif self.rec_algorithm == "CAN":
            self.inverse = args.rec_image_inverse
            postprocess_params = {
                'name': 'CANLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'rec', logger)
        self.benchmark = args.benchmark
        self.use_onnx = args.use_onnx
        if args.benchmark:
            import auto_log
            pid = os.getpid()
            gpu_id = utility.get_infer_gpuid()
            self.autolog = auto_log.AutoLogger(
                model_name="rec",
                model_precision=args.precision,
                batch_size=args.rec_batch_num,
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

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        if self.rec_algorithm == 'NRTR' or self.rec_algorithm == 'ViTSTR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # return padding_im
            image_pil = Image.fromarray(np.uint8(img))
            if self.rec_algorithm == 'ViTSTR':
                img = image_pil.resize([imgW, imgH], Image.BICUBIC)
            else:
                img = image_pil.resize([imgW, imgH], Image.LANCZOS)
            img = np.array(img)
            norm_img = np.expand_dims(img, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            if self.rec_algorithm == 'ViTSTR':
                norm_img = norm_img.astype(np.float32) / 255.
            else:
                norm_img = norm_img.astype(np.float32) / 128. - 1.
            return norm_img
        elif self.rec_algorithm == 'RFL':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
            resized_image = resized_image.astype('float32')
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
            resized_image -= 0.5
            resized_image /= 0.5
            return resized_image

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if isinstance(w, str):
                pass
            elif w is not None and w > 0:
                imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        if self.rec_algorithm == 'RARE':
            if resized_w > self.rec_image_shape[2]:
                resized_w = self.rec_image_shape[2]
            imgW = self.rec_image_shape[2]
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_vl(self, img, image_shape):

        imgC, imgH, imgW = image_shape
        img = img[:, :, ::-1]  # bgr2rgb
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        return resized_image

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0:img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

    def resize_norm_img_sar(self, img, image_shape,
                            width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h = img.shape[0]
        w = img.shape[1]
        valid_ratio = 1.0
        # make sure new_width is an integral multiple of width_divisor.
        width_divisor = int(1 / width_downsample_ratio)
        # resize
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        if imgW_min is not None:
            resize_w = max(imgW_min, resize_w)
        if imgW_max is not None:
            valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
            resize_w = min(imgW_max, resize_w)
        resized_image = cv2.resize(img, (resize_w, imgH))
        resized_image = resized_image.astype('float32')
        # norm 
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        resize_shape = resized_image.shape
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        pad_shape = padding_im.shape

        return padding_im, resize_shape, pad_shape, valid_ratio

    def resize_norm_img_spin(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # return padding_im
        img = cv2.resize(img, tuple([100, 32]), cv2.INTER_CUBIC)
        img = np.array(img, np.float32)
        img = np.expand_dims(img, -1)
        img = img.transpose((2, 0, 1))
        mean = [127.5]
        std = [127.5]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        mean = np.float32(mean.reshape(1, -1))
        stdinv = 1 / np.float32(std.reshape(1, -1))
        img -= mean
        img *= stdinv
        return img

    def resize_norm_img_svtr(self, img, image_shape):

        imgC, imgH, imgW = image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

    def resize_norm_img_abinet(self, img, image_shape):

        imgC, imgH, imgW = image_shape

        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        resized_image = (
            resized_image - mean[None, None, ...]) / std[None, None, ...]
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')

        return resized_image

    def norm_img_can(self, img, image_shape):

        img = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY)  # CAN only predict gray scale image

        if self.inverse:
            img = 255 - img

        if self.rec_image_shape[0] == 1:
            h, w = img.shape
            _, imgH, imgW = self.rec_image_shape
            if h < imgH or w < imgW:
                padding_h = max(imgH - h, 0)
                padding_w = max(imgW - w, 0)
                img_padded = np.pad(img, ((0, padding_h), (0, padding_w)),
                                    'constant',
                                    constant_values=(255))
                img = img_padded

        img = np.expand_dims(img, 0) / 255.0  # h,w,c -> c,h,w
        img = img.astype('float32')

        return img

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()
        if self.benchmark:
            self.autolog.times.start()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = []
                gsrm_word_pos_list = []
                gsrm_slf_attn_bias1_list = []
                gsrm_slf_attn_bias2_list = []
            if self.rec_algorithm == "SAR":
                valid_ratios = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm == "SAR":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                        img_list[indices[ino]], self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    valid_ratio = np.expand_dims(valid_ratio, axis=0)
                    valid_ratios.append(valid_ratio)
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "SRN":
                    norm_img = self.process_image_srn(
                        img_list[indices[ino]], self.rec_image_shape, 8, 25)
                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])
                elif self.rec_algorithm in ["SVTR", "SATRN"]:
                    norm_img = self.resize_norm_img_svtr(img_list[indices[ino]],
                                                         self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm in ["VisionLAN", "PREN"]:
                    norm_img = self.resize_norm_img_vl(img_list[indices[ino]],
                                                       self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == 'SPIN':
                    norm_img = self.resize_norm_img_spin(img_list[indices[ino]])
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "ABINet":
                    norm_img = self.resize_norm_img_abinet(
                        img_list[indices[ino]], self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "RobustScanner":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                        img_list[indices[ino]],
                        self.rec_image_shape,
                        width_downsample_ratio=0.25)
                    norm_img = norm_img[np.newaxis, :]
                    valid_ratio = np.expand_dims(valid_ratio, axis=0)
                    valid_ratios = []
                    valid_ratios.append(valid_ratio)
                    norm_img_batch.append(norm_img)
                    word_positions_list = []
                    word_positions = np.array(range(0, 40)).astype('int64')
                    word_positions = np.expand_dims(word_positions, axis=0)
                    word_positions_list.append(word_positions)
                elif self.rec_algorithm == "CAN":
                    norm_img = self.norm_img_can(img_list[indices[ino]],
                                                 max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                    norm_image_mask = np.ones(norm_img.shape, dtype='float32')
                    word_label = np.ones([1, 36], dtype='int64')
                    norm_img_mask_batch = []
                    word_label_list = []
                    norm_img_mask_batch.append(norm_image_mask)
                    word_label_list.append(word_label)
                else:
                    norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                    max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.benchmark:
                self.autolog.times.stamp()

            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(
                    gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = np.concatenate(
                    gsrm_slf_attn_bias2_list)

                inputs = [
                    norm_img_batch,
                    encoder_word_pos_list,
                    gsrm_word_pos_list,
                    gsrm_slf_attn_bias1_list,
                    gsrm_slf_attn_bias2_list,
                ]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = {"predict": outputs[2]}
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = {"predict": outputs[2]}
            elif self.rec_algorithm == "SAR":
                valid_ratios = np.concatenate(valid_ratios)
                inputs = [
                    norm_img_batch,
                    np.array(
                        [valid_ratios], dtype=np.float32),
                ]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs[0]
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs[0]
            elif self.rec_algorithm == "RobustScanner":
                valid_ratios = np.concatenate(valid_ratios)
                word_positions_list = np.concatenate(word_positions_list)
                inputs = [norm_img_batch, valid_ratios, word_positions_list]

                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs[0]
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs[0]
            elif self.rec_algorithm == "CAN":
                norm_img_mask_batch = np.concatenate(norm_img_mask_batch)
                word_label_list = np.concatenate(word_label_list)
                inputs = [norm_img_batch, norm_img_mask_batch, word_label_list]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs
                else:
                    input_names = self.predictor.get_input_names()
                    input_tensor = []
                    for i in range(len(input_names)):
                        input_tensor_i = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor_i.copy_from_cpu(inputs[i])
                        input_tensor.append(input_tensor_i)
                    self.input_tensor = input_tensor
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs
            else:
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs[0]
                else:
                    self.input_tensor.copy_from_cpu(norm_img_batch)
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    if len(outputs) != 1:
                        preds = outputs
                    else:
                        preds = outputs[0]
                    self.predictor.try_shrink_memory()
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            if self.benchmark:
                self.autolog.times.end(stamp=True)
        return rec_res, time.time() - st


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []

    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )
    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, _ = text_recognizer(img_list)

    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    for ino in range(len(img_list)):
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               rec_res[ino]))
    if args.benchmark:
        text_recognizer.autolog.report()


if __name__ == "__main__":
    main(utility.parse_args())
