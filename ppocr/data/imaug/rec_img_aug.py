# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import cv2
import numpy as np
import random
import copy
from PIL import Image
from .text_image_aug import tia_perspective, tia_stretch, tia_distort
from .abinet_aug import CVGeometry, CVDeterioration, CVColorJitter, SVTRGeometry, SVTRDeterioration
from paddle.vision.transforms import Compose


class RecAug(object):
    def __init__(self,
                 tia_prob=0.4,
                 crop_prob=0.4,
                 reverse_prob=0.4,
                 noise_prob=0.4,
                 jitter_prob=0.4,
                 blur_prob=0.4,
                 hsv_aug_prob=0.4,
                 **kwargs):
        self.tia_prob = tia_prob
        self.bda = BaseDataAugmentation(crop_prob, reverse_prob, noise_prob,
                                        jitter_prob, blur_prob, hsv_aug_prob)

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape

        # tia
        if random.random() <= self.tia_prob:
            if h >= 20 and w >= 20:
                img = tia_distort(img, random.randint(3, 6))
                img = tia_stretch(img, random.randint(3, 6))
            img = tia_perspective(img)

        # bda
        data['image'] = img
        data = self.bda(data)
        return data


class BaseDataAugmentation(object):
    def __init__(self,
                 crop_prob=0.4,
                 reverse_prob=0.4,
                 noise_prob=0.4,
                 jitter_prob=0.4,
                 blur_prob=0.4,
                 hsv_aug_prob=0.4,
                 **kwargs):
        self.crop_prob = crop_prob
        self.reverse_prob = reverse_prob
        self.noise_prob = noise_prob
        self.jitter_prob = jitter_prob
        self.blur_prob = blur_prob
        self.hsv_aug_prob = hsv_aug_prob
        # for GaussianBlur
        self.fil = cv2.getGaussianKernel(ksize=5, sigma=1, ktype=cv2.CV_32F)

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape

        if random.random() <= self.crop_prob and h >= 20 and w >= 20:
            img = get_crop(img)

        if random.random() <= self.blur_prob:
            # GaussianBlur
            img = cv2.sepFilter2D(img, -1, self.fil, self.fil)

        if random.random() <= self.hsv_aug_prob:
            img = hsv_aug(img)

        if random.random() <= self.jitter_prob:
            img = jitter(img)

        if random.random() <= self.noise_prob:
            img = add_gasuss_noise(img)

        if random.random() <= self.reverse_prob:
            img = 255 - img

        data['image'] = img
        return data


class ABINetRecAug(object):
    def __init__(self,
                 geometry_p=0.5,
                 deterioration_p=0.25,
                 colorjitter_p=0.25,
                 **kwargs):
        self.transforms = Compose([
            CVGeometry(
                degrees=45,
                translate=(0.0, 0.0),
                scale=(0.5, 2.),
                shear=(45, 15),
                distortion=0.5,
                p=geometry_p), CVDeterioration(
                    var=20, degrees=6, factor=4, p=deterioration_p),
            CVColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=colorjitter_p)
        ])

    def __call__(self, data):
        img = data['image']
        img = self.transforms(img)
        data['image'] = img
        return data


class RecConAug(object):
    def __init__(self,
                 prob=0.5,
                 image_shape=(32, 320, 3),
                 max_text_length=25,
                 ext_data_num=1,
                 **kwargs):
        self.ext_data_num = ext_data_num
        self.prob = prob
        self.max_text_length = max_text_length
        self.image_shape = image_shape
        self.max_wh_ratio = self.image_shape[1] / self.image_shape[0]

    def merge_ext_data(self, data, ext_data):
        ori_w = round(data['image'].shape[1] / data['image'].shape[0] *
                      self.image_shape[0])
        ext_w = round(ext_data['image'].shape[1] / ext_data['image'].shape[0] *
                      self.image_shape[0])
        data['image'] = cv2.resize(data['image'], (ori_w, self.image_shape[0]))
        ext_data['image'] = cv2.resize(ext_data['image'],
                                       (ext_w, self.image_shape[0]))
        data['image'] = np.concatenate(
            [data['image'], ext_data['image']], axis=1)
        data["label"] += ext_data["label"]
        return data

    def __call__(self, data):
        rnd_num = random.random()
        if rnd_num > self.prob:
            return data
        for idx, ext_data in enumerate(data["ext_data"]):
            if len(data["label"]) + len(ext_data[
                    "label"]) > self.max_text_length:
                break
            concat_ratio = data['image'].shape[1] / data['image'].shape[
                0] + ext_data['image'].shape[1] / ext_data['image'].shape[0]
            if concat_ratio > self.max_wh_ratio:
                break
            data = self.merge_ext_data(data, ext_data)
        data.pop("ext_data")
        return data


class SVTRRecAug(object):
    def __init__(self,
                 aug_type=0,
                 geometry_p=0.5,
                 deterioration_p=0.25,
                 colorjitter_p=0.25,
                 **kwargs):
        self.transforms = Compose([
            SVTRGeometry(
                aug_type=aug_type,
                degrees=45,
                translate=(0.0, 0.0),
                scale=(0.5, 2.),
                shear=(45, 15),
                distortion=0.5,
                p=geometry_p), SVTRDeterioration(
                    var=20, degrees=6, factor=4, p=deterioration_p),
            CVColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=colorjitter_p)
        ])

    def __call__(self, data):
        img = data['image']
        img = self.transforms(img)
        data['image'] = img
        return data


class ClsResizeImg(object):
    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']
        norm_img, _ = resize_norm_img(img, self.image_shape)
        data['image'] = norm_img
        return data


class RecResizeImg(object):
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 eval_mode=False,
                 character_dict_path='./ppocr/utils/ppocr_keys_v1.txt',
                 padding=True,
                 **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.eval_mode = eval_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    def __call__(self, data):
        img = data['image']
        if self.eval_mode or (self.infer_mode and
                              self.character_dict_path is not None):
            norm_img, valid_ratio = resize_norm_img_chinese(img,
                                                            self.image_shape)
        else:
            norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                    self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


class VLRecResizeImg(object):
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 character_dict_path='./ppocr/utils/ppocr_keys_v1.txt',
                 padding=True,
                 **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    def __call__(self, data):
        img = data['image']

        imgC, imgH, imgW = self.image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_w = imgW
        resized_image = resized_image.astype('float32')
        if self.image_shape[0] == 1:
            resized_image = resized_image / 255
            norm_img = resized_image[np.newaxis, :]
        else:
            norm_img = resized_image.transpose((2, 0, 1)) / 255
        valid_ratio = min(1.0, float(resized_w / imgW))

        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


class RFLRecResizeImg(object):
    def __init__(self, image_shape, padding=True, interpolation=1, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

        self.interpolation = interpolation
        if self.interpolation == 0:
            self.interpolation = cv2.INTER_NEAREST
        elif self.interpolation == 1:
            self.interpolation = cv2.INTER_LINEAR
        elif self.interpolation == 2:
            self.interpolation = cv2.INTER_CUBIC
        elif self.interpolation == 3:
            self.interpolation = cv2.INTER_AREA
        else:
            raise Exception("Unsupported interpolation type !!!")

    def __call__(self, data):
        img = data['image']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_img, valid_ratio = resize_norm_img(
            img, self.image_shape, self.padding, self.interpolation)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


class SRNRecResizeImg(object):
    def __init__(self, image_shape, num_heads, max_text_length, **kwargs):
        self.image_shape = image_shape
        self.num_heads = num_heads
        self.max_text_length = max_text_length

    def __call__(self, data):
        img = data['image']
        norm_img = resize_norm_img_srn(img, self.image_shape)
        data['image'] = norm_img
        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            srn_other_inputs(self.image_shape, self.num_heads, self.max_text_length)

        data['encoder_word_pos'] = encoder_word_pos
        data['gsrm_word_pos'] = gsrm_word_pos
        data['gsrm_slf_attn_bias1'] = gsrm_slf_attn_bias1
        data['gsrm_slf_attn_bias2'] = gsrm_slf_attn_bias2
        return data


class SARRecResizeImg(object):
    def __init__(self, image_shape, width_downsample_ratio=0.25, **kwargs):
        self.image_shape = image_shape
        self.width_downsample_ratio = width_downsample_ratio

    def __call__(self, data):
        img = data['image']
        norm_img, resize_shape, pad_shape, valid_ratio = resize_norm_img_sar(
            img, self.image_shape, self.width_downsample_ratio)
        data['image'] = norm_img
        data['resized_shape'] = resize_shape
        data['pad_shape'] = pad_shape
        data['valid_ratio'] = valid_ratio
        return data


class PRENResizeImg(object):
    def __init__(self, image_shape, **kwargs):
        """
        Accroding to original paper's realization, it's a hard resize method here. 
        So maybe you should optimize it to fit for your task better.
        """
        self.dst_h, self.dst_w = image_shape

    def __call__(self, data):
        img = data['image']
        resized_img = cv2.resize(
            img, (self.dst_w, self.dst_h), interpolation=cv2.INTER_LINEAR)
        resized_img = resized_img.transpose((2, 0, 1)) / 255
        resized_img -= 0.5
        resized_img /= 0.5
        data['image'] = resized_img.astype(np.float32)
        return data


class SPINRecResizeImg(object):
    def __init__(self,
                 image_shape,
                 interpolation=2,
                 mean=(127.5, 127.5, 127.5),
                 std=(127.5, 127.5, 127.5),
                 **kwargs):
        self.image_shape = image_shape

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.interpolation = interpolation

    def __call__(self, data):
        img = data['image']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # different interpolation type corresponding the OpenCV
        if self.interpolation == 0:
            interpolation = cv2.INTER_NEAREST
        elif self.interpolation == 1:
            interpolation = cv2.INTER_LINEAR
        elif self.interpolation == 2:
            interpolation = cv2.INTER_CUBIC
        elif self.interpolation == 3:
            interpolation = cv2.INTER_AREA
        else:
            raise Exception("Unsupported interpolation type !!!")
        # Deal with the image error during image loading
        if img is None:
            return None

        img = cv2.resize(img, tuple(self.image_shape), interpolation)
        img = np.array(img, np.float32)
        img = np.expand_dims(img, -1)
        img = img.transpose((2, 0, 1))
        # normalize the image
        img = img.copy().astype(np.float32)
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        img -= mean
        img *= stdinv
        data['image'] = img
        return data


class GrayRecResizeImg(object):
    def __init__(self,
                 image_shape,
                 resize_type,
                 inter_type='Image.LANCZOS',
                 scale=True,
                 padding=False,
                 **kwargs):
        self.image_shape = image_shape
        self.resize_type = resize_type
        self.padding = padding
        self.inter_type = eval(inter_type)
        self.scale = scale

    def __call__(self, data):
        img = data['image']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_shape = self.image_shape
        if self.padding:
            imgC, imgH, imgW = image_shape
            # todo: change to 0 and modified image shape
            h = img.shape[0]
            w = img.shape[1]
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
            norm_img = np.expand_dims(resized_image, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            resized_image = norm_img.astype(np.float32) / 128. - 1.
            padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
            padding_im[:, :, 0:resized_w] = resized_image
            data['image'] = padding_im
            return data
        if self.resize_type == 'PIL':
            image_pil = Image.fromarray(np.uint8(img))
            img = image_pil.resize(self.image_shape, self.inter_type)
            img = np.array(img)
        if self.resize_type == 'OpenCV':
            img = cv2.resize(img, self.image_shape)
        norm_img = np.expand_dims(img, -1)
        norm_img = norm_img.transpose((2, 0, 1))
        if self.scale:
            data['image'] = norm_img.astype(np.float32) / 128. - 1.
        else:
            data['image'] = norm_img.astype(np.float32) / 255.
        return data


class ABINetRecResizeImg(object):
    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']
        norm_img, valid_ratio = resize_norm_img_abinet(img, self.image_shape)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


class SVTRRecResizeImg(object):
    def __init__(self, image_shape, padding=True, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

    def __call__(self, data):
        img = data['image']

        norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


class RobustScannerRecResizeImg(object):
    def __init__(self,
                 image_shape,
                 max_text_length,
                 width_downsample_ratio=0.25,
                 **kwargs):
        self.image_shape = image_shape
        self.width_downsample_ratio = width_downsample_ratio
        self.max_text_length = max_text_length

    def __call__(self, data):
        img = data['image']
        norm_img, resize_shape, pad_shape, valid_ratio = resize_norm_img_sar(
            img, self.image_shape, self.width_downsample_ratio)
        word_positons = np.array(range(0, self.max_text_length)).astype('int64')
        data['image'] = norm_img
        data['resized_shape'] = resize_shape
        data['pad_shape'] = pad_shape
        data['valid_ratio'] = valid_ratio
        data['word_positons'] = word_positons
        return data


def resize_norm_img_sar(img, image_shape, width_downsample_ratio=0.25):
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


def resize_norm_img(img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


def resize_norm_img_srn(img, image_shape):
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


def resize_norm_img_abinet(img, image_shape):
    imgC, imgH, imgW = image_shape

    resized_image = cv2.resize(
        img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
    resized_w = imgW
    resized_image = resized_image.astype('float32')
    resized_image = resized_image / 255.

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    resized_image = (
        resized_image - mean[None, None, ...]) / std[None, None, ...]
    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = resized_image.astype('float32')

    valid_ratio = min(1.0, float(resized_w / imgW))
    return resized_image, valid_ratio


def srn_other_inputs(image_shape, num_heads, max_text_length):

    imgC, imgH, imgW = image_shape
    feature_dim = int((imgH / 8) * (imgW / 8))

    encoder_word_pos = np.array(range(0, feature_dim)).reshape(
        (feature_dim, 1)).astype('int64')
    gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
        (max_text_length, 1)).astype('int64')

    gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
    gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
        [1, max_text_length, max_text_length])
    gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1,
                                  [num_heads, 1, 1]) * [-1e9]

    gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
        [1, max_text_length, max_text_length])
    gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2,
                                  [num_heads, 1, 1]) * [-1e9]

    return [
        encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
        gsrm_slf_attn_bias2
    ]


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def hsv_aug(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


def rad(x):
    """
    rad
    """
    return x * np.pi / 180


def get_warpR(config):
    """
    get_warpR
    """
    anglex, angley, anglez, fov, w, h, r = \
        config.anglex, config.angley, config.anglez, config.fov, config.w, config.h, config.r
    if w > 69 and w < 112:
        anglex = anglex * 1.5

    z = np.sqrt(w**2 + h**2) / 2 / np.tan(rad(fov / 2))
    # Homogeneous coordinate transformation matrix
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0], [
                       0,
                       -np.sin(rad(anglex)),
                       np.cos(rad(anglex)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0], [
                       -np.sin(rad(angley)),
                       0,
                       np.cos(rad(angley)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    r = rx.dot(ry).dot(rz)
    # generate 4 points
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = np.array([dst1, dst2, dst3, dst4])
    org = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    dst = np.zeros((4, 2), np.float32)
    # Project onto the image plane
    dst[:, 0] = list_dst[:, 0] * z / (z - list_dst[:, 2]) + pcenter[0]
    dst[:, 1] = list_dst[:, 1] * z / (z - list_dst[:, 2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    dst1, dst2, dst3, dst4 = dst
    r1 = int(min(dst1[1], dst2[1]))
    r2 = int(max(dst3[1], dst4[1]))
    c1 = int(min(dst1[0], dst3[0]))
    c2 = int(max(dst2[0], dst4[0]))

    try:
        ratio = min(1.0 * h / (r2 - r1), 1.0 * w / (c2 - c1))

        dx = -c1
        dy = -r1
        T1 = np.float32([[1., 0, dx], [0, 1., dy], [0, 0, 1.0 / ratio]])
        ret = T1.dot(warpR)
    except:
        ratio = 1.0
        T1 = np.float32([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        ret = T1
    return ret, (-r1, -c1), ratio, dst


def get_warpAffine(config):
    """
    get_warpAffine
    """
    anglez = config.anglez
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    return rz
