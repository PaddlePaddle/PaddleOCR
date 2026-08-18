"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import cv2
import numpy as np
import math
import random
from PIL import Image
from paddle import get_device


class DecodeImage(object):
    """decode image"""

    def __init__(
        self, img_mode="RGB", channel_first=False, ignore_orientation=False, **kwargs
    ):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        img = data["image"]
        assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype="uint8")
        if self.img_mode == "GRAY":
            # For GRAY mode, decode directly to a single-channel grayscale image.
            decode_flag = cv2.IMREAD_GRAYSCALE
        else:
            # For RGB mode, decode to a 3-channel color image.
            decode_flag = cv2.IMREAD_COLOR

        if self.ignore_orientation:
            decode_flag |= cv2.IMREAD_IGNORE_ORIENTATION

        img = cv2.imdecode(img, decode_flag)

        if img is None:
            return None
        if self.img_mode == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == "RGB":
            assert img.shape[2] == 3, "invalid shape of image[%s]" % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data["image"] = img
        return data


class NormalizeImage(object):
    """normalize image such as subtract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        data["image"] = (img.astype("float32") * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """convert hwc image to chw image"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        data["image"] = img.transpose((2, 0, 1))
        return data


class Fasttext(object):
    def __init__(self, path="None", **kwargs):
        import fasttext

        self.fast_model = fasttext.load_model(path)

    def __call__(self, data):
        label = data["label"]
        fast_label = self.fast_model[label]
        data["fast_label"] = fast_label
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class Pad(object):
    def __init__(self, size=None, size_div=32, **kwargs):
        if size is not None and not isinstance(size, (int, list, tuple)):
            raise TypeError(
                "Type of target_size is invalid. Now is {}".format(type(size))
            )
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.size_div = size_div

    def __call__(self, data):
        img = data["image"]
        img_h, img_w = img.shape[0], img.shape[1]
        if self.size:
            resize_h2, resize_w2 = self.size
            assert (
                img_h < resize_h2 and img_w < resize_w2
            ), "(h, w) of target size should be greater than (img_h, img_w)"
        else:
            resize_h2 = max(
                int(math.ceil(img.shape[0] / self.size_div) * self.size_div),
                self.size_div,
            )
            resize_w2 = max(
                int(math.ceil(img.shape[1] / self.size_div) * self.size_div),
                self.size_div,
            )
        img = cv2.copyMakeBorder(
            img,
            0,
            resize_h2 - img_h,
            0,
            resize_w2 - img_w,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        data["image"] = img
        return data


class Resize(object):
    def __init__(self, size=(640, 640), **kwargs):
        self.size = size

    def resize_image(self, img):
        resize_h, resize_w = self.size
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def __call__(self, data):
        img = data["image"]
        if "polys" in data:
            text_polys = data["polys"]

        img_resize, [ratio_h, ratio_w] = self.resize_image(img)
        if "polys" in data:
            new_boxes = []
            for box in text_polys:
                new_box = []
                for cord in box:
                    new_box.append([cord[0] * ratio_w, cord[1] * ratio_h])
                new_boxes.append(new_box)
            data["polys"] = np.array(new_boxes, dtype=np.float32)
        data["image"] = img_resize
        return data


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.keep_ratio = False
        self.max_side_limit = kwargs.get("max_side_limit", 4000)
        if "image_shape" in kwargs:
            self.image_shape = kwargs["image_shape"]
            self.resize_type = 1
            if "keep_ratio" in kwargs:
                self.keep_ratio = kwargs["keep_ratio"]
        elif "limit_side_len" in kwargs:
            self.limit_side_len = kwargs["limit_side_len"]
            self.limit_type = kwargs.get("limit_type", "min")
        elif "resize_long" in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get("resize_long", 960)
        else:
            self.limit_side_len = 736
            self.limit_type = "min"

    def __call__(self, data):
        img = data["image"]
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data["image"] = img
        data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w])
        if "iluvatar_gpu" in get_device():
            data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w]).astype(
                np.float32
            )
        return data

    def image_padding(self, im, value=0):
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        if self.keep_ratio is True:
            resize_w = ori_w * resize_h / ori_h
            N = math.ceil(resize_w / 32)
            resize_w = N * 32
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # limit the max side
        if self.limit_type == "max":
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif self.limit_type == "min":
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif self.limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception("not support limit type, image ")
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        if max(resize_h, resize_w) > self.max_side_limit:
            print(
                f"Resized image size ({resize_h}x{resize_w}) exceeds max_side_limit of {self.max_side_limit}. "
                f"Resizing to fit within limit."
            )
            ratio = float(self.max_side_limit) / max(resize_h, resize_w)
            resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class E2EResizeForTest(object):
    def __init__(self, **kwargs):
        super(E2EResizeForTest, self).__init__()
        self.max_side_len = kwargs["max_side_len"]
        self.valid_set = kwargs["valid_set"]

    def __call__(self, data):
        img = data["image"]
        src_h, src_w, _ = img.shape
        if self.valid_set == "totaltext":
            im_resized, [ratio_h, ratio_w] = self.resize_image_for_totaltext(
                img, max_side_len=self.max_side_len
            )
        else:
            im_resized, (ratio_h, ratio_w) = self.resize_image(
                img, max_side_len=self.max_side_len
            )
        data["image"] = im_resized
        data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_for_totaltext(self, im, max_side_len=512):
        h, w, _ = im.shape
        resize_w = w
        resize_h = h
        ratio = 1.25
        if h * ratio > max_side_len:
            ratio = float(max_side_len) / resize_h
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image(self, im, max_side_len=512):
        """
        resize image to a size multiple of max_stride which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(max_side_len) / resize_h
        else:
            ratio = float(max_side_len) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)


class KieResize(object):
    def __init__(self, **kwargs):
        super(KieResize, self).__init__()
        self.max_side, self.min_side = kwargs["img_scale"][0], kwargs["img_scale"][1]

    def __call__(self, data):
        img = data["image"]
        points = data["points"]
        src_h, src_w, _ = img.shape
        (
            im_resized,
            scale_factor,
            [ratio_h, ratio_w],
            [new_h, new_w],
        ) = self.resize_image(img)
        resize_points = self.resize_boxes(img, points, scale_factor)
        data["ori_image"] = img
        data["ori_boxes"] = points
        data["points"] = resize_points
        data["image"] = im_resized
        data["shape"] = np.array([new_h, new_w])
        return data

    def resize_image(self, img):
        norm_img = np.zeros([1024, 1024, 3], dtype="float32")
        scale = [512, 1024]
        h, w = img.shape[:2]
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        resize_w, resize_h = int(w * float(scale_factor) + 0.5), int(
            h * float(scale_factor) + 0.5
        )
        max_stride = 32
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(img, (resize_w, resize_h))
        new_h, new_w = im.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        norm_img[:new_h, :new_w, :] = im
        return norm_img, scale_factor, [h_scale, w_scale], [new_h, new_w]

    def resize_boxes(self, im, points, scale_factor):
        points = points * scale_factor
        img_shape = im.shape[:2]
        points[:, 0::2] = np.clip(points[:, 0::2], 0, img_shape[1])
        points[:, 1::2] = np.clip(points[:, 1::2], 0, img_shape[0])
        return points


class SRResize(object):
    def __init__(
        self,
        imgH=32,
        imgW=128,
        down_sample_scale=4,
        keep_ratio=False,
        min_ratio=1,
        mask=False,
        infer_mode=False,
        **kwargs,
    ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.infer_mode = infer_mode

    def __call__(self, data):
        imgH = self.imgH
        imgW = self.imgW
        images_lr = data["image_lr"]
        transform2 = ResizeNormalize(
            (imgW // self.down_sample_scale, imgH // self.down_sample_scale)
        )
        images_lr = transform2(images_lr)
        data["img_lr"] = images_lr
        if self.infer_mode:
            return data

        images_HR = data["image_hr"]
        label_strs = data["label"]
        transform = ResizeNormalize((imgW, imgH))
        images_HR = transform(images_HR)
        data["img_hr"] = images_HR
        return data


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_numpy = np.array(img).astype("float32")
        img_numpy = img_numpy.transpose((2, 0, 1)) / 255
        return img_numpy


class GrayImageChannelFormat(object):
    """
    format gray scale image's channel: (3,h,w) -> (1,h,w)
    Args:
        inverse: inverse gray image
    """

    def __init__(self, inverse=False, **kwargs):
        self.inverse = inverse

    def __call__(self, data):
        img = data["image"]
        img_single_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_expanded = np.expand_dims(img_single_channel, 0)

        if self.inverse:
            data["image"] = np.abs(img_expanded - 1)
        else:
            data["image"] = img_expanded

        data["src_image"] = img
        return data


class RandomPerspective(object):
    """Random perspective transform for OCR detection training.

    Operates on data["image"] (H,W,C) and data["polys"] (N, P, 2).
    Should be placed after IaaAugment and before EastRandomCropData.
    """

    def __init__(
        self,
        prob=0.3,
        degrees=0.0,
        scale=0.2,
        shear=5.0,
        perspective=0.0002,
        fit_output=True,
        fill_value=(123.675, 116.28, 103.53),
        min_area_ratio=0.1,
        **kwargs,
    ):
        self.prob = prob
        self.degrees = degrees
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.fit_output = fit_output
        self.min_area_ratio = min_area_ratio
        if isinstance(fill_value, (int, float)):
            fill_value = (fill_value,) * 3
        self.fill_value = tuple(fill_value)

    def __call__(self, data):
        if random.random() > self.prob:
            return data

        im = data["image"]
        h, w = im.shape[:2]

        # Build perspective matrix
        M_core = self._get_core_matrix(h, w)

        # Compute output bounds
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(
            -1, 1, 2
        )
        warped_corners = cv2.perspectiveTransform(corners, M_core)
        x_min, y_min = warped_corners.min(axis=0).ravel()
        x_max, y_max = warped_corners.max(axis=0).ravel()

        if self.fit_output:
            new_w = int(np.ceil(x_max) - np.floor(x_min))
            new_h = int(np.ceil(y_max) - np.floor(y_min))
            T_fit = np.eye(3, dtype=np.float32)
            T_fit[0, 2] = -np.floor(x_min)
            T_fit[1, 2] = -np.floor(y_min)
            M = T_fit @ M_core
            target_size = (new_w, new_h)
        else:
            T_orig = np.eye(3, dtype=np.float32)
            T_orig[0, 2] = w / 2.0
            T_orig[1, 2] = h / 2.0
            M = T_orig @ M_core
            target_size = (w, h)

        # Warp image
        transformed_im = cv2.warpPerspective(
            im,
            M,
            target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.fill_value,
        )
        data["image"] = transformed_im

        # Transform polys
        polys = data["polys"]
        ignore_tags = data["ignore_tags"]
        texts = data["texts"]

        if len(polys) > 0:
            # polys: (N, P, 2) — flatten to (N*P, 1, 2) for perspectiveTransform
            n = len(polys)
            points_per_poly = (
                polys[0].shape[0] if hasattr(polys[0], "shape") else len(polys[0])
            )
            all_points = np.array(polys, dtype=np.float32).reshape(-1, 1, 2)

            warped_pts = cv2.perspectiveTransform(all_points, M)
            warped_pts = warped_pts.reshape(n, points_per_poly, 2)

            # Compute original areas for filtering
            orig_areas = np.array(
                [
                    cv2.contourArea(p.astype(np.float32))
                    for p in np.array(polys, dtype=np.float32)
                ]
            )
            new_areas = np.array(
                [cv2.contourArea(p.astype(np.float32)) for p in warped_pts]
            )

            tw, th = target_size
            # Filter: area ratio + center within bounds
            centers_x = warped_pts[:, :, 0].mean(axis=1)
            centers_y = warped_pts[:, :, 1].mean(axis=1)
            valid = (
                (new_areas > orig_areas * self.min_area_ratio)
                & (centers_x > 0)
                & (centers_x < tw)
                & (centers_y > 0)
                & (centers_y < th)
            )

            valid_ids = np.where(valid)[0]
            data["polys"] = warped_pts[valid_ids]
            data["ignore_tags"] = [ignore_tags[i] for i in valid_ids]
            data["texts"] = [texts[i] for i in valid_ids]

        return data

    def _get_core_matrix(self, h, w):
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        # Normalize perspective coefficients using 640px as reference size
        # to ensure consistent distortion ratio across different image sizes
        ref_size = 640.0
        max_dim = max(h, w)
        p_normalized = self.perspective * (ref_size / max_dim)

        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-p_normalized, p_normalized)
        P[2, 1] = random.uniform(-p_normalized, p_normalized)

        s = random.uniform(1 - self.scale, 1 + self.scale)
        a = random.uniform(-self.degrees, self.degrees)
        R = np.eye(3, dtype=np.float32)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        return S @ R @ P @ C
