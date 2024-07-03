# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

import math
import numbers
import random

import cv2
import numpy as np
from skimage.util import random_noise


class RandomNoise:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """
        对图片加噪声
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        data["img"] = (
            random_noise(data["img"], mode="gaussian", clip=True) * 255
        ).astype(data["img"].dtype)
        return data


class RandomScale:
    def __init__(self, scales, random_rate):
        """
        :param scales: 尺度
        :param ramdon_rate: 随机系数
        :return:
        """
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data["img"]
        text_polys = data["text_polys"]

        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(self.scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale

        data["img"] = im
        data["text_polys"] = tmp_text_polys
        return data


class RandomRotateImgBox:
    def __init__(self, degrees, random_rate, same_size=False):
        """
        :param degrees: 角度，可以是一个数值或者list
        :param ramdon_rate: 随机系数
        :param same_size: 是否保持和原图一样大
        :return:
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif (
            isinstance(degrees, list)
            or isinstance(degrees, tuple)
            or isinstance(degrees, np.ndarray)
        ):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception("degrees must in Number or list or tuple or np.ndarray")
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data["img"]
        text_polys = data["text_polys"]

        # ---------------------- 旋转图像 ----------------------
        w = im.shape[1]
        h = im.shape[0]
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        if self.same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)
            nh = abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(
            im,
            rot_mat,
            (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4,
        )

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        data["img"] = rot_img
        data["text_polys"] = np.array(rot_text_polys)
        return data


class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        """
        :param input_size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :param ramdon_rate: 随机系数
        :param keep_ratio: 是否保持长宽比
        :return:
        """
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError(
                    "If input_size is a single number, it must be positive."
                )
            size = (size, size)
        elif (
            isinstance(size, list)
            or isinstance(size, tuple)
            or isinstance(size, np.ndarray)
        ):
            if len(size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise Exception("input_size must in Number or list or tuple or np.ndarray")
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data["img"]
        text_polys = data["text_polys"]

        if self.keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale

        data["img"] = im
        data["text_polys"] = text_polys
        return data


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img, (new_width / width, new_height / height)


class ResizeShortSize:
    def __init__(self, short_size, resize_text_polys=True):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data["img"]
        text_polys = data["text_polys"]

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 保证短边 >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            # im, scale = resize_image(im, self.short_size)
            if self.resize_text_polys:
                # text_polys *= scale
                text_polys[:, 0] *= scale[0]
                text_polys[:, 1] *= scale[1]

        data["img"] = im
        data["text_polys"] = text_polys
        return data


class HorizontalFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data["img"]
        text_polys = data["text_polys"]

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]

        data["img"] = flip_im
        data["text_polys"] = flip_text_polys
        return data


class VerticallFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data["img"]
        text_polys = data["text_polys"]

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        data["img"] = flip_im
        data["text_polys"] = flip_text_polys
        return data
