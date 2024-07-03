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

import os
import cv2
import paddle
import random
import pyclipper
import numpy as np
from PIL import Image

import paddle.vision.transforms as transforms

from ppocr.utils.utility import check_install


class RandomScale:
    def __init__(self, short_size=640, **kwargs):
        self.short_size = short_size

    def scale_aligned(self, img, scale):
        oh, ow = img.shape[0:2]
        h = int(oh * scale + 0.5)
        w = int(ow * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        factor_h = h / oh
        factor_w = w / ow
        return img, factor_h, factor_w

    def __call__(self, data):
        img = data["image"]

        h, w = img.shape[0:2]
        random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        scale = (np.random.choice(random_scale) * self.short_size) / min(h, w)
        img, factor_h, factor_w = self.scale_aligned(img, scale)

        data["scale_factor"] = (factor_w, factor_h)
        data["image"] = img
        return data


class MakeShrink:
    def __init__(self, kernel_scale=0.7, **kwargs):
        self.kernel_scale = kernel_scale

    def dist(self, a, b):
        return np.linalg.norm((a - b), ord=2, axis=0)

    def perimeter(self, bbox):
        peri = 0.0
        for i in range(bbox.shape[0]):
            peri += self.dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
        return peri

    def shrink(self, bboxes, rate, max_shr=20):
        check_install("Polygon", "Polygon3")
        import Polygon as plg

        rate = rate * rate
        shrinked_bboxes = []
        for bbox in bboxes:
            area = plg.Polygon(bbox).area()
            peri = self.perimeter(bbox)

            try:
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

                shrinked_bbox = pco.Execute(-offset)
                if len(shrinked_bbox) == 0:
                    shrinked_bboxes.append(bbox)
                    continue

                shrinked_bbox = np.array(shrinked_bbox[0])
                if shrinked_bbox.shape[0] <= 2:
                    shrinked_bboxes.append(bbox)
                    continue

                shrinked_bboxes.append(shrinked_bbox)
            except Exception as e:
                shrinked_bboxes.append(bbox)

        return shrinked_bboxes

    def __call__(self, data):
        img = data["image"]
        bboxes = data["polys"]
        words = data["texts"]
        scale_factor = data["scale_factor"]

        gt_instance = np.zeros(img.shape[0:2], dtype="uint8")  # h,w
        training_mask = np.ones(img.shape[0:2], dtype="uint8")
        training_mask_distance = np.ones(img.shape[0:2], dtype="uint8")

        for i in range(len(bboxes)):
            bboxes[i] = np.reshape(
                bboxes[i]
                * ([scale_factor[0], scale_factor[1]] * (bboxes[i].shape[0] // 2)),
                (bboxes[i].shape[0] // 2, 2),
            ).astype("int32")

        for i in range(len(bboxes)):
            # different value for different bbox
            cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)

            # set training mask to 0
            cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

            # for not accurate annotation, use training_mask_distance
            if words[i] == "###" or words[i] == "???":
                cv2.drawContours(training_mask_distance, [bboxes[i]], -1, 0, -1)

        # make shrink
        gt_kernel_instance = np.zeros(img.shape[0:2], dtype="uint8")
        kernel_bboxes = self.shrink(bboxes, self.kernel_scale)
        for i in range(len(bboxes)):
            cv2.drawContours(gt_kernel_instance, [kernel_bboxes[i]], -1, i + 1, -1)

            # for training mask, kernel and background= 1, box region=0
            if words[i] != "###" and words[i] != "???":
                cv2.drawContours(training_mask, [kernel_bboxes[i]], -1, 1, -1)

        gt_kernel = gt_kernel_instance.copy()
        # for gt_kernel, kernel = 1
        gt_kernel[gt_kernel > 0] = 1

        # shrink 2 times
        tmp1 = gt_kernel_instance.copy()
        erode_kernel = np.ones((3, 3), np.uint8)
        tmp1 = cv2.erode(tmp1, erode_kernel, iterations=1)
        tmp2 = tmp1.copy()
        tmp2 = cv2.erode(tmp2, erode_kernel, iterations=1)

        # compute text region
        gt_kernel_inner = tmp1 - tmp2

        # gt_instance: text instance, bg=0, diff word use diff value
        # training_mask: text instance mask, word=0，kernel and bg=1
        # gt_kernel_instance: text kernel instance, bg=0, diff word use diff value
        # gt_kernel: text_kernel, bg=0，diff word use same value
        # gt_kernel_inner: text kernel reference
        # training_mask_distance: word without anno = 0, else 1

        data["image"] = [
            img,
            gt_instance,
            training_mask,
            gt_kernel_instance,
            gt_kernel,
            gt_kernel_inner,
            training_mask_distance,
        ]
        return data


class GroupRandomHorizontalFlip:
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        imgs = data["image"]

        if random.random() < self.p:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1).copy()
        data["image"] = imgs
        return data


class GroupRandomRotate:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        imgs = data["image"]

        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(
                img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST
            )
            imgs[i] = img_rotation

        data["image"] = imgs
        return data


class GroupRandomCropPadding:
    def __init__(self, target_size=(640, 640), **kwargs):
        self.target_size = target_size

    def __call__(self, data):
        imgs = data["image"]

        h, w = imgs[0].shape[0:2]
        t_w, t_h = self.target_size
        p_w, p_h = self.target_size
        if w == t_w and h == t_h:
            return data

        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w

        if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
            # make sure to crop the text region
            tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
            tl[tl < 0] = 0
            br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
            br[br < 0] = 0
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)

            i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            i = random.randint(0, h - t_h) if h - t_h > 0 else 0
            j = random.randint(0, w - t_w) if w - t_w > 0 else 0

        n_imgs = []
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                s3_length = int(imgs[idx].shape[-1])
                img = imgs[idx][i : i + t_h, j : j + t_w, :]
                img_p = cv2.copyMakeBorder(
                    img,
                    0,
                    p_h - t_h,
                    0,
                    p_w - t_w,
                    borderType=cv2.BORDER_CONSTANT,
                    value=tuple(0 for i in range(s3_length)),
                )
            else:
                img = imgs[idx][i : i + t_h, j : j + t_w]
                img_p = cv2.copyMakeBorder(
                    img,
                    0,
                    p_h - t_h,
                    0,
                    p_w - t_w,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0,),
                )
            n_imgs.append(img_p)

        data["image"] = n_imgs
        return data


class MakeCentripetalShift:
    def __init__(self, **kwargs):
        pass

    def jaccard(self, As, Bs):
        A = As.shape[0]  # small
        B = Bs.shape[0]  # large

        dis = np.sqrt(
            np.sum(
                (
                    As[:, np.newaxis, :].repeat(B, axis=1)
                    - Bs[np.newaxis, :, :].repeat(A, axis=0)
                )
                ** 2,
                axis=-1,
            )
        )

        ind = np.argmin(dis, axis=-1)

        return ind

    def __call__(self, data):
        imgs = data["image"]

        (
            img,
            gt_instance,
            training_mask,
            gt_kernel_instance,
            gt_kernel,
            gt_kernel_inner,
            training_mask_distance,
        ) = (imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5], imgs[6])

        max_instance = np.max(gt_instance)  # num bbox

        # make centripetal shift
        gt_distance = np.zeros((2, *img.shape[0:2]), dtype=np.float32)
        for i in range(1, max_instance + 1):
            # kernel_reference
            ind = gt_kernel_inner == i

            if np.sum(ind) == 0:
                training_mask[gt_instance == i] = 0
                training_mask_distance[gt_instance == i] = 0
                continue

            kpoints = (
                np.array(np.where(ind)).transpose((1, 0))[:, ::-1].astype("float32")
            )

            ind = (gt_instance == i) * (gt_kernel_instance == 0)
            if np.sum(ind) == 0:
                continue
            pixels = np.where(ind)

            points = np.array(pixels).transpose((1, 0))[:, ::-1].astype("float32")

            bbox_ind = self.jaccard(points, kpoints)

            offset_gt = kpoints[bbox_ind] - points

            gt_distance[:, pixels[0], pixels[1]] = offset_gt.T * 0.1

        img = Image.fromarray(img)
        img = img.convert("RGB")

        data["image"] = img
        data["gt_kernel"] = gt_kernel.astype("int64")
        data["training_mask"] = training_mask.astype("int64")
        data["gt_instance"] = gt_instance.astype("int64")
        data["gt_kernel_instance"] = gt_kernel_instance.astype("int64")
        data["training_mask_distance"] = training_mask_distance.astype("int64")
        data["gt_distance"] = gt_distance.astype("float32")

        return data


class ScaleAlignedShort:
    def __init__(self, short_size=640, **kwargs):
        self.short_size = short_size

    def __call__(self, data):
        img = data["image"]

        org_img_shape = img.shape

        h, w = img.shape[0:2]
        scale = self.short_size * 1.0 / min(h, w)
        h = int(h * scale + 0.5)
        w = int(w * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))

        new_img_shape = img.shape
        img_shape = np.array(org_img_shape + new_img_shape)

        data["shape"] = img_shape
        data["image"] = img

        return data
