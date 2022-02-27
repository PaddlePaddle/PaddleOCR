# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
"""
This code is refer from:
https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/transforms.py
"""
import numpy as np
from PIL import Image, ImageDraw
import cv2
from shapely.geometry import Polygon
import math
from ppocr.utils.poly_nms import poly_intersection


class RandomScaling:
    def __init__(self, size=800, scale=(3. / 4, 5. / 2), **kwargs):
        """Random scale the image while keeping aspect.

        Args:
            size (int) : Base size before scaling.
            scale (tuple(float)) : The range of scaling.
        """
        assert isinstance(size, int)
        assert isinstance(scale, float) or isinstance(scale, tuple)
        self.size = size
        self.scale = scale if isinstance(scale, tuple) \
            else (1 - scale, 1 + scale)

    def __call__(self, data):
        image = data['image']
        text_polys = data['polys']
        h, w, _ = image.shape

        aspect_ratio = np.random.uniform(min(self.scale), max(self.scale))
        scales = self.size * 1.0 / max(h, w) * aspect_ratio
        scales = np.array([scales, scales])
        out_size = (int(h * scales[1]), int(w * scales[0]))
        image = cv2.resize(image, out_size[::-1])

        data['image'] = image
        text_polys[:, :, 0::2] = text_polys[:, :, 0::2] * scales[1]
        text_polys[:, :, 1::2] = text_polys[:, :, 1::2] * scales[0]
        data['polys'] = text_polys

        return data


class RandomCropFlip:
    def __init__(self,
                 pad_ratio=0.1,
                 crop_ratio=0.5,
                 iter_num=1,
                 min_area_ratio=0.2,
                 **kwargs):
        """Random crop and flip a patch of the image.

        Args:
            crop_ratio (float): The ratio of cropping.
            iter_num (int): Number of operations.
            min_area_ratio (float): Minimal area ratio between cropped patch
                and original image.
        """
        assert isinstance(crop_ratio, float)
        assert isinstance(iter_num, int)
        assert isinstance(min_area_ratio, float)

        self.pad_ratio = pad_ratio
        self.epsilon = 1e-2
        self.crop_ratio = crop_ratio
        self.iter_num = iter_num
        self.min_area_ratio = min_area_ratio

    def __call__(self, results):
        for i in range(self.iter_num):
            results = self.random_crop_flip(results)

        return results

    def random_crop_flip(self, results):
        image = results['image']
        polygons = results['polys']
        ignore_tags = results['ignore_tags']
        if len(polygons) == 0:
            return results

        if np.random.random() >= self.crop_ratio:
            return results

        h, w, _ = image.shape
        area = h * w
        pad_h = int(h * self.pad_ratio)
        pad_w = int(w * self.pad_ratio)
        h_axis, w_axis = self.generate_crop_target(image, polygons, pad_h,
                                                   pad_w)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return results

        attempt = 0
        while attempt < 50:
            attempt += 1
            polys_keep = []
            polys_new = []
            ignore_tags_keep = []
            ignore_tags_new = []
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin) * (ymax - ymin) < area * self.min_area_ratio:
                # area too small
                continue

            pts = np.stack([[xmin, xmax, xmax, xmin],
                            [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
            pp = Polygon(pts)
            fail_flag = False
            for polygon, ignore_tag in zip(polygons, ignore_tags):
                ppi = Polygon(polygon.reshape(-1, 2))
                ppiou, _ = poly_intersection(ppi, pp, buffer=0)
                if np.abs(ppiou - float(ppi.area)) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area)) < self.epsilon:
                    polys_new.append(polygon)
                    ignore_tags_new.append(ignore_tag)
                else:
                    polys_keep.append(polygon)
                    ignore_tags_keep.append(ignore_tag)

            if fail_flag:
                continue
            else:
                break

        cropped = image[ymin:ymax, xmin:xmax, :]
        select_type = np.random.randint(3)
        if select_type == 0:
            img = np.ascontiguousarray(cropped[:, ::-1])
        elif select_type == 1:
            img = np.ascontiguousarray(cropped[::-1, :])
        else:
            img = np.ascontiguousarray(cropped[::-1, ::-1])
        image[ymin:ymax, xmin:xmax, :] = img
        results['img'] = image

        if len(polys_new) != 0:
            height, width, _ = cropped.shape
            if select_type == 0:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    polys_new[idx] = poly
            elif select_type == 1:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = poly
            else:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = poly
            polygons = polys_keep + polys_new
            ignore_tags = ignore_tags_keep + ignore_tags_new
            results['polys'] = np.array(polygons)
            results['ignore_tags'] = ignore_tags

        return results

    def generate_crop_target(self, image, all_polys, pad_h, pad_w):
        """Generate crop target and make sure not to crop the polygon
        instances.

        Args:
            image (ndarray): The image waited to be crop.
            all_polys (list[list[ndarray]]): All polygons including ground
                truth polygons and ground truth ignored polygons.
            pad_h (int): Padding length of height.
            pad_w (int): Padding length of width.
        Returns:
            h_axis (ndarray): Vertical cropping range.
            w_axis (ndarray): Horizontal cropping range.
        """
        h, w, _ = image.shape
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

        text_polys = []
        for polygon in all_polys:
            rect = cv2.minAreaRect(polygon.astype(np.int32).reshape(-1, 2))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            text_polys.append([box[0], box[1], box[2], box[3]])

        polys = np.array(text_polys, dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w:maxx + pad_w] = 1
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        return h_axis, w_axis


class RandomCropPolyInstances:
    """Randomly crop images and make sure to contain at least one intact
    instance."""

    def __init__(self, crop_ratio=5.0 / 8.0, min_side_ratio=0.4, **kwargs):
        super().__init__()
        self.crop_ratio = crop_ratio
        self.min_side_ratio = min_side_ratio

    def sample_valid_start_end(self, valid_array, min_len, max_start, min_end):

        assert isinstance(min_len, int)
        assert len(valid_array) > min_len

        start_array = valid_array.copy()
        max_start = min(len(start_array) - min_len, max_start)
        start_array[max_start:] = 0
        start_array[0] = 1
        diff_array = np.hstack([0, start_array]) - np.hstack([start_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        start = np.random.randint(region_starts[region_ind],
                                  region_ends[region_ind])

        end_array = valid_array.copy()
        min_end = max(start + min_len, min_end)
        end_array[:min_end] = 0
        end_array[-1] = 1
        diff_array = np.hstack([0, end_array]) - np.hstack([end_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        end = np.random.randint(region_starts[region_ind],
                                region_ends[region_ind])
        return start, end

    def sample_crop_box(self, img_size, results):
        """Generate crop box and make sure not to crop the polygon instances.

        Args:
            img_size (tuple(int)): The image size (h, w).
            results (dict): The results dict.
        """

        assert isinstance(img_size, tuple)
        h, w = img_size[:2]

        key_masks = results['polys']

        x_valid_array = np.ones(w, dtype=np.int32)
        y_valid_array = np.ones(h, dtype=np.int32)

        selected_mask = key_masks[np.random.randint(0, len(key_masks))]
        selected_mask = selected_mask.reshape((-1, 2)).astype(np.int32)
        max_x_start = max(np.min(selected_mask[:, 0]) - 2, 0)
        min_x_end = min(np.max(selected_mask[:, 0]) + 3, w - 1)
        max_y_start = max(np.min(selected_mask[:, 1]) - 2, 0)
        min_y_end = min(np.max(selected_mask[:, 1]) + 3, h - 1)

        for mask in key_masks:
            mask = mask.reshape((-1, 2)).astype(np.int32)
            clip_x = np.clip(mask[:, 0], 0, w - 1)
            clip_y = np.clip(mask[:, 1], 0, h - 1)
            min_x, max_x = np.min(clip_x), np.max(clip_x)
            min_y, max_y = np.min(clip_y), np.max(clip_y)

            x_valid_array[min_x - 2:max_x + 3] = 0
            y_valid_array[min_y - 2:max_y + 3] = 0

        min_w = int(w * self.min_side_ratio)
        min_h = int(h * self.min_side_ratio)

        x1, x2 = self.sample_valid_start_end(x_valid_array, min_w, max_x_start,
                                             min_x_end)
        y1, y2 = self.sample_valid_start_end(y_valid_array, min_h, max_y_start,
                                             min_y_end)

        return np.array([x1, y1, x2, y2])

    def crop_img(self, img, bbox):
        assert img.ndim == 3
        h, w, _ = img.shape
        assert 0 <= bbox[1] < bbox[3] <= h
        assert 0 <= bbox[0] < bbox[2] <= w
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def __call__(self, results):
        image = results['image']
        polygons = results['polys']
        ignore_tags = results['ignore_tags']
        if len(polygons) < 1:
            return results

        if np.random.random_sample() < self.crop_ratio:

            crop_box = self.sample_crop_box(image.shape, results)
            img = self.crop_img(image, crop_box)
            results['image'] = img
            # crop and filter masks
            x1, y1, x2, y2 = crop_box
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            polygons[:, :, 0::2] = polygons[:, :, 0::2] - x1
            polygons[:, :, 1::2] = polygons[:, :, 1::2] - y1

            valid_masks_list = []
            valid_tags_list = []
            for ind, polygon in enumerate(polygons):
                if (polygon[:, ::2] > -4).all() and (
                        polygon[:, ::2] < w + 4).all() and (
                            polygon[:, 1::2] > -4).all() and (
                                polygon[:, 1::2] < h + 4).all():
                    polygon[:, ::2] = np.clip(polygon[:, ::2], 0, w)
                    polygon[:, 1::2] = np.clip(polygon[:, 1::2], 0, h)
                    valid_masks_list.append(polygon)
                    valid_tags_list.append(ignore_tags[ind])

            results['polys'] = np.array(valid_masks_list)
            results['ignore_tags'] = valid_tags_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class RandomRotatePolyInstances:
    def __init__(self,
                 rotate_ratio=0.5,
                 max_angle=10,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0),
                 **kwargs):
        """Randomly rotate images and polygon masks.

        Args:
            rotate_ratio (float): The ratio of samples to operate rotation.
            max_angle (int): The maximum rotation angle.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rotated image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def rotate(self, center, points, theta, center_shift=(0, 0)):
        # rotate points.
        (center_x, center_y) = center
        center_y = -center_y
        x, y = points[:, ::2], points[:, 1::2]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[:, ::2], points[:, 1::2] = _x, _y
        return points

    def cal_canvas_size(self, ori_size, degree):
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    def sample_angle(self, max_angle):
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    def rotate_img(self, img, angle, canvas_size):
        h, w = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
        rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)

        if self.pad_with_fixed_color:
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=self.pad_value)
        else:
            mask = np.zeros_like(img)
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            img_cut = cv2.resize(img_cut, (canvas_size[1], canvas_size[0]))

            mask = cv2.warpAffine(
                mask,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[1, 1, 1])
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[0, 0, 0])
            target_img = target_img + img_cut * mask

        return target_img

    def __call__(self, results):
        if np.random.random_sample() < self.rotate_ratio:
            image = results['image']
            polygons = results['polys']
            h, w = image.shape[:2]

            angle = self.sample_angle(self.max_angle)
            canvas_size = self.cal_canvas_size((h, w), angle)
            center_shift = (int((canvas_size[1] - w) / 2), int(
                (canvas_size[0] - h) / 2))
            image = self.rotate_img(image, angle, canvas_size)
            results['image'] = image
            # rotate polygons
            rotated_masks = []
            for mask in polygons:
                rotated_mask = self.rotate((w / 2, h / 2), mask, angle,
                                           center_shift)
                rotated_masks.append(rotated_mask)
            results['polys'] = np.array(rotated_masks)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class SquareResizePad:
    def __init__(self,
                 target_size,
                 pad_ratio=0.6,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0),
                 **kwargs):
        """Resize or pad images to be square shape.

        Args:
            target_size (int): The target size of square shaped image.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rescales image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        assert isinstance(target_size, int)
        assert isinstance(pad_ratio, float)
        assert isinstance(pad_with_fixed_color, bool)
        assert isinstance(pad_value, tuple)

        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def resize_img(self, img, keep_ratio=True):
        h, w, _ = img.shape
        if keep_ratio:
            t_h = self.target_size if h >= w else int(h * self.target_size / w)
            t_w = self.target_size if h <= w else int(w * self.target_size / h)
        else:
            t_h = t_w = self.target_size
        img = cv2.resize(img, (t_w, t_h))
        return img, (t_h, t_w)

    def square_pad(self, img):
        h, w = img.shape[:2]
        if h == w:
            return img, (0, 0)
        pad_size = max(h, w)
        if self.pad_with_fixed_color:
            expand_img = np.ones((pad_size, pad_size, 3), dtype=np.uint8)
            expand_img[:] = self.pad_value
        else:
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            expand_img = cv2.resize(img_cut, (pad_size, pad_size))
        if h > w:
            y0, x0 = 0, (h - w) // 2
        else:
            y0, x0 = (w - h) // 2, 0
        expand_img[y0:y0 + h, x0:x0 + w] = img
        offset = (x0, y0)

        return expand_img, offset

    def square_pad_mask(self, points, offset):
        x0, y0 = offset
        pad_points = points.copy()
        pad_points[::2] = pad_points[::2] + x0
        pad_points[1::2] = pad_points[1::2] + y0
        return pad_points

    def __call__(self, results):
        image = results['image']
        polygons = results['polys']
        h, w = image.shape[:2]

        if np.random.random_sample() < self.pad_ratio:
            image, out_size = self.resize_img(image, keep_ratio=True)
            image, offset = self.square_pad(image)
        else:
            image, out_size = self.resize_img(image, keep_ratio=False)
            offset = (0, 0)
        results['image'] = image
        try:
            polygons[:, :, 0::2] = polygons[:, :, 0::2] * out_size[
                1] / w + offset[0]
            polygons[:, :, 1::2] = polygons[:, :, 1::2] * out_size[
                0] / h + offset[1]
        except:
            pass
        results['polys'] = polygons

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
