#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
This code is refer from: 
https://github.com/songdejia/EAST/blob/master/data_utils.py
"""
import math
import cv2
import numpy as np
import json
import sys
import os

__all__ = ['EASTProcessTrain']


class EASTProcessTrain(object):
    def __init__(self,
                 image_shape=[512, 512],
                 background_ratio=0.125,
                 min_crop_side_ratio=0.1,
                 min_text_size=10,
                 **kwargs):
        self.input_size = image_shape[1]
        self.random_scale = np.array([0.5, 1, 2.0, 3.0])
        self.background_ratio = background_ratio
        self.min_crop_side_ratio = min_crop_side_ratio
        self.min_text_size = min_text_size

    def preprocess(self, im):
        input_size = self.input_size
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(input_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale)
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        # im = im[:, :, ::-1].astype(np.float32)
        im = im / 255
        im -= img_mean
        im /= img_std
        new_h, new_w, _ = im.shape
        im_padded = np.zeros((input_size, input_size, 3), dtype=np.float32)
        im_padded[:new_h, :new_w, :] = im
        im_padded = im_padded.transpose((2, 0, 1))
        im_padded = im_padded[np.newaxis, :]
        return im_padded, im_scale

    def rotate_im_poly(self, im, text_polys):
        """
        rotate image with 90 / 180 / 270 degre
        """
        im_w, im_h = im.shape[1], im.shape[0]
        dst_im = im.copy()
        dst_polys = []
        rand_degree_ratio = np.random.rand()
        rand_degree_cnt = 1
        if 0.333 < rand_degree_ratio < 0.666:
            rand_degree_cnt = 2
        elif rand_degree_ratio > 0.666:
            rand_degree_cnt = 3
        for i in range(rand_degree_cnt):
            dst_im = np.rot90(dst_im)
        rot_degree = -90 * rand_degree_cnt
        rot_angle = rot_degree * math.pi / 180.0
        n_poly = text_polys.shape[0]
        cx, cy = 0.5 * im_w, 0.5 * im_h
        ncx, ncy = 0.5 * dst_im.shape[1], 0.5 * dst_im.shape[0]
        for i in range(n_poly):
            wordBB = text_polys[i]
            poly = []
            for j in range(4):
                sx, sy = wordBB[j][0], wordBB[j][1]
                dx = math.cos(rot_angle) * (sx - cx)\
                    - math.sin(rot_angle) * (sy - cy) + ncx
                dy = math.sin(rot_angle) * (sx - cx)\
                    + math.cos(rot_angle) * (sy - cy) + ncy
                poly.append([dx, dy])
            dst_polys.append(poly)
        dst_polys = np.array(dst_polys, dtype=np.float32)
        return dst_im, dst_polys

    def polygon_area(self, poly):
        """
        compute area of a polygon
        :param poly:
        :return:
        """
        edge = [(poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
                (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
                (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
                (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])]
        return np.sum(edge) / 2.

    def check_and_validate_polys(self, polys, tags, img_height, img_width):
        """
        check so that the text poly is in the same direction,
        and also filter some invalid polygons
        :param polys:
        :param tags:
        :return:
        """
        h, w = img_height, img_width
        if polys.shape[0] == 0:
            return polys
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

        validated_polys = []
        validated_tags = []
        for poly, tag in zip(polys, tags):
            p_area = self.polygon_area(poly)
            #invalid poly
            if abs(p_area) < 1:
                continue
            if p_area > 0:
                #'poly in wrong direction'
                if not tag:
                    tag = True  #reversed cases should be ignore
                poly = poly[(0, 3, 2, 1), :]
            validated_polys.append(poly)
            validated_tags.append(tag)
        return np.array(validated_polys), np.array(validated_tags)

    def draw_img_polys(self, img, polys):
        if len(img.shape) == 4:
            img = np.squeeze(img, axis=0)
        if img.shape[0] == 3:
            img = img.transpose((1, 2, 0))
            img[:, :, 2] += 123.68
            img[:, :, 1] += 116.78
            img[:, :, 0] += 103.94
        cv2.imwrite("tmp.jpg", img)
        img = cv2.imread("tmp.jpg")
        for box in polys:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
        import random
        ino = random.randint(0, 100)
        cv2.imwrite("tmp_%d.jpg" % ino, img)
        return

    def shrink_poly(self, poly, r):
        """
        fit a poly inside the origin poly, maybe bugs here...
        used for generate the score map
        :param poly: the text poly
        :param r: r in the paper
        :return: the shrinked poly
        """
        # shrink ratio
        R = 0.3
        # find the longer pair
        dist0 = np.linalg.norm(poly[0] - poly[1])
        dist1 = np.linalg.norm(poly[2] - poly[3])
        dist2 = np.linalg.norm(poly[0] - poly[3])
        dist3 = np.linalg.norm(poly[1] - poly[2])
        if dist0 + dist1 > dist2 + dist3:
            # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]),
                               (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]),
                               (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]),
                               (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]),
                               (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
        else:
            ## p0, p3
            # print poly
            theta = np.arctan2((poly[3][0] - poly[0][0]),
                               (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]),
                               (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]),
                               (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]),
                               (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
        return poly

    def generate_quad(self, im_size, polys, tags):
        """
        Generate quadrangle.
        """
        h, w = im_size
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        # (x1, y1, ..., x4, y4, short_edge_norm)
        geo_map = np.zeros((h, w, 9), dtype=np.float32)
        # mask used during traning, to ignore some hard areas
        training_mask = np.ones((h, w), dtype=np.uint8)
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]

            r = [None, None, None, None]
            for i in range(4):
                dist1 = np.linalg.norm(poly[i] - poly[(i + 1) % 4])
                dist2 = np.linalg.norm(poly[i] - poly[(i - 1) % 4])
                r[i] = min(dist1, dist2)
            # score map
            shrinked_poly = self.shrink_poly(
                poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
            cv2.fillPoly(score_map, shrinked_poly, 1)
            cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
            # if the poly is too small, then ignore it during training
            poly_h = min(
                np.linalg.norm(poly[0] - poly[3]),
                np.linalg.norm(poly[1] - poly[2]))
            poly_w = min(
                np.linalg.norm(poly[0] - poly[1]),
                np.linalg.norm(poly[2] - poly[3]))
            if min(poly_h, poly_w) < self.min_text_size:
                cv2.fillPoly(training_mask,
                             poly.astype(np.int32)[np.newaxis, :, :], 0)

            if tag:
                cv2.fillPoly(training_mask,
                             poly.astype(np.int32)[np.newaxis, :, :], 0)

            xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
            # geo map.
            y_in_poly = xy_in_poly[:, 0]
            x_in_poly = xy_in_poly[:, 1]
            poly[:, 0] = np.minimum(np.maximum(poly[:, 0], 0), w)
            poly[:, 1] = np.minimum(np.maximum(poly[:, 1], 0), h)
            for pno in range(4):
                geo_channel_beg = pno * 2
                geo_map[y_in_poly, x_in_poly, geo_channel_beg] =\
                    x_in_poly - poly[pno, 0]
                geo_map[y_in_poly, x_in_poly, geo_channel_beg+1] =\
                    y_in_poly - poly[pno, 1]
            geo_map[y_in_poly, x_in_poly, 8] = \
                1.0 / max(min(poly_h, poly_w), 1.0)
        return score_map, geo_map, training_mask

    def crop_area(self, im, polys, tags, crop_background=False, max_tries=50):
        """
        make random crop from the input image
        :param im:
        :param polys:
        :param tags:
        :param crop_background:
        :param max_tries:
        :return:
        """
        h, w, _ = im.shape
        pad_h = h // 10
        pad_w = w // 10
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w:maxx + pad_w] = 1
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            return im, polys, tags

        for i in range(max_tries):
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
            if xmax - xmin < self.min_crop_side_ratio * w or \
               ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            if polys.shape[0] != 0:
                poly_axis_in_area = (polys[:, :, 0] >= xmin)\
                    & (polys[:, :, 0] <= xmax)\
                    & (polys[:, :, 1] >= ymin)\
                    & (polys[:, :, 1] <= ymax)
                selected_polys = np.where(
                    np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polys = []

            if len(selected_polys) == 0:
                # no text in this area
                if crop_background:
                    im = im[ymin:ymax + 1, xmin:xmax + 1, :]
                    polys = []
                    tags = []
                    return im, polys, tags
                else:
                    continue

            im = im[ymin:ymax + 1, xmin:xmax + 1, :]
            polys = polys[selected_polys]
            tags = tags[selected_polys]
            polys[:, :, 0] -= xmin
            polys[:, :, 1] -= ymin
            return im, polys, tags
        return im, polys, tags

    def crop_background_infor(self, im, text_polys, text_tags):
        im, text_polys, text_tags = self.crop_area(
            im, text_polys, text_tags, crop_background=True)

        if len(text_polys) > 0:
            return None
        # pad and resize image
        input_size = self.input_size
        im, ratio = self.preprocess(im)
        score_map = np.zeros((input_size, input_size), dtype=np.float32)
        geo_map = np.zeros((input_size, input_size, 9), dtype=np.float32)
        training_mask = np.ones((input_size, input_size), dtype=np.float32)
        return im, score_map, geo_map, training_mask

    def crop_foreground_infor(self, im, text_polys, text_tags):
        im, text_polys, text_tags = self.crop_area(
            im, text_polys, text_tags, crop_background=False)

        if text_polys.shape[0] == 0:
            return None
        #continue for all ignore case
        if np.sum((text_tags * 1.0)) >= text_tags.size:
            return None
        # pad and resize image
        input_size = self.input_size
        im, ratio = self.preprocess(im)
        text_polys[:, :, 0] *= ratio
        text_polys[:, :, 1] *= ratio
        _, _, new_h, new_w = im.shape
        #         print(im.shape)
        #         self.draw_img_polys(im, text_polys)
        score_map, geo_map, training_mask = self.generate_quad(
            (new_h, new_w), text_polys, text_tags)
        return im, score_map, geo_map, training_mask

    def __call__(self, data):
        im = data['image']
        text_polys = data['polys']
        text_tags = data['ignore_tags']
        if im is None:
            return None
        if text_polys.shape[0] == 0:
            return None

        #add rotate cases
        if np.random.rand() < 0.5:
            im, text_polys = self.rotate_im_poly(im, text_polys)
        h, w, _ = im.shape
        text_polys, text_tags = self.check_and_validate_polys(text_polys,
                                                              text_tags, h, w)
        if text_polys.shape[0] == 0:
            return None

        # random scale this image
        rd_scale = np.random.choice(self.random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale
        if np.random.rand() < self.background_ratio:
            outs = self.crop_background_infor(im, text_polys, text_tags)
        else:
            outs = self.crop_foreground_infor(im, text_polys, text_tags)

        if outs is None:
            return None
        im, score_map, geo_map, training_mask = outs
        score_map = score_map[np.newaxis, ::4, ::4].astype(np.float32)
        geo_map = np.swapaxes(geo_map, 1, 2)
        geo_map = np.swapaxes(geo_map, 1, 0)
        geo_map = geo_map[:, ::4, ::4].astype(np.float32)
        training_mask = training_mask[np.newaxis, ::4, ::4]
        training_mask = training_mask.astype(np.float32)

        data['image'] = im[0]
        data['score_map'] = score_map
        data['geo_map'] = geo_map
        data['training_mask'] = training_mask
        return data
