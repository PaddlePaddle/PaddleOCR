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
import copy
import cv2
import random
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from ppocr.data.imaug.iaa_augment import IaaAugment
from ppocr.data.imaug.random_crop_data import is_poly_outside_rect


def get_rotate_crop_image(img, points):
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

class CopyPaste(object):
    def __init__(self, objects_paste_ratio=0.2, limit_paste=True,**kwargs):
        self.ext_data_num = 1
        self.objects_paste_ratio = objects_paste_ratio
        self.limit_paste = limit_paste
        augmenter_args = [{
            'type': 'Resize',
            'args': {
                'size': [0.5, 3]
            }
        }]
        self.aug = IaaAugment(augmenter_args)

    def __call__(self, data):
        src_img = data['image']
        src_polys = data['polys'].tolist()
        src_ignores = data['ignore_tags'].tolist()
        ext_data = data['ext_data'][0]
        ext_image = ext_data['image']
        ext_polys = ext_data['polys']
        ext_ignores = ext_data['ignore_tags']

        indexs = [i for i in range(len(ext_ignores)) if not ext_ignores[i]]
        select_num = max(1, min(int(self.objects_paste_ratio * len(ext_polys)), 30))

        random.shuffle(indexs)
        select_idxs = indexs[:select_num]
        select_polys = ext_polys[select_idxs]
        select_ignores = ext_ignores[select_idxs]

        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        ext_image = cv2.cvtColor(ext_image, cv2.COLOR_BGR2RGB)
        src_img = Image.fromarray(src_img).convert('RGBA')
        for poly, tag in zip(select_polys, select_ignores):
            box_img = get_rotate_crop_image(ext_image, poly)

            src_img, box = self.paste_img(src_img, box_img, src_polys)
            if box is not None:
                src_polys.append(box)
                src_ignores.append(tag)
        src_img = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
        h, w = src_img.shape[:2]
        src_polys = np.array(src_polys)
        src_polys[:, :, 0] = np.clip(src_polys[:, :, 0], 0, w)
        src_polys[:, :, 1] = np.clip(src_polys[:, :, 1], 0, h)
        data['image'] = src_img
        data['polys'] = src_polys
        data['ignore_tags'] = np.array(src_ignores)
        return data

    def paste_img(self, src_img, box_img, src_polys):
        box_img_pil = Image.fromarray(box_img).convert('RGBA')
        src_w, src_h = src_img.size
        box_w, box_h = box_img_pil.size
        if box_w > src_w or box_h > src_h:
            return src_img, None
        angle = np.random.randint(0, 360)
        box = np.array([[[0, 0], [box_w, 0], [box_w, box_h], [0, box_h]]])
        box = rotate_bbox(box_img, box, angle)[0]

        paste_x, paste_y = self.select_coord(src_polys, box, src_w - box_w, src_h - box_h)
        if paste_x is None:
            return src_img, None
        box[:, 0] += paste_x
        box[:, 1] += paste_y
        box_img_pil = box_img_pil.rotate(angle, expand=1)
        r, g, b, A = box_img_pil.split()
        src_img.paste(box_img_pil, (paste_x, paste_y), mask=A)

        return src_img, box

    def select_coord(self, src_polys, box, endx, endy):
        if self.limit_paste:
            xmin, ymin, xmax, ymax = box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()
            for _ in range(50):
                paste_x = random.randint(0, endx)
                paste_y = random.randint(0, endy)
                xmin1 = xmin + paste_x
                xmax1 = xmax + paste_x
                ymin1 = ymin + paste_y
                ymax1 = ymax + paste_y

                num_poly_in_rect = 0
                for poly in src_polys:
                    if not is_poly_outside_rect(poly, xmax1, ymin1, xmax1 - xmin1, ymax1 - ymin1):
                        num_poly_in_rect += 1
                        break
                if num_poly_in_rect == 0:
                    return paste_x, paste_y
            return None, None
        else:
            paste_x = random.randint(0, endx)
            paste_y = random.randint(0, endy)
            return paste_x, paste_y

def get_union(pD, pG):
    return Polygon(pD).union(Polygon(pG)).area

def get_intersection_over_union(pD, pG):
    return get_intersection(pD, pG) / get_union(pD, pG)

def get_intersection(pD, pG):
    return Polygon(pD).intersection(Polygon(pG)).area

def rotate_bbox(img, text_polys, angle, scale=1):
    """
    从给定的角度中选择一个角度，对图片和文本框进行旋转
    :param img: 图片
    :param text_polys: 文本框
    :param degrees: 角度，可以是一个数值或者list
    :param same_size: 是否保持和原图一样大
    :return: 旋转后的图片和角度
    """
    # ---------------------- 旋转图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]

    # 角度变弧度
    rangle = np.deg2rad(angle)
    # 计算旋转之后图像的w, h
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
    # 构造仿射矩阵
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # 计算原图中心点到新图中心点的偏移量
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # 更新仿射矩阵
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

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
    return np.array(rot_text_polys, dtype=np.float32)
