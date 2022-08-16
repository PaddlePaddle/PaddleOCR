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

import numpy as np
from PIL import Image
#from torch.utils import data
import cv2
import random
#import torchvision.transforms as transforms
#import torch
import pyclipper
import Polygon as plg
import scipy.io as scio
#import mmcv
import os
import paddle
from paddle.io import Dataset
import paddle.vision.transforms as transforms

tt_root_dir = '/workspace/huangjun12/PaddleProject/OCR/CentripetalText/0.WellTrained/data/total_text/'
tt_train_data_dir = tt_root_dir + 'Images/Train/'
tt_train_gt_dir = tt_root_dir + 'Groundtruth/Polygon/Train/'
tt_test_data_dir = tt_root_dir + 'Images/Test/'
tt_test_gt_dir = tt_root_dir + 'Groundtruth/Polygon/Test/'


def scandir(dir_path, suffix=None):
    """Scan a directory to find the interested files.
    refer to:
    https://github.com/open-mmlab/mmcv/blob/f4167fe1e3d106cd641708d269ddbff568393437/mmcv/utils/path.py#L39
    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, str):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = os.path.relpath(entry.path, root)
                _rel_path = rel_path
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path

    return _scandir(dir_path, suffix)


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def read_mat_lindes(path):
    f = scio.loadmat(path)
    return f


def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    bboxes = []
    words = []
    data = read_mat_lindes(gt_path)

    data_polygt = data['polygt']
    for i, lines in enumerate(data_polygt):
        X = np.array(lines[1])
        Y = np.array(lines[3])

        point_num = len(X[0])
        word = lines[4]
        if len(word) == 0:
            word = '???'
        else:
            word = word[0]
            # word = word[0].encode("utf-8")

        if word == '#':
            word = '###'

        words.append(word)

        arr = np.concatenate([X, Y]).T  # 2,6 -> 6,2

        bbox = []  # x1,y1, x2,y2, ...
        for i in range(point_num):
            bbox.append(arr[i][0])
            bbox.append(arr[i][1])

        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * point_num)  # 数值除以长宽做归一化
        bboxes.append(bbox)

    # 列表，每个单词的位置[[x1,y1,x2,y2,..], [xx1,yy1,xx2,yy2,..]], [w1, w2, ...]
    return bboxes, words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(
            img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=640):
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    img = scale_aligned(img, scale)
    return img


def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

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
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(
                img,
                0,
                p_h - t_h,
                0,
                p_w - t_w,
                borderType=cv2.BORDER_CONSTANT,
                value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(
                img,
                0,
                p_h - t_h,
                0,
                p_w - t_w,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, ))
        n_imgs.append(img_p)
    return n_imgs


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

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
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def jaccard(As, Bs):

    A = As.shape[0]  # 数量大
    B = Bs.shape[0]  # 数量少
    #print(As.shape)
    #print(As[:, np.newaxis, :].repeat(B, axis=1).shape)
    dis = np.sqrt(
        np.sum((As[:, np.newaxis, :].repeat(
            B, axis=1) - Bs[np.newaxis, :, :].repeat(
                A, axis=0))**2,
               axis=-1))

    ind = np.argmin(dis, axis=-1)
    #print(np.min(dis), ind.shape, ind)
    #exit()

    return ind


class CentripetalDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(CentripetalDataSet, self).__init__()
        self.need_reset = False
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        data_dir = dataset_config['data_dir']
        gt_dir = dataset_config['gt_dir']
        self.is_transform = dataset_config['is_transform']
        img_size = dataset_config['img_size']
        self.img_size = img_size if (img_size is None or isinstance(
            img_size, tuple)) else (img_size, img_size)

        self.do_shuffle = loader_config['shuffle']
        self.seed = seed

        self.kernel_scale = dataset_config['kernel_scale']
        self.short_size = dataset_config['short_size']
        self.read_type = dataset_config['read_type']

        # if mode == 'train':
        #     data_dirs = [tt_train_data_dir]
        #     gt_dirs = [tt_train_gt_dir]
        # elif mode == 'eval':
        #     data_dirs = [tt_test_data_dir]
        #     gt_dirs = [tt_test_gt_dir]
        # else:
        #     raise ValueError('Error: mode must be train or test!')

        self.img_paths = []
        self.gt_paths = []

        # 扫描文件夹获取所有图片并打乱 1255张
        img_names = [img_name for img_name in scandir(data_dir, '.jpg')]
        img_names.extend(
            [img_name
             for img_name in scandir(data_dir, '.png')])  # 没有png格式，追加结果为空

        img_paths = []
        gt_paths = []
        for idx, img_name in enumerate(img_names):
            img_path = data_dir + img_name
            img_paths.append(img_path)

            gt_name = 'poly_gt_' + img_name.split('.')[
                0] + '.mat'  #poly_gt_img11.mat 命名规则
            gt_path = gt_dir + gt_name
            gt_paths.append(gt_path)

        self.img_paths.extend(img_paths)
        self.gt_paths.extend(gt_paths)

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        # cv2读取图片
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann(img, gt_path)

        if self.is_transform:
            # 随机缩放
            img = random_scale(img, self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')  # h,w
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        training_mask_distance = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                #根据scale之后的size，缩放回来，同时reshape成[6,2]
                bboxes[i] = np.reshape(
                    bboxes[i] * ([img.shape[1], img.shape[0]] *
                                 (bboxes[i].shape[0] // 2)),
                    (bboxes[i].shape[0] // 2, 2)).astype('int32')
            for i in range(len(bboxes)):
                #一张图片，不同的字框，填充值不一样
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1,
                                 -1)  # 原图，轮廓list框，所有轮廓，颜色，填充模式
                # 上一行的i+1改成(255,255,255)可以显示
                #cv2.imwrite('./gt_distance.png', gt_instance)

                #一张图片，不同的字框，training mask统一填充为0，作为mask
                cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
                #cv2.imwrite('./gt_distance.png', training_mask)

                if words[i] == '###' or words[
                        i] == '???':  #没有精确标注的字符，用training_mask_distance保存
                    cv2.drawContours(training_mask_distance, [bboxes[i]], -1, 0,
                                     -1)

            # print(set(list(gt_instance.flatten())))
            # print(set(list(training_mask.flatten())))

        gt_kernel_instance = np.zeros(img.shape[0:2], dtype='uint8')
        kernel_bboxes = shrink(
            bboxes, self.kernel_scale)  # kernel bbox=bbox按kernel scale收缩得到
        for i in range(len(bboxes)):
            cv2.drawContours(gt_kernel_instance, [kernel_bboxes[i]], -1, i + 1,
                             -1)
            #cv2.imwrite('./gt_distance.png', gt_instance)

            # 对于training mask，字符框填充为0，kernel框和背景填充为1
            if words[i] != '###' and words[i] != '???':
                cv2.drawContours(training_mask, [kernel_bboxes[i]], -1, 1, -1)

        gt_kernel = gt_kernel_instance.copy()
        gt_kernel[gt_kernel > 0] = 1  #所有大于0的位置设置为1，即字符kernel的位置设为1

        # 腐蚀2次
        tmp1 = gt_kernel_instance.copy()
        #cv2.imwrite('./gt_distance.png', tmp1*64)
        erode_kernel = np.ones((3, 3), np.uint8)
        tmp1 = cv2.erode(tmp1, erode_kernel, iterations=1)
        #cv2.imwrite('./gt_distance.png', tmp1*64)
        tmp2 = tmp1.copy()
        tmp2 = cv2.erode(tmp2, erode_kernel, iterations=1)
        #cv2.imwrite('./gt_distance.png', tmp1*64)
        # kernel腐蚀2次的差，是kernel的边界？
        gt_kernel_inner = tmp1 - tmp2
        #cv2.imwrite('./gt_distance.png', gt_kernel_inner*64)

        # gt_instance: 单词框的mask，背景为0，不同单词的框填充值不一样
        # training_mask: 单词框填充为0，kernel框和背景填充为1
        # gt_kernel_instance: 单词框收缩得到的kernel，背景为0，不同单词的kernel框填充值不一样
        # gt_kernel: 同gt_kernel_instance，背景为0，但不同单词的kernel框填充值均为1，不区分instance
        # gt_kernel_inner: 单词gt_kernel_instance腐蚀2次的差，类似kernel的内边界？
        # training_mask_distance: 特殊单词或未标注单词填充为0，其余为1

        if self.is_transform:
            imgs = [
                img, gt_instance, training_mask, gt_kernel_instance, gt_kernel,
                gt_kernel_inner, training_mask_distance
            ]

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernel_instance, gt_kernel, gt_kernel_inner, training_mask_distance = \
                imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5], imgs[6]

        max_instance = np.max(gt_instance)  # 4 单词数量

        # 算centripetal shift
        gt_distance = np.zeros(
            (2, *img.shape[0:2]), dtype=np.float32)  # 似乎在算内边界和外边界的距离
        for i in range(1, max_instance + 1):  # 对每一个kernel实例
            # 对不同单词的框
            ind = (gt_kernel_inner == i)  # kernrl内边界 kernel_reference
            # 如果有单词没有腐蚀边界，单词太小，腐蚀2次之后没了？类似于没有kernel
            # 则将training_mask该单词框全部填充为0，因为没有kernel
            # training_mask_distance也将这个单词框填充为0
            if np.sum(ind) == 0:
                training_mask[gt_instance == i] = 0
                training_mask_distance[gt_instance == i] = 0
                continue

            # 取gt_kernel_inner的位置，kernel边界的位置索引 [540,2] 点的个数，y,x坐标
            kpoints = np.array(np.where(ind)).transpose(
                (1, 0))[:, ::-1].astype('float32')

            # 是单词框的位置，但不是kernel框的位置，就是trainingMask的框？
            ind = (gt_instance == i) * (gt_kernel_instance == 0)  #外边界的位置
            if np.sum(ind) == 0:  # 这个判断好像不太可能触发？
                continue
            pixels = np.where(ind)

            points = np.array(pixels).transpose(
                (1, 0))[:, ::-1].astype('float32')

            # points: 是单词框的位置，但不是kernel框的位置，外边界
            # kpoints: kernel框边界位置，内边界

            bbox_ind = jaccard(points, kpoints)
            #print(bbox_ind.shape, kpoints.shape, points.shape)
            #print(bbox_ind)
            offset_gt = kpoints[bbox_ind] - points
            #print(offset_gt.shape, gt_distance.shape, pixels[1].shape, offset_gt.T.shape)

            gt_distance[:, pixels[0], pixels[
                1]] = offset_gt.T * 0.1  #2维：x的shift和y的shift

            #cv2.imwrite('./gt_distance.png', (64*gt_distance[0]).astype('uint8'))

        #img_name = img_path.split('/')[-1].split('.')[0]
        #self.vis(img_name, img, gt_instance, training_mask, gt_kernel, gt_kernel_instance, gt_kernel_inner, gt_distance)

        img = Image.fromarray(img)
        img = img.convert('RGB')

        if self.is_transform:
            img = transforms.ColorJitter(
                brightness=32.0 / 255, saturation=0.5)(img)

        img = transforms.ToTensor()(img)
        img = transforms.normalize(
            img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #(img)
        gt_kernel = gt_kernel.astype("int64")
        training_mask = training_mask.astype("int64")
        gt_instance = gt_instance.astype("int64")
        gt_kernel_instance = gt_kernel_instance.astype("int64")
        training_mask_distance = training_mask_distance.astype("int64")
        gt_distance = gt_distance.astype("float32")

        # data = dict(
        #     imgs=img,
        #     gt_kernels=gt_kernel,
        #     training_masks=training_mask,
        #     gt_instances=gt_instance,
        #     gt_kernel_instances=gt_kernel_instance,
        #     training_mask_distances=training_mask_distance,
        #     gt_distances=gt_distance
        # )

        return img, gt_kernel, training_mask, gt_instance, gt_kernel_instance, \
        training_mask_distance, gt_distance

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path, self.read_type)
        img_meta = dict(ori_img=img, org_img_size=np.array(img.shape[:2]))

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(
            dict(
                img_size=np.array(img.shape[:2]),
                imgs=img,
                img_name=img_path.split('/')[-1].split('.')[0]))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.normalize(
            img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #(img)

        data = dict(imgs=img, img_metas=img_meta)

        return img, img_meta, img_path

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.prepare_train_data(idx)
        elif self.mode == 'eval':
            return self.prepare_test_data(idx)

    def __len__(self):
        return len(self.img_paths)
