# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:59
# @Author  : zhoujun
import json
import pathlib
import time
import os
import glob
import cv2
import yaml
from typing import Mapping
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser, RawDescriptionHelpFormatter


def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def setup_logger(log_file_path: str=None):
    import logging
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('DBNet.paddle')
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back

    return newFunc


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [
            x.strip().strip('\ufeff').strip('\xef\xbb\xbf')
            for x in f.readlines()
        ]
    return content


def _load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _save_txt, '.json': _save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def _save_txt(data, file_path):
    """
    将一个list的数组写入txt文件里
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def _save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def show_img(imgs: np.ndarray, title='img'):
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


def cal_text_score(texts,
                   gt_texts,
                   training_masks,
                   running_metric_text,
                   thred=0.5):
    training_masks = training_masks.numpy()
    pred_text = texts.numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def order_points_clockwise_list(pts):
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts


def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    if img_path.exists() and img_path.stat(
                    ).st_size > 0 and label_path.exists() and label_path.stat(
                    ).st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
    return train_data


def save_result(result_path, box_list, score_list, is_output_polygon):
    if is_output_polygon:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = score_list[i]
                res.write(result + ',' + str(score) + "\n")
    else:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                score = score_list[i]
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                res.write(result + ',' + str(score) + "\n")


def expand_polygon(polygon):
    """
    对只有一个字符的框进行扩充
    """
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)


def _merge_dict(config, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    Args:
        config: dict onto which the merge is executed
        merge_dct: dct merged into config
    Returns: dct
    """
    for key, value in merge_dct.items():
        sub_keys = key.split('.')
        key = sub_keys[0]
        if key in config and len(sub_keys) > 1:
            _merge_dict(config[key], {'.'.join(sub_keys[1:]): value})
        elif key in config and isinstance(config[key], dict) and isinstance(
                value, Mapping):
            _merge_dict(config[key], value)
        else:
            config[key] = value
    return config


def print_dict(cfg, print_func=print, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(cfg.items()):
        if isinstance(v, dict):
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, print_func, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, print_func, delimiter + 4)
        else:
            print_func("{}{} : {}".format(delimiter * " ", k, v))


class Config(object):
    def __init__(self, config_path, BASE_KEY='base'):
        self.BASE_KEY = BASE_KEY
        self.cfg = self._load_config_with_base(config_path)

    def _load_config_with_base(self, file_path):
        """
               Load config from file.
               Args:
                   file_path (str): Path of the config file to be loaded.
               Returns: global config
               """
        _, ext = os.path.splitext(file_path)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"

        with open(file_path) as f:
            file_cfg = yaml.load(f, Loader=yaml.Loader)

        # NOTE: cfgs outside have higher priority than cfgs in _BASE_
        if self.BASE_KEY in file_cfg:
            all_base_cfg = dict()
            base_ymls = list(file_cfg[self.BASE_KEY])
            for base_yml in base_ymls:
                with open(base_yml) as f:
                    base_cfg = self._load_config_with_base(base_yml)
                    all_base_cfg = _merge_dict(all_base_cfg, base_cfg)

            del file_cfg[self.BASE_KEY]
            file_cfg = _merge_dict(all_base_cfg, file_cfg)
        file_cfg['filename'] = os.path.splitext(os.path.split(file_path)[-1])[0]
        return file_cfg

    def merge_dict(self, args):
        self.cfg = _merge_dict(self.cfg, args)

    def print_cfg(self, print_func=print):
        """
        Recursively visualize a dict and
        indenting acrrording by the relationship of keys.
        """
        print_func('----------- Config -----------')
        print_dict(self.cfg, print_func)
        print_func('---------------------------------------------')

    def save(self, p):
        with open(p, 'w') as f:
            yaml.dump(
                dict(self.cfg), f, default_flow_style=False, sort_keys=False)


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-c", "--config_file", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config_file is not None, \
            "Please specify --config_file=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config


if __name__ == '__main__':
    img = np.zeros((1, 3, 640, 640))
    show_img(img[0][0])
    plt.show()
