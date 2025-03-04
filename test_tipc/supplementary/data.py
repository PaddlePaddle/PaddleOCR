import numpy as np
import paddle
import os
import cv2
import glob


def transform(data, ops=None):
    """transform"""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), "operator config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


class DecodeImage(object):
    """decode image"""

    def __init__(self, img_mode="RGB", channel_first=False, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data["image"]
        assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype="uint8")
        img = cv2.imdecode(img, 1)
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
        data["src_image"] = img
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

        src_img = data["src_image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            src_img = np.array(src_img)
        data["src_image"] = img.transpose((2, 0, 1))

        return data


class SimpleDataset(nn.Dataset):
    def __init__(self, config, mode, logger, seed=None):
        self.logger = logger
        self.mode = mode.lower()

        data_dir = config["Train"]["data_dir"]

        imgs_list = self.get_image_list(data_dir)

        self.ops = create_operators(cfg["transforms"], None)

    def get_image_list(self, img_dir):
        imgs = glob.glob(os.path.join(img_dir, "*.png"))
        if len(imgs) == 0:
            raise ValueError(f"not any images founded in {img_dir}")
        return imgs

    def __getitem__(self, idx):
        return None
