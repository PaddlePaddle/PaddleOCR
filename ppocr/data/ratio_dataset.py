# copyright (c) 2025 PaddlePaddle Authors. All Rights Reserve.
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

import io
import math
import random
import cv2
import lmdb
import numpy as np
from PIL import Image
from paddle.io import Dataset

from .imaug import create_operators, transform


class RatioDataSet(Dataset):

    def __init__(self, config, mode, logger, seed=None):
        super(RatioDataSet, self).__init__()
        self.ds_width = config[mode]["dataset"].get("ds_width", True)
        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        data_dir_list = dataset_config["data_dir_list"]
        self.padding = dataset_config.get("padding", True)
        self.padding_rand = dataset_config.get("padding_rand", False)
        self.padding_doub = dataset_config.get("padding_doub", False)
        max_ratio = dataset_config.get("max_ratio", 12)
        min_ratio = dataset_config.get("min_ratio", 1)
        self.do_shuffle = loader_config["shuffle"]
        data_source_num = len(data_dir_list)
        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert (
            len(ratio_list) == data_source_num
        ), "The length of ratio_list should be the same as the file_list."
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir_list, ratio_list)
        for data_dir in data_dir_list:
            logger.info("Initialize indexs of datasets:%s" % data_dir)
        self.logger = logger
        self.data_idx_order_list = self.dataset_traversal()
        wh_ratio = np.around(np.array(self.get_wh_ratio()))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.ops = create_operators(dataset_config["transforms"], global_config)

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0
        self.base_shape = dataset_config.get(
            "base_shape", [[64, 64], [96, 48], [112, 40], [128, 32]]
        )
        self.base_h = 32

    def get_wh_ratio(self):
        wh_ratio = []
        for idx in range(self.data_idx_order_list.shape[0]):
            lmdb_idx, file_idx = self.data_idx_order_list[idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            wh_key = "wh-%09d".encode() % file_idx
            wh = self.lmdb_sets[lmdb_idx]["txn"].get(wh_key)
            if wh is None:
                img_key = f"image-{file_idx:09d}".encode()
                img = self.lmdb_sets[lmdb_idx]["txn"].get(img_key)
                buf = io.BytesIO(img)
                w, h = Image.open(buf).size
            else:
                wh = wh.decode("utf-8")
                w, h = wh.split("_")
            wh_ratio.append(float(w) / float(h))
        return wh_ratio

    def load_hierarchical_lmdb_dataset(self, data_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, ratio in zip(data_dir_list, ratio_list):
            env = lmdb.open(
                dirpath,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            txn = env.begin(write=False)
            num_samples = int(txn.get("num-samples".encode()))
            lmdb_sets[dataset_idx] = {
                "dirpath": dirpath,
                "env": env,
                "txn": txn,
                "num_samples": num_samples,
                "ratio_num_samples": int(ratio * num_samples),
            }
            dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]["ratio_num_samples"]
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]["ratio_num_samples"]
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(
                random.sample(
                    range(1, self.lmdb_sets[lno]["num_samples"] + 1),
                    self.lmdb_sets[lno]["ratio_num_samples"],
                )
            )
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data."""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype="uint8")
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data["image"]
        h = img.shape[0]
        w = img.shape[1]
        if self.padding_rand and random.random() < 0.5:
            padding = not padding
        imgW, imgH = (
            self.base_shape[gen_ratio - 1]
            if gen_ratio <= 4
            else [self.base_h * gen_ratio, self.base_h]
        )
        use_ratio = imgW // imgH
        if use_ratio >= (w // h) + 2:
            self.error += 1
            return None
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR
            )
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)

            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        if self.padding_doub and random.random() < 0.5:
            padding_im[:, :, -resized_w:] = resized_image
        else:
            padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data["image"] = padding_im
        data["valid_ratio"] = valid_ratio
        data["real_ratio"] = round(w / h)
        return data

    def get_lmdb_sample_info(self, txn, index):
        label_key = "label-%09d".encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode("utf-8")
        img_key = "image-%09d".encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]["txn"], file_idx
        )
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        img, label = sample_info
        data = {"image": img, "label": label}
        outs = transform(data, self.ops[:-1])
        if outs is not None:
            outs = self.resize_norm_img(outs, ratio, padding=self.padding)
            if outs is None:
                ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
                ids = random.sample(ratio_ids, 1)
                return self.__getitem__([img_width, img_height, ids[0], ratio])
            outs = transform(outs, self.ops[-1:])
        if outs is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]
