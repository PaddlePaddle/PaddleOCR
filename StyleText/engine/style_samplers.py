# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import random
import cv2


class DatasetSampler(object):
    def __init__(self, config):
        self.image_home = config["StyleSampler"]["image_home"]
        label_file = config["StyleSampler"]["label_file"]
        self.dataset_with_label = config["StyleSampler"]["with_label"]
        self.height = config["Global"]["image_height"]
        self.index = 0
        with open(label_file, "r") as f:
            label_raw = f.read()
            self.path_label_list = label_raw.split("\n")[:-1]
        assert len(self.path_label_list) > 0
        random.shuffle(self.path_label_list)

    def sample(self):
        if self.index >= len(self.path_label_list):
            random.shuffle(self.path_label_list)
            self.index = 0
        if self.dataset_with_label:
            path_label = self.path_label_list[self.index]
            rel_image_path, label = path_label.split('\t')
        else:
            rel_image_path = self.path_label_list[self.index]
            label = None
        img_path = "{}/{}".format(self.image_home, rel_image_path)
        image = cv2.imread(img_path)
        origin_height = image.shape[0]
        ratio = self.height / origin_height
        width = int(image.shape[1] * ratio)
        height = int(image.shape[0] * ratio)
        image = cv2.resize(image, (width, height))

        self.index += 1
        if label:
            return {"image": image, "label": label}
        else:
            return {"image": image}


def duplicate_image(image, width):
    image_width = image.shape[1]
    dup_num = width // image_width + 1
    image = np.tile(image, reps=[1, dup_num, 1])
    cropped_image = image[:, :width, :]
    return cropped_image
