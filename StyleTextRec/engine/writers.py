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
import os
import cv2
import glob

from utils.logging import get_logger


class SimpleWriter(object):
    def __init__(self, config, tag):
        self.logger = get_logger()
        self.output_dir = config["Global"]["output_dir"]
        self.counter = 0
        self.label_dict = {}
        self.tag = tag
        self.label_file_index = 0

    def save_image(self, image, text_input_label):
        image_home = os.path.join(self.output_dir, "images", self.tag)
        if not os.path.exists(image_home):
            os.makedirs(image_home)

        image_path = os.path.join(image_home, "{}.png".format(self.counter))
        # todo support continue synth
        cv2.imwrite(image_path, image)
        self.logger.info("generate image: {}".format(image_path))

        image_name = os.path.join(self.tag, "{}.png".format(self.counter))
        self.label_dict[image_name] = text_input_label

        self.counter += 1
        if not self.counter % 100:
            self.save_label()

    def save_label(self):
        label_raw = ""
        label_home = os.path.join(self.output_dir, "label")
        if not os.path.exists(label_home):
            os.mkdir(label_home)
        for image_path in self.label_dict:
            label = self.label_dict[image_path]
            label_raw += "{}\t{}\n".format(image_path, label)
        label_file_path = os.path.join(label_home,
                                       "{}_label.txt".format(self.tag))
        with open(label_file_path, "w") as f:
            f.write(label_raw)
        self.label_file_index += 1

    def merge_label(self):
        label_raw = ""
        label_file_regex = os.path.join(self.output_dir, "label",
                                        "*_label.txt")
        label_file_list = glob.glob(label_file_regex)
        for label_file_i in label_file_list:
            with open(label_file_i, "r") as f:
                label_raw += f.read()
        label_file_path = os.path.join(self.output_dir, "label.txt")
        with open(label_file_path, "w") as f:
            f.write(label_raw)
