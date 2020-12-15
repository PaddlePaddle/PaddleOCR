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
import sys
import glob

from utils.config import ArgsParser
from engine.synthesisers import ImageSynthesiser

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))


def synth_image():
    args = ArgsParser().parse_args()
    image_synthesiser = ImageSynthesiser()
    style_image_path = args.style_image
    img = cv2.imread(style_image_path)
    text_corpus = args.text_corpus
    language = args.language

    synth_result = image_synthesiser.synth_image(text_corpus, img, language)
    fake_fusion = synth_result["fake_fusion"]
    fake_text = synth_result["fake_text"]
    fake_bg = synth_result["fake_bg"]
    cv2.imwrite("fake_fusion.jpg", fake_fusion)
    cv2.imwrite("fake_text.jpg", fake_text)
    cv2.imwrite("fake_bg.jpg", fake_bg)


def batch_synth_images():
    image_synthesiser = ImageSynthesiser()

    corpus_file = "../StyleTextRec_data/test_20201208/test_text_list.txt"
    style_data_dir = "../StyleTextRec_data/test_20201208/style_images/"
    save_path = "./output_data/"
    corpus_list = []
    with open(corpus_file, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            substr = line.decode("utf-8").strip("\n").split("\t")
            corpus_list.append(substr)
    style_img_list = glob.glob("{}/*.jpg".format(style_data_dir))
    corpus_num = len(corpus_list)
    style_img_num = len(style_img_list)
    for cno in range(corpus_num):
        for sno in range(style_img_num):
            corpus, lang = corpus_list[cno]
            style_img_path = style_img_list[sno]
            img = cv2.imread(style_img_path)
            synth_result = image_synthesiser.synth_image(corpus, img, lang)
            fake_fusion = synth_result["fake_fusion"]
            fake_text = synth_result["fake_text"]
            fake_bg = synth_result["fake_bg"]
            for tp in range(2):
                if tp == 0:
                    prefix = "%s/c%d_s%d_" % (save_path, cno, sno)
                else:
                    prefix = "%s/s%d_c%d_" % (save_path, sno, cno)
                cv2.imwrite("%s_fake_fusion.jpg" % prefix, fake_fusion)
                cv2.imwrite("%s_fake_text.jpg" % prefix, fake_text)
                cv2.imwrite("%s_fake_bg.jpg" % prefix, fake_bg)
                cv2.imwrite("%s_input_style.jpg" % prefix, img)
            print(cno, corpus_num, sno, style_img_num)


if __name__ == '__main__':
    # batch_synth_images()
    synth_image()
