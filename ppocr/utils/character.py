# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import string
import re
from .check import check_config_params
import sys


class CharacterOps(object):
    """ Convert between text-label and text-index """

    def __init__(self, config):
        self.character_type = config['character_type']
        self.loss_type = config['loss_type']
        if self.character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif self.character_type == "ch":
            character_dict_path = config['character_dict_path']
            add_space = False
            if 'add_space' in config:
                add_space = config['add_space']
            self.character_str = ""
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if add_space:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif self.character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)
        self.beg_str = "sos"
        self.end_str = "eos"
        if self.loss_type == "attention":
            dict_character = [self.beg_str, self.end_str] + dict_character
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if self.character_type == "en":
            text = text.lower()

        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        text = np.array(text_list)
        return text

    def decode(self, text_index, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        char_list = []
        char_num = self.get_char_num()

        if self.loss_type == "attention":
            beg_idx = self.get_beg_end_flag_idx("beg")
            end_idx = self.get_beg_end_flag_idx("end")
            ignored_tokens = [beg_idx, end_idx]
        else:
            ignored_tokens = [char_num]

        for idx in range(len(text_index)):
            if text_index[idx] in ignored_tokens:
                continue
            if is_remove_duplicate:
                if idx > 0 and text_index[idx - 1] == text_index[idx]:
                    continue
            char_list.append(self.character[text_index[idx]])
        text = ''.join(char_list)
        return text

    def get_char_num(self):
        return len(self.character)

    def get_beg_end_flag_idx(self, beg_or_end):
        if self.loss_type == "attention":
            if beg_or_end == "beg":
                idx = np.array(self.dict[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict[self.end_str])
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx"\
                    % beg_or_end
            return idx
        else:
            err = "error in get_beg_end_flag_idx when using the loss %s"\
                % (self.loss_type)
            assert False, err


def cal_predicts_accuracy(char_ops,
                          preds,
                          preds_lod,
                          labels,
                          labels_lod,
                          is_remove_duplicate=False):
    acc_num = 0
    img_num = 0
    for ino in range(len(labels_lod) - 1):
        beg_no = preds_lod[ino]
        end_no = preds_lod[ino + 1]
        preds_text = preds[beg_no:end_no].reshape(-1)
        preds_text = char_ops.decode(preds_text, is_remove_duplicate)

        beg_no = labels_lod[ino]
        end_no = labels_lod[ino + 1]
        labels_text = labels[beg_no:end_no].reshape(-1)
        labels_text = char_ops.decode(labels_text, is_remove_duplicate)
        img_num += 1

        if preds_text == labels_text:
            acc_num += 1
    acc = acc_num * 1.0 / img_num
    return acc, acc_num, img_num


def convert_rec_attention_infer_res(preds):
    img_num = preds.shape[0]
    target_lod = [0]
    convert_ids = []
    for ino in range(img_num):
        end_pos = np.where(preds[ino, :] == 1)[0]
        if len(end_pos) <= 1:
            text_list = preds[ino, 1:]
        else:
            text_list = preds[ino, 1:end_pos[1]]
        target_lod.append(target_lod[ino] + len(text_list))
        convert_ids = convert_ids + list(text_list)
    convert_ids = np.array(convert_ids)
    convert_ids = convert_ids.reshape((-1, 1))
    return convert_ids, target_lod


def convert_rec_label_to_lod(ori_labels):
    img_num = len(ori_labels)
    target_lod = [0]
    convert_ids = []
    for ino in range(img_num):
        target_lod.append(target_lod[ino] + len(ori_labels[ino]))
        convert_ids = convert_ids + list(ori_labels[ino])
    convert_ids = np.array(convert_ids)
    convert_ids = convert_ids.reshape((-1, 1))
    return convert_ids, target_lod
