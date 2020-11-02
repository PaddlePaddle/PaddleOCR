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

#coding=utf-8
import string

class SpecialCharacter(object):
    """
    Special Sign Converter
    """

    def __init__(self):
        special_char_dict_path = "./ppocr/utils/special_character_dict.txt"
        self.special_char = []
        self.normal_char = []
        with open(special_char_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                result = line.split(',')
                if len(result) == 2:
                    self.special_char.append(result[0])
                    self.normal_char.append(result[1])

    def normalText(self, text):
        """
        normal converter, replace special sign
        """
        for index,item in enumerate(self.special_char):
            if text.find(item) >= 0:
                text = text.replace(item, self.normal_char[index])

        return text


if __name__ == "__main__":
    sp = SpecialCharacter()
    print(sp.normalText('２0２1'.decode('utf-8')))