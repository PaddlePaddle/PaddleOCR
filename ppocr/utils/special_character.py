# -*- coding: utf-8 -*
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

class SpecialCharacter(object):
    """
    Special Sign Converter
    """

    def __init__(self, config):
        self.special_char = []
        self.normal_char = []
        if "special_character_dict_path" in config: 
            special_char_dict_path = config["special_character_dict_path"]
            with open(special_char_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.strip("\n").strip("\r\n")
                    result = line.split(',')
                    if len(result) == 2:
                        self.special_char.append(result[0])
                        self.normal_char.append(result[1])
        else:
            self.special_char = ['０','１','２','３','４','５','６','７','８','９','Ａ','Ｂ','Ｃ','Ｄ','Ｅ','Ｆ','Ｇ','Ｈ','Ｉ','Ｊ','Ｋ','Ｌ','Ｍ','Ｎ','Ｏ','Ｐ','Ｑ','Ｒ','Ｓ','Ｔ','Ｕ','Ｖ','Ｗ','Ｘ','Ｙ','Ｚ']
            self.normal_char = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    def normalText(self, text):
        """
        normal converter, replace special sign
        """
        for index,item in enumerate(self.special_char):
            item = item.decode('utf-8')
            if text.find(item) >= 0:
                text = text.replace(item, self.normal_char[index])

        return text


if __name__ == "__main__":
    sp = SpecialCharacter({'special_character_dict_path': './special_character_dict.txt'})
    print(sp.normalText('２0２1'.decode('utf-8')))