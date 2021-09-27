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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import string
import json


class ClsLabelEncode(object):
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list

    def __call__(self, data):
        label = data['label']
        if label not in self.label_list:
            return None
        label = self.label_list.index(label)
        data['label'] = label
        return data


class DetLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes


class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'EN', 'it', 'xi', 'pu', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs',
            'oc', 'rsc', 'bg', 'uk', 'be', 'te', 'ka', 'chinese_cht', 'hi',
            'mr', 'ne', 'latin', 'arabic', 'cyrillic', 'devanagari'
        ]
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.unknown = "UNKNOWN"
        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is {}".format(
                character_type)
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.character_type == "en":
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class NRTRLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='EN_symbol',
                 use_space_char=False,
                 **kwargs):

        super(NRTRLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data['length'] = np.array(len(text))
        text.insert(0, 2)
        text.append(3)
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
        return dict_character


class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class E2ELabelEncodeTest(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='EN',
                 use_space_char=False,
                 **kwargs):
        super(E2ELabelEncodeTest,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def __call__(self, data):
        import json
        padnum = len(self.dict)
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)
        data['polys'] = boxes
        data['ignore_tags'] = txt_tags
        temp_texts = []
        for text in txts:
            text = text.lower()
            text = self.encode(text)
            if text is None:
                return None
            text = text + [padnum] * (self.max_text_len - len(text)
                                      )  # use 36 to pad
            temp_texts.append(text)
        data['texts'] = np.array(temp_texts)
        return data


class E2ELabelEncodeTrain(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        import json
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data


class AttnLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(AttnLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text = [0] + text + [len(self.character) - 1] + [0] * (self.max_text_len
                                                               - len(text) - 2)
        data['label'] = np.array(text)
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class SEEDLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(SEEDLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character + [self.end_str]
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data['length'] = np.array(len(text)) + 1  # conclude eos
        text = text + [len(self.character) - 1] * (self.max_text_len - len(text)
                                                   )
        data['label'] = np.array(text)
        return data


class SRNLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length=25,
                 character_dict_path=None,
                 character_type='en',
                 use_space_char=False,
                 **kwargs):
        super(SRNLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        char_num = len(self.character)
        if text is None:
            return None
        if len(text) > self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text = text + [char_num - 1] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class TableLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 max_elem_length,
                 max_cell_num,
                 character_dict_path,
                 span_weight=1.0,
                 **kwargs):
        self.max_text_length = max_text_length
        self.max_elem_length = max_elem_length
        self.max_cell_num = max_cell_num
        list_character, list_elem = self.load_char_elem_dict(
            character_dict_path)
        list_character = self.add_special_char(list_character)
        list_elem = self.add_special_char(list_elem)
        self.dict_character = {}
        for i, char in enumerate(list_character):
            self.dict_character[char] = i
        self.dict_elem = {}
        for i, elem in enumerate(list_elem):
            self.dict_elem[elem] = i
        self.span_weight = span_weight

    def load_char_elem_dict(self, character_dict_path):
        list_character = []
        list_elem = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            substr = lines[0].decode('utf-8').strip("\r\n").split("\t")
            character_num = int(substr[0])
            elem_num = int(substr[1])
            for cno in range(1, 1 + character_num):
                character = lines[cno].decode('utf-8').strip("\r\n")
                list_character.append(character)
            for eno in range(1 + character_num, 1 + character_num + elem_num):
                elem = lines[eno].decode('utf-8').strip("\r\n")
                list_elem.append(elem)
        return list_character, list_elem

    def add_special_char(self, list_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        list_character = [self.beg_str] + list_character + [self.end_str]
        return list_character

    def get_span_idx_list(self):
        span_idx_list = []
        for elem in self.dict_elem:
            if 'span' in elem:
                span_idx_list.append(self.dict_elem[elem])
        return span_idx_list

    def __call__(self, data):
        cells = data['cells']
        structure = data['structure']['tokens']
        structure = self.encode(structure, 'elem')
        if structure is None:
            return None
        elem_num = len(structure)
        structure = [0] + structure + [len(self.dict_elem) - 1]
        structure = structure + [0] * (self.max_elem_length + 2 - len(structure)
                                       )
        structure = np.array(structure)
        data['structure'] = structure
        elem_char_idx1 = self.dict_elem['<td>']
        elem_char_idx2 = self.dict_elem['<td']
        span_idx_list = self.get_span_idx_list()
        td_idx_list = np.logical_or(structure == elem_char_idx1,
                                    structure == elem_char_idx2)
        td_idx_list = np.where(td_idx_list)[0]

        structure_mask = np.ones(
            (self.max_elem_length + 2, 1), dtype=np.float32)
        bbox_list = np.zeros((self.max_elem_length + 2, 4), dtype=np.float32)
        bbox_list_mask = np.zeros(
            (self.max_elem_length + 2, 1), dtype=np.float32)
        img_height, img_width, img_ch = data['image'].shape
        if len(span_idx_list) > 0:
            span_weight = len(td_idx_list) * 1.0 / len(span_idx_list)
            span_weight = min(max(span_weight, 1.0), self.span_weight)
        for cno in range(len(cells)):
            if 'bbox' in cells[cno]:
                bbox = cells[cno]['bbox'].copy()
                bbox[0] = bbox[0] * 1.0 / img_width
                bbox[1] = bbox[1] * 1.0 / img_height
                bbox[2] = bbox[2] * 1.0 / img_width
                bbox[3] = bbox[3] * 1.0 / img_height
                td_idx = td_idx_list[cno]
                bbox_list[td_idx] = bbox
                bbox_list_mask[td_idx] = 1.0
                cand_span_idx = td_idx + 1
                if cand_span_idx < (self.max_elem_length + 2):
                    if structure[cand_span_idx] in span_idx_list:
                        structure_mask[cand_span_idx] = span_weight

        data['bbox_list'] = bbox_list
        data['bbox_list_mask'] = bbox_list_mask
        data['structure_mask'] = structure_mask
        char_beg_idx = self.get_beg_end_flag_idx('beg', 'char')
        char_end_idx = self.get_beg_end_flag_idx('end', 'char')
        elem_beg_idx = self.get_beg_end_flag_idx('beg', 'elem')
        elem_end_idx = self.get_beg_end_flag_idx('end', 'elem')
        data['sp_tokens'] = np.array([
            char_beg_idx, char_end_idx, elem_beg_idx, elem_end_idx,
            elem_char_idx1, elem_char_idx2, self.max_text_length,
            self.max_elem_length, self.max_cell_num, elem_num
        ])
        return data

    def encode(self, text, char_or_elem):
        """convert text-label into text-index.
        """
        if char_or_elem == "char":
            max_len = self.max_text_length
            current_dict = self.dict_character
        else:
            max_len = self.max_elem_length
            current_dict = self.dict_elem
        if len(text) > max_len:
            return None
        if len(text) == 0:
            if char_or_elem == "char":
                return [self.dict_character['space']]
            else:
                return None
        text_list = []
        for char in text:
            if char not in current_dict:
                return None
            text_list.append(current_dict[char])
        if len(text_list) == 0:
            if char_or_elem == "char":
                return [self.dict_character['space']]
            else:
                return None
        return text_list

    def get_ignored_tokens(self, char_or_elem):
        beg_idx = self.get_beg_end_flag_idx("beg", char_or_elem)
        end_idx = self.get_beg_end_flag_idx("end", char_or_elem)
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end, char_or_elem):
        if char_or_elem == "char":
            if beg_or_end == "beg":
                idx = np.array(self.dict_character[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict_character[self.end_str])
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx of char" \
                              % beg_or_end
        elif char_or_elem == "elem":
            if beg_or_end == "beg":
                idx = np.array(self.dict_elem[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict_elem[self.end_str])
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx of elem" \
                              % beg_or_end
        else:
            assert False, "Unsupport type %s in char_or_elem" \
                              % char_or_elem
        return idx


class SARLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(SARLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1

        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data['length'] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]

        padded_text[:len(target)] = target
        data['label'] = np.array(padded_text)
        return data

    def get_ignored_tokens(self):
        return [self.padding_idx]
