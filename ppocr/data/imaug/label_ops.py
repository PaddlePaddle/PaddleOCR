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
from shapely.geometry import LineString, Point, Polygon


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


class E2ELabelEncode(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='EN',
                 use_space_char=False,
                 **kwargs):
        super(E2ELabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)
        self.pad_num = len(self.dict)  # the length to pad

    def __call__(self, data):
        text_label_index_list, temp_text = [], []
        texts = data['strs']
        for text in texts:
            text = text.lower()
            temp_text = []
            for c_ in text:
                if c_ in self.dict:
                    temp_text.append(self.dict[c_])
            temp_text = temp_text + [self.pad_num] * (self.max_text_len -
                                                      len(temp_text))
            text_label_index_list.append(temp_text)
        data['strs'] = np.array(text_label_index_list)
        return data


class KieLabelEncode(object):
    def __init__(self, character_dict_path, norm=10, directed=False, **kwargs):
        super(KieLabelEncode, self).__init__()
        self.dict = dict({'': 0})
        with open(character_dict_path, 'r') as fr:
            idx = 1
            for line in fr:
                char = line.strip()
                self.dict[char] = idx
                idx += 1
        self.norm = norm
        self.directed = directed

    def compute_relation(self, boxes):
        """Compute relation between every two boxes."""
        x1s, y1s = boxes[:, 0:1], boxes[:, 1:2]
        x2s, y2s = boxes[:, 4:5], boxes[:, 5:6]
        ws, hs = x2s - x1s + 1, np.maximum(y2s - y1s + 1, 1)
        dxs = (x1s[:, 0][None] - x1s) / self.norm
        dys = (y1s[:, 0][None] - y1s) / self.norm
        xhhs, xwhs = hs[:, 0][None] / hs, ws[:, 0][None] / hs
        whs = ws / hs + np.zeros_like(xhhs)
        relations = np.stack([dxs, dys, whs, xhhs, xwhs], -1)
        bboxes = np.concatenate([x1s, y1s, x2s, y2s], -1).astype(np.float32)
        return relations, bboxes

    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        max_len = 300
        recoder_len = max([len(text_ind) for text_ind in text_inds])
        padded_text_inds = -np.ones((len(text_inds), max_len), np.int32)
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds, recoder_len

    def list_to_numpy(self, ann_infos):
        """Convert bboxes, relations, texts and labels to ndarray."""
        boxes, text_inds = ann_infos['points'], ann_infos['text_inds']
        boxes = np.array(boxes, np.int32)
        relations, bboxes = self.compute_relation(boxes)

        labels = ann_infos.get('labels', None)
        if labels is not None:
            labels = np.array(labels, np.int32)
            edges = ann_infos.get('edges', None)
            if edges is not None:
                labels = labels[:, None]
                edges = np.array(edges)
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, -1)
                labels = np.concatenate([labels, edges], -1)
        padded_text_inds, recoder_len = self.pad_text_indices(text_inds)
        max_num = 300
        temp_bboxes = np.zeros([max_num, 4])
        h, _ = bboxes.shape
        temp_bboxes[:h, :h] = bboxes

        temp_relations = np.zeros([max_num, max_num, 5])
        temp_relations[:h, :h, :] = relations

        temp_padded_text_inds = np.zeros([max_num, max_num])
        temp_padded_text_inds[:h, :] = padded_text_inds

        temp_labels = np.zeros([max_num, max_num])
        temp_labels[:h, :h + 1] = labels

        tag = np.array([h, recoder_len])
        return dict(
            image=ann_infos['image'],
            points=temp_bboxes,
            relations=temp_relations,
            texts=temp_padded_text_inds,
            labels=temp_labels,
            tag=tag)

    def convert_canonical(self, points_x, points_y):

        assert len(points_x) == 4
        assert len(points_y) == 4

        points = [Point(points_x[i], points_y[i]) for i in range(4)]

        polygon = Polygon([(p.x, p.y) for p in points])
        min_x, min_y, _, _ = polygon.bounds
        points_to_lefttop = [
            LineString([points[i], Point(min_x, min_y)]) for i in range(4)
        ]
        distances = np.array([line.length for line in points_to_lefttop])
        sort_dist_idx = np.argsort(distances)
        lefttop_idx = sort_dist_idx[0]

        if lefttop_idx == 0:
            point_orders = [0, 1, 2, 3]
        elif lefttop_idx == 1:
            point_orders = [1, 2, 3, 0]
        elif lefttop_idx == 2:
            point_orders = [2, 3, 0, 1]
        else:
            point_orders = [3, 0, 1, 2]

        sorted_points_x = [points_x[i] for i in point_orders]
        sorted_points_y = [points_y[j] for j in point_orders]

        return sorted_points_x, sorted_points_y

    def sort_vertex(self, points_x, points_y):

        assert len(points_x) == 4
        assert len(points_y) == 4

        x = np.array(points_x)
        y = np.array(points_y)
        center_x = np.sum(x) * 0.25
        center_y = np.sum(y) * 0.25

        x_arr = np.array(x - center_x)
        y_arr = np.array(y - center_y)

        angle = np.arctan2(y_arr, x_arr) * 180.0 / np.pi
        sort_idx = np.argsort(angle)

        sorted_points_x, sorted_points_y = [], []
        for i in range(4):
            sorted_points_x.append(points_x[sort_idx[i]])
            sorted_points_y.append(points_y[sort_idx[i]])

        return self.convert_canonical(sorted_points_x, sorted_points_y)

    def __call__(self, data):
        import json
        label = data['label']
        annotations = json.loads(label)
        boxes, texts, text_inds, labels, edges = [], [], [], [], []
        for ann in annotations:
            box = ann['points']
            x_list = [box[i][0] for i in range(4)]
            y_list = [box[i][1] for i in range(4)]
            sorted_x_list, sorted_y_list = self.sort_vertex(x_list, y_list)
            sorted_box = []
            for x, y in zip(sorted_x_list, sorted_y_list):
                sorted_box.append(x)
                sorted_box.append(y)
            boxes.append(sorted_box)
            text = ann['transcription']
            texts.append(ann['transcription'])
            text_ind = [self.dict[c] for c in text if c in self.dict]
            text_inds.append(text_ind)
            labels.append(ann['label'])
            edges.append(ann.get('edge', 0))
        ann_infos = dict(
            image=data['image'],
            points=boxes,
            texts=texts,
            text_inds=text_inds,
            edges=edges,
            labels=labels)

        return self.list_to_numpy(ann_infos)


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
