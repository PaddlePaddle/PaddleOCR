# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle

from .rec_postprocess import AttnLabelDecode


class TableLabelDecode(AttnLabelDecode):
    """  """

    def __init__(self, character_dict_path, **kwargs):
        super(TableLabelDecode, self).__init__(character_dict_path)
        self.td_token = ['<td>', '<td', '<eb></eb>', '<td></td>']

    def __call__(self, preds, batch=None):
        structure_probs = preds['structure_probs']
        bbox_preds = preds['loc_preds']
        if isinstance(structure_probs, paddle.Tensor):
            structure_probs = structure_probs.numpy()
        if isinstance(bbox_preds, paddle.Tensor):
            bbox_preds = bbox_preds.numpy()
        shape_list = batch[-1]
        result = self.decode(structure_probs, bbox_preds, shape_list)
        if len(batch) == 1:  # only contains shape
            return result

        label_decode_result = self.decode_label(batch)
        return result, label_decode_result

    def decode(self, structure_probs, bbox_preds, shape_list):
        """convert text-label into text-index.
        """
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            score_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                text = self.character[char_idx]
                if text in self.td_token:
                    bbox = bbox_preds[batch_idx, idx]
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append([structure_list, np.mean(score_list)])
            bbox_batch_list.append(np.array(bbox_list))
        result = {
            'bbox_batch_list': bbox_batch_list,
            'structure_batch_list': structure_batch_list,
        }
        return result

    def decode_label(self, batch):
        """convert text-label into text-index.
        """
        structure_idx = batch[1]
        gt_bbox_list = batch[2]
        shape_list = batch[-1]
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                structure_list.append(self.character[char_idx])

                bbox = gt_bbox_list[batch_idx][idx]
                if bbox.sum() != 0:
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
            structure_batch_list.append(structure_list)
            bbox_batch_list.append(bbox_list)
        result = {
            'bbox_batch_list': bbox_batch_list,
            'structure_batch_list': structure_batch_list,
        }
        return result

    def _bbox_decode(self, bbox, shape):
        h, w, ratio_h, ratio_w, pad_h, pad_w = shape
        src_h = h / ratio_h
        src_w = w / ratio_w
        bbox[0::2] *= src_w
        bbox[1::2] *= src_h
        return bbox


class TableMasterLabelDecode(TableLabelDecode):
    """  """

    def __init__(self, character_dict_path, box_shape='ori', **kwargs):
        super(TableMasterLabelDecode, self).__init__(character_dict_path)
        self.box_shape = box_shape
        assert box_shape in [
            'ori', 'pad'
        ], 'The shape used for box normalization must be ori or pad'

    def add_special_char(self, dict_character):
        self.beg_str = '<SOS>'
        self.end_str = '<EOS>'
        self.unknown_str = '<UKN>'
        self.pad_str = '<PAD>'
        dict_character = dict_character
        dict_character = dict_character + [
            self.unknown_str, self.beg_str, self.end_str, self.pad_str
        ]
        return dict_character

    def get_ignored_tokens(self):
        pad_idx = self.dict[self.pad_str]
        start_idx = self.dict[self.beg_str]
        end_idx = self.dict[self.end_str]
        unknown_idx = self.dict[self.unknown_str]
        return [start_idx, end_idx, pad_idx, unknown_idx]

    def _bbox_decode(self, bbox, shape):
        h, w, ratio_h, ratio_w, pad_h, pad_w = shape
        if self.box_shape == 'pad':
            h, w = pad_h, pad_w
        bbox[0::2] *= w
        bbox[1::2] *= h
        bbox[0::2] /= ratio_w
        bbox[1::2] /= ratio_h
        return bbox
