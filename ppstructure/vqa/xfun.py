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

import json
import os
import cv2
import numpy as np
import paddle
import copy
from paddle.io import Dataset

__all__ = ["XFUNDataset"]


class XFUNDataset(Dataset):
    """
    Example:
        print("=====begin to build dataset=====")
        from paddlenlp.transformers import LayoutXLMTokenizer
        tokenizer = LayoutXLMTokenizer.from_pretrained("/paddle/models/transformers/layoutxlm-base-paddle/")
        tok_res = tokenizer.tokenize("Maribyrnong")
        # res = tokenizer.convert_ids_to_tokens(val_data["input_ids"][0])
        dataset = XfunDatasetForSer(
            tokenizer,
            data_dir="./zh.val/",
            label_path="zh.val/xfun_normalize_val.json",
            img_size=(224,224))
        print(len(dataset))

        data = dataset[0]
        print(data.keys())
        print("input_ids: ", data["input_ids"])
        print("labels: ", data["labels"])
        print("token_type_ids: ", data["token_type_ids"])
        print("words_list: ", data["words_list"])
        print("image shape: ", data["image"].shape)
    """

    def __init__(self,
                 tokenizer,
                 data_dir,
                 label_path,
                 contains_re=False,
                 label2id_map=None,
                 img_size=(224, 224),
                 pad_token_label_id=None,
                 add_special_ids=False,
                 return_attention_mask=True,
                 load_mode='all',
                 max_seq_len=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.label_path = label_path
        self.contains_re = contains_re
        self.label2id_map = label2id_map
        self.img_size = img_size
        self.pad_token_label_id = pad_token_label_id
        self.add_special_ids = add_special_ids
        self.return_attention_mask = return_attention_mask
        self.load_mode = load_mode
        self.max_seq_len = max_seq_len

        if self.pad_token_label_id is None:
            self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

        self.all_lines = self.read_all_lines()

        self.entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}
        self.return_keys = {
            'bbox': {
                'type': 'np',
                'dtype': 'int64'
            },
            'input_ids': {
                'type': 'np',
                'dtype': 'int64'
            },
            'labels': {
                'type': 'np',
                'dtype': 'int64'
            },
            'attention_mask': {
                'type': 'np',
                'dtype': 'int64'
            },
            'image': {
                'type': 'np',
                'dtype': 'float32'
            },
            'token_type_ids': {
                'type': 'np',
                'dtype': 'int64'
            },
            'entities': {
                'type': 'dict'
            },
            'relations': {
                'type': 'dict'
            }
        }

        if load_mode == "all":
            self.encoded_inputs_all = self._parse_label_file_all()

    def pad_sentences(self,
                      encoded_inputs,
                      max_seq_len=512,
                      pad_to_max_seq_len=True,
                      return_attention_mask=True,
                      return_token_type_ids=True,
                      truncation_strategy="longest_first",
                      return_overflowing_tokens=False,
                      return_special_tokens_mask=False):
        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
            max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs["input_ids"])
            if self.tokenizer.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                        "input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.tokenizer.pad_token_type_id] * difference)
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs[
                    "input_ids"] + [self.tokenizer.pad_token_id] * difference
                encoded_inputs["labels"] = encoded_inputs[
                    "labels"] + [self.pad_token_label_id] * difference
                encoded_inputs["bbox"] = encoded_inputs[
                    "bbox"] + [[0, 0, 0, 0]] * difference
            elif self.tokenizer.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [
                        1
                    ] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        [self.tokenizer.pad_token_type_id] * difference +
                        encoded_inputs["token_type_ids"])
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * difference + encoded_inputs["input_ids"]
                encoded_inputs["labels"] = [
                    self.pad_token_label_id
                ] * difference + encoded_inputs["labels"]
                encoded_inputs["bbox"] = [
                    [0, 0, 0, 0]
                ] * difference + encoded_inputs["bbox"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        return encoded_inputs

    def truncate_inputs(self, encoded_inputs, max_seq_len=512):
        for key in encoded_inputs:
            if key == "sample_id":
                continue
            length = min(len(encoded_inputs[key]), max_seq_len)
            encoded_inputs[key] = encoded_inputs[key][:length]
        return encoded_inputs

    def read_all_lines(self, ):
        with open(self.label_path, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
        return lines

    def _parse_label_file_all(self):
        """
        parse all samples
        """
        encoded_inputs_all = []
        for line in self.all_lines:
            encoded_inputs_all.extend(self._parse_label_file(line))
        return encoded_inputs_all

    def _parse_label_file(self, line):
        """
        parse single sample
        """

        image_name, info_str = line.split("\t")
        image_path = os.path.join(self.data_dir, image_name)

        def add_imgge_path(x):
            x['image_path'] = image_path
            return x

        encoded_inputs = self._read_encoded_inputs_sample(info_str)
        if self.contains_re:
            encoded_inputs = self._chunk_re(encoded_inputs)
        else:
            encoded_inputs = self._chunk_ser(encoded_inputs)
        encoded_inputs = list(map(add_imgge_path, encoded_inputs))
        return encoded_inputs

    def _read_encoded_inputs_sample(self, info_str):
        """
        parse label info
        """
        # read text info
        info_dict = json.loads(info_str)
        height = info_dict["height"]
        width = info_dict["width"]

        words_list = []
        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        gt_label_list = []

        if self.contains_re:
            # for re
            entities = []
            relations = []
            id2label = {}
            entity_id_to_index_map = {}
            empty_entity = set()
        for info in info_dict["ocr_info"]:
            if self.contains_re:
                # for re
                if len(info["text"]) == 0:
                    empty_entity.add(info["id"])
                    continue
                id2label[info["id"]] = info["label"]
                relations.extend([tuple(sorted(l)) for l in info["linking"]])

            # x1, y1, x2, y2
            bbox = info["bbox"]
            label = info["label"]
            bbox[0] = int(bbox[0] * 1000.0 / width)
            bbox[2] = int(bbox[2] * 1000.0 / width)
            bbox[1] = int(bbox[1] * 1000.0 / height)
            bbox[3] = int(bbox[3] * 1000.0 / height)

            text = info["text"]
            encode_res = self.tokenizer.encode(
                text, pad_to_max_seq_len=False, return_attention_mask=True)

            gt_label = []
            if not self.add_special_ids:
                # TODO: use tok.all_special_ids to remove
                encode_res["input_ids"] = encode_res["input_ids"][1:-1]
                encode_res["token_type_ids"] = encode_res["token_type_ids"][1:
                                                                            -1]
                encode_res["attention_mask"] = encode_res["attention_mask"][1:
                                                                            -1]
            if label.lower() == "other":
                gt_label.extend([0] * len(encode_res["input_ids"]))
            else:
                gt_label.append(self.label2id_map[("b-" + label).upper()])
                gt_label.extend([self.label2id_map[("i-" + label).upper()]] *
                                (len(encode_res["input_ids"]) - 1))
            if self.contains_re:
                if gt_label[0] != self.label2id_map["O"]:
                    entity_id_to_index_map[info["id"]] = len(entities)
                    entities.append({
                        "start": len(input_ids_list),
                        "end":
                        len(input_ids_list) + len(encode_res["input_ids"]),
                        "label": label.upper(),
                    })
            input_ids_list.extend(encode_res["input_ids"])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend([bbox] * len(encode_res["input_ids"]))
            gt_label_list.extend(gt_label)
            words_list.append(text)

        encoded_inputs = {
            "input_ids": input_ids_list,
            "labels": gt_label_list,
            "token_type_ids": token_type_ids_list,
            "bbox": bbox_list,
            "attention_mask": [1] * len(input_ids_list),
            # "words_list": words_list,
        }
        encoded_inputs = self.pad_sentences(
            encoded_inputs,
            max_seq_len=self.max_seq_len,
            return_attention_mask=self.return_attention_mask)
        encoded_inputs = self.truncate_inputs(encoded_inputs)

        if self.contains_re:
            relations = self._relations(entities, relations, id2label,
                                        empty_entity, entity_id_to_index_map)
            encoded_inputs['relations'] = relations
            encoded_inputs['entities'] = entities
        return encoded_inputs

    def _chunk_ser(self, encoded_inputs):
        encoded_inputs_all = []
        seq_len = len(encoded_inputs['input_ids'])
        chunk_size = 512
        for chunk_id, index in enumerate(range(0, seq_len, chunk_size)):
            chunk_beg = index
            chunk_end = min(index + chunk_size, seq_len)
            encoded_inputs_example = {}
            for key in encoded_inputs:
                encoded_inputs_example[key] = encoded_inputs[key][chunk_beg:
                                                                  chunk_end]

            encoded_inputs_all.append(encoded_inputs_example)
        return encoded_inputs_all

    def _chunk_re(self, encoded_inputs):
        # prepare data
        entities = encoded_inputs.pop('entities')
        relations = encoded_inputs.pop('relations')
        encoded_inputs_all = []
        chunk_size = 512
        for chunk_id, index in enumerate(
                range(0, len(encoded_inputs["input_ids"]), chunk_size)):
            item = {}
            for k in encoded_inputs:
                item[k] = encoded_inputs[k][index:index + chunk_size]

            # select entity in current chunk
            entities_in_this_span = []
            global_to_local_map = {}  #
            for entity_id, entity in enumerate(entities):
                if (index <= entity["start"] < index + chunk_size and
                        index <= entity["end"] < index + chunk_size):
                    entity["start"] = entity["start"] - index
                    entity["end"] = entity["end"] - index
                    global_to_local_map[entity_id] = len(entities_in_this_span)
                    entities_in_this_span.append(entity)

            # select relations in current chunk
            relations_in_this_span = []
            for relation in relations:
                if (index <= relation["start_index"] < index + chunk_size and
                        index <= relation["end_index"] < index + chunk_size):
                    relations_in_this_span.append({
                        "head": global_to_local_map[relation["head"]],
                        "tail": global_to_local_map[relation["tail"]],
                        "start_index": relation["start_index"] - index,
                        "end_index": relation["end_index"] - index,
                    })
            item.update({
                "entities": reformat(entities_in_this_span),
                "relations": reformat(relations_in_this_span),
            })
            item['entities']['label'] = [
                self.entities_labels[x] for x in item['entities']['label']
            ]
            encoded_inputs_all.append(item)
        return encoded_inputs_all

    def _relations(self, entities, relations, id2label, empty_entity,
                   entity_id_to_index_map):
        """
        build relations
        """
        relations = list(set(relations))
        relations = [
            rel for rel in relations
            if rel[0] not in empty_entity and rel[1] not in empty_entity
        ]
        kv_relations = []
        for rel in relations:
            pair = [id2label[rel[0]], id2label[rel[1]]]
            if pair == ["question", "answer"]:
                kv_relations.append({
                    "head": entity_id_to_index_map[rel[0]],
                    "tail": entity_id_to_index_map[rel[1]]
                })
            elif pair == ["answer", "question"]:
                kv_relations.append({
                    "head": entity_id_to_index_map[rel[1]],
                    "tail": entity_id_to_index_map[rel[0]]
                })
            else:
                continue
        relations = sorted(
            [{
                "head": rel["head"],
                "tail": rel["tail"],
                "start_index": get_relation_span(rel, entities)[0],
                "end_index": get_relation_span(rel, entities)[1],
            } for rel in kv_relations],
            key=lambda x: x["head"], )
        return relations

    def load_img(self, image_path):
        # read img
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_h, resize_w = self.img_size
        im_shape = img.shape[0:2]
        im_scale_y = resize_h / im_shape[0]
        im_scale_x = resize_w / im_shape[1]
        img_new = cv2.resize(
            img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=2)
        mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
        std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
        img_new = img_new / 255.0
        img_new -= mean
        img_new /= std
        img = img_new.transpose((2, 0, 1))
        return img

    def __getitem__(self, idx):
        if self.load_mode == "all":
            data = copy.deepcopy(self.encoded_inputs_all[idx])
        else:
            data = self._parse_label_file(self.all_lines[idx])[0]

        image_path = data.pop('image_path')
        data["image"] = self.load_img(image_path)

        return_data = {}
        for k, v in data.items():
            if k in self.return_keys:
                if self.return_keys[k]['type'] == 'np':
                    v = np.array(v, dtype=self.return_keys[k]['dtype'])
                return_data[k] = v
        return return_data

    def __len__(self, ):
        if self.load_mode == "all":
            return len(self.encoded_inputs_all)
        else:
            return len(self.all_lines)


def get_relation_span(rel, entities):
    bound = []
    for entity_index in [rel["head"], rel["tail"]]:
        bound.append(entities[entity_index]["start"])
        bound.append(entities[entity_index]["end"])
    return min(bound), max(bound)


def reformat(data):
    new_data = {}
    for item in data:
        for k, v in item.items():
            if k not in new_data:
                new_data[k] = []
            new_data[k].append(v)
    return new_data
