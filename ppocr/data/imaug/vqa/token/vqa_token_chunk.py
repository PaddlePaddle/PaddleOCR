# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from collections import defaultdict


class VQASerTokenChunk(object):
    def __init__(self, max_seq_len=512, infer_mode=False, **kwargs):
        self.max_seq_len = max_seq_len
        self.infer_mode = infer_mode

    def __call__(self, data):
        encoded_inputs_all = []
        seq_len = len(data['input_ids'])
        for index in range(0, seq_len, self.max_seq_len):
            chunk_beg = index
            chunk_end = min(index + self.max_seq_len, seq_len)
            encoded_inputs_example = {}
            for key in data:
                if key in [
                        'label', 'input_ids', 'labels', 'token_type_ids',
                        'bbox', 'attention_mask'
                ]:
                    if self.infer_mode and key == 'labels':
                        encoded_inputs_example[key] = data[key]
                    else:
                        encoded_inputs_example[key] = data[key][chunk_beg:
                                                                chunk_end]
                else:
                    encoded_inputs_example[key] = data[key]

            encoded_inputs_all.append(encoded_inputs_example)
        if len(encoded_inputs_all) == 0:
            return None
        return encoded_inputs_all[0]


class VQAReTokenChunk(object):
    def __init__(self,
                 max_seq_len=512,
                 entities_labels=None,
                 infer_mode=False,
                 **kwargs):
        self.max_seq_len = max_seq_len
        self.entities_labels = {
            'HEADER': 0,
            'QUESTION': 1,
            'ANSWER': 2
        } if entities_labels is None else entities_labels
        self.infer_mode = infer_mode

    def __call__(self, data):
        # prepare data
        entities = data.pop('entities')
        relations = data.pop('relations')
        encoded_inputs_all = []
        for index in range(0, len(data["input_ids"]), self.max_seq_len):
            item = {}
            for key in data:
                if key in [
                        'label', 'input_ids', 'labels', 'token_type_ids',
                        'bbox', 'attention_mask'
                ]:
                    if self.infer_mode and key == 'labels':
                        item[key] = data[key]
                    else:
                        item[key] = data[key][index:index + self.max_seq_len]
                else:
                    item[key] = data[key]
            # select entity in current chunk
            entities_in_this_span = []
            global_to_local_map = {}  #
            for entity_id, entity in enumerate(entities):
                if (index <= entity["start"] < index + self.max_seq_len and
                        index <= entity["end"] < index + self.max_seq_len):
                    entity["start"] = entity["start"] - index
                    entity["end"] = entity["end"] - index
                    global_to_local_map[entity_id] = len(entities_in_this_span)
                    entities_in_this_span.append(entity)

            # select relations in current chunk
            relations_in_this_span = []
            for relation in relations:
                if (index <= relation["start_index"] < index + self.max_seq_len
                        and index <= relation["end_index"] <
                        index + self.max_seq_len):
                    relations_in_this_span.append({
                        "head": global_to_local_map[relation["head"]],
                        "tail": global_to_local_map[relation["tail"]],
                        "start_index": relation["start_index"] - index,
                        "end_index": relation["end_index"] - index,
                    })
            item.update({
                "entities": self.reformat(entities_in_this_span),
                "relations": self.reformat(relations_in_this_span),
            })
            if len(item['entities']) > 0:
                item['entities']['label'] = [
                    self.entities_labels[x] for x in item['entities']['label']
                ]
                encoded_inputs_all.append(item)
        if len(encoded_inputs_all) == 0:
            return None
        return encoded_inputs_all[0]

    def reformat(self, data):
        new_data = defaultdict(list)
        for item in data:
            for k, v in item.items():
                new_data[k].append(v)
        return new_data
