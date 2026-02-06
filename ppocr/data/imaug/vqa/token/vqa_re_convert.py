# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np


class TensorizeEntitiesRelations(object):
    def __init__(self, max_seq_len=512, infer_mode=False, **kwargs):
        self.max_seq_len = max_seq_len
        self.infer_mode = infer_mode

    def __call__(self, data):
        entities = data["entities"]
        relations = data["relations"]

        entities_new = np.full(
            shape=[self.max_seq_len + 1, 3], fill_value=-1, dtype="int64"
        )
        entities_new[0, 0] = len(entities["start"])
        entities_new[0, 1] = len(entities["end"])
        entities_new[0, 2] = len(entities["label"])
        entities_new[1 : len(entities["start"]) + 1, 0] = np.array(entities["start"])
        entities_new[1 : len(entities["end"]) + 1, 1] = np.array(entities["end"])
        entities_new[1 : len(entities["label"]) + 1, 2] = np.array(entities["label"])

        relations_new = np.full(
            shape=[self.max_seq_len * self.max_seq_len + 1, 2],
            fill_value=-1,
            dtype="int64",
        )
        relations_new[0, 0] = len(relations["head"])
        relations_new[0, 1] = len(relations["tail"])
        relations_new[1 : len(relations["head"]) + 1, 0] = np.array(relations["head"])
        relations_new[1 : len(relations["tail"]) + 1, 1] = np.array(relations["tail"])

        data["entities"] = entities_new
        data["relations"] = relations_new
        return data
