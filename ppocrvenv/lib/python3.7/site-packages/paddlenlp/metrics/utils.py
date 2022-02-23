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


def default_trans_func(output, label, seq_mask, vocab):
    seq_mask = np.expand_dims(seq_mask, axis=2).repeat(output.shape[2], axis=2)
    output = output * seq_mask
    idx = np.argmax(output, axis=2)
    cand, ref_list = [], []
    for i in range(idx.shape[0]):
        token_list = []
        for j in range(idx.shape[1]):
            if seq_mask[i][j][0] == 0:
                break
            token_list.append(vocab[idx[i][j]])
        cand.append(token_list)

    label = np.squeeze(label, axis=2)
    for i in range(label.shape[0]):
        token_list = []
        for j in range(label.shape[1]):
            if seq_mask[i][j][0] == 0:
                break
            token_list.append(vocab[label[i][j]])

        ref_list.append([token_list])
    return cand, ref_list
