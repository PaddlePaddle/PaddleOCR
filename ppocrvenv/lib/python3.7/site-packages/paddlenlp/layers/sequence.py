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


def sequence_mask(seq_ids, valid_lengths):
    """
    To boost the performance, this sequence_mask is different with paddle.fluid.layers.sequence_mask

    Args:
        seq_ids (Tensor):
            The whole sequence index, a tensor with a shape of [batch_size, sequence_length].
        valid_lengths (Tensor):
            The valid length of every sequence, a tensor with a shape of [batch_size].

    Returns:
        Tensor: Returns the output sequence mask `mask`.
        Its dtype is `bool` and has a shape of [batch_size, sequence_length].
    """
    lengths_exp = valid_lengths.unsqueeze(1)
    mask = seq_ids < lengths_exp

    return mask
