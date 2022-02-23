#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numbers
import numpy as np

try:
    from collections.abc import Sequence, Mapping
except:
    from collections import Sequence, Mapping

FIELD_PREFIX = "_paddle_field_"


def _flatten_batch(batch):
    """
    For lod_blocking_queue only receive tensor array, flatten batch
    data, extract numpy.array data out as a list of numpy.array to
    send to lod_blocking_queue, and save the batch data structure
    such as fields in other types (str, int, etc) or key-value map
    of dictionaries
    """

    def _flatten(batch, flat_batch, structure, field_idx):
        if isinstance(batch, Sequence):
            for field in batch:
                if isinstance(field, (np.ndarray, paddle.Tensor)):
                    structure.append('{}{}'.format(FIELD_PREFIX, field_idx))
                    flat_batch.append(field)
                    field_idx += 1
                elif isinstance(field, (str, bytes, numbers.Number)):
                    structure.append(field)
                elif isinstance(field, Sequence):
                    field_struct, field_idx = _flatten(field, flat_batch, [],
                                                       field_idx)
                    structure.append(field_struct)
                elif isinstance(field, Mapping):
                    field_struct, field_idx = _flatten(field, flat_batch, {},
                                                       field_idx)
                    structure.append(field_struct)
                else:
                    structure.append(field)
        elif isinstance(batch, Mapping):
            for k, field in batch.items():
                if isinstance(field, (np.ndarray, paddle.Tensor)):
                    structure[k] = '{}{}'.format(FIELD_PREFIX, field_idx)
                    flat_batch.append(field)
                    field_idx += 1
                elif isinstance(field, (str, bytes, numbers.Number)):
                    structure[k] = field
                elif isinstance(field, Sequence):
                    field_struct, field_idx = _flatten(field, flat_batch, [],
                                                       field_idx)
                    structure[k] = field_struct
                elif isinstance(field, Mapping):
                    field_struct, field_idx = _flatten(field, flat_batch, {},
                                                       field_idx)
                    structure[k] = field_struct
                else:
                    structure[k] = field
        else:
            raise TypeError("wrong flat data type: {}".format(type(batch)))

        return structure, field_idx

    # sample only contains single fields
    if not isinstance(batch, Sequence):
        flat_batch = []
        structure, _ = _flatten([batch], flat_batch, [], 0)
        return flat_batch, structure[0]
    flat_batch = []
    structure, _ = _flatten(batch, flat_batch, [], 0)
    return flat_batch, structure


def _restore_batch(flat_batch, structure):
    """
    After reading list of Tensor data from lod_blocking_queue outputs,
    use this function to restore the batch data structrue, replace
    :attr:`_paddle_field_x` with data from flat_batch
    """

    def _restore(structure, field_idx):
        if isinstance(structure, Sequence):
            for i, field in enumerate(structure):
                if isinstance(field, str) and field.startswith(FIELD_PREFIX):
                    cur_field_idx = int(field.replace(FIELD_PREFIX, ''))
                    field_idx = max(field_idx, cur_field_idx)
                    assert flat_batch[cur_field_idx] is not None, \
                                "flat_batch[{}] parsed repeatly"
                    structure[i] = flat_batch[cur_field_idx]
                    flat_batch[cur_field_idx] = None
                elif isinstance(field, (str, bytes, numbers.Number)):
                    continue
                elif isinstance(field, (Sequence, Mapping)):
                    field_idx = _restore(structure[i], field_idx)
        elif isinstance(structure, Mapping):
            for k, field in structure.items():
                if isinstance(field, str) and field.startswith(FIELD_PREFIX):
                    cur_field_idx = int(field.replace(FIELD_PREFIX, ''))
                    field_idx = max(field_idx, cur_field_idx)
                    assert flat_batch[cur_field_idx] is not None, \
                                "flat_batch[{}] parsed repeatly"
                    structure[k] = flat_batch[cur_field_idx]
                    flat_batch[cur_field_idx] = None
                elif isinstance(field, (str, bytes, numbers.Number)):
                    continue
                elif isinstance(field, (Sequence, Mapping)):
                    field_idx = _restore(structure[k], field_idx)
        else:
            raise TypeError("wrong flat data type: {}".format(type(structure)))

        return field_idx

    assert isinstance(flat_batch, Sequence), \
            "flat_batch is not a list or tuple"

    # no np.array in dataset, no output tensor from blocking queue
    # simply return structure
    if len(flat_batch) == 0:
        return structure

    # sample only contains single fields
    if isinstance(structure, (str, bytes)):
        assert structure == '{}{}'.format(FIELD_PREFIX, 0), \
                "invalid structure: {}".format(structure)
        return flat_batch[0]
    field_idx = _restore(structure, 0)
    assert field_idx + 1 == len(flat_batch), "Tensor parse incomplete"
    return structure
