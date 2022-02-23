#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import logging
from ..log_helper import get_logger
from collections.abc import Sequence, Mapping

_WARNING_TO_LOG = True


class _DatasetFetcher(object):
    def __init__(self, dataset, auto_collate_batch, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collate_batch = auto_collate_batch
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    # NOTE: fetch function here perform the whole pipeline of dataset
    #       reading and data trasforms of a batch in each calling, this
    #       may take a long time inside, if DataLoader is exit outside,
    #       fetch need to perceive exit situation, so we pass done_event
    #       here for fetch to check exit status
    # NOTE: if DataLoadet exit by `break`, performing GPU tensor operations,
    #       e.g. to_tensor may cause SIGSEGV in thread, so we pass the
    #       done_event argument to check DataLoader exit status between
    #       ecah sample processing in the batch
    def fetch(self, batch_indices, done_event=None):
        raise NotImplementedError("'fetch' not implement for class {}".format(
            self.__class__.__name__))

    def _log_warning(self):
        # only log warning on GPU 0 when distributed launch
        from ...distributed import get_world_size, get_rank
        if get_world_size() >= 2 and get_rank() != 0:
            return

        warn_str = "Detect dataset only contains single fileds, return format " \
                   "changed since Paddle 2.1. In Paddle <= 2.0, DataLoader add " \
                   "a list surround output data(e.g. return [data]), and in " \
                   "Paddle >= 2.1, DataLoader return the single filed directly " \
                   "(e.g. return data). For example, in following code: \n\n"
        warn_str += \
                "import numpy as np\n" \
                "from paddle.io import DataLoader, Dataset\n\n" \
                "class RandomDataset(Dataset):\n" \
                "    def __getitem__(self, idx):\n" \
                "        data = np.random.random((2, 3)).astype('float32')\n\n" \
                "        return data\n\n" \
                "    def __len__(self):\n" \
                "        return 10\n\n" \
                "dataset = RandomDataset()\n" \
                "loader = DataLoader(dataset, batch_size=1)\n" \
                "data = next(loader())\n\n"

        warn_str += "In Paddle <= 2.0, data is in format '[Tensor(shape=(1, 2, 3), " \
                    "dtype=float32)]', and in Paddle >= 2.1, data is in format" \
                    " 'Tensor(shape=(1, 2, 3), dtype=float32)'\n"

        logger = get_logger(
            "DataLoader", logging.INFO, fmt='%(levelname)s: %(message)s')
        logger.warning(warn_str)


class _IterableDatasetFetcher(_DatasetFetcher):
    def __init__(self, dataset, auto_collate_batch, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(
            dataset, auto_collate_batch, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, batch_indices, done_event=None):

        if self.auto_collate_batch:
            data = []
            for _ in batch_indices:
                if done_event is None or not done_event.is_set():
                    try:
                        data.append(next(self.dataset_iter))
                    except StopIteration:
                        break
                else:
                    return None

            if len(data) == 0 or (self.drop_last and
                                  len(data) < len(batch_indices)):
                raise StopIteration

            global _WARNING_TO_LOG
            if not isinstance(data[0], (Sequence, Mapping)) \
                    and _WARNING_TO_LOG:
                self._log_warning()
                _WARNING_TO_LOG = False
        else:
            data = next(self.dataset_iter)

        if self.collate_fn:
            data = self.collate_fn(data)
        return data


class _MapDatasetFetcher(_DatasetFetcher):
    def __init__(self, dataset, auto_collate_batch, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collate_batch,
                                                 collate_fn, drop_last)

    def fetch(self, batch_indices, done_event=None):
        if self.auto_collate_batch:
            data = []
            for idx in batch_indices:
                if done_event is None or not done_event.is_set():
                    data.append(self.dataset[idx])
                else:
                    return None

            global _WARNING_TO_LOG
            if not isinstance(data[0], (Sequence, Mapping)) \
                    and _WARNING_TO_LOG:
                self._log_warning()
                _WARNING_TO_LOG = False
        else:
            data = self.dataset[batch_indices]

        if self.collate_fn:
            data = self.collate_fn(data)
        return data
