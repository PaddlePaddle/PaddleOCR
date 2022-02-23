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

import collections
import functools
import math
import six

import numpy as np
import paddle.distributed as dist


class SamplerHelper(object):
    """
    The class is to help construct iterable sampler used for 
    :class:`paddle.io.DataLoader`. It wraps a dataset and uses its 
    :meth:`__getitem__` method. Every subclass of :class:`SamplerHelper` has 
    to provide an :meth:`__iter__` method, providing a way to iterate over 
    indices of dataset elements, and a :meth:`__len__` method that returns the 
    length of the returned iterators.

    The class also can be used as batch iterator instead of indices iterator 
    when `iterator` yield samples rather than indices by initializing `iterator` 
    with a iterable dataset.

    .. note:: 
        The :meth:`__len__` method isn't strictly required by 
        :class:`paddle.io.DataLoader`, but is expected in any calculation 
        involving the length of a :class:`paddle.io.DataLoader`.

    Args:
        dataset (Dataset): Input dataset for :class:`SamplerHelper`.
        iterable (Iterable, optional): Iterator of dataset. Default: None.
    """

    # chain sampler
    def __init__(self, dataset, iterable=None):
        self.data_source = dataset
        self.iterable = iterable
        if isinstance(dataset, collections.Iterable) and iterable is None:
            # iterable-style datasets
            self.iterable = dataset

    def __iter__(self):
        if self.iterable is None:
            return iter(range(len(self.data_source)))
        elif isinstance(self.iterable, collections.Iterable):
            return iter(self.iterable)
        elif callable(self.iterable):
            return self.iterable()
        else:
            raise ValueError(
                "`iterable` should be None, instance of Iterable or callable "
                "producing generator.")

    def __len__(self):
        # Allow some samplers have different length with `len(data_source)`,
        # such as batch sampler.
        if hasattr(self, "_length"):
            return self._length
        else:
            return len(self.data_source)

    @property
    def length(self):
        """
        Returns the length.
        """

        # since `len()` only produce integer, use length property to get None
        # for uncertain length. samplers can set length if necessary.
        try:
            length = len(self)
        except Exception:
            length = None
        return length

    @length.setter
    def length(self, length):
        self._length = length

    def apply(self, fn):
        # Transformation functions would be performed. It includes 
        # :meth:`shuffle`, :meth:`sort`, :meth:`fit` and :meth:`shard`.
        # Args:
        #     fn (callable): Transformation functions to be performed.
        # Returns:
        #     SamplerHelper: A new transformed :class:`SamplerHelper` object.

        rs = fn(self)
        if isinstance(rs, (list, tuple)):
            iterable, data_source = rs
        else:
            iterable, data_source = rs, self.data_source
        sampler = type(self)(data_source, iterable)
        return sampler

    def shuffle(self, buffer_size=-1, seed=None):
        """
        Shuffles the dataset according to the given buffer size and random seed.

        Args:
            buffer_size (int, optional): Buffer size for shuffle. If 
                `buffer_size < 0` or more than the length of the dataset, 
                `buffer_size` is the length of the dataset. Default: -1. 
            seed (int, optional): Seed for the random. Default: None.

        Returns:
            SamplerHelper: A new shuffled :class:`SamplerHelper` object.
            
        Example:
            .. code-block:: python

                from paddlenlp.data import SamplerHelper
                from paddle.io import Dataset

                class MyDataset(Dataset):
                    def __init__(self):
                        super(MyDataset, self).__init__()
                        self.data = [
                            [[1, 2, 3, 4], [1]],
                            [[5, 6, 7], [0]],
                            [[8, 9], [1]],
                        ]

                    def __getitem__(self, index):
                        data = self.data[index][0]
                        label = self.data[index][1]
                        return data, label

                    def __len__(self):
                        return len(self.data)

                dataset = MyDataset()
                sampler = SamplerHelper(dataset)
                print(list(sampler))    # indices of dataset elements
                # [0, 1, 2]

                sampler = sampler.shuffle(seed=2)
                print(list(sampler))    # indices of dataset elements
                # [2, 1, 0]
        """
        if seed is not None:
            random_generator = np.random.RandomState(seed)
        else:  # use the global random generator
            random_generator = np.random

        def _impl():
            buf = []
            for idx in iter(self):
                buf.append(idx)
                if buffer_size > 0 and len(buf) >= buffer_size:
                    random_generator.shuffle(buf)
                    for b in buf:
                        yield b
                    buf = []
            if len(buf) > 0:
                random_generator.shuffle(buf)
                for b in buf:
                    yield b

        return type(self)(self.data_source, _impl)

    def sort(self, cmp=None, key=None, reverse=False, buffer_size=-1):
        """
        Sorts the dataset according to given callable :meth:`cmp` or :meth:`key`.

        Args:
            cmp (callable, optional): The function of comparison. Default: None. 
            key (callable, optional): The function of key. Default: None.
            reverse (bool, optional): Whether to reverse when sorting the data 
                samples. If True, it means in descending order, and False means 
                in ascending order. Default: False.
            buffer_size (int, optional): Buffer size for sort. If 
                `buffer_size < 0` or `buffer_size` is more than the length 
                of the data, `buffer_size` will be set to the length of the data. 
                Default: -1.

        Returns:
            SamplerHelper: A new sorted :class:`SamplerHelper` object.
            
        Example:
            .. code-block:: python

                from paddlenlp.data import SamplerHelper
                from paddle.io import Dataset

                class MyDataset(Dataset):
                    def __init__(self):
                        super(MyDataset, self).__init__()
                        self.data = [
                            [[1, 2, 3, 4], [1]],
                            [[5, 6, 7], [0]],
                            [[8, 9], [1]],
                        ]

                    def __getitem__(self, index):
                        data = self.data[index][0]
                        label = self.data[index][1]
                        return data, label

                    def __len__(self):
                        return len(self.data)

                dataset = MyDataset()
                sampler = SamplerHelper(dataset)
                print(list(sampler))    # indices of dataset elements
                # [0, 1, 2]

                # Sorted in ascending order by the length of the first field 
                # of the sample
                key = (lambda x, data_source: len(data_source[x][0]))
                sampler = sampler.sort(key=key)
                print(list(sampler))    # indices of dataset elements
                # [2, 1, 0]
        """
        if key:
            key_wrapper = (lambda x: key(x, self.data_source))
        elif cmp:
            key_wrapper = functools.cmp_to_key(
                lambda x, y: cmp(x, y, self.data_source))
        else:
            key_wrapper = (lambda x: len(self.data_source[x]))

        def _impl():
            data_source = self.data_source
            buf = []
            for idx in iter(self):
                buf.append(idx)
                if buffer_size > 0 and len(buf) >= buffer_size:
                    buf = sorted(buf, key=key_wrapper, reverse=reverse)
                    for b in buf:
                        yield b
                    buf = []
            if len(buf) > 0:
                buf = sorted(buf, key=key_wrapper, reverse=reverse)
                for b in buf:
                    yield b

        return type(self)(self.data_source, _impl)

    def batch(self, batch_size, drop_last=False, batch_size_fn=None, key=None):
        """
        Batches the dataset according to given `batch_size`.

        Args:
            batch_size (int): The batch size.
            drop_last (bool, optional): Whether to drop the last mini batch. 
                Default: False.
            batch_size_fn (callable, optional): It accepts four arguments: 
                index of data source, the length of minibatch, the size of
                minibatch so far and data source, and it returns the size of
                mini batch so far. Actually, the returned value can be anything
                and would used as argument `size_so_far` in `key`. If None, it
                would return the length of mini match. Default: None.
            key (callable, optional): The function of key. It accepts the size of minibatch so far
                and the length of minibatch, and returns what to be compared
                with `batch_size`. If None, only the size of mini batch so far
                would be compared with `batch_size`. Default: None.

        Returns:
            SamplerHelper: A new batched :class:`SamplerHelper` object.
            
        Example:
            .. code-block:: python

                from paddlenlp.data import SamplerHelper
                from paddle.io import Dataset

                class MyDataset(Dataset):
                    def __init__(self):
                        super(MyDataset, self).__init__()
                        self.data = [
                            [[1, 2, 3, 4], [1]],
                            [[5, 6, 7], [0]],
                            [[8, 9], [1]],
                        ]

                    def __getitem__(self, index):
                        data = self.data[index][0]
                        label = self.data[index][1]
                        return data, label

                    def __len__(self):
                        return len(self.data)

                dataset = MyDataset()
                sampler = SamplerHelper(dataset)
                print(list(sampler))    # indices of dataset elements
                # [0, 1, 2]

                sampler = sampler.batch(batch_size=2)
                print(list(sampler))    # indices of dataset elements
                # [[0, 1], [2]]
        """
        _key = lambda size_so_far, minibatch_len: size_so_far

        ori_batch_size_fn = batch_size_fn
        if batch_size_fn is None:
            batch_size_fn = lambda new, count, sofar, data_source: count
        key = _key if key is None else key

        def _impl():
            data_source = self.data_source
            minibatch, size_so_far = [], 0
            for idx in iter(self):
                minibatch.append(idx)
                size_so_far = batch_size_fn(idx,
                                            len(minibatch), size_so_far,
                                            data_source)
                if key(size_so_far, len(minibatch)) == batch_size:
                    yield minibatch
                    minibatch, size_so_far = [], 0
                elif key(size_so_far, len(minibatch)) > batch_size:
                    if len(minibatch) == 1:
                        raise ValueError(
                            "Please increase the value of `batch_size`, or limit the max length of batch."
                        )
                    yield minibatch[:-1]
                    minibatch, size_so_far = minibatch[-1:], batch_size_fn(
                        idx, 1, 0, data_source)
            if minibatch and not drop_last:
                yield minibatch

        sampler = type(self)(self.data_source, _impl)
        if ori_batch_size_fn is None and self.length is not None:
            sampler.length = (self.length + int(not drop_last) *
                              (batch_size - 1)) // batch_size
        else:
            sampler.length = None

        return sampler

    def shard(self, num_replicas=None, rank=None):
        """
        Slices the dataset for multi GPU training.

        Args:
            num_replicas (int, optional): The number of training process, and 
                is also the number of GPU cards used in training. If None, it 
                will be set by :meth:`paddle.distributed.get_world_size` method. 
                Default: None.
            rank (int, optional): The id of current training process. Equal 
                to the value of the environment variable PADDLE_TRAINER_ID. If 
                None, it will be intialized by :meth:`paddle.distributed.get_rank` 
                method. Default: None.

        Returns:
            SamplerHelper: A new sliced :class:`SamplerHelper` object.
            
        Example:
            .. code-block:: python

                from paddlenlp.data import SamplerHelper
                from paddle.io import Dataset

                class MyDataset(Dataset):
                    def __init__(self):
                        super(MyDataset, self).__init__()
                        self.data = [
                            [[1, 2, 3, 4], [1]],
                            [[5, 6, 7], [0]],
                            [[8, 9], [1]],
                        ]

                    def __getitem__(self, index):
                        data = self.data[index][0]
                        label = self.data[index][1]
                        return data, label

                    def __len__(self):
                        return len(self.data)

                dataset = MyDataset()
                sampler = SamplerHelper(dataset)
                print(list(sampler))    # indices of dataset elements
                # [0, 1, 2]

                sampler = sampler.shard(num_replicas=2)
                print(list(sampler))    # indices of dataset elements
                # [0, 2]
        """
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        def _impl():
            for i, idx in enumerate(self):
                if i % num_replicas == rank:
                    yield idx
            if i % num_replicas != num_replicas - 1 and rank > i % num_replicas:
                # use last samples to make it evenly divisible
                yield idx

        sampler = type(self)(self.data_source, _impl)
        if self.length is not None:
            sampler.length = int(math.ceil(self.length * 1.0 / num_replicas))
        else:
            sampler.length = None
        return sampler

    def list(self):
        # Produce a sampler with a `listiterator` when calling `iter`. Since 
        # `list` would fetch all contents at time, thus it can get accurate 
        # length.

        def _impl():
            indices = list(iter(self))
            self.length = len(indices)
            return iter(indices)

        return type(self)(self.data_source, _impl)
