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

from __future__ import print_function

import paddle
from .. import framework

__all__ = [
    "Dataset", "IterableDataset", "TensorDataset", "ComposeDataset",
    "ChainDataset", "random_split", "Subset"
]


class Dataset(object):
    """
    An abstract class to encapsulate methods and behaviors of datasets.

    All datasets in map-style(dataset samples can be get by a given key)
    should be a subclass of `paddle.io.Dataset`. All subclasses should
    implement following methods:

    :code:`__getitem__`: get sample from dataset with a given index. This
    method is required by reading dataset sample in :code:`paddle.io.DataLoader`.

    :code:`__len__`: return dataset sample number. This method is required
    by some implements of :code:`paddle.io.BatchSampler`

    see :code:`paddle.io.DataLoader`.

    Examples:
        
        .. code-block:: python

            import numpy as np
            from paddle.io import Dataset
            
            # define a random dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label
                
                def __len__(self):
                    return self.num_samples
            
            dataset = RandomDataset(10)
            for i in range(len(dataset)):
                print(dataset[i])

    """

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))


class IterableDataset(Dataset):
    """
    An abstract class to encapsulate methods and behaviors of iterable datasets.

    All datasets in iterable-style (can only get sample one by one sequentially, like
    a Python iterator) should be a subclass of `paddle.io.IterableDataset`. All subclasses should
    implement following methods:

    :code:`__iter__`: yield sample sequentially. This method is required by reading dataset sample in :code:`paddle.io.DataLoader`.

    .. note::
        do not implement :code:`__getitem__` and :code:`__len__` in IterableDataset, should not be called either.

    see :code:`paddle.io.DataLoader`.

    Examples:
        
        .. code-block:: python

            import numpy as np
            from paddle.io import IterableDataset
            
            # define a random dataset
            class RandomDataset(IterableDataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __iter__(self):
                    for i in range(self.num_samples):
                        image = np.random.random([784]).astype('float32')
                        label = np.random.randint(0, 9, (1, )).astype('int64')
                        yield image, label
            
            dataset = RandomDataset(10)
            for img, lbl in dataset:
                print(img, lbl)

    When :attr:`num_workers > 0`, each worker has a different copy of the dataset object and
    will yield whole dataset samples, which means samples in dataset will be repeated in
    :attr:`num_workers` times. If it is required for each sample to yield only once, there
    are two methods to configure different copy in each worker process to avoid duplicate data
    among workers as follows. In both the methods, worker information that can be getted in
    a worker process by `paddle.io.get_worker_info` will be needed.

    Example 1: splitting data copy in each worker in :code:`__iter__`

        .. code-block:: python

            import math
            import paddle
            import numpy as np
            from paddle.io import IterableDataset, DataLoader, get_worker_info

            class SplitedIterableDataset(IterableDataset):
                def __init__(self, start, end):
                    self.start = start
                    self.end = end

                def __iter__(self):
                    worker_info = get_worker_info()
                    if worker_info is None:
                        iter_start = self.start
                        iter_end = self.end
                    else:
                        per_worker = int(
                            math.ceil((self.end - self.start) / float(
                                worker_info.num_workers)))
                        worker_id = worker_info.id
                        iter_start = self.start + worker_id * per_worker
                        iter_end = min(iter_start + per_worker, self.end)

                    for i in range(iter_start, iter_end):
                        yield np.array([i])

            dataset = SplitedIterableDataset(start=2, end=9)
            dataloader = DataLoader(
                dataset,
                num_workers=2,
                batch_size=1,
                drop_last=True)

            for data in dataloader:
                print(data)
                # outputs: [2, 5, 3, 6, 4, 7]

    Example 2: splitting data copy in each worker by :code:`worker_init_fn`

        .. code-block:: python

            import math
            import paddle
            import numpy as np
            from paddle.io import IterableDataset, DataLoader, get_worker_info

            class RangeIterableDataset(IterableDataset):
                def __init__(self, start, end):
                    self.start = start
                    self.end = end

                def __iter__(self):
                    for i in range(self.start, self.end):
                        yield np.array([i])

            dataset = RangeIterableDataset(start=2, end=9)

            def worker_init_fn(worker_id):
                worker_info = get_worker_info()

                dataset = worker_info.dataset
                start = dataset.start
                end = dataset.end
                num_per_worker = int(
                    math.ceil((end - start) / float(worker_info.num_workers)))

                worker_id = worker_info.id
                dataset.start = start + worker_id * num_per_worker
                dataset.end = min(dataset.start + num_per_worker, end)

            dataloader = DataLoader(
                dataset,
                num_workers=2,
                batch_size=1,
                drop_last=True,
                worker_init_fn=worker_init_fn)

            for data in dataloader:
                print(data) 
            # outputs: [2, 5, 3, 6, 4, 7]

    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    def __getitem__(self, idx):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__len__', self.__class__.__name__))


class TensorDataset(Dataset):
    """
    Dataset defined by a list of tensors.

    Each tensor should be in shape of [N, ...], while N is the sample number,
    and ecah tensor contains a field of sample, :code:`TensorDataset` retrieve
    each sample by indexing tensors in the 1st dimension.

    Args:
        tensors(list|tuple): A list/tuple of tensors with same shape in the 1st dimension.

    Returns:
        Dataset: a Dataset instance wrapping tensors.

    Examples:

        .. code-block:: python
        
            import numpy as np
            import paddle
            from paddle.io import TensorDataset


            input_np = np.random.random([2, 3, 4]).astype('float32')
            input = paddle.to_tensor(input_np)
            label_np = np.random.random([2, 1]).astype('int32')
            label = paddle.to_tensor(label_np)

            dataset = TensorDataset([input, label])

            for i in range(len(dataset)):
                input, label = dataset[i]
                print(input, label)

    """

    def __init__(self, tensors):
        if not framework.in_dygraph_mode():
            raise RuntimeError(
                "TensorDataset con only be used in imperative mode")
        assert all([tensor.shape[0] == tensors[0].shape[0] for tensor in tensors]), \
                "tensors not have same shape of the 1st dimension"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class ComposeDataset(Dataset):
    """
    A Dataset which composes fields of multiple datasets.

    This dataset is used for composing fileds of multiple map-style
    datasets of same length.

    Args:
        datasets(list of Dataset): List of datasets to be composed.

    Returns:
        Dataset: A Dataset which composes fields of multiple datasets.

    Examples:

        .. code-block:: python
        
            import numpy as np
            import paddle
            from paddle.io import Dataset, ComposeDataset


            # define a random dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([32]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label
                
                def __len__(self):
                    return self.num_samples

            dataset = ComposeDataset([RandomDataset(10), RandomDataset(10)])
            for i in range(len(dataset)):
                image1, label1, image2, label2 = dataset[i]
                print(image1)
                print(label1)
                print(image2)
                print(label2)
            
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "input datasets shoule not be empty"
        for i, dataset in enumerate(self.datasets):
            assert isinstance(dataset, Dataset), \
                    "each input dataset should be paddle.io.Dataset"
            assert not isinstance(dataset, IterableDataset), \
                    "paddle.io.IterableDataset not supported"
            if i > 0:
                assert len(dataset) == len(self.datasets[i-1]), \
                        "lengths of datasets should be same"

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        sample = []
        for dataset in self.datasets:
            sample.extend(to_list(dataset[idx]))
        return tuple(sample)


class ChainDataset(IterableDataset):
    """
    A Dataset which chains multiple iterable-tyle datasets.

    This dataset is used for assembling multiple datasets which should
    be :code:`paddle.io.IterableDataset`.

    Args:
        datasets(list of Dataset): List of datasets to be chainned.

    Returns:
        Dataset: A Dataset which chains fields of multiple datasets.

    Examples:

        .. code-block:: python
        
            import numpy as np
            import paddle
            from paddle.io import IterableDataset, ChainDataset


            # define a random dataset
            class RandomDataset(IterableDataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __iter__(self):
                    for i in range(10):
                        image = np.random.random([32]).astype('float32')
                        label = np.random.randint(0, 9, (1, )).astype('int64')
                        yield image, label
                
            dataset = ChainDataset([RandomDataset(10), RandomDataset(10)])
            for image, label in iter(dataset):
                print(image, label)
            
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "input datasets shoule not be empty"
        for i, dataset in enumerate(self.datasets):
            assert isinstance(dataset, IterableDataset), \
                    "ChainDataset only support paddle.io.IterableDataset"

    def __iter__(self):
        for dataset in self.datasets:
            for sample in dataset:
                yield sample


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    
    Args:
        dataset (Dataset): The whole Dataset.
        indices (sequence): Indices in the whole set selected for subset.

    Returns:
        Dataset: A Dataset which is the subset of the original dataset.
    
    Example code:

        .. code-block:: python

            import paddle
            from paddle.io import Subset

            # example 1:
            a = paddle.io.Subset(dataset=range(1, 4), indices=[0, 2])
            print(list(a))
            # [1, 3]

            # example 2:
            b = paddle.io.Subset(dataset=range(1, 4), indices=[1, 1])
            print(list(b))
            # [2, 2]
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths, generator=None):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator, optional): Generator used for the random permutation. Default is None then the DefaultGenerator is used in manual_seed().

     Returns:
        Datasets: A list of subset Datasets, which are the non-overlapping subsets of the original Dataset.

    Example code:

        .. code-block:: python

            import paddle
            from paddle.io import random_split

            a_list = paddle.io.random_split(range(10), [3, 7])
            print(len(a_list)) 
            # 2

            for idx, v in enumerate(a_list[0]):
                print(idx, v)

            # output of the first subset
            # 0 1
            # 1 3
            # 2 9

            for idx, v in enumerate(a_list[1]):
                print(idx, v)
            # output of the second subset
            # 0 5
            # 1 7
            # 2 8
            # 3 6
            # 4 0
            # 5 2
            # 6 4
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )
    # TODO(@Joejiong): support Variable or Tensor type with .tolist class member function.
    # For example var.item() and var.tolist()
    indices = paddle.randperm(sum(lengths)).numpy().tolist()
    return [
        Subset(dataset, indices[offset - length:offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def _accumulate(iterable, fn=lambda x, y: x + y):
    """
    Return running totals
    
    Args:
        iterable: any iterable object for example dataset.
        y (x): one element in the iterable object.
        fn (x, y): Defaults to lambdax.

    Yields:
        yields total from beginning iterator to current iterator.

    Example code:
    
        .. code-block:: python
        
            _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
            _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    """

    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total
