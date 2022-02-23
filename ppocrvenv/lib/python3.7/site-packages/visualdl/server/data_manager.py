# Copyright (c) 2020 VisualDL Authors. All Rights Reserve.
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
# =======================================================================

import threading
import random
import collections

DEFAULT_PLUGIN_MAXSIZE = {
    "scalar": 1000,
    "image": 10,
    "histogram": 100,
    "embeddings": 50000000,
    "audio": 10,
    "pr_curve": 300,
    "roc_curve": 300,
    "meta_data": 100,
    "text": 10,
    "hyper_parameters": 10000
}


class Reservoir(object):
    """A map-to-arrays dict, with deterministic Reservoir Sampling.

    Store each reservoir bucket by key, and each bucket is a list sampling
    with reservoir algorithm.
    """

    def __init__(self, max_size, seed=0):
        """Creates a new reservoir.

        Args:
            max_size: The number of values to keep in the reservoir for each tag,
                if max_size is zero, all values will be kept in bucket.
            seed: The seed to initialize a random.Random().
            num_item_index: The index of data to add.

        Raises:
          ValueError: If max_size is not a nonnegative integer.
        """
        if max_size < 0 or max_size != round(max_size):
            raise ValueError("Max_size must be nonnegative integer.")
        self._max_size = max_size
        self._buckets = collections.defaultdict(
            lambda: _ReservoirBucket(max_size=self._max_size,
                                     random_instance=random.Random(seed))
        )
        self._mutex = threading.Lock()

    @property
    def keys(self):
        """Return all keys in self._buckets.

        Returns:
            All keys in reservoir buckets.
        :return:
        """
        with self._mutex:
            return list(self._buckets.keys())

    def _exist_in_keys(self, key):
        """Determine if key exists.

        Args:
            key: Key to determine if exists.

        Returns:
            True if key exists in buckets.keys, otherwise False.
        """
        return True if key in self._buckets.keys() else False

    def exist_in_keys(self, run, tag):
        """Determine if run_tag exists.

        For usage habits of VisualDL, actually call self._exist_in_keys()

        Args:
            run: Identity of one tablet.
            tag: Identity of one record in tablet.

        Returns:
            True if run_tag exists in buckets.keys, otherwise False.
        """
        key = run + "/" + tag
        return self._exist_in_keys(key)

    def _get_num_items_index(self, key):
        keys = self.keys
        if key not in keys:
            raise KeyError("Key %s not in buckets.keys()" % key)
        return self._buckets[key].num_items_index

    def get_num_items_index(self, run, tag):
        key = run + "/" + tag
        return self._get_num_items_index(key)

    def _get_items(self, key):
        """Get items with tag "key"

        Args:
            key: Key to finding bucket in reservoir buckets.

        Returns:
            One bucket in reservoir buckets by key.
        """
        keys = self.keys
        with self._mutex:
            if key not in keys:
                raise KeyError("Key %s not in buckets.keys()" % key)
            return self._buckets[key].items

    def get_items(self, run, tag):
        """Get items with tag 'run_tag'

        For usage habits of VisualDL, actually call self._get_items()

        Args:
            run: Identity of one tablet.
            tag: Identity of one record in tablet.

        Returns:
            One bucket in reservoir buckets by run and tag.
        """
        key = run + "/" + tag
        return self._get_items(key)

    def _add_item(self, key, item):
        """Add a new item to reservoir buckets with given tag as key.

        If bucket with key has not yet reached full size, each item will be
        added.

        If bucket with key is full, each item will be added with same
        probability.

        Add new item to buckets will always valid because self._buckets is a
        collection.defaultdict.

        Args:
            key: Tag of one bucket to add new item.
            item: New item to add to bucket.
        """
        with self._mutex:
            self._buckets[key].add_item(item)

    def _add_scalar_item(self, key, item):
        """Add a new scalar item to reservoir buckets with given tag as key.

        If bucket with key has not yet reached full size, each item will be
        added.

        If bucket with key is full, each item will be added with same
        probability.

        Add new item to buckets will always valid because self._buckets is a
        collection.defaultdict.

        Args:
            key: Tag of one bucket to add new item.
            item: New item to add to bucket.
        """
        with self._mutex:
            self._buckets[key].add_scalar_item(item)

    def add_item(self, run, tag, item):
        """Add a new item to reservoir buckets with given tag as key.

        For usage habits of VisualDL, actually call self._add_items()

        Args:
            run: Identity of one tablet.
            tag: Identity of one record in tablet.
            item: New item to add to bucket.
        """
        key = run + "/" + tag
        self._add_item(key, item)

    def add_scalar_item(self, run, tag, item):
        """Add a new scalar item to reservoir buckets with given tag as key.

        For usage habits of VisualDL, actually call self._add_items()

        Args:
            run: Identity of one tablet.
            tag: Identity of one record in tablet.
            item: New item to add to bucket.
        """
        key = run + "/" + tag
        self._add_scalar_item(key, item)

    def _cut_tail(self, key):
        with self._mutex:
            self._buckets[key].cut_tail()

    def cut_tail(self, run, tag):
        """Pop the last item in reservoir buckets.

        Sometimes the tail of the retrieved data is abnormal 0. This
        method is used to handle this problem.

        Args:
            run: Identity of one tablet.
            tag: Identity of one record in tablet.
        """
        key = run + "/" + tag
        self._cut_tail(key)


class _ReservoirBucket(object):
    """Data manager for sampling data, use reservoir sampling.
    """

    def __init__(self, max_size, random_instance=None):
        """Create a _ReservoirBucket instance.

        Args:
            max_size: The maximum size of reservoir bucket. If max_size is
                zero, the bucket has unbounded size.
            random_instance: The random number generator. If not specified,
                default to random.Random(0)
            num_item_index: The index of data to add.

        Raises:
            ValueError: If args max_size is not a nonnegative integer.
        """
        if max_size < 0 or max_size != round(max_size):
            raise ValueError("Max_size must be nonnegative integer.")
        self._max_size = max_size
        self._random = random_instance if random_instance is not None else \
            random.Random(0)
        self._items = []
        self._mutex = threading.Lock()
        self._num_items_index = 0

        self.max_scalar = None
        self.min_scalar = None

    def add_item(self, item):
        """ Add an item to bucket, replacing an old item with probability.

        Use reservoir sampling to add a new item to sampling bucket,
        each item in a steam has same probability stay in the bucket.

        Args:
            item: The item to add to reservoir bucket.
        """
        with self._mutex:
            if len(self._items) < self._max_size or self._max_size == 0:
                self._items.append(item)
            else:
                r = self._random.randint(1, self._num_items_index)
                if r < self._max_size:
                    self._items.pop(r)
                    self._items.append(item)
                else:
                    self._items[-1] = item
            self._num_items_index += 1

    def add_scalar_item(self, item):
        """ Add an scalar item to bucket, replacing an old item with probability.

        Use reservoir sampling to add a new item to sampling bucket,
        each item in a steam has same probability stay in the bucket.

        Args:
            item: The item to add to reservoir bucket.
        """
        with self._mutex:
            if not self.max_scalar or self.max_scalar.value < item.value:
                self.max_scalar = item
            if not self.min_scalar or self.min_scalar.value > item.value:
                self.min_scalar = item

            if len(self._items) < self._max_size or self._max_size == 0:
                self._items.append(item)
            else:
                if item.id == self.min_scalar.id or item.id == self.max_scalar.id:
                    r = self._random.randint(1, self._max_size - 1)
                else:
                    r = self._random.randint(1, self._num_items_index)
                if r < self._max_size:
                    if self._items[r].id == self.min_scalar.id or self._items[r].id == self.max_scalar.id:
                        if r - 1 > 0:
                            r = r - 1
                        elif r + 1 < self._max_size:
                            r = r + 1
                    self._items.pop(r)
                    self._items.append(item)
                else:
                    self._items[-1] = item

            self._num_items_index += 1

    @property
    def items(self):
        """Get self._items

        Returns:
            All items.
        """
        with self._mutex:
            return self._items

    @property
    def num_items_index(self):
        with self._mutex:
            return self._num_items_index

    def cut_tail(self):
        """Pop the last item in reservoir buckets.

        Sometimes the tail of the retrieved data is abnormal 0. This
        method is used to handle this problem.
        """
        with self._mutex:
            self._items.pop()
            self._num_items_index -= 1


class DataManager(object):
    """Data manager for all plugin.
    """

    def __init__(self):
        """Create a data manager for all plugin.

        All kinds of plugin has own reservoir, stored in a dict with plugin
        name as key.

        """
        self._reservoirs = {
            "scalar":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["scalar"]),
            "histogram":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["histogram"]),
            "image":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["image"]),
            "embeddings":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["embeddings"]),
            "audio":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["audio"]),
            "pr_curve":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["pr_curve"]),
            "roc_curve":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["roc_curve"]),
            "meta_data":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["meta_data"]),
            "text":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["text"]),
            "hyper_parameters":
            Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE["hyper_parameters"])
        }
        self._mutex = threading.Lock()

    def add_reservoir(self, plugin):
        """Add reservoir to reservoirs.

        Every reservoir is attached to one plugin.

        Args:
            plugin: Key to get one reservoir bucket for one specified plugin.
        """
        with self._mutex:
            if plugin not in self._reservoirs.keys():
                self._reservoirs.update({
                    plugin:
                    Reservoir(max_size=DEFAULT_PLUGIN_MAXSIZE[plugin])
                })

    def get_reservoir(self, plugin):
        """Get reservoir by plugin as key.

        Args:
            plugin: Key to get one reservoir bucket for one specified plugin.

        Returns:
            Reservoir bucket for plugin.
        """
        with self._mutex:
            if plugin not in self._reservoirs.keys():
                raise KeyError("Key %s not in reservoirs." % plugin)
            return self._reservoirs[plugin]

    def add_item(self, plugin, run, tag, item):
        """Add item to one plugin reservoir bucket.

        Use 'run', 'tag' for usage habits of VisualDL.

        Args:
            plugin: Key to get one reservoir bucket.
            run: Each tablet has different 'run'.
            tag: Tag will be used to generate paths of tablets.
            item: The item to add to reservoir bucket.
        """
        with self._mutex:
            if 'scalar' == plugin:
                self._reservoirs[plugin].add_scalar_item(run, tag, item)
            else:
                self._reservoirs[plugin].add_item(run, tag, item)

    def get_keys(self):
        """Get all plugin buckets name.

        Returns:
            All plugin keys.
        """
        with self._mutex:
            return self._reservoirs.keys()


default_data_manager = DataManager()
