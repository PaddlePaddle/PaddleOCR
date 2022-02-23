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

import numpy as np
import zipfile
import re
import random
import functools
import six

import paddle
from paddle.io import Dataset
import paddle.compat as cpt
from paddle.dataset.common import _check_exists_and_download

__all__ = []

age_table = [1, 18, 25, 35, 45, 50, 56]

URL = 'https://dataset.bj.bcebos.com/movielens%2Fml-1m.zip'
MD5 = 'c4d9eecfca2ab87c1945afe126590906'


class MovieInfo(object):
    """
    Movie id, title and categories information are stored in MovieInfo.
    """

    def __init__(self, index, categories, title):
        self.index = int(index)
        self.categories = categories
        self.title = title

    def value(self, categories_dict, movie_title_dict):
        """
        Get information from a movie.
        """
        return [[self.index], [categories_dict[c] for c in self.categories],
                [movie_title_dict[w.lower()] for w in self.title.split()]]

    def __str__(self):
        return "<MovieInfo id(%d), title(%s), categories(%s)>" % (
            self.index, self.title, self.categories)

    def __repr__(self):
        return self.__str__()


class UserInfo(object):
    """
    User id, gender, age, and job information are stored in UserInfo.
    """

    def __init__(self, index, gender, age, job_id):
        self.index = int(index)
        self.is_male = gender == 'M'
        self.age = age_table.index(int(age))
        self.job_id = int(job_id)

    def value(self):
        """
        Get information from a user.
        """
        return [[self.index], [0 if self.is_male else 1], [self.age],
                [self.job_id]]

    def __str__(self):
        return "<UserInfo id(%d), gender(%s), age(%d), job(%d)>" % (
            self.index, "M"
            if self.is_male else "F", age_table[self.age], self.job_id)

    def __repr__(self):
        return str(self)


class Movielens(Dataset):
    """
    Implementation of `Movielens 1-M <https://grouplens.org/datasets/movielens/1m/>`_ dataset.

    Args:
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' or 'test' mode. Default 'train'.
        test_ratio(float): split ratio for test sample. Default 0.1.
        rand_seed(int): random seed. Default 0.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of Movielens 1-M dataset

    Examples:

        .. code-block:: python

            import paddle
            from paddle.text.datasets import Movielens

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, category, title, rating):
                    return paddle.sum(category), paddle.sum(title), paddle.sum(rating)


            movielens = Movielens(mode='train')

            for i in range(10):
                category, title, rating = movielens[i][-3:]
                category = paddle.to_tensor(category)
                title = paddle.to_tensor(title)
                rating = paddle.to_tensor(rating)

                model = SimpleNet()
                category, title, rating = model(category, title, rating)
                print(category.numpy().shape, title.numpy().shape, rating.numpy().shape)

    """

    def __init__(self,
                 data_file=None,
                 mode='train',
                 test_ratio=0.1,
                 rand_seed=0,
                 download=True):
        assert mode.lower() in ['train', 'test'], \
            "mode should be 'train', 'test', but got {}".format(mode)
        self.mode = mode.lower()

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(data_file, URL, MD5,
                                                        'sentiment', download)

        self.test_ratio = test_ratio
        self.rand_seed = rand_seed

        np.random.seed(rand_seed)
        self._load_meta_info()
        self._load_data()

    def _load_meta_info(self):
        pattern = re.compile(r'^(.*)\((\d+)\)$')
        self.movie_info = dict()
        self.movie_title_dict = dict()
        self.categories_dict = dict()
        self.user_info = dict()
        with zipfile.ZipFile(self.data_file) as package:
            for info in package.infolist():
                assert isinstance(info, zipfile.ZipInfo)
                title_word_set = set()
                categories_set = set()
                with package.open('ml-1m/movies.dat') as movie_file:
                    for i, line in enumerate(movie_file):
                        line = cpt.to_text(line, encoding='latin')
                        movie_id, title, categories = line.strip().split('::')
                        categories = categories.split('|')
                        for c in categories:
                            categories_set.add(c)
                        title = pattern.match(title).group(1)
                        self.movie_info[int(movie_id)] = MovieInfo(
                            index=movie_id, categories=categories, title=title)
                        for w in title.split():
                            title_word_set.add(w.lower())

                for i, w in enumerate(title_word_set):
                    self.movie_title_dict[w] = i

                for i, c in enumerate(categories_set):
                    self.categories_dict[c] = i

                with package.open('ml-1m/users.dat') as user_file:
                    for line in user_file:
                        line = cpt.to_text(line, encoding='latin')
                        uid, gender, age, job, _ = line.strip().split("::")
                        self.user_info[int(uid)] = UserInfo(
                            index=uid, gender=gender, age=age, job_id=job)

    def _load_data(self):
        self.data = []
        is_test = self.mode == 'test'
        with zipfile.ZipFile(self.data_file) as package:
            with package.open('ml-1m/ratings.dat') as rating:
                for line in rating:
                    line = cpt.to_text(line, encoding='latin')
                    if (np.random.random() < self.test_ratio) == is_test:
                        uid, mov_id, rating, _ = line.strip().split("::")
                        uid = int(uid)
                        mov_id = int(mov_id)
                        rating = float(rating) * 2 - 5.0

                        mov = self.movie_info[mov_id]
                        usr = self.user_info[uid]
                        self.data.append(usr.value() + \
                                         mov.value(self.categories_dict, self.movie_title_dict) + \
                                         [[rating]])

    def __getitem__(self, idx):
        data = self.data[idx]
        return tuple([np.array(d) for d in data])

    def __len__(self):
        return len(self.data)
