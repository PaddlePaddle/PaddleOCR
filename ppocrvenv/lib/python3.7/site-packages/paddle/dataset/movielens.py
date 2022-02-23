# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
Movielens 1-M dataset.

Movielens 1-M dataset contains 1 million ratings from 6000 users on 4000
movies, which was collected by GroupLens Research. This module will download
Movielens 1-M dataset from
http://files.grouplens.org/datasets/movielens/ml-1m.zip and parse training
set and test set into paddle reader creators.

"""

from __future__ import print_function

import numpy as np
import zipfile
import paddle.dataset.common
import paddle.utils.deprecated as deprecated
import re
import random
import functools
import six
import paddle.compat as cpt

__all__ = []

age_table = [1, 18, 25, 35, 45, 50, 56]

#URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
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

    def value(self):
        """
        Get information from a movie.
        """
        return [
            self.index, [CATEGORIES_DICT[c] for c in self.categories],
            [MOVIE_TITLE_DICT[w.lower()] for w in self.title.split()]
        ]

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
        return [self.index, 0 if self.is_male else 1, self.age, self.job_id]

    def __str__(self):
        return "<UserInfo id(%d), gender(%s), age(%d), job(%d)>" % (
            self.index, "M"
            if self.is_male else "F", age_table[self.age], self.job_id)

    def __repr__(self):
        return str(self)


MOVIE_INFO = None
MOVIE_TITLE_DICT = None
CATEGORIES_DICT = None
USER_INFO = None


def __initialize_meta_info__():
    fn = paddle.dataset.common.download(URL, "movielens", MD5)
    global MOVIE_INFO
    if MOVIE_INFO is None:
        pattern = re.compile(r'^(.*)\((\d+)\)$')
        with zipfile.ZipFile(file=fn) as package:
            for info in package.infolist():
                assert isinstance(info, zipfile.ZipInfo)
                MOVIE_INFO = dict()
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
                        MOVIE_INFO[int(movie_id)] = MovieInfo(
                            index=movie_id, categories=categories, title=title)
                        for w in title.split():
                            title_word_set.add(w.lower())

                global MOVIE_TITLE_DICT
                MOVIE_TITLE_DICT = dict()
                for i, w in enumerate(title_word_set):
                    MOVIE_TITLE_DICT[w] = i

                global CATEGORIES_DICT
                CATEGORIES_DICT = dict()
                for i, c in enumerate(categories_set):
                    CATEGORIES_DICT[c] = i

                global USER_INFO
                USER_INFO = dict()
                with package.open('ml-1m/users.dat') as user_file:
                    for line in user_file:
                        line = cpt.to_text(line, encoding='latin')
                        uid, gender, age, job, _ = line.strip().split("::")
                        USER_INFO[int(uid)] = UserInfo(
                            index=uid, gender=gender, age=age, job_id=job)
    return fn


def __reader__(rand_seed=0, test_ratio=0.1, is_test=False):
    fn = __initialize_meta_info__()
    np.random.seed(rand_seed)
    with zipfile.ZipFile(file=fn) as package:
        with package.open('ml-1m/ratings.dat') as rating:
            for line in rating:
                line = cpt.to_text(line, encoding='latin')
                if (np.random.random() < test_ratio) == is_test:
                    uid, mov_id, rating, _ = line.strip().split("::")
                    uid = int(uid)
                    mov_id = int(mov_id)
                    rating = float(rating) * 2 - 5.0

                    mov = MOVIE_INFO[mov_id]
                    usr = USER_INFO[uid]
                    yield usr.value() + mov.value() + [[rating]]


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def __reader_creator__(**kwargs):
    return lambda: __reader__(**kwargs)


train = functools.partial(__reader_creator__, is_test=False)
test = functools.partial(__reader_creator__, is_test=True)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def get_movie_title_dict():
    """
    Get movie title dictionary.
    """
    __initialize_meta_info__()
    return MOVIE_TITLE_DICT


def __max_index_info__(a, b):
    if a.index > b.index:
        return a
    else:
        return b


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def max_movie_id():
    """
    Get the maximum value of movie id.
    """
    __initialize_meta_info__()
    return six.moves.reduce(__max_index_info__, list(MOVIE_INFO.values())).index


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def max_user_id():
    """
    Get the maximum value of user id.
    """
    __initialize_meta_info__()
    return six.moves.reduce(__max_index_info__, list(USER_INFO.values())).index


def __max_job_id_impl__(a, b):
    if a.job_id > b.job_id:
        return a
    else:
        return b


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def max_job_id():
    """
    Get the maximum value of job id.
    """
    __initialize_meta_info__()
    return six.moves.reduce(__max_job_id_impl__,
                            list(USER_INFO.values())).job_id


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def movie_categories():
    """
    Get movie categories dictionary.
    """
    __initialize_meta_info__()
    return CATEGORIES_DICT


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def user_info():
    """
    Get user info dictionary.
    """
    __initialize_meta_info__()
    return USER_INFO


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def movie_info():
    """
    Get movie info dictionary.
    """
    __initialize_meta_info__()
    return MOVIE_INFO


def unittest():
    for train_count, _ in enumerate(train()()):
        pass
    for test_count, _ in enumerate(test()()):
        pass

    print(train_count, test_count)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Movielens",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def fetch():
    paddle.dataset.common.download(URL, "movielens", MD5)


if __name__ == '__main__':
    unittest()
