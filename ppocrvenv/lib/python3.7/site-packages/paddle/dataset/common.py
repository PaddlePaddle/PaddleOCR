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

from __future__ import print_function

import requests
import hashlib
import os
import errno
import shutil
import six
import sys
import importlib
import paddle.dataset
import six.moves.cPickle as pickle
import glob
import paddle

__all__ = []

HOME = os.path.expanduser('~')
DATA_HOME = os.path.join(HOME, '.cache', 'paddle', 'dataset')


# When running unit tests, there could be multiple processes that
# trying to create DATA_HOME directory simultaneously, so we cannot
# use a if condition to check for the existence of the directory;
# instead, we use the filesystem as the synchronization mechanism by
# catching returned errors.
def must_mkdirs(path):
    try:
        os.makedirs(DATA_HOME)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


must_mkdirs(DATA_HOME)


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, module_name, md5sum, save_name=None):
    dirname = os.path.join(DATA_HOME, module_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname,
                            url.split('/')[-1]
                            if save_name is None else save_name)

    if os.path.exists(filename) and md5file(filename) == md5sum:
        return filename

    retry = 0
    retry_limit = 3
    while not (os.path.exists(filename) and md5file(filename) == md5sum):
        if os.path.exists(filename):
            sys.stderr.write("file %s  md5 %s\n" % (md5file(filename), md5sum))
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError("Cannot download {0} within retry limit {1}".
                               format(url, retry_limit))
        sys.stderr.write("Cache file %s not found, downloading %s \n" %
                         (filename, url))
        sys.stderr.write("Begin to download\n")
        try:
            r = requests.get(url, stream=True)
            total_length = r.headers.get('content-length')

            if total_length is None:
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            else:
                with open(filename, 'wb') as f:
                    chunk_size = 4096
                    total_length = int(total_length)
                    total_iter = total_length / chunk_size + 1
                    log_interval = total_iter // 20 if total_iter > 20 else 1
                    log_index = 0
                    bar = paddle.hapi.progressbar.ProgressBar(
                        total_iter, name='item')
                    for data in r.iter_content(chunk_size=chunk_size):
                        f.write(data)
                        log_index += 1
                        bar.update(log_index, {})
                        if log_index % log_interval == 0:
                            bar.update(log_index)

        except Exception as e:
            # re-try
            continue
    sys.stderr.write("\nDownload finished\n")
    sys.stdout.flush()
    return filename


def fetch_all():
    for module_name in [
            x for x in dir(paddle.dataset) if not x.startswith("__")
    ]:
        if "fetch" in dir(
                importlib.import_module("paddle.dataset.%s" % module_name)):
            getattr(
                importlib.import_module("paddle.dataset.%s" % module_name),
                "fetch")()


def split(reader, line_count, suffix="%05d.pickle", dumper=pickle.dump):
    """
    you can call the function as:

    split(paddle.dataset.cifar.train10(), line_count=1000,
        suffix="imikolov-train-%05d.pickle")

    the output files as:

    |-imikolov-train-00000.pickle
    |-imikolov-train-00001.pickle
    |- ...
    |-imikolov-train-00480.pickle

    :param reader: is a reader creator
    :param line_count: line count for each file
    :param suffix: the suffix for the output files, should contain "%d"
                means the id for each file. Default is "%05d.pickle"
    :param dumper: is a callable function that dump object to file, this
                function will be called as dumper(obj, f) and obj is the object
                will be dumped, f is a file object. Default is cPickle.dump.
    """
    if not callable(dumper):
        raise TypeError("dumper should be callable.")
    lines = []
    indx_f = 0
    for i, d in enumerate(reader()):
        lines.append(d)
        if i >= line_count and i % line_count == 0:
            with open(suffix % indx_f, "w") as f:
                dumper(lines, f)
                lines = []
                indx_f += 1
    if lines:
        with open(suffix % indx_f, "w") as f:
            dumper(lines, f)


def cluster_files_reader(files_pattern,
                         trainer_count,
                         trainer_id,
                         loader=pickle.load):
    """
    Create a reader that yield element from the given files, select
    a file set according trainer count and trainer_id

    :param files_pattern: the files which generating by split(...)
    :param trainer_count: total trainer count
    :param trainer_id: the trainer rank id
    :param loader: is a callable function that load object from file, this
                function will be called as loader(f) and f is a file object.
                Default is cPickle.load
    """

    def reader():
        if not callable(loader):
            raise TypeError("loader should be callable.")
        file_list = glob.glob(files_pattern)
        file_list.sort()
        my_file_list = []
        for idx, fn in enumerate(file_list):
            if idx % trainer_count == trainer_id:
                print("append file: %s" % fn)
                my_file_list.append(fn)
        for fn in my_file_list:
            with open(fn, "r") as f:
                lines = loader(f)
                for line in lines:
                    yield line

    return reader


def _check_exists_and_download(path, url, md5, module_name, download=True):
    if path and os.path.exists(path):
        return path

    if download:
        return paddle.dataset.common.download(url, module_name, md5)
    else:
        raise ValueError('{} not exists and auto download disabled'.format(
            path))
