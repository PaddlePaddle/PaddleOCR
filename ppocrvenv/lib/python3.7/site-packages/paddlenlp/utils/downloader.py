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
import os
import sys
import os.path as osp
import shutil
import json
import requests
import hashlib
import tarfile
import zipfile
import time
import uuid
import threading
from collections import OrderedDict
from .env import DOWNLOAD_SERVER, SUCCESS_STATUS, FAILED_STATUS

try:
    from tqdm import tqdm
except:

    class tqdm(object):
        def __init__(self, total=None, **kwargs):
            self.total = total
            self.n = 0

        def update(self, n):
            self.n += n
            if self.total is None:
                sys.stderr.write("\r{0:.1f} bytes".format(self.n))
            else:
                sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(
                    self.total)))
            sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.write('\n')


from .log import logger

__all__ = ['get_weights_path_from_url']

COMMUNITY_MODEL_PREFIX = "https://bj.bcebos.com/paddlenlp/models/transformers/community/"

WEIGHTS_HOME = osp.expanduser("~/.cache/paddle/hapi/weights")

DOWNLOAD_RETRY_LIMIT = 3

nlp_models = OrderedDict((
    ('RoBERTa-zh-base',
     'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz'
     ),
    ('RoBERTa-zh-large',
     'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.tar.gz'
     ),
    ('ERNIE-v2-en-base',
     'https://ernie.bj.bcebos.com/ERNIE_Base_en_stable-2.0.0.tar.gz'),
    ('ERNIE-v2-en-large',
     'https://ernie.bj.bcebos.com/ERNIE_Large_en_stable-2.0.0.tar.gz'),
    ('XLNet-cased-base',
     'https://xlnet.bj.bcebos.com/xlnet_cased_L-12_H-768_A-12.tgz'),
    ('XLNet-cased-large',
     'https://xlnet.bj.bcebos.com/xlnet_cased_L-24_H-1024_A-16.tgz'),
    ('ERNIE-v1-zh-base',
     'https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz'),
    ('ERNIE-v1-zh-base-max-len-512',
     'https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz'),
    ('BERT-en-uncased-large-whole-word-masking',
     'https://bert-models.bj.bcebos.com/wwm_uncased_L-24_H-1024_A-16.tar.gz'),
    ('BERT-en-cased-large-whole-word-masking',
     'https://bert-models.bj.bcebos.com/wwm_cased_L-24_H-1024_A-16.tar.gz'),
    ('BERT-en-uncased-base',
     'https://bert-models.bj.bcebos.com/uncased_L-12_H-768_A-12.tar.gz'),
    ('BERT-en-uncased-large',
     'https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz'),
    ('BERT-en-cased-base',
     'https://bert-models.bj.bcebos.com/cased_L-12_H-768_A-12.tar.gz'),
    ('BERT-en-cased-large',
     'https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz'),
    ('BERT-multilingual-uncased-base',
     'https://bert-models.bj.bcebos.com/multilingual_L-12_H-768_A-12.tar.gz'),
    ('BERT-multilingual-cased-base',
     'https://bert-models.bj.bcebos.com/multi_cased_L-12_H-768_A-12.tar.gz'),
    ('BERT-zh-base',
     'https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz'), ))


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def get_weights_path_from_url(url, md5sum=None):
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.
    Args:
        url (str): download url
        md5sum (str): md5 sum of download package
    
    Returns:
        str: a local path to save downloaded weights.
    Examples:
        .. code-block:: python
            from paddle.utils.download import get_weights_path_from_url
            resnet18_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams'
            local_weight_path = get_weights_path_from_url(resnet18_pretrained_weight_url)
    """
    path = get_path_from_url(url, WEIGHTS_HOME, md5sum)
    return path


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.split(url)[-1]
    fpath = fname
    return osp.join(root_dir, fpath)


def get_path_from_url(url, root_dir, md5sum=None, check_exist=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.
    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str): md5 sum of download package
    
    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    from paddle.fluid.dygraph.parallel import ParallelEnv

    assert is_url(url), "downloading from {} not a url".format(url)
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)

    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        logger.info("Found {}".format(fullpath))
    else:
        if ParallelEnv().local_rank % 8 == 0:
            fullpath = _download(url, root_dir, md5sum)
        else:
            while not os.path.exists(fullpath):
                time.sleep(1)

    if ParallelEnv().local_rank % 8 == 0:
        if tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath):
            fullpath = _decompress(fullpath)

    return fullpath


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))

        logger.info("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                with tqdm(
                        total=int(total_size),
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    return fullname


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.info("File {} md5 check failed, {}(calc) != "
                    "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True


def _md5(text):
    """
    Calculate the md5 value of the input text.
    """

    md5code = hashlib.md5(text.encode())
    return md5code.hexdigest()


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info("Decompressing {}...".format(fname))

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    return uncompressed_path


def _uncompress_file_zip(filepath):
    files = zipfile.ZipFile(filepath, 'r')
    file_list = files.namelist()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)
        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _uncompress_file_tar(filepath, mode="r:*"):
    files = tarfile.open(filepath, mode)
    file_list = files.getnames()
    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)
        files.extractall(file_dir, files.getmembers())
    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        files.extractall(file_dir, files.getmembers())
    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)

        files.extractall(os.path.join(file_dir, rootpath), files.getmembers())

    files.close()

    return uncompressed_path


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find(os.sep) < -1:
        return True
    return False


def _is_a_single_dir(file_list):
    new_file_list = []
    for file_path in file_list:
        if '/' in file_path:
            file_path = file_path.replace('/', os.sep)
        elif '\\' in file_path:
            file_path = file_path.replace('\\', os.sep)
        new_file_list.append(file_path)

    file_name = new_file_list[0].split(os.sep)[0]
    for i in range(1, len(new_file_list)):
        if file_name != new_file_list[i].split(os.sep)[0]:
            return False
    return True


class DownloaderCheck(threading.Thread):
    """
    Check the resource applicability  when downloading the models.
    """

    def __init__(self, task, command="taskflow", addition=None):
        threading.Thread.__init__(self)
        self.command = command
        self.task = task
        self.addition = addition
        self.hash_flag = _md5(str(uuid.uuid1())[9:18]) + "-" + str(
            int(time.time()))

    def uri_path(self, server_url, api):
        srv = server_url
        if server_url.endswith('/'):
            srv = server_url[:-1]
        if api.startswith('/'):
            srv += api
        else:
            api = '/' + api
            srv += api
        return srv

    def request_check(self, task, command, addition):
        if task is None:
            return SUCCESS_STATUS
        payload = {'word': self.task}
        api_url = self.uri_path(DOWNLOAD_SERVER, 'search')
        cache_path = os.path.join("～")
        if os.path.exists(cache_path):
            extra = {
                "command": self.command,
                "mtime": os.stat(cache_path).st_mtime,
                "hub_name": self.hash_flag
            }
        else:
            extra = {
                "command": self.command,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "hub_name": self.hash_flag
            }
        if addition is not None:
            extra.update({"addition": addition})
        try:
            import paddle
            payload['hub_version'] = " "
            payload['paddle_version'] = paddle.__version__.split('-')[0]
            payload['extra'] = json.dumps(extra)
            r = requests.get(api_url, payload, timeout=1).json()
            if r.get("update_cache", 0) == 1:
                return SUCCESS_STATUS
            else:
                return FAILED_STATUS
        except Exception as err:
            return FAILED_STATUS

    def run(self):
        self.request_check(self.task, self.command, self.addition)
