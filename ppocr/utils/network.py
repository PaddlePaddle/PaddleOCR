# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import shutil
import tarfile
import requests
import os.path as osp
import paddle.distributed as dist
from tqdm import tqdm

from ppocr.utils.logging import get_logger

MODELS_DIR = os.path.expanduser("~/.paddleocr/models/")
DOWNLOAD_RETRY_LIMIT = 3


def download_with_progressbar(url, save_path):
    logger = get_logger()
    if save_path and os.path.exists(save_path):
        logger.info(f"Path {save_path} already exists. Skipping...")
        return
    else:
        # Mainly used to solve the problem of downloading data from different
        # machines in the case of multiple machines. Different nodes will download
        # data, and the same node will only download data once.
        if dist.get_rank() == 0:
            _download(url, save_path)
        else:
            while not os.path.exists(save_path):
                time.sleep(1)


def _download(url, save_path):
    """
    Download from url, save to path.

    url (str): download url
    save_path (str): download to given path
    """
    logger = get_logger()

    fname = osp.split(url)[-1]
    retry_cnt = 0

    while not osp.exists(save_path):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError(
                "Download from {} failed. " "Retry limit reached".format(url)
            )

        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            logger.info(
                "Downloading {} from {} failed {} times with exception {}".format(
                    fname, url, retry_cnt + 1, str(e)
                )
            )
            time.sleep(1)
            continue

        if req.status_code != 200:
            raise RuntimeError(
                "Downloading from {} failed with code "
                "{}!".format(url, req.status_code)
            )

        # For protecting download interupted, download to
        # tmp_file firstly, move tmp_file to save_path
        # after download finished
        tmp_file = save_path + ".tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_file, "wb") as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_file, save_path)

    return save_path


def maybe_download(model_storage_directory, url):
    # using custom model
    tar_file_name_list = [".pdiparams", ".pdiparams.info", ".pdmodel"]
    if not os.path.exists(
        os.path.join(model_storage_directory, "inference.pdiparams")
    ) or not os.path.exists(os.path.join(model_storage_directory, "inference.pdmodel")):
        assert url.endswith(".tar"), "Only supports tar compressed package"
        tmp_path = os.path.join(model_storage_directory, url.split("/")[-1])
        print("download {} to {}".format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, "r") as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if member.name.endswith(tar_file_name):
                        filename = "inference" + tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(os.path.join(model_storage_directory, filename), "wb") as f:
                    f.write(file.read())
        os.remove(tmp_path)


def maybe_download_params(model_path):
    if os.path.exists(model_path) or not is_link(model_path):
        return model_path
    else:
        url = model_path
    tmp_path = os.path.join(MODELS_DIR, url.split("/")[-1])
    print("download {} to {}".format(url, tmp_path))
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_with_progressbar(url, tmp_path)
    return tmp_path


def is_link(s):
    return s is not None and s.startswith("http")


def confirm_model_dir_url(model_dir, default_model_dir, default_url):
    url = default_url
    if model_dir is None or is_link(model_dir):
        if is_link(model_dir):
            url = model_dir
        file_name = url.split("/")[-1][:-4]
        model_dir = default_model_dir
        model_dir = os.path.join(model_dir, file_name)
    return model_dir, url
