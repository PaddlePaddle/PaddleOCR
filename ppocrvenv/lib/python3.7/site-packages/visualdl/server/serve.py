#!/user/bin/env python

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

import requests
import json
import os

from visualdl.io import bfile
from visualdl.reader.reader import is_VDLRecord_file
from visualdl.utils.dir import CONFIG_PATH
from visualdl.server.log import logger


def get_server_url():
    with open(CONFIG_PATH, 'r') as fp:
        server_url = json.load(fp)['server_url']
    return server_url


def apply_for_token():
    url = get_server_url() + '/sts/'
    res = requests.post(url=url).json()

    return res


def get_url(path='', model='', **kwargs):
    server_url = get_server_url() + '/url/'
    data = json.dumps({'path': path, 'model': model})
    headers = {"Content-Type": "application/json"}
    res = requests.post(url=server_url, headers=headers, data=data).json()
    err_code = res.get('code')
    msg = res.get('msg')

    if '000000' == err_code:
        url = msg.get('url')
        return url
    else:
        logger.error(msg)
        return


def get_vdl_log_file(logdirs):
    """Get logs.

    Every dir(means `run` in vdl) has only one log(meads `actual log file`).

    Returns:
        walks: A dict like {"exp1": "vdlrecords.1587375595.log",
                            "exp2": "vdlrecords.1587375685.log"}
    """
    walks = {}
    walks_temp = {}
    for logdir in logdirs:
        for root, dirs, files in bfile.walk(logdir):
            walks.update({root: files})

    for run, tags in walks.items():
        tags_temp = [tag for tag in tags if
                     is_VDLRecord_file(path=bfile.join(run, tag), check=False)]
        tags_temp.sort(reverse=True)
        if len(tags_temp) > 0:
            walks_temp.update({run: tags_temp[0]})

    return walks_temp


def upload_to_dev(logdir=None, model=None):
    if not logdir and not model:
        logger.error("Must specify directory to upload via `--logdir` or specify model to upload via `--model`.")
        return
    walks = {}
    if logdir:
        walks = get_vdl_log_file(logdir)
        if not walks:
            logger.error("There is no valid log file in %s" % logdir)
            return

    res = apply_for_token()

    err_code = res.get('code')
    msg = res.get('msg')

    if '000000' == err_code:
        sts_ak = msg.get('sts_ak')
        sts_sk = msg.get('sts_sk')
        sts_token = msg.get('token')
        bucket_id = msg.get('dir')
    else:
        logger.error(msg)
        return

    if not sts_ak or not sts_sk or not sts_token:
        return
    bos_fs = bfile.BosConfigClient(bos_ak=sts_ak,
                                   bos_sk=sts_sk,
                                   bos_sts=sts_token)

    for key, value in walks.items():
        filename = bos_fs.join(key, value)
        bos_fs.upload_object_from_file(path=bucket_id, filename=filename)

    if model:
        if os.path.getsize(model) > 1024 * 1024 * 100:
            logger.error('Size of model must less than 100M.')
        else:
            bos_fs.upload_object_from_file(path=bucket_id, filename=model)
    url = get_url(path=bucket_id, model=model)

    print("View your visualization results at: `%s`." % url)
