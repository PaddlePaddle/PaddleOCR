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
import lmdb
import cv2
import shutil
from tqdm import tqdm
import json
import numpy as np


def checkImageIsValid(image_bin):
    if image_bin is None:
        return False
    image_buf = np.fromstring(image_bin, dtype=np.uint8)
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h * img_w == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode('utf-8'), str(v).encode('utf-8'))


def convert2lmdb(data_root_dir, label_file_path, lmdb_out_dir, is_check=False):
    """Convert Icdar15 format data to Lmdb format.
    Args:
        data_root_dir: the root dir of total imgs or img subdirs.
        label_file_path: icdar2015 format label's path.
        lmdb_out_dir: output lmdb dir.
        is_check: check the image whether it is valid or not, if set True, may
            slow down the lmdb conversion speed. 
    """
    if os.path.exists(lmdb_out_dir) and os.path.isdir(lmdb_out_dir):
        while True:
            print(f'{lmdb_out_dir} already exist, delete or not? [y/n]')
            Yn = input().strip()
            if Yn in ['Y', 'y']:
                shutil.rmtree(lmdb_out_dir)
                break
            if Yn in ['N', 'n']:
                return

    os.makedirs(lmdb_out_dir)
    env = lmdb.open(lmdb_out_dir, map_size=1099511627776)
    cache = {}
    cnt = 1  # in lmdb_dataset.py, idx start from 1
    with open(label_file_path, 'r', encoding='utf-8') as fp1:
        lines = fp1.read().strip().split('\n')
        nums = len(lines)
        for i in tqdm(range(nums), desc='making lmdb...'):
            relative_img_path, label = lines[i].split('\t')
            img_path = os.path.join(data_root_dir, relative_img_path)
            if not os.path.exists(img_path):
                print(f'Img path: {img_path} isn\'t exist, continue.')
                continue
            with open(img_path, 'rb') as fp2:
                image_bin = fp2.read()
                if is_check and not checkImageIsValid(image_bin):
                    print(
                        f'Img path: {img_path} is an invalid image, continue.')
                    continue
                image_key = 'image-%09d' % cnt
                label_key = 'label-%09d' % cnt
                cache[image_key] = image_bin
                cache[label_key] = label
                if cnt % 1000 == 0:
                    writeCache(env, cache)
                    cache = {}
                cnt += 1
        cache['num-samples'] = str(nums - 1)
        writeCache(env, cache)
        print(f'Created lmdb dataset with {nums} samples successfully')
