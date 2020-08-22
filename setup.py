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

from setuptools import setup
from io import open

with open('requirments.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()
    requirements.append('tqdm')


def readme():
    with open('doc/doc_en/whl.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='paddleocr',
    packages=['paddleocr'],
    package_dir={'paddleocr': ''},
    include_package_data=True,
    entry_points={"console_scripts": ["paddleocr= paddleocr.paddleocr:main"]},
    version='0.0.3',
    install_requires=requirements,
    license='Apache License 2.0',
    description='Awesome OCR toolkits based on PaddlePaddle （8.6M ultra-lightweight pre-trained model, support training and deployment among server, mobile, embeded and IoT devices',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Baidu PaddlePaddle',
    url='https://github.com/PaddlePaddle/PaddleOCR',
    download_url='https://github.com/PaddlePaddle/PaddleOCR.git',
    keywords=[
        'ocr textdetection textrecognition paddleocr crnn east star-net rosetta ocrlite db chineseocr chinesetextdetection chinesetextrecognition'
    ],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python', 'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
    ], )
