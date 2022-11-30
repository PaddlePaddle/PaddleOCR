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

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()
    requirements.append('tqdm')


def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='PPOCRLabel',
    packages=['PPOCRLabel'],
    package_data = {'PPOCRLabel': ['libs/*','resources/strings/*','resources/icons/*']},
    package_dir={'PPOCRLabel': ''},
    include_package_data=True,
    entry_points={"console_scripts": ["PPOCRLabel= PPOCRLabel.PPOCRLabel:main"]},
    version='2.1.3',
    install_requires=requirements,
    license='Apache License 2.0',
    description='PPOCRLabelv2 is a semi-automatic graphic annotation tool suitable for OCR field, with built-in PP-OCR model to automatically detect and re-recognize data. It is written in Python3 and PyQT5, supporting rectangular box, table, irregular text and key information annotation modes. Annotations can be directly used for the training of PP-OCR detection and recognition models.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/PaddlePaddle/PaddleOCR',
    download_url='https://github.com/PaddlePaddle/PaddleOCR.git',
    keywords=[
        'ocr textdetection textrecognition paddleocr crnn east star-net rosetta ocrlite db chineseocr chinesetextdetection chinesetextrecognition'
    ],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
    ], )