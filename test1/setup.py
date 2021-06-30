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

from setuptools import setup
from io import open
import shutil

with open('../requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()
    requirements.append('tqdm')


def readme():
    with open('api_ch.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


shutil.copytree('./table', './test1/table')
shutil.copyfile('./predict_system.py', './test1/predict_system.py')
shutil.copyfile('./utility.py', './test1/utility.py')
shutil.copytree('../ppocr', './ppocr')
shutil.copytree('../tools', './tools')
shutil.copyfile('../LICENSE', './LICENSE')

setup(
    name='paddlestructure',
    packages=['paddlestructure'],
    package_dir={'paddlestructure': ''},
    include_package_data=True,
    entry_points={"console_scripts": ["paddlestructure= paddlestructure.paddlestructure:main"]},
    version='1.0',
    install_requires=requirements,
    license='Apache License 2.0',
    description='Awesome OCR toolkits based on PaddlePaddle （8.6M ultra-lightweight pre-trained model, support training and deployment among server, mobile, embeded and IoT devices',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/PaddlePaddle/PaddleOCR',
    download_url='https://github.com/PaddlePaddle/PaddleOCR.git',
    keywords=[
        'ocr textdetection textrecognition paddleocr crnn east star-net rosetta ocrlite db chineseocr chinesetextdetection chinesetextrecognition'
    ],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
    ], )

shutil.rmtree('ppocr')
shutil.rmtree('tools')
shutil.rmtree('test1')
os.remove('LICENSE')
