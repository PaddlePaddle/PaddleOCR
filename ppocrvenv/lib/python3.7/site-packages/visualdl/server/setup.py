# Copyright (c) 2017 VisualDL Authors. All Rights Reserve.
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

from setuptools import setup

packages = [
    'visualdl', 'visualdl.onnx', 'visualdl.mock', 'visualdl.frontend.dist'
]

setup(
    name="visualdl",
    version="0.0.1",
    packages=packages,
    package_data={'visualdl.frontend.dist': ['*', 'fonts/*']},
    include_package_data=True,
    install_requires=['flask>=0.12.1'],
    url='http://www.baidu.com/',
    license='Apache 2.0')
