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
import sys
import os
import runpy

ori_path = os.getcwd()
os.chdir("ppocr/postprocess/pse_postprocess/pse")
_saved_argv = sys.argv
try:
    sys.argv = ["setup.py", "build_ext", "--inplace"]
    runpy.run_path("setup.py", run_name="__main__")
except SystemExit as e:
    if e.code != 0:
        raise RuntimeError(
            "Cannot compile pse: {}, if your system is windows, you need to install all the default components of `desktop development using C++` in visual studio 2019+".format(
                os.path.dirname(os.path.realpath(__file__))
            )
        )
finally:
    sys.argv = _saved_argv
    os.chdir(ori_path)

from .pse import pse
