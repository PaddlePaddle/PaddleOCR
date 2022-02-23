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

import os
import json


VDL_SERVER = "https://www.paddlepaddle.org.cn/paddle/visualdl/service/server"

default_vdl_config = {
    'server_url': VDL_SERVER
}

USER_HOME = os.path.expanduser('~')
VDL_HOME = os.path.join(USER_HOME, '.visualdl')
CONF_HOME = os.path.join(VDL_HOME, 'conf')
CONFIG_PATH = os.path.join(CONF_HOME, 'config.json')


def init_vdl_config():
    if not os.path.exists(CONF_HOME):
        os.makedirs(CONF_HOME, exist_ok=True)
    if not os.path.exists(CONFIG_PATH) or 0 == os.path.getsize(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as fp:
            fp.write(json.dumps(default_vdl_config))
