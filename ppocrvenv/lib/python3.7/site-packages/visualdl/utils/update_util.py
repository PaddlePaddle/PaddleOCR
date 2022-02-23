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

import threading
import hashlib
import requests
from visualdl import __version__
from visualdl.proto.record_pb2 import DESCRIPTOR


def md5(text):
    if isinstance(text, str):
        text = text.encode("utf8")
    md5 = hashlib.md5()
    md5.update(text)
    return md5.hexdigest()


class PbUpdater(threading.Thread):
    def __init__(self, product='normal'):
        self.product = product
        threading.Thread.__init__(self)

    def update_pb(self,
                  version=__version__,
                  md5_code=md5(str(DESCRIPTOR))
                  ):
        payload = {
            "data": {
                "version": version,
                "md5": md5_code,
                "product": self.product
            }
        }
        url = 'https://paddlepaddle.org.cn/paddlehub/stat?from=vdl'
        try:
            r = requests.post(url=url, json=payload)
            if r.json().get("update_flag", 0) == 1:
                pb_bin = r.json().get("pb_bin")
                with open('/visualdl/proto/record_pb2.py', mode='wb') as fp:
                    fp.write(pb_bin)
                    print('Update pb file successfully.')
        except Exception:
            pass

    def run(self):
        self.update_pb(version=__version__,
                       md5_code=md5(str(DESCRIPTOR))
                       )
