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

import logging


logger = logging.getLogger('vdl_logger')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s'))
logger.addHandler(handler)

info_logger = logging.getLogger('visualdl.info')
info_logger.setLevel(logging.INFO)
info_handler = logging.StreamHandler()
info_handler.setFormatter(logging.Formatter('%(message)s'))
info_logger.addHandler(info_handler)
info_logger.propagate = False
info = info_logger.info


def init_logger(verbose):
    level = max(logging.ERROR - verbose * 10, logging.NOTSET)

    logger.setLevel(level)
    logging.getLogger('werkzeug').setLevel(level)
