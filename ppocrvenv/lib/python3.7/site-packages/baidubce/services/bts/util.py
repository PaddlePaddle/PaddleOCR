# Copyright 2014 Baidu, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""
This module provides some python2 and python3's compatible functions for BTS.
"""

import sys
import logging

from baidubce.exception import BceClientError
from baidubce.services.bts import PYTHON_VERSION_ERROR

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY3:
    import urllib.parse
else:
    import urllib

_logger = logging.getLogger(__name__)


def _encode(s):
    """
    _encode

    :param s: s
    :type s: string

    :return:
    :rtype string
    """
    if PY2:
        return urllib.quote(s)
    if PY3:
        return urllib.parse.quote(s)

    ex = BceClientError(PYTHON_VERSION_ERROR)
    _logger.debug(ex)
    raise ex


def _decode(s):
    """
    _decode

    :param s: s
    :type s: string

    :return:
    :rtype string
    """
    if PY2:
        return urllib.unquote(s)
    if PY3:
        return urllib.parse.unquote(s)

    ex = BceClientError(PYTHON_VERSION_ERROR)
    _logger.debug(ex)
    raise ex

