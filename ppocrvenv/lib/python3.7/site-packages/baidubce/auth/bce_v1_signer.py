# Copyright 2014 Baidu, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.

"""
This module provides authentication functions for bce services.
"""
from __future__ import absolute_import
from builtins import str
from builtins import bytes
import hashlib
import hmac
import logging

from baidubce.http import http_headers
from baidubce import utils
from baidubce import compat


_logger = logging.getLogger(__name__)


def _get_canonical_headers(headers, headers_to_sign=None):
    headers = headers or {}

    if headers_to_sign is None or len(headers_to_sign) == 0:
        headers_to_sign = set([b"host",
                               b"content-md5",
                               b"content-length",
                               b"content-type"])
    result = []
    for k in headers:
        k_lower = k.strip().lower()
        value = utils.convert_to_standard_string(headers[k]).strip()
        if k_lower.startswith(http_headers.BCE_PREFIX) \
                or k_lower in headers_to_sign:
            str_tmp = b"%s:%s" % (utils.normalize_string(k_lower), utils.normalize_string(value))
            result.append(str_tmp)
    result.sort()
    return (b'\n').join(result)


def sign(credentials, http_method, path, headers, params,
         timestamp=0, expiration_in_seconds=1800, headers_to_sign=None):
    """
    Create the authorization
    """

    _logger.debug('Sign params: %s %s %s %s %d %d %s' % (
        http_method, path, headers, params, timestamp, expiration_in_seconds, headers_to_sign))

    headers = headers or {}
    params = params or {}

    sign_key_info = b'bce-auth-v1/%s/%s/%d' % (
        credentials.access_key_id,
        utils.get_canonical_time(timestamp),
        expiration_in_seconds)
    sign_key = hmac.new(
        credentials.secret_access_key,
        sign_key_info,
        hashlib.sha256).hexdigest()

    canonical_uri = path
    canonical_querystring = utils.get_canonical_querystring(params, True)

    canonical_headers = _get_canonical_headers(headers, headers_to_sign)
    string_to_sign = (b'\n').join([
        http_method, canonical_uri, 
        canonical_querystring, canonical_headers
        ])
    sign_result = hmac.new(compat.convert_to_bytes(sign_key), string_to_sign, hashlib.sha256).hexdigest()
    # convert to bytes
    sign_result = compat.convert_to_bytes(sign_result)

    if headers_to_sign:
        result = b'%s/%s/%s' % (sign_key_info, (b';').join(headers_to_sign), sign_result)
    else:
        result = b'%s//%s' % (sign_key_info, sign_result)

    _logger.debug('sign_key=[%s] sign_string=[%d bytes][ %s ]' %
                  (sign_key, len(string_to_sign), string_to_sign))
    _logger.debug('result=%s' % result)
    return result
