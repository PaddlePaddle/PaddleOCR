# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
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
This module provides a client class for CDN.
"""

import copy
import logging
import uuid

from baidubce import bce_base_client
from baidubce.auth import bce_v1_signer
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_methods
from baidubce.utils import required

_logger = logging.getLogger(__name__)


class DumapClient(bce_base_client.BceBaseClient):
    """
    DumapClient
    """

    def __init__(self, config=None):
        bce_base_client.BceBaseClient.__init__(self, config)

    @required(app_id=(bytes, str), uri=(bytes, str), params=dict)
    def call_open_api(self, app_id=None, uri=None, params=None, body=None, method=b'GET', config=None):
        """
        call open_api
        :param app_id: app_id
        :type app_id: string
        :param uri: open api uri
        :type uri: string
        :param params: dict
        :type params:request params
        :param body: request body (default: None)
        :type body: string
        :param method: http method (default GET)
        :type method: http_methods
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
                """
        response = self._send_request(
            http_method=method,
            path=uri,
            params=params,
            body=body,
            headers={b'x-app-id': app_id},
            config=config
        )

        if response.body:
            return response.body.decode("utf-8")
        else:
            return response

    @staticmethod
    def _merge_config(self, config):
        if config is None:
            return self.config
        else:
            new_config = copy.copy(self.config)
            new_config.merge_non_none_values(config)
            return new_config

    def _send_request(
            self, http_method, path,
            body=None, headers=None, params=None,
            config=None):
        config = self._merge_config(self, config)
        headers[b'x-bce-request-id'] = uuid.uuid4()
        headers[b'Content-Type'] = b'application/json;charset=utf-8'

        return bce_http_client.send_request(
            config, sign_wrapper([b'host', b'x-bce-date', b'x-bce-request-id', b'x-app-id']),
            [handler.parse_error, parse_none],
            http_method, path, body, headers, params)


def sign_wrapper(headers_to_sign):
    """wrapper the bce_v1_signer.sign()."""

    def _wrapper(credentials, http_method, path, headers, params):
        return bce_v1_signer.sign(credentials, http_method, path, headers, params,
                                  headers_to_sign=headers_to_sign)

    return _wrapper


def parse_none(http_response, response):
    """If the body is not empty, convert it to a python object and set as the value of
    response.body. http_response is always closed if no error occurs.

    :param http_response: the http_response object returned by HTTPConnection.getresponse()
    :type http_response: httplib.HTTPResponse

    :param response: general response object which will be returned to the caller
    :type response: baidubce.BceResponse

    :return: always true
    :rtype bool
    """
    body = http_response.read()
    if body:
        response.body = body
    http_response.close()
    return True
