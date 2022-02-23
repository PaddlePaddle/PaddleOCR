# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
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
This module provides a client class for CERT.
"""

from __future__ import unicode_literals

import copy
import json
import logging

from baidubce import bce_base_client
from baidubce import utils
from baidubce.auth import bce_v1_signer
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_content_types
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.services.cert import cert_model

_logger = logging.getLogger(__name__)


class CertClient(bce_base_client.BceBaseClient):
    """
    CertClient
    """
    prefix = b"/v1/certificate"

    def __init__(self, config=None):
        bce_base_client.BceBaseClient.__init__(self, config)

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
            config=None,
            body_parser=None):
        config = self._merge_config(self, config)
        if body_parser is None:
            body_parser = handler.parse_json
        headers = headers or {}
        headers[http_headers.CONTENT_TYPE] = http_content_types.JSON

        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, utils.append_uri(CertClient.prefix, path), body, headers, params)

    def create_cert(self, cert_create_request, config=None):
        """
        create certificate

        :param cert_create_request: certificate base informations
        :type cert_create_request: cert_model.CertCreateRequest

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.POST, '', body=json.dumps(cert_create_request.__dict__),
            config=config)

    def list_user_certs(self, config=None):
        """
        list user's certificates

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '',
            config=config)

    def get_cert_info(self, cert_id, config=None):
        """
        get a certificate information by id

        :param cert_id: certificate id
        :type cert_id: string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/' + cert_id,
            config=config)

    def delete_cert(self, cert_id, config=None):
        """
        delete a certificate by id
        :param cert_id: certificate id
        :type cert_id: string
        :param config: None
        :type config: baidubce.BceClientConfiguration
        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.DELETE, '/' + cert_id,
            config=config)

    def replace_cert(self, cert_id, cert_create_request, config=None):
        """
        delete a certificate by id

        :param cert_id: certificate id
        :type cert_id: string

        :param cert_create_request: certificate base informations
        :type cert_create_request: cert_model.CertCreateRequest

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT, '/' + cert_id,
            body=json.dumps(cert_create_request.__dict__),
            params={'certData': ''},
            config=config)
