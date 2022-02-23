# -*- coding: utf-8 -*-

# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

"""
This module provides a client class for EIP group.
"""

import copy
import json
import logging
import uuid

from baidubce import utils
from baidubce import bce_base_client
from baidubce.auth import bce_v1_signer
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_methods
from baidubce.services.eip import eip_group_model
from baidubce.utils import required

_logger = logging.getLogger(__name__)


default_billing_to_purchase_created = eip_group_model.Billing('Prepaid')
default_billing_to_purchase_reserved = eip_group_model.Billing()


class EipGroupClient(bce_base_client.BceBaseClient):
    """
    EIP group sdk client
    """
    version = b'/v1'
    prefix = b'/eipgroup'

    def __init__(self, config=None):
        bce_base_client.BceBaseClient.__init__(self, config)

    def _merge_config(self, config=None):
        """
        :param config:
        :type config: baidubce.BceClientConfiguration
        :return:
        """
        if config is None:
            return self.config
        else:
            new_config = copy.copy(self.config)
            new_config.merge_non_none_values(config)
            return new_config

    def _send_request(self, http_method, path,
                      body=None, headers=None, params=None,
                      config=None, body_parser=None):
        config = self._merge_config(config)
        if body_parser is None:
            body_parser = handler.parse_json
        if headers is None:
            headers = {b'Accept': b'*/*',
                       b'Content-Type': b'application/json;charset=utf-8'}
        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, path, body, headers, params)

    @required(eip_count=int,
              bandwidth_in_mbps=int)
    def create_eip_group(self, eip_count, bandwidth_in_mbps,
                         name=None, client_token=None,
                         billing=None, config=None):
        """
        Create a shared bandwidth EIP group with specified options.
        Real-name authentication is required before creating EIP groups.
        Only prepaid EIP groups is supported.

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if client token is provided.
        :type client_token: string

        :param eip_count:
            Numbers of EIP addresses in the EIP group.
            The minimum number of public IP addresses is two,
            and the maximum number multiplies 5Mbps mustn't exceed the
            total amount of shared bandwidth package.
        :type eip_count: int

        :param bandwidth_in_mbps:
            Public Internet bandwidth in unit Mbps. For Prepaid EIP groups,
            this value must be integer between 10 and 200.
        :type bandwidth_in_mbps: int

        :param billing:
            Billing information.
        :type billing: eip_group_model.Billing

        :param name:
            The name of EIP group that will be created.
            The name, beginning with letter, should have the length between
            1 and 65 bytes, and could contain alphabets, numbers or '-_/.'.
            If not specified, the service will generate it automatically.
        :type name: string

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = self._get_path()
        if client_token is None:
            client_token = generate_client_token()
        params = {b'clientToken': client_token}
        if billing is None:
            billing = default_billing_to_purchase_created
        body = {
            'eipCount': eip_count,
            'bandwidthInMbps': bandwidth_in_mbps,
            'billing': billing.__dict__
        }
        if name is not None:
            body['name'] = name
        return self._send_request(http_methods.POST,
                                  path, body=json.dumps(body),
                                  params=params, config=config)

    def list_eip_groups(self, id=None, name=None, status=None,
                        marker=None, max_keys=None, config=None):
        """
        Return a list of EIP groups, according to the ID,
        name or status of EIP group. If not specified,
        returns a full list of EIP groups in VPC.

        :param id:
            The id of specified EIP group.
        :type id: string

        :param name:
            The name of specified EIP group.
        :type name: string

        :param status:
            The status of specified EIP group.
        :type status: string

        :param marker:
            The optional parameter marker specified in the original
            request to specify where in the results to begin listing.
            Together with the marker, specifies the list result which
            listing should begin. If the marker is not specified,
            the list result will listing from the first one.
        :type marker: string

        :param max_keys:
            The optional parameter to specifies the max number of
            list result to return.
            The default value is 1000.
        :type max_keys: int

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = self._get_path()
        params = {}
        if id is not None:
            params[b'id'] = id
        if name is not None:
            params[b'name'] = name
        if status is not None:
            params[b'status'] = status
        if marker is not None:
            params[b'marker'] = marker
        if max_keys is not None:
            params[b'maxKeys'] = max_keys
        return self._send_request(http_methods.GET, path,
                                  params=params, config=config)

    @required(id=(bytes, str))
    def get_eip_group(self, id, config=None):
        """
        Get the detail information of specified EIP group.

        :param id:
            The id of specified EIP group.
        :type id: string

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = utils.append_uri(self._get_path(), id)
        return self._send_request(http_methods.GET, path, config=config)

    @required(id=(bytes, str), name=(bytes, str))
    def update_eip_group(self, id, name, client_token=None, config=None):
        """
        Update the name of specified EIP group.

        :param id:
            The id of specified EIP group.
        :type id: string

        :param name:
            The new name of the EIP group
        :type name: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if clientToken is provided.
            If the clientToken is not specified by user,
            a random String generated by default algorithm will be used.
        :type client_token: string

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = utils.append_uri(self._get_path(), id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'update': None,
            b'clientToken': client_token
        }
        body = {
            'name': name
        }
        return self._send_request(http_methods.PUT,
                                  path, body=json.dumps(body),
                                  params=params, config=config)

    @required(id=(bytes, str), bandwidth_in_mbps=int)
    def resize_eip_group_bandwidth(self, id, bandwidth_in_mbps,
                                   client_token=None, config=None):
        """
        Resize the bandwidth of a specified EIP group.

        :param id:
            The id of specified EIP group.
        :type id: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if clientToken is provided.
            If the clientToken is not specified by user,
            a random String generated by default algorithm will be used.
        :type client_token: string

        :param bandwidth_in_mbps:
            The new bandwidth of EIP group.
            For prepaid EIP groups, this value must be integer
            between 10 and 200.
        :type bandwidth_in_mbps: int

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = utils.append_uri(self._get_path(), id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'resize': None,
            b'clientToken': client_token
        }
        body = {
            'bandwidthInMbps': bandwidth_in_mbps
        }
        return self._send_request(http_methods.PUT,
                                  path, body=json.dumps(body),
                                  params=params, config=config)

    @required(id=(bytes, str), eip_add_count=int)
    def resize_eip_group_count(self, id, eip_add_count,
                               client_token=None, config=None):
        """
        Resize the EIP count of a specified EIP group.

        :param id:
            The id of specified EIP group.
        :type id: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if clientToken is provided.
            If the clientToken is not specified by user,
            a random String generated by default algorithm will be used.
        :type client_token: string

        :param eip_add_count:
            The increase number of EIP addresses in the EIP group.
            This value must larger than zero, and the maximum number multiplies
            5Mbps mustn't exceed the total amount of shared bandwidth package.
        :type eip_add_count: int

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = utils.append_uri(self._get_path(), id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'resize': None,
            b'clientToken': client_token
        }
        body = {
            'eipAddCount': eip_add_count
        }
        return self._send_request(http_methods.PUT,
                                  path, body=json.dumps(body),
                                  params=params, config=config)

    @required(id=(bytes, str))
    def purchase_reserved_eip_group(self, id, client_token=None,
                                    billing=None, config=None):
        """
        Renew specified EIP group.
        EIP groups cannot can not be renewed during resizing process.

        :param id:
            The id of EIP group.
        :type id: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if clientToken is provided.
            If the clientToken is not specified by user,
            a random String generated by default algorithm will be used.
        :type client_token: string

        :param billing:
            Billing information.
        :type billing: eip_group_model.Billing

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = utils.append_uri(self._get_path(), id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'purchaseReserved': None,
            b'clientToken': client_token
        }
        if billing is None:
            billing = default_billing_to_purchase_reserved
        body = {
            'billing': billing.__dict__
        }
        return self._send_request(http_methods.PUT,
                                  path, body=json.dumps(body),
                                  params=params, config=config)

    @staticmethod
    def _get_path(prefix=None):
        """
        :type prefix: string
        :param prefix: path prefix
        """
        if prefix is None:
            prefix = EipGroupClient.prefix
        return utils.append_uri(EipGroupClient.version, prefix)


def generate_client_token_by_uuid():
    """
    The default method to generate the random string for client_token
    if the optional parameter client_token is not specified by the user.

    :return:
    :rtype string
    """
    return str(uuid.uuid4())


generate_client_token = generate_client_token_by_uuid
