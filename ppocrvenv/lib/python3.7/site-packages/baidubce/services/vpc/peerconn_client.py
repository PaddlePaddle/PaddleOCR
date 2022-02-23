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
This module provides a client class for peer connection.
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
from baidubce.services.vpc import peerconn_model
from baidubce.utils import required

_logger = logging.getLogger(__name__)


default_billing_to_purchase_created = peerconn_model.Billing('Postpaid')
default_billing_to_purchase_reserved = peerconn_model.Billing()


class PeerConnClient(bce_base_client.BceBaseClient):
    """
    Peer connection sdk client
    """
    version = b'/v1'
    prefix = b'/peerconn'

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

    @required(bandwidth_in_mbps=int,
              local_vpc_id=(bytes, str),
              peer_vpc_id=(bytes, str),
              peer_region=(bytes, str))
    def create_peerconn(self, bandwidth_in_mbps, local_vpc_id,
                        peer_vpc_id, peer_region, description=None,
                        local_if_name=None, peer_account_id=None,
                        peer_if_name=None, client_token=None,
                        billing=None, config=None):
        """
        Create Peer connection.
        For peer connections within the same region, only postpaid is
        supported.
        For peer connections between different accounts, the peer connections
        are available only after the remote account accepts the connections.
        For peer connections within the same account, peer connection will be
        accepted automatically.
        There can be only one peer connection between any two VPCs.
        Peer connection endpoints cannot be the same VPC.
        If both local VPC and remote VPC are transit VPCs, peer connection
        cannot be established.

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if client token is provided.
        :type client_token: string

        :param bandwidth_in_mbps:
            Network bandwidth (in unit of Mbps) of peer connection.
        :type bandwidth_in_mbps: int

        :param description:
            Description of peer connection.
        :type description: string

        :param local_if_name:
            Name of local interface of peer connection.
        :type local_if_name: string

        :param local_vpc_id:
            Local side VPC id of peer connection.
        :type local_vpc_id: string

        :param peer_account_id:
            Remote account id of peer connection.
            Used only when the peer connection connects two different accounts.
        :type peer_account_id: string

        :param peer_vpc_id:
            Remote side VPC id of peer connection.
        :type peer_vpc_id: string

        :param peer_region:
            Remote side region of peer connection.
        :type peer_region: string

        :param peer_if_name:
            Name of remote interface of peer connection.
        :type peer_if_name: string

        :param billing:
            Billing information.
        :type billing: nat_model.Billing

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = self._get_path()
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'clientToken': client_token
        }
        if billing is None:
            billing = default_billing_to_purchase_created
        body = {
            'bandwidthInMbps': bandwidth_in_mbps,
            'localVpcId': local_vpc_id,
            'peerVpcId': peer_vpc_id,
            'peerRegion': peer_region,
            'billing': billing.__dict__
        }
        if description is not None:
            body['description'] = description
        if local_if_name is not None:
            body['localIfName'] = local_if_name
        if peer_account_id is not None:
            body['peerAccountId'] = peer_account_id
        if peer_if_name is not None:
            body['peerIfName'] = peer_if_name
        return self._send_request(http_methods.POST,
                                  path, body=json.dumps(body),
                                  params=params, config=config)

    @required(vpc_id=(bytes, str))
    def list_peerconns(self, vpc_id, marker=None,
                       max_keys=None, config=None):
        """
        Return a list of peer connections.

        :param vpc_id:
            VPC id that peer connections connect to.
        :type vpc_id: string

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
        params = {
            b'vpcId': vpc_id
        }
        if marker is not None:
            params[b'marker'] = marker
        if max_keys is not None:
            params[b'maxKeys'] = max_keys
        return self._send_request(http_methods.GET,
                                  path, params=params, config=config)

    @required(peer_conn_id=(bytes, str))
    def get_peerconn(self, peer_conn_id, config=None):
        """
        Get the detail information of specified peer connection.

        :param peer_conn_id:
            The id of specified peer connection.
        :type peer_conn_id: string

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = utils.append_uri(self._get_path(), peer_conn_id)
        return self._send_request(http_methods.GET, path, config=config)

    @required(peer_conn_id=(bytes, str), local_if_id=(bytes, str))
    def update_peerconn(self, peer_conn_id, local_if_id, description=None,
                        local_if_name=None, client_token=None, config=None):
        """
        Update the interface name or description of specified peer connection.

        :param peer_conn_id:
            The id of specified peer connection.
        :type peer_conn_id: string

        :param local_if_id:
            Local interface id of peer connection
        :type local_if_id: string

        :param description:
            Description of peer connection.
        :type description: string

        :param local_if_name:
            Name of local interface of peer connection.
        :type local_if_name: string

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
        path = utils.append_uri(self._get_path(), peer_conn_id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'clientToken': client_token
        }
        body = {
            'localIfId': local_if_id
        }
        if description is not None:
            body['description'] = description
        if local_if_name is not None:
            body['localIfName'] = local_if_name
        return self._send_request(http_methods.PUT,
                                  path, body=json.dumps(body),
                                  params=params, config=config)

    @required(peer_conn_id=(bytes, str), action=(bytes, str))
    def handle_peerconn(self, peer_conn_id, action, client_token=None,
                        config=None):
        """
        Accept or reject peer connection request.
        Timeout period of connection request is 7 days.
        When timeout or the remote side rejects the connection,
        the status of peer connection on initiator side is consulting failed.

        :param peer_conn_id:
            The id of specified peer connection.
        :type peer_conn_id: string

        :param action:
            'accept': when accepting the peer connection.
            'reject': when rejecting the peer connection.
        :type action: string

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
        path = utils.append_uri(self._get_path(), peer_conn_id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'clientToken': client_token
        }
        params[action] = None
        return self._send_request(http_methods.PUT,
                                  path, params=params, config=config)

    @required(peer_conn_id=(bytes, str))
    def delete_peerconn(self, peer_conn_id, client_token=None,
                        config=None):
        """
        Delete peer connection.
        For peer connections between different accounts,
        only initiator can perform this operation.
        One cannot delete prepaid peer connections that are not expired.
        Consulting failed prepaid peer connections can be deleted.

        :param peer_conn_id:
            The id of specified peer connection.
        :type peer_conn_id: string

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
        path = utils.append_uri(self._get_path(), peer_conn_id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'clientToken': client_token
        }
        return self._send_request(http_methods.DELETE,
                                  path, params=params, config=config)

    @required(peer_conn_id=(bytes, str), new_bandwidth_in_mbps=int)
    def resize_peerconn(self, peer_conn_id, new_bandwidth_in_mbps,
                        client_token=None, config=None):
        """
        Scale down/up the bandwidth of specified peer connection.
        For peer connections between different accounts,
        only initiator can perform this operation.
        Prepaid peer connection can only scale up.
        Postpaid peer connection can scale up or down.

        :param peer_conn_id:
            The id of specified peer connection.
        :type peer_conn_id: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if clientToken is provided.
            If the clientToken is not specified by user,
            a random String generated by default algorithm will be used.
        :type client_token: string

        :param new_bandwidth_in_mbps:
            The new bandwidth of the peer connection.
        :type new_bandwidth_in_mbps: int

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = utils.append_uri(self._get_path(), peer_conn_id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'resize': None,
            b'clientToken': client_token
        }
        body = {
            'newBandwidthInMbps': new_bandwidth_in_mbps
        }
        return self._send_request(http_methods.PUT,
                                  path, body=json.dumps(body),
                                  params=params, config=config)

    @required(peer_conn_id=(bytes, str))
    def purchase_reserved_peerconn(self, peer_conn_id, client_token=None,
                                   billing=None, config=None):
        """
        Renew specified peer connection.
        Postpaid peer connection cannot be renewed.
        For peer connections between different accounts, only the initiator can
        perform this operation.

        :param peer_conn_id:
            The id of specified peer connection.
        :type peer_conn_id: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if clientToken is provided.
            If the clientToken is not specified by user,
            a random String generated by default algorithm will be used.
        :type client_token: string

        :param billing:
            Billing information.
        :type billing: peerconn_model.Billing

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = utils.append_uri(self._get_path(), peer_conn_id)
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

    @required(peer_conn_id=(bytes, str), role=(bytes, str))
    def open_peerconn_dns_sync(self, peer_conn_id, role,
                               client_token=None, config=None):
        """
        Open DNS sync between VPCs connected by peer connection.
        DNS sync can be opened only when the status of
        peer connection is available.
        DNS sync cannot be opened when the DNS status of
        peer connection is syncing or closing.

        :param peer_conn_id:
            The id of specified peer connection.
        :type peer_conn_id: string

        :param role:
            'initiator': for VPC where peer connection is initiated.
            'acceptor': for VPC where peer connection is accepted.
        :type role: string

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
        path = utils.append_uri(self._get_path(), peer_conn_id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'open': None,
            b'role': role,
            b'clientToken': client_token
        }
        return self._send_request(http_methods.PUT,
                                  path, params=params, config=config)

    @required(peer_conn_id=(bytes, str), role=(bytes, str))
    def close_peerconn_dns_sync(self, peer_conn_id, role,
                                client_token=None, config=None):
        """
        Close DNS sync between VPCs connected by peer connection.
        DNS sync can be closed only when the status of peer connection
        is available. DNS sync cannot be closed when the DNS status
        of peer connection is syncing or closing.

        :param peer_conn_id:
            The id of specified peer connection.
        :type peer_conn_id: string

        :param role:
            'initiator': for VPC where peer connection is initiated.
            'acceptor': for VPC where peer connection is accepted.
        :type role: string

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
        path = utils.append_uri(self._get_path(), peer_conn_id)
        if client_token is None:
            client_token = generate_client_token()
        params = {
            b'close': None,
            b'role': role,
            b'clientToken': client_token
        }
        return self._send_request(http_methods.PUT,
                                  path, params=params, config=config)

    @staticmethod
    def _get_path(prefix=None):
        """
        :type prefix: string
        :param prefix: path prefix
        """
        if prefix is None:
            prefix = PeerConnClient.prefix
        return utils.append_uri(PeerConnClient.version, prefix)


def generate_client_token_by_uuid():
    """
    The default method to generate the random string for client_token
    if the optional parameter client_token is not specified by the user.

    :return:
    :rtype string
    """
    return str(uuid.uuid4())


generate_client_token = generate_client_token_by_uuid
