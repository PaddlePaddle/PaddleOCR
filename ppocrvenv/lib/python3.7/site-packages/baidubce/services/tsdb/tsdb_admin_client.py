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
This module provides a client class for TSDB.
"""

import copy
import json
import logging

from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_content_types
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.services.tsdb import tsdb_handler


_logger = logging.getLogger(__name__)


class TsdbAdminClient(BceBaseClient):
    """
    sdk client
    """
    def __init__(self, config=None):
        BceBaseClient.__init__(self, config)

    def create_database(
            self,
            client_token,
            database_name,
            ingest_datapoints_monthly,
            purchase_length,
            description=None,
            store_bytes_quota=None,
            coupon_name=None):
        """
        create_database

        :param client_token: a unique id for idempotence
        :type client_token: string
        :param database_name: name of database
        :type database_name: string
        :param description: optional, description for database
        :type description: string
        :param ingest_datapoints_monthly: max ingest datapoints count per month,unit Million
        :type ingest_datapoints_monthly: int
        :param store_bytes_quota: optional, unit GB
        :type store_bytes_quota: int
        :param purchase_length: purchase length, unit Month
        :type purchase_length: int
        :param coupon_name: optional, coupon number
        :type coupon_name: type

        :return: {database_id:,charge:,expired_time:order_id:}
        :rtype: baidubce.bce_response.BceResponse
        """

        path = b"/v1/database"
        params = {"clientToken": client_token}
        body = json.dumps({
            "databaseName": database_name.decode(),
            "description": description.decode(),
            "ingestDataPointsMonthly": ingest_datapoints_monthly,
            "storeBytesQuota": store_bytes_quota,
            "purchaseLength": purchase_length,
            "couponName": coupon_name.decode()
        }).encode('utf-8')
        return self._send_request(http_methods.POST, path=path, body=body,
                                  params=params, body_parser=tsdb_handler.parse_json)

    def delete_database(self, database_id):
        """
        delete database

        :param database_id: database id to delete
        :type database_id: string

        :return: bce_request_id
        :rtype: baidubce.bce_response.BceResponses
        """
        path = b'/v1/database/' + database_id
        return self._send_request(http_methods.DELETE, path, body_parser=tsdb_handler.parse_json)

    def get_database(self, database_id):
        """
        get database

        :param database_id: database id to delete
        :type database_id: string
        
        :return: database info
        :rtype: baidubce.bce_response.BceResponse
        """

        path = b'/v1/database/' + database_id
        return self._send_request(http_methods.GET, path, body_parser=tsdb_handler.parse_json)

    def get_all_databases(self):
        """
        get all databases
        
        :return: database dict
        :rtype: baidubce.bce_response.BceResponse
        """

        path = b'/v1/database'
        return self._send_request(http_methods.GET, path, body_parser=tsdb_handler.parse_json)

    def _merge_config(self, config):
        if config is None:
            return self.config
        else:
            new_config = copy.copy(self.config)
            new_config.merge_non_none_values(config)
            return new_config

    def _send_request(
            self, http_method, path,
            body=None,
            params=None,
            headers=None,
            config=None,
            body_parser=None):
        config = self._merge_config(config)
        if headers is None:
            headers = {http_headers.CONTENT_TYPE: http_content_types.JSON}
        if body_parser is None:
            body_parser = handler.parse_json
        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, path, body, headers, params)
