# -*- coding: utf-8 -*-

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
This module provides a client class for DDC.
"""

import copy
import json

from baidubce import utils
from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_methods
from baidubce.utils import required


class DdcClient(BceBaseClient):
    """
    Ddc base sdk client
    """
    version = '/v1'
    prefix = '/ddc'

    def __init__(self, config=None):
        """
        :type config: baidubce.BceClientConfiguration
        :param config:
        """
        BceBaseClient.__init__(self, config)
        
    @required(instance_id=(str), db_name=(str), table_name=(str))
    def lazydrop_create_hard_link(self, instance_id=None, db_name=None, table_name=None, config=None):
        """
            Create hard link.

            :param instance_id:
                The id of instance
            :type instance_id: string

            :param db_name:
                The database name
            :type db_name: string

            :param table_name:
                The table name
            :type table_name: string

            :param config:
            :type config: baidubce.BceClientConfiguration

            :return: 

            :rtype baidubce.bce_response.BceResponse
        """
        path = DdcClient.version + DdcClient.prefix\
        + "/instance/" + instance_id + "/database/" + db_name + "/table/link"
        body = {}
        if table_name is not None:
            body['tableName'] = table_name
        
        return self._send_request(http_methods.POST, path, body=json.dumps(body), config=config)
    
    @required(instance_id=(str), db_name=(str), table_name=(str))
    def lazydrop_delete_hard_link(self, instance_id=None, db_name=None, table_name=None, config=None):
        """
            Delete hard link.

            :param instance_id:
                The id of instance
            :type instance_id: string

            :param db_name:
                The database name
            :type db_name: string

            :param table_name:
                The table name
            :type table_name: string

            :param config:
            :type config: baidubce.BceClientConfiguration

            :return: 

            :rtype baidubce.bce_response.BceResponse
        """
        path = DdcClient.version + DdcClient.prefix\
        + "/instance/" + instance_id + "/database/" + db_name + "/table/" + table_name + "/link"
        
        return self._send_request(http_methods.DELETE, path, config=config)
    
    @required(instance_id=(str), log_type=(str), datetime=(str))
    def list_log_by_instance_id(self, instance_id=None, log_type=None, datetime=None, config=None):
        """
            Delete hard link.

            :param instance_id:
                The id of instance
            :type instance_id: string

            :param log_type:
                LogType
            :type log_type: string

            :param datetime:
                Datetime
            :type datetime: string

            :param config:
            :type config: baidubce.BceClientConfiguration

            :return: 

            :rtype baidubce.bce_response.BceResponse
        """
        path = '/v2' + DdcClient.prefix + '/instance/' + instance_id + '/logs'
        params = {}
        if log_type is not None:
            params['logType'] = log_type
            
        if datetime is not None:
            params['datetime'] = datetime
        return self._send_request(http_methods.GET, path, params=params, config=config)
    
    @required(instance_id=(str), log_id=(str))
    def get_log_by_id(self, instance_id=None, log_id=None, download_valid_time_in_sec=None, config=None):
        """
            Delete hard link.

            :param instance_id:
                The id of instance
            :type instance_id: string

            :param log_id:
                LogId
            :type log_id: string

            :param download_valid_time_in_sec:
                downloadValidTimeInSec
            :type download_valid_time_in_sec: Integer

            :param config:
            :type config: baidubce.BceClientConfiguration

            :return: 

            :rtype baidubce.bce_response.BceResponse
        """
        path = DdcClient.version + DdcClient.prefix + "/instance/" + instance_id + "/logs/" + log_id
        params = {}
        if download_valid_time_in_sec is not None:
            params['downloadValidTimeInSec'] = download_valid_time_in_sec
        return self._send_request(http_methods.GET, path, params=params, config=config)
    
    @staticmethod
    def _get_path(prefix=None):
        """
        :type prefix: string
        :param prefix: path prefix
        """
        if prefix is None:
            prefix = DdcClient.prefix
        return utils.append_uri(DdcClient.version, prefix)

    def _merge_config(self, config):
        """

        :type config: baidubce.BceClientConfiguration
        :param config:
        :return:
        """
        if config is None:
            return self.config
        else:
            new_config = copy.copy(self.config)
            new_config.merge_non_none_values(config)
            return new_config

    def _send_request(self, http_method, path, body=None, headers=None, params=None,
                      config=None, body_parser=None):
        """

        :param http_method:
        :param path:
        :param body:
        :param headers:
        :param params:

        :type config: baidubce.BceClientConfiguration
        :param config:

        :param body_parser:

        :return: baidubce.BceResponse
        """
        config = self._merge_config(config)
        if body_parser is None:
            body_parser = handler.parse_json

        if headers is None:
            headers = {b'Accept': b'*/*', b'Content-Type': b'application/json;charset=utf-8'}

        return bce_http_client.send_request(config, bce_v1_signer.sign,
                                            [handler.parse_error, body_parser],
                                            http_method, path.encode(), body, headers, params)
