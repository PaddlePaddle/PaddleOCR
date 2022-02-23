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

import io
import copy
import json
import logging
import gzip

from baidubce import bce_client_configuration
from baidubce import utils
from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_content_types
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.services.tsdb import tsdb_handler



_logger = logging.getLogger(__name__)


class TsdbClient(BceBaseClient):
    """
    sdk client
    """
    def __init__(self, config, database=None):
        self.database = database
        BceBaseClient.__init__(self, config)

    def write_datapoints(self, datapoints, use_gzip=True):
        """
        write datapoints

        :param datapoints: a list of datapoint dict
        :type datapoints: list
        :param use_gzip: open gzip compress
        :type use_gzip: boolean
        """

        path = b'/v1/datapoint'
        body = json.dumps({"datapoints": datapoints}).encode('utf-8')
        headers={http_headers.CONTENT_TYPE: http_content_types.JSON}
        if use_gzip:
            body = self._gzip_compress(body)
            headers[http_headers.CONTENT_ENCODING] = b'gzip'
        return self._send_request(
                http_methods.POST,
                path=path,
                body=body,
                headers=headers,
                body_parser=tsdb_handler.parse_json
            )

    def get_metrics(self):
        """
        list metrics

        :return: a list of metric
        :rtype: baidubce.bce_response.BceResponse
        """

        path = b"/v1/metric"
        return self._send_request(http_methods.GET, path=path, body_parser=tsdb_handler.parse_json)

    def get_fields(self, metric):
        """
        get fields

        :type metric: string
        :param metric:

        :return: field dict. {field1:{type: 'Number'},field2:{type: 'String'}}
        :rtype: baidubce.bce_response.BceResponse
        """
        metric = utils.convert_to_standard_string(metric)
        path = b'/v1/metric/' + metric + b'/field'
        return self._send_request(http_methods.GET, path=path, body_parser=tsdb_handler.parse_json)

    def get_tags(self, metric):
        """
        get tags

        :type metric: string
        :param metric:

        :return: {tagk1:[tagk11,tagk21,..],tagk2:[tagk21,tagk22,..]..}
        :rtype: baidubce.bce_response.BceResponse
        """
        metric = utils.convert_to_standard_string(metric)
        path = b'/v1/metric/' + metric + b'/tag'
        return self._send_request(http_methods.GET, path=path, body_parser=tsdb_handler.parse_json)

    def get_datapoints(self, query_list, disable_presampling=False):
        """
        query datapoints

        :param query_list: a list of query dict
        :type query_list: list
        :param disable_presampling: open of close presampling result query
        :type disable_presampling: boolean

        :return: a list of result dict
        :rtype: baidubce.bce_response.BceResponse
        """

        path = b'/v1/datapoint'
        params = {'query': '', 'disablePresampling': disable_presampling}
        body = json.dumps({"queries": query_list})
        return self._send_request(http_methods.PUT, path=path, params=params,
                body=body, body_parser=tsdb_handler.parse_json)

    def get_rows_with_sql(self, statement):
        """
        get_rows_with_sql

        :param statement: sql statement
        :type statement: string

        :return: {rows:[[],[],...], columns: []}
        :rtype: baidubce.bce_response.BceResponse
        """

        path = b'/v1/row'
        params = {'sql': statement}
        return self._send_request(http_methods.GET, path=path, params=params,
                body_parser=tsdb_handler.parse_json)

    def generate_pre_signed_url(self,
                                query_list,
                                timestamp=0,
                                expiration_in_seconds=1800,
                                disable_presampling=False,
                                headers=None,
                                headers_to_sign=None,
                                protocol=None,
                                config=None):
        """
        Get an authorization url with expire time

        :type timestamp: int
        :param timestamp: None

        :type expiration_in_seconds: int
        :param expiration_in_seconds: None

        :type options: dict
        :param options: None

        :return:
            **URL string**
        """

        path = b'/v1/datapoint'
        params = {
            'query': json.dumps({"queries": query_list}),
            'disablePresampling': disable_presampling
        }
        return self._generate_pre_signed_url(path, timestamp, expiration_in_seconds,
                        params, headers, headers_to_sign, protocol, config)

    def generate_pre_signed_url_with_sql(self,
                                statement,
                                timestamp=0,
                                expiration_in_seconds=1800,
                                headers=None,
                                headers_to_sign=None,
                                protocol=None,
                                config=None):
        """
        Get an authorization url with sql

        :type timestamp: int
        :param timestamp: None

        :type expiration_in_seconds: int
        :param expiration_in_seconds: None

        :type options: dict
        :param options: None

        :return:
            **URL string**
        """

        path = b'/v1/row'
        params = {'sql': statement}
        return self._generate_pre_signed_url(path, timestamp, expiration_in_seconds,
                        params, headers, headers_to_sign, protocol, config)

    def _generate_pre_signed_url(
            self, path, timestamp=0,
            expiration_in_seconds=1800,
            params=None,
            headers=None,
            headers_to_sign=None,
            protocol=None,
            config=None):
        """
        Get an authorization url with expire time

        :type timestamp: int
        :param timestamp: None

        :type expiration_in_seconds: int
        :param expiration_in_seconds: None

        :type options: dict
        :param options: None

        :return:
            **URL string**
        """

        config = self._merge_config(config)
        headers = headers or {}
        params = params or {}
        # specified protocol > protocol in endpoint > default protocol
        endpoint_protocol, endpoint_host, endpoint_port = utils.parse_host_port(
                config.endpoint, config.protocol)
        protocol = protocol or endpoint_protocol

        full_host = endpoint_host
        if endpoint_port != config.protocol.default_port:
            full_host += b':' + str(endpoint_port)
        headers[http_headers.HOST] = full_host

        params[http_headers.AUTHORIZATION.lower()] = bce_v1_signer.sign(
            config.credentials,
            http_methods.GET,
            path,
            headers,
            params,
            timestamp,
            expiration_in_seconds,
            headers_to_sign)
        
        return "%s://%s%s?%s" % (protocol.name,
                                 full_host.decode(),
                                 path.decode(),
                                 utils.get_canonical_querystring(params, False).decode())
    def _gzip_compress(self, str):
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="w") as f:
            f.write(str)
        return out.getvalue()

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
            headers=None,
            params=None,
            config=None,
            body_parser=None):
        config = self._merge_config(config)
        if headers is None:
            headers = {http_headers.CONTENT_TYPE: http_content_types.JSON}
        if body_parser is None:
            body_parser = handler.parse_json
        if self.database is not None:
            if params is None:
                params = {}
            params.update({b'database': self.database})
        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, path, body, headers, params)
