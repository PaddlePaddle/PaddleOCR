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
This module provides a client class for BTS.
"""

import copy
import json
import logging

from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.exception import BceClientError
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_content_types
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.services import bts
from baidubce.services.bts import INVALID_ARGS_ERROR
from baidubce.services.bts.model import batch_query_row_args_2_dict
from baidubce.services.bts.model import create_instance_args_2_dict
from baidubce.services.bts.model import CreateInstanceArgs
from baidubce.services.bts.model import create_table_args_2_dict
from baidubce.services.bts.model import update_table_args_2_dict
from baidubce.services.bts.model import query_row_args_2_dict
from baidubce.services.bts.model import scan_args_2_dict
from baidubce.services.bts.util import _decode
from baidubce.services.bts.util import _encode

_logger = logging.getLogger(__name__)


class BtsClient(BceBaseClient):
    """
    BTS Client
    """
    def __init__(self, config=None):
        BceBaseClient.__init__(self, config)

    # ------------- instance operation -----------
    def create_instance(self, instance_name, create_instance_args=None, config=None):
        """
        create instance

        :param instance_name: instance name
        :type instance_name: string
        :param create_instance_args: arguments for create instance
        :type create_instance_args: CreateInstanceArgs
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = bts.URL_PREFIX + b"/" + instance_name
        if create_instance_args is None:
            create_instance_args = CreateInstanceArgs()
        return self._send_request(http_methods.PUT, path=path, config=config,
                                  body=json.dumps(create_instance_args, default=create_instance_args_2_dict),
                                  headers={http_headers.CONTENT_TYPE: http_content_types.JSON})

    def drop_instance(self, instance_name, config=None):
        """
        drop instance

        :param instance_name: instance name
        :type instance_name: string
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = bts.URL_PREFIX + b"/" + instance_name
        return self._send_request(http_methods.DELETE, path=path, config=config)

    def list_instances(self, config=None):
        """
        list instances

        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b"/v1/instances"
        return self._send_request(http_methods.GET, path=path, config=config)

    def show_instance(self, instance_name, config=None):
        """
        show instance

        :param instance_name: instance name
        :type instance_name: string
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = bts.URL_PREFIX + b"/" + instance_name
        return self._send_request(http_methods.GET, path=path, config=config)

    # ------------- table operation -----------
    def create_table(self, instance_name, table_name, create_table_args, config=None):
        """
        create table

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param create_table_args: arguments for create table
        :type create_table_args: CreateTableArgs
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name
        return self._send_request(http_methods.PUT, path=path, config=config,
                                  body=json.dumps(create_table_args, default=create_table_args_2_dict),
                                  headers={http_headers.CONTENT_TYPE: http_content_types.JSON})

    def update_table(self, instance_name, table_name, update_table_args, config=None):
        """
        update table

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param update_table_args: arguments for update table
        :type update_table_args: UpdateTableArgs
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name
        return self._send_request(http_methods.PUT, path=path, config=config,
                                  body=json.dumps(update_table_args, default=update_table_args_2_dict),
                                  headers={http_headers.CONTENT_TYPE: http_content_types.JSON})

    def drop_table(self, instance_name, table_name, config=None):
        """
        drop table

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name
        return self._send_request(http_methods.DELETE, path=path, config=config)

    def show_table(self, instance_name, table_name, config=None):
        """
        show table

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name
        return self._send_request(http_methods.GET, path=path, config=config)

    def list_tables(self, instance_name, config=None):
        """
        list tables

        :param instance_name: instance name
        :type instance_name: string
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = bts.URL_PREFIX + b"/" + instance_name + b"/tables"
        return self._send_request(http_methods.GET, path=path, config=config)

    # ------------- row operation -----------
    def put_row(self, instance_name, table_name, put_row_args, config=None):
        """
        put row

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param put_row_args: arguments for put row
        :type put_row_args: Row
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        if put_row_args is None or put_row_args.rowkey == "":
            ex = BceClientError(INVALID_ARGS_ERROR)
            _logger.debug(ex)
            raise ex
        try:
            put_row_args.rowkey = _encode(put_row_args.rowkey)
            for i in range(len(put_row_args.cells)):
                put_row_args.cells[i]["value"] = _encode(put_row_args.cells[i]["value"])
        except Exception as ex:
            raise ex

        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name + b"/row"
        return self._send_request(http_methods.PUT, path=path, config=config,
                                  body=json.dumps(put_row_args.__dict__),
                                  headers={http_headers.CONTENT_TYPE: http_content_types.JSON})

    def batch_put_row(self, instance_name, table_name, batch_put_row_args, config=None):
        """
        batch put row

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param batch_put_row_args: arguments for batch put row
        :type batch_put_row_args: BatchPutRowArgs
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        if batch_put_row_args is None:
            ex = BceClientError(INVALID_ARGS_ERROR)
            _logger.debug(ex)
            raise ex
        try:
            for i in range(len(batch_put_row_args.rows)):
                if batch_put_row_args.rows[i]["rowkey"] == "":
                    ex = BceClientError(INVALID_ARGS_ERROR)
                    _logger.debug(ex)
                    raise ex
                batch_put_row_args.rows[i]["rowkey"] = _encode(batch_put_row_args.rows[i]["rowkey"])
                for j in range(len(batch_put_row_args.rows[i]["cells"])):
                    batch_put_row_args.rows[i]["cells"][j]["value"] = \
                        _encode(batch_put_row_args.rows[i]["cells"][j]["value"])
        except Exception as ex:
            raise ex

        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name + b"/rows"
        return self._send_request(http_methods.PUT, path=path, config=config,
                                  body=json.dumps(batch_put_row_args.__dict__),
                                  headers={http_headers.CONTENT_TYPE: http_content_types.JSON})

    def delete_row(self, instance_name, table_name, delete_row_args, config=None):
        """
        delete row

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param delete_row_args: arguments for delete row
        :type delete_row_args: QueryRowArgs
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        if delete_row_args is None or delete_row_args.rowkey == "":
            ex = BceClientError(INVALID_ARGS_ERROR)
            _logger.debug(ex)
            raise ex
        try:
            delete_row_args.rowkey = _encode(delete_row_args.rowkey)
        except Exception as ex:
            raise ex

        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name + b"/row"
        return self._send_request(http_methods.DELETE, path=path, config=config,
                                  body=json.dumps(delete_row_args, default=query_row_args_2_dict),
                                  headers={http_headers.CONTENT_TYPE: http_content_types.JSON})

    def batch_delete_row(self, instance_name, table_name, batch_delete_row_args, config=None):
        """
        batch delete row

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param batch_delete_row_args: arguments for batch delete row
        :type batch_delete_row_args: BatchQueryRowArgs
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        if batch_delete_row_args is None:
            ex = BceClientError(INVALID_ARGS_ERROR)
            _logger.debug(ex)
            raise ex
        try:
            for i in range(len(batch_delete_row_args.rows)):
                if batch_delete_row_args.rows[i]["rowkey"] == "":
                    ex = BceClientError(INVALID_ARGS_ERROR)
                    _logger.debug(ex)
                    raise ex
                batch_delete_row_args.rows[i]["rowkey"] = _encode(batch_delete_row_args.rows[i]["rowkey"])
        except Exception as ex:
            raise ex

        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name + b"/rows"
        return self._send_request(http_methods.DELETE, path=path, config=config,
                                  body=json.dumps(batch_delete_row_args, default=batch_query_row_args_2_dict),
                                  headers={http_headers.CONTENT_TYPE: http_content_types.JSON})

    def get_row(self, instance_name, table_name, get_row_args, config=None):
        """
        get row

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param get_row_args: arguments for get row
        :type get_row_args: QueryRowArg
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        try:
            if get_row_args is None or get_row_args.rowkey == "":
                ex = BceClientError(INVALID_ARGS_ERROR)
                _logger.debug(ex)
                raise ex
            get_row_args.rowkey = _encode(get_row_args.rowkey)
        except Exception as ex:
            raise ex

        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name + b"/row"
        response = self._send_request(http_methods.GET, path=path, config=config,
                                      body=json.dumps(get_row_args, default=query_row_args_2_dict),
                                      headers={http_headers.CONTENT_TYPE: http_content_types.JSON})
        try:
            if response.result is not None:
                response.result[0].rowkey = _decode(str(response.result[0].rowkey))
                for i in range(len(response.result[0].cells)):
                    response.result[0].cells[i].value = _decode(str(response.result[0].cells[i].value))
        except Exception as ex:
            raise ex
        return response

    def batch_get_row(self, instance_name, table_name, batch_get_row_args, config=None):
        """
        batch get row

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param batch_get_row_args: arguments for batch get row
        :type batch_get_row_args: BatchQueryRowArgs
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        if batch_get_row_args is None:
            ex = BceClientError(INVALID_ARGS_ERROR)
            _logger.debug(ex)
            raise ex
        try:
            for i in range(len(batch_get_row_args.rows)):
                if batch_get_row_args.rows[i]["rowkey"] == "":
                    ex = BceClientError(INVALID_ARGS_ERROR)
                    _logger.debug(ex)
                    raise ex
                batch_get_row_args.rows[i]["rowkey"] = _encode(batch_get_row_args.rows[i]["rowkey"])
        except Exception as ex:
            raise ex

        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name + b"/rows"
        response = self._send_request(http_methods.GET, path=path, config=config,
                                      body=json.dumps(batch_get_row_args, default=batch_query_row_args_2_dict),
                                      headers={http_headers.CONTENT_TYPE: http_content_types.JSON})
        try:
            if response.result is not None:
                for i in range(len(response.result)):
                    response.result[i].rowkey = _decode(str(response.result[i].rowkey))
                    for j in range(len(response.result[i].cells)):
                        response.result[i].cells[j].value = _decode(str(response.result[i].cells[j].value))
        except Exception as ex:
            raise ex
        return response

    def scan(self, instance_name, table_name, scan_args, config=None):
        """
        scan

        :param instance_name: instance name
        :type instance_name: string
        :param table_name: table name
        :type table_name: string
        :param scan_args: arguments for scan
        :type scan_args: ScanArgs
        :param config: None
        :type config: BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        if scan_args is None:
            ex = BceClientError(INVALID_ARGS_ERROR)
            _logger.debug(ex)
            raise ex
        try:
            if scan_args.start_rowkey is not "":
                scan_args.start_rowkey = _encode(scan_args.start_rowkey)
            if scan_args.stop_rowkey is not "":
                scan_args.stop_rowkey = _encode(scan_args.stop_rowkey)
        except Exception as ex:
            raise ex

        path = bts.URL_PREFIX + b"/" + instance_name + b"/table/" + table_name + b"/rows"
        response = self._send_request(http_methods.GET, path=path, config=config,
                                      body=json.dumps(scan_args, default=scan_args_2_dict),
                                      headers={http_headers.CONTENT_TYPE: http_content_types.JSON})
        try:
            if response.result is not None:
                for i in range(len(response.result)):
                    response.result[i].rowkey = _decode(str(response.result[i].rowkey))
                    for j in range(len(response.result[i].cells)):
                        response.result[i].cells[j].value = _decode(str(response.result[i].cells[j].value))
        except Exception as ex:
            raise ex
        return response

    def _merge_config(self, config):
        if config is None:
            return self.config
        else:
            new_config = copy.copy(self.config)
            new_config.merge_non_none_values(config)
            return new_config

    # ------------- Http Request -----------
    def _send_request(
            self, http_method, path,
            body=None, headers=None, params=None,
            config=None,
            body_parser=None):
        config = self._merge_config(config)
        if body_parser is None:
            body_parser = handler.parse_json

        if config.security_token is not None:
            headers = headers or {}
            headers[http_headers.STS_SECURITY_TOKEN] = config.security_token

        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, path, body, headers, params)


