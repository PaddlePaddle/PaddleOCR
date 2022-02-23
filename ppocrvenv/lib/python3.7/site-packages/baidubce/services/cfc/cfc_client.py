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
This module provides a client class for CFC.
API Reference: https://cloud.baidu.com/doc/CFC/index.html
"""

import copy
import json
import logging
import baidubce
import sys
import time
import traceback
import base64

from baidubce import bce_base_client
from baidubce import utils
from baidubce.auth import bce_v1_signer
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_content_types
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.services.cfc import cfc_handler
from baidubce.services.cfc import models
from baidubce.exception import BceClientError
from baidubce.exception import BceServerError
from baidubce.utils import required
from baidubce import compat


if compat.PY3:
    from urllib.parse import quote
else:
    from urllib import quote

_logger = logging.getLogger(__name__)


class CfcClient(bce_base_client.BceBaseClient):
    """
    CdnClient
    """
    prefix = '/v1'

    def __init__(self, config=None):
        bce_base_client.BceBaseClient.__init__(self, config)

    def invocations(self, function_name, invocation_type="RequestResponse",
                    log_type="None", body=None, qualifier=None, config=None):
        """
        invoking function

        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param invocation_type: (required)  Event/RequestResponse/DryRun
        :type invocation_type string

        :param  log_type: None / Tail You can set this optional parameter to Tail in the request only if you
                         specify the InvocationType parameter with value RequestResponse. In this case,CFC
                         returns the base64-encoded last 4 KB of log data produced by your cfc function in
                         the x-bce-log-result header.
        :type log_type string

        :param qualifier Minimum length of 1. Maximum length of 128. You can use function versions or function aliases.
                         If you don't, the default is $LATEST.
        :type qualifier string

        :param body: json

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: httplib.httpresponse
        """
        params = {}
        params["invocationType"] = invocation_type
        params["logType"] = log_type
        if qualifier is not None:
            params["Qualifier"] = qualifier
        if body is None:
            body = {}
        return self._send_request(
            http_methods.POST,
            '/functions/' + function_name + '/invocations',
            body=json.dumps(body),
            params=params,
            config=config, special=True)

    def invoke(self, function_name, invocation_type="RequestResponse",
                    log_type="None", body=None, qualifier=None, config=None):
        """
        invoking function

        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param invocation_type: (required)  Event/RequestResponse/DryRun
        :type invocation_type string

        :param  log_type: None / Tail You can set this optional parameter to Tail in the request only if you
                         specify the InvocationType parameter with value RequestResponse. In this case,CFC
                         returns the base64-encoded last 4 KB of log data produced by your cfc function in
                         the x-bce-log-result header.
        :type log_type string

        :param qualifier Minimum length of 1. Maximum length of 128. You can use function versions or function aliases.
                         If you don't, the default is $LATEST.
        :type qualifier string

        :param body: json

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: httplib.httpresponse
        """
        return self.invocations(function_name, invocation_type, log_type, body, qualifier, config)

    def create_function(self, function_name, description=None, environment=None,
                        handler=None, memory_size=128, region='bj',
                        zip_file=None, publish=False, run_time='python2',
                        timeout=3, dry_run=False, code_zip_file=None, config=None):
        """
        Create cfc function

        :param function_name  (required)
        :type function_name string

        :param description - A short, user-defined function description. Minimum length of 0. Maximum length of 256.
        :type description string

        :param environment environment's configuration settings.
        :type environment String to string map, [a-zA-Z]([a-zA-Z0-9_])+

        :param handler  (required) Maximum length of 128. [^\s]+
        :type handler string

        :param memory_size  The amount of memory, in MB.The default value is 128 MB. The value must be a multiple of 64
                            MB. From 128M to 3008M. Now it only supports 128M.
        :type memory_size int

        :param region  bj or gz. Now it only supports bj.
        :type region string

        :param zip_file  (required)
        :type zip_file bytes The contents of your zip file containing your deployment package.
                        the contents of the zip file must be base64-encoded.

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :param publish  This boolean parameter can be used to request CFC to create the CFC function and publish
                        a version as an atomic operation.
        :type publish: boolean

        :param run_time: python2 | nodejs6.11
        :type run_time: string

        :param timeout: 1-300 The default is 3 seconds.
        :type timeout: int

        :param code_zip_file: the file path of the zipped code.
        :type code_zip_file: string

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        if environment is None:
            environment = {}
        data = {
            'Code': {
                'Publish': publish,
                'DryRun': dry_run
            },
            'Description': description,
            'Region': region,
            'Timeout': timeout,
            'FunctionName': function_name,
            'Handler': handler,
            'Runtime': run_time,
            'MemorySize': memory_size,
            'Environment': {
                'Variables': environment
            }
        }
        if code_zip_file:
            code_file = open(code_zip_file, 'rb')
            code = code_file.read()
            code_base64 = base64.b64encode(code).decode('utf-8')
            data['Code']['ZipFile'] = code_base64
        else:
            data['Code']['ZipFile'] = zip_file
        params = {}
        return self._send_request(
            http_methods.POST,
            '/functions',
            body=json.dumps(data),
            params=params,
            config=config)

    def list_functions(self, function_version=None, page=None, page_size=None,
                       marker=None, max_items=None, config=None):
        """
        List cfc function

        :param function_version
        :type function_version string

        :param page
        :type page int

        :param page_size
        :type page_size int

        :param marker
        :type marker int

        :param max_items
        :type max_items int

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if function_version is not None:
            params['FunctionVersion'] = function_version
        if page is not None:
            params['page'] = page
        if page_size is not None:
            params['pageSize'] = page_size
        if marker is not None:
            params['Marker'] = marker
        if max_items is not None:
            params['MaxItems'] = max_items
        return self._send_request(
            http_methods.GET,
            '/functions/',
            body={},
            params=params,
            config=config)

    def get_function(self, function_name, qualifier=None, config=None):
        """
        get function

        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param qualifier Minimum length of 1. Maximum length of 128. You can use function versions or function aliases.
                         If you don't, the default is $LATEST.
        :type qualifier string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if qualifier is not None:
            params["Qualifier"] = qualifier
        return self._send_request(
            http_methods.GET,
            '/functions/' + function_name,
            body={},
            params=params,
            config=config)

    def delete_function(self, function_name, qualifier=None, config=None):
        """
        delete_function

        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param qualifier Minimum length of 1. Maximum length of 128. You can use function versions or function aliases.
                         If you don't, the default is $LATEST.
        :type qualifier string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if qualifier is not None:
            params["Qualifier"] = qualifier
        return self._send_request(
            http_methods.DELETE,
            '/functions/' + function_name,
            body={},
            params={},
            config=config)

    def update_function_code(self, function_name, zip_file=None,
                             publish=None, dry_run=None, config=None):
        """
        update_function_code

        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param zip_file
        :type zip_file bytes The contents of your zip file containing your deployment package.
                        the contents of the zip file must be base64-encoded.
        :param publish  This boolean parameter can be used to request CFC to create the CFC function and publish
                        a version as an atomic operation.
        :type publish: boolean

        :param dry_run  This boolean parameter can be used to test your request to CFC to update the function and
                        publish a version as an atomic operation. It will do all necessary computation and validation
                        of your code but will not upload it or a publish a version. Each time this operation is invoked,
                        the CodeSha256 hash value of the provided code will also be computed and returned in the
                        response.
        :type dry_run: boolean

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        body = {}
        if zip_file is not None:
            body["ZipFile"] = zip_file
        if publish is not None:
            body["Publish"] = publish
        if dry_run is not None:
            body["DryRun"] = dry_run

        return self._send_request(
            http_methods.PUT,
            '/functions/' + function_name + '/code',
            body=json.dumps(body),
            params={},
            config=config)

    def get_function_configuration(self, function_name, qualifier=None, config=None):
        """
        get_function_configuration
        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param qualifier Minimum length of 1. Maximum length of 128. You can use function versions or function aliases.
                         If you don't, the default is $LATEST.
        :type qualifier string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if qualifier is not None:
            params["Qualifier"] = qualifier

        return self._send_request(
            http_methods.GET,
            '/functions/' + function_name + "/configuration",
            body={},
            params={},
            config=config)

    def update_function_configuration(self, function_name, description=None, environment=None,
                                      handler=None, run_time=None,
                                      timeout=None, config=None):
        """
        update_function_configuration
        :param function_name  (required)
        :type function_name string

        :param description - A short, user-defined function description. Minimum length of 0. Maximum length of 256.
        :type description string

        :param environment environment's configuration settings.
        :type environment String to string map, [a-zA-Z]([a-zA-Z0-9_])+

        :param handler  (required) Maximum length of 128. [^\s]+
        :type handler string

        :param run_time: python2 | nodejs6.11
        :type run_time: string

        :param timeout: 1-300 The default is 3 seconds.
        :type timeout: int

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        data = {}
        if description is not None:
            data["Description"] = description
        if environment is not None:
            data["Environment"] = {"Variables": environment}
        if handler is not None:
            data["Handler"] = handler
        if run_time is not None:
            data["Runtime"] = run_time
        if timeout is not None:
            data["Timeout"] = timeout
        return self._send_request(
            http_methods.PUT,
            '/functions/' + function_name + '/configuration',
            body=json.dumps(data),
            params={},
            config=config)

    def list_versions_by_function(self, function_name, marker=None, max_items=None, config=None):
        """
        list_versions_by_function
        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

                :param marker
        :type marker int

        :param max_items
        :type max_items int

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if marker is not None:
            params["Marker"] = marker
        if max_items is not None:
            params["MaxItems"] = max_items
        return self._send_request(
            http_methods.GET,
            '/functions/' + function_name + "/versions",
            body={},
            params=params,
            config=config)

    def publish_version(self, function_name, description=None, config=None):
        """
        publish_version
        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param description  A short, user-defined function publish description. Minimum length of 0. Maximum length of
                            256.
        :type description string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if description is not None:
            params["Description"] = description

        return self._send_request(
            http_methods.POST,
            '/functions/' + function_name + "/versions",
            body={},
            params=params,
            config=config)

    def list_aliases(self, function_name, function_version=None,
                     marker=None, max_items=None, config=None):
        """
        list_aliases
        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param function_version
        :type function_version string

        :param marker
        :type marker int

        :param max_items
        :type max_items int

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if function_version is not None:
            params["FunctionVersion"] = function_version
        if marker is not None:
            params["Marker"] = marker
        if max_items is not None:
            params["MaxItems"] = max_items

        return self._send_request(
            http_methods.GET,
            '/functions/' + function_name + "/aliases",
            body={},
            params=params,
            config=config)

    def create_alias(self, function_name, function_version=None,
                     name=None, description=None, config=None):
        """
        create_alias
        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param name alias name
        :type name string

        :param function_version function version for which you are creating the alias. (\$LATEST|[0-9]+)
        :type function_version string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :param description   Description of the alias.
        :type description string

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        data = {}
        if description is not None:
            data["Description"] = description
        if function_version is not None:
            data["FunctionVersion"] = function_version
        if name is not None:
            data["Name"] = name

        return self._send_request(
            http_methods.POST,
            '/functions/' + function_name + "/aliases",
            body=json.dumps(data),
            params={},
            config=config)

    def get_alias(self, function_name, alias_name, config=None):
        """
        get_alias
        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param alias_name alias name
        :type alias_name string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET,
            '/functions/' + function_name + "/aliases/" + alias_name,
            body={},
            params={},
            config=config)

    def update_alias(self, function_name, alias_name, function_version=None,
                     description=None, config=None):
        """
        update_alias
        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param alias_name alias name
        :type alias_name string

        :param function_version function version for which you are update the alias. (\$LATEST|[0-9]+)
        :type function_version string

        :param description   Description of the alias.
        :type description string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        data = {}
        if description is not None:
            data["Description"] = description
        if function_version is not None:
            data["FunctionVersion"] = function_version

        return self._send_request(
            http_methods.PUT,
            '/functions/' + function_name + "/aliases/" + alias_name,
            body=json.dumps(data),
            params={},
            config=config)

    def delete_alias(self, function_name, alias_name, config=None):
        """
        delete_alias
        :param function_name  (required) The cfc function name. You can specify a function name (function_name)
                                or you can specify Baidu Resource Name (BRN) of the function (for example,
                                brn:bce:cfc:bj:account-id:function:function_name). Cfc also allows you
                                to specify a simple BRN (for example, account_id:function_name). The length of BRN is
                                limited to 1 to 140 characters. The function name is limited to 64 characters in length.
        :type function_name string

        :param alias_name alias name
        :type alias_name string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.DELETE,
            '/functions/' + function_name + "/aliases/" + alias_name,
            body={},
            params={},
            config=config)

    def list_triggers(self, function_brn, config=None):
        """
        list_triggers
        :param function_brn  (required) The cfc function brn.
        :type function_brn string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return: message result as following format
            {
              "Relation": [
                {
                  "RelationId": "brn:bce:cfc-http-trigger:bj:cd64f99c69d7c404b61de0a4f1865834:b8542048977633ad0a867aefc33fd32a/cfc/GET/cfc/docs",
                  "Sid": "cfc-dfbd5359-1afb-49e7-81d7-16554056c79d",
                  "Source": "cfc-http-trigger/v1/CFCAPI",
                  "Target": "brn:bce:cfc:bj:cd64f99c69d7c404b61de0a4f1865834:function:helloCFCDocs:$LATEST",
                  "Data": {
                    "AuthType": "anonymous",
                    "Brn": "brn:bce:cfc-http-trigger:bj:cd64f99c69d7c404b61de0a4f1865834:b8542048977633ad0a867aefc33fd32a/cfc/GET/cfc/docs",
                    "EndpointPrefix": "https://6ewfn1337kndc.cfc-execute.bj.baidubce.com",
                    "Method": "GET",
                    "ResourcePath": "/cfc/docs"
                  }
                },
                {
                  "RelationId": "brn:bce:cfc-http-trigger:bj:cd64f99c69d7c404b61de0a4f1865834:b8542048977633ad0a867aefc33fd32a/cfc/POST,PUT/cfc/docs/edit",
                  "Sid": "cfc-dfbd5359-1afb-49e7-81d7-16554056c79d",
                  "Source": "cfc-http-trigger/v1/CFCAPI",
                  "Target": "brn:bce:cfc:bj:cd64f99c69d7c404b61de0a4f1865834:function:helloCFCDocs:$LATEST",
                  "Data": {
                    "AuthType": "anonymous",
                    "Brn": "brn:bce:cfc-http-trigger:bj:cd64f99c69d7c404b61de0a4f1865834:b8542048977633ad0a867aefc33fd32a/cfc/POST,PUT/cfc/docs/edit",
                    "EndpointPrefix": "https://6ewfn1337kndc.cfc-execute.bj.baidubce.com",
                    "Method": "POST,PUT",
                    "ResourcePath": "/cfc/docs/edit"
                  }
                }
              ]
            }
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if function_brn is not None:
            params["FunctionBrn"] = function_brn
        return self._send_request(
            http_methods.GET,
            '/relation',
            body={},
            params=params,
            config=config)

    def create_trigger(self, function_brn, source=None, trigger_data=None, config=None):
        """
        create_trigger
        :param function_brn  (required) The cfc function brn.
        :type function_brn string

        :param source. The trigger source type. For example, models.CRONTAB_TRIGGER
        :type source string

        :param trigger_data The trigger data. You can get details from
            https://cloud.baidu.com/doc/CFC/s/Kjwvz47o9/#relationconfiguration
        :type trigger_data models.AbstractTriggerDateModel

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:  message result as following format
            {
                "Relation": {
                    "Sid": "cfc-c53bef4e-2cac-4bc6-84c1-2d5ec2f8bec2",
                    "RelationId" : "00457f0b-20d8-4f3d-8555-c2u121f38313",
                    "Source": "string",
                    "Target": "brn:bce:cfc:bj:640c8817bd1de2928d47256dd0620ce5:function:test:$LATEST",
                    "Data": {
                        "EventType":["PutObject", "PostObject"],
                        "Prefix":"images/",
                        "Suffix":".jpg",
                        "Status":"enabled"
                    }
                }
            }
        :rtype: baidubce.bce_response.BceResponse
        """
        data = {}
        if function_brn is not None:
            data["Target"] = function_brn
        if source is not None:
            data["Source"] = source
        if trigger_data is not None:
            if isinstance(trigger_data, models.AbstractTriggerDataModel):
                data["Source"] = trigger_data.get_trigger_source()
                data["Data"] = trigger_data.serialize()
            else:
                data["Data"] = trigger_data
        return self._send_request(
            http_methods.POST,
            '/relation',
            body=json.dumps(data),
            params={},
            config=config)

    def update_trigger(self, function_brn, relation_id, trigger_data, source=None, config=None):
        """
        create_trigger
        :param function_brn  (required) The cfc function brn.
        :type function_brn string

        :param source  (required) The trigger source type. For example, models.CRONTAB_TRIGGER
        :type source string

        :param relation_id  (required) The relation_id.
        :type relation_id string

        :param trigger_data The trigger data. You can get details from
            https://cloud.baidu.com/doc/CFC/s/Kjwvz47o9/#relationconfiguration
        :type trigger_data models.AbstractTriggerDateModel

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return: message result as following format
            {
                "Relation": {
                    "Sid": "cfc-c53bef4e-2cac-4bc6-84c1-2d5ec2f8bec2",
                    "RelationId" : "00457f0b-20d8-4f3d-8555-c2u121f38313",
                    "Source": "string",
                    "Target": "brn:bce:cfc:bj:640c8817bd1de2928d47256dd0620ce5:function:test:$LATEST",
                    "Data": {
                        "EventType":["PutObject", "PostObject"],
                        "Prefix":"images/",
                        "Suffix":".jpg",
                        "Status":"enabled"
                    }
                }
            }
        :rtype: baidubce.bce_response.BceResponse
        """
        data = {}
        if function_brn is not None:
            data["Target"] = function_brn
        if source is not None:
            data["Source"] = source
        if trigger_data is not None:
            if isinstance(trigger_data, models.AbstractTriggerDataModel):
                data["Source"] = trigger_data.get_trigger_source()
                data["Data"] = trigger_data.serialize()
            else:
                data["Data"] = trigger_data
        if relation_id is not None:
            data["RelationId"] = relation_id

        return self._send_request(
            http_methods.PUT,
            '/relation',
            body=json.dumps(data),
            params={},
            config=config)

    def delete_trigger(self, function_brn, source, relation_id, config=None):
        """
        delete_trigger
        :param function_brn  (required) The cfc function brn.
        :type function_brn string

        :param source  (required) The trigger source type.
        :type source string

        :param relation_id  (required) The relation_id.
        :type relation_id string

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if function_brn is not None:
            params["Target"] = function_brn
        if source is not None:
            params["Source"] = source
        if relation_id is not None:
            params["RelationId"] = relation_id
        return self._send_request(
            http_methods.DELETE,
            '/relation',
            body={},
            params=params,
            config=config)

    @staticmethod
    def _encode_function_name(self, function_name):
        return ''

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
            body_parser=None, special=False):
        config = self._merge_config(self, config)
        if body_parser is None:
            body_parser = cfc_handler.parse_json
        headers = headers or {}
        headers[http_headers.CONTENT_TYPE] = http_content_types.JSON
        if config.security_token is not None:
            headers = headers or {}
            headers[http_headers.STS_SECURITY_TOKEN] = config.security_token
        return self.send_request(config, bce_v1_signer.sign,
                                 [cfc_handler.parse_error, body_parser],
                                 http_method, CfcClient.prefix + path,
                                 body, headers, params, special)

    def send_request(
            self,
            config,
            sign_function,
            response_handler_functions,
            http_method, path, body, headers, params, special=False):
        """
        Send request to BCE services.
        :param config
        :type config: baidubce.BceClientConfiguration
        :param sign_function:
        :param response_handler_functions:
        :type response_handler_functions: list
        :param request:
        :type request: baidubce.internal.InternalRequest
        :return:
        :rtype: baidubce.BceResponse
        """
        t = int(time.time())
        _logger.debug(b'%s request start: %s %s, %s, %s, %d',
                      http_method, path, headers, params, body, t)
        headers = headers or {}
        if config.security_token is not None:
            headers[http_headers.STS_SECURITY_TOKEN] = config.security_token
        headers_to_sign = [b"host",
                           b"content-length",
                           b"content-type"]

        should_get_new_date = False
        if http_headers.BCE_DATE not in headers:
            should_get_new_date = True
        headers[http_headers.HOST] = config.endpoint

        for k in headers:
            k_lower = k.strip().lower()
            if k_lower.startswith(http_headers.BCE_PREFIX):
                headers_to_sign.append(k_lower)
        user_agent = 'bce-sdk-python/%s/%s/%s' % (
            compat.convert_to_string(baidubce.SDK_VERSION), sys.version, sys.platform)
        user_agent = user_agent.replace('\n', '')
        user_agent = compat.convert_to_bytes(user_agent)
        headers[http_headers.USER_AGENT] = user_agent
        body = compat.convert_to_bytes(body)
        if not body:
            headers[http_headers.CONTENT_LENGTH] = 0
        elif isinstance(body, bytes):
            headers[http_headers.CONTENT_LENGTH] = len(body)
        elif http_headers.CONTENT_LENGTH not in headers:
            raise ValueError(b'No %s is specified.' % http_headers.CONTENT_LENGTH)

        # store the offset of fp body
        offset = None
        if hasattr(body, "tell") and hasattr(body, "seek"):
            offset = body.tell()

        protocol, host, port = utils.parse_host_port(config.endpoint, config.protocol)
        path = quote(path)

        headers[http_headers.HOST] = host
        if port != config.protocol.default_port:
            headers[http_headers.HOST] += b':' + compat.convert_to_bytes(port)
        path = compat.convert_to_bytes(path)
        headers[http_headers.AUTHORIZATION] = sign_function(
            config.credentials, http_method, path, headers, params, headers_to_sign=headers_to_sign)
        encoded_params = utils.get_canonical_querystring(params, False)
        if len(encoded_params) > 0:
            uri = path + b'?' + encoded_params
        else:
            uri = path
        bce_http_client.check_headers(headers)

        retries_attempted = 0
        errors = []
        while True:
            conn = None
            try:
                if should_get_new_date is True:
                    headers[http_headers.BCE_DATE] = utils.get_canonical_time()
                    headers_to_sign.append(http_headers.BCE_DATE)

                headers[http_headers.AUTHORIZATION] = sign_function(
                    config.credentials, http_method, path, headers, params, headers_to_sign=headers_to_sign)
                if retries_attempted > 0 and offset is not None:
                    body.seek(offset)

                conn = bce_http_client._get_connection(protocol, host,
                                                       port, config.connection_timeout_in_mills)

                _logger.debug('request args:method=%s, uri=%s, headers=%s,patams=%s, body=%s',
                              http_method, uri, headers, params, body)

                http_response = bce_http_client._send_http_request(
                    conn, http_method, uri, headers, body, config.send_buf_size)

                headers_list = http_response.getheaders()

                # on py3 ,values of headers_list is decoded with ios-8859-1 from
                # utf-8 binary bytes

                # headers_list[*][0] is lowercase on py2
                # headers_list[*][0] is raw value py3
                if compat.PY3 and isinstance(headers_list, list):
                    temp_heads = []
                    for k, v in headers_list:
                        k = k.encode('latin-1').decode('utf-8')
                        v = v.encode('latin-1').decode('utf-8')
                        k = k.lower()
                        temp_heads.append((k, v))
                    headers_list = temp_heads

                _logger.debug(
                    'request return: status=%d, headers=%s' % (http_response.status, headers_list))
                # cfc invoke return doesn't have to be json
                if special:
                    return http_response

                response = bce_http_client.BceResponse()
                response.set_metadata_from_headers(dict(headers_list))
                for handler_function in response_handler_functions:
                    if handler_function(http_response, response):
                        break

                return response
            except Exception as e:
                if conn is not None:
                    conn.close()
                # insert ">>>>" before all trace back lines and then save it
                errors.append('\n'.join('>>>>' + line for line in traceback.format_exc().splitlines()))
                if config.retry_policy.should_retry(e, retries_attempted):
                    delay_in_millis = config.retry_policy.get_delay_before_next_retry_in_millis(
                        e, retries_attempted)
                    time.sleep(delay_in_millis / 1000.0)
                else:
                    raise bce_http_client.BceHttpClientError('Unable to execute HTTP request. '
                                                             'Retried %d times. All trace backs:\n'
                                                             '%s' % (retries_attempted,
                                                                     '\n'.join(errors)), e)
        retries_attempted += 1
