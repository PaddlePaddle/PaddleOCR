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
This module provides a client for IAM.
"""

import copy
import json
import logging

from future.utils import iteritems

from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_content_types
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.services import iam

_logger = logging.getLogger(__name__)


class IamClient(BceBaseClient):
    """
    sdk client
    """

    def __init__(self, config=None):
        BceBaseClient.__init__(self, config)

    def _send_iam_request(self,
                          http_method,
                          path,
                          body=None,
                          headers=None,
                          params=None,
                          config=None,
                          body_parser=None):
        config = self._merge_config(config)
        path = iam.URL_PREFIX + path
        if body_parser is None:
            body_parser = handler.parse_json

        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, path, body, headers, params)

    def _merge_config(self, config):
        if config is None:
            return self.config
        else:
            new_config = copy.copy(self.config)
            new_config.merge_non_none_values(config)
            return new_config

    # ######################################### #role management# #################################################### #

    def get_role(self, role_name):
        """
        :type role_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/role/" + role_name

        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def create_role(self, create_role_request):
        """
        :type create_role_request: dict

        :return:
            **HttpResponse**
        """
        if create_role_request is None:
            body = None
        else:
            if not isinstance(create_role_request, dict):
                raise TypeError(b'create_role_request should be dict')
            else:
                body = json.dumps(create_role_request)

        path = b"/role"
        return self._send_iam_request(
            http_methods.POST,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            body=body
        )

    def update_role(self, role_name, update_role_request):
        """
        :type role_name: bytes
        :type update_role_request: dict

        :return:
            **HttpResponse**
        """
        if update_role_request is None:
            body = None
        else:
            if not isinstance(update_role_request, dict):
                raise TypeError(b'update_role_request should be dict')
            else:
                body = json.dumps(update_role_request)

        path = b"/role/" + role_name
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            body=body
        )

    def delete_role(self, role_name):
        """
        :type role_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/role/" + role_name
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def list_role(self):
        """
        :return:
            **HttpResponse**
        """
        path = b"/role"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    # ######################################### #policy management# ################################################## #

    def create_policy(self, create_policy_request):
        """
        :type create_policy_request: dict

        :return:
            **HttpResponse**
        """
        if create_policy_request is None:
            body = None
        else:
            if not isinstance(create_policy_request, dict):
                raise TypeError(b'create_policy_request should be dict')
            else:
                body = json.dumps(create_policy_request)

        path = b"/policy"
        return self._send_iam_request(
            http_methods.POST,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            body=body
        )

    def get_policy(self, policy_name, policy_type):
        """
        :type policy_name: bytes
        :type policy_type: bytes
        :return:
            **HttpResponse**
        """
        path = b"/policy/" + policy_name
        params = {b"policyType": policy_type}
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def delete_policy(self, policy_name):
        """
        :type policy_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/policy/" + policy_name
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def list_policy(self, policy_type=None, name_filter=None):
        """
        :type policy_type: bytes

        :type name_filter: bytes
        :param name_filter: bytes

        :return:
            **HttpResponse**
        """
        path = b"/policy"
        params = {b"policyType": policy_type, b"nameFilter": name_filter}
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def attach_policy_to_user(self, user_name, policy_name, policy_type=None):
        """
        :type user_name: bytes
        :type policy_name: bytes

        :type policy_type: bytes
        :param policy_type: None

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/policy/" + policy_name
        params = {b"policyType": policy_type}
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def detach_policy_from_user(self, user_name, policy_name, policy_type=None):
        """
        :type user_name: bytes
        :type policy_name: bytes

        :type policy_type: bytes
        :param policy_type: None

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/policy/" + policy_name
        params = {b"policyType": policy_type}
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def list_policies_from_user(self, user_name):
        """
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/policy"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def attach_policy_to_group(self, group_name, policy_name, policy_type=None):
        """
        :type group_name: bytes
        :type policy_name: bytes

        :type policy_type: bytes
        :param policy_type: None

        :return:
            **HttpResponse**
        """
        path = b"/group/" + group_name + b"/policy/" + policy_name
        params = {"policyType": policy_type}
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def detach_policy_from_group(self, group_name, policy_name, policy_type=None):
        """
        :type group_name: bytes
        :type policy_name: bytes

        :type policy_type: bytes
        :param policy_type: None

        :return:
            **HttpResponse**
        """
        path = b"/group/" + group_name + b"/policy/" + policy_name
        params = {b"policyType": policy_type}
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def list_policies_from_group(self, group_name):
        """
        :type group_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/group/" + group_name + b"/policy"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def attach_policy_to_role(self, role_name, policy_name, policy_type=None):
        """
        :type role_name: bytes
        :type policy_name: bytes

        :type policy_type: bytes
        :param policy_type: None

        :return:
            **HttpResponse**
        """
        path = b"/role/" + role_name + b"/policy/" + policy_name
        params = {b"policyType": policy_type}
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def detach_policy_from_role(self, role_name, policy_name, policy_type=None):
        """
        :type role_name: bytes
        :type policy_name: bytes

        :type policy_type: bytes
        :param policy_type: None

        :return:
            **HttpResponse**
        """
        path = b"/role/" + role_name + b"/policy/" + policy_name
        params = {b"policyType": policy_type}
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def list_policies_from_role(self, role_name):
        """
        :type role_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/role/" + role_name + b"/policy"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    # ######################################### #user management# #################################################### #
    def create_user(self, create_user_request):
        """
        :type create_user_request: dict

        :return:
            **HttpResponse**
        """
        if create_user_request is None:
            body = None
        else:
            if not isinstance(create_user_request, dict):
                raise TypeError(b'create_user_request should be dict')
            else:
                body = json.dumps(create_user_request)

        path = b'/user'
        return self._send_iam_request(
            http_methods.POST,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            body=body
        )

    def get_user(self, user_name):
        """
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b'/user/' + user_name

        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def update_user(self, user_name, update_user_request):
        """
        :type user_name: bytes
        :type update_user_request: dict
        :return:
            **HttpResponse**
        """
        if update_user_request is None:
            body = None
        else:
            if not isinstance(update_user_request, dict):
                raise TypeError(b'update_user_request should be dict')
            else:
                body = json.dumps(update_user_request)

        path = b"/user/" + user_name
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            body=body
        )

    def delete_user(self, user_name):
        """
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def list_user(self):
        """
        :return:
            **HttpResponse**
        """
        path = b"/user"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def update_user_login_profile(self, user_name, update_user_login_profile_request):
        """
        :type user_name: bytes
        :type update_user_login_profile_request: dict

        :return:
            **HttpResponse**
        """
        if update_user_login_profile_request is None:
            body = None
        else:
            if not isinstance(update_user_login_profile_request, dict):
                raise TypeError(b'update_user_login_profile_request should be dict')
            else:
                body = json.dumps(update_user_login_profile_request)

        path = b"/user/" + user_name + b"/loginProfile"
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            body=body
        )

    def get_user_login_profile(self, user_name):
        """
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/loginProfile"

        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def delete_user_login_profile(self, user_name):
        """
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/loginProfile"
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def create_user_accesskey(self, user_name):
        """
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/accesskey"
        return self._send_iam_request(
            http_methods.POST,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def disable_user_accesskey(self, user_name, accesskey_id):
        """
        :type user_name: bytes
        :type accesskey_id: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/accesskey/" + accesskey_id
        params = {"disable": ""}
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def enable_user_accesskey(self, user_name, accesskey_id):
        """
        :type user_name: bytes
        :type accesskey_id: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/accesskey/" + accesskey_id
        params = {"enable": ""}
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            params=params
        )

    def delete_user_accesskey(self, user_name, accesskey_id):
        """
        :type user_name: bytes
        :type accesskey_id: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/accesskey/" + accesskey_id
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def list_user_accesskey(self, user_name):
        """
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/accesskey"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    # ######################################### #group management# ################################################### #

    def create_group(self, create_group_request):
        """
        :type create_group_request:dict

        :return:
            **HttpResponse**
        """
        if create_group_request is None:
            body = None
        else:
            if not isinstance(create_group_request, dict):
                raise TypeError(b'create_group_request should be dict')
            else:
                body = json.dumps(create_group_request)

        path = b"/group"
        return self._send_iam_request(
            http_methods.POST,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            body=body
        )

    def get_group(self, group_name):
        """
        :type group_name: bytes
        :return:
            **HttpResponse**
        """
        path = b"/group/" + group_name
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def update_group(self, group_name, update_group_request):
        """
        :type group_name: bytes
        :type update_group_request: dict

        :return:
            **HttpResponse**
        """

        if update_group_request is None:
            body = None
        else:
            if not isinstance(update_group_request, dict):
                raise TypeError(b'update_group_request should be dict')
            else:
                body = json.dumps(update_group_request)

        path = b"/group/" + group_name
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path,
            body=body
        )

    def delete_group(self, group_name):
        """
        :type group_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/group/" + group_name
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def list_group(self):
        """
        :return:
            **HttpResponse**
        """
        path = b"/group"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def add_user_to_group(self, group_name, user_name):
        """
        :type group_name: bytes
        :type user_name: bytes
        :return:
            **HttpResponse**
        """
        path = b"/group/" + group_name + b"/user/" + user_name
        return self._send_iam_request(
            http_methods.PUT,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def remove_user_from_group(self, group_name, user_name):
        """
        :type group_name: bytes
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/group/" + group_name + b"/user/" + user_name
        return self._send_iam_request(
            http_methods.DELETE,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def list_user_group(self, user_name):
        """
        :type user_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/user/" + user_name + b"/group"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )

    def list_group_user(self, group_name):
        """
        :type group_name: bytes

        :return:
            **HttpResponse**
        """
        path = b"/group/" + group_name + b"/user"
        return self._send_iam_request(
            http_methods.GET,
            headers={http_headers.CONTENT_TYPE: http_content_types.JSON},
            path=path
        )
