#! usr/bin/python
# coding=utf-8

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
This module provides a client class for SMS.
"""

import copy
import json
import logging
import uuid

import http.client

import baidubce.services.sms.model as model
from baidubce import utils
from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.exception import BceClientError
from baidubce.exception import BceServerError
from baidubce.http import bce_http_client
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.services import sms
from baidubce.utils import required

_logger = logging.getLogger(__name__)


def _parse_result(http_response, response):
    if http_response.status / 100 == http.client.CONTINUE / 100:
        raise BceClientError('Can not handle 1xx http status code')
    bse = None
    body = http_response.read()
    if body:
        d = json.loads(body)

        if 'message' in d and 'code' in d and 'requestId' in d:
            r_code = d['code']
            # 1000 means success
            if r_code != '1000':
                bse = BceServerError(d['message'],
                                     code=d['code'],
                                     request_id=d['requestId'])
            else:
                response.__dict__.update(
                    json.loads(body, object_hook=utils.dict_to_python_object).__dict__)
                http_response.close()
                return True
        elif http_response.status / 100 == http.client.OK / 100:
            response.__dict__.update(
                json.loads(body, object_hook=utils.dict_to_python_object).__dict__)
            http_response.close()
            return True
    elif http_response.status / 100 == http.client.OK / 100:
        return True

    if bse is None:
        bse = BceServerError(http_response.reason, request_id=response.metadata.bce_request_id)
    bse.status_code = http_response.status
    raise bse


class SmsClient(BceBaseClient):
    """
    Sms sdk client
    """

    def __init__(self, config=None):
        if config is not None:
            self._check_config_type(config)
        BceBaseClient.__init__(self, config)

    @required(config=BceClientConfiguration)
    def _check_config_type(self, config):
        return True

    @required(signature_id=str, template_id=str, type=str, mobile=str, content_var_dict=dict)
    def send_message(self, signature_id, template_id, mobile, content_var_dict, config=None, custom=None,
                     user_ext_id=None, merchant_url_id=None):
        """
        Send message

        :param signature_id: The unique code identifying message signature, can be obtained from cloud.baidu.com
        :type  signature_id: string or unicode

        :param template_id: The unique code identifying message content template, can be obtained from cloud.baidu.com
        :type  template_id: string or unicode

        :param mobile: The target mobile, use "," as separators if you have multiple targets.
        :type  mobile: string or unicode

        :param content_var_dict: A map like "{"template param name": "template param content"}
        :type  content_var_dict: dict

        :param config: None
        :type  config: BceClientConfiguration

        :param custom: The user-defined param
        :type  custom: string or unicode

        :param user_ext_id: The user-defined channel code
        :type  user_ext_id: string or unicode

        :param merchant_url_id: The id of callback url specified by user
        :type  merchant_url_id: string or unicode

        :return: Object
            {
              "request_id": "5e6dacd5-8815-4183-8255-4ff079bf24e6",
              "code": "1000",
              "message": "成功",
              "data": [
                {
                  "code": "1000",
                  "message": "成功",
                  "mobile": "13800138000",
                  "message_id": "e325ea68-02c1-47ad-8844-c7b93cafaeba_13800138000"
                }
              ]
            }
        """
        data = {
            'signatureId': signature_id,
            'template': template_id,
            'mobile': mobile,
            'contentVar': content_var_dict,
            'custom': custom,
            'userExtId': user_ext_id,
            'merchantUrlId': merchant_url_id
        }

        return self._send_request(http_methods.POST, 'sendSms', body=json.dumps(data), config=config, api_version=1)

    @required(content=str, content_type=str, country_type=str)
    def create_signature(self, content, content_type, description=None, country_type="DOMESTIC",
                         signature_file_base_64=None, signature_file_format=None, config=None):
        """
        Create signature
        :param content: Signature content, only Chinese and English characters and numbers are allowed.
        :type  content: string or unicode

        :param content_type: Signature type, one of "Enterprise, MobileApp, Web, WeChatPublic, Brand, Else"
        :type  content_type: string or unicode

        :param description: Description of the signature
        :type  description: string or unicode

        :param country_type: The country or region in which the template can be used. Default value is "DOMESTIC".
                             The value of countryType could be DOMESTIC or INTERNATIONAL or GLOBAL.
                             DOMESTIC: the template can only be used in Mainland China.
                             INTERNATIONAL: the template can only be used out of Mainland China.
                             GLOBAL: the template can only be used all over the world.
        :type  country_type: string or unicode

        :param signature_file_base_64: The base64 encoding string of the signature certificate picture
        :type  signature_file_base_64: string or unicode

        :param signature_file_format: The format of the signature certificate picture, only one of JPG, PNG,
                                      JPEG allowed.
        :type  signature_file_format: string or unicode

        :param config: None
        :type  config: BceClientConfiguration
        :return: Object
            {
              "signature_id": "sms-sign-WWejWQ54455",
              "status": "SUBMITTED"
            }

        """
        data = {
            "content": content,
            "contentType": content_type,
            "countryType": country_type
        }
        if description:
            data["description"] = description
        if signature_file_base_64:
            data["signatureFileBase64"] = signature_file_base_64
        if signature_file_format:
            data["signatureFileFormat"] = signature_file_format
        return model.CreateSignatureResponse(self._send_request(http_methods.POST, 'signatureApply',
                                                                params={"clientToken": uuid.uuid4()},
                                                                body=json.dumps(data), config=config, api_version=2))

    @required(content=str, content_type=str, country_type=str, signature_id=str)
    def update_signature(self, content, content_type, country_type, signature_id, description=None,
                         signature_file_base_64=None, signature_file_format=None, config=None):
        """
        Update signature
        :param content: Signature content
        :type  content: string or unicode

        :param content_type: Signature type, one of "Enterprise, MobileApp, Web, WeChatPublic, Brand, Else"
        :type  content_type: string or unicode

        :param signature_id: The unique code identifying the signature
        :type  signature_id: string or unicode

        :param description: Description of the signature
        :type  description: string or unicode

        :param country_type: The country or region in which the template can be used.
                             The value of countryType could be DOMESTIC or INTERNATIONAL or GLOBAL.
                             DOMESTIC: the template can only be used in Mainland China.
                             INTERNATIONAL: the template can only be used out of Mainland China.
                             GLOBAL: the template can only be used all over the world.
        :type  country_type: string or unicode

        :param signature_file_base_64: The base64 encoding string of the signature certificate picture
        :type  signature_file_base_64: string or unicode

        :param signature_file_format: The format of the signature certificate picture, only one of JPG, PNG,
                                      JPEG allowed.
        :type  signature_file_format: string or unicode

        :param config: None
        :type  config: BceClientConfiguration
        :return: Object
            {
              "content": "Baidu",
              "content_type": "Enterprise",
              "description": "test sdk",
              "country_type": "DOMESTIC",
              "signature_file_base64": "test-string-base64encoded",
              "signature_file_format": "png"
            }
        """
        data = {
            "content": content,
            "contentType": content_type,
            "countryType": country_type
        }

        if description:
            data["description"] = description
        if signature_file_base_64:
            data["signatureFileBase64"] = signature_file_base_64
        if signature_file_format:
            data["signatureFileFormat"] = signature_file_format
        return self._send_request(http_methods.PUT, 'signatureApply', key=signature_id,
                                  body=json.dumps(data), config=config, api_version=2)

    @required(signature_id=str)
    def get_signature_detail(self, signature_id, config=None):
        """
        Get signature detail
        :param signature_id: The unique code identifying the signature
        :type  signature_id: string or unicode

        :param config: None
        :type  config: BceClientConfiguration

        :return: Object
            {
              "signature_id": "sms-sign-WWejWQ54455",
              "user_id": "bbede3f8c42e4113b6971fd09a57f494",
              "content": "Baidu",
              "content_type": "MobileApp",
              "description": "test sdk",
              "review": "",
              "status": "SUBMITTED",
              "country_type": "GLOBAL",
            }
        """
        return model.GetSignatureResponse(self._send_request(http_methods.GET, 'signatureApply', key=signature_id,
                                                             config=config, api_version=2))

    @required(signature_id=str)
    def delete_signature(self, signature_id, config=None):
        """
        Delete signature
        :param signature_id: The unique code identifying the signature
        :type  signature_id: string or unicode

        :param config: None
        :type  config: BceClientConfiguration

        :return:
        """
        return self._send_request(http_methods.DELETE, 'signatureApply', key=signature_id, config=config, api_version=2)

    @required(name=str, content=str, sms_type=str, country_type=str, description=str)
    def create_template(self, name, content, sms_type, country_type, description, config=None):
        """
        Create template with specific name and content

        :param name: Template name
        :type  name: string or unicode

        :param content: Template content like 'this is ${APP}, your code is ${VID}'
        :type  content: string or unicode

        :param sms_type: Business type of the template content, can be obtained from cloud.baidu.com
        :type  sms_type: string or unicode

        :param country_type: The country or region in which the template can be used.
                             The value of countryType could be DOMESTIC or INTERNATIONAL or GLOBAL.
                             DOMESTIC: the template can only be used in Mainland China.
                             INTERNATIONAL: the template can only be used out of Mainland China.
                             GLOBAL: the template can only be used all over the world.
        :type  country_type: string or unicode

        :param description: Description of the template
        :type  description: string or unicode

        :param config: None
        :type  config: BceClientConfiguration

        :return: Object
            {
                "template_id": "sms-tmpl-wHoJXL09355",
                "status": "SUBMITTED",
            }

        :rtype: baidubce.bce_response.BceResponse
        """
        data = {'name': name,
                'content': content,
                'smsType': sms_type,
                'countryType': country_type,
                'description': description}

        return model.CreateTemplateResponse(self._send_request(http_methods.POST, 'template',
                                                               params={"clientToken": uuid.uuid4()},
                                                               body=json.dumps(data), config=config, api_version=2))

    @required(template_id=str, name=str, content=str, sms_type=str, country_type=str)
    def update_template(self, template_id, name, content, sms_type, country_type, description=None, config=None):
        """
        Update template when audition failed.
        :param template_id: The unique code identifying the template
        :type  template_id: string or unicode

        :param name: the name of template
        :type  name: string or unicode

        :param content: the content of template,such as 'this is ${APP}, your code is ${VID}'
        :type  content: string or unicode

        :param sms_type: The business type of the template content, can be obtained from cloud.baidu.com
        :type  sms_type: string or unicode

        :param country_type: The country or region in which the template can be used.
                             The value of countryType could be DOMESTIC or INTERNATIONAL or GLOBAL.
                             DOMESTIC: the template can only be used in Mainland China.
                             INTERNATIONAL: the template can only be used out of Mainland China.
                             GLOBAL: the template can only be used all over the world.
        :type  country_type: string or unicode

        :param description: Description of the template
        :type  description: string or unicode

        :param config: None
        :type  config: BceClientConfiguration

        :return:
        """
        data = {'name': name,
                'content': content,
                'smsType': sms_type,
                'countryType': country_type}
        if description:
            data["description"] = description
        return self._send_request(http_method=http_methods.PUT, function_name='template', key=template_id,
                                  body=json.dumps(data), config=config, api_version=2)

    @required(template_id=str)
    def get_template_detail(self, template_id, config=None):
        """
        Get template detail
        :param template_id: The ID of message template
        :type  template_id: string or unicode

        :param config: None
        :type  config: BceClientConfiguration

        :return:
            {
              "template_id": "sms-tmpl-wHoJXL09355",
              "user_id": "bbede3f8c42e4113b6971fd09a57f494",
              "name": "TemplateNameTest",
              "content": "${code}",
              "sms_type": "CommonNotice",
              "description": "test modify",
              "review": "通过",
              "status": "APPROVED",
              "country_type": "INTERNATIONAL",
            }
        """
        return model.GetTemplateResponse(self._send_request(http_methods.GET, function_name='template', key=template_id,
                                                            config=config, api_version=2))

    @required(template_id=str)
    def delete_template(self, template_id, config=None):
        """
        Delete template
        :param template_id: The ID of message template
        :type  template_id: string or unicode

        :param config: None
        :type  config: BceClientConfiguration

        :return:
        """
        return self._send_request(http_method=http_methods.DELETE, function_name='template', key=template_id,
                                  config=config, api_version=2)

    def query_quota_rate(self, config=None):
        """
        Query quota and rate-limit detail
        :param config: None
        :type  config: BceClientConfiguration

        :return:
            {
              "quota_oer_day": 100,
              "quota_per_month": 1000,
              "quota_remain_today": 100,
              "quota_remain_this_month": 1000,
              "quota_white_list": true,
              "rate_limit_per_mobile_per_sign_by_minute": 5,
              "rate_limit_per_mobile_per_sign_by_hour": 10,
              "rate_limit_per_mobile_per_sign_by_day": 50,
              "rate_limit_white_list": true
            }
        """
        return model.QueryQuotaResponse(self._send_request(http_methods.GET, function_name="quota",
                                                           params={"userQuery": ""}, config=config, api_version=2))

    @required(quota_per_day=int, quota_per_month=int, rate_limit_per_mobile_per_sign_by_minute=int,
              rate_limit_per_mobile_per_sign_by_hour=int, rate_limit_per_mobile_per_sign_by_day=int)
    def update_quota_rate(self, quota_per_day, quota_per_month, rate_limit_per_mobile_per_sign_by_minute,
                          rate_limit_per_mobile_per_sign_by_hour, rate_limit_per_mobile_per_sign_by_day, config=None):
        """

        :param quota_per_day: Upper bound of the response-success request counts in one natural day.
        :type  quota_per_day: int

        :param quota_per_month: Upper bound of the response-success request counts in one natural month.
        :type  quota_per_month: int

        :param rate_limit_per_mobile_per_sign_by_minute: The limit with same mobile and signature in one minute
        :type  rate_limit_per_mobile_per_sign_by_minute: int

        :param rate_limit_per_mobile_per_sign_by_hour: Hourly limit with same mobile and signature
        :type  rate_limit_per_mobile_per_sign_by_hour: int

        :param rate_limit_per_mobile_per_sign_by_day: Daily rate limit with same mobile and signature
        :type  rate_limit_per_mobile_per_sign_by_day: int

        :param config: None
        :type  config: BceClientConfiguration

        :return:
        """
        data = {
            "quotaPerDay": quota_per_day,
            "quotaPerMonth": quota_per_month,
            "rateLimitPerMobilePerSignByMinute": rate_limit_per_mobile_per_sign_by_minute,
            "rateLimitPerMobilePerSignByHour": rate_limit_per_mobile_per_sign_by_hour,
            "rateLimitPerMobilePerSignByDay": rate_limit_per_mobile_per_sign_by_day
        }
        return self._send_request(http_methods.PUT, function_name="quota", body=json.dumps(data),
                                  config=config, api_version=2)

    @staticmethod
    def _get_path_v3(config, function_name=None, key=None):
        return utils.append_uri(sms.URL_PREFIX_V3, function_name, key)

    @staticmethod
    def _get_path_v3_2(config, function_name=None, key=None):
        return utils.append_uri(sms.URL_PREFIX_V3_2, function_name, key)

    @staticmethod
    def _bce_sms_sign(credentials, http_method, path, headers, params,
                      timestamp=0, expiration_in_seconds=1800,
                      headers_to_sign=None):

        headers_to_sign_list = [b"host",
                                b"content-md5",
                                b"content-length",
                                b"content-type"]

        if headers_to_sign is None or len(headers_to_sign) == 0:
            headers_to_sign = []
            for k in headers:
                k_lower = k.strip().lower()
                if k_lower.startswith(http_headers.BCE_PREFIX) or k_lower in headers_to_sign_list:
                    headers_to_sign.append(k_lower)
            headers_to_sign.sort()
        else:
            for k in headers:
                k_lower = k.strip().lower()
                if k_lower.startswith(http_headers.BCE_PREFIX):
                    headers_to_sign.append(k_lower)
            headers_to_sign.sort()

        return bce_v1_signer.sign(credentials,
                                  http_method,
                                  path,
                                  headers,
                                  params,
                                  timestamp,
                                  expiration_in_seconds,
                                  headers_to_sign)

    def _merge_config(self, config):
        if config is None:
            return self.config
        else:
            self._check_config_type(config)
            new_config = copy.copy(self.config)
            new_config.merge_non_none_values(config)
            return new_config

    def _send_request(self, http_method, function_name=None, key=None, body=None, headers=None, params=None,
                      config=None, body_parser=None, api_version=1):
        config = self._merge_config(config)
        path = {1: SmsClient._get_path_v3,
                2: SmsClient._get_path_v3_2,
                }[api_version](config, function_name, key)

        if body_parser is None:
            body_parser = _parse_result

        if headers is None:
            headers = {b'Accept': b'*/*', b'Content-Type': b'application/json;charset=utf-8'}

        return bce_http_client.send_request(config, SmsClient._bce_sms_sign, [body_parser], http_method, path, body,
                                            headers, params)
