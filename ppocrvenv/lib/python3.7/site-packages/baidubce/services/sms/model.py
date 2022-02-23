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
This module defines some Response classes for BTS
"""
from baidubce.bce_response import BceResponse


class CreateSignatureResponse(BceResponse):
    """
    Create Signature Response
    """

    def __init__(self, bce_response):
        super(CreateSignatureResponse, self).__init__()
        self.signature_id = str(bce_response.signature_id)
        self.status = str(bce_response.status)
        self.metadata = bce_response.metadata


class GetSignatureResponse(BceResponse):
    """
    Get Signature Response
    """

    def __init__(self, bce_response):
        super(GetSignatureResponse, self).__init__()
        self.signature_id = str(bce_response.signature_id)
        self.user_id = str(bce_response.user_id)
        self.content = str(bce_response.content)
        self.country_type = str(bce_response.country_type)
        self.content_type = str(bce_response.content_type)
        self.review = str(bce_response.review)
        self.status = str(bce_response.status)
        self.metadata = bce_response.metadata


class CreateTemplateResponse(BceResponse):
    """
    Create Template Response
    """

    def __init__(self, bce_response):
        super(CreateTemplateResponse, self).__init__()
        self.template_id = str(bce_response.template_id)
        self.status = str(bce_response.status)
        self.metadata = bce_response.metadata


class GetTemplateResponse(BceResponse):
    """
    Get Template Response
    """

    def __init__(self, bce_response):
        super(GetTemplateResponse, self).__init__()
        self.template_id = str(bce_response.template_id)
        self.user_id = str(bce_response.user_id)
        self.name = str(bce_response.name)
        self.content = str(bce_response.content)
        self.sms_type = str(bce_response.sms_type)
        self.description = str(bce_response.description)
        self.review = str(bce_response.review)
        self.status = str(bce_response.status)
        self.country_type = str(bce_response.country_type)
        self.metadata = bce_response.metadata


class QueryQuotaResponse(BceResponse):
    """
    Query Quota Response
    """

    def __init__(self, bce_response):
        super(QueryQuotaResponse, self).__init__()
        self.quota_per_day = bce_response.quota_per_day
        self.quota_per_month = bce_response.quota_per_month
        self.quota_remain_today = bce_response.quota_remain_today
        self.quota_remain_this_month = bce_response.quota_remain_this_month
        self.quota_white_list = bce_response.quota_white_list
        self.rate_limit_per_mobile_per_sign_by_minute = bce_response.rate_limit_per_mobile_per_sign_by_minute
        self.rate_limit_per_mobile_per_sign_by_hour = bce_response.rate_limit_per_mobile_per_sign_by_hour
        self.rate_limit_per_mobile_per_sign_by_day = bce_response.reate_limit_per_mobile_per_sign_by_day
        self.rate_limit_white_list = bce_response.rate_limit_white_list
        self.metadata = bce_response.metadata
