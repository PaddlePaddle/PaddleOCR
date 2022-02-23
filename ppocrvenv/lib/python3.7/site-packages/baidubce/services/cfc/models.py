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
This module provides models for CFC-SDK.
"""

import json

DUEDGE_TRIGGER = 'duedge'

DUEROS_TRIGGER = 'dueros'

CRONTAB_TRIGGER = 'cfc-crontab-trigger/v1/'

HTTP_TRIGGER = 'cfc-http-trigger/v1/CFCAPI'

CDN_TRIGGER = 'cdn'

BOS_TRIGGER = 'bos'


class AbstractTriggerDataModel(object):
    """Base class for all trigger data."""

    def serialize(self):
        """
        serialize

        :return
        :rtype dict
        """
        d = vars(self)
        ret = {}
        for k in d:
            if d[k] is not None:
                ret[k] = d[k]
        return ret

    def get_trigger_source(self):
        """
        trigger source

        :return
        :rtype string
        """
        return ''

    def __repr__(self):
        return '%s' % json.dumps(self.serialize())


class CrontabTriggerData(AbstractTriggerDataModel):
    """
    Crontab Trigger Data
    :param brn  (required) The url path.
    :type brn string
    :param name  (required) The trigger name. 1-30 length.Pattern: ^[a-zA-Z0-9-_]+$
    :type name string.The name of the trigger that you are creating or updating
    :param schedule_expression  (required) Schedule expression.The details see
        https://cloud.baidu.com/doc/CFC/s/Zjxl9lbed.
        For example, "cron(0 * * * *)" or "rate(10 minutes)".
    :type schedule_expression string.
    :param enabled.
    :type enabled bool. Enables the trigger.
    :param custom_input.
    :type custom_input json.
    """

    def __init__(self, brn=None, name=None, schedule_expression=None, enabled=False, custom_input=None):
        self.Input = custom_input
        self.Brn = brn
        self.Name = name
        self.ScheduleExpression = schedule_expression
        self.UUID = None
        if enabled:
            self.Enabled = 'Enabled'
        else:
            self.Enabled = 'Disabled'

    def set_status(self, enabled=False):
        """
        set crontab status

        :param enabled.
        :type enabled bool. Enables the trigger.
        :return
        :rtype string
        """
        if enabled:
            self.Enabled = 'Enabled'
        else:
            self.Enabled = 'Disabled'

    def get_trigger_source(self):
        return CRONTAB_TRIGGER


class HttpTriggerData(AbstractTriggerDataModel):
    """
    Http Trigger Data
    :param resource_path  (required) The url path.
    :type resource_path string
    :param method  (required) The http method. eg "GET,HEAD"
    :type method string
    :param auth_type  (required) Authentication type.
    :type auth_type string. eg anonymous | iam
    """

    def __init__(self, resource_path=None, method=None, auth_type=None):
        self.ResourcePath = resource_path
        self.Method = method
        self.AuthType = auth_type

    def get_trigger_source(self):
        return HTTP_TRIGGER


class CdnTriggerData(AbstractTriggerDataModel):
    """
    Cdn Trigger Data
    :param event_type  (required) Cdn event type. The details see
        https://cloud.baidu.com/doc/CFC/s/Kjwvz47o9/#relationconfiguration.
    :type event_type string
    :param domains. Domain list.
    :type domains list of string
    :param remark.
    :type remark string.
    :param status. Enables the trigger.
    :type status bool.
    """

    def __init__(self, event_type=None, domains=None, remark=None, status=False):
        self.EventType = event_type
        self.Domains = domains
        self.Remark = remark
        if status:
            self.Status = 'enabled'
        else:
            self.Status = 'disabled'

    def set_status(self, enabled=False):
        """
        set cdn trigger status

        :param enabled.
        :type enabled bool. Enables the trigger.
        :return
        :rtype string
        """
        if enabled:
            self.Status = 'enabled'
        else:
            self.Status = 'disabled'

    def get_trigger_source(self):
        return CDN_TRIGGER


class BOSTriggerData(AbstractTriggerDataModel):
    """
    BOS Trigger Data
    :param event_type  (required) BOS event type. The details see
        https://cloud.baidu.com/doc/CFC/s/Kjwvz47o9/#relationconfiguration.
    :type event_type list of string
    :param resource. For example, /prefix*suffix  /my.img  /my*img
    :type resource string
    :param name. The name of the trigger that you are creating or updating
    :type name string
    :param status. Enables the trigger.
    :type status bool.
    """

    def __init__(self, bucket=None, event_type=None, resource=None, name=None, status=False):
        self.Resource = resource
        self.EventType = event_type
        self.Name = name
        self.Bucket = bucket
        if status:
            self.Status = 'enabled'
        else:
            self.Status = 'disabled'

    def set_status(self, enabled=False):
        """
        set bos trigger status

        :param enabled.
        :type enabled bool. Enables the trigger.
        :return
        :rtype string
        """
        if enabled:
            self.Status = 'enabled'
        else:
            self.Status = 'disabled'

    def get_trigger_source(self):
        return BOS_TRIGGER + '/' + self.Bucket
