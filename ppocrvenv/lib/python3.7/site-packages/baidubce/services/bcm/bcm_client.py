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
This module provides a client class for BCM.
"""
import copy

from baidubce import bce_base_client, utils, compat
from baidubce.auth import bce_v1_signer
from baidubce.http import handler, bce_http_client, http_methods


class BcmClient(bce_base_client.BceBaseClient):
    """
    BCM base sdk client
    """

    prefix = b'/json-api'
    version = b'/v1'

    def __init__(self, config=None):
        bce_base_client.BceBaseClient.__init__(self, config)

    def _merge_config(self, config=None):
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

        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, BcmClient.prefix + BcmClient.version + path, body, headers, params)

    def get_metric_data(self, user_id=None, scope=None, metric_name=None,
                        dimensions=None, statistics=None, start_time=None,
                        end_time=None, period_in_second=None, config=None):
        """
        Return metric data of product instances owned by the authenticated user.

        This site may help you: https://cloud.baidu.com/doc/BCM/s/9jwvym3kb

        :param user_id:
            Master account ID
        :type user_id: string

        :param scope:
            Cloud product namespace, eg: BCE_BCC.
        :type scope: string

        :param metric_name:
            The metric name of baidu cloud monitor, eg: CpuIdlePercent.
        :type metric_name: string

        :param dimensions:
            Consists of dimensionName: dimensionValue.
            Use semicolons when items have multiple dimensions,
            such as dimensionName: dimensionValue; dimensionName: dimensionValue.
            Only one dimension value can be specified for the same dimension.
            eg: InstanceId:fakeid-2222
        :type dimensions: string

        :param statistics:
            According to the format of statistics1,statistics2,statistics3,
            the optional values are `average`, `maximum`, `minimum`, `sum`, `sampleCount`
        :type statistics: string

        :param start_time:
            Query start time.
            Please refer to the date and time, UTC date indication
        :type start_time: string

        :param end_time:
            Query end time.
            Please refer to the date and time, UTC date indication
        :type end_time: string

        :param period_in_second:
            Statistical period.
            Multiples of 60 in seconds (s).
        :type period_in_second: int

        :param config:
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        user_id = compat.convert_to_bytes(user_id)
        scope = compat.convert_to_bytes(scope)
        metric_name = compat.convert_to_bytes(metric_name)
        path = b'/metricdata/%s/%s/%s' % (user_id, scope, metric_name)
        params = {}

        if dimensions is not None:
            params[b'dimensions'] = dimensions
        if statistics is not None:
            params[b'statistics[]'] = statistics
        if start_time is not None:
            params[b'startTime'] = start_time
        if end_time is not None:
            params[b'endTime'] = end_time
        if period_in_second is not None:
            params[b'periodInSecond'] = period_in_second

        return self._send_request(http_methods.GET, path, params=params, config=config)
