# Copyright 2019 Baidu, Inc.
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
This module defines a common param class for CDN stats query.
"""


class CdnStatsParam(object):
    """Param of Cdn stats query."""

    def __init__(self, metric=None, start_time=None, end_time=None, period=None, key_type=None, key=None, groupBy=None,
                 prov=None, isp=None, level=None, protocol=None, extra=None):
        self.metric = metric
        self.startTime = start_time
        self.endTime = end_time
        self.period = period
        self.key_type = key_type
        self.key = key
        self.groupBy = groupBy
        self.prov = prov
        self.isp = isp
        self.level = level
        self.protocol = protocol
        self.extra = extra
