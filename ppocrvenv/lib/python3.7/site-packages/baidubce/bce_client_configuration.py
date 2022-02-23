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
This module defines a common configuration class for BCE.
"""

from future.utils import iteritems
from builtins import str
from builtins import bytes
import baidubce.protocol
import baidubce.region
from baidubce.retry.retry_policy import BackOffRetryPolicy
from baidubce import compat


class BceClientConfiguration(object):
    """Configuration of Bce client."""

    def __init__(self,
                 credentials=None,
                 endpoint=None,
                 protocol=None,
                 region=None,
                 connection_timeout_in_mills=None,
                 send_buf_size=None,
                 recv_buf_size=None,
                 retry_policy=None,
                 security_token=None,
                 cname_enabled=False,
                 backup_endpoint=None):
        self.credentials = credentials
        self.endpoint = compat.convert_to_bytes(endpoint) if endpoint is not None else endpoint
        self.protocol = protocol
        self.region = region
        self.connection_timeout_in_mills = connection_timeout_in_mills
        self.send_buf_size = send_buf_size
        self.recv_buf_size = recv_buf_size
        if retry_policy is None:
            self.retry_policy = BackOffRetryPolicy()
        else:
            self.retry_policy = retry_policy
        self.security_token = security_token
        self.cname_enabled = cname_enabled
        self.backup_endpoint = compat.convert_to_bytes(backup_endpoint) if backup_endpoint is not None else backup_endpoint

    def merge_non_none_values(self, other):
        """

        :param other:
        :return:
        """
        for k, v in iteritems(other.__dict__):
            if v is not None:
                self.__dict__[k] = v


DEFAULT_PROTOCOL = baidubce.protocol.HTTP
DEFAULT_REGION = baidubce.region.BEIJING
DEFAULT_CONNECTION_TIMEOUT_IN_MILLIS = 50 * 1000
DEFAULT_SEND_BUF_SIZE = 1024 * 1024
DEFAULT_RECV_BUF_SIZE = 10 * 1024 * 1024
DEFAULT_CONFIG = BceClientConfiguration(
    protocol=DEFAULT_PROTOCOL,
    region=DEFAULT_REGION,
    connection_timeout_in_mills=DEFAULT_CONNECTION_TIMEOUT_IN_MILLIS,
    send_buf_size=DEFAULT_SEND_BUF_SIZE,
    recv_buf_size=DEFAULT_RECV_BUF_SIZE,
    retry_policy=BackOffRetryPolicy())
