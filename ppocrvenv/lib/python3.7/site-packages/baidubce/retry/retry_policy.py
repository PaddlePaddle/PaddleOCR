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

import http.client
import logging
from builtins import str
from builtins import bytes

from baidubce.exception import BceServerError


_logger = logging.getLogger(__name__)


class NoRetryPolicy(object):
    """A policy that never retries."""

    def should_retry(self, error, retries_attempted):
        """Always returns False.

        :param error: ignored
        :param retries_attempted: ignored
        :return: always False
        :rtype: bool
        """
        return False

    def get_delay_before_next_retry_in_millis(self, error, retries_attempted):
        """Always returns 0.

        :param error: ignored
        :param retries_attempted: ignored
        :return: always 0
        :rtype: int
        """
        return 0


class BackOffRetryPolicy(object):
    """A policy that retries with exponential back-off strategy.

    This policy will keep retrying until the maximum number of retries is reached. The delay time
    will be a fixed interval for the first time then 2 * interval for the second, 4 * internal for
    the third, and so on. In general, the delay time will be 2^number_of_retries_attempted*interval.

    When a maximum of delay time is specified, the delay time will never exceed this limit.
    """

    def __init__(self,
                 max_error_retry=3,
                 max_delay_in_millis=20 * 1000,
                 base_interval_in_millis=300):
        """
        :param max_error_retry: the maximum number of retries.
        :type max_error_retry: int
        :param max_delay_in_millis: the maximum of delay time in milliseconds.
        :type max_delay_in_millis: int
        :param base_interval_in_millis: the base delay interval in milliseconds.
        :type base_interval_in_millis: int
        :raise ValueError if max_error_retry or max_delay_in_millis is negative.
        """
        if max_error_retry < 0:
            raise ValueError(b'max_error_retry should be a non-negative integer.')
        if max_delay_in_millis < 0:
            raise ValueError(b'max_delay_in_millis should be a non-negative integer.')

        self.max_error_retry = max_error_retry
        self.max_delay_in_millis = max_delay_in_millis
        self.base_interval_in_millis = base_interval_in_millis

    def should_retry(self, error, retries_attempted):
        """Return true if the http client should retry the request.

        :param error: the caught error.
        :type error: Exception
        :param retries_attempted: the number of retries which has been attempted before.
        :type retries_attempted: int
        :return: true if the http client should retry the request.
        :rtype: bool
        """

        # stop retrying when the maximum number of retries is reached
        if retries_attempted >= self.max_error_retry:
            return False

        # always retry on IOError
        if isinstance(error, IOError):
            _logger.debug(b'Retry for IOError.')
            return True

        # Only retry on a subset of service exceptions
        if isinstance(error, BceServerError):
            if error.status_code == http.client.INTERNAL_SERVER_ERROR:
                _logger.debug(b'Retry for internal server error.')
                return True
            if error.status_code == http.client.SERVICE_UNAVAILABLE:
                _logger.debug(b'Retry for service unavailable.')
                return True
            if error.code == BceServerError.REQUEST_EXPIRED:
                _logger.debug(b'Retry for request expired.')
                return True

        return False

    def get_delay_before_next_retry_in_millis(self, error, retries_attempted):
        """Returns the delay time in milliseconds before the next retry.

        :param error: the caught error.
        :type error: Exception
        :param retries_attempted: the number of retries which has been attempted before.
        :type retries_attempted: int
        :return: the delay time in milliseconds before the next retry.
        :rtype: int
        """
        if retries_attempted < 0:
            return 0
        delay_in_millis = (1 << retries_attempted) * self.base_interval_in_millis
        if delay_in_millis > self.max_delay_in_millis:
            return self.max_delay_in_millis
        return delay_in_millis
