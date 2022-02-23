# -*- coding: utf-8 -*-

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
This module provide billing information and eip status condition.
"""


class Billing(object):
    """
    billing information
    """

    def __init__(self, payment_timing=None, billing_method=None, reservation_length=None,
                 reservation_time_unit=None):
        """
        :type payment_timing: string
        :param payment_timing: The pay time of the payment, default value 'Postpaid'

        :type billing_method: string
        :param billing_method: The way of eip charging, default value 'ByBandwidth'

        :type reservation_length: int
        :param reservation_length: purchase length

        :type reservation_time_unit: string
        :param reservation_time_unit: time unit of purchasing，default 'Month'
        """
        self.payment_timing = payment_timing
        self.billing_method = billing_method
        self.reservation_length = reservation_length
        self.reservation_time_unit = reservation_time_unit or 'Month'