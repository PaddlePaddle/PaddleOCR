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
This module provides models for BES-SDK.
"""


class Billing(object):
    """
    Billing Class
    """

    def __init__(self, payment_type, time):
        self.paymentType = payment_type
        self.time = time


class Module(object):
    """
    Module Class
    """

    def __init__(self, type=None,
                 instance_num=None,
                 version=None,
                 slot_type=None,
                 desire_instance_num=None,
                 disk_slot_info=None):
        if type is not None:
            self.type = type
        if instance_num is not None:
            self.instanceNum = instance_num
        if version is not None:
            self.version = version
        if slot_type is not None:
            self.slotType = slot_type
        if desire_instance_num is not None:
            self.desireInstanceNum = desire_instance_num
        if disk_slot_info is not None:
            self.diskSlotInfo = disk_slot_info.__dict__


class DiskSlotInfo(object):
    """
    DiskSlotInfo class
    """

    def __init__(self, type=None, size=None, instance_num=None):
        if type is not None:
            self.type = type
        if size is not None:
            self.size = size
