#! usr/bin/python
# -*-coding:utf-8 -*-
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
from json import JSONEncoder


class Billing(object):
    """
	This class define billing.
	param: paymentTiming:
		The pay time of the payment,
	param: reservationLength:
		The duration to buy in specified time unit,
	param: reservationTimeUnit:
		The time unit to specify the duration ,only "Month" can be used now.
	"""

    def __init__(self, pay_method='Prepaid', reservationLength=1, reservationTimeUnit='Month'):
        self.paymentTiming = pay_method
        self.reservation = {
            'reservationLength': reservationLength,
            'reservationTimeUnit': reservationTimeUnit
        }

    def get_pay_method(self):
        """
            get instance current pay_method:Prepaid/Postpaid
        """
        return self.paymentTiming


class SubnetMap(object):
    """
    SubnetMap:contains zoneName and subnetId
    """

    def __init__(self, zone_name, subnet_id):
        super(SubnetMap, self).__init__()
        self.zone_name = str(zone_name)
        self.subnet_id = str(subnet_id)


class CreateInstanceResponse(BceResponse):
    """
    Create Instance Response
    """

    def __init__(self, bce_response):
        super(CreateInstanceResponse, self).__init__()
        self.instance_ids = bce_response.instance_ids


class ListInstanceResponse(BceResponse):
    """
    List Instance Response
    """

    def __init__(self, bce_response):
        super(ListInstanceResponse, self).__init__()
        self.max_keys = bce_response.max_keys
        self.marker = str(bce_response.marker)
        self.next_marker = str(bce_response.next_marker)
        self.is_truncated = bce_response.is_truncated
        self.instances = bce_response.instances


class GetInstanceResponse(BceResponse):
    """
    Get Instance Response
    """

    def __init__(self, bce_response):
        super(GetInstanceResponse, self).__init__()
        self.instance_id = str(bce_response.instance_id)
        self.instance_name = str(bce_response.instance_name)
        self.instance_status = str(bce_response.instance_status)
        self.cluster_type = str(bce_response.cluster_type)
        self.engine = str(bce_response.engine)
        self.engine_version = str(bce_response.engine_version)
        self.vnet_ip = str(bce_response.vnet_ip)
        self.port = str(bce_response.port)
        self.instance_create_time = bce_response.instance_create_time
        self.instance_expire_time = bce_response.instance_expire_time
        self.capacity = bce_response.capacity
        self.used_capacity = bce_response.used_capacity
        self.payment_timing = str(bce_response.payment_timing)
        self.zone_names = bce_response.zone_names
        self.vpc_id = str(bce_response.vpc_id)
        self.subnets = bce_response.subnets
        self.auto_renew = bce_response.auto_renew
        self.shard_num = bce_response.shard_num
        self.store_type = bce_response.store_type


class ListAvailableZoneResponse(BceResponse):
    """
    List available zone.
    """

    def __init__(self, bce_response):
        super(ListAvailableZoneResponse, self).__init__()
        self.zones = bce_response.zones


class ListSubnetResponse(BceResponse):
    """
    List available zone.
    """

    def __init__(self, bce_response):
        super(ListSubnetResponse, self).__init__()
        self.subnets = bce_response.subnets
        if self.subnets is None:
            self.subnets = []


class ListNodeTypeResponse(BceResponse):
    """
    List nodetypes for scs.
    """

    def __init__(self, bce_response):
        super(ListNodeTypeResponse, self).__init__()
        self.default_node_type_list = bce_response.default_node_type_list
        self.cluster_node_type_list = bce_response.cluster_node_type_list
        self.hsdb_node_type_list = bce_response.hsdb_node_type_list


class NodeType(object):
    """
    NodeType model
    """

    def __init__(self, obj):
        super(NodeType, self).__init__()
        self.node_type = str(obj.node_type)
        self.cpu_num = int(obj.cpu_num)
        self.instance_flavor = int(obj.instance_flavor)
        self.network_throughput_in_gbps = float(obj.network_throughput_in_gbps)
        self.peak_qps = int(obj.peak_qps)
        self.max_connections = int(obj.max_connections)
        self.allowed_node_num_list = obj.allowed_node_num_list
        self.allowed_replication_num_list = obj.allowed_replication_num_list


class Tag(object):
    """
    Tag model
    """

    def __init__(self, key, value):
        super(Tag, self).__init__()
        self.tag_key = str(key)
        self.tag_value = str(value)

    def __repr__(self):
        return repr((self.tag_key, self.tag_value))


class ListSecurityIpResponse(BceResponse):
    """
    List IP whitelist of instances which allow access to
    """

    def __init__(self, bce_response):
        super(ListSecurityIpResponse, self).__init__()
        self.security_ips = bce_response.security_ips


class ListParameterResponse(BceResponse):
    """
    List configuration parameters and runtime parameters of scs instance
    """

    def __init__(self, bce_response):
        super(ListParameterResponse, self).__init__()
        self.parameters = bce_response.parameters


class Parameter(object):
    """
    Configuration parameters and runtime parameters of scs instance
    """

    def __init__(self, obj):
        super(Parameter, self).__init__()
        self.default = str(obj.default)
        self.force_restart = int(obj.force_restart)
        self.name = str(obj.name)
        self.value = str(obj.value)


class ModifyParameterRequest(object):
    """
    A request model to modify instance parameter
    """

    def __init__(self, obj):
        super(ModifyParameterRequest, self).__init__()
        self.name = str(obj.name)
        self.value = str(obj.value)


class ListBackupResponse(BceResponse):
    """
    List backups of instance.
    """

    def __init__(self, bce_response):
        super(ListBackupResponse, self).__init__()
        self.total_count = bce_response.total_count
        self.backups = bce_response.backups


class Backup(object):
    """
    Backup model
    """

    def __init__(self, obj):
        super(Backup, self).__init__()
        self.backup_type = str(obj.backup_type)
        self.comment = str(obj.comment)
        self.records = obj.records
        self.start_time = str(obj.start_time)


class BackupRecord(object):
    """
    Backup record model
    """

    def __init__(self, obj):
        super(BackupRecord, self).__init__()
        self.backup_record_id = str(obj.backup_record_id)
        self.backup_status = str(obj.backup_status)
        self.duration = int(obj.duration)
        self.object_size = int(obj.object_size)
        self.shard_name = str(obj.shard_name)
        self.start_time = str(obj.start_time)


class GetBackupResponse(BceResponse):
    """
    Get backup detail response model
    """

    def __init__(self, bce_response):
        super(GetBackupResponse, self).__init__()
        self.url = str(bce_response.url)
        self.url_expiration = str(bce_response.url_expiration)


class JsonWrapper(JSONEncoder):
    """
        custom json encoder for class
    """

    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, SubnetMap):
            return {
                'zoneName': obj.zone_name,
                'subnetId': obj.subnet_id
            }
        if isinstance(obj, Tag):
            return {
                'tagKey': obj.tag_key,
                'tagValue': obj.tag_value
            }
        return JSONEncoder.default(self, obj)
