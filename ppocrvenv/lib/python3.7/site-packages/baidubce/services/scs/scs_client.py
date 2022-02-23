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
This module provides a client class for SCS.
"""
from __future__ import unicode_literals

import copy
import json
import logging
import uuid

import http.client

import baidubce.services.scs.model as model
from baidubce import utils
from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.exception import BceClientError
from baidubce.exception import BceServerError
from baidubce.http import bce_http_client
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.services import scs
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


class ScsClient(BceBaseClient):
    """
    Scs sdk client
    """

    def __init__(self, config=None):
        if config is not None:
            self._check_config_type(config)
        BceBaseClient.__init__(self, config)

    @required(config=BceClientConfiguration)
    def _check_config_type(self, config):
        return True

    @required(engine_version=(str), instance_name=(str), cluster_type=(str),
              node_type=(str), shard_num=int, proxy_num=int,
              replication_num=int, store_type=int, vpc_id=(str), subnets=list, port=int, purchase_count=int,
              auto_renew_time_unit=(str), auto_renew_time=int, billing=model.Billing)
    def create_instance(self, engine_version, instance_name, cluster_type, node_type, shard_num, replication_num,
                        proxy_num=0, store_type=None, vpc_id=None, subnets=None, port=6379, purchase_count=1,
                        auto_renew_time_unit='month', auto_renew_time=None, billing=model.Billing(), config=None):
        """
        Create instance with specific config

        :param engine_version:
        :type engine_version:string or unicode

        :param instance_name: Instance name
        :type  name: string or unicode

        :param cluster_type: default/master_slave or cluster
        :type  cluster_type: string or unicode

        :param node_type: Node specification
        :type  node_type: string or unicode

        :param shard_num: number of shard
        :type  shard_num: int

        :param replication_num: number of replication
        :type  replication_num: int

        :param store_type: default is 0;High performance memory:0;ssd-native:1
        :type  store_type: int

        :param vpc_id: vpc id,use default vpc id if not provide.
        :type  vpc_id: str

        :param subnets: list of model.SubnetMap
        :type  subnets: list

        :param port: default is 6379
        :type  port: int

        :param purchase_count: Number of purchases, default is 1
        :type  purchase_count: int

        :param auto_renew_time_unit: Renew monthly or yearly,value in ['month','year]
        :type  auto_renew_time_unit: str

        :param auto_renew_time: If billing is Prepay, the automatic renewal time is 1-9
         when auto_renew_time_unit is 'month' and 1-3 when auto_renew_time_unit is 'year'
        :type  auto_renew_time: int

        :param billing: default billing is Prepay 1 month
        :type  billing: model.Billing

        :param config: None
        :type  config: BceClientConfiguration

        :return: Object
            {
                "instance_ids": ["scs-bj-wHoJXL09355"]
            }

        :rtype: baidubce.bce_response.BceResponse
        """
        data = {
            'billing': billing.__dict__,
            'instanceName': instance_name,
            'nodeType': node_type,
            'port': port,
            'engineVersion': engine_version,
            'storeType': store_type,
            'purchaseCount': purchase_count,
            'shardNum': shard_num,
            'proxyNum': proxy_num,
            'replicationNum': replication_num,
            'clusterType': cluster_type,
            'vpcId': vpc_id,
            'subnets': subnets,
            'autoRenewTimeUnit': auto_renew_time_unit,
            'autoRenewTime': auto_renew_time
        }

        return model.CreateInstanceResponse(self._send_request(http_methods.POST, 'instance',
                                                               params={"clientToken": uuid.uuid4()},
                                                               body=json.dumps(data, cls=model.JsonWrapper, indent=4),
                                                               config=config,
                                                               api_version=2))

    @required(engine_version=(str), instance_id=(str),
              node_type=(str), shard_num=int, proxy_num=int, billing=model.Billing)
    def resize_instance(self, instance_id, node_type, shard_num=None, engine_version=None, billing=None, config=None):
        """
        Create instance with specific config

        :param engine_version:
        :type engine_version:string or unicode

        :type  name: string or unicode

        :param node_type: Node specification
        :type  node_type: string or unicode

        :param instance_id: number of shard
        :type  shard_num: int

        :param shard_num: number of proxy node
        :type  shard_num: int

        :param billing: default billing is Prepay 1 month
        :type  billing: model.Billing

        :param config: None
        :type  config: BceClientConfiguration

        """
        response = self.get_instance_detail(instance_id)
        billing = model.Billing(pay_method=response.payment_timing)
        if engine_version is None:
            engine_version = response.engine_version
        if shard_num is None:
            shard_num = response.shard_num
        data = {
            'billing': billing.__dict__,
            'nodeType': node_type,
            'engineVersion': engine_version,
            'shardNum': shard_num
        }
        key = instance_id + '/resize'
        return model.CreateInstanceResponse(self._send_request(http_methods.PUT, 'instance', key=key,
                                                               body=json.dumps(data), config=config))

    @required(instance_id=(str))
    def delete_instance(self, instance_id, config=None):
        """
        Delete instance
        :param instance_id: The ID of instance
        :type  instance_id: string or unicode

        :param config: None
        :type  config: BceClientConfiguration

        :return:
        """
        return self._send_request(http_method=http_methods.DELETE, function_name='instance', key=instance_id,
                                  config=config, api_version=1)

    def list_instances(self, marker=None, max_keys=1000, config=None):
        """
        Get instances in current region
        :param marker: start position
        :param max_keys: max count per page
        :param config:
        :return:
        """
        params = {}
        if marker is not None:
            params[b'marker'] = marker
        if max_keys is not None:
            params[b'maxKeys'] = max_keys
        return model.ListInstanceResponse(self._send_request(http_methods.GET,
                                                             function_name='instance',
                                                             params=params,
                                                             config=config))

    @required(instance_id=(str))
    def get_instance_detail(self, instance_id, config=None):
        """
        Get instance detail
        :param instance_id: The ID of instance
        :type  instance_id: string or unicode

        :param config: None
        :type  config: BceClientConfiguration

        :return:
            {
                "instance_id":"scs-bj-cxisuftlkquj",
                "instance_name":"post101",
                "instance_status":"_running",
                "cluster_type":"master_slave",
                "engine":"redis",
                "engine_version":"3.2",
                "vnet_ip":"10.107.231.11",
                "domain":"redis.zsfqaybijgbv.scs.bj.baidubce.com",
                "port":"6379",
                "instance_create_time":"2018-11-13_t05:37:49_z",
                "capacity":1,
                "used_capacity":0.06,
                "payment_timing":"_postpaid",
                "vpc_id":"vpc-1n1wqxfu4iuu",
                "auto_renew":"true",
                "subnets":[
                    {
                        "name":"系统预定义子网_c",
                        "subnet_id":"sbn-0ynnfkyymh8z",
                        "zone_name":"cn-bj-c",
                        "cidr":"192.168.32.0/20"
                    },
                    {
                        "name":"系统预定义子网",
                        "subnet_id":"sbn-rvv87cdd0gv9",
                        "zone_name":"cn-bj-a",
                        "cidr":"192.168.0.0/20"
                    }
                ],
                "zone_names":[
                    "cn-bj-a",
                    "cn-bj-c"
                ]
            }
        """
        return model.GetInstanceResponse(self._send_request(http_methods.GET, function_name='instance', key=instance_id,
                                                            config=config, api_version=2))

    @required(instance_id=(str), instance_name=(str))
    def rename_instance(self, instance_id, instance_name, config=None):
        """
        Update the instance_name of instance
        :param instance_id: the ID of instance
        :type str

        :param instance_name: a new instance name for instance
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        data = {'instanceName': instance_name}
        key = instance_id + '/rename'
        return self._send_request(http_method=http_methods.PUT, function_name='instance', key=key,
                                  body=json.dumps(data), config=config, api_version=1)

    @required(instance_id=(str))
    def restart_instance(self, instance_id, config=None):
        """
        restart the instance
        :param instance_id: the ID of instance
        :type str

        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        key = instance_id + '/restart'
        return self._send_request(http_method=http_methods.PUT, function_name='instance', key=key,
                                  config=config, api_version=2)

    @required(instance_id=(str), domain=(str))
    def rename_instance_domain(self, instance_id, domain, config=None):
        """
        Update the domain of instance
        :param instance_id: the ID of instance
        :type str

        :param domain: a new domain for instance.
        New instance domain name.
        Naming rules of domain name:
            1. It is composed of lowercase letters and numbers;
            2. It begins with a lowercase letter;
            3. The length is between 3-30
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        data = {'domain': domain}
        key = instance_id + '/renameDomain'
        return self._send_request(http_method=http_methods.PUT, function_name='instance', key=key,
                                  body=json.dumps(data), config=config, api_version=1)

    @required(instance_id=(str), password=(str))
    def flush_instance(self, instance_id, password='', config=None):
        """
        Clear the data of SCS instance
        :param instance_id: the ID of instance
        :type str

        :param password: the password of instance.
        if no password is set, an empty string is passed.
        The password needs to be encrypted.
        Please refer to the definition of password encryption transmission specification for details.
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        password = ''
        if len(password) > 0:
            secret_access_key = self.config.credentials.secret_access_key
            password = utils.aes128_encrypt_16char_key(password, secret_access_key)
        data = {
            'password': password
        }
        key = instance_id + '/flush'
        return self._send_request(http_method=http_methods.PUT, function_name='instance', key=key,
                                  body=json.dumps(data), config=config, api_version=1)

    def list_available_zones(self, config=None):
        """
        lost available zone of current region
        :param config:
        :type BceClientConfiguration
        :return:
        """
        return model.ListAvailableZoneResponse(self._send_request(http_methods.GET, function_name="zone",
                                                                  config=config))

    def list_subnets(self, vpc_id=None, zone_name=None, config=None):
        """
        list subnet by vpcId or zoneName
        :param vpc_id: ID of the VPC to which it belongs
        :param zone_name: The name of the zone it belongs to
        :param config:
        :type BceClientConfiguration
        :return:
        """
        params = {}
        if vpc_id is not None:
            params[b'vpcId'] = vpc_id
        if zone_name is not None:
            params[b'zoneName'] = zone_name
        return model.ListSubnetResponse(self._send_request(http_methods.GET, function_name="subnet",
                                                           params=params, config=config))

    def list_nodetypes(self, config=None):
        """
        List nodetypes
        :param config:
        :type BceClientConfiguration
        :return:
        """
        return model.ListNodeTypeResponse(self._send_request(http_methods.GET, function_name="nodetypes",
                                                             config=config, api_version=2))

    @required(instance_id=(str), change_tags=list)
    def bind_tags(self, instance_id, change_tags, config=None):
        """
        Bind tags to instance
        :param change_tags: tag list to bind to instance
        :type list of Tag

        :param instance_id: the ID of instance
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """

        data = {
            'changeTags': change_tags
        }
        key = instance_id + '/bindTag'
        return self._send_request(http_method=http_methods.PUT, function_name='instance',
                                  key=key, body=json.dumps(data, cls=model.JsonWrapper, indent=4), config=config)

    @required(instance_id=(str), change_tags=list)
    def unbind_tags(self, instance_id, change_tags, config=None):
        """
        Unbind tags to instance
        :param change_tags: tag list to unbind
        :type list of Tag

        :param instance_id: the ID of instance
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        data = {
            'changeTags': change_tags
        }
        key = instance_id + '/unBindTag'
        return self._send_request(http_method=http_methods.PUT, function_name='instance',
                                  key=key, body=json.dumps(data, cls=model.JsonWrapper, indent=4), config=config)

    @required(instance_id=(str))
    def set_as_master(self, instance_id, config=None):
        """
        Set an SCS instance as the primary region of the live instance group
        The cluster created by default is master
        :param instance_id: the ID of instance
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """

        key = instance_id + '/setAsMaster'
        return self._send_request(http_method=http_methods.PUT, function_name='instance',
                                  key=key, config=config)

    @required(instance_id=(str), master_domain=(str), master_port=int)
    def set_as_slave(self, instance_id, master_domain, master_port, config=None):
        """
        Set SCS instance A as a slave of a master SCS instance B
        :param master_port: the port of master redis B
        :type int

        :param master_domain:the domain of master redis B
        if B in a different region,peer-2-peer dns copy muster be turn on.
        :type str/unicode

        :param instance_id: the ID of instance
        :type str/unicode

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        data = {
            'masterDomain': master_domain,
            'masterPort': master_port
        }
        key = instance_id + '/setAsSlave'
        return self._send_request(http_method=http_methods.PUT, function_name='instance',
                                  key=key, body=json.dumps(data), config=config)

    @required(instance_id=(str))
    def list_security_ip(self, instance_id, config=None):
        """
        List securityIps of a instance
        :param instance_id: ID of instanceId
        :param config:
        :return:
        """
        key = instance_id + "/securityIp"
        return model.ListSecurityIpResponse(self._send_request(http_methods.GET, function_name="instance",
                                                               key=key, config=config))

    @required(instance_id=(str), security_ips=list)
    def add_security_ips(self, instance_id, security_ips, config=None):
        """
        Add security_ips to access to instance
        :param security_ips:
        :type list

        :param instance_id: the ID of instance
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        data = {
            'securityIps': security_ips
        }
        key = instance_id + '/securityIp'
        return self._send_request(http_method=http_methods.PUT, function_name='instance',
                                  key=key, body=json.dumps(data), config=config)

    @required(instance_id=(str), security_ips=list)
    def delete_security_ips(self, instance_id, security_ips, config=None):
        """
        Delete security_ips to access to instance
        :param security_ips:
        :type list

        :param instance_id: the ID of instance
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        data = {
            'securityIps': security_ips
        }
        key = instance_id + '/securityIp'
        return self._send_request(http_method=http_methods.DELETE, function_name='instance',
                                  key=key, body=json.dumps(data), config=config)

    @required(instance_id=(str), password=(str))
    def modify_password(self, instance_id, password, config=None):
        """
        Update the password for scs instance
        :param password:
        :type list

        :param instance_id: the ID of instance
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """

        secret_access_key = self.config.credentials.secret_access_key
        data = {
            'password': utils.aes128_encrypt_16char_key(password, secret_access_key)
        }
        key = instance_id + '/modifyPassword'
        return self._send_request(http_method=http_methods.PUT, function_name='instance',
                                  key=key, body=json.dumps(data), config=config)

    @required(instance_id=(str))
    def list_parameters(self, instance_id, config=None):
        """
        List parameters of a instance
        :param instance_id: ID of instanceId
        :param config:
        :return:
        """
        key = instance_id + "/parameter"
        return model.ListParameterResponse(self._send_request(http_methods.GET, function_name="instance",
                                                              key=key, config=config))

    @required(instance_id=(str), name=(str), time=(str), expire_days=int)
    def modify_parameter(self, instance_id, name, value, config=None):
        """
        Update the parameter of instance
        :param name: parameter name
        :param value: parameter value
        :param instance_id: the ID of instance
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        data = {
            'parameter': {
                'name': name,
                'value': value
            }
        }
        key = instance_id + '/parameter'
        return self._send_request(http_method=http_methods.PUT, function_name='instance',
                                  key=key, body=json.dumps(data), config=config)

    @required(instance_id=(str))
    def list_backups(self, instance_id, config=None):
        """
        List backup of a instance
        :param instance_id: ID of instanceId
        :param config:
        :return:
        """
        key = instance_id + "/backup"
        return model.ListBackupResponse(self._send_request(http_methods.GET, function_name="instance",
                                                           key=key, config=config))

    @required(instance_id=(str), backup_record_id=(str))
    def get_backup(self, instance_id, backup_record_id, config=None):
        """
        List backup of a instance
        :param backup_record_id: backup record ID
        :param instance_id: ID of instanceId
        :param config:
        :return:
        """
        key = instance_id + "/backup/" + backup_record_id
        return model.GetBackupResponse(self._send_request(http_methods.GET, function_name="instance",
                                                          key=key, config=config))

    @required(instance_id=(str), days=(str), time=(str), expire_days=int)
    def modify_backup_policy(self, instance_id, days, time, expire_days, config=None):
        """
        Update the instance_name of instance
        :param instance_id: the ID of instance
        :type str

        :param expire_days: Backup file expiration time
        :type int

        :param time: when to backup.utc time.eg:01:05:00
        :param days: the duration to backup.eg:Sun,Wed,Thu,Fri,Sta
        :type str

        :param config: None
        :type BceClientConfiguration

        :return:
        """
        data = {
            'backupDays': days,
            'backupTime': time,
            'expireDay': expire_days
        }
        key = instance_id + '/modifyBackupPolicy'
        return self._send_request(http_method=http_methods.PUT, function_name='instance', key=key,
                                  body=json.dumps(data), config=config)

    @staticmethod
    def _get_path_v1(config, function_name=None, key=None):
        return utils.append_uri(scs.URL_PREFIX_V1, function_name, key)

    @staticmethod
    def _get_path_v2(config, function_name=None, key=None):
        return utils.append_uri(scs.URL_PREFIX_V2, function_name, key)

    @staticmethod
    def _bce_scs_sign(credentials, http_method, path, headers, params,
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
        if params is None:
            params = {"clientToken": uuid.uuid4()}
        config = self._merge_config(config)
        path = {
            1: ScsClient._get_path_v1,
            2: ScsClient._get_path_v2,
        }[api_version](config, function_name, key)

        if body_parser is None:
            body_parser = _parse_result

        if headers is None:
            headers = {b'Accept': b'*/*', b'Content-Type': b'application/json;charset=utf-8'}

        return bce_http_client.send_request(config, ScsClient._bce_scs_sign, [body_parser], http_method, path, body,
                                            headers, params)
