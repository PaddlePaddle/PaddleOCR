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
This module provides a client class for BBC.
"""
import copy
import json
import logging
import random
import string
import uuid

from baidubce import bce_base_client
from baidubce.auth import bce_v1_signer
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_methods
from baidubce.utils import aes128_encrypt_16char_key
from baidubce.utils import required
from baidubce.services.bbc import bbc_model
from baidubce import compat

_logger = logging.getLogger(__name__)

default_billing_to_purchase_created = bbc_model.Billing('Postpaid')



class BbcClient(bce_base_client.BceBaseClient):
    """
    Bbc client sdk
    """

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
                      config=None, body_parser=None, api_version=b'/v1'):
        config = self._merge_config(config)
        if body_parser is None:
            body_parser = handler.parse_json

        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, api_version + path, body, headers, params)



    @required(flavor_id = (bytes, str), image_id = (bytes, str), raid_id = (bytes, str))
    def create_instance(self, flavor_id, image_id, raid_id, root_disk_size_in_gb=20, purchase_count=1,
                        zone_name=None, subnet_id=None, billing=None, name=None, admin_pass=None,
                        auto_renew_time_unit=None, auto_renew_time=0, deploy_set_id=None, enable_ht=False,
                        security_group_id=None, client_token=None, config=None):

        """
        Create a bbc instance with the specified options.
        You must fill the field of clientToken, which is especially for keeping idempotent.
        This is an asynchronous interface.

        :param flavor_id:
            The id of flavor, list all available flavors in BbcClient.list_flavors.
        :type flavor_id: string

        :param image_id:
            The id of image, list all available images in BbcClient.list_images.
        :type image_id: string

        :param raid_id:
            The id of raid, list all available raids in BbcClient.get_flavor_raid.
        :type raid_id: string

        :param root_disk_size_in_gb:
            System disk size of the physical machine to be created
        :type root_disk_size_in_gb: int

        :param purchase_count:
            The number of instances to buy, the default value is 1.
        :type purchase_count: int

        :param zone_name:
            The optional parameter to specify the available zone for the instance.
            See more detail through list_zones method
        :type zone_name: string

        :param subnet_id:
            The optional parameter to specify the id of subnet from vpc, optional param
             default value is default subnet from default vpc
        :type subnet_id: string

        :param billing:
            Billing information.
        :type billing: bbc_model.Billing

        :param name:
            The optional parameter to desc the instance that will be created.
        :type name: string

        :param enable_ht:
            The optional parameter to enable instance hyperthread.
        :type name: bool

        :param admin_pass:
            The optional parameter to specify the password for the instance.
            If specify the adminPass,the adminPass must be a 8-16 characters String
            which must contains letters, numbers and symbols.
            The symbols only contains "!@#$%^*()".
            The adminPass will be encrypted in AES-128 algorithm
            with the substring of the former 16 characters of user SecretKey.
            If not specify the adminPass, it will be specified by an random string.
            See more detail on
            https://cloud.baidu.com/doc/BBC/s/3jwvxu9iz#%E5%AF%86%E7%A0%81%E5%8A%A0%E5%AF%86%E4%BC%A0%E8%BE%93%E8%A7%84%E8%8C%83
        :type admin_pass: string

        :param auto_renew_time_unit
            The parameter to specify the unit of the auto renew time.
            The auto renew time unit can be "month" or "year".
            The default value is "month".
        :type auto_renew_time_unit: string

        :param auto_renew_time
            The parameter to specify the auto renew time, the default value is 0.
        :type auto_renew_time: string

        :deploy_set_id:
            The id of the deploy set
        :type deploy_set_id: string

        :param security_group_id:
            The optional parameter to specify the securityGroupId of the instance
            vpcId of the securityGroupId must be the same as the vpcId of subnetId
        :type security_group_id: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if client token is provided.
            If the clientToken is not specified by the user,
            a random String generated by default algorithm will be used.
            See more detail at
            https://bce.baidu.com/doc/BCC/API.html#.E5.B9.82.E7.AD.89.E6.80.A7
        :type client_token: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b'/instance'
        params = {}
        if client_token is None:
            params['clientToken'] = generate_client_token()
        else:
            params['clientToken'] = client_token


        if root_disk_size_in_gb == 0:
            root_disk_size_in_gb = 20
        if purchase_count < 1:
            purchase_count = 1

        if billing is None:
            billing = default_billing_to_purchase_created

        body = {
            'flavorId': flavor_id,
            'imageId': image_id,
            'raidId': raid_id,
            'rootDiskSizeInGb': root_disk_size_in_gb,
            'purchaseCount': purchase_count,
            'billing': billing.__dict__
        }
        if zone_name is not None:
            body['zoneName'] = zone_name
        if subnet_id is not None:
            body['subnetId'] = subnet_id
        if security_group_id is not None:
            body['securityGroupId'] = security_group_id
        if name is not None:
            body['name'] = name
        if deploy_set_id is not None:
            body['deploySetId'] = deploy_set_id
        if enable_ht:
            body['enableHt'] = True
        if admin_pass is not None:
            secret_access_key = self.config.credentials.secret_access_key
            cipher_admin_pass = aes128_encrypt_16char_key(admin_pass, secret_access_key)
            body['adminPass'] = cipher_admin_pass
        if auto_renew_time_unit is None:
            body['autoRenewTimeUnit'] = "month"
        else:
            body['autoRenewTimeUnit'] = auto_renew_time_unit
        if auto_renew_time != 0:
            body['autoRenewTime'] = auto_renew_time
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)


    def list_instances(self, marker=None, max_keys=1000, internal_ip=None, config=None):
        """
        Return a list of instances owned by the authenticated user.

        :param marker:
            The optional parameter marker specified in the original request to specify
            where in the results to begin listing.
            Together with the marker, specify the list result which listing should begin.
            If the marker is not specified, the list result will listing from the first one.
        :type marker: string

        :param max_keys:
            The optional parameter to specify the max number of list result to return.
            The default value is 1000.
        :type max_keys: int

        :param internal_ip:
            The identified internal ip of instance.
        :type internal_ip: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """

        path = b'/instance'
        params = {}

        if marker is not None:
            params['marker'] = marker
        if max_keys is not None:
            params['maxKeys'] = max_keys
        if internal_ip is not None:
            params['internalIp'] = internal_ip

        return self._send_request(http_methods.GET, path, params=params, config=config)



    @required(instance_id=(bytes, str))
    def get_instance(self, instance_id, contains_failed=False, config=None):
        """
        Get the detailed information of specified instance.

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param contains_failed:
            The optional parameters to get the failed message.If true, it means get the failed message.
        :type contains_failed: boolean

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s' % instance_id
        params = {}
        if contains_failed:
            params['containsFailed'] = contains_failed
        return self._send_request(http_methods.GET, path, params=params, config=config)

    @required(instance_id=(bytes, str))
    def start_instance(self, instance_id, config=None):
        """
        Starting the instance owned by the user.
        You can start the instance only when the instance is Stopped,
        otherwise, it's will get 409 errorCode.
        This is an asynchronous interface.

        :param instance_id: id of instance proposed to start
        :type instance_id: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s' % instance_id
        params = {
            'start': None
        }
        return self._send_request(http_methods.PUT, path, params=params, config=config)

    @required(instance_id=(bytes, str))
    def stop_instance(self, instance_id, force_stop=False, config=None):
        """
        Stopping the instance owned by the user.
        You can stop the instance only when the instance is Running,
        otherwise, it's will get 409 errorCode.
        This is an asynchronous interface.

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param force_stop:
            The optional parameter to stop the instance forcibly.If true,
            it will stop the instance just like power off immediately
            and it may result in losing important data which have not been written to disk.
        :type force_stop: boolean

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s' % instance_id
        body = {
            'forceStop': force_stop
        }
        params = {
            'stop': None
        }
        return self._send_request(http_methods.PUT, path, json.dumps(body),
                                  params=params, config=config)

    @required(instance_id=(bytes, str))
    def reboot_instance(self, instance_id, force_stop=False, config=None):
        """
        Rebooting the instance owned by the user.
        You can reboot the instance only when the instance is Running,
        otherwise, it's will get 409 errorCode.
        This is an asynchronous interface.

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param force_stop:
            The optional parameter to stop the instance forcibly.If true,
            it will stop the instance just like power off immediately
            and it may result in losing important data which have not been written to disk.
        :type force_stop: boolean

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s' % instance_id
        body = {
            'forceStop': force_stop
        }
        params = {
            'reboot': None
        }
        return self._send_request(http_methods.PUT, path, json.dumps(body),
                                  params=params, config=config)

    @required(instance_id=str)
    def batch_add_ip(self, instance_id, private_ips=None, secondary_private_ip_address_count=None, config=None):
        """
        batch_add_ip
        :param instance_id:
        :param private_ips:
        :param secondary_private_ip_address_count:
        :param config:
        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b'/instance/batchAddIp'
        body = {
            'instanceId': instance_id,
        }
        if private_ips is not None:
            body['privateIps'] = private_ips
        if secondary_private_ip_address_count is not None:
            body['secondaryPrivateIpAddressCount'] = secondary_private_ip_address_count
        params = {

        }
        return self._send_request(http_methods.PUT, path, json.dumps(body),
                                  params=params, config=config)

    @required(instance_id=str, private_ips=list)
    def batch_delete_ip(self, instance_id, private_ips, config=None):
        """
        :param instance_id:
        :param private_ips:
        :param config:
        :return:
        """
        path = b'/instance/batchDelIp'
        body = {
            'instanceId': instance_id,
            'privateIps': private_ips,
        }
        params = {

        }
        return self._send_request(http_methods.PUT, path, json.dumps(body),
                                  params=params, config=config)

    @required(instance_id=(bytes, str),
              renew_time_unit=(bytes, str),
              renew_time=int)
    def create_auto_renew_rules(self, instance_id, renew_time_unit, renew_time, config=None):
        """
        Creating auto renew rules for the bbc.
        It only works for the prepaid bbc.

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param renew_time_unit
            The parameter to specify the unit of the renew time.
            The renew time unit can be "month" or "year".
        :type renew_time_unit: string

        :param renew_time
            The parameter to specify the renew time.
        :type renew_time: int
        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/batchCreateAutoRenewRules'
        body = {
            'instanceId': instance_id,
            'renewTimeUnit': renew_time_unit,
            'renewTime': renew_time
        }
        params = {

        }
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(instance_id=(bytes, str))
    def delete_auto_renew_rules(self, instance_id, config=None):
        """
        Deleting auto renew rules for the bbc.
        It only works for the prepaid bbc.

        :param instance_id:
            The id of instance.
        :type instance_id: string
        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/batchDeleteAutoRenewRules'
        body = {
            'instanceId': instance_id
        }
        params = {

        }
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(instance_id=(bytes, str),
              name=(bytes, str))
    def modify_instance_name(self, instance_id, name, config=None):
        """
        Modifying the name of the instance.
        You can modify the instance name only when the instance is Running or Stopped ,
        otherwise, it's will get 409 errorCode.

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param name:
            The new value for instance's name.
        :type name: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s' % instance_id
        body = {
            'name': name
        }
        params = {
            'rename': None
        }
        return self._send_request(http_methods.PUT, path, json.dumps(body),
                                  params=params, config=config)



    def modify_instance_desc(self, instance_id, desc, config=None):
        """
        Modifying the description  of the instance.
        You can modify the description only when the instance is Running or Stopped ,
        otherwise, it's will get 409 errorCode.

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param desc:
            The new description of the instance.
        :type name: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s' % instance_id
        body = {
            'desc': desc
        }
        params = {
            'updateDesc': None
        }
        return self._send_request(http_methods.PUT, path, json.dumps(body),
                                  params=params, config=config)

    @required(instance_id=(bytes, str),
              image_id=(bytes, str),
              admin_pass=(bytes, str))
    def rebuild_instance(self, instance_id, image_id, admin_pass,
                         is_preserve_data=True, raid_id=None, sys_root_size=20, config=None):
        """
        Rebuilding the instance owned by the user.
        After rebuilding the instance,
        all of snapshots created from original instance system disk will be deleted,
        all of customized images will be saved for using in the future.
        This is an asynchronous interface

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param image_id:
            The id of the image which is used to rebuild the instance.
        :type image_id: string

        :param admin_pass:
            The admin password to login the instance.
            The admin password will be encrypted in AES-128 algorithm
            with the substring of the former 16 characters of user SecretKey.
            See more detail on
            https://cloud.baidu.com/doc/BBC/s/3jwvxu9iz#%E5%AF%86%E7%A0%81%E5%8A%A0%E5%AF%86%E4%BC%A0%E8%BE%93%E8%A7%84%E8%8C%83
        :type admin_pass: string

        :param is_preserve_data:
            Whether or not to retain data, the default is true. The raid_id and
            sys_root_size fields do not take effect when the value is true
        :type is_preserve_data: bool

        :param raid_id:
            The id of raid. See more details on
            https://cloud.baidu.com/doc/BBC/s/Bjwvxu9ul#%E6%9F%A5%E8%AF%A2raid
        :type raid_id: string

        :param sys_root_size:
            System root partition size, default is 20G, value range is 20-100
        :type sys_root_size: int

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        secret_access_key = self.config.credentials.secret_access_key
        cipher_admin_pass = aes128_encrypt_16char_key(admin_pass, secret_access_key)
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s' % instance_id
        body = {
            'imageId': image_id,
            'adminPass': cipher_admin_pass,
            'isPreserveData': is_preserve_data,
            'raidId': raid_id,
            'sysRootSize': sys_root_size
        }
        params = {
            'rebuild': None
        }
        return self._send_request(http_methods.PUT, path, json.dumps(body),
                                  params=params, config=config)

    @required(instance_id=(bytes, str))
    def release_instance(self, instance_id, config=None):
        """
        Releasing the instance owned by the user.
        Only the Postpaid instance or Prepaid which is expired can be released.
        After releasing the instance,
        all of the data will be deleted.
        all of snapshots created from original instance system disk will be deleted,
        all of customized images created from original instance system disk will be reserved.

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        instance_id = compat.convert_to_bytes(instance_id)
        api_version = b'/v2'
        path = b'/instance/%s' % instance_id
        return self._send_request(http_methods.DELETE, path, config=config, api_version=api_version)

    @required(instance_id=(bytes, str),
              admin_pass=(bytes, str))
    def modify_instance_password(self, instance_id, admin_pass, config=None):
        """
        Modifying the password of the instance.
        You can change the instance password only when the instance is Running or Stopped ,
        otherwise, it's will get 409 errorCode.
        This is an asynchronous interface.

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param admin_pass:
            The new password to update.
            The adminPass will be encrypted in AES-128 algorithm
            with the substring of the former 16 characters of user SecretKey.
        :type admin_pass: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        secret_access_key = self.config.credentials.secret_access_key
        cipher_admin_pass = aes128_encrypt_16char_key(admin_pass, secret_access_key)
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s' % instance_id
        body = {
            'adminPass': cipher_admin_pass
        }
        params = {
            'changePass': None
        }
        return self._send_request(http_methods.PUT, path, json.dumps(body),
                                  params=params, config=config)

    @required(bbc_ids = list)
    def get_vpc_subnet(self, bbc_ids, config=None):
        """
        Query VPC / Subnet information by BBC instance id

        :param bbc_ids:
            List of BBC instance IDs that need to query VPC / Subnet information
        :type bbc_ids: list

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b'/vpcSubnet'
        body = {
            'bbcIds': bbc_ids
        }

        return self._send_request(http_methods.POST, path, body=json.dumps(body), config = config)

    @required(instance_id=(bytes, str), change_tags = list)
    def unbind_tags(self, instance_id, change_tags, config=None):
        """
        Unbind the tags of existing instances

        :param instance_id:
            The id of instance.
        :type instance_id: string

        :param change_tags:
            List of tags to be unbind
        :type change_tags: list

        :return:
        :rtype baidubce.bce_response.BceResponse

        """
        instance_id = compat.convert_to_bytes(instance_id)
        path = b'/instance/%s/tag' % instance_id

        params = {
            'unbind': None
        }

        body = {
            'changeTags': change_tags
        }

        return self._send_request(http_methods.PUT, path, body=json.dumps(body),
                                  params=params, config=config)


    def list_flavors(self, config=None):
        """
        :return:
        :rtype baidubce.bce_response.BceResponse

        """
        path = b'/flavor'
        return self._send_request(http_methods.GET, path, config = config)

    @required(flavor_id = (bytes, str))
    def get_flavor(self, flavor_id, config=None):
        """
        :param flavor_id:
            The id of flavor.
        :type flavor_id: string

        :return:
        :rtype baidubce.bce_response.BceResponse

        """
        flavor_id = compat.convert_to_bytes(flavor_id)
        path = b'/flavor/%s' % flavor_id
        return self._send_request(http_methods.GET, path, config=config)

    @required(flavor_id=(bytes, str))
    def get_flavor_raid(self, flavor_id, config=None):
        """
        :param flavor_id:
            The id of flavor.
        :type flavor_id: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        flavor_id = compat.convert_to_bytes(flavor_id)
        path = b'/flavorRaid/%s' % flavor_id
        return self._send_request(http_methods.GET, path, config=config)

    @required(image_name=(bytes, str),
              instance_id=(bytes, str))
    def create_image_from_instance_id(self,
                                      image_name,
                                      instance_id,
                                      client_token=None,
                                      config=None):
        """
        Creating a customized image which can be used for creating instance.
        You can create an image from an instance with this method.
        While creating an image from an instance, the instance must be Running or Stopped,
        otherwise, it's will get 409 errorCode.
        This is an asynchronous interface.

        :param image_name:
            The name for the image that will be created.
            The name length from 1 to 65,only contains letters,digital and underline.
        :type image_name: string

        :param instance_id:
            The optional parameter specify the id of the instance which will be used to create the new image.
            When instanceId and snapshotId are specified ,only instanceId will be used.
        :type instance_id: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if client token is provided.
            If the clientToken is not specified by the user,
            a random String generated by default algorithm will be used.
            See more detail at
            https://bce.baidu.com/doc/BCC/API.html#.E5.B9.82.E7.AD.89.E6.80.A7
        :type client_token: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b'/image'
        params = None
        if client_token is None:
            params = {
                'clientToken': generate_client_token()
            }
        else:
            params = {
                'clientToken': client_token
            }
        body = {
            'imageName': image_name,
            'instanceId': instance_id
        }

        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    def list_images(self, image_type='All', marker=None, max_keys=1000, config=None):
        """
        Listing images owned by the authenticated user.

        :param image_type:
            The optional parameter to filter image to list.
            See more detail at
            https://bce.baidu.com/doc/BCC/API.html#ImageType"
        :type image_type: menu{'All', System', 'Custom', 'Integration'}

        :param marker:
            The optional parameter marker specified in the original request to specify
            where in the results to begin listing.
            Together with the marker, specifies the list result which listing should begin.
            If the marker is not specified, the list result will listing from the first one.
        :type marker: string

        :param max_keys:
            The optional parameter to specify the max number of list result to return.
            The default value is 1000.
        :type max_keys: int

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b'/image'
        params = {
            'imageType': image_type
        }
        if marker is not None:
            params['marker'] = marker
        if max_keys is not None:
            params['maxKeys'] = max_keys

        return self._send_request(http_methods.GET, path, params=params, config=config)

    @required(image_id=(bytes, str))
    def get_image(self, image_id, config=None):
        """
        Get the detail information of specified image.

        :param image_id:
            The id of image.
        :type image_id: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        image_id = compat.convert_to_bytes(image_id)
        path = b'/image/%s' % image_id
        return self._send_request(http_methods.GET, path, config=config)

    @required(image_id=(bytes, str))
    def delete_image(self, image_id, config=None):
        """
        Deleting the specified image.
        Only the customized image can be deleted,
        otherwise, it's will get 403 errorCode.

        :param image_id:
            The id of image.
        :type image_id: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        image_id = compat.convert_to_bytes(image_id)
        path = b'/image/%s' % image_id
        return self._send_request(http_methods.DELETE, path, config=config)

    def get_operation_log(self, marker=None, max_keys=100, start_time=None, end_time=None, config=None):
        """
        Querying information about physical machine operation logs

        :param marker:
            The optional parameter marker specified in the original request to specify
            where in the results to begin listing.
            Together with the marker, specifies the list result which listing should begin.
            If the marker is not specified, the list result will listing from the first one.
        :type marker: string

        :param max_keys:
            The optional parameter to specify the max number of list result to return.
            The default value is 1000.
        :type max_keys: int

        :param start_time:
            The start time of the physical machine operation (UTC time),
            the format is yyyy-MM-dd'T'HH: mm: ss'Z ', if it is empty, query the operation log of the day
        :type start_time: string

        :params end_time
            The end time of the physical machine operation (UTC time),
            the format is yyyy-MM-dd'T'HH: mm: ss'Z ', if it is empty, query the operation log of the day
        :type end_time: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b'/operationLog'
        params = {}
        if marker is not None:
            params['marker'] = marker
        if max_keys is not None:
            params['maxKeys'] = max_keys
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time

        return self._send_request(http_methods.GET, path, params=params, config=config)

    @required(concurrency=int, strategy=(bytes, str))
    def create_deploy_set(self, concurrency, strategy, name=None, desc=None, client_token=None, config=None):
        """
        Create a deploy set based on a specified policy and concurrency

        :param concurrency:
            Deployment set concurrency, range [1,5]
        :type concurrency: int

        :param strategy:
            Deployment set strategy, currently BBC strategy only supports: "tor_ha", "host_ha"
        :type strategy: string

        :param name:
            Deployment set name, supports uppercase and lowercase letters, numbers,
            Chinese, and -_ /. Special characters, must start with a letter, length 1-65
        :type name: string

        :param desc:
            Deployment set description
        :type desc: string

        :param client_token:
            An ASCII string whose length is less than 64.
            The request will be idempotent if client token is provided.
            If the clientToken is not specified by the user,
            a random String generated by default algorithm will be used.
            See more detail at
            https://bce.baidu.com/doc/BCC/API.html#.E5.B9.82.E7.AD.89.E6.80.A7
        :type client_token: string

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b'/deployset'
        params = {}
        if client_token is None:
            params['clientToken'] = generate_client_token()
        else:
            params['clientToken'] = client_token
        body = {
            "strategy": strategy,
            "concurrency": concurrency
        }
        if name is not None:
            body["name"] = name
        if desc is not None:
            body['desc'] = desc
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    def list_deploy_sets(self, config=None):
        """
        List all deploy sets

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        path = b'/deployset'
        return self._send_request(http_methods.GET, path, config=config)

    @required(deploy_set_ids=(bytes, str))
    def get_deploy_set(self, deploy_set_id, config=None):
        """
        Get the specified deploy set

        :param deploy_set_id:
            The id of the deploy set
        :type deploy_set_id: String

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        deploy_set_id = compat.convert_to_bytes(deploy_set_id)
        path = b'/deployset/%s' % deploy_set_id
        return self._send_request(http_methods.GET, path, config=config)

    @required(deploy_set_ids=(bytes, str))
    def delete_deploy_set(self, deploy_set_id, config=None):
        """
        Delete the specified deploy sets

        :param deploy_set_ids:
            The ids of the deploy sets you want to delete
        :type deploy_set_ids: list

        :return:
        :rtype baidubce.bce_response.BceResponse
        """
        deploy_set_id = compat.convert_to_bytes(deploy_set_id)
        path = b'/deployset/%s' % deploy_set_id
        return self._send_request(http_methods.DELETE, path, config=config)









def generate_client_token_by_uuid():
    """
    The default method to generate the random string for client_token
    if the optional parameter client_token is not specified by the user.
    :return:
    :rtype string
    """
    return str(uuid.uuid4())


def generate_client_token_by_random():
    """
    The alternative method to generate the random string for client_token
    if the optional parameter client_token is not specified by the user.
    :return:
    :rtype string
    """
    client_token = ''.join(random.sample(string.ascii_letters + string.digits, 36))
    return client_token


generate_client_token = generate_client_token_by_uuid



