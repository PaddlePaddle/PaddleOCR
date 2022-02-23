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
This module provides a client class for KMS.
"""

import copy
import json
import logging
import random
import string
import uuid

from baidubce.bce_base_client import BceBaseClient
from baidubce.utils import required
from baidubce.auth import bce_v1_signer
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_methods
import base64

_logger = logging.getLogger(__name__)

class KmsClient(BceBaseClient):
    """
    sdk client
    """
    def __init__(self, config=None):
        BceBaseClient.__init__(self, config)

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
        if headers is None:
            headers = {b'Accept': b'*/*',
                       b'Content-Type': b'application/json;charset=utf-8'}
        return bce_http_client.send_request(config, bce_v1_signer.sign,
                                            [handler.parse_error, body_parser],
                                            http_method, path, body, headers,
                                            params)

    @required(protectedBy=(bytes, str), keySpec=(bytes, str), origin=(bytes, str))
    def create_masterKey(self, description, protectedBy, keySpec,
                        origin, keyUsage="ENCRYPT_DECRYPT", config=None):
        """
        create a master key with the specified options.
        :type description: string
        :param description: a description about the master key

        :type protectedBy: constants.ProtectedBy
        :param protectedBy: the protect level about the master key, you can choose HSM or SOFTWARE

        :type keySpec: constants.KeySpec
        :param keySpec:  key specification about the master key. now you can choose the BAIDU_AES_256, 
        AES_128, AES_256, RSA_1024, RSA_2048, RSA_4096

        :type keyUsage: string
        :param keyUsage:  default "ENCRYPT_DECRYPT"

        :type origin: constants.Origin
        :param origin:  origin of the master key. you can choose BAIDU_KMS or EXTERNAL
        """
        path = b'/'
        params = {}
        params['action'] = b'CreateKey'
        body={}
        if description:
            body['description'] = description
        body['protectedBy'] = protectedBy
        body['keySpec'] = keySpec
        body['origin'] = origin
        body['keyUsage'] = keyUsage
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(limit=int)
    def list_masterKey(self, limit, marker="", config=None):
        """
        list your masterkey 
        :type limit: int
        :param limit: the number of masterKey you want list

        :type marker: string
        :param marker: the marker keyid , kms will search from the marker, default ""
        """
        path = b'/'
        params = {}
        params['action'] = b'ListKeys'
        body={}
        body['limit'] = limit
        body['marker'] = marker
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes), plaintext=(str, bytes))
    def encrypt(self, keyId, plaintext, config=None):
        """
        encrypt the plaintext
        :type keyId: string
        :param keyId: indicate kms will use which masterkey to encrypt
        
        :type plaintext: string
        :param plaintext: the plaintext need encrypted by kms
        """
        path = b'/'
        params = {}
        params['action'] = b'Encrypt'
        body={}
        body['keyId'] = keyId
        body['plaintext'] = plaintext
        try:
            base64.b64decode(plaintext)
        except TypeError:
            raise TypeError("please input base64 string")
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes), ciphertext=(str, bytes))
    def decrypt(self, keyId, ciphertext, config=None):
        """
        decrypt the ciphertext
        :type keyId: string
        :param keyId: indicate kms will use which masterkey to decrypt

        :type ciphertext: string
        :param ciphertext:  the ciphertext need decrypted by kms
        """
        path = b'/'
        params = {}
        params['action'] = b'Decrypt'
        body={}
        body['keyId'] = keyId
        body['ciphertext'] = ciphertext
        try:
            base64.b64decode(ciphertext)
        except TypeError:
            raise TypeError("please input base64 string")
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes), keySpec=(str, bytes))
    def generate_dataKey(self, keyId, keySpec, numberOfBytes=-1, config=None):
        """
        generate a data key by master key
        :type keyId: string
        :param keyId: indicate kms will use which masterkey to generate data key

        :type keySpec: string
        :param keySpec: AES_128 or AES_256

        :type numberOfBytes: int
        :param numberOfBytes: The length of data key
        """
        path = b'/'
        params = {}
        params['action'] = b'GenerateDataKey'
        body={}
        body['keyId'] = keyId
        if keySpec != "AES_128" and keySpec != "AES_256":
            raise ValueError("only support AES_128 and AES_256")
        body['keySpec'] = keySpec
        body['numberOfBytes'] = numberOfBytes
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes))
    def enable_masterKey(self, keyId, config=None):
        """
        enable your master key
        :type keyId: string
        :param keyId: the keyId of masterkey will be enable
        """
        path = b'/'
        params = {}
        params['action'] = b'EnableKey'
        body={}
        body['keyId'] = keyId
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes))
    def disable_masterKey(self, keyId, config=None):
        """
        disable your master key
        :type keyId: string
        :param keyId: the keyId of masterkey will be diable
        """
        path = b'/'
        params = {}
        params['action'] = b'DisableKey'
        body={}
        body['keyId'] = keyId
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes), pendingWindowInDays=int)
    def scheduleDelete_masterKey(self, keyId, pendingWindowInDays, config=None):
        """
        schedule delete master key
        :type keyId: string
        :param keyId: the keyId of masterkey will be deleted

        :type pendingWindowInDays: int
        :pram pendingWindowInDays: kms will wait pendingWindowInDays day then delete the key
        """
        path = b'/'
        params = {}
        params['action'] = b'ScheduleKeyDeletion'
        body={}
        body['keyId'] = keyId
        if pendingWindowInDays > 30 or pendingWindowInDays < 7:
            raise ValueError("please input pendingWindowInDays >=7 and <=30")
        body['pendingWindowInDays'] = pendingWindowInDays
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes))
    def cancelDelete_maaterKey(self, keyId, config=None):
        """
        cancel delete master key
        :type keyId: string
        :param keyId: the keyId of masterkey will cancel delete
        """
        path = b'/'
        params = {}
        params['action'] = b'CancelKeyDeletion'
        body={}
        body['keyId'] = keyId
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes))
    def describe_masterKey(self, keyId, config=None):
        """
        descript the master key
        :type keyId: string
        :param keyId: the keyId of masterkey 
        """
        path = b'/'
        params = {}
        params['action'] = b'DescribeKey'
        body={}
        body['keyId'] = keyId
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes))
    def get_parameters_for_import(self, keyId, publicKeyEncoding, wrappingAlgorithm="RSAES_PKCS1_V1_5",
                                wrappingKeySpec="RSA_2048", config=None):
        """
        get parameters for import
        :type keyId: string
        :param keyId: the keyId of masterkey

        :type wrappingAlgorithm: string
        :param wrappingAlgorithm: the algorithm for user encrypt local key

        :type wrappingKeySpec:string
        :param wrappingKeySpec: the pubkey spec for user encrypt local key
        """
        path = b'/'
        params = {}
        params['action'] = b'GetParametersForImport'
        body={}
        body['keyId'] = keyId
        if wrappingAlgorithm != "RSAES_PKCS1_V1_5":
            raise TypeError("only support RSAES_PKCS1_V1_5")
        body['wrappingAlgorithm'] = wrappingAlgorithm
        if wrappingKeySpec != "RSA_2048":
            raise TypeError("only support RSA_2048")
        body['wrappingKeySpec'] = wrappingKeySpec
        if publicKeyEncoding != "RAW_HEX" and publicKeyEncoding != "BASE64" and publicKeyEncoding != "PEM":
            raise ValueError("only support RAW_HEX or BASE64 or PEM")
        body['publicKeyEncoding'] = publicKeyEncoding
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes), importToken=(str, bytes), encryptedKey=(str, bytes), keySpec=(str, bytes))
    def import_symmetricMasterKey(self, keyId, importToken, encryptedKey, keySpec,
                                keyUsage="ENCRYPT_DECRYPT", config=None):
        """
        import symmetric key
        :type keyId: string
        :param keyId: the keyId of masterkey

        :type importToken: string
        :param importToken: token from import parameter

        :type encryptedKey: string
        :param encryptedKey: the symmetric key encrypted by pubkey

        :type keySpec: string
        :param keySpec: the import key spec

        :type keyUsage: string
        :param keyUsage: default "ENCRYPT_DECRYPT"
        """
        path = b'/'
        params = {}
        params['action'] = b'ImportKey'
        body={}
        body['keyId'] = keyId
        body['importToken'] = importToken
        body['encryptedKey'] = encryptedKey
        body['keySpec'] = keySpec
        body['keyUsage'] = keyUsage
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)

    @required(keyId=(str, bytes),
            importToken=(str, bytes),
            asymmetricKeySpec=(str, bytes),
            asymmetricKeyUsage=(str, bytes),
            encryptedKeyEncryptionKey=(str, bytes),
            asymmetricKey=object)
    def import_asymmetricMasterKey(self, keyId, importToken, asymmetricKeySpec, encryptedKeyEncryptionKey,
                                asymmetricKeyUsage="ENCRYPT_DECRYPT", config=None, **kwargs):
        """
        import asymmetric key
        :type keyId: string
        :param keyId: the keyId of masterkey

        :type importToken: string
        :param importToken: token from import parameter

        :type asymmetricKeySpec: string
        :param asymmetricKeySpec: the import key spec

        :type encryptedKeyEncryptionKey: string
        :param encryptedKeyEncryptionKey: EncryptionKey

        :type asymmetricKey: **args
        :param asymmetricKey: include publicKeyDer encryptedD encryptedP encryptedQ encryptedDp encryptedDq encryptedQinv
        """
        path = b'/'
        params = {}
        params['action'] = b'ImportAsymmetricKey'
        body={}
        body['keyId'] = keyId
        body['importToken'] = importToken
        body['asymmetricKeySpec'] = asymmetricKeySpec
        body['asymmetricKeyUsage'] = asymmetricKeyUsage
        body['encryptedKeyEncryptionKey'] = encryptedKeyEncryptionKey
        body['encryptedRsaKey'] = {}
        if kwargs['publicKeyDer'] is None:
            raise ValueError('arg "publicKeyDer" should not be None')
        body['encryptedRsaKey']['publicKeyDer'] = kwargs['publicKeyDer']
        if kwargs['encryptedD'] is None:
            raise ValueError('arg "encryptedD" should not be None')
        body['encryptedRsaKey']['encryptedD'] = kwargs['encryptedD']
        if kwargs['encryptedP'] is None:
            raise ValueError('arg "encryptedP" should not be None')
        body['encryptedRsaKey']['encryptedP'] = kwargs['encryptedP']
        if kwargs['encryptedQ'] is None:
            raise ValueError('arg "encryptedQ" should not be None')
        body['encryptedRsaKey']['encryptedQ'] = kwargs['encryptedQ']
        if kwargs['encryptedDp'] is None:
            raise ValueError('arg "encryptedDp" should not be None')
        body['encryptedRsaKey']['encryptedDp'] = kwargs['encryptedDp']
        if kwargs['encryptedDq'] is None:
            raise ValueError('arg "encryptedDq" should not be None')
        body['encryptedRsaKey']['encryptedDq'] = kwargs['encryptedDq']
        if kwargs['encryptedQinv'] is None:
            raise ValueError('arg "encryptedQinv" should not be None')
        body['encryptedRsaKey']['encryptedQinv'] = kwargs['encryptedQinv']
        return self._send_request(http_methods.POST, path, json.dumps(body),
                                  params=params, config=config)




