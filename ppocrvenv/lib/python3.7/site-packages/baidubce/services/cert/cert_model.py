# Copyright 2020 Baidu, Inc.
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
This module provides models for CERT-SDK
see more detail in https://cloud.baidu.com/doc/Reference/s/Gjwvz27xu
"""


class CertCreateRequest(object):
    """
    This class define certificate creation information

    param: cert_name:
        The certificate name.

    param: cert_server_data:
        The SSL client/server certificate in base-64 format.

    param: cert_private_data:
        The private key in base-64 format.

    param: cert_link_data:
        The certificate chain without server certificate.

    param: cert_type:
        The certificate type,
        available values are [1,2] now,
        1 means cert_server_data is a server certificate
        2 means cert_server_data is a client certificate
    """

    def __init__(self, cert_name, cert_server_data, cert_private_data=None, cert_link_data=None, cert_type=1):
        self.certName = cert_name
        self.certServerData = cert_server_data
        self.certPrivateData = cert_private_data
        self.certLinkData = cert_link_data
        self.certType = cert_type
