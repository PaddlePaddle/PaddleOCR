# -*- coding: utf-8 -*-

"""
This module provide billing information and IkeConfig and IPSec
"""


class Billing(object):
    """
    billing information
    """

    def __init__(self, payment_timing=None, billing_method=None, reservation_length=None,
                 reservation_time_unit=None):
        """
        :type payment_timing: string
        :param payment_timing: 'Prepaid'  'Postpaid'

        :type billing_method: string
        :param billing_method: 'ByTraffic' 'ByBandwidt'

        :type reservation_length: int
        :param reservation_length: purchase length

        :type reservation_time_unit: string
        :param reservation_time_unit: time unit of purchasing，currently only supports monthly
        """
        self.payment_timing = payment_timing
        self.billing_method = billing_method
        self.reservation_length = reservation_length or 1
        self.reservation_time_unit = reservation_time_unit or 'Month'


class IkeConfig(object):
    """
    IKE Configuration example
    """

    def __init__(self, ike_version=None, ike_mode=None, ike_enc_alg=None, ike_auth_alg=None, ike_pfs=None,
                 ike_lifeTime=None):
        """
        :type ike_version: string
        :param ike_version: Version, value range ：v1/v2

        :type ike_mode: string
        :param ike_mode: Negotiation mode, value range ：main/aggressive

        :type ike_enc_alg: string
        :param ike_enc_alg: Encryption algorithm, value range ：aes/aes192/aes256/3des

        :type ike_auth_alg: string
        :param ike_auth_alg: Authentication algorithm, value range ：sha1/md5

        :type ike_pfs: string
        :param ike_pfs: DH Grouping, value range ：group2/group5/group14/group24

        :type ike_lifeTime: string
        :param ike_lifeTime: SA Life cycle, value range ：60-86400
        """
        self.ike_version = ike_version
        self.ike_mode = ike_mode
        self.ike_enc_alg = ike_enc_alg
        self.ike_auth_alg = ike_auth_alg
        self.ike_pfs = ike_pfs
        self.ike_lifeTime = ike_lifeTime


class IpsecConfig(object):
    """
    IPSec Configuration example
    """
    def __init__(self, ipsec_enc_alg=None, ipsec_auth_alg=None, ipsec_pfs=None, ipsec_lifetime=None):
        """
        :type ipsec_enc_alg: string
        :param ipsec_enc_alg: Encryption algorithm, value range ：aes/aes192/aes256/3des

        :type ipsec_auth_alg: string
        :param ipsec_auth_alg: Authentication algorithm, value range ：sha1/md5

        :type ipsec_pfs: string
        :param ipsec_pfs: group2/group5/group14/group24

        :type ipsec_lifetime: string
        :param ipsec_lifetime: SA Life cycle, value range ：180-86400
        """
        self.ipsec_enc_alg = ipsec_enc_alg
        self.ipsec_auth_alg = ipsec_auth_alg
        self.ipsec_pfs = ipsec_pfs
        self.ipsec_lifetime = ipsec_lifetime