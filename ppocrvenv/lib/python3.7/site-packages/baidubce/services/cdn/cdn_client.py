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
This module provides a client class for CDN.
"""

import copy
import json
import logging
import baidubce

from baidubce import bce_base_client
from baidubce.auth import bce_v1_signer
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_content_types
from baidubce.http import http_headers
from baidubce.http import http_methods
from baidubce.exception import BceClientError
from baidubce.exception import BceServerError
from baidubce.utils import required
from baidubce import utils
from baidubce.services.cdn.cdn_stats_param import CdnStatsParam

_logger = logging.getLogger(__name__)


class CdnClient(bce_base_client.BceBaseClient):
    """
    CdnClient
    """
    prefix = b"/v2"

    def __init__(self, config=None):
        bce_base_client.BceBaseClient.__init__(self, config)

    def list_domains(self, config=None):
        """
        get domain list

        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET,
            '/domain',
            config=config)

    def list_user_domains(self, status, rule=None, config=None):
        """
        get user domain list

        :param status: Specify the domain whose return 'status' is status
        :type status: string
        :param rule: domain Fuzzy Matching
        :type rule: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['status'] = status 

        if rule is not None:
            params['rule'] = rule

        return self._send_request(
            http_methods.GET, '/user/domains',
            params=params,
            config=config)

    def valid_domain(self, domain, config=None):
        """
        query if a domain can be create
        
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET,
            '/domain/' + domain + '/valid',
            config=config)

    def get_domain_cache_full_url(self, domain, config=None):
        """
        get cache full url of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'cacheFullUrl': ''},
            config=config)

    
    def set_domain_cache_share(self, domain, cache_share, config=None):
        """
        update cacheShare of the domain
        :param domain: the domain name
        :type domain: string
        :param cache_share: detailed configuration of cacheShare
        :type cache_share: dictionary
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'cacheShare': ''},
            body=json.dumps({'cacheShare': cache_share}),
            config=config)


    def get_domain_cache_share(self, domain, config=None):
        """
        get cacheShare configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'cacheShare': ''},
            config=config)

    
    def set_domain_traffic_limit(self, domain, traffic_limit, config=None):
        """
        update trafficLimit of the domain
        :param domain: the domain name
        :type domain: string
        :param traffic_limit: detailed configuration of trafficLimit
        :type traffic_limit: dictionary
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'trafficLimit': ''},
            body=json.dumps({'trafficLimit': traffic_limit}),
            config=config)


    def get_domain_traffic_limit(self, domain, config=None):
        """
        get trafficLimit configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'trafficLimit': ''},
            config=config)

    
    def set_domain_ua_acl(self, domain, ua_acl, config=None):
        """
        update uaAcl of the domain
        :param domain: the domain name
        :type domain: string
        :param ua_acl: detailed configuration of uaAcl
        :type ua_acl: dictionary
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'uaAcl': ''},
            body=json.dumps({'uaAcl': ua_acl}),
            config=config)


    def get_domain_ua_acl(self, domain, config=None):
        """
        get uaAcl configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'uaAcl': ''},
            config=config)


    def set_domain_origin_protocol(self, domain, origin_protocol, config=None):
        """
        update originProtocol of the domain
        :param domain: the domain name
        :type domain: string
        :param origin_protocol: detailed configuration of originProtocol
        :type origin_protocol: dictionary
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'originProtocol': ''},
            body=json.dumps({'originProtocol': origin_protocol}),
            config=config)


    def get_domain_origin_protocol(self, domain, config=None):
        """
        get originProtocol configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'originProtocol': ''},
            config=config)


    def set_domain_retry_origin(self, domain, retry_origin, config=None):
        """
        update retryOrigin of the domain
        :param domain: the domain name
        :type domain: string
        :param retry_origin: detailed configuration of retryOrigin
        :type retry_origin: dictionary
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'retryOrigin': ''},
            body=json.dumps({'retryOrigin': retry_origin}),
            config=config)


    def get_domain_retry_origin(self, domain, config=None):
        """
        get retryOrigin configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'retryOrigin': ''},
            config=config)


    def set_domain_ipv6_dispatch(self, domain, ipv6_dispatch, config=None):
        """
        update ipv6Dispatch of the domain
        :param domain: the domain name
        :type domain: string
        :param ipv6_dispatch: detailed configuration of IPv6 access and IPv6 back to source
        :type ipv6_dispatch: IPv6Config
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'ipv6Dispatch': ''},
            body=json.dumps({'ipv6Dispatch': ipv6_dispatch}),
            config=config)


    def get_domain_ipv6_dispatch(self, domain, config=None):
        """
        get ipv6Dispatch configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'ipv6Dispatch': ''},
            config=config)

    
    def set_domain_quic(self, domain, quic, config=None):
        """
        update QUIC of the domain
        :param domain: the domain name
        :type domain: string
        :param quic: detailed configuration of Quick UDP Internet Connection
        :type quic: bool
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'quic': ''},
            body=json.dumps({'quic': quic}),
            config=config)


    def get_domain_quic(self, domain, config=None):
        """
        get quic configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'quic': ''},
            config=config)

    
    def set_domain_offline_mode(self, domain, offline_mode, config=None):
        """
        update offlineMode of the domain
        :param domain: the domain name
        :type domain: string
        :param offline_mode: detailed configuration of offlineMode
        :type offline_mode: bool
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'offlineMode': ''},
            body=json.dumps({'offlineMode': offline_mode}),
            config=config)


    def get_domain_offline_mode(self, domain, config=None):
        """
        get offlineMode configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'offlineMode': ''},
            config=config)


    def set_domain_ocsp(self, domain, ocsp, config=None):
        """
        update OCSP configuration of the domain
        :param domain: the domain name
        :type domain: string
        :param ocsp: detailed configuration of Online Certificate Status Protocol
        :type ocsp: bool
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'ocsp': ''},
            body=json.dumps({'ocsp': ocsp}),
            config=config)


    def get_domain_ocsp(self, domain, config=None):
        """
        get ocsp configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'ocsp': ''},
            config=config)


    def set_domain_error_page(self, domain, error_page, config=None):
        """
        update error_page of the domain
        :param domain: the domain name
        :type domain: string
        :param error_page: Detailed configuration of custom error jump pages
        :type error_page: list<ErrorPage>
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'errorPage': ''},
            body=json.dumps({'errorPage': error_page}),
            config=config)

    def get_domain_error_page(self, domain, config=None):
        """
        get error page configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'errorPage': ''},
            config=config)


    def get_domain_referer_acl(self, domain, config=None):
        """
        get referer acl configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'refererACL': ''},
            config=config)


    def get_domain_ip_acl(self, domain, config=None):
        """
        get ip acl configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'ipACL': ''},
            config=config)


    def set_domain_cors(self, domain, cors, config=None):
        """
        update cors of the domain
        :param domain: the domain name
        :type domain: string
        :param cors: Accelerating Cross-domain Settings for Domain Names
        :type cors
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'cors': ''},
            body=json.dumps({'cors': cors}),
            config=config)

    def get_domain_cors(self, domain, config=None):
        """
        get cors configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'cors': ''},
            config=config)

    def set_domain_access_limit(self, domain, access_limit, config=None):
        """
        update access limit of the domain
        :param domain: the domain name
        :type domain: string
        :param access_limit: Setting IP Access Limitation
        :type AccessLimit
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'accessLimit': ''},
            body=json.dumps({'accessLimit': access_limit}),
            config=config)

    def get_domain_access_limit(self, domain, config=None):
        """
        get access limit configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'accessLimit': ''},
            config=config)

    def set_domain_client_ip(self, domain, client_ip, config=None):
        """
        update Getting Real User IP Settings
        :param domain: the domain name
        :type domain: string
        :param client_ip: Getting Real User IP Settings
        :type ClientIp
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'clientIp': ''},
            body=json.dumps({'clientIp': client_ip}),
            config=config)

    def get_domain_client_ip(self, domain, config=None):
        """
        get client ip configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'clientIp': ''},
            config=config)

    def set_domain_range_switch(self, domain, range_switch, config=None):
        """
        update range backsource Settings
        :param domain: the domain name
        :type domain: string
        :param range_switch: rangre backsource settings
        :type bool
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'rangeSwitch': ''},
            body=json.dumps({'rangeSwitch': range_switch}),
            config=config)

    def get_domain_range_switch(self, domain, config=None):
        """
        get range switch configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'rangeSwitch': ''},
            config=config)

    def set_domain_mobile_access(self, domain, mobile_access, config=None):
        """
        update mobile access Settings
        :param domain: the domain name
        :type domain: string
        :param mobile_access: mobile access settings
        :type MobileAccess
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'mobileAccess': ''},
            body=json.dumps({'mobileAccess': mobile_access}),
            config=config)

    def get_domain_mobile_access(self, domain, config=None):
        """
        get mobile access configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'mobileAccess': ''},
            config=config)

    def set_domain_http_header(self, domain, http_header, config=None):
        """
        update http header Settings
        :param domain: the domain name
        :type domain: string
        :param http_header: http header settings
        :type HttpHeader
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'httpHeader': ''},
            body=json.dumps({'httpHeader': http_header}),
            config=config)

    def get_domain_http_header(self, domain, config=None):
        """
        get http header configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'httpHeader': ''},
            config=config)

    def set_domain_file_trim(self, domain, file_trim, config=None):
        """
        update file trim Settings
        :param domain: the domain name
        :type domain: string
        :param file_trim: file trim settings
        :type bool
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'fileTrim': ''},
            body=json.dumps({'fileTrim': file_trim}),
            config=config)

    def get_domain_file_trim(self, domain, config=None):
        """
        get file trim configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'fileTrim': ''},
            config=config)

    def set_domain_media_drag(self, domain, media_drag, config=None):
        """
        update media drag Settings
        :param domain: the domain name
        :type domain: string
        :param media_drag: media drag settings
        :type MediaDragConf
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'mediaDrag': ''},
            body=json.dumps({'mediaDragConf': media_drag}),
            config=config)

    def get_domain_media_drag(self, domain, config=None):
        """
        get range switch configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'mediaDrag': ''},
            config=config)

    def set_domain_compress(self, domain, compress, config=None):
        """
        update compress Settings
        :param domain: the domain name
        :type domain: string
        :param compress: media drag settings
        :type compress
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'compress': ''},
            body=json.dumps({'compress': compress}),
            config=config)

    def get_domain_compress(self, domain, config=None):
        """
        get compress configuration of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'compress': ''},
            config=config)

    def get_domain_records(self, Type=None, start_time=None, end_time=None,
            url=None, marker=None, config=None):
        """
        Query refresh and preload records
        :param Type: None
        :type string
        :param start_time: None
        :type Timestamp
        :param end_time: None
        :type Timestamp
        :param url: None
        :type string
        :param marker: None
        :type string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """

        params = {}

        if Type is not None:
            params['type'] = Type

        if start_time is not None:
            params['startTime'] = start_time

        if end_time is not None:
            params['endTime'] = end_time

        if url is not None:
            params['url'] = url
        
        if marker is not None:
            params['marker'] = marker

        if params is None:
            params = {}

        return self._send_request(
            http_methods.GET, '/cache/records',
            params = params,
            config=config)

    def set_dsa(self, action, config=None):
        """
        set das enable
        :param action: dsa enable
        :type string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/dsa',
            body=json.dumps({'action': action}),
            config=config)

    def set_domain_dsa(self, domain, dsa, config=None):
        """
        update domain dsa Settings
        :param domain: the domain name
        :type domain: string
        :param dsa: domain dsa settings
        :type DSA
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'dsa': ''},
            body=json.dumps({'dsa': dsa}),
            config=config)

    def get_dsa_domains(self, config=None):
        """
        get dsa domain list
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET,
            '/dsa/domain',
            config=config)

    def get_log_list(self, log, config=None):
        """
        get log list
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.POST,
            '/log/list',
            body=json.dumps(log),
            config=config)

    def create_domain(self, domain, origin, other=None, config=None):
        """
        create domain
        :param domain: the domain name
        :type domain: string
        :param origin: the origin address list
        :type origin: list<OriginPeer>
        :param other: the other config
        :type other: list<CONFIG>
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        body = {'origin': origin}
        if other is not None:
            body.update(other)

        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain,
            body=json.dumps(body),
            config=config)

    def delete_domain(self, domain, config=None):
        """
        delete a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.DELETE,
            '/domain/' + domain,
            config=config)

    def enable_domain(self, domain, config=None):
        """
        enable a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.POST,
            '/domain/' + domain,
            params={'enable': ''},
            config=config)

    def disable_domain(self, domain, config=None):
        """
        disable a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.POST,
            '/domain/' + domain,
            params={'disable': ''},
            config=config)

    def get_domain_config(self, domain, config=None):
        """
        get configuration of the domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        if domain[0] == '*':
            domain = '%2A' + domain[1:]

        return self._send_request(
            http_methods.GET,
            '/domain/' + domain + '/config',
            config=config)

    def set_domain_multi_configs(self, domain, multi_configs, config=None):
        """
        update multiConfigs of the domain
        :param domain: the domain name
        :type domain: string
        :param multi_configs: the domain config list
        :type multi_configs: list<config>
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'multiConfigs': ''},
            body=json.dumps({'multiConfigs': multi_configs}),
            config=config)

    def set_domain_origin(self, domain, origin, other=None, config=None):
        """
        update origin address of the domain
        :param domain: the domain name
        :type domain: string
        :param origin: the origin address list
        :type origin: list<OriginPeer>
        :param other: origin extension configuration
        :type other: dictionary
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        body = {'origin': origin}
        if other is not None:
            body.update(other)

        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'origin': ''},
            body=json.dumps(body),
            config=config)

    def get_domain_cache_ttl(self, domain, config=None):
        """
        get cache rules of a domain
        :param domain: the domain name
        :type domain: string
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/domain/' + domain + '/config',
            params={'cacheTTL': ''},
            config=config)

    @required(domain=str, rules=list)
    def set_domain_cache_ttl(self, domain, rules, config=None):
        """
        set cache rules of a domain
        :param domain: the domain name
        :type domain: string
        :param rules: cache rules
        :type rules: list
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'cacheTTL': ''},
            body=json.dumps({'cacheTTL': rules}),
            config=config)

    def set_domain_cache_full_url(self, domain, flag, config=None):
        """
        set if use the full url as cache key
        :param domain: the domain name
        :type domain: string
        :param flag: 
        :type flag: bool
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'cacheFullUrl': ''},
            body=json.dumps({'cacheFullUrl': flag}),
            config=config)

    @required(domain=str)
    def set_domain_referer_acl(self, domain,
                            blackList=None, whiteList=None,
                            allowEmpty=True, config=None):
        """
        set request referer access control
        :param domain: the domain name
        :type domain: string
        :param blackList: referer blackList
        :type blackList: list
        :param whitelist: referer whitelist
        :type whitelist: list
        :param allowempty: allow empty referer?
        :type allowempty: bool
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        acl = {}
        acl['allowEmpty'] = allowEmpty

        if blackList is not None:
            acl['blackList'] = blackList
        if whiteList is not None:
            acl['whiteList'] = whiteList

        return self._send_request(
            http_methods.PUT, '/domain/' + domain + '/config',
            params={'refererACL': ''},
            body=json.dumps({'refererACL': acl}),
            config=config)

    @required(domain=str)
    def set_domain_ip_acl(self, domain, blackList=None, whiteList=None, config=None):
        """
        set request ip access control
        :param domain: the domain name
        :type domain: string
        :param blackList: ip blackList
        :type blackList: list
        :param whitelist: ip whitelist
        :type whitelist: list
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        acl = {}

        if blackList is not None:
            acl['blackList'] = blackList
        if whiteList is not None:
            acl['whiteList'] = whiteList

        return self._send_request(
            http_methods.PUT, '/domain/' + domain + '/config',
            params={'ipACL': ''},
            body=json.dumps({'ipACL': acl}),
            config=config)

    @required(domain=str, https=dict)
    def set_domain_https(self, domain, https, config=None):
        """
        set request ip access control
        :param domain: the domain name
        :type domain: string
        :param https: https config
        :type https: dict
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """

        return self._send_request(
            http_methods.PUT, '/domain/' + domain + '/config',
            params={'https': ''},
            body=json.dumps({'https': https}),
            config=config)

    @required(domain=str, limitRate=int)
    def set_domain_limit_rate(self, domain, limitRate, config=None):
        """
        set limit rate
        :param domain: the domain name
        :type domain: string
        :param limitRate: limit rate value (Byte/s)
        :type limitRate: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT, '/domain/' + domain + '/config',
            params={'limitRate': ''},
            body=json.dumps({'limitRate': limitRate}),
            config=config)

    @required(domain=str, request_auth=dict)
    def set_domain_request_auth(self, domain, requestAuth, config=None):
        """
        set request auth
        :param domain: the domain
        :type domain: string
        :param requestAuth: request auth config
        :type requestAuth: dict
        :param config: None
        :type config: baidubce.BceClientConfiguration
        :return:
        :rtype: baidubce.bce_response.BceResponse

        """
        return self._send_request(
            http_methods.PUT, '/domain/' + domain + '/config',
            params={'requestAuth': ''},
            body=json.dumps({'requestAuth': requestAuth}),
            config=config)

    @required(param=CdnStatsParam)
    def get_domain_stats(self, param, config=None):
        """
        query stats of the domain or uid or tagId, eg : flow pv
        :param param: the stats query param
        :type param: cdn_stats_param.CdnStatsParam
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.POST,
            '/stat/query',
            body=json.dumps(param.__dict__),
            config=config)

    def get_domain_pv_stat(self, domain=None,
                        startTime=None, endTime=None,
                        period=300, withRegion=None, config=None):
        """
        query pv and qps of the domain
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param withRegion: if need client region distribution
        :type withRegion: any
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period
        params['withRegion'] = withRegion
        return self._send_request(
            http_methods.GET, '/stat/pv',
            params=params,
            config=config)

    def get_domain_flow_stat(self, domain=None,
                        startTime=None, endTime=None,
                        period=300, withRegion=None, config=None):
        """
        query bandwidth of the domain
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param withRegion: if need client region distribution
        :type withRegion: any
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period
        params['withRegion'] = withRegion

        return self._send_request(
            http_methods.GET, '/stat/flow',
            params=params,
            config=config)

    def get_domain_src_flow_stat(self, domain=None,
                        startTime=None, endTime=None,
                        period=300, config=None):
        """
        query origin bandwidth of the domain
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period

        return self._send_request(
            http_methods.GET, '/stat/srcflow',
            params=params,
            config=config)

    def get_domain_hitrate_stat(self, domain=None,
                        startTime=None, endTime=None,
                        period=300, config=None):
        """
        query hit rate of the domain
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period
        return self._send_request(
            http_methods.GET, '/stat/hitrate',
            params=params,
            config=config)

    def get_domain_httpcode_stat(self, domain=None,
                                startTime=None, endTime=None,
                                period=300, withRegion=None, config=None):
        """
        query http response code of a domain or all domains of the user
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param withRegion: if need client region distribution
        :type withRegion: any
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period
        params['withRegion'] = withRegion
        return self._send_request(
            http_methods.GET, '/stat/httpcode',
            params=params,
            config=config)

    def get_domain_topn_url_stat(self, domain=None,
                                startTime=None, endTime=None,
                                period=300, config=None):
        """
        query top n url of the domain or all domains of the user
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period

        return self._send_request(
            http_methods.GET, '/stat/topn/url',
            params=params,
            config=config)

    def get_domain_topn_referer_stat(self, domain=None,
                                    startTime=None, endTime=None,
                                    period=300, config=None):
        """
        query top n referer of the domain or all domains of the user
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period

        return self._send_request(
            http_methods.GET, '/stat/topn/referer',
            params=params,
            config=config)

    def get_domain_uv_stat(self, domain=None,
                        startTime=None, endTime=None,
                        period=3600, config=None):
        """
        query the total number of client of a domain or all domains of the user
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period

        return self._send_request(
            http_methods.GET, '/stat/uv',
            params=params,
            config=config)

    def get_domain_avg_speed_stat(self, domain=None,
                                startTime=None, endTime=None,
                                period=300, config=None):
        """
        query average of the domain or all domains of the user
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param period: time interval of query result
        :type period: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['domain'] = domain
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['period'] = period

        return self._send_request(
            http_methods.GET, '/stat/avgspeed',
            params=params,
            config=config)

    @required(tasks=list)
    def purge(self, tasks, config=None):
        """
        purge the cache of specified url or directory
        :param tasks: url or directory list to purge
        :type tasks: list
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        body = {}
        body['tasks'] = tasks
        return self._send_request(
            http_methods.POST, '/cache/purge',
            config=config, body=json.dumps(body))

    def list_purge_tasks(self, id=None, url=None,
                        startTime=None, endTime=None,
                        marker=None, config=None):
        """
        query the status of purge tasks
        :param id: purge task id to query
        :type id: string
        :param url: purge url to query
        :type url: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param marker: 'nextMarker' get from last query
        :type marker: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['id'] = id
        params['url'] = url
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['marker'] = marker

        return self._send_request(
            http_methods.GET, '/cache/purge',
            params=params,
            config=config)

    @required(tasks=list)
    def prefetch(self, tasks, config=None):
        """
        prefetch the source of specified url from origin
        :param tasks: url or directory list need prefetch
        :type tasks: list
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        body = {}
        body['tasks'] = tasks
        return self._send_request(
            http_methods.POST,
            '/cache/prefetch',
            config=config, body=json.dumps(body))

    def list_prefetch_tasks(self, id=None, url=None,
                            startTime=None, endTime=None,
                            marker=None, config=None):
        """
        query the status of prefetch tasks
        :param id: prefetch task id to query
        :type id: string
        :param url: prefetch url to query
        :type url: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param marker: 'nextMarker' get from last query
        :type marker: int
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        params['id'] = id
        params['url'] = url
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        params['marker'] = marker

        return self._send_request(
            http_methods.GET, '/cache/prefetch',
            params=params,
            config=config)

    def list_quota(self, config=None):
        """
        query purge quota of the user
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(http_methods.GET,
                                '/cache/quota',
                                config=config)

    def get_domain_log(self, domain, startTime, endTime, config=None):
        """
        get log of the domain in specified period of time
        :param domain: the domain name
        :type domain: string
        :param startTime: query start time
        :type startTime: Timestamp
        :param endTime: query end time
        :type endTime: Timestamp
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        params = {}
        if startTime is not None:
            params['startTime'] = startTime

        if endTime is not None:
            params['endTime'] = endTime

        return self._send_request(
            http_methods.GET,
            '/log/' + domain + '/log',
            params=params,
            config=config)

    def ip_query(self, action, ip, config=None):
        """
        check specified ip whether belongs to Baidu CDN
        :param action: 'describeIp'
        :type action: string
        :param ip: specified ip
        :type ip: string
        """
        params = {}
        params['action'] = action
        params['ip'] = ip
        if params is None:
            params = {}
        return self._send_request(
            http_methods.GET, '/utils',
            params=params,
            config=config)

    def list_nodes(self, config=None):
        """
        Baidu back source IP address segment query

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET, '/nodes/list',
            params={},
            config=config)

    @required(domain=str)
    def set_seo(self, domain, push_record=False, directory_origin=False, config=None):
        """
        set seo
        :param domain: the domain name
        :type domain: string
        :param push_record: push record to baidu or not
        :type param: boolean
        :param directory_origin: directory access origin or not
        :param config: None
        :type config: baidubce.BceClientConfiguration

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        body = dict()
        body['pushRecord'] = "ON" if push_record else "OFF"
        body['diretlyOrigin'] = "ON" if directory_origin else "OFF"

        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'seoSwitch': ''},
            body=json.dumps({'seoSwitch': body}),
            config=config)

    @required(domain=str)
    def get_seo(self, domain, config=None):
        """
        get seo configuration.
        :param domain: the domain name
        :type domain: string

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.GET,
            '/domain/' + domain + '/config',
            params={'seoSwitch': ''},
            config=config)

    @required(domain=str)
    def set_follow_protocol(self, domain, follow, config=None):
        """
        set follow protocol.
        :param domain: the domain name
        :type domain: string
        :param follow: follow protocol or not
        :type follow: boolean

        :return:
        :rtype: baidubce.bce_response.BceResponse
        """
        return self._send_request(
            http_methods.PUT,
            '/domain/' + domain + '/config',
            params={'followProtocol': ''},
            body=json.dumps({'followProtocol': follow}),
            config=config)

    @staticmethod
    def _merge_config(self, config):
        if config is None:
            return self.config
        else:
            new_config = copy.copy(self.config)
            new_config.merge_non_none_values(config)
            return new_config

    def _send_request(
            self, http_method, path,
            body=None, headers=None, params=None,
            config=None,
            body_parser=None):
        config = self._merge_config(self, config)
        if body_parser is None:
            body_parser = handler.parse_json

        return bce_http_client.send_request(
            config, bce_v1_signer.sign, [handler.parse_error, body_parser],
            http_method, utils.append_uri(CdnClient.prefix, path), body, headers, params)
