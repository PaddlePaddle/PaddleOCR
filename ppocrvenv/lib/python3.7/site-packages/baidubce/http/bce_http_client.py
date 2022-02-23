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
This module provide http request function for bce services.
"""
from future.utils import iteritems, iterkeys, itervalues
from builtins import str, bytes
import logging
import http.client
import sys
import time
import traceback

import baidubce
from baidubce import compat
from baidubce import utils
from baidubce.bce_response import BceResponse
from baidubce.exception import BceHttpClientError
from baidubce.exception import BceClientError
from baidubce.http import http_headers

_logger = logging.getLogger(__name__)


def _get_connection(protocol, host, port, connection_timeout_in_millis):
    """
    :param protocol
    :type protocol: baidubce.protocol.Protocol
    :param endpoint
    :type endpoint: str
    :param connection_timeout_in_millis
    :type connection_timeout_in_millis int
    """
    host = compat.convert_to_string(host)
    if protocol.name == baidubce.protocol.HTTP.name:
        return http.client.HTTPConnection(
            host=host, port=port, timeout=connection_timeout_in_millis / 1000)
    elif protocol.name == baidubce.protocol.HTTPS.name:
        return http.client.HTTPSConnection(
            host=host, port=port, timeout=connection_timeout_in_millis / 1000)
    else:
        raise ValueError(
            'Invalid protocol: %s, either HTTP or HTTPS is expected.' % protocol)


def _send_http_request(conn, http_method, uri, headers, body, send_buf_size):
    # putrequest() need that http_method and uri is Ascii on Py2 and unicode \
    # on Py3
    http_method = compat.convert_to_string(http_method)
    uri = compat.convert_to_string(uri)
    conn.putrequest(http_method, uri, skip_host=True, skip_accept_encoding=True)

    for k, v in iteritems(headers):
        k = utils.convert_to_standard_string(k)
        v = utils.convert_to_standard_string(v)
        conn.putheader(k, v)
    conn.endheaders()

    if body:
        if isinstance(body, (bytes, str)):
            conn.send(body)
        else:
            total = int(headers[http_headers.CONTENT_LENGTH])
            sent = 0
            while sent < total:
                size = total - sent
                if size > send_buf_size:
                    size = send_buf_size
                buf = body.read(size)
                if not buf:
                    raise BceClientError(
                        'Insufficient data, only %d bytes available while %s is %d' % (
                            sent, http_headers.CONTENT_LENGTH, total))
                conn.send(buf)
                sent += len(buf)

    return conn.getresponse()


def check_headers(headers):
    """
    check value in headers, if \n in value, raise
    :param headers:
    :return:
    """
    for k, v in iteritems(headers):
        if isinstance(v, (bytes, str)) and \
        b'\n' in compat.convert_to_bytes(v):
            raise BceClientError(r'There should not be any "\n" in header[%s]:%s' % (k, v))


def send_request(
        config,
        sign_function,
        response_handler_functions,
        http_method, path, body, headers, params, use_backup_endpoint=False):
    """
    Send request to BCE services.

    :param config
    :type config: baidubce.BceClientConfiguration

    :param sign_function:

    :param response_handler_functions:
    :type response_handler_functions: list

    :param request:
    :type request: baidubce.internal.InternalRequest

    :return:
    :rtype: baidubce.BceResponse
    """
    _logger.debug(b'%s request start: %s %s, %s, %s',
                  http_method, path, headers, params, body)
    headers = headers or {}

    user_agent = 'bce-sdk-python/%s/%s/%s' % (
        compat.convert_to_string(baidubce.SDK_VERSION), sys.version, sys.platform)
    user_agent = user_agent.replace('\n', '')
    user_agent = compat.convert_to_bytes(user_agent)
    headers[http_headers.USER_AGENT] = user_agent
 
    should_get_new_date = False
    if http_headers.BCE_DATE not in headers:
        should_get_new_date = True

    request_endpoint = config.endpoint
    if use_backup_endpoint:
        request_endpoint = config.backup_endpoint

    headers[http_headers.HOST] = request_endpoint

    if isinstance(body, str):
        body = body.encode(baidubce.DEFAULT_ENCODING)
    if not body:
        headers[http_headers.CONTENT_LENGTH] = 0
    elif isinstance(body, bytes):
        headers[http_headers.CONTENT_LENGTH] = len(body)
    elif http_headers.CONTENT_LENGTH not in headers:
        raise ValueError(b'No %s is specified.' % http_headers.CONTENT_LENGTH)
    # store the offset of fp body
    offset = None
    if hasattr(body, "tell") and hasattr(body, "seek"):
        offset = body.tell()

    protocol, host, port = utils.parse_host_port(request_endpoint, config.protocol)

    headers[http_headers.HOST] = host
    if port != config.protocol.default_port:
        headers[http_headers.HOST] += b':' + compat.convert_to_bytes(port)

    headers[http_headers.AUTHORIZATION] = sign_function(
        config.credentials, http_method, path, headers, params)

    encoded_params = utils.get_canonical_querystring(params, False)
    if len(encoded_params) > 0:
        uri = path + b'?' + encoded_params
    else:
        uri = path
    check_headers(headers)

    retries_attempted = 0
    errors = []
    while True:
        conn = None
        try:
            # restore the offset of fp body when retrying
            if should_get_new_date is True:
                headers[http_headers.BCE_DATE] = utils.get_canonical_time()

            headers[http_headers.AUTHORIZATION] = sign_function(
                config.credentials, http_method, path, headers, params)

            if retries_attempted > 0 and offset is not None:
                body.seek(offset)
            
            conn = _get_connection(protocol, host, port, config.connection_timeout_in_mills)

            _logger.debug('request args:method=%s, uri=%s, headers=%s,patams=%s, body=%s',
                    http_method, uri, headers, params, body)

            http_response = _send_http_request(
                conn, http_method, uri, headers, body, config.send_buf_size)
            
            headers_list = http_response.getheaders()

            # on py3 ,values of headers_list is decoded with ios-8859-1 from
            # utf-8 binary bytes

            # headers_list[*][0] is lowercase on py2
            # headers_list[*][0] is raw value py3
            if compat.PY3 and isinstance(headers_list, list):
                temp_heads = []
                for k, v in headers_list:
                    k = k.encode('latin-1').decode('utf-8')
                    v = v.encode('latin-1').decode('utf-8')
                    k = k.lower()
                    temp_heads.append((k, v))
                headers_list = temp_heads

            _logger.debug(
                'request return: status=%d, headers=%s' % (http_response.status, headers_list))
            response = BceResponse()
            response.set_metadata_from_headers(dict(headers_list))

            for handler_function in response_handler_functions:
                if handler_function(http_response, response):
                    break
            return response
        except Exception as e:
            if conn is not None:
                conn.close()

            # insert ">>>>" before all trace back lines and then save it
            errors.append('\n'.join('>>>>' + line for line in traceback.format_exc().splitlines()))

            if config.retry_policy.should_retry(e, retries_attempted):
                delay_in_millis = config.retry_policy.get_delay_before_next_retry_in_millis(
                    e, retries_attempted)
                time.sleep(delay_in_millis / 1000.0)
            else:
                raise BceHttpClientError('Unable to execute HTTP request. Retried %d times. '
                                         'All trace backs:\n%s' % (retries_attempted,
                                                                   '\n'.join(errors)), e)

        retries_attempted += 1
