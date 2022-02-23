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
This module provide some tools for bce client.
"""
# str() generator unicode,bytes() for ASCII
from __future__ import absolute_import
from builtins import str, bytes
from future.utils import iteritems, iterkeys, itervalues
from baidubce import compat

import os
import re
import datetime
import hashlib
import base64
import string
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from Crypto.Cipher import AES
import baidubce
from baidubce.http import http_headers

import codecs

DEFAULT_CNAME_LIKE_LIST = [b".cdn.bcebos.com"]

def get_md5_from_fp(fp, offset=0, length=-1, buf_size=8192):
    """
    Get MD5 from file by fp.

    :type fp: FileIO
    :param fp: None

    :type offset: long
    :param offset: None

    :type length: long
    :param length: None
    =======================
    :return:
        **file_size, MD(encode by base64)**
    """

    origin_offset = fp.tell()
    if offset:
        fp.seek(offset)
    md5 = hashlib.md5()
    while True:
        bytes_to_read = buf_size
        if bytes_to_read > length > 0:
            bytes_to_read = length
        buf = fp.read(bytes_to_read)
        if not buf:
            break
        md5.update(buf)
        if length > 0:
            length -= len(buf)
        if length == 0:
            break
    fp.seek(origin_offset)
    return base64.standard_b64encode(md5.digest())


def get_canonical_time(timestamp=0):
    """
    Get cannonical time.

    :type timestamp: int
    :param timestamp: None
    =======================
    :return:
        **string of canonical_time**
    """
    if timestamp == 0:
        utctime = datetime.datetime.utcnow()
    else:
        utctime = datetime.datetime.utcfromtimestamp(timestamp)
    return b"%04d-%02d-%02dT%02d:%02d:%02dZ" % (
        utctime.year, utctime.month, utctime.day,
        utctime.hour, utctime.minute, utctime.second)


def is_ip(s):
    """
    Check a string whether is a legal ip address.

    :type s: string
    :param s: None
    =======================
    :return:
        **Boolean**
    """
    try:
        tmp_list = s.split(b':')
        s = tmp_list[0]
        if s == b'localhost':
            return True
        tmp_list = s.split(b'.')
        if len(tmp_list) != 4:
            return False
        else:
            for i in tmp_list:
                if int(i) < 0 or int(i) > 255:
                    return False
    except:
        return False
    return True


def convert_to_standard_string(input_string):
    """
    Encode a string to utf-8.

    :type input_string: string
    :param input_string: None
    =======================
    :return:
        **string**
    """
    #if isinstance(input_string, str):
    #    return input_string.encode(baidubce.DEFAULT_ENCODING)
    #elif isinstance(input_string, bytes):
    #    return input_string
    #else:
    #    return str(input_string).encode("utf-8")
    return compat.convert_to_bytes(input_string)

def convert_header2map(header_list):
    """
    Transfer a header list to dict

    :type s: list
    :param s: None
    =======================
    :return:
        **dict**
    """
    header_map = {}
    for a, b in header_list:
        if isinstance(a, bytes):
            a = a.strip(b'\"')
        if isinstance(b, bytes):
            b = b.strip(b'\"')
        header_map[a] = b
    return header_map


def safe_get_element(name, container):
    """
    Get element from dict which the lower of key and name are equal.

    :type name: string
    :param name: None

    :type container: dict
    :param container: None
    =======================
    :return:
        **Value**
    """
    for k, v in iteritems(container):
        if k.strip().lower() == name.strip().lower():
            return v
    return ""


def check_redirect(res):
    """
    Check whether the response is redirect.

    :type res: HttpResponse
    :param res: None

    :return:
        **Boolean**
    """
    is_redirect = False
    try:
        if res.status == 301 or res.status == 302:
            is_redirect = True
    except:
        pass
    return is_redirect


def _get_normalized_char_list():
    """"
    :return:
        **ASCII string**
    """
    ret = ['%%%02X' % i for i in range(256)]
    for ch in string.ascii_letters + string.digits + '.~-_':
        ret[ord(ch)] = ch
    if isinstance(ret[0], str):
        ret = [s.encode("utf-8") for s in ret]
    return ret
_NORMALIZED_CHAR_LIST = _get_normalized_char_list()


def normalize_string(in_str, encoding_slash=True):
    """
    Encode in_str.
    When encoding_slash is True, don't encode skip_chars, vice versa.

    :type in_str: string
    :param in_str: None

    :type encoding_slash: Bool
    :param encoding_slash: None
    ===============================
    :return:
        **ASCII  string**
    """
    tmp = []
    for ch in convert_to_standard_string(in_str):
        # on python3, ch is int type
        sep = ''
        index = -1
        if isinstance(ch, int):
            # on py3
            sep = chr(ch).encode("utf-8")
            index = ch
        else:
            sep = ch
            index = ord(ch)
        if sep == b'/' and not encoding_slash:
            tmp.append(b'/')
        else:
            tmp.append(_NORMALIZED_CHAR_LIST[index])
    return (b'').join(tmp)


def append_uri(base_uri, *path_components):
    """
    Append path_components to the end of base_uri in order, and ignore all empty strings and None

    :param base_uri: None
    :type base_uri: string

    :param path_components: None

    :return: the final url
    :rtype: str
    """
    tmp = [base_uri]
    for path in path_components:
        if path:
            tmp.append(normalize_string(path, False))
    if len(tmp) > 1:
        tmp[0] = tmp[0].rstrip(b'/')
        tmp[-1] = tmp[-1].lstrip(b'/')
        for i in range(1, len(tmp) - 1):
            tmp[i] = tmp[i].strip(b'/')
    return (b'/').join(tmp)


def check_bucket_valid(bucket):
    """
    Check bucket name whether is legal.

    :type bucket: string
    :param bucket: None
    =======================
    :return:
        **Boolean**
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-"
    if len(bucket) < 3 or len(bucket) > 63:
        return False
    if bucket[-1] == "-" or bucket[-1] == "_":
        return False
    if not (('a' <= bucket[0] <= 'z') or ('0' <= bucket[0] <= '9')):
        return False
    for i in bucket:
        if not i in alphabet:
            return False
    return True


def guess_content_type_by_file_name(file_name):
    """
    Get file type by filename.

    :type file_name: string
    :param file_name: None
    =======================
    :return:
        **Type Value**
    """
    mime_map = dict()
    mime_map["js"] = "application/javascript"
    mime_map["xlsx"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    mime_map["xltx"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.template"
    mime_map["potx"] = "application/vnd.openxmlformats-officedocument.presentationml.template"
    mime_map["ppsx"] = "application/vnd.openxmlformats-officedocument.presentationml.slideshow"
    mime_map["pptx"] = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    mime_map["sldx"] = "application/vnd.openxmlformats-officedocument.presentationml.slide"
    mime_map["docx"] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    mime_map["dotx"] = "application/vnd.openxmlformats-officedocument.wordprocessingml.template"
    mime_map["xlam"] = "application/vnd.ms-excel.addin.macroEnabled.12"
    mime_map["xlsb"] = "application/vnd.ms-excel.sheet.binary.macroEnabled.12"
    try:
        file_name = compat.convert_to_string(file_name)
        name = os.path.basename(file_name.lower())
        suffix = name.split('.')[-1]
        if suffix in iterkeys(mime_map):
            mime_type = mime_map[suffix]
        else:
            import mimetypes

            mimetypes.init()
            mime_type = mimetypes.types_map.get("." + suffix, 'application/octet-stream')
    except:
        mime_type = 'application/octet-stream'
    if not mime_type:
        mime_type = 'application/octet-stream'

    return compat.convert_to_bytes(mime_type)


_first_cap_regex = re.compile('(.)([A-Z][a-z]+)')
_number_cap_regex = re.compile('([a-z])([0-9]{2,})')
_end_cap_regex = re.compile('([a-z0-9])([A-Z])')


def pythonize_name(name):
    """Convert camel case to a "pythonic" name.
    Examples::
        pythonize_name('CamelCase') -> 'camel_case'
        pythonize_name('already_pythonized') -> 'already_pythonized'
        pythonize_name('HTTPRequest') -> 'http_request'
        pythonize_name('HTTPStatus200Ok') -> 'http_status_200_ok'
        pythonize_name('UPPER') -> 'upper'
        pythonize_name('ContentMd5')->'content_md5'
        pythonize_name('') -> ''
    """
    if name == "eTag":
        return "etag"
    s1 = _first_cap_regex.sub(r'\1_\2', name)
    s2 = _number_cap_regex.sub(r'\1_\2', s1)
    return _end_cap_regex.sub(r'\1_\2', s2).lower()


def get_canonical_querystring(params, for_signature):
    """

    :param params:
    :param for_signature:
    :return:
    """
    if params is None:
        return ''
    result = []
    for k, v in iteritems(params):
        if not for_signature or k.lower != http_headers.AUTHORIZATION.lower():
            if v is None:
                v = ''
            result.append(b'%s=%s' % (normalize_string(k), normalize_string(v)))
    result.sort()
    return (b'&').join(result)


def print_object(obj):
    """

    :param obj:
    :return:
    """
    tmp = []
    for k, v in iteritems(obj.__dict__):
        if not k.startswith('__') and k != "raw_data":
            if isinstance(v, bytes):
                tmp.append("%s:'%s'" % (k, v))
            # str is unicode
            elif isinstance(v, str):
                tmp.append("%s:u'%s'" % (k, v))
            else:
                tmp.append('%s:%s' % (k, v))
    return '{%s}' % ','.join(tmp)

class Expando(object):
    """
    Expandable class
    """
    def __init__(self, attr_dict=None):
        if attr_dict:
            self.__dict__.update(attr_dict)

    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError
        return None

    def __repr__(self):
        return print_object(self)


def dict_to_python_object(d):
    """

    :param d:
    :return:
    """
    attr = {}
    for k, v in iteritems(d):
        if not isinstance(k, compat.string_types):
            k = compat.convert_to_string(k)
        k = pythonize_name(k)
        attr[k] = v
    return Expando(attr)


def required(**types):
    """
    decorator of input param check
    :param types:
    :return:
    """
    def _required(f):
        def _decorated(*args, **kwds):
            for i, v in enumerate(args):
                if f.__code__.co_varnames[i] in types:
                    if v is None:
                        raise ValueError('arg "%s" should not be None' %
                                         (f.__code__.co_varnames[i]))
                    if not isinstance(v, types[f.__code__.co_varnames[i]]):
                        raise TypeError('arg "%s"= %r does not match %s' %
                                        (f.__code__.co_varnames[i],
                                         v,
                                         types[f.__code__.co_varnames[i]]))
            for k, v in iteritems(kwds):
                if k in types:
                    if v is None:
                        raise ValueError('arg "%s" should not be None' % k)
                    if not isinstance(v, types[k]):
                        raise TypeError('arg "%s"= %r does not match %s' % (k, v, types[k]))
            return f(*args, **kwds)
        _decorated.__name__ = f.__name__
        return _decorated
    return _required


def parse_host_port(endpoint, default_protocol):
    """
    parse protocol, host, port from endpoint in config

    :type: string
    :param endpoint: endpoint in config

    :type: baidubce.protocol.HTTP or baidubce.protocol.HTTPS
    :param default_protocol: if there is no scheme in endpoint,
                              we will use this protocol as default
    :return: tuple of protocol, host, port
    """
    # netloc should begin with // according to RFC1808
    if b"//" not in endpoint:
        endpoint = b"//" + endpoint

    try:
        # scheme in endpoint dominates input default_protocol
        parse_result = urlparse(
                endpoint,
                compat.convert_to_bytes(default_protocol.name))
    except Exception as e:
        raise ValueError('Invalid endpoint:%s, error:%s' % (endpoint,
            compat.convert_to_string(e)))

    if parse_result.scheme == compat.convert_to_bytes(baidubce.protocol.HTTP.name):
        protocol = baidubce.protocol.HTTP
        port = baidubce.protocol.HTTP.default_port
    elif parse_result.scheme == compat.convert_to_bytes(baidubce.protocol.HTTPS.name):
        protocol = baidubce.protocol.HTTPS
        port = baidubce.protocol.HTTPS.default_port
    else:
        raise ValueError('Unsupported protocol %s' % parse_result.scheme)
    host = parse_result.hostname
    if parse_result.port is not None:
        port = parse_result.port

    return protocol, host, port

"""
def aes128_encrypt_16char_key(adminpass, secretkey):
    
    #Python2:encrypt admin password by AES128
    
    pad_it = lambda s: s + (16 - len(s) % 16) * chr(16 - len(s) % 16)
    key = secretkey[0:16]
    mode = AES.MODE_ECB
    cryptor = AES.new(key, mode, key)
    cipheradminpass = cryptor.encrypt(pad_it(adminpass)).encode('hex')
    return cipheradminpass
"""


def aes128_encrypt_16char_key(adminpass, secretkey):
    """

    :param adminpass: adminpass
    :param secretkey: secretkey
    :return: cipheradminpass
    """

    # Python3: encrypt admin password by AES128

    pad_it = lambda s: s + (16 - len(s) % 16) * chr(16 - len(s) % 16)
    key = secretkey[0:16]
    mode = AES.MODE_ECB
    cryptor = AES.new(key, mode)
    pad_admin = pad_it(adminpass)
    byte_pad_admin = pad_admin.encode(encoding='utf-8')

    cryptoradminpass = cryptor.encrypt(byte_pad_admin)
    #print(cryptoradminpass)

    #cipheradminpass = cryptor.encrypt(byte_pad_admin).encode('hex')
    byte_cipheradminpass = codecs.encode(cryptoradminpass, 'hex_codec')
    #print(byte_cipheradminpass)

    cipheradminpass = byte_cipheradminpass.decode(encoding='utf-8')
    #print(cipheradminpass)

    return cipheradminpass

def is_cname_like_host(host):
    """
    :param host: custom domain
    :return: domain end with cdn endpoint or not
    """
    if host is None:
        return False
    for suffix in DEFAULT_CNAME_LIKE_LIST:
        if host.lower().endswith(suffix):
            return True
    return False
