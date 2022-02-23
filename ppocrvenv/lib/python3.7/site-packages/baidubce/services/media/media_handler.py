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
This module provides general http handler functions for processing http responses from media services.
"""

import json
from baidubce import utils


def dict_to_python_object(d):
    """
    dict to python object without converting camel key to a "pythonic" name

    :param d:
    :return:
    """
    attr = {}
    for k, v in utils.iteritems(d):
        if not isinstance(k, utils.compat.string_types):
            k = utils.compat.convert_to_string(k)
        attr[k] = v
    return utils.Expando(attr)


def parse_json(http_response, response):
    """If the body is not empty, convert it to a python object and set as the value of
    response.body. http_response is always closed if no error occurs.

    :param http_response: the http_response object returned by HTTPConnection.getresponse()
    :type http_response: httplib.HTTPResponse

    :param response: general response object which will be returned to the caller
    :type response: baidubce.BceResponse

    :return: always true
    :rtype bool
    """
    body = http_response.read()
    if body:
        body = utils.compat.convert_to_string(body)
        response.__dict__.update(json.loads(body, object_hook=dict_to_python_object).__dict__)
    http_response.close()
    return True
