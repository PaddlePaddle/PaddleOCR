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
This module provides general http handler functions for processing http responses from bos services.
"""

import json
from baidubce import utils
from baidubce.exception import BceServerError
from baidubce.http import handler
from builtins import str
from builtins import bytes

def parse_copy_object_response(http_response, response):
    """
    response parser for copy object
    """
    TRANSFER_ENCODING = b'transfer-encoding'
    headers_list = {k: v for k, v in http_response.getheaders()}
    if headers_list.get(TRANSFER_ENCODING, b'not exist') == b'chunked':
        body = http_response.read()
        if body:
            d = json.loads(body)
            if b'code' in d:
                http_response.close()
                raise BceServerError(d[b'message'], code=d[b'code'], request_id=d[b'requestId'])
            else:
                response.__dict__.update(
                    json.loads(body, object_hook=utils.dict_to_python_object).__dict__)
                http_response.close()
        else:
            e = BceServerError(http_response.reason, request_id=response.metadata.bce_request_id)
            http_response.close()
            raise e
        return True
    else:
        return handler.parse_json(http_response, response)
