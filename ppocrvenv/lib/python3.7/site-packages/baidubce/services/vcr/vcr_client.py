# Copyright 2017-2019 Baidu, Inc.
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
This module provides a client class for VCR.
"""

import copy
import json
import logging
from builtins import str
from builtins import bytes

from baidubce import compat

from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_methods
from baidubce.utils import required

_logger = logging.getLogger(__name__)


class VcrClient(BceBaseClient):
    """
    vcr client
    """

    def __init__(self, config=None):
        BceBaseClient.__init__(self, config)

    @required(source=(bytes, str))
    def put_media(self, source, auth=None, description=None,
                  preset=None, notification=None, config=None):
        """
        Check a media.
        :param source: media source
        :type source: string
        :param auth: media source auth param
        :type auth: string
        :param description: media description
        :type description: string
        :param preset: check preset name
        :type preset: string
        :param notification: notification name
        :type notification: string
        :return: **Http Response**
        """
        body = {
            'source': source
        }
        if auth is not None:
            body['auth'] = auth
        if description is not None:
            body['description'] = description
        if preset is not None:
            body['preset'] = preset
        if notification is not None:
            body['notification'] = notification
        return self._send_request(http_methods.PUT, b'/v1/media',
                                  body=json.dumps(body),
                                  config=config)

    @required(source=(bytes, str))
    def get_media(self, source, config=None):
        """
        :param source: media source
        :type source: string
        :return: **Http Response**
        """
        return self._send_request(http_methods.GET, b'/v1/media',
                                  params={b'source': source},
                                  config=config)

    @required(source=(bytes, str))
    def put_audio(self, source, auth=None, description=None,
                  preset=None, notification=None, config=None):
        """
        Check an audio.
        :param source: audio source
        :type source: string
        :param auth: audio source auth param
        :type auth: string
        :param description: audio description
        :type description: string
        :param preset: check preset name
        :type preset: string
        :param notification: notification name
        :type notification: string
        :return: **Http Response**
        """
        body = {
            'source': source
        }
        if auth is not None:
            body['auth'] = auth
        if description is not None:
            body['description'] = description
        if preset is not None:
            body['preset'] = preset
        if notification is not None:
            body['notification'] = notification
        return self._send_request(http_methods.PUT, b'/v2/audio',
                                  body=json.dumps(body),
                                  config=config)

    @required(source=(bytes, str))
    def get_audio(self, source, config=None):
        """
        :param source: audio source
        :type source: string
        :return: **Http Response**
        """
        return self._send_request(http_methods.GET, b'/v2/audio',
                                  params={b'source': source},
                                  config=config)

    @required(source=(bytes, str))
    def put_image(self, source, preset=None, config=None):
        """
        :param source: image source
        :type source: string
        :param preset: check preset name
        :type preset: string
        :return: **Http Response**
        """
        body = {
            'source': source
        }
        if preset is not None:
            body['preset'] = preset
        return self._send_request(http_methods.PUT, b'/v1/image',
                                  body=json.dumps(body),
                                  config=config)

    @required(source=(bytes, str))
    def put_image_async_check(self, source, preset=None, notification=None, description=None,
                              config=None):
        """
        :param source: image source
        :type source: string
        :param preset: check preset name
        :type preset: string
        :param description: image description
        :type description: string
        :param notification: notification name
        :type notification: string
        :return: **Http Response**
        """
        body = {
            'source': source
        }
        if preset is not None:
            body['preset'] = preset
        if description is not None:
            body['description'] = description
        if notification is not None:
            body['notification'] = notification
        return self._send_request(http_methods.PUT, b'/v2/image',
                                  body=json.dumps(body),
                                  config=config)

    @required(source=(bytes, str))
    def get_image_async_check_result(self, source, preset=None, config=None):
        """
        :param source: image source
        :type source: string
        :param preset: check preset name
        :type preset: string
        :return: **Http Response**
        """
        params = {b'source': source}
        if preset is not None:
            params[b'preset'] = preset
        return self._send_request(http_methods.GET, b'/v2/image',
                                  params=params,
                                  config=config)

    @required(text=(bytes, str))
    def put_text(self, text, preset=None, config=None):
        """
        :param text: text to check
        :type text: string
        :param preset: check preset name
        :type preset: string
        :return: **Http Response**
        """
        body = {
            'text': text
        }
        if preset is not None:
            body['preset'] = preset
        return self._send_request(http_methods.PUT, b'/v1/text',
                                  body=json.dumps(body),
                                  config=config)

    @required(lib=(bytes, str), brief=(bytes, str), image=(bytes, str))
    def add_face_image(self, lib, brief, image, config=None):
        """
        :param lib: private face lib
        :param brief: private face brief
        :param image: private face image url
        :return: **Http Response**
        """
        body = {
            'brief': brief,
            'image': image
        }
        return self._send_request(http_methods.POST,
                                  b'/v1/face/lib/%s' % compat.convert_to_bytes(lib),
                                  body=json.dumps(body),
                                  config=config)

    @required(lib=(bytes, str), brief=(bytes, str))
    def del_face_brief(self, lib, brief, config=None):
        """
        :param lib: private face lib
        :param brief: private face brief
        :return: **Http Response**
        """
        params = {
            b'brief': brief
        }
        return self._send_request(http_methods.DELETE,
                                  b'/v1/face/lib/%s' % compat.convert_to_bytes(lib),
                                  params=params,
                                  config=config)

    @required(lib=(bytes, str), brief=(bytes, str), image=(bytes, str))
    def del_face_image(self, lib, brief, image, config=None):
        """
        :param lib: private face lib
        :param brief: private face brief
        :param image: private face image
        :return: **Http Response**
        """
        params = {
            b'brief': brief,
            b'image': image
        }
        return self._send_request(http_methods.DELETE,
                                  b'/v1/face/lib/%s' % compat.convert_to_bytes(lib),
                                  params=params,
                                  config=config)

    @required(lib=(bytes, str))
    def get_face_lib(self, lib, config=None):
        """
        :param lib: private face lib
        :return: **Http Response**
        """
        return self._send_request(http_methods.GET,
                                  b'/v1/face/lib/%s' % compat.convert_to_bytes(lib),
                                  config=config)

    @required(lib=(bytes, str), brief=(bytes, str))
    def get_face_brief(self, lib, brief, config=None):
        """
        :param lib: private face lib
        :param brief: private face brief
        :return: **Http Response**
        """
        params = {
            b'brief': brief
        }
        return self._send_request(http_methods.GET,
                                  b'/v1/face/lib/%s' % compat.convert_to_bytes(lib),
                                  params=params,
                                  config=config)

    @required(lib=(bytes, str), brief=(bytes, str), image=(bytes, str))
    def add_logo_image(self, lib, brief, image, config=None):
        """
        :param lib: private logo lib
        :param brief: private logo brief
        :param image: private logo image
        :return: **Http Response**
        """
        body = {
            'brief': brief,
            'image': image
        }
        return self._send_request(http_methods.POST,
                                  b'/v1/logo/lib/%s' % compat.convert_to_bytes(lib),
                                  body=json.dumps(body),
                                  config=config)

    @required(lib=(bytes, str), brief=(bytes, str))
    def del_logo_brief(self, lib, brief, config=None):
        """
        :param lib: private logo lib
        :param brief: private logo brief
        :return: **Http Response**
        """
        params = {
            b'brief': brief
        }
        return self._send_request(http_methods.DELETE,
                                  b'/v1/logo/lib/%s' % compat.convert_to_bytes(lib),
                                  params=params,
                                  config=config)

    @required(lib=(bytes, str), image=(bytes, str))
    def del_logo_image(self, lib, image, config=None):
        """
        :param lib: private logo lib
        :param image: private logo image
        :return: **Http Response**
        """
        params = {
            b'image': image
        }
        return self._send_request(http_methods.DELETE,
                                  b'/v1/logo/lib/%s' % compat.convert_to_bytes(lib),
                                  params=params,
                                  config=config)

    @required(lib=(bytes, str))
    def get_logo_lib(self, lib, config=None):
        """
        :param lib: private logo lib
        :return: **Http Response**
        """
        return self._send_request(http_methods.GET,
                                  b'/v1/logo/lib/%s' % compat.convert_to_bytes(lib),
                                  config=config)

    @required(lib=(bytes, str), brief=(bytes, str))
    def get_logo_brief(self, lib, brief, config=None):
        """
        :param lib: private logo lib
        :param brief: private logo brief
        :return: **Http Response**
        """
        params = {
            b'brief': brief
        }
        return self._send_request(http_methods.GET,
                                  b'/v1/logo/lib/%s' % compat.convert_to_bytes(lib),
                                  params=params,
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
            http_method, path, body, headers, params)
