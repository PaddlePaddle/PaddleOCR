# Copyright 2019 Baidu, Inc.
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
This module provides a client class for MVS.
"""

# from baidubce.services.vca.vca_client import VcaClient
#
# from baidubce.services.vcr.vcr_client import VcrClient

import copy
import json
import logging
from builtins import str
from builtins import bytes

from baidubce.auth import bce_v1_signer
from baidubce.bce_base_client import BceBaseClient
from baidubce.http import bce_http_client
from baidubce.http import handler
from baidubce.http import http_methods
from baidubce.utils import required

_logger = logging.getLogger(__name__)


class MvsClient(BceBaseClient):
    """
    mvs client
    """
    def __init__(self, config=None):
        """init"""
        BceBaseClient.__init__(self, config)

    @required(source=(bytes, str))
    def insert_video(self, video_lib, source, notification, description=None, config=None):
        """
        insert a video.
        :param video_lib: video lib
        :type video_lib: string
        :param source: video source
        :type source: string
        :param notification: notification name
        :type notification: string
        :param description: description for this request
        :type description: string
        :return: **BceResponse**
        """
        body = {
            'source': source,
            'notification': notification
        }
        if description is not None:
            body['description'] = description
        return self._send_request(http_methods.PUT, str.encode('/v2/videolib/' + video_lib),
                                  headers={b'Content-Type': b'application/json'},
                                  body=json.dumps(body),
                                  config=config)

    @required(source=(bytes, str))
    def search_video_by_video(self, video_lib, source, notification, description=None, config=None):
        """
        search video by video.
        :param video_lib: video lib
        :type video_lib: string
        :param source: video source
        :type source: string
        :param notification: notification name
        :type notification: string
        :param description: description for this request
        :type description: string
        :return: **BceResponse**
        """
        body = {
            'source': source,
            'notification': notification
        }
        if description is not None:
            body['description'] = description
        return self._send_request(http_methods.POST, str.encode('/v2/videolib/' + video_lib),
                                  headers={b'Content-Type': b'application/json'},
                                  params={b'searchByVideo': b''},
                                  body=json.dumps(body),
                                  config=config)

    @required(source=(bytes, str))
    def insert_image(self, image_lib, source, description=None, config=None):
        """
        insert an image.
        :param image_lib: image lib
        :type image_lib: string
        :param source: video source
        :type source: string
        :param description: description for this request
        :type description: string
        :return: **BceResponse**
        """
        body = {
            'source': source
        }
        if description is not None:
            body['description'] = description
        return self._send_request(http_methods.PUT, str.encode('/v2/imagelib/' + image_lib),
                                  headers={b'Content-Type': b'application/json'},
                                  body=json.dumps(body),
                                  config=config)

    @required(source=(bytes, str))
    def search_image_by_image(self, image_lib, source, description=None, config=None):
        """
        search video by video.
        :param video_lib: video lib
        :type video_lib: string
        :param source: video source
        :type source: string
        :param description: description for this request
        :type description: string
        :return: **BceResponse**
        """
        body = {
            'source': source
        }
        if description is not None:
            body['description'] = description
        return self._send_request(http_methods.POST, str.encode('/v2/imagelib/' + image_lib),
                                  headers={b'Content-Type': b'application/json'},
                                  params={b'searchByImage': b''},
                                  body=json.dumps(body),
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
