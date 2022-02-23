# Copyright (c) 2017 VisualDL Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================

import sys
import socket
from argparse import ArgumentParser

from visualdl import __version__
from visualdl.server.log import (init_logger, logger)

default_host = None
default_port = 8040
default_cache_timeout = 20
default_public_path = '/app'
default_product = 'normal'

support_themes = ['light', 'dark']


class DefaultArgs(object):
    def __init__(self, args):
        self.logdir = args.get('logdir')
        self.host = args.get('host', default_host)
        self.port = args.get('port', default_port)
        self.cache_timeout = args.get('cache_timeout', default_cache_timeout)
        self.language = args.get('language')
        self.public_path = args.get('public_path')
        self.api_only = args.get('api_only', False)
        self.open_browser = args.get('open_browser', False)
        self.model = args.get('model', '')
        self.product = args.get('product', default_product)
        self.telemetry = args.get('telemetry', True)
        self.theme = args.get('theme', None)
        self.dest = args.get('dest', '')
        self.behavior = args.get('behavior', '')


def get_host(host=default_host, port=default_port):
    if not host:
        host = socket.getfqdn()
        try:
            socket.create_connection((host, port), timeout=1)
        except socket.error:
            host = 'localhost'
    return host


def validate_args(args):
    # if not in API mode, public path cannot be set to root path
    if not args.api_only and args.public_path == '/':
        logger.error('Public path cannot be set to root path.')
        sys.exit(-1)

    # public path must start with `/`
    if args.public_path is not None and not args.public_path.startswith('/'):
        logger.error('Public path should always start with a `/`.')
        sys.exit(-1)

    # theme not support
    if args.theme is not None and args.theme not in support_themes:
        logger.error('Theme {} is not support.'.format(args.theme))
        sys.exit(-1)


def format_args(args):
    # set default public path according to API mode option
    if args.public_path is None:
        args.public_path = '' if args.api_only else default_public_path
    else:
        args.public_path = args.public_path.rstrip('/')

    # don't open browser in API mode
    if args.api_only:
        args.open_browser = False

    # set host to localhost if host is not set
    if not args.host:
        args.host = get_host(args.host, args.port)

    return args


class ParseArgs(object):
    def __init__(self, **kwargs):
        args = DefaultArgs(kwargs)
        validate_args(args)
        args = format_args(args)

        self.logdir = args.logdir
        self.host = args.host
        self.port = args.port
        self.cache_timeout = args.cache_timeout
        self.language = args.language
        self.public_path = args.public_path
        self.api_only = args.api_only
        self.open_browser = args.open_browser
        self.model = args.model
        self.product = args.product
        self.telemetry = args.telemetry
        self.theme = args.theme
        self.dest = args.dest
        self.behavior = args.behavior


def parse_args():
    """
    :return:
    """
    parser = ArgumentParser(
        prog="VisualDL",
        description="VisualDL, a tool to visualize deep learning.",
        epilog="For more information: https://github.com/PaddlePaddle/VisualDL"
    )

    parser.add_argument(
        "--logdir",
        action="store",
        nargs="+",
        help="log file directory")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=default_port,
        action="store",
        help="port of %(prog)s board")
    parser.add_argument(
        "-t",
        "--host",
        type=str,
        default=default_host,
        action="store",
        help="bind %(prog)s board to ip/host")
    parser.add_argument(
        "--model",
        type=str,
        action="store",
        default="",
        help="model file path")
    parser.add_argument(
        "--cache-timeout",
        action="store",
        dest="cache_timeout",
        type=float,
        default=default_cache_timeout,
        help="memory cache timeout duration in seconds (default: %(default)s)", )
    parser.add_argument(
        "-L",
        "--language",
        type=str,
        action="store",
        default=None,
        help="specify the default language")
    parser.add_argument(
        "--public-path",
        type=str,
        action="store",
        dest="public_path",
        default=None,
        help="set public path"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        dest="api_only",
        default=False,
        help="serve api only"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="set log level, use -vvv... to get more information"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {}".format(__version__)
    )
    parser.add_argument(
        "--product",
        type=str,
        action="store",
        default=default_product,
        help="specify the product")
    parser.add_argument(
        "--disable-telemetry",
        action="store_false",
        dest="telemetry",
        default=True,
        help="disable telemetry"
    )
    parser.add_argument(
        "--theme",
        action="store",
        dest="theme",
        default=None,
        choices=support_themes,
        help="set theme"
    )
    parser.add_argument(
        'dest',
        nargs='?',
        help='set destination for log'
    )
    parser.add_argument(
        "behavior",
        nargs='?'
    )

    args = parser.parse_args()

    init_logger(args.verbose)

    return vars(args)
