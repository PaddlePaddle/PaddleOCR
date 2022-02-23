#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Http Server."""

import logging

import six
# NOTE: HTTPServer has a different name in python2 and python3
from http.server import HTTPServer
import http.server as SimpleHTTPServer

import time
import threading
import socket

__all__ = []


def get_logger(name, level, fmt):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler('http.log', mode='w')
    formatter = logging.Formatter(fmt=fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


_http_server_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class KVHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    """
    kv handler class for kv http server,
    it defines the way to get/set kv in server.
    """

    def do_GET(self):
        """
        get method for kv handler, get value according to key.
        """
        log_str = "GET " + self.address_string() + self.path
        paths = self.path.split('/')
        if len(paths) < 3:
            print('len of request path must be 3: ' + self.path)
            self.send_status_code(400)
            return
        _, scope, key = paths
        with self.server.kv_lock:
            value = self.server.kv.get(scope, {}).get(key)
        if value is None:
            log_str += ' , key not found: ' + key
            self.send_status_code(404)
        else:
            log_str += ' , key found: ' + key
            self.send_response(200)
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)
        _http_server_logger.info(log_str)

    def do_PUT(self):
        """
        put method for kv handler, set value according to key.
        """
        log_str = "PUT " + self.address_string() + self.path
        paths = self.path.split('/')
        if len(paths) < 3:
            print('len of request path must be 3: ' + self.path)
            self.send_status_code(400)
            return
        _, scope, key = paths
        content_length = int(self.headers['Content-Length'])
        try:
            value = self.rfile.read(content_length)
        except:
            print("receive error invalid request")
            self.send_status_code(404)
            return
        with self.server.kv_lock:
            if self.server.kv.get(scope) is None:
                self.server.kv[scope] = {}
            self.server.kv[scope][key] = value
        self.send_status_code(200)
        _http_server_logger.info(log_str)

    def do_DELETE(self):
        """
        delete method for kv handler, set value according to key.
        """
        log_str = "DELETE " + self.address_string() + self.path
        paths = self.path.split('/')
        if len(paths) < 3:
            print('len of request path must be 3: ' + self.path)
            self.send_status_code(400)
            return
        _, scope, key = paths
        with self.server.delete_kv_lock:
            if self.server.delete_kv.get(scope) is None:
                self.server.delete_kv[scope] = set()
            self.server.delete_kv[scope].add(key)
        self.send_status_code(200)
        _http_server_logger.info(log_str)

    def log_message(self, format, *args):
        """
        ignore all logging messages in kv handler.
        """
        pass

    def send_status_code(self, code):
        """
        send status code back to client.
        """
        self.send_response(code)
        self.send_header("Content-Length", 0)
        self.end_headers()


class KVHTTPServer(HTTPServer, object):
    """
    it is a http server storing kv pairs.
    """

    def __init__(self, port, handler):
        """Init."""
        super(KVHTTPServer, self).__init__(('', port), handler)
        self.delete_kv_lock = threading.Lock()
        self.delete_kv = {}
        self.kv_lock = threading.Lock()
        self.kv = {}

    def get_deleted_size(self, key):
        """
        get deleted size in key.
        """
        ret = 0
        with self.delete_kv_lock:
            ret = len(self.delete_kv.get(key, set()))
        return ret


class KVServer:
    """
    it is a server storing kv pairs, has a http server inside.
    """

    def __init__(self, port, size={}):
        """Init."""
        self.http_server = KVHTTPServer(port, KVHandler)
        self.listen_thread = None
        self.size = size

    def start(self):
        """
        start server until user calls stop to let it quit.
        """
        self.listen_thread = threading.Thread(
            target=lambda: self.http_server.serve_forever())
        self.listen_thread.start()

    def stop(self):
        """
        stop server and clear its resources.
        """
        self.http_server.shutdown()
        self.listen_thread.join()
        self.http_server.server_close()

    def should_stop(self):
        """
        return whether the server should stop.

        Returns:
            ret(bool): whether the server should stop
        """
        for key in self.size:
            s = self.http_server.get_deleted_size(key)
            if s != self.size.get(key, 0):
                return False
        return True
