# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import six
import os


class Command(object):
    def __init__(self, server, name):
        import etcd3

        srv, port = server.split(':')
        self.etcd = etcd3.client(host=srv, port=port)

        self.prefix = "/paddle/" + name
        self.node_prefix = self.prefix + '/nodes'
        self.np_path = self.prefix + '/np'

    def set_np(self, np):
        self.etcd.put(self.np_path, six.b('{}'.format(np)))

    def scale_np(self, np):
        if self.etcd.get(self.np_path)[0] != None:
            self.set_np(np)
            return True
        return False

    def clean(self):
        self.etcd.delete_prefix(self.prefix)

    def close(self):
        self.etcd.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Elastic Command')
    parser.add_argument(
        "--elastic_server", type=str, help="etcd server host:port")
    parser.add_argument("--job_id", type=str, help="job unique id")
    parser.add_argument("--np", type=int, help="job pod/node number")
    parser.add_argument("action", type=str, help="action to take")

    args = parser.parse_args()

    server = args.elastic_server or os.getenv('PADDLE_ELASTIC_SERVER')
    name = args.job_id or os.getenv('PADDLE_ELASTIC_JOB_ID')

    np = args.np or int(os.getenv('PADDLE_ELASTIC_NP', 0))

    cmd = Command(server, name)

    if args.action == "scale":
        cmd.scale_np(np)

    if args.action == "clean":
        cmd.clean()

    print("action {} done".format(args.action))

    cmd.close()
