#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from .. import core
from . import ps_instance
from google.protobuf import text_format

__all__ = ['Fleet']


class Fleet(object):
    """
    
    """

    def __init__(self):
        self.instance_ = ps_instance.PaddlePSInstance()
        self.fleet_ = core.FleetWrapper()

    def stop(self):
        self.instance_.barrier_worker()
        if self.instance.is_first_worker():
            self.fleet_.stop_server()
        self.instance_.barrier_worker()
        self.instance_.barrier_all()
        self.instance.finalize()

    def init_pserver(self, opt_info):
        if "fleet_desc" in opt_info:
            self.dist_desc_str_ = text_format.MessageToString(opt_info[
                "fleet_desc"])
            self.dist_desc_ = opt_info["fleet_desc"]
        else:
            print(
                "You should run distributed optimization to get opt_info first")
            sys.exit(-1)
        self.fleet_.init_server(self.dist_desc_str_)
        ip = self.fleet_.start_server()
        self.instance_.set_ip(ip)
        self.instance.barrier_all()
        ips = self.instance.gather_ips()
        self.fleet.gather_servers(ips, self.instance_.get_node_cnt())
        self.instance_.barrier_all()

    def init_worker(self, opt_info):
        if "fleet_desc" in opt_info:
            self.dist_desc_str_ = text_format.MessageToString(opt_info[
                "fleet_desc"])
            self.dist_desc_ = opt_info["fleet_desc"]
        else:
            print(
                "You should run distributed optimization to get opt_info first")
            sys.exit(-1)
        self.instance_.barrier_all()
        ips = self.instance.gather_ips()
        self.fleet_.init_worker(self.dist_desc_str_, ips,
                                self.instance_.get_node_cnt(),
                                self.instance._rankid)
        self.instance.barrier_worker()

    def init_pserver_model(self):
        if self.instance_.is_first_worker():
            self.fleet_.init_model()
        self.instance_.barrier_worker()

    def save_pserver_model(self, save_path):
        self.fleet_.save_model(save_path)
