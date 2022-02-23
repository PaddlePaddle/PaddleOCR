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
# limitations under the License.
"""Fleet Utils."""

import collections
import copy
import json
import logging
import math
import numpy as np
import os
import sys
import time
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.log_helper import get_logger
from paddle.distributed.fleet.utils.fs import LocalFS, HDFSClient
from . import utils
OpRole = core.op_proto_and_checker_maker.OpRole

__all__ = ["FleetUtil"]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s %(levelname)s: %(message)s')

fleet = None


class FleetUtil(object):
    """
    FleetUtil provides some common functions for users' convenience.

    Examples:
        .. code-block:: python

          from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
          fleet_util = FleetUtil()
          fleet_util.rank0_print("my log")

    """

    def __init__(self, mode="pslib"):
        global fleet
        op_maker = core.op_proto_and_checker_maker
        self.op_role_key = op_maker.kOpRoleAttrName()
        if mode == "pslib":
            from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet as fleet_pslib
            fleet = fleet_pslib
        elif mode == "transpiler":
            from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet as fleet_transpiler
            fleet = fleet_transpiler
        else:
            raise ValueError(
                "Please choose one mode from [\"pslib\", \"transpiler\"]")

    def rank0_print(self, s):
        """
        Worker of rank 0 print some log.

        Args:
            s(str): string to print

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.rank0_print("my log")

        """
        if fleet.worker_index() != 0:
            return
        print(s)
        sys.stdout.flush()

    def rank0_info(self, s):
        """
        Worker of rank 0 print some log info.

        Args:
            s(str): string to log

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.rank0_info("my log info")

        """
        if fleet.worker_index() != 0:
            return
        _logger.info(s)

    def rank0_error(self, s):
        """
        Worker of rank 0 print some log error.

        Args:
            s(str): string to log

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.rank0_error("my log error")

        """
        if fleet.worker_index() != 0:
            return
        _logger.error(s)

    def set_zero(self,
                 var_name,
                 scope=fluid.global_scope(),
                 place=fluid.CPUPlace(),
                 param_type="int64"):
        """
        Set tensor of a Variable to zero.

        Args:
            var_name(str): name of Variable
            scope(Scope): Scope object, default is fluid.global_scope()
            place(Place): Place object, default is fluid.CPUPlace()
            param_type(str): param data type, default is int64

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.set_zero(myvar.name, myscope)

        """
        param = scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype(param_type)
        param.set(param_array, place)

    def print_global_auc(self,
                         scope=fluid.global_scope(),
                         stat_pos="_generated_var_2",
                         stat_neg="_generated_var_3",
                         print_prefix=""):
        r"""
        Print global auc of all distributed workers.

        Args:
            scope(Scope): Scope object, default is fluid.global_scope()
            stat_pos(str): name of auc pos bucket Variable
            stat_neg(str): name of auc neg bucket Variable
            print_prefix(str): prefix of print auc

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.print_global_auc(myscope, stat_pos=stat_pos.name,
                                          stat_neg=stat_neg.name)

              # below is part of model
              emb = my_slot_net(slots, label) # emb can be fc layer of size 1
              similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(\
                  emb, min=-15.0, max=15.0), name="similarity_norm")\
              binary_predict = fluid.layers.concat(input=[\
                  fluid.layers.elementwise_sub(\
                      fluid.layers.ceil(similarity_norm), similarity_norm),\
                  similarity_norm], axis=1)
              auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, \
                  stat_neg] = fluid.layers.auc(input=binary_predict,\
                                               label=label, curve='ROC',\
                                               num_thresholds=4096)

        """
        auc_value = self.get_global_auc(scope, stat_pos, stat_neg)
        self.rank0_print(print_prefix + " global auc = %s" % auc_value)

    def get_global_auc(self,
                       scope=fluid.global_scope(),
                       stat_pos="_generated_var_2",
                       stat_neg="_generated_var_3"):
        """
        Get global auc of all distributed workers.

        Args:
            scope(Scope): Scope object, default is fluid.global_scope()
            stat_pos(str): name of auc pos bucket Variable
            stat_neg(str): name of auc neg bucket Variable

        Returns:
            auc_value(float), total_ins_num(int)

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              auc_value, _ = fleet_util.get_global_auc(myscope,
                                                       stat_pos=stat_pos,
                                                       stat_neg=stat_neg)

        """
        if scope.find_var(stat_pos) is None or scope.find_var(stat_neg) is None:
            self.rank0_print("not found auc bucket")
            return None
        fleet._role_maker._barrier_worker()
        # auc pos bucket
        pos = np.array(scope.find_var(stat_pos).get_tensor())
        # auc pos bucket shape
        old_pos_shape = np.array(pos.shape)
        # reshape to one dim
        pos = pos.reshape(-1)
        global_pos = np.copy(pos) * 0
        # mpi allreduce
        fleet._role_maker._all_reduce(pos, global_pos)
        # reshape to its original shape
        global_pos = global_pos.reshape(old_pos_shape)

        # auc neg bucket
        neg = np.array(scope.find_var(stat_neg).get_tensor())
        old_neg_shape = np.array(neg.shape)
        neg = neg.reshape(-1)
        global_neg = np.copy(neg) * 0
        fleet._role_maker._all_reduce(neg, global_neg)
        global_neg = global_neg.reshape(old_neg_shape)

        # calculate auc
        num_bucket = len(global_pos[0])
        area = 0.0
        pos = 0.0
        neg = 0.0
        new_pos = 0.0
        new_neg = 0.0
        total_ins_num = 0
        for i in range(num_bucket):
            index = num_bucket - 1 - i
            new_pos = pos + global_pos[0][index]
            total_ins_num += global_pos[0][index]
            new_neg = neg + global_neg[0][index]
            total_ins_num += global_neg[0][index]
            area += (new_neg - neg) * (pos + new_pos) / 2
            pos = new_pos
            neg = new_neg

        auc_value = None
        if pos * neg == 0 or total_ins_num == 0:
            auc_value = 0.5
        else:
            auc_value = area / (pos * neg)

        fleet._role_maker._barrier_worker()
        return auc_value

    def load_fleet_model_one_table(self, table_id, path):
        """
        load pslib model to one table

        Args:
            table_id(int): load model to one table, default is None, which mean
                           load all table.
            path(str): model path

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.load_fleet_model("hdfs:/my/model/path", table_id=1)
        """
        fleet.load_one_table(table_id, path)

    def load_fleet_model(self, path, mode=0):
        """
        load pslib model

        Args:
            path(str): model path
            mode(str): 0 or 1, which means load checkpoint or delta model,
                       default is 0

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()

              fleet_util.load_fleet_model("hdfs:/my/model/path")

              fleet_util.load_fleet_model("hdfs:/my/model/path", mode=0)

        """
        fleet.init_server(path, mode=mode)

    def save_fleet_model(self, path, mode=0):
        """
        save pslib model

        Args:
            path(str): model path
            mode(str): 0 or 1, which means save checkpoint or delta model,
                       default is 0

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_fleet_model("hdfs:/my/model/path")

        """
        fleet.save_persistables(None, path, mode=mode)

    def _get_xbox_str(self,
                      output_path,
                      day,
                      model_path,
                      xbox_base_key,
                      data_path,
                      hadoop_fs_name,
                      monitor_data={},
                      mode="patch"):
        xbox_dict = collections.OrderedDict()
        if mode == "base":
            xbox_dict["id"] = str(xbox_base_key)
        elif mode == "patch":
            xbox_dict["id"] = str(int(time.time()))
        else:
            print("warning: unknown mode %s, set it to patch" % mode)
            mode = "patch"
            xbox_dict["id"] = str(int(time.time()))
        xbox_dict["key"] = str(xbox_base_key)
        if model_path.startswith("hdfs:") or model_path.startswith("afs:"):
            model_path = model_path[model_path.find(":") + 1:]
        xbox_dict["input"] = hadoop_fs_name + model_path.rstrip("/") + "/000"
        xbox_dict["record_count"] = "111111"
        xbox_dict["partition_type"] = "2"
        xbox_dict["job_name"] = "default_job_name"
        xbox_dict["ins_tag"] = "feasign"
        xbox_dict["ins_path"] = data_path
        job_id_with_host = os.popen("echo -n ${JOB_ID}").read().strip()
        instance_id = os.popen("echo -n ${INSTANCE_ID}").read().strip()
        start_pos = instance_id.find(job_id_with_host)
        end_pos = instance_id.find("--")
        if start_pos != -1 and end_pos != -1:
            job_id_with_host = instance_id[start_pos:end_pos]
        xbox_dict["job_id"] = job_id_with_host
        # currently hard code here, set monitor_data empty string
        xbox_dict["monitor_data"] = ""
        xbox_dict["monitor_path"] = output_path.rstrip("/") + "/monitor/" \
                                    + day + ".txt"
        xbox_dict["mpi_size"] = str(fleet.worker_num())
        return json.dumps(xbox_dict)

    def write_model_donefile(self,
                             output_path,
                             day,
                             pass_id,
                             xbox_base_key,
                             hadoop_fs_name,
                             hadoop_fs_ugi,
                             hadoop_home="$HADOOP_HOME",
                             donefile_name="donefile.txt"):
        """
        write donefile when save model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id
            xbox_base_key(str|int): xbox base key
            hadoop_fs_name(str): hdfs/afs fs name
            hadoop_fs_ugi(str): hdfs/afs fs ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"
            donefile_name(str): donefile name, default is "donefile.txt"

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.write_model_donefile(output_path="hdfs:/my/output",
                                              model_path="hdfs:/my/model",
                                              day=20190723,
                                              pass_id=66,
                                              xbox_base_key=int(time.time()),
                                              hadoop_fs_name="hdfs://xxx",
                                              hadoop_fs_ugi="user,passwd")

        """
        day = str(day)
        pass_id = str(pass_id)
        xbox_base_key = int(xbox_base_key)

        if pass_id != "-1":
            suffix_name = "/%s/%s/" % (day, pass_id)
            model_path = output_path.rstrip("/") + suffix_name
        else:
            suffix_name = "/%s/0/" % day
            model_path = output_path.rstrip("/") + suffix_name

        if fleet.worker_index() == 0:
            donefile_path = output_path + "/" + donefile_name
            content  = "%s\t%lu\t%s\t%s\t%d" % (day, xbox_base_key,\
                                                model_path, pass_id, 0)
            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)
            if client.is_file(donefile_path):
                pre_content = client.cat(donefile_path)
                pre_content_list = pre_content.split("\n")
                day_list = [i.split("\t")[0] for i in pre_content_list]
                pass_list = [i.split("\t")[3] for i in pre_content_list]
                exist = False
                for i in range(len(day_list)):
                    if int(day) == int(day_list[i]) and \
                            int(pass_id) == int(pass_list[i]):
                        exist = True
                        break
                if not exist:
                    with open(donefile_name, "w") as f:
                        f.write(pre_content + "\n")
                        f.write(content + "\n")
                    client.delete(donefile_path)
                    client.upload(donefile_name, output_path)
                    self.rank0_error("write %s/%s %s succeed" % \
                                      (day, pass_id, donefile_name))
                else:
                    self.rank0_error("not write %s because %s/%s already "
                                     "exists" % (donefile_name, day, pass_id))
            else:
                with open(donefile_name, "w") as f:
                    f.write(content + "\n")
                client.upload(donefile_name, output_path)
                self.rank0_error("write %s/%s %s succeed" % \
                               (day, pass_id, donefile_name))
        fleet._role_maker._barrier_worker()

    def write_xbox_donefile(self,
                            output_path,
                            day,
                            pass_id,
                            xbox_base_key,
                            data_path,
                            hadoop_fs_name,
                            hadoop_fs_ugi,
                            monitor_data={},
                            hadoop_home="$HADOOP_HOME",
                            donefile_name=None):
        """
        write delta donefile or xbox base donefile

        Args:
            output_path(str): output path
            day(str|int): training day of model
            pass_id(str|int): training pass id of model
            xbox_base_key(str|int): xbox base key
            data_path(str|list): training data path
            hadoop_fs_name(str): hdfs/afs fs name
            hadoop_fs_ugi(str): hdfs/afs fs ugi
            monitor_data(dict): metrics
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"
            donefile_name(str): donefile name, default is None"

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.write_xbox_donefile(
                  output_path="hdfs:/my/output/",
                  model_path="hdfs:/my/output/20190722/01",
                  day=20190722,
                  pass_id=1,
                  xbox_base_key=int(time.time()),
                  data_path="hdfs:/my/data/",
                  hadoop_fs_name="hdfs://xxx",
                  hadoop_fs_ugi="user,passwd",
                  monitor_data={}
                  )

        """
        day = str(day)
        pass_id = str(pass_id)
        xbox_base_key = int(xbox_base_key)
        mode = None

        if pass_id != "-1":
            mode = "patch"
            suffix_name = "/%s/delta-%s/" % (day, pass_id)
            model_path = output_path.rstrip("/") + suffix_name
            if donefile_name is None:
                donefile_name = "xbox_patch_done.txt"
        else:
            mode = "base"
            suffix_name = "/%s/base/" % day
            model_path = output_path.rstrip("/") + suffix_name
            if donefile_name is None:
                donefile_name = "xbox_base_done.txt"

        if isinstance(data_path, list):
            data_path = ",".join(data_path)

        if fleet.worker_index() == 0:
            donefile_path = output_path + "/" + donefile_name
            xbox_str = self._get_xbox_str(output_path, day, model_path, \
                    xbox_base_key, data_path, hadoop_fs_name, monitor_data={},
                    mode=mode)
            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)
            if client.is_file(donefile_path):
                pre_content = client.cat(donefile_path)
                last_dict = json.loads(pre_content.split("\n")[-1])
                last_day = last_dict["input"].split("/")[-3]
                last_pass = last_dict["input"].split("/")[-2].split("-")[-1]
                exist = False
                if int(day) < int(last_day) or \
                        int(day) == int(last_day) and \
                        int(pass_id) <= int(last_pass):
                    exist = True
                if not exist:
                    with open(donefile_name, "w") as f:
                        f.write(pre_content + "\n")
                        f.write(xbox_str + "\n")
                    client.delete(donefile_path)
                    client.upload(donefile_name, output_path)
                    self.rank0_error("write %s/%s %s succeed" % \
                                      (day, pass_id, donefile_name))
                else:
                    self.rank0_error("not write %s because %s/%s already "
                                     "exists" % (donefile_name, day, pass_id))
            else:
                with open(donefile_name, "w") as f:
                    f.write(xbox_str + "\n")
                client.upload(donefile_name, output_path)
                self.rank0_error("write %s/%s %s succeed" % \
                               (day, pass_id, donefile_name))
        fleet._role_maker._barrier_worker()

    def write_cache_donefile(self,
                             output_path,
                             day,
                             pass_id,
                             key_num,
                             hadoop_fs_name,
                             hadoop_fs_ugi,
                             hadoop_home="$HADOOP_HOME",
                             donefile_name="sparse_cache.meta",
                             **kwargs):
        """
        write cache donefile

        Args:
            output_path(str): output path
            day(str|int): training day of model
            pass_id(str|int): training pass id of model
            key_num(str|int): save cache return value
            hadoop_fs_name(str): hdfs/afs fs name
            hadoop_fs_ugi(str): hdfs/afs fs ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"
            donefile_name(str): donefile name, default is "sparse_cache.meta"
            kwargs(dict): user defined properties
                          file_num(int): cache file num
                          table_id(int): cache table id

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.write_cache_donefile(
                  output_path="hdfs:/my/output/",
                  day=20190722,
                  pass_id=1,
                  key_num=123456,
                  hadoop_fs_name="hdfs://xxx",
                  hadoop_fs_ugi="user,passwd",
                  )

        """
        day = str(day)
        pass_id = str(pass_id)
        key_num = int(key_num)
        file_num = kwargs.get("file_num", 16)
        table_id = kwargs.get("table_id", 0)

        if pass_id != "-1":
            suffix_name = "/%s/delta-%s/%03d_cache" % (day, pass_id, table_id)
            model_path = output_path.rstrip("/") + suffix_name
        else:
            suffix_name = "/%s/base/%03d_cache" % (day, table_id)
            model_path = output_path.rstrip("/") + suffix_name

        if fleet.worker_index() == 0:
            donefile_path = model_path + "/" + donefile_name
            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)
            if client.is_file(donefile_path):
                self.rank0_error( \
                    "not write because %s already exists" % donefile_path)
            else:
                meta_str = "file_prefix:part\npart_num:%s\nkey_num:%d\n" \
                           % (file_num, key_num)
                with open(donefile_name, "w") as f:
                    f.write(meta_str)
                client.upload(donefile_name, model_path)
                self.rank0_error("write %s succeed" % donefile_path)
        fleet._role_maker._barrier_worker()

    def load_model(self, output_path, day, pass_id):
        """
        load pslib model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.load_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = "/%s/%s/" % (day, pass_id)
        load_path = output_path + suffix_name
        self.rank0_error("going to load_model %s" % load_path)
        self.load_fleet_model(load_path)
        self.rank0_error("load_model done")

    def save_model(self, output_path, day, pass_id):
        """
        save pslib model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = "/%s/%s/" % (day, pass_id)
        model_path = output_path + suffix_name
        self.rank0_print("going to save_model %s" % model_path)
        self.save_fleet_model(model_path)
        self.rank0_print("save_model done")

    def save_batch_model(self, output_path, day):
        """
        save batch model

        Args:
            output_path(str): output path
            day(str|int): training day

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_batch_model("hdfs:/my/path", 20190722)

        """
        day = str(day)
        suffix_name = "/%s/0/" % day
        model_path = output_path + suffix_name
        self.rank0_print("going to save_model %s" % model_path)
        fleet.save_persistables(None, model_path, mode=3)
        self.rank0_print("save_batch_model done")

    def save_delta_model(self, output_path, day, pass_id):
        """
        save delta model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_batch_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = "/%s/delta-%s/" % (day, pass_id)
        model_path = output_path + suffix_name
        self.rank0_print("going to save_delta_model %s" % model_path)
        fleet.save_persistables(None, model_path, mode=1)
        self.rank0_print("save_delta_model done")

    def save_xbox_base_model(self, output_path, day):
        """
        save xbox base model

        Args:
            output_path(str): output path
            day(str|int): training day

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_xbox_base_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        suffix_name = "/%s/base/" % day
        model_path = output_path + suffix_name
        self.rank0_print("going to save_xbox_base_model " + model_path)
        fleet.save_persistables(None, model_path, mode=2)
        self.rank0_print("save_xbox_base_model done")

    def save_cache_model(self, output_path, day, pass_id, mode=1, **kwargs):
        """
        save cache model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id
            mode(str|int): save mode
            kwargs(dict): user defined properties
                          table_id(int): table id to save cache

        Returns:
            key_num(int): cache key num

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_cache_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        pass_id = str(pass_id)
        mode = int(mode)
        table_id = kwargs.get("table_id", 0)
        suffix_name = "/%s/delta-%s" % (day, pass_id)
        model_path = output_path.rstrip("/") + suffix_name
        self.rank0_print("going to save_cache_model %s" % model_path)
        key_num = fleet.save_cache_model(
            None, model_path, mode=mode, table_id=table_id)
        self.rank0_print("save_cache_model done")
        return key_num

    def save_cache_base_model(self, output_path, day, **kwargs):
        """
        save cache model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id
            kwargs(dict): user defined properties
                          table_id(int): table id to save cache

        Returns:
            key_num(int): cache key num

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_cache_base_model("hdfs:/my/path", 20190722)

        """
        day = str(day)
        table_id = kwargs.get("table_id", 0)
        suffix_name = "/%s/base" % day
        model_path = output_path.rstrip("/") + suffix_name
        self.rank0_print("going to save_cache_base_model %s" % model_path)
        key_num = fleet.save_cache_model(
            None, model_path, mode=2, table_id=table_id)
        self.rank0_print("save_cache_base_model done")
        return key_num

    def pull_all_dense_params(self, scope, program):
        """
        pull all dense params in trainer of rank 0

        Args:
            scope(Scope): fluid Scope
            program(Program): fluid Program

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.pull_all_dense_params(my_scope, my_program)

        """
        fleet._role_maker._barrier_worker()
        if fleet._role_maker.is_first_worker():
            prog_id = str(id(program))
            tables = fleet._opt_info["program_id_to_worker"][prog_id].\
                get_desc().dense_table
            prog_conf = fleet._opt_info['program_configs'][prog_id]
            prog_tables = {}
            for key in prog_conf:
                if "dense" not in key:
                    continue
                for table_id in prog_conf[key]:
                    prog_tables[int(table_id)] = 0
            for table in tables:
                if int(table.table_id) not in prog_tables:
                    continue
                var_name_list = []
                for i in range(0, len(table.dense_variable_name)):
                    var_name = table.dense_variable_name[i]
                    if scope.find_var(var_name) is None:
                        raise ValueError("var " + var_name +
                                         " not found in scope " +
                                         "when pull dense")
                    var_name_list.append(var_name)
                fleet._fleet_ptr.pull_dense(scope,
                                            int(table.table_id), var_name_list)
        fleet._role_maker._barrier_worker()

    def save_paddle_inference_model(self,
                                    executor,
                                    scope,
                                    program,
                                    feeded_vars,
                                    target_vars,
                                    output_path,
                                    day,
                                    pass_id,
                                    hadoop_fs_name,
                                    hadoop_fs_ugi,
                                    hadoop_home="$HADOOP_HOME",
                                    save_combine=True):
        """
        save paddle inference model, and upload to hdfs dnn_plugin path

        Args:
            executor(Executor): fluid Executor
            scope(Scope): fluid Scope
            program(Program): fluid Program
            feeded_vars(list[Variable]): feed vars
            target_vars(list[variable]): fetch vars
            output_path(str): hdfs/afs output path
            day(str|int): training day
            pass_id(str|int): training pass
            hadoop_fs_name(str): hadoop fs name
            hadoop_fs_ugi(str): hadoop fs ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"
            save_combine(bool): whether to save in a file or separate files,
                                default is True

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_paddle_inference_model(exe,
                                                     join_scope,
                                                     join_program,
                                                     feeded_vars,
                                                     target_vars,
                                                     "hdfs:/my/output/path/",
                                                     day=20190727,
                                                     pass_id=6,
                                                     hadoop_fs_name="xxx",
                                                     hadoop_fs_ugi="xxx,xxx")
        """
        day = str(day)
        pass_id = str(pass_id)
        feeded_var_names = [i.name for i in feeded_vars]
        model_name = "inference_model"
        # pull dense before save
        self.pull_all_dense_params(scope, program)
        if fleet.worker_index() == 0:
            with fluid.scope_guard(scope):
                if save_combine:
                    fluid.io.save_inference_model(
                        dirname=model_name,
                        feeded_var_names=feeded_var_names,
                        target_vars=target_vars,
                        executor=executor,
                        main_program=program.clone(),
                        params_filename="params")
                else:
                    fluid.io.save_inference_model(
                        dirname=model_name,
                        feeded_var_names=feeded_var_names,
                        target_vars=target_vars,
                        executor=executor,
                        main_program=program.clone())

            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)

            if pass_id == "-1":
                dest = "%s/%s/base/dnn_plugin/" % (output_path, day)
            else:
                dest = "%s/%s/delta-%s/dnn_plugin/" % (output_path, day,
                                                       pass_id)
            if not client.is_exist(dest):
                client.makedirs(dest)

            client.upload(model_name, dest, multi_processes=5, overwrite=True)

        fleet._role_maker._barrier_worker()

    def save_paddle_params(self,
                           executor,
                           scope,
                           program,
                           model_name,
                           output_path,
                           day,
                           pass_id,
                           hadoop_fs_name,
                           hadoop_fs_ugi,
                           hadoop_home="$HADOOP_HOME",
                           var_names=None,
                           save_combine=True):
        """
        save paddle model, and upload to hdfs dnn_plugin path

        Args:
            executor(Executor): fluid Executor
            scope(Scope): fluid Scope
            program(Program): fluid Program
            model_name(str): save model local dir or filename
            output_path(str): hdfs/afs output path
            day(str|int): training day
            pass_id(str|int): training pass
            hadoop_fs_name(str): hadoop fs name
            hadoop_fs_ugi(str): hadoop fs ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"
            var_names(list): save persistable var names, default is None
            save_combine(bool): whether to save in a file or separate files,
                                default is True

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_paddle_params(exe,
                                            join_scope,
                                            join_program,
                                            "paddle_dense.model.0",
                                            "hdfs:/my/output/path/",
                                            day=20190727,
                                            pass_id=6,
                                            hadoop_fs_name="xxx",
                                            hadoop_fs_ugi="xxx,xxx",
                                            var_names=join_all_var_names)
              fleet_util.save_paddle_params(exe,
                                            join_scope,
                                            join_program,
                                            "paddle_dense.model.usr.0",
                                            "hdfs:/my/output/path/",
                                            day=20190727,
                                            pass_id=6,
                                            hadoop_fs_name="xxx",
                                            hadoop_fs_ugi="xxx,xxx",
                                            var_names=join_user_var_names)
              fleet_util.save_paddle_params(exe,
                                            join_scope,
                                            join_program,
                                            "paddle_dense.model.item.0",
                                            "hdfs:/my/output/path/",
                                            day=20190727,
                                            pass_id=6,
                                            hadoop_fs_name="xxx",
                                            hadoop_fs_ugi="xxx,xxx",
                                            var_names=join_user_item_names)

        """
        day = str(day)
        pass_id = str(pass_id)
        # pull dense before save
        self.pull_all_dense_params(scope, program)
        if fleet.worker_index() == 0:
            vars = [program.global_block().var(i) for i in var_names]
            with fluid.scope_guard(scope):
                if save_combine:
                    fluid.io.save_vars(
                        executor, "./", program, vars=vars, filename=model_name)
                else:
                    fluid.io.save_vars(executor, model_name, program, vars=vars)

            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)

            if pass_id == "-1":
                dest = "%s/%s/base/dnn_plugin/" % (output_path, day)
            else:
                dest = "%s/%s/delta-%s/dnn_plugin/" % (output_path, day,
                                                       pass_id)
            if not client.is_exist(dest):
                client.mkdirs(dest)
            client.upload(model_name, dest, multi_processes=5, overwrite=True)

        fleet._role_maker._barrier_worker()

    def get_last_save_xbox_base(self,
                                output_path,
                                hadoop_fs_name,
                                hadoop_fs_ugi,
                                hadoop_home="$HADOOP_HOME"):
        r"""
        get last saved base xbox info from xbox_base_done.txt

        Args:
            output_path(str): output path
            hadoop_fs_name(str): hdfs/afs fs_name
            hadoop_fs_ugi(str): hdfs/afs fs_ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"

        Returns:
            [last_save_day, last_path, xbox_base_key]
            last_save_day(int): day of saved model
            last_path(str): model path
            xbox_base_key(int): xbox key

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              last_save_day, last_path, xbox_base_key = \
                  fleet_util.get_last_save_xbox_base("hdfs:/my/path", 20190722,
                                                     88)

        """
        donefile_path = output_path + "/xbox_base_done.txt"
        configs = {
            "fs.default.name": hadoop_fs_name,
            "hadoop.job.ugi": hadoop_fs_ugi
        }
        client = HDFSClient(hadoop_home, configs)
        if not client.is_file(donefile_path):
            return [-1, -1, int(time.time())]
        pre_content = client.cat(donefile_path)
        last_dict = json.loads(pre_content.split("\n")[-1])
        last_day = int(last_dict["input"].split("/")[-3])
        last_path = "/".join(last_dict["input"].split("/")[:-1])
        xbox_base_key = int(last_dict["key"])
        return [last_day, last_path, xbox_base_key]

    def get_last_save_xbox(self,
                           output_path,
                           hadoop_fs_name,
                           hadoop_fs_ugi,
                           hadoop_home="$HADOOP_HOME"):
        r"""
        get last saved xbox info from xbox_patch_done.txt

        Args:
            output_path(str): output path
            hadoop_fs_name(str): hdfs/afs fs_name
            hadoop_fs_ugi(str): hdfs/afs fs_ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"

        Returns:
            [last_save_day, last_save_pass, last_path, xbox_base_key]
            last_save_day(int): day of saved model
            last_save_pass(int): pass id of saved
            last_path(str): model path
            xbox_base_key(int): xbox key

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              last_save_day, last_save_pass, last_path, xbox_base_key = \
                  fleet_util.get_last_save_xbox("hdfs:/my/path", 20190722, 88)

        """
        donefile_path = output_path + "/xbox_patch_done.txt"
        configs = {
            "fs.default.name": hadoop_fs_name,
            "hadoop.job.ugi": hadoop_fs_ugi
        }
        client = HDFSClient(hadoop_home, configs)
        if not client.is_file(donefile_path):
            return [-1, -1, "", int(time.time())]
        pre_content = client.cat(donefile_path)
        last_dict = json.loads(pre_content.split("\n")[-1])
        last_day = int(last_dict["input"].split("/")[-3])
        last_pass = int(last_dict["input"].split("/")[-2].split("-")[-1])
        last_path = "/".join(last_dict["input"].split("/")[:-1])
        xbox_base_key = int(last_dict["key"])
        return [last_day, last_pass, last_path, xbox_base_key]

    def get_last_save_model(self,
                            output_path,
                            hadoop_fs_name,
                            hadoop_fs_ugi,
                            hadoop_home="$HADOOP_HOME"):
        r"""
        get last saved model info from donefile.txt

        Args:
            output_path(str): output path
            hadoop_fs_name(str): hdfs/afs fs_name
            hadoop_fs_ugi(str): hdfs/afs fs_ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"

        Returns:
            [last_save_day, last_save_pass, last_path, xbox_base_key]
            last_save_day(int): day of saved model
            last_save_pass(int): pass id of saved
            last_path(str): model path
            xbox_base_key(int): xbox key

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              last_save_day, last_save_pass, last_path, xbox_base_key = \
                  fleet_util.get_last_save_model("hdfs:/my/path", 20190722, 88)

        """
        last_save_day = -1
        last_save_pass = -1
        last_path = ""
        donefile_path = output_path + "/donefile.txt"
        configs = {
            "fs.default.name": hadoop_fs_name,
            "hadoop.job.ugi": hadoop_fs_ugi
        }
        client = HDFSClient(hadoop_home, configs)
        if not client.is_file(donefile_path):
            return [-1, -1, "", int(time.time())]
        content = client.cat(donefile_path)
        content = content.split("\n")[-1].split("\t")
        last_save_day = int(content[0])
        last_save_pass = int(content[3])
        last_path = content[2]
        xbox_base_key = int(content[1])
        return [last_save_day, last_save_pass, last_path, xbox_base_key]

    def get_online_pass_interval(self, days, hours, split_interval,
                                 split_per_pass, is_data_hourly_placed):
        """
        get online pass interval

        Args:
            days(str): days to train
            hours(str): hours to train
            split_interval(int|str): split interval
            split_per_pass(int}str): split per pass
            is_data_hourly_placed(bool): is data hourly placed

        Returns:
            online_pass_interval(list)

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              online_pass_interval = fleet_util.get_online_pass_interval(
                  days="{20190720..20190729}",
                  hours="{0..23}",
                  split_interval=5,
                  split_per_pass=2,
                  is_data_hourly_placed=False)

        """
        days = os.popen("echo -n " + days).read().split(" ")
        hours = os.popen("echo -n " + hours).read().split(" ")
        split_interval = int(split_interval)
        split_per_pass = int(split_per_pass)
        splits_per_day = 24 * 60 // split_interval
        pass_per_day = splits_per_day // split_per_pass
        left_train_hour = int(hours[0])
        right_train_hour = int(hours[-1])

        start = 0
        split_path = []
        for i in range(splits_per_day):
            h = start // 60
            m = start % 60
            if h < left_train_hour or h > right_train_hour:
                start += split_interval
                continue
            if is_data_hourly_placed:
                split_path.append("%02d" % h)
            else:
                split_path.append("%02d%02d" % (h, m))
            start += split_interval

        start = 0
        online_pass_interval = []
        for i in range(pass_per_day):
            online_pass_interval.append([])
            for j in range(start, start + split_per_pass):
                online_pass_interval[i].append(split_path[j])
            start += split_per_pass

        return online_pass_interval

    def get_global_metrics(self,
                           scope=fluid.global_scope(),
                           stat_pos_name="_generated_var_2",
                           stat_neg_name="_generated_var_3",
                           sqrerr_name="sqrerr",
                           abserr_name="abserr",
                           prob_name="prob",
                           q_name="q",
                           pos_ins_num_name="pos",
                           total_ins_num_name="total"):
        r"""
        get global metrics, including auc, bucket_error, mae, rmse,
        actual_ctr, predicted_ctr, copc, mean_predict_qvalue, total_ins_num.

        Args:
            scope(Scope): Scope object, default is fluid.global_scope()
            stat_pos_name(str): name of auc pos bucket Variable
            stat_neg_name(str): name of auc neg bucket Variable
            sqrerr_name(str): name of sqrerr Variable
            abserr_name(str): name of abserr Variable
            prob_name(str): name of prob Variable
            q_name(str): name of q Variable
            pos_ins_num_name(str): name of pos ins num Variable
            total_ins_num_name(str): name of total ins num Variable

        Returns:
            [auc, bucket_error, mae, rmse, actual_ctr, predicted_ctr, copc,
             mean_predict_qvalue, total_ins_num]

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              metric_list = fleet_util.get_global_metrics(myscope,
                                                          stat_pos.name,
                                                          stat_neg.name,
                                                          local_sqrerr.name,
                                                          local_abserr.name,
                                                          local_prob.name,
                                                          local_q.name,
                                                          local_pos_ins.name,
                                                          local_total_ins.name)

              # below is part of example model
              label = fluid.layers.data(name="click", shape=[-1, 1],\
                  dtype="int64", lod_level=0, append_batch_size=False)
              emb = my_slot_net(slots, label) # emb can be fc layer of size 1
              similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(\
                  emb, min=-15.0, max=15.0), name="similarity_norm")\
              binary_predict = fluid.layers.concat(input=[\
                  fluid.layers.elementwise_sub(\
                      fluid.layers.ceil(similarity_norm), similarity_norm),\
                  similarity_norm], axis=1)
              auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, \
                  stat_neg] = fluid.layers.auc(input=binary_predict,\
                                               label=label, curve='ROC',\
                                               num_thresholds=4096)
              local_sqrerr, local_abserr, local_prob, local_q, local_pos_ins,\
                  local_total_ins = fluid.contrib.layers.ctr_metric_bundle(\
                      similarity_norm, label)

        """
        if scope.find_var(stat_pos_name) is None or \
                scope.find_var(stat_neg_name) is None:
            self.rank0_print("not found auc bucket")
            return [None] * 9
        elif scope.find_var(sqrerr_name) is None:
            self.rank0_print("not found sqrerr_name=%s" % sqrerr_name)
            return [None] * 9
        elif scope.find_var(abserr_name) is None:
            self.rank0_print("not found abserr_name=%s" % abserr_name)
            return [None] * 9
        elif scope.find_var(prob_name) is None:
            self.rank0_print("not found prob_name=%s" % prob_name)
            return [None] * 9
        elif scope.find_var(q_name) is None:
            self.rank0_print("not found q_name=%s" % q_name)
            return [None] * 9
        elif scope.find_var(pos_ins_num_name) is None:
            self.rank0_print("not found pos_ins_num_name=%s" % pos_ins_num_name)
            return [None] * 9
        elif scope.find_var(total_ins_num_name) is None:
            self.rank0_print("not found total_ins_num_name=%s" % \
                             total_ins_num_name)
            return [None] * 9

        # barrier worker to ensure all workers finished training
        fleet._role_maker._barrier_worker()

        # get auc
        auc = self.get_global_auc(scope, stat_pos_name, stat_neg_name)
        pos = np.array(scope.find_var(stat_pos_name).get_tensor())
        # auc pos bucket shape
        old_pos_shape = np.array(pos.shape)
        # reshape to one dim
        pos = pos.reshape(-1)
        global_pos = np.copy(pos) * 0
        # mpi allreduce
        fleet._role_maker._all_reduce(pos, global_pos)
        # reshape to its original shape
        global_pos = global_pos.reshape(old_pos_shape)
        # auc neg bucket
        neg = np.array(scope.find_var(stat_neg_name).get_tensor())
        old_neg_shape = np.array(neg.shape)
        neg = neg.reshape(-1)
        global_neg = np.copy(neg) * 0
        fleet._role_maker._all_reduce(neg, global_neg)
        global_neg = global_neg.reshape(old_neg_shape)

        num_bucket = len(global_pos[0])

        def get_metric(name):
            metric = np.array(scope.find_var(name).get_tensor())
            old_metric_shape = np.array(metric.shape)
            metric = metric.reshape(-1)
            global_metric = np.copy(metric) * 0
            fleet._role_maker._all_reduce(metric, global_metric)
            global_metric = global_metric.reshape(old_metric_shape)
            return global_metric[0]

        global_sqrerr = get_metric(sqrerr_name)
        global_abserr = get_metric(abserr_name)
        global_prob = get_metric(prob_name)
        global_q_value = get_metric(q_name)
        # note: get ins_num from auc bucket is not actual value,
        # so get it from metric op
        pos_ins_num = get_metric(pos_ins_num_name)
        total_ins_num = get_metric(total_ins_num_name)
        neg_ins_num = total_ins_num - pos_ins_num

        mae = global_abserr / total_ins_num
        rmse = math.sqrt(global_sqrerr / total_ins_num)
        return_actual_ctr = pos_ins_num / total_ins_num
        predicted_ctr = global_prob / total_ins_num
        mean_predict_qvalue = global_q_value / total_ins_num
        copc = 0.0
        if abs(predicted_ctr > 1e-6):
            copc = return_actual_ctr / predicted_ctr

        # calculate bucket error
        last_ctr = -1.0
        impression_sum = 0.0
        ctr_sum = 0.0
        click_sum = 0.0
        error_sum = 0.0
        error_count = 0.0
        click = 0.0
        show = 0.0
        ctr = 0.0
        adjust_ctr = 0.0
        relative_error = 0.0
        actual_ctr = 0.0
        relative_ctr_error = 0.0
        k_max_span = 0.01
        k_relative_error_bound = 0.05
        for i in range(num_bucket):
            click = global_pos[0][i]
            show = global_pos[0][i] + global_neg[0][i]
            ctr = float(i) / num_bucket
            if abs(ctr - last_ctr) > k_max_span:
                last_ctr = ctr
                impression_sum = 0.0
                ctr_sum = 0.0
                click_sum = 0.0
            impression_sum += show
            ctr_sum += ctr * show
            click_sum += click
            if impression_sum == 0:
                continue
            adjust_ctr = ctr_sum / impression_sum
            if adjust_ctr == 0:
                continue
            relative_error = \
                           math.sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum))
            if relative_error < k_relative_error_bound:
                actual_ctr = click_sum / impression_sum
                relative_ctr_error = abs(actual_ctr / adjust_ctr - 1)
                error_sum += relative_ctr_error * impression_sum
                error_count += impression_sum
                last_ctr = -1

        bucket_error = error_sum / error_count if error_count > 0 else 0.0

        return [
            auc, bucket_error, mae, rmse, return_actual_ctr, predicted_ctr,
            copc, mean_predict_qvalue, int(total_ins_num)
        ]

    def print_global_metrics(self,
                             scope=fluid.global_scope(),
                             stat_pos_name="_generated_var_2",
                             stat_neg_name="_generated_var_3",
                             sqrerr_name="sqrerr",
                             abserr_name="abserr",
                             prob_name="prob",
                             q_name="q",
                             pos_ins_num_name="pos",
                             total_ins_num_name="total",
                             print_prefix=""):
        r"""
        print global metrics, including auc, bucket_error, mae, rmse,
        actual_ctr, predicted_ctr, copc, mean_predict_qvalue, total_ins_num.

        Args:
            scope(Scope): Scope object, default is fluid.global_scope()
            stat_pos_name(str): name of auc pos bucket Variable
            stat_neg_name(str): name of auc neg bucket Variable
            sqrerr_name(str): name of sqrerr Variable
            abserr_name(str): name of abserr Variable
            prob_name(str): name of prob Variable
            q_name(str): name of q Variable
            pos_ins_num_name(str): name of pos ins num Variable
            total_ins_num_name(str): name of total ins num Variable
            print_prefix(str): print prefix

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.print_global_metrics(myscope,
                                              stat_pos.name,
                                              stat_neg.name,
                                              local_sqrerr.name,
                                              local_abserr.name,
                                              local_prob.name,
                                              local_q.name,
                                              local_pos_ins.name,
                                              local_total_ins.name)

              # below is part of model
              label = fluid.layers.data(name="click", shape=[-1, 1],\
                  dtype="int64", lod_level=0, append_batch_size=False)
              emb = my_slot_net(slots, label) # emb can be fc layer of size 1
              similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(\
                  emb, min=-15.0, max=15.0), name="similarity_norm")\
              binary_predict = fluid.layers.concat(input=[\
                  fluid.layers.elementwise_sub(\
                      fluid.layers.ceil(similarity_norm), similarity_norm),\
                  similarity_norm], axis=1)
              auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, \
                  stat_neg] = fluid.layers.auc(input=binary_predict,\
                                               label=label, curve='ROC',\
                                               num_thresholds=4096)
              local_sqrerr, local_abserr, local_prob, local_q, local_pos_ins, \
                  local_total_ins = fluid.contrib.layers.ctr_metric_bundle(\
                      similarity_norm, label)

        """
        if scope.find_var(stat_pos_name) is None or \
                scope.find_var(stat_neg_name) is None:
            self.rank0_print("not found auc bucket")
            return
        elif scope.find_var(sqrerr_name) is None:
            self.rank0_print("not found sqrerr_name=%s" % sqrerr_name)
            return
        elif scope.find_var(abserr_name) is None:
            self.rank0_print("not found abserr_name=%s" % abserr_name)
            return
        elif scope.find_var(prob_name) is None:
            self.rank0_print("not found prob_name=%s" % prob_name)
            return
        elif scope.find_var(q_name) is None:
            self.rank0_print("not found q_name=%s" % q_name)
            return
        elif scope.find_var(pos_ins_num_name) is None:
            self.rank0_print("not found pos_ins_num_name=%s" % pos_ins_num_name)
            return
        elif scope.find_var(total_ins_num_name) is None:
            self.rank0_print("not found total_ins_num_name=%s" % \
                             total_ins_num_name)
            return

        auc, bucket_error, mae, rmse, actual_ctr, predicted_ctr, copc,\
            mean_predict_qvalue, total_ins_num = self.get_global_metrics(\
            scope, stat_pos_name, stat_neg_name, sqrerr_name, abserr_name,\
            prob_name, q_name, pos_ins_num_name, total_ins_num_name)
        self.rank0_print("%s global AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f "
                         "RMSE=%.6f Actural_CTR=%.6f Predicted_CTR=%.6f "
                         "COPC=%.6f MEAN Q_VALUE=%.6f Ins number=%s" %
                         (print_prefix, auc, bucket_error, mae, rmse,
                          actual_ctr, predicted_ctr, copc, mean_predict_qvalue,
                          total_ins_num))

    def program_type_trans(self, prog_dir, prog_fn, is_text):
        return utils.program_type_trans(prog_dir, prog_fn, is_text)

    def draw_from_program_file(self, model_filename, is_text, output_dir,
                               output_filename):
        """draw program from file"""
        program = utils.load_program(model_filename, is_text)
        utils.graphviz(program.global_block(), output_dir, output_filename)

    def draw_from_program(self, program, output_dir, output_name):
        """draw Program"""
        utils.graphviz(program.global_block(), output_dir, output_name)

    def check_two_programs(self, config):
        train_prog = utils.load_program(config.train_prog_path,
                                        config.is_text_train_program)
        pruned_prog = utils.load_program(config.pruned_prog_path,
                                         config.is_text_pruned_program)
        if config.draw:
            pruned_dir = os.path.dirname(config.pruned_prog_path)
            self.draw_from_program(pruned_prog, pruned_dir,
                                   config.draw_out_name)
        res = utils.check_pruned_program_vars(train_prog, pruned_prog)
        if res:
            _logger.info("check_programs succeed.")
        else:
            _logger.info(
                "check_programs failed. pruned program and train program not match!"
            )
        return res

    def check_vars_and_dump(self, config):
        _logger.info("start check_vars_and_dump.")
        results = utils.check_saved_vars_try_dump(
            config.dump_model_dir, config.dump_program_filename,
            config.is_text_dump_program, config.feed_config,
            config.fetch_config, config.batch_size, config.save_params_filename)
        _logger.info("check_vars_and_dump succeed.")
        return results

    def parse_program_proto(self, prog_path, is_text, output_dir):
        """
        Parse program.proto into a more readable format. 
        This function will generate three files: 
        output_dir/vars_all.log,
        output_dir/vars_persistable.log,
        output_dir/ops.log.

        Args:
            prog_path(str): proto file path to be parsed.
            is_text(bool): proto file is human-readale format or not(binary).
            output_dir(str): output dir.

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              program_path = "./program.pbtxt"
              is_text = True
              output_dir = "/tmp/"
              fleet_util.parse_program_proto(program_path, is_text, output_dir)
        """
        program = utils.load_program(prog_path, is_text)
        utils.parse_program(program, output_dir)

    def _is_optimizer_op(self, op):
        return self.op_role_key in op.attr_names and \
                int(op.all_attrs()[self.op_role_key]) & int(OpRole.Optimize)

    def split_program_by_device(self, program):
        ops_list = []
        type_list = []
        pre = None
        type_cpu = "cpu"
        for op in program.global_block().ops:
            if self._is_optimizer_op(op):
                break
            if op.has_attr("op_device"):
                cur_attr = op.attr("op_device") if op.attr(
                    "op_device") != "" else type_cpu
                if pre is None or pre != cur_attr:
                    ops_list.append([])
                    type_list.append(cur_attr)
                ops_list[-1].append(op)
                pre = cur_attr
        l = len(type_list)
        i = 0
        type_heter = None
        while i < l:
            while i < l and type_list[i] == type_cpu:
                i += 1
            if i == l:
                break

            type_heter = type_list[i]
            i += 1
            start = i
            valid = True
            while i < l and type_list[i] != type_heter:
                if type_list[i] != type_cpu:
                    valid = False
                    break
                i += 1

            if i == l:
                break
            elif not valid:
                continue

            for j in range(start, i):
                for op in ops_list[j]:
                    op._set_attr("op_device", type_heter)
                type_list[j] = type_heter
                j += 1

        pre = None
        merged_ops_list = []
        merged_type_list = []
        for i in range(l):
            if pre is None or pre != type_list[i]:
                merged_ops_list.append([])
                merged_type_list.append(type_list[i])
            merged_ops_list[-1].extend(ops_list[i])
            pre = type_list[i]

        data_vars = set()
        for k in program.global_block().vars:
            var = program.global_block().var(k)
            if not var.persistable:
                data_vars.add(var.name)

        l = len(merged_ops_list)
        inputs_pre = set()
        outputs_pre = set()
        in_from_pre = [[] for i in range(l)]
        for i in range(l):
            inputs = set()
            outputs = set()
            for op in merged_ops_list[i]:
                for input in op.input_names:
                    for tmp in op.input(input):
                        if tmp not in outputs:
                            inputs.add(tmp)
                for output in op.output_names:
                    for tmp in op.output(output):
                        outputs.add(tmp)
            if i == 0:
                in_from_pre[i] = []
            elif i == 1:
                in_from_pre[i] = (outputs_pre | data_vars) & inputs
            else:
                in_from_pre[i] = outputs_pre & inputs
            inputs_pre = copy.deepcopy(inputs)
            outputs_pre = copy.deepcopy(outputs)

        l = len(in_from_pre)
        start_list = []
        end_list = []
        send_list = [[] for i in range(l)]
        sum = 0
        program_list = []
        for i in range(l):
            start_list.append(sum)
            end_list.append(sum + len(merged_ops_list[i]) - 1)
            sum += len(merged_ops_list[i])
            if i < l - 1:
                send_list[i].extend(list(in_from_pre[i + 1]))
            prog = program.clone()
            if merged_type_list[i] != type_cpu:
                prog = prog._prune_with_input(
                    list(in_from_pre[i]), list(send_list[i]))
                program_list.append(prog)
            else:
                program_list.append(prog)
        recv_list = [list(i) for i in in_from_pre]
        found = False
        heter_index = None
        for i in range(len(merged_type_list)):
            t = merged_type_list[i]
            if t != type_cpu:
                if found:
                    print("only one region of program can be heter")
                found = True
                heter_index = i
        if heter_index is None:
            print("warning: non heter program")
            return None
        else:
            return [start_list[heter_index], end_list[heter_index], send_list[heter_index], \
                    recv_list[heter_index], program_list[heter_index]]
