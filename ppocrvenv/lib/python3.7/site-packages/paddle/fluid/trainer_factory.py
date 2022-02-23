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
"""Defination of TrainerFactory."""

import threading
import time
import logging
import numpy as np
from paddle.fluid.log_helper import get_logger

local_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

from .trainer_desc import MultiTrainer, DistMultiTrainer, PipelineTrainer, HeterXpuTrainer, PSGPUTrainer, HeterPipelineTrainer
from .device_worker import Hogwild, DownpourSGD, Section, DownpourSGDOPT, HeterSection
from .framework import Variable
from multiprocessing import Process, Manager

__all__ = ["TrainerFactory", "FetchHandlerMonitor"]


class TrainerFactory(object):
    """
    Create trainer and device worker.
    If opt_info is not None, it will get configs from opt_info,
    otherwise create MultiTrainer and Hogwild.
    """

    def __init__(self):
        pass

    def _create_trainer(self, opt_info=None):
        trainer = None
        device_worker = None
        if not opt_info:
            # default is MultiTrainer + Hogwild
            trainer = MultiTrainer()
            device_worker = Hogwild()
            trainer._set_device_worker(device_worker)
        else:
            trainer_class = opt_info.get("trainer", "MultiTrainer")
            device_worker_class = opt_info.get("device_worker", "Hogwild")
            trainer = globals()[trainer_class]()
            device_worker = globals()[device_worker_class]()

            # for debug tools
            if opt_info is not None:
                if opt_info.get("trainers") is not None:
                    trainer._set_trainers(opt_info["trainers"])
                if opt_info.get("trainer_id") is not None:
                    trainer._set_trainer_id(opt_info["trainer_id"])
                if opt_info.get("dump_slot") is not None:
                    trainer._set_dump_slot(opt_info["dump_slot"])
                if opt_info.get("mpi_rank") is not None:
                    trainer._set_mpi_rank(opt_info["mpi_rank"])
                if opt_info.get("mpi_size") is not None:
                    trainer._set_mpi_size(opt_info["mpi_size"])
                if opt_info.get("dump_fields") is not None and len(
                        opt_info.get("dump_fields")) != 0:
                    trainer._set_dump_fields(opt_info["dump_fields"])
                if opt_info.get("dump_fields_path") is not None and len(
                        opt_info.get("dump_fields_path")) != 0:
                    trainer._set_dump_fields_path(opt_info["dump_fields_path"])
                if opt_info.get("dump_file_num") is not None:
                    trainer._set_dump_file_num(opt_info["dump_file_num"])
                if opt_info.get("dump_converter") is not None:
                    trainer._set_dump_converter(opt_info["dump_converter"])
                if opt_info.get("dump_param") is not None and len(
                        opt_info.get("dump_param")) != 0:
                    trainer._set_dump_param(opt_info["dump_param"])
                if opt_info.get("worker_places") is not None:
                    trainer._set_worker_places(opt_info["worker_places"])
                if opt_info.get("use_ps_gpu") is not None:
                    trainer._set_use_ps_gpu(opt_info["use_ps_gpu"])
                if opt_info.get("enable_random_dump") is not None:
                    trainer._set_enable_random_dump(opt_info[
                        "enable_random_dump"])
                if opt_info.get("dump_interval") is not None:
                    trainer._set_dump_interval(opt_info["dump_interval"])
                if opt_info.get("random_with_lineid") is not None:
                    trainer._set_random_with_lineid(opt_info[
                        "random_with_lineid"])

            if "fleet_desc" in opt_info:
                device_worker._set_fleet_desc(opt_info["fleet_desc"])
                trainer._set_fleet_desc(opt_info["fleet_desc"])
                if opt_info.get("use_cvm") is not None:
                    trainer._set_use_cvm(opt_info["use_cvm"])
                if opt_info.get("no_cvm") is not None:
                    trainer._set_no_cvm(opt_info["no_cvm"])
                if opt_info.get(
                        "scale_sparse_gradient_with_batch_size") is not None:
                    trainer._set_scale_sparse_grad_with_batch_size(opt_info[
                        "scale_sparse_gradient_with_batch_size"])
                if opt_info.get("scale_datanorm") is not None:
                    trainer._set_scale_datanorm(opt_info["scale_datanorm"])
                if opt_info.get("adjust_ins_weight") is not None:
                    trainer._set_adjust_ins_weight(opt_info[
                        "adjust_ins_weight"])
                if opt_info.get("copy_table") is not None:
                    trainer._set_copy_table_config(opt_info["copy_table"])
                if opt_info.get("check_nan_var_names") is not None:
                    trainer._set_check_nan_var_names(opt_info[
                        "check_nan_var_names"])
                if opt_info.get("loss_names") is not None:
                    trainer._set_loss_names(opt_info["loss_names"])
            trainer._set_device_worker(device_worker)
        return trainer


class FetchHandlerMonitor(object):
    """
    Defination of FetchHandlerMonitor class,
    it's for fetch handler.
    """

    def __init__(self, scope, handler):
        self.fetch_instance = handler
        self.fetch_thread = threading.Thread(
            target=self.handler_launch_func, args=(scope, self.fetch_instance))
        self.running_lock = threading.Lock()
        self.running = False

    def handler_launch_func(self, scope, handler):
        fetch_instance = handler
        period_secs = fetch_instance.period_secs
        var_name_to_key = {}
        for key in fetch_instance.var_dict:
            if isinstance(fetch_instance.var_dict[key], Variable):
                var_name_to_key[fetch_instance.var_dict[key].name] = key
            else:
                local_logger.warning("the value of {} is not a Variable".format(
                    key))
                var_name_to_key["None.var"] = key
        elapsed_secs = 0
        while True:
            self.running_lock.acquire()
            if self.running == False:
                break
            if elapsed_secs < period_secs:
                # TODO(guru4elephant): needs customized condition
                time.sleep(1)
                elapsed_secs += 1
            else:
                elapsed_secs = 0
                fetch_dict = {}
                for key in var_name_to_key:
                    var = scope.find_var(key)
                    fetch_dict[key] = var
                    if var == None:
                        local_logger.warning("{} value currently not available".
                                             format(var_name_to_key[key]))
                res_dict = {}
                for key in fetch_dict:
                    user_name = var_name_to_key[key]
                    if fetch_dict[key] == None:
                        res_dict[user_name] = None
                        continue
                    else:
                        res_dict[user_name] = fetch_dict[key].get_tensor()

                    lod = res_dict[user_name].lod()
                    if len(lod) > 0:
                        raise RuntimeError("Some of your fetched tensors \
                                            hold LoD information. \
                                            They can not be completely cast \
                                            to Python ndarray. We can \
                                            not return LoDTensor itself directly, \
                                            please choose another targets")
                    if res_dict[user_name]._is_initialized():
                        res_dict[user_name] = np.array(res_dict[user_name])
                    else:
                        res_dict[user_name] = None
                fetch_instance.handler(res_dict)
            self.running_lock.release()

    def start(self):
        """
        start monitor,
        it will start a monitor thread.
        """
        self.running_lock.acquire()
        self.running = True
        self.running_lock.release()
        self.fetch_thread.setDaemon(True)
        self.fetch_thread.start()

    def stop(self):
        self.running_lock.acquire()
        self.running = False
        self.running_lock.release()
