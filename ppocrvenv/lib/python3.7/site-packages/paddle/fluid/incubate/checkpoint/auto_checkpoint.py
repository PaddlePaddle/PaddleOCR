# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import logging
import hashlib
import json
import os
import six
import time
import collections
from threading import Thread, current_thread
from contextlib import contextmanager

from paddle.fluid import unique_name, compiler
from .checkpoint_saver import SerializableBase, CheckpointSaver, PaddleModel
from paddle.fluid.framework import in_dygraph_mode, Program

g_train_epoch_range = None
g_checker = None

logger = None

generator = unique_name.UniqueNameGenerator()

CONST_CHECKPOINT = "checkpoint"
CONST_MEMORYINIT = "memory_init"

# auto checkpoint by dataloader event.
CONST_DACP_TYPE = "dacp"
# auto checkpoint by loop range.
CONST_ACP_TYPE = "acp"
g_acp_type = None
g_program_attr = {}  # program_name->can_be_auto_checkpoint


def _get_logger(log_level, name="auto_checkpoint"):
    global logger
    if logger != None:
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    return logger


def _thread_checker():
    assert current_thread().name == "MainThread", \
        "auto checkpoint must run under main thread"


class AutoCheckpointChecker(object):
    def __init__(self):
        self._run_env = None
        self._platform = None
        self._job_id = None
        self._hdfs_home = None
        self._hdfs_name = None
        self._hdfs_ugi = None
        self._hdfs_checkpoint_path = None
        self._trainer_id = None
        self._ce_test = None

        self._run_env = os.getenv("PADDLE_RUNNING_ENV")
        if self._run_env != "PADDLE_EDL_AUTO_CHECKPOINT":
            return

        try:
            self._platform = os.environ["PADDLE_RUNNING_PLATFORM"]
            self._job_id = os.environ["PADDLE_JOB_ID"]
            self._hdfs_home = os.environ["PADDLE_EDL_HDFS_HOME"]
            self._hdfs_name = os.environ["PADDLE_EDL_HDFS_NAME"]
            self._hdfs_ugi = os.environ["PADDLE_EDL_HDFS_UGI"]
            self._hdfs_checkpoint_path = os.environ[
                "PADDLE_EDL_HDFS_CHECKPOINT_PATH"]
            self._trainer_id = int(os.environ["PADDLE_TRAINER_ID"])

            self._ce_test = int(os.getenv("PADDLE_EDL_ONLY_FOR_CE_TEST", "0"))
            self._fs_cache = os.getenv("PADDLE_EDL_FS_CACHE", ".cache")

            self._save_checkpoint_inter = int(
                os.getenv("PADDLE_EDL_SAVE_CHECKPOINT_INTER", "900"))  # s

            if not self._ce_test:
                assert len(self._hdfs_home) > 3 and \
                    len(self._hdfs_name) > 6 and \
                    len(self._hdfs_ugi) > 3 and \
                    len(self._hdfs_checkpoint_path) > 0, "hdfs environ must set"
            else:
                assert len(self._hdfs_home) > 3 and \
                    len(self._hdfs_checkpoint_path) > 0, "hdfs environ must set"

        except Exception as e:
            logger.fatal("exception:{}".format(e))
            sys.exit(1)

    def get_range_checkpoint_path(self, name):
        return "{}/{}/range/{}".format(self.hdfs_checkpoint_path, self.job_id,
                                       name)

    def get_exe_checkpoint_path(self, name):
        return "{}/{}/exe/{}".format(self.hdfs_checkpoint_path, self.job_id,
                                     name)

    def get_job_path(self):
        return "{}/{}".format(self.hdfs_checkpoint_path, self.job_id)

    @property
    def save_checkpoint_inter(self):
        return self._save_checkpoint_inter

    def valid(self):
        if in_dygraph_mode():
            return False

        return self._run_env is not None and \
            self._platform is not None and \
            self._job_id is not None and \
            self._hdfs_home is not None and \
            self._hdfs_name is not None and \
            self._hdfs_ugi is not None and \
            self._hdfs_checkpoint_path is not None and \
            self._trainer_id is not None

    def __str__(self):
        return "run_env:{} platform:{} job_id:{} \
            hdfs_home:{} hdfs_name:{} hdfs_ugi:{} \
            hdfs_checkpoint_path:{} trainer_id:{} ce_test".format(
            self._run_env, self._platform, self._hdfs_home, self._hdfs_name,
            self._hdfs_ugi, self._hdfs_checkpoint_path, self._trainer_id,
            self._ce_test)

    @property
    def trainer_id(self):
        return self._trainer_id

    @property
    def run_env(self):
        return self._run_env

    @property
    def platform(self):
        return self._platform

    @property
    def job_id(self):
        return self._job_id

    @property
    def hdfs_home(self):
        return self._hdfs_home

    @property
    def hdfs_name(self):
        return self._hdfs_name

    @property
    def ce_test(self):
        return self._ce_test

    @property
    def hdfs_ugi(self):
        return self._hdfs_ugi

    @property
    def hdfs_checkpoint_path(self):
        return self._hdfs_checkpoint_path

    @staticmethod
    def generate_range_name():
        return generator("_range_")


class ExeTrainStatus(SerializableBase):
    def __init__(self):
        self._epoch_no = -1  # start epoch_no
        self._hash_key = None
        self._key = None
        self._checkpoint_path = None
        self._checkpoint_no = None
        self._restored_from = None
        self._exe = None
        self._program = None
        self._exe_name = None
        self._program_name = None

        self._file_name = "exe_train_status"

    def __eq__(self, t):
        return self._epoch_no == t._epoch_no and \
            self._hash_key == t._hash_key and \
            self._key == t._key and \
            self._checkpoint_path == t._checkpoint_path and \
            self._checkpoint_no == t._checkpoint_no and \
            self._exe_name == t._exe_name and \
            self._program_name == t._program_name

    def __ne__(self, t):
        return not self == t

    def serialize(self, path):
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'w') as f:
            s = self._serialize()
            f.write(s)

    def _serialize(self, pop_keys=["restored_from"]):
        d = self._to_dict()
        for k in pop_keys:
            d.pop(k, None)
        return json.dumps(d)

    def deserialize(self, path):
        d = None
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'r') as f:
            s = f.read()
            self._deserialize(s)

    def _deserialize(self, s):
        d = json.loads(s)
        self._epoch_no = d["epoch_no"]
        self._key = d["key"]
        self._hash_key = d["hash_key"]
        self._checkpoint_path = d["checkpoint_path"]
        self._checkpoint_no = d["checkpoint_no"]
        self._exe_name = d["exe_name"]
        self._program_name = d["program_name"]

    def _to_dict(self):
        return {
            "epoch_no": self._epoch_no,
            "key": self._key,
            "hash_key": self._hash_key,
            "checkpoint_path": self._checkpoint_path,
            "restored_from": self._restored_from,
            "exe_name": self._exe_name,
            "program_name": self._program_name,
            "checkpoint_no": self._checkpoint_no
        }

    def __str__(self):
        return self._serialize([])


class TrainEpochRange(SerializableBase):
    def __init__(self,
                 max_epoch_num,
                 name,
                 checkpoint_inter=None,
                 restored=True):
        self._max_epoch_num = max_epoch_num
        self._epoch_no = -1  # current epoch_no
        self._name = name
        self._restored_from = None
        self._exe_status = {}
        self._flag_generated = False

        self._checker = g_checker
        if checkpoint_inter is not None:
            self._save_checkpoint_inter = checkpoint_inter
        else:
            self._save_checkpoint_inter = self._checker.save_checkpoint_inter
        assert self._save_checkpoint_inter >= 0, "checkpointer:{} must >=0".format(
            self._save_checkpoint_inter)
        self._last_checkpoint_time = time.time()

        self._load_cp_nos = None
        self._checkpoint_epoch_no = None

        if not self._checker.valid():
            return

        self._file_name = "range_train_status"

        if not restored:
            return

        self._checkpoint_path = self._checker.get_range_checkpoint_path(name)

        config = {
            "fs.default.name": self._checker.hdfs_name,
            "hadoop.job.ugi": self._checker.hdfs_ugi
        }

        if self._checker.ce_test:
            config = None

        from paddle.distributed.fleet.utils.fs import HDFSClient
        self._hdfs = HDFSClient(self._checker.hdfs_home, config)

        self._cper = CheckpointSaver(self._hdfs)

        _thread_checker()

        self._get_last_valid_checkpoint()

    def _look_for_valid(self, cp_nos):
        cps = []
        epoch_no = -1
        for i in cp_nos[::-1]:
            t = TrainEpochRange(self._max_epoch_num, self.name, restored=False)
            self._cper.load_checkpoint(
                self._checkpoint_path, [t],
                self._checker.trainer_id,
                checkpoint_no=i,
                local_cache_path=self._checker._fs_cache)
            cps.append(t)
            logger.debug("look for valid:{} t:{}".format(i, t._serialize()))
            if epoch_no < 0:
                epoch_no = t._epoch_no
            else:
                if epoch_no - t._epoch_no >= 1:
                    return t, i
        return None, None

    def _get_last_valid_checkpoint(self):
        self._load_cp_nos = self._cper.get_checkpoint_no(self._checkpoint_path)
        logger.info("find checkpoint nos:{}".format(self._load_cp_nos))

        if len(self._load_cp_nos) < 1:
            self._restored_from = CONST_MEMORYINIT
            return

        if g_acp_type == CONST_ACP_TYPE:
            # get the last one
            self._cper.load_checkpoint(
                self._checkpoint_path, [self],
                self._checker.trainer_id,
                local_cache_path=self._checker._fs_cache)
            self._restored_from = CONST_CHECKPOINT
            self._checkpoint_epoch_no = self._epoch_no

            logger.info("load tain_epoch_range checkpoint:{}".format(
                self._serialize()))

        elif g_acp_type == CONST_DACP_TYPE:
            t, i = self._look_for_valid(self._load_cp_nos)
            if t is None:
                self._restored_from = CONST_MEMORYINIT
                return

            self._cper.load_checkpoint(
                self._checkpoint_path, [self],
                self._checker.trainer_id,
                checkpoint_no=i,
                local_cache_path=self._checker._fs_cache)

            self._restored_from = CONST_CHECKPOINT
            self._checkpoint_epoch_no = self._epoch_no
            logger.info("load tain_epoch_range checkpoint:{}".format(
                self._serialize()))
        else:
            assert False, "not supported acp_type:{}".format(g_acp_type)

    def _to_dict(self):
        d = {
            "max_epoch_num": self._max_epoch_num,
            "epoch_no": self._epoch_no,
            "name": self._name,
            "checkpoint_path": self._checkpoint_path,
            "restored_from": self._restored_from,
            "checkpoint_epoch_no": self._checkpoint_epoch_no
        }
        return d

    def __str__(self):
        return self._serialize([])

    @property
    def name(self):
        return self._name

    def serialize(self, path):
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'w') as f:
            s = self._serialize()
            f.write(s)

    def _serialize(self, pop_keys=["restored_from", "checkpoint_epoch_no"]):
        # self
        d = self._to_dict()
        for k in pop_keys:
            d.pop(k, None)

        # registerd exes
        d["exe_status"] = {}
        e = d["exe_status"]
        for k, t in six.iteritems(self._exe_status):
            e[t._key] = t._serialize()
        return json.dumps(d)

    @property
    def restored_from(self):
        return self._restored_from

    def deserialize(self, path):
        d = None
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'r') as f:
            d = json.load(f)

        # self
        self._max_epoch_num = d["max_epoch_num"]
        self._epoch_no = d["epoch_no"]
        self._name = d["name"]
        self._checkpoint_path = d["checkpoint_path"]

        # exes status
        e = d["exe_status"]
        for k, v in six.iteritems(e):
            t = ExeTrainStatus()
            t._deserialize(v)
            self._exe_status[k] = t

    def next(self):
        _thread_checker()

        if self._max_epoch_num < 0:
            self._max_epoch_num = sys.maxint

        assert self._epoch_no >= -1, "self._epoch_no:{} must >=-1".format(
            self._epoch_no)

        self._last_checkpoint_time = time.time()
        start = self._epoch_no + 1
        logger.info("started epoch_no:{} max_epoch_num:{}".format(
            start, self._max_epoch_num))

        for i in range(start, self._max_epoch_num):
            self._epoch_no = i
            yield i

            self.save_checkpoint()

    def get(self):
        return self._epoch_no

    def save_checkpoint(self):
        # not save last one because exe and program can't be restored.
        if self._checker.trainer_id == 0:

            if time.time() - self._last_checkpoint_time >= \
                    self._save_checkpoint_inter:
                if g_acp_type == CONST_ACP_TYPE:
                    # not save the last one
                    if self._max_epoch_num > 0 and self._epoch_no != self._max_epoch_num - 1:
                        self._save_checkpoint()
                elif g_acp_type == CONST_DACP_TYPE:
                    self._save_checkpoint()
                else:
                    assert False, "not supported acp_type:{}".format(g_acp_type)
            self._last_checkpoint_time = time.time()

    def _save_checkpoint(self):
        """
        status => /jobid/xxx_range_xx/range/
        model =>                       /exe/
        """
        if not self._checker.valid():
            return

        e = self._exe_status
        for k, t in six.iteritems(self._exe_status):
            m = PaddleModel(t._exe, t._program)
            p = self._checker.get_exe_checkpoint_path(t._hash_key)
            t._epoch_no = self.get()
            path, checkpoint_no = self._cper.save_checkpoint(
                p, [m],
                self._checker.trainer_id,
                local_cache_path=self._checker._fs_cache)
            # index info
            t._checkpoint_path = path
            t._checkpoint_no = checkpoint_no

            e[t._key] = t

            logger.debug("save executor checkpoint:{}".format(t._serialize()))

        if len(self._exe_status) > 0:
            self._cper.save_checkpoint(
                self._checkpoint_path, [self],
                local_cache_path=self._checker._fs_cache)
            logger.info("save train_epoch_range checkpoint:{}".format(
                self._serialize()))

            self._generate_flag()

    def _generate_flag(self):
        if self._flag_generated:
            return

        name = "can_be_auto_checkpoint.flag"
        path = self._checker.get_job_path() + "/" + name
        logger.info("this job can_be_auto_checkpoint")
        self._hdfs.mkdirs(self._checker.get_job_path())
        self._hdfs.touch(path, exist_ok=True)

        self._flag_generated = True


def _get_train_epoch_range():
    return g_train_epoch_range


def _check_program_oprole(program):
    global_block = program.global_block()
    has_backward = False
    has_opt = False
    for idx, op in enumerate(global_block.ops):
        if op._is_backward_op():
            has_backward = True

        if op._is_optimize_op():
            has_opt = True

        if has_backward and has_opt:
            return True

    return False


def _can_auto_checkpoint(prog):
    if not isinstance(prog, compiler.CompiledProgram) and \
            not isinstance(prog, Program):
        return False

    if isinstance(prog, compiler.CompiledProgram):
        if prog._program is None or \
                prog._program._is_distributed:
            return False
    else:
        if prog._is_distributed:
            return False

    program = _get_valid_program(prog)

    if program._auto_checkpoint_name in g_program_attr:
        if not g_program_attr[program._auto_checkpoint_name]:
            return False
    else:
        ret = False
        if isinstance(program, compiler.CompiledProgram):
            ret = _check_program_oprole(program._program)
        else:
            ret = _check_program_oprole(program)

        g_program_attr[program._auto_checkpoint_name] = ret
        if not ret:
            logger.debug("program {} need't to auto checkpoint".format(
                program._auto_checkpoint_name))
            return False

    return g_checker.valid() and g_train_epoch_range is not None


def _get_running_key(exe_name, program_name):
    return "{}_{}".format(exe_name, program_name)


def _get_checker():
    _get_logger(20)
    global g_checker
    if g_checker is None:
        g_checker = AutoCheckpointChecker()

    return g_checker


def _normal_yield(max_epoch_num):
    if max_epoch_num < 0:
        max_epoch_num = sys.maxint
    for i in range(0, max_epoch_num):
        yield i

    return


def train_epoch_range(max_epoch_num, save_checkpoint_inter=None):
    global g_acp_type
    if not _get_checker().valid():
        logger.warning(
            "auto checkpoint will take effect  automaticly on PaddleCloud")
        for i in _normal_yield(max_epoch_num):
            yield i

        return

    if g_acp_type == CONST_DACP_TYPE:
        for i in _normal_yield(max_epoch_num):
            yield i

        return

    g_acp_type = CONST_ACP_TYPE
    logger.info("acp_type:{}".format(g_acp_type))

    global g_train_epoch_range
    try:
        g_train_epoch_range = TrainEpochRange(
            max_epoch_num,
            g_checker.generate_range_name(),
            checkpoint_inter=save_checkpoint_inter)

        for i in g_train_epoch_range.next():
            yield i
    finally:
        g_train_epoch_range = None


def _get_valid_program(prog):
    if isinstance(prog, compiler.CompiledProgram):
        return prog._program

    return prog


def _auto_checkpoint(exe, prog):
    _get_checker()

    assert exe._auto_checkpoint_name != None
    if not _can_auto_checkpoint(prog):
        return

    program = _get_valid_program(prog)
    assert program._auto_checkpoint_name != None

    exe_status = g_train_epoch_range._exe_status
    key = _get_running_key(exe._auto_checkpoint_name,
                           program._auto_checkpoint_name)

    if g_train_epoch_range.restored_from == CONST_CHECKPOINT:
        assert key in exe_status, "when restored key:{} must be in train_epoch_range:{}".format(
            key, g_train_epoch_range)

    t = None
    if key in exe_status:
        t = exe_status[key]
        if t._restored_from is None:
            a = CheckpointSaver(g_train_epoch_range._hdfs)
            m = PaddleModel(exe, program)
            a.load_checkpoint(
                g_checker.get_exe_checkpoint_path(key), [m],
                trainer_id=g_checker.trainer_id,
                checkpoint_no=t._checkpoint_no,
                local_cache_path=g_checker._fs_cache)
            t._restored_from = CONST_CHECKPOINT
            logger.info("load executor checkpoint {}".format(t))
        t._exe = exe
        t._program = program
        t._epoch_no = g_train_epoch_range.get()
    else:
        t = ExeTrainStatus()
        t._epoch_no = g_train_epoch_range.get()
        t._hash_key = key
        t._key = key
        t._restored_from = CONST_MEMORYINIT
        t._exe = exe
        t._program = program
        t._exe_name = exe._auto_checkpoint_name
        t._program_name = program._auto_checkpoint_name

        # register this <exe,program,io>
        exe_status[key] = t

        logger.info("not found checkpoint, so train from epoch 0")

    _thread_checker()
