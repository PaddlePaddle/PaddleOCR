#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""HDFS Utils."""

import os
import sys
import subprocess
import multiprocessing
from datetime import datetime

import re
import copy
import errno
import time
import logging
import six
#from . import fs
from paddle.distributed.fleet.utils.fs import FS, LocalFS, FSFileExistsError, FSFileNotExistsError, ExecuteError, FSTimeOut, FSShellCmdAborted
from paddle.fluid import core
import functools

import shutil

__all__ = ["HDFSClient"]


def _handle_errors(max_time_out=None):
    def decorator(f):
        @functools.wraps(f)
        def handler(*args, **kwargs):
            o = args[0]
            time_out = max_time_out
            if time_out is None:
                time_out = float(o._time_out) / 1000.0
            else:
                time_out /= 1000.0
            inter = float(o._sleep_inter) / 1000.0

            start = time.time()
            last_print_time = start
            while True:
                try:
                    return f(*args, **kwargs)
                #important: only ExecuteError need to retry
                except ExecuteError as e:
                    if time.time() - start >= time_out:
                        raise FSTimeOut("args:{} timeout:{}".format(
                            args, time.time() - start))

                    time.sleep(inter)

                if time.time() - last_print_time > 30:
                    print("hadoop operator timeout:args:{} timeout:{}".format(
                        args, time.time() - start))
                    last_print_time = time.time()

        return handler

    return decorator


class HDFSClient(FS):
    def __init__(
            self,
            hadoop_home,
            configs,
            time_out=5 * 60 * 1000,  #ms
            sleep_inter=1000):  #ms
        # Raise exception if JAVA_HOME not exists.

        self.pre_commands = []
        hadoop_bin = '%s/bin/hadoop' % hadoop_home
        self.pre_commands.append(hadoop_bin)
        dfs = 'fs'
        self.pre_commands.append(dfs)

        if configs:
            for k, v in six.iteritems(configs):
                config_command = '-D%s=%s' % (k, v)
                self.pre_commands.append(config_command)

        self._time_out = time_out
        self._sleep_inter = sleep_inter
        self._base_cmd = " ".join(self.pre_commands)
        self._bd_err_re = re.compile(
            r'\s?responseErrorMsg\s?\:.*, errorCode\:\s?[0-9]+, path\:')

    def _run_cmd(self, cmd, redirect_stderr=False):
        exe_cmd = "{} -{}".format(self._base_cmd, cmd)
        ret, output = core.shell_execute_cmd(exe_cmd, 0, 0, redirect_stderr)
        ret = int(ret)
        if ret == 134:
            raise FSShellCmdAborted(cmd)
        return ret, output.splitlines()

    @_handle_errors()
    def list_dirs(self, fs_path):
        if not self.is_exist(fs_path):
            return []

        dirs, files = self._ls_dir(fs_path)
        return dirs

    @_handle_errors()
    def ls_dir(self, fs_path):
        """	
        list directory under fs_path, and only give the pure name, not include the fs_path	
        """
        if not self.is_exist(fs_path):
            return [], []

        return self._ls_dir(fs_path)

    def _ls_dir(self, fs_path):
        cmd = "ls {}".format(fs_path)
        ret, lines = self._run_cmd(cmd)

        if ret != 0:
            raise ExecuteError(cmd)

        dirs = []
        files = []
        for line in lines:
            arr = line.split()
            if len(arr) != 8:
                continue

            p = os.path.basename(arr[7])
            if arr[0][0] == 'd':
                dirs.append(p)
            else:
                files.append(p)

        return dirs, files

    def _test_match(self, lines):
        for l in lines:
            m = self._bd_err_re.match(l)
            if m != None:
                return m

        return None

    @_handle_errors()
    def is_dir(self, fs_path):
        if not self.is_exist(fs_path):
            return False

        return self._is_dir(fs_path)

    def _is_dir(self, fs_path):
        cmd = "test -d {}".format(fs_path, redirect_stderr=True)
        ret, lines = self._run_cmd(cmd)
        if ret:
            # other error
            if self._test_match(lines):
                raise ExecuteError(cmd)

            return False

        return True

    def is_file(self, fs_path):
        if not self.is_exist(fs_path):
            return False

        return not self._is_dir(fs_path)

    @_handle_errors()
    def is_exist(self, fs_path):
        cmd = "ls {} ".format(fs_path)
        ret, out = self._run_cmd(cmd, redirect_stderr=True)
        if ret != 0:
            for l in out:
                if "No such file or directory" in l:
                    return False
            raise ExecuteError(cmd)

        return True

    # can't retry
    def upload(self, local_path, fs_path):
        if self.is_exist(fs_path):
            raise FSFileExistsError("{} exists".format(fs_path))

        local = LocalFS()
        if not local.is_exist(local_path):
            raise FSFileNotExistsError("{} not exists".format(local_path))

        return self._try_upload(local_path, fs_path)

    @_handle_errors()
    def _try_upload(self, local_path, fs_path):
        cmd = "put {} {}".format(local_path, fs_path)
        ret = 0
        try:
            ret, lines = self._run_cmd(cmd)
            if ret != 0:
                raise ExecuteError(cmd)
        except Exception as e:
            self.delete(fs_path)
            raise e

    # can't retry
    def download(self, fs_path, local_path):
        if self.is_exist(local_path):
            raise FSFileExistsError("{} exists".format(local_path))

        if not self.is_exist(fs_path):
            raise FSFileNotExistsError("{} not exits".format(fs_path))

        return self._try_download(fs_path, local_path)

    @_handle_errors()
    def _try_download(self, fs_path, local_path):
        cmd = "get {} {}".format(fs_path, local_path)
        ret = 0
        try:
            ret, lines = self._run_cmd(cmd)
            if ret != 0:
                raise ExecuteError(cmd)
        except Exception as e:
            local_fs = LocalFS()
            local_fs.delete(local_path)
            raise e

    @_handle_errors()
    def mkdirs(self, fs_path):
        if self.is_exist(fs_path):
            return

        out_hdfs = False

        cmd = "mkdir {} ".format(fs_path)
        ret, out = self._run_cmd(cmd, redirect_stderr=True)
        if ret != 0:
            for l in out:
                if "No such file or directory" in l:
                    out_hdfs = True
                    break
            if not out_hdfs:
                raise ExecuteError(cmd)

        if out_hdfs and not self.is_exist(fs_path):
            cmd = "mkdir -p {}".format(fs_path)
            ret, lines = self._run_cmd(cmd)
            if ret != 0:
                raise ExecuteError(cmd)

    def mv(self, fs_src_path, fs_dst_path, overwrite=False, test_exists=True):
        if overwrite and self.is_exist(fs_dst_path):
            self.delete(fs_dst_path)

        if test_exists:
            if not self.is_exist(fs_src_path):
                raise FSFileNotExistsError("{} is not exists".format(
                    fs_src_path))

            if self.is_exist(fs_dst_path):
                raise FSFileExistsError("{} exists already".format(fs_dst_path))

        return self._try_mv(fs_src_path, fs_dst_path)

    @_handle_errors()
    def _try_mv(self, fs_src_path, fs_dst_path):
        cmd = "mv {} {}".format(fs_src_path, fs_dst_path)
        ret = 0
        try:
            ret, _ = self._run_cmd(cmd)
            if ret != 0:
                raise ExecuteError(cmd)
        except Exception as e:
            if not self.is_exist(fs_src_path) and \
                    self.is_exist(fs_dst_path):
                return
            raise e

    def _rmr(self, fs_path):
        cmd = "rmr {}".format(fs_path)
        ret, _ = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError(cmd)

    def _rm(self, fs_path):
        cmd = "rm {}".format(fs_path)
        ret, _ = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError(cmd)

    @_handle_errors()
    def delete(self, fs_path):
        if not self.is_exist(fs_path):
            return

        is_dir = self._is_dir(fs_path)
        if is_dir:
            return self._rmr(fs_path)

        return self._rm(fs_path)

    def touch(self, fs_path, exist_ok=True):
        if self.is_exist(fs_path):
            if exist_ok:
                return
            raise FSFileExistsError

        return self._touchz(fs_path)

    @_handle_errors()
    def _touchz(self, fs_path):
        cmd = "touchz {}".format(fs_path)
        ret, _ = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

    def need_upload_download(self):
        return True
