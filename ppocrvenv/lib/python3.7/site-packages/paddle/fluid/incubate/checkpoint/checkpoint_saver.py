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

from ...compiler import CompiledProgram


class SerializableBase(object):
    def serialize(self, path):
        raise NotImplementedError

    def deserialize(self, path):
        raise NotImplementedError


class PaddleModel(SerializableBase):
    def __init__(self, exe, program):
        self._exe = exe
        self._origin_program = program
        self._program = program
        if isinstance(program, CompiledProgram):
            self._program = program._program

        self._file_name = "_paddle_fleet_param__"

    def serialize(self, path):
        from ...io import save_persistables
        save_persistables(
            executor=self._exe,
            dirname=path,
            main_program=self._program,
            filename=self._file_name)

    def deserialize(self, path):
        from ...io import load_persistables
        load_persistables(
            executor=self._exe,
            dirname=path,
            main_program=self._program,
            filename=self._file_name)


class CheckpointSaver(object):
    def __init__(self, fs):
        self._fs = fs
        self._checkpoint_prefix = "__paddle_checkpoint__"

    def save_checkpoint(self,
                        path,
                        slists,
                        trainer_id=None,
                        local_cache_path=".cache"):
        """
        Serialize objects in slists to path
        Return really saved path and checkpoint_no
        """
        if not self._fs.is_exist(path):
            self._fs.mkdirs(path)
        else:
            assert self._fs.is_dir(path), "path:{} must be a directory".format(
                path)

        max_no = self._get_last_checkpoint_no(path)
        if max_no < 0:
            max_no = -1
        max_no += 1

        real_path = "{}/{}.{}".format(path, self._checkpoint_prefix, max_no)
        tmp_path = "{}.tmp".format(real_path)
        saved_path = tmp_path

        from paddle.distributed.fleet.utils.fs import LocalFS
        local_fs = LocalFS()

        cache_path = None
        if self._fs.need_upload_download():
            cache_path = "{}/{}.{}.saved_cache".format(
                local_cache_path, self._checkpoint_prefix, max_no)

            if trainer_id is not None:
                cache_path = "{}.{}".format(cache_path, trainer_id)

            if not local_fs.is_exist(cache_path):
                local_fs.mkdirs(cache_path)
            else:
                assert local_fs.is_dir(cache_path), \
                    "cache path:{} must be a directory".format(cache_path)

            saved_path = cache_path

        for s in slists:
            s.serialize(saved_path)

        if self._fs.need_upload_download():
            self._fs.delete(tmp_path)
            self._fs.upload(cache_path, tmp_path)
            local_fs.delete(cache_path)
        self._fs.mv(tmp_path, real_path)

        return real_path, max_no

    def load_checkpoint(self,
                        path,
                        slists,
                        trainer_id,
                        local_cache_path=".cache",
                        checkpoint_no=None,
                        ignore_empty=True):
        """
        Deserialize objects in slists from path
        Return really load path
        """
        if checkpoint_no is None:
            max_no = self._get_last_checkpoint_no(path)

            if not ignore_empty:
                assert max_no >= 0, "Can't find checkpoint"

            if max_no < 0:
                return None

            checkpoint_no = max_no
        else:
            assert isinstance(checkpoint_no, int)
            assert checkpoint_no >= 0

        from paddle.distributed.fleet.utils.fs import LocalFS
        local_fs = LocalFS()
        if self._fs.need_upload_download():
            cache_path = "{}/{}.{}.load_cache".format(
                local_cache_path, self._checkpoint_prefix, checkpoint_no)

            if trainer_id is not None:
                cache_path = "{}.{}".format(cache_path, trainer_id)

            if not local_fs.is_exist(local_cache_path):
                local_fs.mkdirs(local_cache_path)
            if local_fs.is_exist(cache_path):
                local_fs.delete(cache_path)

        real_path = "{}/{}.{}".format(path, self._checkpoint_prefix,
                                      checkpoint_no)
        load_path = real_path
        if self._fs.need_upload_download():
            self._fs.download(real_path, cache_path)
            load_path = cache_path

        for s in slists:
            s.deserialize(load_path)

        if self._fs.need_upload_download() and cache_path:
            local_fs.delete(cache_path)

        return real_path

    def get_checkpoint_no(self, root_path):
        a = []
        dirs = self._fs.list_dirs(root_path)
        for d in dirs:
            g = d.split(".")
            if len(g) != 2:
                continue

            if g[0] != self._checkpoint_prefix:
                continue

            try:
                n = int(g[1])
                a.append(n)
            except:
                continue

        a.sort()
        return a

    def _get_last_checkpoint_no(self, root_path):
        """
        only get the first depth
        """
        a = self.get_checkpoint_no(root_path)
        if len(a) > 0:
            return a[-1]

        return -1

    def clean_redundant_checkpoints(self, root_path, reserved=[]):
        max_no = self._get_last_checkpoint_no(root_path)
        if max_no < 0:
            return

        s = set(reserved)
        if len(s) == 0:
            s.add(max_no)

        dirs = self._fs.list_dirs(root_path)
        for d in dirs:
            g = d.split(".")
            if len(g) != 2:
                continue

            if g[0] != self._checkpoint_prefix:
                continue

            try:
                n = int(g[1])
                if n not in s:
                    path = "{}/{}.{}".format(root_path, self._checkpoint_prefix,
                                             n)
                    self._fs.delete(path)
            except Exception as e:
                print(e)
                continue
