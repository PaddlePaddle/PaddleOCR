# Copyright (c) 2020 VisualDL Authors. All Rights Reserve.
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

import collections
from functools import partial  # noqa: F401
from visualdl.io import bfile
from visualdl.component import components
from visualdl.reader.record_reader import RecordReader
from visualdl.server.data_manager import default_data_manager
from visualdl.proto import record_pb2
from visualdl.utils.string_util import decode_tag, encode_tag


def is_VDLRecord_file(path, check=False):
    """Determine whether it is a VDL log file according to the file name.

    File name of a VDL log file must contain `vdlrecords`.

    Args:
        path: File name to determine.
        check: Check file is valid or not.

    Returns:
        True if the file is a VDL log file, otherwise false.
    """
    if "vdlrecords" not in path:
        return False
    if check:
        _reader = RecordReader(filepath=path)
        meta_data = _reader.get_next()
        record = record_pb2.Record()
        record.ParseFromString(meta_data)
        if 'meta_data_tag' != record.values[0].tag:
            return False
    return True


class LogReader(object):
    """Log reader to read vdl log, support for frontend api in lib.py.

    """

    def __init__(self, logdir='', file_path=''):
        """Instance of LogReader

        Args:
            logdir: The dir include vdl log files, multiple subfolders allowed.
        """
        if isinstance(logdir, str):
            self.dir = [logdir]
        else:
            self.dir = logdir

        self.reader = None
        self.readers = {}
        self.walks = None
        self._tags = {}
        self.name2tags = {}
        self.tags2name = {}

        self.file_readers = {}

        # {'run': {'scalar': {'tag1': data, 'tag2': data}}}
        self._log_datas = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

        if file_path:
            self._log_data = collections.defaultdict(lambda: collections.defaultdict(list))
            self.get_file_reader(file_path=file_path)
            remain = self.get_remain()
            self.read_log_data(remain=remain)
            components_name = components.keys()
            for name in components_name:
                exec("self.get_%s=partial(self.get_data, '%s')" % (name, name))
        elif logdir:
            self.data_manager = default_data_manager
            self.load_new_data(update=True)
            self._a_tags = {}

            self._model = ""

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_path):
        self._model = model_path
        with bfile.BFile(model_path, 'rb') as bfp:
            if not bfp.isfile(model_path):
                print("Model path %s should be file path, please check this path." % model_path)
            else:
                if bfile.exists(model_path):
                    self._model = model_path
                else:
                    print("Model path %s is invalid, please check this path." % model_path)

    @property
    def logdir(self):
        return self.dir

    def parsing_from_proto(self, component, proto_datas):
        data = []
        if 'scalar' == component:
            for item in proto_datas:
                data.append([item.id, item.tag, item.timestamp, item.value])
            return data
        return proto_datas

    def _get_log_tags(self):
        component_keys = self._log_data.keys()
        log_tags = {}
        for key in component_keys:
            _tags = list(self._log_data[key].keys())
            tags = list(map(lambda x: encode_tag(x), _tags))
            log_tags[key] = tags

        return log_tags

    def get_log_data(self, component, run, tag):
        if (run in self._log_datas.keys() and
                component in self._log_datas[run].keys() and
                tag in self._log_datas[run][component].keys()):
            return self._log_datas[run][component][tag]
        else:
            file_path = bfile.join(run, self.walks[run])
            reader = self._get_file_reader(file_path=file_path, update=False)
            remain = self.get_remain(reader=reader)
            data = self.read_log_data(remain=remain, update=False)[component][tag]
            data = self.parsing_from_proto(component, data)
            self._log_datas[run][component][tag] = data
            return data

    def get_tags(self):
        return self._get_log_tags()

    def get_data(self, component, tag):
        return self._log_data[component][decode_tag(tag)]

    def parse_from_bin(self, record_bin):
        """Register to self._tags by component type.

        Args:
            record_bin: Binary data from vdl log file.
        """
        record = record_pb2.Record()
        record.ParseFromString(record_bin)
        value = record.values[0]
        tag = decode_tag(value.tag)
        path = bfile.join(self.reader.dir, tag)

        if path not in self._tags.keys():
            value_type = value.WhichOneof("one_value")
            if "value" == value_type:
                component = "scalar"
            elif "image" == value_type:
                component = "image"
            elif "embeddings" == value_type:
                component = "embeddings"
            elif "audio" == value_type:
                component = "audio"
            elif "histogram" == value_type:
                component = "histogram"
            elif "pr_curve" == value_type:
                component = "pr_curve"
            elif "roc_curve" == value_type:
                component = "roc_curve"
            elif "meta_data" == value_type:
                self.update_meta_data(record)
                component = "meta_data"
            elif "text" == value_type:
                component = "text"
            elif "hparam" == value_type:
                component = "hyper_parameters"
            else:
                raise TypeError("Invalid value type `%s`." % value_type)
            self._tags[path] = component

        return self._tags[path], self.reader.dir, tag, value

    def update_meta_data(self, record):
        meta = record.values[0].meta_data
        if meta.display_name:
            self.name2tags[meta.display_name] = self.reader.dir
            self.tags2name[self.reader.dir] = meta.display_name

    def get_all_walk(self):
        self.walks = {}
        for dir in self.dir:
            for root, dirs, files in bfile.walk(dir):
                self.walks.update({root: files})

    def logs(self, update=False):
        """Get logs.

        Every dir(means `run` in vdl) has only one log(meads `actual log file`).

        Returns:
            walks: A dict like {"exp1": "vdlrecords.1587375595.log",
                                "exp2": "vdlrecords.1587375685.log"}
        """
        if self.walks is None or update is True:
            self.get_all_walk()

            walks_temp = {}
            for run, tags in self.walks.items():
                tags_temp = [tag for tag in tags if is_VDLRecord_file(tag)]
                tags_temp.sort(reverse=True)
                if len(tags_temp) > 0:
                    walks_temp.update({run: tags_temp[0]})
            self.walks = walks_temp
        return self.walks

    def get_log_reader(self, dir, log):
        """Get log reader for every vdl log file.

        Get instance of class RecordReader base on BFile. Note that each
        `log` may contain multi `tag`.

        Args:
            dir: Dir name of log.
            log: Vdl log file name.
        """
        if self.walks is None:
            self.logs()
        if self.walks.get(dir, None) != log:
            raise FileNotFoundError("Can't find file %s.", (dir + "/" + log))

        filepath = bfile.join(dir, log)
        if filepath not in self.readers.keys():
            self.register_reader(filepath, dir)
        self.reader = self.readers[filepath]
        return self.reader

    def get_file_reader(self, file_path):
        """Get file reader for specified vdl log file.

        Get instance of class RecordReader base on BFile.

        Args:
            file_path: Vdl log file path.
        """
        return self._get_file_reader(file_path, True)

    def _get_file_reader(self, file_path, update=True):
        if update:
            self.register_reader(file_path)
            self.reader = self.readers[file_path]
            self.reader.dir = file_path
            return self.reader
        else:
            reader = RecordReader(filepath=file_path)
            return reader

    def register_reader(self, path, dir=None, update=True):
        if update:
            if path not in list(self.readers.keys()):
                reader = RecordReader(filepath=path, dir=dir)
                self.readers[path] = reader
        else:
            pass

    def register_readers(self, update=False):
        """Register all readers for all vdl log files.

        Args:
            update: Need update if `update` is True.
        """
        self.logs(update)
        for dir, path in self.walks.items():
            filepath = bfile.join(dir, path)
            self.register_reader(filepath, dir)

    def add_remain(self):
        """Add remain data to data_manager.

        Add remain data to data manager according its component type and tag
        one by one.
        """
        for reader in self.readers.values():
            self.reader = reader
            remain = self.reader.get_remain()
            for item in remain:
                component, dir, tag, record = self.parse_from_bin(item)

                self.data_manager.add_item(component, self.reader.dir, tag,
                                           record)

    def get_remain(self, reader=None):
        """Get all remain data by self.reader.
        """
        if self.reader is None and reader is None:
            raise RuntimeError("Please specify log path!")
        return self.reader.get_remain() if reader is None else reader.get_remain()

    def read_log_data(self, remain, update=True):
        """Parse data from log file without sampling.

        Args:
            remain: Raw data from log file.
        """
        _log_data = collections.defaultdict(lambda: collections.defaultdict(list))
        for item in remain:
            component, dir, tag, record = self.parse_from_bin(item)
            _log_data[component][tag].append(record)
        if update:
            self._log_data = _log_data
        return _log_data

    @property
    def log_data(self):
        return self._log_data

    def runs(self, update=True):
        self.logs(update=update)
        return list(self.walks.keys())

    def tags(self):
        if self._tags is None:
            self.add_remain()
        return self._tags

    def components(self, update=False):
        """Get components type used by vdl.
        """
        if self.logdir is None:
            return set()
        if update is True:
            self.load_new_data(update=update)
        components_set = set(self._tags.values())

        return components_set

    def load_new_data(self, update=True):
        """Load remain data.

        Make sure all readers for every vdl log file are registered, load all
        remain data.
        """
        if self.logdir is not None:
            self.register_readers(update=update)
            self.add_remain()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
