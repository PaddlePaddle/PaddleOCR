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

from visualdl.io import bfile
import struct


class _RecordReader(object):
    def __init__(self, filepath=None):
        if filepath is None:
            raise FileNotFoundError('No filename provided, cannot read Events')
        if not bfile.exists(filepath):
            raise FileNotFoundError(
                '{} does not point to valid Events file'.format(filepath))

        self._curr_event = None
        self.file_handle = bfile.BFile(filepath, 'rb')

    def get_next(self):
        # Read the header
        self._curr_event = None
        header_str = self.file_handle.read(8)
        if len(header_str) != 8:
            # Hit EOF so raise and exit
            raise EOFError('No more events to read on LFS.')
        header = struct.unpack('Q', header_str)
        header_len = int(header[0])
        event_str = self.file_handle.read(header_len)

        self._curr_event = event_str

    def record(self):
        return self._curr_event


class _RecordReaderIterator(object):
    """A iterator of record reader.
    """

    def __init__(self, filepath):
        self._reader = _RecordReader(filepath=filepath)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._reader.get_next()
        except EOFError:
            raise StopIteration
        return self._reader.record()


class RecordReader(object):
    """Record reader of log file.

    Get one data or all data with this class.
    """

    def __init__(self, filepath, dir=None):
        self._filepath = filepath
        self._dir = dir
        self._reader = _RecordReaderIterator(filepath)

    def get_next(self, update=False):
        """Get next data in log file.

        Args:
            update (boolean): Get writer again if `update` is True.
        """
        if update:
            self._reader = _RecordReaderIterator(self._filepath)
        return self._reader.__next__()

    def get_all(self, update=False):
        """Get all data in log file.

        Args:
            update (boolean): Get writer again if `update` is True.
        """
        if update:
            self._reader = _RecordReaderIterator(self._filepath)
        return list(self._reader)

    def get_remain(self, update=False):
        """Get remain data in log file.

        Args:
            update (boolean): Get writer again if `update` is True.
        """
        if update:
            return self.get_all()
        results = []
        for item in self._reader:
            results.append(item)
        return results

    @property
    def dir(self):
        return self._dir

    @dir.setter
    def dir(self, value):
        self._dir = value
