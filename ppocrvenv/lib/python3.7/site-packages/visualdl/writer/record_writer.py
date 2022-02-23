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
from visualdl.utils.crc32 import masked_crc32c
from visualdl.proto import record_pb2
import struct
import time
import queue
import threading
import os

QUEUE_TIMEOUT = os.getenv("VDL_QUEUE_TIMEOUT")
if isinstance(QUEUE_TIMEOUT, str):
    QUEUE_TIMEOUT = int(QUEUE_TIMEOUT)


class RecordWriter(object):
    """Package data with crc32 or not.
    """

    def __init__(self, writer):
        self._writer = writer

    def write(self, data):
        """Package and write data to disk.

        Args:
            data (string or bytes): Data to write to disk.
        """
        header = struct.pack('<Q', len(data))
        self._writer.write(header + data)

    def write_crc(self, data):
        """Package data with crc32 and write to disk.

        Format of a single record: (little-endian)
        uint64    length
        uint32    masked crc of length
        byte      data[length]
        uint32    masked crc of data

        Args:
            data (string or bytes): Data to write to disk.
        """
        header = struct.pack('<Q', len(data))
        header_crc = struct.pack('<I', masked_crc32c(header))
        footer_crc = struct.pack('<I', masked_crc32c(data))
        self._writer.write(header + header_crc + data + footer_crc)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()

    @property
    def closed(self):
        return self._writer.closed


class RecordFileWriter(object):
    """ Writers `Record` protocol buffers to an records file one by one.

    The `RecordFileWriter` class create an records file in the specified
    directory and asynchronously writes `Record` protocol buffers to this
    file.
    """

    def __init__(self,
                 logdir,
                 max_queue_size=10,
                 flush_secs=120,
                 filename_suffix='',
                 filename=''):
        self._logdir = logdir
        if not bfile.exists(logdir):
            bfile.makedirs(logdir)

        if filename:
            if 'vdlrecords' in filename:
                self._file_name = bfile.join(logdir, filename)
                if bfile.exists(self._file_name):
                    print(
                        '`{}` is exists, VisualDL will add logs to it.'.format(
                            self._file_name))
            else:
                fn = "vdlrecords.%010d.log%s" % (time.time(), filename_suffix)
                self._file_name = bfile.join(logdir, fn)
                print('Since the log filename should contain `vdlrecords`, '
                      'the filename is invalid and `{}` will replace `{}`'.
                      format(  # noqa: E501
                          fn, filename))
        else:
            self._file_name = bfile.join(
                logdir,
                "vdlrecords.%010d.log%s" % (time.time(), filename_suffix))

        self._general_file_writer = bfile.BFile(self._file_name, "wb")
        self._async_writer = _AsyncWriter(
            RecordWriter(self._general_file_writer), max_queue_size,
            flush_secs)
        # TODO(shenyuhan) Maybe file_version in future.
        # _record = record_pb2.Record()
        # self.add_record(_record)
        self.flush()

    def get_logdir(self):
        return self._logdir

    def get_filename(self):
        return self._file_name

    def add_record(self, record):
        if not isinstance(record, record_pb2.Record):
            raise TypeError("Expected an record_pb2.Record proto, "
                            " but got %s" % type(record))
        a = record.SerializeToString()
        self._async_writer.write(a)

    def flush(self):
        self._async_writer.flush()

    def close(self):
        self._async_writer.close()


class _AsyncWriter(object):
    def __init__(self, record_writer, flush_secs=120, max_queue_size=20):
        """Start a sub-thread to handle data writing.

        Args:
            record_writer (visualdl.record_writer.RecordWriter):
        """

        self._record_writer = record_writer
        self._closed = False
        self._bytes_queue = queue.Queue(max_queue_size)
        self._worker = _AsyncWriterThread(self._bytes_queue,
                                          self._record_writer, flush_secs)
        self._lock = threading.Lock()
        self._worker.start()

    def write(self, bytestring):
        with self._lock:
            if self._closed:
                raise IOError("Writer is closed.")
            try:
                self._bytes_queue.put(bytestring, timeout=QUEUE_TIMEOUT)
            except queue.Full:
                print('This data was not written to the log due to timeout.')

    def flush(self):
        with self._lock:
            if self._closed:
                raise IOError("Writer is closed.")
            # Waiting all data to flush of writer.
            self._bytes_queue.join()
            self._record_writer.flush()

    def close(self):
        if not self._closed:
            with self._lock:
                if not self._closed:
                    self._closed = True
                    self._worker.stop()
                    self._record_writer.flush()
                    self._record_writer.close()


class _AsyncWriterThread(threading.Thread):
    def __init__(self, data_queue, record_writer, flush_secs):
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = data_queue
        self._record_writer = record_writer
        self._flush_secs = flush_secs
        self._next_flush_time = 0
        self._has_pending_data = False
        self._shutdown_signal = object()

    def stop(self):
        self._queue.put(self._shutdown_signal)
        self.join()

    def run(self):
        has_unresolved_bug = False
        while True:
            now = time.time()
            queue_wait_duration = self._next_flush_time - now
            data = None
            try:
                if queue_wait_duration > 0:
                    data = self._queue.get(True, queue_wait_duration)
                else:
                    data = self._queue.get(False)

                if data == self._shutdown_signal:
                    return

                self._record_writer.write(data)
                self._has_pending_data = True
            except queue.Empty:
                pass
            except Exception as e:
                # prevent the main thread from deadlock due to writing error.
                if not has_unresolved_bug:
                    print('Warning: Writing data Error, Due to unresolved Exception {}'.format(e))
                    print('Warning: Writing data to FileSystem failed since {}.'.format(
                        time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
                has_unresolved_bug = True
                pass
            finally:
                if data:
                    self._queue.task_done()

            now = time.time()
            if now > self._next_flush_time:
                if self._has_pending_data:
                    self._record_writer.flush()
                    self._has_pending_data = False
                self._next_flush_time = now + self._flush_secs
