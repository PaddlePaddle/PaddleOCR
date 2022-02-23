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

import os
import tempfile
import hashlib
import base64
import time

try:
    import hdfs
    from hdfs.util import HdfsError
    HDFS_ENABLED = True
except ImportError:
    HDFS_ENABLED = False
try:
    from baidubce.services.bos.bos_client import BosClient
    from baidubce import exception
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.auth.bce_credentials import BceCredentials
    BOS_ENABLED = True
except ImportError:
    BOS_ENABLED = False

# Note: Some codes here refer to TensorBoardX.
# A good default block size depends on the system in question.
# A somewhat conservative default chosen here.
_DEFAULT_BLOCK_SIZE = 16 * 1024 * 1024


def content_md5(buffer):
    md5 = hashlib.md5()
    md5.update(buffer)
    return base64.standard_b64encode(md5.digest())


class FileFactory(object):
    def __init__(self):
        self._register_factories = {}

    def register_filesystem(self, path, filesystem):
        self._register_factories.update({path: filesystem})

    def get_filesystem(self, path):
        if path.startswith(
                'hdfs://') and "hdfs" not in self._register_factories:
            if not HDFS_ENABLED:
                raise RuntimeError('Please install module named "hdfs".')
            try:
                default_file_factory.register_filesystem(
                    "hdfs", HDFileSystem())
            except hdfs.util.HdfsError:
                raise RuntimeError(
                    "Please initialize `~/.hdfscli.cfg` for HDFS.")
        elif path.startswith(
                'bos://') and "bos" not in self._register_factories:
            if not BOS_ENABLED:
                raise RuntimeError(
                    'Please install module named "bce-python-sdk".')
            default_file_factory.register_filesystem("bos", BosFileSystem())

        prefix = ""
        index = path.find("://")
        if index >= 0:
            prefix = path[:index]
        fs = self._register_factories.get(prefix, None)
        if fs is None:
            raise ValueError("No recognized filesystem for prefix %s" % prefix)
        return fs


default_file_factory = FileFactory()


class LocalFileSystem(object):
    def __init__(self):
        pass

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def makedirs(path):
        os.makedirs(path)

    @staticmethod
    def join(path, *paths):
        return os.path.join(path, *paths)

    def isfile(self, filename):
        return os.path.isfile(filename)

    def read_file(self, filename, binary_mode=True):
        mode = "rb" if binary_mode else "r"
        with open(filename, mode) as reader:
            data = reader.read()
        return data

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        mode = "rb" if binary_mode else "r"
        encoding = None if binary_mode else "utf-8"
        offset = None
        if continue_from is not None:
            offset = continue_from.get("last_offset", None)
        with open(filename, mode=mode, encoding=encoding) as fp:
            if offset is not None:
                fp.seek(offset)
            data = fp.read(size)
            continue_from_token = {"last_offset": fp.tell()}
            return data, continue_from_token

    def _write(self, filename, file_content, mode):
        encoding = None if "b" in mode else "utf-8"
        with open(filename, mode, encoding=encoding) as fp:
            fp.write(file_content)

    def append(self, filename, file_content, binary_mode=False):
        try:
            self._write(filename, file_content, "ab" if binary_mode else "a")
        except FileNotFoundError:
            self.makedirs(os.path.dirname(filename))

    def write(self, filename, file_content, binary_mode=False):
        try:
            self._write(filename, file_content, "ab" if binary_mode else "a")
        except FileNotFoundError:
            self.makedirs(os.path.dirname(filename))
        # self._write(filename, file_content, "wb" if binary_mode else "w")

    def walk(self, dir):
        if 'posix' == os.name:
            return os.walk(dir, followlinks=True)
        return os.walk(dir)


default_file_factory.register_filesystem("", LocalFileSystem())


class HDFileSystem(object):
    def __init__(self):
        self.cli = hdfs.config.Config().get_client('dev')

    def exists(self, path):
        if self.cli.status(hdfs_path=path[7:], strict=False) is None:
            return False
        else:
            return True

    def isfile(self, filename):
        return exists(filename)

    def read_file(self, filename, binary_mode=True):
        with self.cli.read(hdfs_path=filename[7:]) as reader:
            data = reader.read()
        return data

    def makedirs(self, path):
        self.cli.makedirs(hdfs_path=path[7:])

    @staticmethod
    def join(path, *paths):
        result = os.path.join(path, *paths)
        result.replace('\\', '/')
        return result

    def read(self, filename, binary_mode=False, size=0, continue_from=None):
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("last_offset", 0)

        encoding = None if binary_mode else "utf-8"
        try:
            with self.cli.read(
                    hdfs_path=filename[7:], offset=offset,
                    encoding=encoding) as reader:
                data = reader.read()
                continue_from_token = {"last_offset": offset + len(data)}
                return data, continue_from_token
        except HdfsError:
            raise EOFError('No more events to read on HDFS.')

    def append(self, filename, file_content, binary_mode=False):
        self.cli.write(hdfs_path=filename[7:], data=file_content, append=True)

    def write(self, filename, file_content, binary_mode=False):
        self.cli.write(hdfs_path=filename[7:], data=file_content, append=True)
        # self.cli.write(hdfs_path=filename[7:], data=file_content)

    def walk(self, dir):
        walks = self.cli.walk(hdfs_path=dir[7:])
        return (['hdfs://' + root, dirs, files] for root, dirs, files in walks)


def get_object_info(path):
    path = path[6:]
    index = path.index('/')
    bucket_name = path[0:index]
    object_key = path[index + 1:]
    return bucket_name, object_key


class BosConfigClient(object):
    def __init__(self, bos_ak, bos_sk, bos_sts, bos_host="bj.bcebos.com"):
        self.config = BceClientConfiguration(
            credentials=BceCredentials(bos_ak, bos_sk),
            endpoint=bos_host,
            security_token=bos_sts)
        self.bos_client = BosClient(self.config)

    def exists(self, path):
        bucket_name, object_key = get_object_info(path)
        try:
            self.bos_client.get_object_meta_data(bucket_name, object_key)
            return True
        except exception.BceError:
            return False

    def makedirs(self, path):
        if not path.endswith('/'):
            path += '/'
        if self.exists(path):
            return
        bucket_name, object_key = get_object_info(path)
        if not object_key.endswith('/'):
            object_key += '/'
        init_data = b''
        self.bos_client.append_object(
            bucket_name=bucket_name,
            key=object_key,
            data=init_data,
            content_md5=content_md5(init_data),
            content_length=len(init_data))

    @staticmethod
    def join(path, *paths):
        result = os.path.join(path, *paths)
        result.replace('\\', '/')
        return result

    def upload_object_from_file(self, path, filename):
        if not self.exists(path):
            self.makedirs(path)
        bucket_name, object_key = get_object_info(path)

        object_key = self.join(object_key, filename)
        # if not object_key.endswith('/'):
        #     object_key += '/'
        print('Uploading file `%s`' % filename)
        self.bos_client.put_object_from_file(
            bucket=bucket_name, key=object_key, file_name=filename)


class BosFileSystem(object):
    def __init__(self, write_flag=True):
        if write_flag:
            self.max_contents_count = 1
            self.max_contents_time = 1
            self.get_bos_config()
            self.bos_client = BosClient(self.config)
            self.file_length_map = {}

            self._file_contents_to_add = b''
            self._file_contents_count = 0
            self._start_append_time = time.time()

    def get_bos_config(self):
        bos_host = os.getenv("BOS_HOST")
        if not bos_host:
            raise KeyError('${BOS_HOST} is not found.')
        access_key_id = os.getenv("BOS_AK")
        if not access_key_id:
            raise KeyError('${BOS_AK} is not found.')
        secret_access_key = os.getenv("BOS_SK")
        if not secret_access_key:
            raise KeyError('${BOS_SK} is not found.')
        self.max_contents_count = int(os.getenv('BOS_CACHE_COUNT', 1))
        self.max_contents_time = int(os.getenv('BOS_CACHE_TIME', 1))
        bos_sts = os.getenv("BOS_STS")
        self.config = BceClientConfiguration(
            credentials=BceCredentials(access_key_id, secret_access_key),
            endpoint=bos_host,
            security_token=bos_sts)

    def set_bos_config(self, bos_ak, bos_sk, bos_sts,
                       bos_host="bj.bcebos.com"):
        self.config = BceClientConfiguration(
            credentials=BceCredentials(bos_ak, bos_sk),
            endpoint=bos_host,
            security_token=bos_sts)
        self.bos_client = BosClient(self.config)

    def renew_bos_client_from_server(self):
        import requests
        import json
        from visualdl.utils.dir import CONFIG_PATH
        with open(CONFIG_PATH, 'r') as fp:
            server_url = json.load(fp)['server_url']
        url = server_url + '/sts/'
        res = requests.post(url=url).json()
        err_code = res.get('code')
        msg = res.get('msg')
        if '000000' == err_code:
            sts_ak = msg.get('sts_ak')
            sts_sk = msg.get('sts_sk')
            sts_token = msg.get('token')
            self.set_bos_config(sts_ak, sts_sk, sts_token)
        else:
            print('Renew bos client error. Error msg: {}'.format(msg))
            return

    def isfile(self, filename):
        return exists(filename)

    def read_file(self, filename, binary=True):
        bucket_name, object_key = get_object_info(filename)
        result = self.bos_client.get_object_as_string(bucket_name, object_key)
        return result

    def exists(self, path):
        bucket_name, object_key = get_object_info(path)
        try:
            self.bos_client.get_object_meta_data(bucket_name, object_key)
            return True
        except exception.BceError:
            return False

    def get_meta(self, bucket_name, object_key):
        return self.bos_client.get_object_meta_data(bucket_name, object_key)

    def makedirs(self, path):
        if not path.endswith('/'):
            path += '/'
        if self.exists(path):
            return
        bucket_name, object_key = get_object_info(path)
        if not object_key.endswith('/'):
            object_key += '/'
        init_data = b''
        self.bos_client.append_object(
            bucket_name=bucket_name,
            key=object_key,
            data=init_data,
            content_md5=content_md5(init_data),
            content_length=len(init_data))

    @staticmethod
    def join(path, *paths):
        result = os.path.join(path, *paths)
        result.replace('\\', '/')
        return result

    def read(self, filename, binary_mode=False, size=0, continue_from=None):
        bucket_name, object_key = get_object_info(filename)
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("last_offset", 0)
        length = int(
            self.get_meta(bucket_name, object_key).metadata.content_length)
        if offset < length:
            data = self.bos_client.get_object_as_string(
                bucket_name=bucket_name,
                key=object_key,
                range=[offset, length - 1])
        else:
            data = b''

        continue_from_token = {"last_offset": length}
        return data, continue_from_token

    def ready_to_append(self):
        if self._file_contents_count >= self.max_contents_count or \
                time.time() - self._start_append_time > self.max_contents_time:
            return True
        else:
            return False

    def append(self, filename, file_content, binary_mode=False, force=False):
        self._file_contents_to_add += file_content
        self._file_contents_count += 1

        if not force and not self.ready_to_append():
            return
        file_content = self._file_contents_to_add
        bucket_name, object_key = get_object_info(filename)
        if not self.exists(filename):
            init_data = b''
            try:
                self.bos_client.append_object(
                    bucket_name=bucket_name,
                    key=object_key,
                    data=init_data,
                    content_md5=content_md5(init_data),
                    content_length=len(init_data))
            except (exception.BceServerError, exception.BceHttpClientError):
                self.renew_bos_client_from_server()
                self.bos_client.append_object(
                    bucket_name=bucket_name,
                    key=object_key,
                    data=init_data,
                    content_md5=content_md5(init_data),
                    content_length=len(init_data))
                return
        content_length = len(file_content)

        try:
            offset = self.get_meta(bucket_name,
                                   object_key).metadata.content_length
            self.bos_client.append_object(
                bucket_name=bucket_name,
                key=object_key,
                data=file_content,
                content_md5=content_md5(file_content),
                content_length=content_length,
                offset=offset)
        except (exception.BceServerError, exception.BceHttpClientError):
            self.renew_bos_client_from_server()
            offset = self.get_meta(bucket_name,
                                   object_key).metadata.content_length
            self.bos_client.append_object(
                bucket_name=bucket_name,
                key=object_key,
                data=file_content,
                content_md5=content_md5(file_content),
                content_length=content_length,
                offset=offset)

        self._file_contents_to_add = b''
        self._file_contents_count = 0
        self._start_append_time = time.time()

    def write(self, filename, file_content, binary_mode=False):
        self.append(filename, file_content, binary_mode=False)

        # bucket_name, object_key = BosFileSystem._get_object_info(filename)
        #
        # self.bos_client.append_object(bucket_name=bucket_name,
        #                               key=object_key,
        #                               data=file_content,
        #                               content_md5=content_md5(file_content),
        #                               content_length=len(file_content))

    def walk(self, dir):
        class WalkGenerator():
            def __init__(self, bucket_name, contents):
                self.contents = None
                self.length = 0
                self.bucket = bucket_name
                self.handle_contents(contents)
                self.count = 0

            def handle_contents(self, contents):
                contents_map = {}
                for item in contents:
                    try:
                        rindex = item.rindex('/')
                        key = item[0:rindex]
                        value = item[rindex + 1:]
                    except ValueError:
                        key = '.'
                        value = item
                    if key in contents_map.keys():
                        contents_map[key].append(value)
                    else:
                        contents_map[key] = [value]
                temp_walk = []
                for key, value in contents_map.items():
                    temp_walk.append([
                        BosFileSystem.join('bos://' + self.bucket, key), [],
                        value
                    ])
                self.length = len(temp_walk)
                self.contents = temp_walk

            def __iter__(self):
                return self

            def __next__(self):
                if self.count < self.length:
                    self.count += 1
                    return self.contents[self.count - 1]
                else:
                    raise StopIteration

        bucket_name, object_key = get_object_info(dir)

        if object_key in ['.', './']:
            prefix = None
        else:
            prefix = object_key if object_key.endswith(
                '/') else object_key + '/'
        response = self.bos_client.list_objects(bucket_name, prefix=prefix)
        contents = [content.key for content in response.contents]
        return WalkGenerator(bucket_name, contents)


class BFile(object):
    def __init__(self, filename, mode):
        if mode not in ('r', 'rb', 'br', 'w', 'wb', 'bw'):
            raise NotImplementedError("mode {} not supported by "
                                      "BFile.".format(mode))
        self._filename = filename
        self.fs = default_file_factory.get_filesystem(filename)
        self.fs_supports_append = hasattr(self.fs, 'append')
        self.buff = None
        self.buff_chunk_size = _DEFAULT_BLOCK_SIZE
        self.buff_offset = 0
        self.continuation_token = None
        self.write_temp = None
        self.write_started = False
        self.binary_mode = 'b' in mode
        self.write_mode = 'w' in mode
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.buff = None
        self.buff_offset = 0
        self.continuation_token = None

    def __iter__(self):
        return self

    def isfile(self, filename):
        return self.fs.isfile(filename)

    def _read_buffer_to_offset(self, new_buff_offset):
        """Read buffer from index self.buffer_offset to index new_buff_offset.

        self.buff_offset marks the last position of the last read,
        new_buff_offset indicates the last position of this read.
        self.buff_offset will be updated by new_buff_offset after this read.

        Returns:
            self.buff[i1: i2]: Content of self.buff.
        """
        old_buff_offset = self.buff_offset
        read_size = min(len(self.buff), new_buff_offset) - old_buff_offset
        self.buff_offset += read_size
        return self.buff[old_buff_offset:old_buff_offset + read_size]

    def read_file(self, filename, binnary=True):
        return self.fs.read_file(filename, binnary)

    def read(self, n=None):
        """Read `n` or all contents of self.buff or file.

        Returns:
            result: Data from self.buff or file.
        """
        result = None
        # If self.buff is not none and length of self.buff more than
        # self.buff_offset, means there are some content in self.buff have
        # not been read.
        if self.buff and len(self.buff) > self.buff_offset:
            if n is not None:
                chunk = self._read_buffer_to_offset(self.buff_offset + n)
                # If length of data in self.buff is more than `n`, then read `n`
                # data from local buffer.
                if len(chunk) == n:
                    return chunk
                result = chunk
                # The length of all data in self.buff may less than `n`,
                # so we should read other `n-length(self.buff)` data.
                n -= len(chunk)
            # If n is none, read all data in self.buff.
            else:
                # add all local buffer and update offsets
                result = self._read_buffer_to_offset(len(self.buff))

        # self.buff is empty if program is here.
        # Read from filesystem.
        # If n is not none, read max(n, self.buff_chunk_size) data from file,
        # otherwise read all data from file.
        # TODO(shenhuhan) N is limited to max_buff, but all-data is unlimited?
        read_size = max(self.buff_chunk_size, n) if n is not None else None
        self.buff, self.continuation_token = self.fs.read(
            self._filename, self.binary_mode, read_size,
            self.continuation_token)
        self.buff_offset = 0

        if n is not None:
            chunk = self._read_buffer_to_offset(n)
        else:
            # add all local buffer and update offsets
            chunk = self._read_buffer_to_offset(len(self.buff))
        result = result + chunk if result else chunk

        return result

    def write(self, file_content):
        """Write contents to file.

        Args:
            file_content: Contents waiting to be written to file.
        """
        if not self.write_mode:
            raise RuntimeError("File not opened in write mode")
        if self.closed:
            raise RuntimeError("File already closed")

        if self.fs_supports_append:
            if not self.write_started:
                self.fs.write(self._filename, file_content, self.binary_mode)
                self.write_started = True
            else:
                self.fs.append(self._filename, file_content, self.binary_mode)
        else:
            # add to temp file, but wait for flush to write to final filesystem
            if self.write_temp is None:
                mode = "w+b" if self.binary_mode else "w+"
                self.write_temp = tempfile.TemporaryFile(mode)
            self.write_temp.write(file_content)

    def __next__(self):
        line = None
        while True:
            if not self.buff:
                # read one unit into the buffer
                line = self.read(1)
                if line and (line[-1] == '\n' or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()
            else:
                index = self.buff.find('\n', self.buff_offset)
                if index != -1:
                    # include line until now plus newline
                    chunk = self.read(index + 1 - self.buff_offset)
                    line = line + chunk if line else chunk
                    return line

                # read one unit past end of buffer
                chunk = self.read(len(self.buff) + 1 - self.buff_offset)
                line = line + chunk if line else chunk
                if line and (line[-1] == '\n' or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()

    def next(self):
        return self.__next__()

    def flush(self):
        """Flush data to disk.
        """
        if self.closed:
            raise RuntimeError("File already closed")
        if not self.fs_supports_append:
            if self.write_temp is not None:
                # read temp file from the beginning
                self.write_temp.flush()
                self.write_temp.seek(0)
                chunk = self.write_temp.read()
                if chunk is not None:
                    # write full contents and keep in temp file
                    self.fs.write(self._filename, chunk, self.binary_mode)
                    self.write_temp.seek(len(chunk))

    def close(self):
        if isinstance(self.fs, BosFileSystem):
            try:
                self.fs.append(
                    self._filename, b'', self.binary_mode, force=True)
            except Exception:
                pass
        self.flush()
        if self.write_temp is not None:
            self.write_temp.close()
            self.write_temp = None
            self.write_started = False
        self.closed = True


def exists(path):
    return default_file_factory.get_filesystem(path).exists(path)


def makedirs(path):
    return default_file_factory.get_filesystem(path).makedirs(path)


def join(path, *paths):
    return default_file_factory.get_filesystem(path).join(path, *paths)


def walk(dir):
    return default_file_factory.get_filesystem(dir).walk(dir)
