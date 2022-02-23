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


class FileSystem(object):
    """
    A file system that support hadoop client desc. 

    Args:
        fs_type (string): fs_type, for example is "afs"
        user (string): hadoop param
        passwd (string): hadoop param
        hadoop bin (string): hadoop param
    Examples:
        fs = FileSystm()
    """

    def __init__(self,
                 fs_type="afs",
                 uri="afs://xx",
                 user=None,
                 passwd=None,
                 hadoop_bin=""):
        assert user != None
        assert passwd != None
        assert hadoop_bin != None
        import ps_pb2 as pslib
        self.fs_client = pslib.FsClientParameter()
        self.fs_client.uri = uri
        self.fs_client.user = user
        self.fs_client.passwd = passwd
        #self.fs_client.buffer_size = 0
        self.fs_client.hadoop_bin = hadoop_bin
        #self.fs_client.afs_conf = afs_conf if not afs_conf else ""

    def get_desc(self):
        """
        get hadoop desc.
        """
        return self.fs_client


class MPIHelper(object):
    """
    MPIHelper is a wrapper of mpi4py, support get_rank get_size etc.
    Args:
        No params
    Examples:
        mh = MPIHelper()
        mh.get_ip()
    """

    def __init__(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.MPI = MPI

    def get_rank(self):
        return self.comm.Get_rank()

    def get_size(self):
        return self.comm.Get_size()

    def get_ip(self):
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        return local_ip

    def get_hostname(self):
        import socket
        return socket.gethostname()

    def finalize(self):
        self.MPI.Finalize()
