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
"""This is definition of dataset class, which is high performance IO."""

import paddle
from paddle.fluid.proto import data_feed_pb2
from google.protobuf import text_format
import paddle.fluid.core as core

__all__ = []


class DatasetBase(object):
    """ Base dataset class. """

    def __init__(self):
        """ Init. """
        # define class name here
        # to decide whether we need create in memory instance
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        self.proto_desc.pipe_command = "cat"
        self.dataset = core.Dataset("MultiSlotDataset")
        self.thread_num = 1
        self.filelist = []
        self.use_ps_gpu = False
        self.psgpu = None

    def init(self,
             batch_size=1,
             thread_num=1,
             use_var=[],
             pipe_command="cat",
             input_type=0,
             fs_name="",
             fs_ugi="",
             download_cmd="cat"):
        """
        should be called only once in user's python scripts to initialize setings of dataset instance. 
        Normally, it is called by InMemoryDataset or QueueDataset.

        Args:
            batch_size(int): batch size. It will be effective during training. default is 1.
            thread_num(int): thread num, it is the num of readers. default is 1.
            use_var(list): list of variables. Variables which you will use. default is [].
            pipe_command(str): pipe command of current dataset. A pipe command is a UNIX pipeline command that can be used only. default is "cat"
            input_type(int): the input type of generated input. 0 is for one sample, 1 is for one batch. defalut is 0.
            fs_name(str): fs name. default is "".
            fs_ugi(str): fs ugi. default is "".
            download_cmd(str): customized download command. default is "cat"


        """
        self._set_batch_size(batch_size)
        self._set_thread(thread_num)
        self._set_use_var(use_var)
        self._set_pipe_command(pipe_command)
        self._set_input_type(input_type)
        self._set_hdfs_config(fs_name, fs_ugi)
        self._set_download_cmd(download_cmd)

    def _set_pipe_command(self, pipe_command):
        """
        Set pipe command of current dataset
        A pipe command is a UNIX pipeline command that can be used only

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.dataset.DatasetBase()
              dataset._set_pipe_command("python my_script.py")

        Args:
            pipe_command(str): pipe command

        """
        self.proto_desc.pipe_command = pipe_command

    def _set_batch_size(self, batch_size):
        """
        Set batch size. Will be effective during training

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.DatasetBase()
              dataset._set_batch_size(128)

        Args:
            batch_size(int): batch size

        """
        self.proto_desc.batch_size = batch_size

    def _set_thread(self, thread_num):
        """
        Set thread num, it is the num of readers.

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.DatasetBase()
              dataset._set_thread(12)

        Args:
            thread_num(int): thread num
        """
        self.dataset.set_thread_num(thread_num)
        self.thread_num = thread_num

    def set_filelist(self, filelist):
        """
        Set file list in current worker. The filelist is indicated by a list of file names (string).

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.DatasetBase()
              dataset.set_filelist(['a.txt', 'b.txt'])

        Args:
            filelist(list[str]): list of file names of inputs.
        """
        self.dataset.set_filelist(filelist)
        self.filelist = filelist

    def _set_input_type(self, input_type):
        self.proto_desc.input_type = input_type

    def _set_use_var(self, var_list):
        """
        Set Variables which you will use.

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.DatasetBase()
              dataset._set_use_var([data, label])

        Args:
            var_list(list): variable list
        """
        multi_slot = self.proto_desc.multi_slot_desc
        for var in var_list:
            slot_var = multi_slot.slots.add()
            slot_var.is_used = True
            slot_var.name = var.name
            if var.lod_level == 0:
                slot_var.is_dense = True
                slot_var.shape.extend(var.shape)
            if var.dtype == core.VarDesc.VarType.FP32:
                slot_var.type = "float"
            elif var.dtype == core.VarDesc.VarType.INT64:
                slot_var.type = "uint64"
            else:
                raise ValueError(
                    "Currently, paddle.distributed.fleet.dataset only supports dtype=float32 and dtype=int64"
                )

    def _set_hdfs_config(self, fs_name, fs_ugi):
        """
        Set hdfs config: fs name ad ugi

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.DatasetBase()
              dataset._set_hdfs_config("my_fs_name", "my_fs_ugi")

        Args:
            fs_name(str): fs name
            fs_ugi(str): fs ugi
        """
        self.dataset.set_hdfs_config(fs_name, fs_ugi)

    def _set_download_cmd(self, download_cmd):
        """
        Set customized download cmd: download_cmd

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.DatasetBase()
              dataset._set_download_cmd("./read_from_afs")

        Args:
            download_cmd(str): customized download command
        """
        self.dataset.set_download_cmd(download_cmd)

    def _prepare_to_run(self):
        """
        Set data_feed_desc before load or shuffle,
        user no need to call this function.
        """
        if self.thread_num > len(self.filelist):
            self.thread_num = len(self.filelist)
        self.dataset.set_thread_num(self.thread_num)
        self.dataset.set_data_feed_desc(self._desc())
        self.dataset.create_readers()

    def _set_use_ps_gpu(self, use_ps_gpu):
        """
        set use_ps_gpu flag

        Args:
            use_ps_gpu: bool
        """
        self.use_ps_gpu = use_ps_gpu
        # if not defined heterps with paddle, users will not use psgpu
        if not core._is_compiled_with_heterps():
            self.use_ps_gpu = 0
        elif self.use_ps_gpu:
            self.psgpu = core.PSGPU()

    def _finish_to_run(self):
        self.dataset.destroy_readers()

    def _desc(self):
        """
        Returns a protobuf message for this DataFeedDesc

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.DatasetBase()
              print(dataset._desc())

        Returns:
            A string message
        """
        return text_format.MessageToString(self.proto_desc)

    def _dynamic_adjust_before_train(self, thread_num):
        pass

    def _dynamic_adjust_after_train(self):
        pass

    def _check_use_var_with_data_generator(self, var_list, data_generator_class,
                                           test_file):
        """
         Var consistency insepection of use_var_list and data_generator data.

        Examples:
            .. code-block:: python

              # required: skiptest
              import paddle
              from dataset_generator import CTRDataset
              dataset = paddle.distributed.fleet.DatasetBase()
              generator_class = CTRDataset()
              dataset._check_use_var_with_data_generator([data, label], generator_class, "data/part-00000")

        Args:
            var_list(list): variable list
            data_generator_class(class): data_generator class
            test_file(str): local test file path
        """

        f = open(test_file, "r")
        var_len = len(var_list)

        while True:
            line = f.readline()
            if line:
                line_iter = data_generator_class.generate_sample(line)
                for user_parsed_line in line_iter():
                    data_gen_len = len(user_parsed_line)
                    if var_len != data_gen_len:
                        raise ValueError(
                            "var length mismatch error: var_list = %s vs data_generator = %s"
                            % (var_len, data_gen_len))

                    for i, ele in enumerate(user_parsed_line):
                        if len(ele[1]) == 0:
                            raise ValueError(
                                "var length error: var %s's length in data_generator is 0"
                                % ele[0])

                        if var_list[
                                i].dtype == core.VarDesc.VarType.FP32 and not all(
                                    isinstance(ele, float) for ele in ele[1]):
                            raise TypeError(
                                "var dtype mismatch error: var name = %s, var type in var_list = %s, while var in data_generator contains non-float value, which is %s \n"
                                "Please check if order of var_list and data_generator are aligned. \n"
                                "Please check if var's type in data_generator is correct."
                                % (ele[0], "float", ele[1]))

                        if (var_list[i].dtype == core.VarDesc.VarType.INT64 or
                                var_list[i].dtype == core.VarDesc.VarType.INT32
                            ) and not all(
                                isinstance(ele, int) for ele in ele[1]):
                            raise TypeError(
                                "var dtype mismatch error: var name = %s, var type in var_list = %s, while var in data_generator contains non-int value, which is %s \n"
                                "Please check if order of var_list and data_generator are aligned. \n"
                                "Please check if var's type in data_generator is correct."
                                % (ele[0], "int", ele[1]))

            else:
                break

        f.close()


class InMemoryDataset(DatasetBase):
    """
    :api_attr: Static Graph
    
    It will load data into memory and shuffle data before training.

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()
            dataset = paddle.distributed.InMemoryDataset()

    """

    def __init__(self):
        """ Init. """
        super(InMemoryDataset, self).__init__()
        self.proto_desc.name = "MultiSlotInMemoryDataFeed"
        self.fleet_send_batch_size = None
        self.is_user_set_queue_num = False
        self.queue_num = None
        self.parse_ins_id = False
        self.parse_content = False
        self.parse_logkey = False
        self.merge_by_sid = True
        self.enable_pv_merge = False
        self.merge_by_lineid = False
        self.fleet_send_sleep_seconds = None

    def _init_distributed_settings(self, **kwargs):
        """
        :api_attr: Static Graph

        should be called only once in user's python scripts to initialize distributed-related setings of dataset instance
        Args:
            kwargs: Keyword arguments. Currently, we support following keys in **kwargs:

            merge_size(int): ins size to merge, if merge_size > 0, set merge by line id, 
                             instances of same line id will be merged after shuffle, 
                             you should parse line id in data generator. default is -1.
            parse_ins_id(bool): Set if Dataset need to parse ins_id. default is False.
            parse_content(bool): Set if Dataset need to parse content. default is False.
            fleet_send_batch_size(int): Set fleet send batch size in one rpc, default is 1024
            fleet_send_sleep_seconds(int): Set fleet send sleep time, default is 0
            fea_eval(bool): Set if Dataset need to do feature importance evaluation using slots shuffle.
                            default is False.
            candidate_size(int): if fea_eval is set True, set the candidate size used in slots shuffle.

        Examples:
            .. code-block:: python

              import paddle
              paddle.enable_static()
              dataset = paddle.distributed.InMemoryDataset()
              dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=[])
              dataset._init_distributed_settings(
                    parse_ins_id=True,
                    parse_content=True,
                    fea_eval=True,
                    candidate_size=10000)
              
        """
        merge_size = kwargs.get("merge_size", -1)
        if merge_size > 0:
            self._set_merge_by_lineid(merge_size)

        parse_ins_id = kwargs.get("parse_ins_id", False)
        self._set_parse_ins_id(parse_ins_id)

        parse_content = kwargs.get("parse_content", False)
        self._set_parse_content(parse_content)

        fleet_send_batch_size = kwargs.get("fleet_send_batch_size", None)
        if fleet_send_batch_size:
            self._set_fleet_send_batch_size(fleet_send_batch_size)

        fleet_send_sleep_seconds = kwargs.get("fleet_send_sleep_seconds", None)
        if fleet_send_sleep_seconds:
            self._set_fleet_send_sleep_seconds(fleet_send_sleep_seconds)

        fea_eval = kwargs.get("fea_eval", False)
        if fea_eval:
            candidate_size = kwargs.get("candidate_size", 10000)
            self._set_fea_eval(candidate_size, True)

    def update_settings(self, **kwargs):
        """
        :api_attr: Static Graph

        should be called in user's python scripts to update setings of dataset instance.

        Args:
            kwargs: Keyword arguments. Currently, we support following keys in **kwargs,
                    including single node settings and advanced distributed related settings:
            batch_size(int): batch size. It will be effective during training. default is 1.
            thread_num(int): thread num, it is the num of readers. default is 1.
            use_var(list): list of variables. Variables which you will use. default is [].
            input_type(int): the input type of generated input. 0 is for one sample, 1 is for one batch. defalut is 0.
            fs_name(str): fs name. default is "".
            fs_ugi(str): fs ugi. default is "".
            pipe_command(str): pipe command of current dataset. A pipe command is a UNIX pipeline command that can be used only. default is "cat"
            download_cmd(str): customized download command. default is "cat"
            data_feed_type(str): data feed type used in c++ code. default is "MultiSlotInMemoryDataFeed".
            queue_num(int): Dataset output queue num, training threads get data from queues. default is-1, which is set same as thread number in c++.

            merge_size(int): ins size to merge, if merge_size > 0, set merge by line id, 
                             instances of same line id will be merged after shuffle, 
                             you should parse line id in data generator. default is -1.
            parse_ins_id(bool): Set if Dataset need to parse ins_id. default is False.
            parse_content(bool): Set if Dataset need to parse content. default is False.
            fleet_send_batch_size(int): Set fleet send batch size in one rpc, default is 1024
            fleet_send_sleep_seconds(int): Set fleet send sleep time, default is 0
            fea_eval(bool): Set if Dataset need to do feature importance evaluation using slots shuffle.
                            default is False.
            candidate_size(int): if fea_eval is set True, set the candidate size used in slots shuffle.

        Examples:
            .. code-block:: python

                import paddle    
                paddle.enable_static()

                dataset = paddle.distributed.InMemoryDataset()
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=[])
                dataset._init_distributed_settings(
                    parse_ins_id=True,
                    parse_content=True,
                    fea_eval=True,
                    candidate_size=10000)
                dataset.update_settings(batch_size=2)
            
        """
        for key in kwargs:
            if key == "pipe_command":
                self._set_pipe_command(kwargs[key])
            elif key == "batch_size":
                self._set_batch_size(kwargs[key])
            elif key == "thread_num":
                self._set_thread(kwargs[key])
            elif key == "use_var":
                self._set_use_var(kwargs[key])
            elif key == "input_type":
                self._set_input_type(kwargs[key])
            elif key == "fs_name" and "fs_ugi" in kwargs:
                self._set_hdfs_config(kwargs[key], kwargs["fs_ugi"])
            elif key == "download_cmd":
                self._set_download_cmd(kwargs[key])
            elif key == "merge_size" and kwargs.get("merge_size", -1) > 0:
                self._set_merge_by_lineid(kwargs[key])
            elif key == "parse_ins_id":
                self._set_parse_ins_id(kwargs[key])
            elif key == "parse_content":
                self._set_parse_content(kwargs[key])
            elif key == "fleet_send_batch_size":
                self._set_fleet_send_batch_size(kwargs[key])
            elif key == "fleet_send_sleep_seconds":
                self._set_fleet_send_sleep_seconds(kwargs[key])
            elif key == "fea_eval" and kwargs[key] == True:
                candidate_size = kwargs.get("candidate_size", 10000)
                self._set_fea_eval(candidate_size, True)

    def init(self, **kwargs):
        """
        :api_attr: Static Graph

        should be called only once in user's python scripts to initialize setings of dataset instance
        
        Args:
            kwargs: Keyword arguments. Currently, we support following keys in **kwargs:
            
            batch_size(int): batch size. It will be effective during training. default is 1.
            thread_num(int): thread num, it is the num of readers. default is 1.
            use_var(list): list of variables. Variables which you will use. default is [].
            input_type(int): the input type of generated input. 0 is for one sample, 1 is for one batch. defalut is 0.
            fs_name(str): fs name. default is "".
            fs_ugi(str): fs ugi. default is "".
            pipe_command(str): pipe command of current dataset. A pipe command is a UNIX pipeline command that can be used only. default is "cat"
            download_cmd(str): customized download command. default is "cat"
            data_feed_type(str): data feed type used in c++ code. default is "MultiSlotInMemoryDataFeed".
            queue_num(int): Dataset output queue num, training threads get data from queues. default is -1, which is set same as thread number in c++.

        Examples:
            .. code-block:: python

                import paddle
                import os
                paddle.enable_static()

                with open("test_queue_dataset_run_a.txt", "w") as f:
                    data = "2 1 2 2 5 4 2 2 7 2 1 3"
                    f.write(data)
                with open("test_queue_dataset_run_b.txt", "w") as f:
                    data = "2 1 2 2 5 4 2 2 7 2 1 3"
                    f.write(data)

                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)

                dataset = paddle.distributed.InMemoryDataset()
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                dataset.set_filelist(
                    ["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"])
                dataset.load_into_memory()
                
                place = paddle.CPUPlace()
                exe = paddle.static.Executor(place)
                startup_program = paddle.static.Program()
                main_program = paddle.static.Program()
                exe.run(startup_program)

                exe.train_from_dataset(main_program, dataset)
                
                os.remove("./test_queue_dataset_run_a.txt")
                os.remove("./test_queue_dataset_run_b.txt")

        """
        batch_size = kwargs.get("batch_size", 1)
        thread_num = kwargs.get("thread_num", 1)
        use_var = kwargs.get("use_var", [])
        input_type = kwargs.get("input_type", 0)
        fs_name = kwargs.get("fs_name", "")
        fs_ugi = kwargs.get("fs_ugi", "")
        pipe_command = kwargs.get("pipe_command", "cat")
        download_cmd = kwargs.get("download_cmd", "cat")

        super(InMemoryDataset, self).init(
            batch_size=batch_size,
            thread_num=thread_num,
            use_var=use_var,
            pipe_command=pipe_command,
            input_type=input_type,
            fs_name=fs_name,
            fs_ugi=fs_ugi,
            download_cmd=download_cmd)

        data_feed_type = kwargs.get("data_feed_type",
                                    "MultiSlotInMemoryDataFeed")
        self._set_feed_type(data_feed_type)

        if kwargs.get("queue_num", -1) > 0:
            queue_num = kwargs.get("queue_num", -1)
            self._set_queue_num(queue_num)

    def _set_feed_type(self, data_feed_type):
        """
        Set data_feed_desc
        """
        self.proto_desc.name = data_feed_type

    def _prepare_to_run(self):
        """
        Set data_feed_desc before load or shuffle,
        user no need to call this function.
        """
        if self.thread_num <= 0:
            self.thread_num = 1
        self.dataset.set_thread_num(self.thread_num)
        if self.queue_num is None:
            self.queue_num = self.thread_num
        self.dataset.set_queue_num(self.queue_num)
        self.dataset.set_parse_ins_id(self.parse_ins_id)
        self.dataset.set_parse_content(self.parse_content)
        self.dataset.set_parse_logkey(self.parse_logkey)
        self.dataset.set_merge_by_sid(self.merge_by_sid)
        self.dataset.set_enable_pv_merge(self.enable_pv_merge)
        self.dataset.set_data_feed_desc(self._desc())
        self.dataset.create_channel()
        self.dataset.create_readers()

    def _dynamic_adjust_before_train(self, thread_num):
        if not self.is_user_set_queue_num:
            if self.use_ps_gpu:
                self.dataset.dynamic_adjust_channel_num(thread_num, True)
            else:
                self.dataset.dynamic_adjust_channel_num(thread_num, False)
        self.dataset.dynamic_adjust_readers_num(thread_num)

    def _dynamic_adjust_after_train(self):
        if not self.is_user_set_queue_num:
            if self.use_ps_gpu:
                self.dataset.dynamic_adjust_channel_num(self.thread_num, True)
            else:
                self.dataset.dynamic_adjust_channel_num(self.thread_num, False)
        self.dataset.dynamic_adjust_readers_num(self.thread_num)

    def _set_queue_num(self, queue_num):
        """
        Set Dataset output queue num, training threads get data from queues

        Args:
            queue_num(int): dataset output queue num

        Examples:
            .. code-block:: python

              import paddle
              paddle.enable_static()
              dataset = paddle.distributed.InMemoryDataset()
              dataset._set_queue_num(12)

        """
        self.is_user_set_queue_num = True
        self.queue_num = queue_num

    def _set_parse_ins_id(self, parse_ins_id):
        """
        Set if Dataset need to parse insid

        Args:
            parse_ins_id(bool): if parse ins_id or not

        Examples:
            .. code-block:: python

              import paddle
              paddle.enable_static()
              dataset = paddle.distributed.InMemoryDataset()
              dataset._set_parse_ins_id(True)

        """
        self.parse_ins_id = parse_ins_id

    def _set_parse_content(self, parse_content):
        """
        Set if Dataset need to parse content

        Args:
            parse_content(bool): if parse content or not

        Examples:
            .. code-block:: python

              import paddle
              paddle.enable_static()
              dataset = paddle.distributed.InMemoryDataset()
              dataset._set_parse_content(True)

        """
        self.parse_content = parse_content

    def _set_fleet_send_batch_size(self, fleet_send_batch_size=1024):
        """
        Set fleet send batch size, default is 1024

        Args:
            fleet_send_batch_size(int): fleet send batch size

        Examples:
            .. code-block:: python

              import paddle
              paddle.enable_static()
              dataset = paddle.distributed.InMemoryDataset()
              dataset._set_fleet_send_batch_size(800)

        """
        self.fleet_send_batch_size = fleet_send_batch_size

    def _set_fleet_send_sleep_seconds(self, fleet_send_sleep_seconds=0):
        """
        Set fleet send sleep time, default is 0

        Args:
            fleet_send_sleep_seconds(int): fleet send sleep time

        Examples:
            .. code-block:: python

              import paddle
              paddle.enable_static()
              dataset = paddle.distributed.InMemoryDataset()
              dataset._set_fleet_send_sleep_seconds(2)

        """
        self.fleet_send_sleep_seconds = fleet_send_sleep_seconds

    def _set_merge_by_lineid(self, merge_size=2):
        """
        Set merge by line id, instances of same line id will be merged after
        shuffle, you should parse line id in data generator.

        Args:
            merge_size(int): ins size to merge. default is 2.

        Examples:
            .. code-block:: python

              import paddle
              paddle.enable_static()
              dataset = paddle.distributed.InMemoryDataset()
              dataset._set_merge_by_lineid()

        """
        self.dataset.set_merge_by_lineid(merge_size)
        self.merge_by_lineid = True
        self.parse_ins_id = True

    def _set_generate_unique_feasigns(self, generate_uni_feasigns, shard_num):
        self.dataset.set_generate_unique_feasigns(generate_uni_feasigns)
        self.gen_uni_feasigns = generate_uni_feasigns
        self.local_shard_num = shard_num

    def _generate_local_tables_unlock(self, table_id, fea_dim, read_thread_num,
                                      consume_thread_num, shard_num):
        self.dataset.generate_local_tables_unlock(
            table_id, fea_dim, read_thread_num, consume_thread_num, shard_num)

    def load_into_memory(self, is_shuffle=False):
        """
        :api_attr: Static Graph
        
        Load data into memory

        Args:
            is_shuffle(bool): whether to use local shuffle, default is False

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()
                
                dataset = paddle.distributed.InMemoryDataset()
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.load_into_memory()
        """
        self._prepare_to_run()
        if not self.use_ps_gpu:
            self.dataset.load_into_memory()
        elif core._is_compiled_with_heterps():
            self.psgpu.set_dataset(self.dataset)
            self.psgpu.load_into_memory(is_shuffle)

    def preload_into_memory(self, thread_num=None):
        """
        :api_attr: Static Graph

        Load data into memory in async mode

        Args:
            thread_num(int): preload thread num

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()

                dataset = paddle.distributed.InMemoryDataset()
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.preload_into_memory()
                dataset.wait_preload_done()
        """
        self._prepare_to_run()
        if thread_num is None:
            thread_num = self.thread_num
        self.dataset.set_preload_thread_num(thread_num)
        self.dataset.create_preload_readers()
        self.dataset.preload_into_memory()

    def wait_preload_done(self):
        """
        :api_attr: Static Graph

        Wait preload_into_memory done

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()

                dataset = paddle.distributed.InMemoryDataset()
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.preload_into_memory()
                dataset.wait_preload_done()
        """
        self.dataset.wait_preload_done()
        self.dataset.destroy_preload_readers()

    def local_shuffle(self):
        """
        :api_attr: Static Graph

        Local shuffle

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()

                dataset = paddle.distributed.InMemoryDataset()
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.load_into_memory()
                dataset.local_shuffle()
        """
        self.dataset.local_shuffle()

    def global_shuffle(self, fleet=None, thread_num=12):
        """
        :api_attr: Static Graph

        Global shuffle.
        Global shuffle can be used only in distributed mode. i.e. multiple
        processes on single machine or multiple machines training together.
        If you run in distributed mode, you should pass fleet instead of None.

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()

                dataset = paddle.distributed.InMemoryDataset()
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.load_into_memory()
                dataset.global_shuffle()

        Args:
            fleet(Fleet): fleet singleton. Default None.
            thread_num(int): shuffle thread num. Default is 12.

        """
        trainer_num = 1
        if fleet is not None:
            fleet._role_maker.barrier_worker()
            trainer_num = fleet.worker_num()
        if self.fleet_send_batch_size is None:
            self.fleet_send_batch_size = 1024
        if self.fleet_send_sleep_seconds is None:
            self.fleet_send_sleep_seconds = 0
        self.dataset.register_client2client_msg_handler()
        self.dataset.set_trainer_num(trainer_num)
        self.dataset.set_fleet_send_batch_size(self.fleet_send_batch_size)
        self.dataset.set_fleet_send_sleep_seconds(self.fleet_send_sleep_seconds)
        if fleet is not None:
            fleet._role_maker.barrier_worker()
        self.dataset.global_shuffle(thread_num)
        if fleet is not None:
            fleet._role_maker.barrier_worker()
        if self.merge_by_lineid:
            self.dataset.merge_by_lineid()
        if fleet is not None:
            fleet._role_maker.barrier_worker()

    def release_memory(self):
        """
        :api_attr: Static Graph
        
        Release InMemoryDataset memory data, when data will not be used again.

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()
                
                dataset = paddle.distributed.InMemoryDataset()
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.load_into_memory()
                dataset.global_shuffle()
                exe = paddle.static.Executor(paddle.CPUPlace())
                startup_program = paddle.static.Program()
                main_program = paddle.static.Program()
                exe.run(startup_program)
                exe.train_from_dataset(main_program, dataset)
                dataset.release_memory()

        """
        self.dataset.release_memory()

    def get_memory_data_size(self, fleet=None):
        """
        :api_attr: Static Graph

        Get memory data size, user can call this function to know the num
        of ins in all workers after load into memory.

        Note:
            This function may cause bad performance, because it has barrier

        Args:
            fleet(Fleet): Fleet Object.

        Returns:
            The size of memory data.

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()

                dataset = paddle.distributed.InMemoryDataset()
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.load_into_memory()
                print dataset.get_memory_data_size()

        """
        import numpy as np
        local_data_size = self.dataset.get_memory_data_size()
        local_data_size = np.array([local_data_size])
        if fleet is not None:
            global_data_size = local_data_size * 0
            fleet._role_maker.all_reduce_worker(local_data_size,
                                                global_data_size)
            return global_data_size[0]
        return local_data_size[0]

    def get_shuffle_data_size(self, fleet=None):
        """
        :api_attr: Static Graph

        Get shuffle data size, user can call this function to know the num
        of ins in all workers after local/global shuffle.

        Note:
            This function may cause bad performance to local shuffle,
            because it has barrier. It does not affect global shuffle.

        Args:
            fleet(Fleet): Fleet Object.

        Returns:
            The size of shuffle data.

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()
                
                dataset = paddle.distributed.InMemoryDataset()
                dataset = paddle.distributed.InMemoryDataset()
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.load_into_memory()
                dataset.global_shuffle()
                print dataset.get_shuffle_data_size()

        """
        import numpy as np
        local_data_size = self.dataset.get_shuffle_data_size()
        local_data_size = np.array([local_data_size])
        if fleet is not None:
            global_data_size = local_data_size * 0
            fleet._role_maker.all_reduce_worker(local_data_size,
                                                global_data_size)
            return global_data_size[0]
        return local_data_size[0]

    def _set_fea_eval(self, record_candidate_size, fea_eval=True):
        """
        set fea eval mode for slots shuffle to debug the importance level of
        slots(features), fea_eval need to be set True for slots shuffle.
        
        Args:
            record_candidate_size(int): size of instances candidate to shuffle 
                                        one slot
            fea_eval(bool): whether enable fea eval mode to enable slots shuffle.
                            default is True.
            
        Examples:
            .. code-block:: python

            import paddle
            paddle.enable_static()
            dataset = paddle.distributed.InMemoryDataset()
            dataset._set_fea_eval(1000000, True)

        """
        if fea_eval:
            self.dataset.set_fea_eval(fea_eval, record_candidate_size)
        self.fea_eval = fea_eval

    def slots_shuffle(self, slots):
        """
        Slots Shuffle 
        Slots Shuffle is a shuffle method in slots level, which is usually used 
        in sparse feature with large scale of instances. To compare the metric, i.e.
        auc while doing slots shuffle on one or several slots with baseline to 
        evaluate the importance level of slots(features).
        
        Args:
            slots(list[string]): the set of slots(string) to do slots shuffle.

        Examples:
            .. code-block:: python

                import paddle
                paddle.enable_static()
                
                dataset = paddle.distributed.InMemoryDataset()
                dataset._init_distributed_settings(fea_eval=True)
                slots = ["slot1", "slot2", "slot3", "slot4"]
                slots_vars = []
                for slot in slots:
                    var = paddle.static.data(
                        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
                    slots_vars.append(var)
                dataset.init(
                    batch_size=1,
                    thread_num=2,
                    input_type=1,
                    pipe_command="cat",
                    use_var=slots_vars)
                filelist = ["a.txt", "b.txt"]
                dataset.set_filelist(filelist)
                dataset.load_into_memory()
                dataset.slots_shuffle(['slot1'])
        """
        if self.fea_eval:
            slots_set = set(slots)
            self.dataset.slots_shuffle(slots_set)


class QueueDataset(DatasetBase):
    """
    :api_attr: Static Graph

    QueueDataset, it will process data streamly.

    Examples:
        .. code-block:: python

          import paddle
          dataset = paddle.distributed.QueueDataset()

    """

    def __init__(self):
        """
        Initialize QueueDataset
        """
        super(QueueDataset, self).__init__()
        self.proto_desc.name = "MultiSlotDataFeed"

    def init(self, **kwargs):
        """
        :api_attr: Static Graph

        should be called only once in user's python scripts to initialize setings of dataset instance
        """
        super(QueueDataset, self).init(**kwargs)

    def _prepare_to_run(self):
        """
        Set data_feed_desc/thread num/filelist before run,
        user no need to call this function.
        """
        if self.thread_num > len(self.filelist):
            self.thread_num = len(self.filelist)
        if self.thread_num == 0:
            self.thread_num = 1
        self.dataset.set_thread_num(self.thread_num)
        self.dataset.set_filelist(self.filelist)
        self.dataset.set_data_feed_desc(self._desc())
        self.dataset.create_readers()


class FileInstantDataset(DatasetBase):
    """
    FileInstantDataset, it will process data streamly.

    Examples:
        .. code-block:: python

          import paddle
          dataset = paddle.distributed.fleet.FileInstantDataset()
    """

    def __init__(self):
        """
        Initialize FileInstantDataset
        """
        super(FileInstantDataset, self).__init__()
        self.proto_desc.name = "MultiSlotFileInstantDataFeed"

    def init(self, **kwargs):
        """
        should be called only once in user's python scripts to initialize setings of dataset instance
        """
        super(FileInstantDataset, self).init(**kwargs)


class BoxPSDataset(InMemoryDataset):
    """
    BoxPSDataset: derived from InMemoryDataset.

    Examples:
        .. code-block:: python

          import paddle
          dataset = paddle.distributed.fleet.BoxPSDataset()
    """

    def __init__(self):
        """
        Initialize BoxPSDataset
        """
        super(BoxPSDataset, self).__init__()
        self.boxps = core.BoxPS(self.dataset)
        self.proto_desc.name = "PaddleBoxDataFeed"

    def init(self, **kwargs):
        """
        should be called only once in user's python scripts to initialize setings of dataset instance
        """
        super(BoxPSDataset, self).init(**kwargs)

        rank_offset = kwargs.get("rank_offset", "")
        self._set_rank_offset(rank_offset)
        pv_batch_size = kwargs.get("pv_batch_size", 1)
        self._set_pv_batch_size(pv_batch_size)
        parse_logkey = kwargs.get("parse_logkey", False)
        self._set_parse_logkey(parse_logkey)
        merge_by_sid = kwargs.get("merge_by_sid", False)
        self._set_merge_by_sid(merge_by_sid)
        enable_pv_merge = kwargs.get("enable_pv_merge", False)
        self._set_enable_pv_merge(enable_pv_merge)

    def _set_rank_offset(self, rank_offset):
        """
        Set rank_offset for merge_pv. It set the message of Pv.

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              dataset._set_rank_offset("rank_offset")

        Args:
            rank_offset(str): rank_offset's name

        """
        self.proto_desc.rank_offset = rank_offset

    def _set_pv_batch_size(self, pv_batch_size):
        """
        Set pv batch size. It will be effective during enable_pv_merge

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              dataset._set_pv_batch_size(128)
        Args:
            pv_batch_size(int): pv batch size

        """
        self.proto_desc.pv_batch_size = pv_batch_size

    def _set_parse_logkey(self, parse_logkey):
        """
        Set if Dataset need to parse logkey

        Args:
            parse_content(bool): if parse logkey or not

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              dataset._set_parse_logkey(True)

        """
        self.parse_logkey = parse_logkey

    def _set_merge_by_sid(self, merge_by_sid):
        """
        Set if Dataset need to merge sid. If not, one ins means one Pv.

        Args:
            merge_by_sid(bool): if merge sid or not

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              dataset._set_merge_by_sid(True)

        """
        self.merge_by_sid = merge_by_sid

    def _set_enable_pv_merge(self, enable_pv_merge):
        """
        Set if Dataset need to merge pv.

        Args:
            enable_pv_merge(bool): if enable_pv_merge or not

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              dataset._set_enable_pv_merge(True)

        """
        self.enable_pv_merge = enable_pv_merge

    def set_date(self, date):
        """
        Workaround for date
        """
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:])
        self.boxps.set_date(year, month, day)

    def begin_pass(self):
        """
        Begin Pass
        Notify BoxPS to load sparse parameters of next pass to GPU Memory 

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              dataset.begin_pass()
        """
        self.boxps.begin_pass()

    def end_pass(self, need_save_delta):
        """
        End Pass
        Notify BoxPS that current pass ended 
        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              dataset.end_pass(True)
        """
        self.boxps.end_pass(need_save_delta)

    def wait_preload_done(self):
        """
        Wait async preload done
        Wait Until Feed Pass Done
        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              filelist = ["a.txt", "b.txt"]
              dataset.set_filelist(filelist)
              dataset.preload_into_memory()
              dataset.wait_preload_done()
        """
        self.boxps.wait_feed_pass_done()

    def load_into_memory(self):
        """
        Load next pass into memory and notify boxps to fetch its emb from SSD
        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              filelist = ["a.txt", "b.txt"]
              dataset.set_filelist(filelist)
              dataset.load_into_memory()
	    """
        self._prepare_to_run()
        self.boxps.load_into_memory()

    def preload_into_memory(self):
        """
        Begin async preload next pass while current pass may be training
        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              filelist = ["a.txt", "b.txt"]
              dataset.set_filelist(filelist)
              dataset.preload_into_memory()
        """
        self._prepare_to_run()
        self.boxps.preload_into_memory()

    def _dynamic_adjust_before_train(self, thread_num):
        if not self.is_user_set_queue_num:
            self.dataset.dynamic_adjust_channel_num(thread_num, True)
        self.dataset.dynamic_adjust_readers_num(thread_num)

    def _dynamic_adjust_after_train(self):
        pass

    def slots_shuffle(self, slots):
        """
        Slots Shuffle 
        Slots Shuffle is a shuffle method in slots level, which is usually used 
        in sparse feature with large scale of instances. To compare the metric, i.e.
        auc while doing slots shuffle on one or several slots with baseline to 
        evaluate the importance level of slots(features).
        
        Args:
            slots(list[string]): the set of slots(string) to do slots shuffle.

        Examples:
            import paddle
            dataset = paddle.distributed.fleet.BoxPSDataset()
            dataset.set_merge_by_lineid()
            #suppose there is a slot 0
            dataset.slots_shuffle(['0'])
        """
        slots_set = set(slots)
        self.boxps.slots_shuffle(slots_set)

    def set_current_phase(self, current_phase):
        """
        Set current phase in train. It is useful for untest.
        current_phase : 1 for join, 0 for update.

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              filelist = ["a.txt", "b.txt"]
              dataset.set_filelist(filelist)
              dataset.load_into_memory()
              dataset.set_current_phase(1)

        """
        self.dataset.set_current_phase(current_phase)

    def get_pv_data_size(self):
        """
        Get memory data size of Pv, user can call this function to know the pv num
        of ins in all workers after load into memory.

        Note:
            This function may cause bad performance, because it has barrier

        Returns:
            The size of memory pv data.

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              filelist = ["a.txt", "b.txt"]
              dataset.set_filelist(filelist)
              dataset.load_into_memory()
              print dataset.get_pv_data_size()

        """
        return self.dataset.get_pv_data_size()

    def preprocess_instance(self):
        """
        Merge pv instance and convey it from input_channel to input_pv_channel. 
        It will be effective when enable_pv_merge_ is True.

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              filelist = ["a.txt", "b.txt"]
              dataset.set_filelist(filelist)
              dataset.load_into_memory()
              dataset.preprocess_instance()

        """
        self.dataset.preprocess_instance()

    def postprocess_instance(self):
        """
        Divide pv instance and convey it to input_channel.

        Examples:
            .. code-block:: python

              import paddle
              dataset = paddle.distributed.fleet.BoxPSDataset()
              filelist = ["a.txt", "b.txt"]
              dataset.set_filelist(filelist)
              dataset.load_into_memory()
              dataset.preprocess_instance()
              exe.train_from_dataset(dataset)
              dataset.postprocess_instance()

        """
        self.dataset.postprocess_instance()
