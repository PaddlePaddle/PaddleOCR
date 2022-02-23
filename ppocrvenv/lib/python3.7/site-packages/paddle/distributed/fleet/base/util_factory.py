#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Fleet Utils."""
"""distributed operations"""
"""basic collective operations in python"""
"""remote file system"""

from ..utils.fs import FS, LocalFS, HDFSClient
from paddle.fluid.proto import framework_pb2
from paddle.fluid.framework import Program
from paddle.fluid import debugger
from google.protobuf import text_format
import paddle.fluid as fluid
from collections import OrderedDict
from paddle.fluid import core
import subprocess
import os
import numpy as np

__all__ = []


class UtilFactory(object):
    def _create_util(self, context=None):
        util = UtilBase()
        if context is not None and "valid_strategy" in context:
            util._set_strategy(context["valid_strategy"])
        if context is not None and "role_maker" in context:
            util._set_role_maker(context["role_maker"])
        return util


class UtilBase(object):
    def __init__(self):
        self.role_maker = None
        self.dist_strategy = None

    def _set_strategy(self, dist_strategy):
        self.dist_strategy = dist_strategy

    def _set_role_maker(self, role_maker):
        self.role_maker = role_maker

    def _set_file_system(self, fs_client):
        assert isinstance(
            fs_client, FS
        ), "fs_client must be the instance of paddle.distributed.fleet.utils.FS"
        self.fs_client = fs_client

    def all_reduce(self, input, mode="sum", comm_world="worker"):
        """
        All reduce `input` between specified collection. This is a distributed API.

        Args:
            input (list|numpy.array): The input variable to do all_reduce between specified collection.
            mode (str): "sum" or "min" or "max".
            comm_world (str, optional): Collection used to execute all_reduce operation. Supported collections incude `worker` , `server` and `all` . The default is `worker` .

        Returns:
            output(Numpy.array|None): A numpy array with the same shape as the `input` .

        Examples:
            .. code-block:: python

                # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .
                import paddle.distributed.fleet as fleet
                from paddle.distributed.fleet import PaddleCloudRoleMaker
                import sys
                import numpy as np
                import os

                os.environ["PADDLE_WITH_GLOO"] = "2"

                def train():
                    role = PaddleCloudRoleMaker(
                        is_collective=False,
                        init_gloo=True,
                        path="./tmp_gloo")
                    fleet.init(role)

                    if fleet.is_server():
                        input = [1, 2]
                        output = fleet.util.all_reduce(input, "sum", "server")
                        print(output)
                        # [2, 4]
                    elif fleet.is_worker():
                        input = np.array([3, 4])
                        output = fleet.util.all_reduce(input, "sum", "worker")
                        print(output)
                        # [6, 8]
                    output = fleet.util.all_reduce(input, "sum", "all")
                    print(output)
                    # [8, 12]
                if __name__ == "__main__":
                    train()
        """
        return self.role_maker._all_reduce(input, mode, comm_world)

    def barrier(self, comm_world="worker"):
        """
        Barrier between specified collection.

        Args:
            comm_world (str, optional): Collection used to execute barrier operation. Supported collections incude `worker` , `server` and `all` . The default is `worker` .

        Examples:

            .. code-block:: python

                # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .

                import paddle.distributed.fleet as fleet
                from paddle.distributed.fleet import PaddleCloudRoleMaker
                import sys
                import os

                os.environ["PADDLE_WITH_GLOO"] = "2"

                def train():
                    role = PaddleCloudRoleMaker(
                        is_collective=False,
                        init_gloo=True,
                        path="./tmp_gloo")
                    fleet.init(role)

                    if fleet.is_server():
                        fleet.util.barrier("server")
                        print("all server arrive here")
                    elif fleet.is_worker():
                        fleet.util.barrier("worker")
                        print("all server arrive here")
                    fleet.util.barrier("all")
                    print("all servers and workers arrive here")

                if __name__ == "__main__":
                    train()
        """
        self.role_maker._barrier(comm_world)

    def all_gather(self, input, comm_world="worker"):
        """
        All gather `input` between specified collection.

        Args:
            input (Int|Float): The input variable to do all_gather between specified collection.
            comm_world (str, optional): Collection used to execute all_reduce operation. Supported collections incude `worker` , `server` and `all` . The default is `worker` .

        Returns:
            output (List): A list of gathered values.

        Examples:

            .. code-block:: python

                # Save the following code in `train.py` , and then execute the command `fleetrun --server_num 2 --worker_num 2 train.py` .
                import paddle.distributed.fleet as fleet
                from paddle.distributed.fleet import PaddleCloudRoleMaker
                import sys
                import os

                os.environ["PADDLE_WITH_GLOO"] = "2"

                def train():
                    role = PaddleCloudRoleMaker(
                        is_collective=False,
                        init_gloo=True,
                        path="./tmp_gloo")
                    fleet.init(role)

                    if fleet.is_server():
                        input = fleet.server_index()
                        output = fleet.util.all_gather(input, "server")
                        print(output)
                        # output = [0, 1]
                    elif fleet.is_worker():
                        input = fleet.worker_index()
                        output = fleet.util.all_gather(input, "worker")
                        # output = [0, 1]
                        print(output)
                    output = fleet.util.all_gather(input, "all")
                    print(output)
                    # output = [0, 1, 0, 1]

                if __name__ == "__main__":
                    train()
        """

        return self.role_maker._all_gather(input, comm_world)

    def _broadcast(self):
        pass

    def _scatter(self):
        pass

    def get_file_shard(self, files):
        """
        Split files before distributed training, and return filelist assigned to the current trainer.

        .. code-block:: text

            example 1: files is [a, b, c ,d, e]  and trainer_num = 2, then trainer
                    0 gets [a, b, c] and trainer 1 gets [d, e].
            example 2: files is [a, b], and trainer_num = 3, then trainer 0 gets
                    [a], trainer 1 gets [b],  trainer 2 gets []

        Args:
            files(list): File list need to be read.

        Returns:
            List: Files belong to this worker.

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                from paddle.distributed.fleet import UserDefinedRoleMaker

                role = UserDefinedRoleMaker(
                    is_collective=False,
                    init_gloo=False,
                    current_id=0,
                    role=fleet.Role.WORKER,
                    worker_endpoints=["127.0.0.1:6003", "127.0.0.1:6004"],
                    server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])
                fleet.init(role)

                files = fleet.util.get_file_shard(["file1", "file2", "file3"])
                print(files)
                # files = ["file1", "file2"]
        """
        if not isinstance(files, list):
            raise TypeError("files should be a list of file need to be read.")

        trainer_id = self.role_maker._worker_index()
        trainers = self.role_maker._worker_num()

        remainder = len(files) % trainers
        blocksize = int(len(files) / trainers)

        blocks = [blocksize] * trainers
        for i in range(remainder):
            blocks[i] += 1

        trainer_files = [[]] * trainers
        begin = 0
        for i in range(trainers):
            trainer_files[i] = files[begin:begin + blocks[i]]
            begin += blocks[i]

        return trainer_files[trainer_id]

    def print_on_rank(self, message, rank_id):
        """
        Woker of rank `rank_id` print some message. 

        Args:
            message(str): Log to be printed.
            rank_id(int): trainer id.

        Examples:

            .. code-block:: python

                import paddle.distributed.fleet as fleet
                from paddle.distributed.fleet import UserDefinedRoleMaker

                role = UserDefinedRoleMaker(
                    is_collective=False,
                    init_gloo=False,
                    current_id=0,
                    role=fleet.Role.WORKER,
                    worker_endpoints=["127.0.0.1:6003", "127.0.0.1:6004"],
                    server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])
                fleet.init(role)

                fleet.util.print_on_rank("I'm worker 0", 0)
        """
        if self.role_maker._worker_index() != rank_id:
            return
        print(message)

    def _save_program(self, program, model_filename='__model__', is_text=False):
        if is_text:
            with open(model_filename, "w") as f:
                f.write(str(program))
        else:
            with open(model_filename, "wb") as f:
                f.write(program.desc.serialize_to_string())

    def _load_program(self, path, is_text):
        def load_program_binary(path):
            """load program from binary string file"""
            with open(path, "rb") as f:
                program_desc_str = f.read()
            return Program.parse_from_string(program_desc_str)

        def load_program_text(path):
            """load program from human-readable text file"""
            with open(path, "r") as f:
                program_desc_text = f.read()

            prog_desc = framework_pb2.ProgramDesc()
            text_format.Merge(program_desc_text, prog_desc)
            return Program.parse_from_string(prog_desc.SerializeToString())

        if is_text:
            return load_program_text(path)
        else:
            return load_program_binary(path)

    def _program_type_trans(self, prog_dir, prog_fn, is_text):
        prog = self._load_program(os.path.join(prog_dir, prog_fn), is_text)
        prog_out_fn = prog_fn + ".bin" if is_text else prog_fn + ".pbtxt"
        self._save_program(prog,
                           os.path.join(prog_dir, prog_out_fn), 1 - is_text)
        return prog_out_fn

    def _visualize_graphviz(self, program, output_dir, output_filename):
        block = program.global_block()
        dot_path = os.path.join(output_dir, output_filename + '.dot')
        pdf_path = os.path.join(output_dir, output_filename + '.pdf')
        debugger.draw_block_graphviz(block, path=dot_path)
        cmd = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        p.wait()

    def _proto_check(self, config):
        train_prog = self._load_program(config.train_prog_path,
                                        config.is_text_train_program)
        pruned_prog = self._load_program(config.pruned_prog_path,
                                         config.is_text_pruned_program)

        is_match = True

        pruned_vars = [(v.name, v) for v in pruned_prog.list_vars()
                       if fluid.io.is_persistable(v)]
        pruned_vars = OrderedDict(pruned_vars)
        pruned_vars_name = [name for name in pruned_vars]
        print("persistable vars in pruned program: {}".format(pruned_vars_name))

        # feed and fetch op is added in pruned program when pruning, not need to be found in train program
        feed_fetch_type_list = [
            core.VarDesc.VarType.FEED_MINIBATCH, core.VarDesc.VarType.FETCH_LIST
        ]

        for var_name in pruned_vars:
            var = pruned_vars[var_name]
            # feed and fetch op is added in pruned program when pruning, not need to be found in train program
            if var.type in feed_fetch_type_list:
                break
            try:
                train_prog_var = train_prog.global_block().var(var_name)
            except ValueError as e:
                print(
                    "Not find variable '%s' in train program. please check pruning."
                    % var_name)
                is_match = False
                continue
            if var.shape != train_prog_var.shape or var.dtype != train_prog_var.dtype:
                print(
                    "variable: {} not match. in pruned program shape: {} dtype:{}, in train program shape: {} dtype: {}".
                    format(var_name, var.shape, var.dtype, train_prog_var.shape,
                           train_prog_var.dtype))
                is_match = False
        return is_match

    def _params_check(self, config):
        def feed_gen(batch_size, feeded_vars_dims, feeded_vars_filelist):
            def reader(batch_size, fn, dim):
                data = []
                if isinstance(dim, list) or isinstance(dim, tuple):
                    shape = list(dim)
                    _temp = 1
                    for x in dim:
                        _temp = _temp * x
                    dim = _temp
                else:
                    shape = [dim]

                shape = [batch_size] + shape
                dim = dim * batch_size

                for line in open(fn, 'r'):
                    fields = line.strip().split(' ')
                    fields = [float(d) for d in fields]
                    while len(fields) >= dim:
                        tmp = fields[:dim]
                        fields = fields[dim:]
                        data.append(np.array(tmp).reshape(shape))
                return data

            batch_feed = []
            for i, fn in enumerate(feeded_vars_filelist):
                batch_feed.append(reader(batch_size, fn, feeded_vars_dims[i]))
            return batch_feed

        prog = self._load_program(
            os.path.join(config.dump_model_dir, config.dump_program_filename),
            config.is_text_dump_program)
        if config.is_text_dump_program:
            model_filename = self._program_type_trans(
                config.dump_model_dir, config.dump_program_filename,
                config.is_text_dump_program)

        saved_params = [
            v for v in prog.list_vars() if fluid.io.is_persistable(v)
        ]
        print("persistable vars in dump program: {}".format(
            [v.name for v in saved_params]))

        def check_not_expected_ops(prog, not_expected_op_types):
            op_types_set = set()
            for op in prog.global_block().ops:
                if op.type in not_expected_op_types and op.type not in op_types_set:
                    op_types_set.add(op.type)
            return op_types_set

        not_expected_op_types = check_not_expected_ops(prog, ["lookup_table"])
        if len(not_expected_op_types) > 0:
            print(
                "find op type '{}' in program, please check if your program is pruned correctly !".
                format(list(not_expected_op_types)))
            return False

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            inference_program, feed_target_names, fetch_targets = \
                fluid.io.load_inference_model(config.dump_model_dir, exe, model_filename=model_filename,
                                              params_filename=config.save_params_filename)

            # check program vars and saved vars shape
            orig_para_shape = {
                each_var.name: tuple(each_var.desc.shape())
                for each_var in saved_params
            }
            for each_var in saved_params:
                var_temp = fluid.global_scope().find_var(each_var.name)
                assert var_temp != None, "can't not find var: " + each_var.name
                new_shape = (np.array(var_temp.get_tensor())).shape
                assert each_var.name in orig_para_shape, each_var.name + "MUST in var list"
                orig_shape = orig_para_shape.get(each_var.name)
                if new_shape != orig_shape:
                    raise RuntimeError(
                        "Shape not matching: the Program requires a parameter with a shape of ({}), "
                        "while the loaded parameter (namely [ {} ]) has a shape of  ({}).".
                        format(orig_shape, each_var.name, new_shape))

            # check feed/fetch vars in program and config
            feed_config = config.feed_config
            fetch_config = config.fetch_config
            fetch_targets_names = [v.name for v in fetch_targets]
            if not feed_target_names:
                print("warning! no feed targets in program.")
            if not fetch_targets_names:
                print("warning! no fetch targets in program.")
            fetch_list = fetch_targets
            feed_name_list = feed_target_names
            if feed_config.feeded_vars_names is not None and feed_target_names != feed_config.feeded_vars_names:
                print(
                    "warning! feed vars in program and config are diff: feed in program: {}. feed in config {}.".
                    format(feed_target_names, feed_config.feeded_vars_names))
                feed_name_list = feed_config.feeded_vars_names
                # remove feed op in inference_program. new feed op will be added in exe.run
                global_block = inference_program.global_block()
                need_to_remove_op_index = []
                for i, op in enumerate(global_block.ops):
                    op.desc.set_is_target(False)
                    if op.type == "feed":  # only remove feed op here
                        need_to_remove_op_index.append(i)
                for index in need_to_remove_op_index[::-1]:
                    global_block._remove_op(index)
            if fetch_config.fetch_vars_names is not None and fetch_targets_names != fetch_config.fetch_vars_names:
                print(
                    "warning! fetch vars in program and config are diff: fetch in program: {}. fetch in config {}.".
                    format(fetch_targets_names, fetch_config.fetch_vars_names))
                fetch_list = [
                    inference_program.global_block().var(i)
                    for i in fetch_config.fetch_vars_names
                ]
                # remove fetch op in inference_program. new fetch op will be added in exe.run
                global_block = inference_program.global_block()
                need_to_remove_op_index = []
                for i, op in enumerate(global_block.ops):
                    op.desc.set_is_target(False)
                    if op.type == "fetch":  # only remove fetch op here
                        need_to_remove_op_index.append(i)
                for index in need_to_remove_op_index[::-1]:
                    global_block._remove_op(index)

            # if fetch_list have lod tensor
            return_numpy = all([v.lod_level == 0 for v in fetch_list])

            # try dump fetch_targets
            feed_tensors = []
            assert len(feed_config.feeded_vars_names) == len(
                feed_config.feeded_vars_dims) == len(
                    feed_config.feeded_vars_types)
            # check program vars and feed tensor shape in config
            for i in range(len(feed_config.feeded_vars_names)):
                var = inference_program.global_block().var(
                    feed_config.feeded_vars_names[i])
                if not isinstance(feed_config.feeded_vars_dims[i],
                                  (list, tuple)):
                    tensor_shape = (feed_config.feeded_vars_dims[i], )
                else:
                    tensor_shape = tuple(feed_config.feeded_vars_dims[i])
                feed_config.feeded_vars_dims[i] = tensor_shape
                var_shape = var.shape[1:]
                if tensor_shape != var_shape:
                    raise RuntimeError(
                        "feed variable '{}' shape not match. infer program  shape: {}. feed tensor shape: {}".
                        format(feed_config.feeded_vars_names[i], var_shape,
                               tensor_shape))

            if not feed_config.feeded_vars_filelist:
                print("generate random feed vars.")
                for i in range(len(feed_config.feeded_vars_names)):
                    var = inference_program.global_block().var(
                        feed_config.feeded_vars_names[i])
                    # create fake feed tensor. if lod_level > 1, should create_lod_tensor()
                    if var.lod_level == 0:
                        feed_tensors.append(
                            np.array(
                                np.random.random(
                                    tuple([config.batch_size] + list(
                                        feed_config.feeded_vars_dims[i]))),
                                dtype=feed_config.feeded_vars_types[i]))
                    elif var.lod_level == 1:
                        t = np.array(
                            np.random.random(
                                tuple([config.batch_size] + list(
                                    feed_config.feeded_vars_dims[i]))),
                            dtype=feed_config.feeded_vars_types[i])
                        feed_tensors.append(
                            fluid.create_lod_tensor(t, [[1] * config.batch_size
                                                        ], place))
                    else:
                        raise RuntimeError(
                            "vars with lod_level >= 2 is not supported now in this infer program check tool."
                        )
                results = exe.run(inference_program,
                                  feed={
                                      name: feed_tensors[i]
                                      for i, name in enumerate(feed_name_list)
                                  },
                                  fetch_list=fetch_list,
                                  return_numpy=return_numpy)
            else:
                print("load feed vars from files: {}.".format(
                    feed_config.feeded_vars_filelist))
                feed_vars = [
                    inference_program.global_block().var(
                        feed_config.feeded_vars_names[i])
                    for i in range(len(feed_config.feeded_vars_names))
                ]
                feeder = fluid.DataFeeder(feed_list=feed_vars, place=place)
                batch_feed = feed_gen(config.batch_size,
                                      feed_config.feeded_vars_dims,
                                      feed_config.feeded_vars_filelist)
                slots = [batch_feed]
                results = exe.run(inference_program,
                                  feed=feeder.feed(slots),
                                  fetch_list=fetch_list,
                                  return_numpy=return_numpy)
            for i, v in enumerate(fetch_list):
                print("fetch_targets name: %s" % v.name)
                print("fetch_targets: {}".format(results[i]))
            return results
