#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Defination of trainers."""

import sys
import os
__all__ = [
    'TrainerDesc', 'MultiTrainer', 'DistMultiTrainer', 'PipelineTrainer',
    'HeterXpuTrainer', 'HeterPipelineTrainer'
]


class TrainerDesc(object):
    '''
    Set proto from python to c++.
    Can be initialized from train_desc.
    '''

    def __init__(self):
        '''
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        with open(proto_file, 'r') as f:
            text_format.Parse(f.read(), self.proto_desc)
        '''
        # Workaround for relative import in protobuf under python3
        # TODO: should be fixed
        cur_path = os.path.dirname(__file__)
        if cur_path not in sys.path:
            sys.path.append(cur_path)
        if cur_path + "/proto" not in sys.path:
            sys.path.append(cur_path + "/proto")

        from proto import trainer_desc_pb2
        self.proto_desc = trainer_desc_pb2.TrainerDesc()
        import multiprocessing as mp
        # set default thread num == cpu count
        self.proto_desc.thread_num = mp.cpu_count()
        self._fleet_desc = None
        self._device_worker = None
        self._program = None
        self._infer = False

    def _set_heter_info(self, ret):
        #ret = = fu.split_program_by_device(program)
        #start_list, end_list, send_list, recv_list, program_list = fu.split_program_by_device(program)
        #if len(start_list) != 3:
        #    print("start_list len=", len(start_list), " will not set heter info")
        #    return
        #for i in start_list[0]:
        #    self.proto_desc.op_run_start_idx.append(i)
        #for i in end_list[0]:
        #    self.proto_desc.op_run_end_idx.append(i)
        #for i in send_list[0]:
        #    self.proto_desc.op_run_send_list.append(i)
        #for i in recv_list[0]:
        #    self.proto_desc.op_run_recv_list.append(i)
        if ret is None:
            return
        #for i in ret[0]: # start_list[1]:
        #    self.proto_desc.xpu_start_idx.append(i)
        self.proto_desc.xpu_start_idx = ret[0]

        #for i in ret[1]:  #end_list[1]:
        #    self.proto_desc.o_end_idx.append(i)
        self.proto_desc.xpu_end_idx = ret[1]
        for i in ret[2]:  # send_list[1]:
            self.proto_desc.xpu_send_list.append(i)
        for i in ret[3]:  # recv_list[1]:
            self.proto_desc.xpu_recv_list.append(i)

        #for i in start_list[2]:
        #    self.proto_desc.op_run_end_start_idx.append(i)
        #for i in end_list[2]:
        #    self.proto_desc.op_run_end_idx.append(i)
        #for i in send_list[2]:
        #    self.proto_desc.op_run_end_send_list.append(i)
        #for i in recv_list[2]:
        #    self.proto_desc.op_run_end_recv_list.append(i)

    def _set_fetch_var_and_info(self, fetch_vars, fetch_info, print_period):
        # convert fetch_info to list
        fetch_info = list(fetch_info)
        for i, v in enumerate(fetch_vars):
            self.proto_desc.fetch_config.fetch_var_names.extend([v.name])
            self.proto_desc.fetch_config.fetch_var_str_format.extend(
                [fetch_info[i]])
        self.proto_desc.fetch_config.print_period = print_period

    def _set_debug(self, debug):
        self.proto_desc.debug = debug

    def _set_thread(self, thread_num):
        self.proto_desc.thread_num = thread_num

    def _set_device_worker(self, device_worker):
        self._device_worker = device_worker

    def _set_infer(self, infer):
        self._infer = infer

    def _set_fleet_desc(self, fleet_desc):
        self._fleet_desc = fleet_desc

    def _gen_trainer_desc(self):
        pass

    def _set_program(self, program):
        self._program = program

    def _set_trainer_id(self, trainer_id):
        self.proto_desc.trainer_id = trainer_id

    def _set_trainers(self, trainers):
        for trainer_num in trainers:
            self.proto_desc.trainers.append(trainer_num)

    def _set_use_cvm(self, use_cvm=False):
        self.proto_desc.use_cvm = use_cvm

    def _set_no_cvm(self, no_cvm=False):
        self.proto_desc.no_cvm = no_cvm

    def _set_scale_sparse_grad_with_batch_size(
            self, scale_sparse_gradient_with_batch_size=True):
        self.proto_desc.scale_sparse_gradient_with_batch_size = scale_sparse_gradient_with_batch_size

    def _set_scale_datanorm(self, scale_datanorm=-1):
        self.proto_desc.scale_datanorm = scale_datanorm

    def _set_dump_slot(self, dump_slot):
        self.proto_desc.dump_slot = dump_slot

    def _set_mpi_rank(self, mpi_rank):
        self.proto_desc.mpi_rank = mpi_rank

    def _set_mpi_size(self, mpi_size):
        self.proto_desc.mpi_size = mpi_size

    def _set_dump_fields(self, dump_fields):
        for field in dump_fields:
            self.proto_desc.dump_fields.append(field)

    def _set_dump_fields_path(self, path):
        self.proto_desc.dump_fields_path = path

    def _set_dump_file_num(self, dump_file_num):
        self.proto_desc.dump_file_num = dump_file_num

    def _set_user_define_dump_filename(self, user_define_dump_filename):
        self.proto_desc.user_define_dump_filename = user_define_dump_filename

    def _set_dump_converter(self, converter):
        self.proto_desc.dump_converter = converter

    def _set_enable_random_dump(self, enable_random_dump):
        self.proto_desc.enable_random_dump = enable_random_dump

    def _set_dump_interval(self, dump_interval):
        self.proto_desc.dump_interval = dump_interval

    def _set_random_with_lineid(self, random_with_lineid):
        self.proto_desc.random_with_lineid = random_with_lineid

    def _set_dump_param(self, dump_param):
        for param in dump_param:
            self.proto_desc.dump_param.append(param)

    def _set_worker_places(self, worker_places):
        for place in worker_places:
            self.proto_desc.worker_places.append(place)

    def _set_use_ps_gpu(self, use_ps_gpu=False):
        self.proto_desc.use_ps_gpu = use_ps_gpu

    def _set_thread_barrier(self, thread_barrier):
        self.proto_desc.thread_barrier = thread_barrier

    def _set_check_nan_var_names(self, check_nan_var_names):
        for var in check_nan_var_names:
            self.proto_desc.check_nan_var_names.append(var)

    def _set_loss_names(self, loss_names):
        for loss in loss_names:
            self.proto_desc.loss_names.append(loss)

    def _set_adjust_ins_weight(self, config_dict):
        self.proto_desc.adjust_ins_weight_config.need_adjust = \
                config_dict.get("need_adjust", False)
        self.proto_desc.adjust_ins_weight_config.nid_slot = \
                config_dict.get("nid_slot", "")
        self.proto_desc.adjust_ins_weight_config.nid_adjw_threshold = \
                config_dict.get("nid_adjw_threshold", 0.0)
        self.proto_desc.adjust_ins_weight_config.nid_adjw_ratio = \
                config_dict.get("nid_adjw_ratio", 0.0)
        self.proto_desc.adjust_ins_weight_config.ins_weight_slot = \
                config_dict.get("ins_weight_slot", "")

    def _set_copy_table_config(self, config_dict):
        config = self.proto_desc.copy_table_config
        config.need_copy = config_dict.get("need_copy", False)
        config.batch_num = config_dict.get("batch_num", 100)

        src_sparse_tables = config_dict.get("src_sparse_tables", [])
        if not isinstance(src_sparse_tables, list):
            src_sparse_tables = [src_sparse_tables]
        dest_sparse_tables = config_dict.get("dest_sparse_tables", [])
        if not isinstance(dest_sparse_tables, list):
            dest_sparse_tables = [dest_sparse_tables]
        if len(src_sparse_tables) != len(dest_sparse_tables):
            raise ValueError(
                "len(src_sparse_tables) != len(dest_sparse_tables)," \
                " %s vs %s" % (len(src_sparse_tables), \
                len(dest_sparse_tables)))
        for i in src_sparse_tables:
            config.src_sparse_tables.append(i)
        for i in dest_sparse_tables:
            config.dest_sparse_tables.append(i)

        src_dense_tables = config_dict.get("src_dense_tables", [])
        if not isinstance(src_dense_tables, list):
            src_dense_tables = [src_dense_tables]
        dest_dense_tables = config_dict.get("dest_dense_tables", [])
        if not isinstance(dest_dense_tables, list):
            dest_dense_tables = [dest_dense_tables]
        if len(src_dense_tables) != len(dest_dense_tables):
            raise ValueError(
                "len(src_dense_tables) != len(dest_dense_tables)," \
                " %s vs %s" % (len(src_dense_tables), \
                len(dest_dense_tables)))
        for i in src_dense_tables:
            config.src_dense_tables.append(i)
        for i in dest_dense_tables:
            config.dest_dense_tables.append(i)

        # user can also specify dense variables to copy,
        # instead of copy dense table
        src_var_list = config_dict.get("src_var_list", [])
        if not isinstance(src_var_list, list):
            src_var_list = [src_var_list]
        dest_var_list = config_dict.get("dest_var_list", [])
        if not isinstance(dest_var_list, list):
            dest_var_list = [dest_var_list]
        if len(src_var_list) != len(dest_var_list):
            raise ValueError(
                "len(src_var_list) != len(dest_var_list), %s vs" \
                " %s" % (len(src_var_list), len(dest_var_list)))
        for i in src_var_list:
            config.src_var_list.append(i)
        for i in dest_var_list:
            config.dest_var_list.append(i)

        dependency_map = config_dict.get("dependency_map", {})
        for key in dependency_map:
            m = config.table_denpendency_map.add()
            m.key = key
            values = dependency_map[key]
            if not isinstance(values, list):
                values = [values]
            if len(values) != 1:
                raise ValueError("dependency len %s != 1" % len(values))
            for value in values:
                m.values.append(value)
        config.dense_pull_after_copy = \
            config_dict.get("dense_pull_after_copy", True)
        config.enable_dependency = \
            config_dict.get("enable_dependency", False)
        config.sparse_copy_by_feasign = \
            config_dict.get("sparse_copy_by_feasign", True)

    def _desc(self):
        from google.protobuf import text_format
        return self.proto_desc.SerializeToString()

    def __str__(self):
        from google.protobuf import text_format
        return text_format.MessageToString(self.proto_desc)


class MultiTrainer(TrainerDesc):
    '''
    Implement of MultiTrainer.
    Can be init from TrainerDesc.
    '''

    def __init__(self):
        super(MultiTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(MultiTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(MultiTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "MultiTrainer"
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)


class DistMultiTrainer(TrainerDesc):
    """
    Implement of DistMultiTrainer.
    It's for Distributed training.
    """

    def __init__(self):
        super(DistMultiTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(DistMultiTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(DistMultiTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "DistMultiTrainer"
        if self._program == None:
            raise RuntimeError("None Program")
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)


class HeterXpuTrainer(TrainerDesc):
    """
    Implement of HeterXpuTrainer.
    It's for Distributed training.
    """

    def __init__(self):
        super(HeterXpuTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(HeterXpuTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(HeterXpuTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "HeterXpuTrainer"
        if self._program == None:
            raise RuntimeError("None Program")
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)


class PSGPUTrainer(TrainerDesc):
    """
    Implement of PSGPUTrainer.
    It's for Distributed training.
    """

    def __init__(self):
        super(PSGPUTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(PSGPUTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(PSGPUTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "PSGPUTrainer"
        if self._program == None:
            raise RuntimeError("None Program")
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)


class HeterPipelineTrainer(TrainerDesc):
    """
    Implement of HeterPipelineTrainer.
    It's for HeterPS Pipeline training.
    """

    def __init__(self):
        super(HeterPipelineTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(HeterPipelineTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(HeterPipelineTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "HeterPipelineTrainer"
        if self._program == None:
            raise RuntimeError("None Program")
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)


class PipelineTrainer(TrainerDesc):
    """
    Implement of PipelineTrainer.
    It's for Pipeline.
    """

    def __init__(self):
        super(PipelineTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(PipelineTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(PipelineTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "PipelineTrainer"
        if self._program == None:
            raise RuntimeError("None Program")
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)
