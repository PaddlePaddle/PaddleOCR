# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import defaultdict
from paddle.fluid.framework import Program, Block, Operator
from paddle.fluid.framework import in_dygraph_mode
import paddle.fluid.core as core
import paddle.distributed.fleet as fleet
import numpy as np


class HybridParallelInferenceHelper(object):
    """
    A helper class to split program for inference with hybrid parallelism.
    
    Args:
        startup_program (Program): the startup program.
        main_program (Program): the main program.
        num_mp (int): number of model parallel degree. Default ``1``.
        num_pp (int): number of pipeline parallel degree. Default ``1``.
        micro_batch_size (int): number of micro batch size. Default ``1``.
        beam_size (int): number of beam search size. Default ``1``.
        init_comm (bool): wheter if initilize comminication group. Default ``True``.
        role_maker (RoleMakerBase or subclass): user custom define RoleMakerBase.
            If ``role_maker==None``, then use PaddleCloudRoleMaker. Default ``None``.
    
    Returns:
        None.
        
    Write Paradigm:
    
    .. code-block:: bash
        :name: bash-example1
        
        # while op pattern
        with paddle.fluid.device_guard(f'{device}:all'):
            # init global cond
            max_len = layers.fill_constant(shape=[1], dtype="int64", value=10, force_cpu=False)
            step_idx = layers.fill_constant(shape=[1], dtype="int64", value=0, force_cpu=False)
            cond_int = layers.fill_constant(shape=[1], dtype="int64", value=0, force_cpu=False, name="cond_int")
            cond = layers.cast(step_idx < max_len, dtype="bool")
            while_op = layers.While(cond, is_test=True)
            
            # init global lod_tensor_array for generation task
            arr = layers.array_write(data, step_idx)
            
        with while_op.block():
            with paddle.fluid.device_guard(f'{device}:all'):
                # read data from global lod_tensor_array
                element_in_arr = layers.array_read(array=arr, i=step_idx)
                # write placehold data to global lod_tensor_array,
                # it need for send_v2 of lod_tensor_array
                layers.increment(x=step_idx, value=1.0, in_place=True)
                layers.array_write(element_in_arr, i=step_idx, array=arr)
                
            with paddle.fluid.device_guard(f'{device}:0'):
                ... some code
                
            with paddle.fluid.device_guard(f'{device}:1'):
                ... some code
                
            with paddle.fluid.device_guard(f'{device}:{num_pp-1}'):
                # generate some data in while block and write to global lod_tensor_array
                # that they are read in next while step.
                # we will using send_v2 to send global lod_tensor_array to other pipeline and sync
                layers.array_write(other_var, i=step_idx, array=arr)
                
                # update cond and assign to cond_int, we will sync cond_int
                layers.assign(layers.cast(cond, dtype="int32"), cond_int)
                
            with paddle.fluid.device_guard(f'{model._device}:all'):
                # the code below must at end of while block and exists in device:all
                layers.assign(layers.cast(cond_int, dtype='bool'), cond)
                
        with paddle.fluid.device_guard(f'{model._device}:all'):
            # use a empty lod_tensor_array to clear lod_tensor_array
            layers.assign(layers.create_array(data.dtype), arr)
            
            
    Examples:
    
    .. code-block:: python
        :name: code-example1
    
        # required: distributed
        import os
        import numpy as np
        import paddle
        import paddle.fluid.layers as layers
        import paddle.distributed.fleet as fleet
        paddle.enable_static()

        nranks = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
        rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        dev_id = int(os.getenv("FLAGS_selected_gpus", 0))

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        if nranks > 1:
            dist_strategy = fleet.DistributedStrategy()
            dist_strategy.without_graph_optimization = True
            fleet.init(is_collective=True, strategy=dist_strategy)

        device = "gpu"

        with paddle.static.program_guard(main_program, startup_program):
            with paddle.fluid.device_guard(f'{device}:0'):
                X = paddle.static.data(name='X', shape=[None, 2], dtype='float32')

            with paddle.fluid.device_guard(f'{device}:all'):
                max_len = layers.fill_constant(
                    shape=[1], dtype="int64", value=5, force_cpu=False, name="n")
                step_idx = layers.fill_constant(
                    shape=[1], dtype="int64", value=0, force_cpu=False, name="i")

                data = layers.array_write(X, step_idx)

                cond_int = layers.fill_constant(shape=[1], dtype="int64", value=0, force_cpu=False, name="cond_int")
                cond = layers.less_than(x=step_idx, y=max_len)
                while_op = layers.While(cond, is_test=True)

            with while_op.block():
                with paddle.fluid.device_guard(f'{device}:all'):
                    input = layers.array_read(array=data, i=step_idx)
                    layers.increment(x=step_idx, value=1.0, in_place=True)
                    layers.array_write(input, i=step_idx, array=data)

                with paddle.fluid.device_guard(f'{device}:0'):
                    param_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
                    weight1 = paddle.static.create_parameter(
                        shape=[2, 5], dtype='float32', attr=param_attr, is_bias=False)
                    hidden1 = paddle.matmul(input, weight1)

                with paddle.fluid.device_guard(f'{device}:1'):
                    param_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(2.0))
                    weight2 = paddle.static.create_parameter(
                        shape=[5, 2], dtype='float32', attr=param_attr, is_bias=False)
                    hidden2 = paddle.matmul(hidden1, weight2)

                    layers.array_write(hidden2, i=step_idx, array=data)

                    # update cond and assign to cond_int, we will sync cond_int
                    layers.less_than(x=step_idx, y=max_len, cond=cond)
                    layers.assign(layers.cast(cond, dtype="int32"), cond_int)

                with paddle.fluid.device_guard(f'{device}:all'):
                    # the code below must at end of while block and exists in device:all
                    layers.assign(layers.cast(cond_int, dtype='bool'), cond)

            with paddle.fluid.device_guard(f'{device}:all'):
                out = layers.create_array(data.dtype)
                layers.assign(data, out)

            with paddle.fluid.device_guard(f'{device}:all'):
                # use a empty lod_tensor_array to clear lod_tensor_array
                layers.assign(layers.create_array(data.dtype), data)

        helper = fleet.HybridParallelInferenceHelper(startup_program, main_program, micro_batch_size=2, num_pp=2, init_comm=nranks>1)
        helper.gen_infer_program(['array_write_0.out'], ['cond_int.tmp_0'])

        exe = paddle.static.Executor(paddle.CUDAPlace(dev_id))
        exe.run(startup_program)
        
        np.random.seed(2333)
        for step in range(5):
            init_data = np.random.uniform(low=0.0, high=1.0, size=[2, 2]).astype('float32')
            [res] = exe.run(main_program, feed={"X": init_data}, fetch_list=[out])
            print('-------- step', step, ' --------')
            print(res)
    """

    def __init__(self,
                 startup_program,
                 main_program,
                 num_mp=1,
                 num_pp=1,
                 micro_batch_size=1,
                 beam_size=1,
                 init_comm=True,
                 role_maker=None):

        assert isinstance(startup_program, Program)
        assert isinstance(main_program, Program)

        self._device = None
        if core.is_compiled_with_npu():
            self._device = "npu"
        elif core.is_compiled_with_cuda():
            self._device = "gpu"
        assert self._device, "Only gpu and npu are supported."
        assert not in_dygraph_mode(), "Only static mode is supported."

        op_maker = core.op_proto_and_checker_maker
        self._op_role = op_maker.OpRole
        self._op_role_key = op_maker.kOpRoleAttrName()
        self._op_device_key = op_maker.kOpDeviceAttrName()

        self._param_device_map = dict()

        self._pipeline_pair = []
        self._pipeline_pair_in_while = []
        self._pp_ring_map = dict()
        self.ring_id = 20  # Just a magic number

        self.micro_batch_size = micro_batch_size
        self.beam_size = beam_size
        self.init_comm = init_comm

        self._output_var_to_op = None
        self._input_var_to_op = None
        self._main_program = main_program
        self._startup_program = startup_program

        if role_maker is None:
            self.role_maker = fleet.base.role_maker.PaddleCloudRoleMaker(
                is_collective=True)
        else:
            if isinstance(role_maker, fleet.base.role_maker.RoleMakerBase):
                assert role_maker._is_collective == True
                self.role_maker = role_maker

        # communication_group info
        self.mp_ring_id = 0
        self.global_ring_id = 1

        self.endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.endpoints[self.role_maker._worker_index()]
        self.rank = self.role_maker._worker_index()
        self.nranks = self.role_maker._worker_num()
        assert num_mp * num_pp == self.nranks
        self.num_pp = num_pp
        self.num_mp = num_mp

        # global ring info
        self.global_endpoints = self.endpoints
        self.global_rank = self.rank
        self.global_nranks = self.nranks

        arr = np.arange(0, self.num_pp * self.num_mp).reshape(
            [self.num_pp, self.num_mp])
        ipp, imp = np.where(arr == self.rank)
        ipp = ipp[0]
        imp = imp[0]
        self.mp_group = arr[ipp, :]
        self.pp_group = arr[:, imp]

        self._stage = ipp

    def _init_communication_group(self):
        dev_ids = []
        for pair in self._pipeline_pair:
            prev_id, cur_id = pair
            if prev_id not in dev_ids:
                dev_ids.append(prev_id)
            if cur_id not in dev_ids:
                dev_ids.append(cur_id)
        num_pp = len(dev_ids)
        num_pp = max(1, num_pp)
        assert num_pp == self.num_pp, 'num_pp: {}, self.num_pp: {}'.format(
            num_pp, self.num_pp)

        collective_helper = fleet.meta_optimizers.common.CollectiveHelper(
            self.role_maker, wait_port=False)

        # Create global rings
        collective_helper._init_communicator(
            self._startup_program, self.current_endpoint, self.global_endpoints,
            self.global_rank, self.global_ring_id, True, self.global_ring_id,
            True)

        # Create mp rings
        if self.num_mp > 1:
            mp_endpoints = [self.endpoints[mp_idx] for mp_idx in self.mp_group]
            mp_rank = [
                idx for idx, mp_idx in enumerate(self.mp_group)
                if mp_idx == self.rank
            ][0]
            collective_helper._init_communicator(
                self._startup_program, self.current_endpoint, mp_endpoints,
                mp_rank, self.mp_ring_id, True, self.global_ring_id, True)

        # Create pipeline rings
        if self.num_pp > 1:
            for pair in self._pipeline_pair:
                pair_key = pair[0] * 1000 + pair[1]
                ring_id = self._pp_ring_map[pair_key]

                first_node = self.pp_group[pair[0]]
                second_node = self.pp_group[pair[1]]
                if self.rank != first_node and self.rank != second_node:
                    collective_helper._init_communicator(
                        self._startup_program, None, None, None, None, False,
                        self.global_ring_id, True)
                    continue

                pipeline_endpoints = [
                    self.endpoints[first_node], self.endpoints[second_node]
                ]
                pipeline_rank = 0 if self.rank == first_node else 1
                collective_helper._init_communicator(
                    self._startup_program, self.current_endpoint,
                    pipeline_endpoints, pipeline_rank, ring_id, False,
                    self.global_ring_id, True)

    def _get_input_output_info(self, block):
        '''
        Get info of op input and output.
        '''
        # A map from output var to op which generate it.
        output_var_to_op = defaultdict(list)
        # A map from var to op which takes it as input.
        input_var_to_op = defaultdict(list)

        for index, op in enumerate(block.ops):
            for var_name in op.input_arg_names:
                input_var_to_op[var_name].append([op, index])
            for var_name in op.output_arg_names:
                output_var_to_op[var_name].append([op, index])

        return output_var_to_op, input_var_to_op

    def _update_param_device_map(self):
        """
        Get the device info for parameters.
        """
        params = [param.name for param in self._main_program.all_parameters()]
        for each_block in self._main_program.blocks:
            for op in each_block.ops:
                for var_name in op.input_arg_names:
                    if not var_name in params or var_name in self._param_device_map:
                        continue
                    device = op.attr(self._op_device_key)

                    self._param_device_map[var_name] = device

    def _split_program(self, program, stage, block_idx):
        """
        Split a program and get the one with the given pipeline stage.

        Args:
            stage (int): pipeline stage
            block_idx (int): block index
            
        Returns:
            used_var_names (set): used var names in block_idx block
        """

        used_var_names = set()
        block = program.block(block_idx)
        op_idx = 0
        for op in list(block.ops):
            op_stage = op.attr(self._op_device_key).split(':')[1]
            # Copy ops whose op_device set to "gpu:all" to all sections.
            if op_stage == "all" or int(op_stage) == stage:
                op_idx += 1
                if op.type == "while":
                    sub_block_id = int(op.attr('sub_block').id)
                    sub_used_var_names = self._split_program(program, stage,
                                                             sub_block_id)

                    used_var_names.update(sub_used_var_names)

                    input_idxs = []
                    input_arg_names = op.input("X")
                    for i, name in enumerate(input_arg_names):
                        if name not in sub_used_var_names:
                            input_idxs.append(i)
                    if len(input_idxs) > 0:
                        for i in reversed(input_idxs):
                            input_arg_names.pop(i)
                        op.desc.set_input("X", input_arg_names)

                    output_idxs = []
                    output_arg_names = op.output("Out")
                    for i, name in enumerate(output_arg_names):
                        if name not in sub_used_var_names:
                            output_idxs.append(i)
                    if len(output_idxs) > 0:
                        for i in reversed(output_idxs):
                            output_arg_names.pop(i)
                        op.desc.set_output("Out", output_arg_names)

                for var_name in op.input_arg_names + op.output_arg_names:
                    used_var_names.add(var_name)
            else:
                block._remove_op(op_idx)

        for var_name in list(block.vars.keys()):
            if not var_name in used_var_names:
                block._remove_var(var_name)

        return used_var_names

#     def _find_post_op(self, index, var_name):
#         """
#         Find the post op that has variable named var_name as input.
#         """
#         # bugfix for uniform hybrid parallelism
#         if '.cast_fp32' in var_name:
#             var_name = var_name.replace('.cast_fp32', '')
#         if '.cast_fp16' in var_name:
#             var_name = var_name.replace('.cast_fp16', '')

#         post_ops = self._input_var_to_op[var_name]
#         if post_ops == None: return None
#         result_op = None
#         for post_op, post_idx in reversed(post_ops):
#             if post_idx > index:
#                 result_op = post_op
#                 break
#         return result_op

    def _find_prev_op(self, index, var_name):
        """
        Find the previous op of op with index that outputs
        variable named var_name.
        """
        prev_ops = self._output_var_to_op[var_name]
        if prev_ops == None: return None
        result_op = None
        for prev_op, prev_idx in reversed(prev_ops):
            if prev_idx < index:
                result_op = prev_op
                break
        return result_op

    def _add_op_device_attr(self, block):
        """
        Add op_device attrribute for ops in block that have 
        not that attribute set.
        
        Args:
            block (Block): the block to process.
        """
        assert isinstance(block, Block)

        # Ops should be copied to all pipeline stages.
        device_all_ops = [
            "create_py_reader",
            "read",
            "create_double_buffer_reader",
            "while",
        ]

        for op in block.ops:
            if op.type in device_all_ops:
                # We use "gpu:all" to represent an op should be put on all
                # pipeline stages, such as read ops. Note that: "gpu:all"
                # is only used by pipeline as an indicator.
                op._set_attr(self._op_device_key, self._device + ":all")
            if op.type == "while":
                sub_block_id = op.attr('sub_block').id
                sub_block = block.program.block(sub_block_id)
                self._add_op_device_attr(sub_block)

    def _check_validation(self, block):
        """
        Check whether ops in a block have both the op_device and the 
        op_role attributes set.
        """
        assert isinstance(block, Block)

        pre_stage_id = None
        for op in block.ops:
            assert op.has_attr(self._op_role_key), (
                "{} has no {} set .".format(op.type, self._op_role_key))
            op_role = op.attr(self._op_role_key)
            assert op_role == int(self._op_role.Forward), (
                "Only forward is supported for inference.")
            if not op._has_kernel(op.type):
                assert op.type in ["while", "conditional_block"], (
                    "The only supported op without kernel is while.")
                sub_block_id = op.attr('sub_block').id
                sub_block = block.program.block(sub_block_id)
                self._check_validation(sub_block)
            assert op.has_attr(self._op_device_key), (
                "{} has no {} set.".format(op.type, self._op_device_key))

            device = op.attr(self._op_device_key)
            assert device, (
                "{} has no {} set.".format(op.type, self._op_device_key))
            if device.split(':')[1] == "all": continue

            dev_type = device.split(':')[0]
            assert dev_type == self._device
            stage_id = int(device.split(':')[1])
            pre_stage_id = stage_id

    def _insert_sendrecv_ops_for_boundaries(self, block, is_while_block):
        """
        Insert a pair of send and recv ops for every two
        consecutive ops on different devices.
        """
        # A map from var to device where op takes it as input,
        # avoiding multiple send and recv ops.
        input_var_to_device = dict()

        extra_index_info = {'index': 0, }

        for index, op in enumerate(list(block.ops)):
            cur_device = op.attr(self._op_device_key)
            if cur_device.split(':')[-1] == "all": continue
            for var_name in op.input_arg_names:
                if not block.has_var(var_name) and block._find_var_recursive(
                        var_name):
                    continue
                var = block.var(var_name)
                # skip data var
                if var.is_data: continue
                prev_device = None
                generate_ops = self._output_var_to_op.get(var_name)
                if generate_ops is None:
                    if var_name not in self._param_device_map:
                        continue
                    prev_device = self._param_device_map[var_name]

                prev_op = self._find_prev_op(index, var_name)

                if not prev_device:
                    prev_device = prev_op.attr(self._op_device_key) \
                        if prev_op else None

                if prev_device is None or prev_device.split(":")[-1] == "all":
                    continue

                if prev_device == cur_device: continue

                if var_name not in input_var_to_device:
                    input_var_to_device[var_name] = []
                if (cur_device, prev_device) in input_var_to_device[var_name]:
                    continue

                assert self._device == cur_device.split(':')[
                    0], "More than one device type found."
                device_type = cur_device.split(':')[0] + ':'

                def _insert_send_recv(cur_id, prev_id):
                    assert cur_id > prev_id
                    cur_dev = device_type + str(cur_id)
                    prev_dev = device_type + str(prev_id)
                    if (cur_dev, prev_dev) in input_var_to_device[var_name]:
                        return

                    if cur_id - prev_id > 1:
                        _insert_send_recv(cur_id - 1, prev_id)
                        _insert_send_recv(cur_id, cur_id - 1)
                        input_var_to_device[var_name].append(
                            (cur_dev, prev_dev))
                        return

                    assert cur_id - prev_id == 1
                    input_var_to_device[var_name].append((cur_dev, prev_dev))

                    op_role = op.attr(self._op_role_key)
                    var = block.vars[var_name]
                    pair = (prev_id, cur_id)
                    if is_while_block and pair not in self._pipeline_pair_in_while:
                        self._pipeline_pair_in_while.append(pair)

                    # 1000 is just a magic number
                    pair_key = prev_id * 1000 + cur_id
                    if pair not in self._pipeline_pair:
                        self._pipeline_pair.append(pair)
                        self._pp_ring_map[pair_key] = self.ring_id
                        ring_id = self.ring_id
                        self.ring_id += 1
                    else:
                        ring_id = self._pp_ring_map[pair_key]

                    block._insert_op_without_sync(
                        index=index + extra_index_info['index'],
                        type='send_v2',
                        inputs={'X': var},
                        attrs={
                            self._op_device_key: prev_dev,
                            self._op_role_key: op_role,
                            'use_calc_stream': True,
                            'peer': 1,
                            'ring_id': ring_id
                        })
                    extra_index_info['index'] += 1
                    var_shape = list(var.shape)
                    if var_shape[0] < 0:
                        if is_while_block:
                            var_shape[
                                0] = self.micro_batch_size * self.beam_size
                        else:
                            var_shape[0] = self.micro_batch_size

                    block._insert_op_without_sync(
                        index=index + extra_index_info['index'],
                        type='recv_v2',
                        outputs={'Out': [var]},
                        attrs={
                            'out_shape': var_shape,
                            'dtype': var.dtype,
                            self._op_device_key: cur_dev,
                            self._op_role_key: op_role,
                            'use_calc_stream': True,
                            'peer': 0,
                            'ring_id': ring_id
                        })
                    extra_index_info['index'] += 1

                _insert_send_recv(
                    int(cur_device.split(':')[1]),
                    int(prev_device.split(':')[1]))
        block._sync_with_cpp()

    def _insert_sendrecv_ops_in_while_block(
            self, block, sync_in_while_lastpp2firstpp_var_names,
            sync_in_while_var_names, stage):
        dev_ids = []
        for pair in self._pipeline_pair_in_while:
            prev_id, cur_id = pair
            if prev_id not in dev_ids:
                dev_ids.append(prev_id)
            if cur_id not in dev_ids:
                dev_ids.append(cur_id)

        if len(dev_ids) == 0:
            return

        first_id = min(dev_ids)
        last_id = max(dev_ids)

        assert len(block.ops) > 2, "It must have more than 2 ops in while sub block, " \
            "layers.assign(layers.cast(cond_int, dtype='bool'), cond) must at end of while block, " \
            "because nccl cannot send bool dtype var"
        index = len(block.ops) - 2

        for prev_id in dev_ids:
            if prev_id == cur_id: continue
            assert cur_id > prev_id

            pair = (prev_id, cur_id)
            # 1000 is just a magic number
            pair_key = prev_id * 1000 + cur_id
            if pair not in self._pipeline_pair:
                self._pipeline_pair.append(pair)
                self._pp_ring_map[pair_key] = self.ring_id
                ring_id = self.ring_id
                self.ring_id += 1
            else:
                ring_id = self._pp_ring_map[pair_key]

            if cur_id == last_id and prev_id == first_id:
                var_names = sync_in_while_lastpp2firstpp_var_names + sync_in_while_var_names
            else:
                var_names = sync_in_while_var_names

            for var_name in var_names:
                var = block._var_recursive(var_name)
                if stage == cur_id:
                    block._insert_op_without_sync(
                        index=index,
                        type='send_v2',
                        inputs={'X': var},
                        attrs={
                            self._op_device_key:
                            self._device + ':' + str(cur_id),
                            self._op_role_key: int(self._op_role.Forward),
                            'use_calc_stream': True,
                            'peer': 0,
                            'ring_id': ring_id
                        })
                else:
                    var_shape = list(var.shape)
                    var_shape[0] = self.micro_batch_size if var_shape[
                        0] < 0 else var_shape[0]
                    block._insert_op_without_sync(
                        index=index,
                        type='recv_v2',
                        outputs={'Out': [var]},
                        attrs={
                            'out_shape': var_shape,
                            'dtype': var.dtype,
                            self._op_device_key:
                            self._device + ':' + str(prev_id),
                            self._op_role_key: int(self._op_role.Forward),
                            'use_calc_stream': True,
                            'peer': 1,
                            'ring_id': ring_id
                        })
                index += 1
        block._sync_with_cpp()

    def _get_while_block(self):
        """
        Get the while sub-block.
        """
        main_block = self._main_program.global_block()
        num_while = 0
        sub_block_id = None
        for op in main_block.ops:
            assert num_while < 2, "More than one while op found."
            if op.type == 'while':
                sub_block_id = op.attr('sub_block').id
                num_while += 1
        if sub_block_id: return op, self._main_program.block(sub_block_id)
        return None, None

    def gen_infer_program(self,
                          sync_in_while_lastpp2firstpp_var_names=None,
                          sync_in_while_var_names=None,
                          debug=False):
        """
        Generate inference program.
        Params:
            sync_in_while_lastpp2firstpp_var_names (list(str)): the vars in the last pipeline 
                that need to send var to first pipeline and exclude bool dtype var
            sync_in_while_var_names (list(str)): the vars sync among all pipeline in while block
                e.g cond. Note that cond cannot be bool dtype.
            debug (bool): the flag indicate debug
        """
        main_block = self._main_program.global_block()
        startup_block = self._startup_program.global_block()

        if debug:
            with open(f'main_program.txt', 'w') as f:
                f.write(str(self._main_program))
            with open(f'startup_program.txt', 'w') as f:
                f.write(str(self._startup_program))

        # step1: add op_device attribute for all ops
        self._add_op_device_attr(startup_block)
        self._check_validation(startup_block)
        self._add_op_device_attr(main_block)
        self._check_validation(main_block)

        # step2: add send/recv ops
        self._update_param_device_map()
        # step2.1: add send/recv for main_block
        out_var_to_op, in_var_to_op = self._get_input_output_info(main_block)
        self._output_var_to_op = out_var_to_op
        self._input_var_to_op = in_var_to_op
        self._insert_sendrecv_ops_for_boundaries(main_block, False)

        # step2.2: add send/recv for while_block
        while_op, while_block = self._get_while_block()
        if while_block:
            out_var_to_op, in_var_to_op = self._get_input_output_info(
                while_block)
            self._output_var_to_op = out_var_to_op
            self._input_var_to_op = in_var_to_op

            self._insert_sendrecv_ops_for_boundaries(while_block, True)

            self._insert_sendrecv_ops_in_while_block(
                while_block, sync_in_while_lastpp2firstpp_var_names,
                sync_in_while_var_names, self._stage)

        # step3: split programs
        self._split_program(self._startup_program, self._stage, 0)
        self._split_program(self._main_program, self._stage, 0)

        if debug:
            with open(f'main_program.txt.{self.rank}', 'w') as f:
                f.write(str(self._main_program))
            with open(f'startup_program.txt.{self.rank}', 'w') as f:
                f.write(str(self._startup_program))

        if self.init_comm:
            self._init_communication_group()
