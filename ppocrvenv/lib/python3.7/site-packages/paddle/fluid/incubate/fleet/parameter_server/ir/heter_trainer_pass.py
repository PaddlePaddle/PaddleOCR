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

from __future__ import print_function
import warnings

import paddle.fluid.core as core
import paddle.fluid.framework as framework

from paddle.fluid.transpiler.details.program_utils import delete_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import find_heter_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import union_forward_gradient_op
from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import create_heter_program
from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import create_trainer_program
from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import find_block_joints
from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import find_op_input_output
from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import get_vars_name_in_block


def split_heter_worker_ops_pass(program, config, stage_id, device):
    """
    split heter worker program from origin-program
    1. find heter op (located on different device)
    2. find input&output of every heter-block
    3. create heter worker program, add listen&serv op
    """
    default_deveice = "cpu"
    program, heter_ops, _, program_block_ops = find_heter_ops(program,
                                                              default_deveice)
    if len(heter_ops) == 0:
        warnings.warn(
            "Currently running in Heter Parameter Server mode, but no OP running on heterogeneous devices, Please check your code."
        )
        return program

    program_block_ops = union_forward_gradient_op(program_block_ops)
    block_vars_detail = find_block_joints(program, program_block_ops, heter_ops)
    heter_program = framework.Program()
    create_heter_program(program, config, heter_program, program_block_ops,
                         heter_ops, block_vars_detail, device, stage_id)
    return heter_program


def split_trainer_ops_pass(program, config, default_device="cpu"):
    """
    split cpu-trainer program from origin-program
    1. find heter op (located on different device)
    2. find input&output of every heter-block
    3. create cpu-trainer program, add send&recv op 
    """
    # Todo: support user define default_device (MrChengmo)
    default_device_ = default_device
    program, heter_ops, default_ops, program_block_ops = find_heter_ops(
        program, default_device_)
    program_block_ops = union_forward_gradient_op(program_block_ops)

    block_vars_detail = find_block_joints(program, program_block_ops, heter_ops)
    trainer_program = program.clone()
    create_trainer_program(trainer_program, program, config, program_block_ops,
                           block_vars_detail)
    return trainer_program
