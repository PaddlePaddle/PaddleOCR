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

from __future__ import print_function, absolute_import
import os
import sys
import logging
import subprocess
import numpy as np
from collections import OrderedDict
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.log_helper import get_logger

from google.protobuf import text_format
from paddle.fluid import debugger
from paddle.fluid.framework import Program
from paddle.fluid.proto import framework_pb2

__all__ = [
    "load_program", "save_program", "program_type_trans",
    "check_saved_vars_try_dump", "parse_program", "check_pruned_program_vars",
    "graphviz"
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

persistable_vars_out_fn = "vars_persistable.log"
all_vars_out_fn = "vars_all.log"
ops_out_fn = "ops.log"

feed_fetch_type_list = [
    core.VarDesc.VarType.FEED_MINIBATCH, core.VarDesc.VarType.FETCH_LIST
]
not_expected_op_types = ["lookup_table"]


def load_program(model_filename, is_text=False):
    if is_text:
        return load_program_text(model_filename)
    return load_program_binary(model_filename)


def load_program_binary(model_filename):
    """load program from binary string file"""
    with open(model_filename, "rb") as f:
        program_desc_str = f.read()
    return Program.parse_from_string(program_desc_str)


def load_program_text(model_filename):
    """load program from human-readable text file"""
    with open(model_filename, "r") as f:
        program_desc_text = f.read()

    prog_desc = framework_pb2.ProgramDesc()
    text_format.Merge(program_desc_text, prog_desc)
    return Program.parse_from_string(prog_desc.SerializeToString())


def save_program(program, model_filename='__model__', is_text=False):
    if is_text:
        with open(model_filename, "w") as f:
            f.write(str(program))
    else:
        with open(model_filename, "wb") as f:
            f.write(program.desc.serialize_to_string())


def check_pruned_program_vars(train_prog, pruned_prog):
    is_match = True

    pruned_vars = [(v.name, v) for v in pruned_prog.list_vars()
                   if fluid.io.is_persistable(v)]
    pruned_vars = OrderedDict(pruned_vars)
    pruned_vars_name = [name for name in pruned_vars]
    logger.info("persistable vars in pruned program: {}".format(
        pruned_vars_name))

    for var_name in pruned_vars:
        var = pruned_vars[var_name]
        # feed and fetch op is added in pruned program when pruning, not need to be found in train program
        if var.type in feed_fetch_type_list:
            break
        try:
            train_prog_var = train_prog.global_block().var(var_name)
        except ValueError as e:
            logger.error(
                "not find variable '%s' in train program. please check pruning."
                % var_name)
            logger.error(e)
            continue
        if var.shape != train_prog_var.shape or var.dtype != train_prog_var.dtype:
            logger.error(
                "variable: {} not match. in pruned program shape: {} dtype:{}, in train program shape: {} dtype: {}".
                format(var_name, var.shape, var.dtype, train_prog_var.shape,
                       train_prog_var.dtype))
            is_match = False
    return is_match


def graphviz(block, output_dir="", filename='debug'):
    dot_path = os.path.join(output_dir, filename + '.dot')
    pdf_path = os.path.join(output_dir, filename + '.pdf')
    debugger.draw_block_graphviz(block, path=dot_path)
    cmd = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    p.wait()


def program_type_trans(prog_dir, prog_fn, is_text):
    prog = load_program(os.path.join(prog_dir, prog_fn), is_text)
    prog_out_fn = prog_fn + ".bin" if is_text else prog_fn + ".pbtxt"
    save_program(prog, os.path.join(prog_dir, prog_out_fn), 1 - is_text)
    return prog_out_fn


def append_save_op(block, var, path):
    block.append_op(
        type='save', inputs={'X': [var]}, outputs={},
        attrs={'file_path': path})


def append_load_op(block, var, path):
    block.append_op(
        type='load',
        inputs={},
        outputs={'Out': [var]},
        attrs={'file_path': path})


def save_var(np_array, var_name, shape_list, dtype, save_path):
    program = fluid.Program()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    with fluid.program_guard(program):
        d0_data = fluid.layers.data(var_name, shape=shape_list, dtype=dtype)
        append_save_op(program.global_block(), d0_data, save_path)
        exe.run(feed={var_name: np_array}, fetch_list=[])


def load_var(var_name, shape_list, dtype, save_path):
    program = fluid.Program()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    with fluid.program_guard(program):
        d0_data = fluid.layers.data(var_name, shape=shape_list, dtype=dtype)
        append_load_op(program.global_block(), d0_data, save_path)
        outs = exe.run(feed={}, fetch_list=[d0_data])
        return outs


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


def feed_gen(batch_size, feeded_vars_dims, feeded_vars_filelist):
    batch_feed = []
    for i, fn in enumerate(feeded_vars_filelist):
        batch_feed.append(reader(batch_size, fn, feeded_vars_dims[i]))
    return batch_feed


def try_load_model_vars(dump_dir, dump_prog_fn, is_text_dump_program,
                        batch_size, feed_config, fetch_config, save_filename,
                        saved_params):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        if is_text_dump_program:
            dump_prog_fn = program_type_trans(dump_dir, dump_prog_fn,
                                              is_text_dump_program)
        inference_program, feed_target_names, fetch_targets = \
            fluid.io.load_inference_model(dump_dir, exe, model_filename=dump_prog_fn,
                                          params_filename=save_filename)

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
        fetch_targets_names = [v.name for v in fetch_targets]
        if not feed_target_names:
            logger.warning("no feed targets in program.")
        if not fetch_targets_names:
            logger.warning("no fetch targets in program.")
        fetch_list = fetch_targets
        feed_name_list = feed_target_names
        if feed_config.feeded_vars_names is not None and feed_target_names != feed_config.feeded_vars_names:
            logger.warning(
                "feed vars in program and config are diff: feed in program: {}. feed in config {}.".
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
            logger.warning(
                "fetch vars in program and config are diff: fetch in program: {}. fetch in config {}.".
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
            feed_config.feeded_vars_dims) == len(feed_config.feeded_vars_types)
        # check program vars and feed tensor shape in config
        for i in range(len(feed_config.feeded_vars_names)):
            var = inference_program.global_block().var(
                feed_config.feeded_vars_names[i])
            if not isinstance(feed_config.feeded_vars_dims[i], (list, tuple)):
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
            logger.info("generate random feed vars.")
            for i in range(len(feed_config.feeded_vars_names)):
                var = inference_program.global_block().var(
                    feed_config.feeded_vars_names[i])
                # create fake feed tensor. if lod_level > 1, should create_lod_tensor()
                if var.lod_level == 0:
                    feed_tensors.append(
                        np.array(
                            np.random.random(
                                tuple([batch_size] + list(
                                    feed_config.feeded_vars_dims[i]))),
                            dtype=feed_config.feeded_vars_types[i]))
                elif var.lod_level == 1:
                    t = np.array(
                        np.random.random(
                            tuple([batch_size] + list(
                                feed_config.feeded_vars_dims[i]))),
                        dtype=feed_config.feeded_vars_types[i])
                    feed_tensors.append(
                        fluid.create_lod_tensor(t, [[1] * batch_size], place))
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
            logger.info("load feed vars from files: {}.".format(
                feed_config.feeded_vars_filelist))
            feed_vars = [
                inference_program.global_block().var(
                    feed_config.feeded_vars_names[i])
                for i in range(len(feed_config.feeded_vars_names))
            ]
            feeder = fluid.DataFeeder(feed_list=feed_vars, place=place)
            batch_feed = feed_gen(batch_size, feed_config.feeded_vars_dims,
                                  feed_config.feeded_vars_filelist)
            slots = [batch_feed]
            results = exe.run(inference_program,
                              feed=feeder.feed(slots),
                              fetch_list=fetch_list,
                              return_numpy=return_numpy)
        for i, v in enumerate(fetch_list):
            logger.info("fetch_targets name: %s" % v.name)
            logger.info("fetch_targets: {}".format(results[i]))
        return results


def check_not_expected_ops(prog):
    op_types_set = set()
    for op in prog.global_block().ops:
        if op.type in not_expected_op_types and op.type not in op_types_set:
            logger.warning(
                "find op type '{}' in program, please check if your program is pruned correctly !".
                format(op.type))
            op_types_set.add(op.type)


def check_saved_vars_try_dump(dump_dir,
                              dump_prog_fn,
                              is_text_dump_program,
                              feed_config,
                              fetch_config,
                              batch_size=1,
                              save_filename=None):
    dump_prog = load_program(
        os.path.join(dump_dir, dump_prog_fn), is_text_dump_program)
    saved_params = [
        v for v in dump_prog.list_vars() if fluid.io.is_persistable(v)
    ]
    logger.info("persistable vars in dump program: {}".format(
        [v.name for v in saved_params]))

    check_not_expected_ops(dump_prog)

    return try_load_model_vars(dump_dir, dump_prog_fn, is_text_dump_program,
                               batch_size, feed_config, fetch_config,
                               save_filename, saved_params)


def parse_program(program, output_dir):
    # persistable vars
    output = {}
    persistable_vars = [
        v for v in program.list_vars() if fluid.io.is_persistable(v)
    ]
    output["persistable_vars"] = [{
        'name': str(v.name),
        'shape': str(v.shape),
        'lod_level': int(v.lod_level),
        'dtype': str(v.dtype),
        'type': str(v.type)
    } for v in persistable_vars]
    with open(os.path.join(output_dir, persistable_vars_out_fn), 'w') as f:
        f.write("persistable vars:\n")
        for var in output["persistable_vars"]:
            f.write(str(var))
            f.write("\n")

    # all vars
    all_vars = [v for v in program.list_vars()]
    output["all_vars"] = [{
        'name': str(v.name),
        'shape': str(v.shape),
        'lod_level': int(v.lod_level),
        'dtype': str(v.dtype)
    } if v.type not in feed_fetch_type_list else {
        'name': str(v.name),
        'type': str(v.type)
    } for v in all_vars]
    with open(os.path.join(output_dir, all_vars_out_fn), 'w') as f:
        f.write("all vars:\n")
        for var in output["all_vars"]:
            f.write(str(var))
            f.write("\n")

    # ops
    ops = program.global_block().ops
    output["ops"] = [{
        'type': op.type,
        'input_arg_names': str(op.input_arg_names),
        'output_arg_names': str(op.output_arg_names)
    } for op in ops]
    with open(os.path.join(output_dir, ops_out_fn), 'w') as f:
        f.write("ops:\n")
        for op in output["ops"]:
            f.write(str(op))
            f.write("\n")
