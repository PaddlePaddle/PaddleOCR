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

import os
import six
import pickle
import numpy as np

import paddle
from paddle import compat as cpt
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid import backward
from paddle.fluid import unique_name
from paddle.fluid.dygraph import layers
from paddle.fluid.layers import nn
from paddle.fluid.layers.utils import _hash_with_id
from paddle.fluid.dygraph.base import switch_to_static_graph
from paddle.fluid.framework import in_dygraph_mode

__all__ = ['TranslatedLayer']

INFER_MODEL_SUFFIX = ".pdmodel"
INFER_PARAMS_SUFFIX = ".pdiparams"
INFER_PARAMS_INFO_SUFFIX = ".pdiparams.info"

LOADED_VAR_SUFFIX = "load"
PARAMETER_NAME_PREFIX = "param"
BUFFER_NAME_PREFIX = "buffer"


def _load_program_desc(model_file_path):
    # 1. parse program desc
    with open(model_file_path, "rb") as f:
        program_desc_str = f.read()

    program_desc = core.ProgramDesc(program_desc_str)
    if not core._is_program_version_supported(program_desc._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program_desc._version())

    return program_desc


def _is_persistable(var_desc):
    if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var_desc.type() == core.VarDesc.VarType.READER or \
            var_desc.type() == core.VarDesc.VarType.RAW:
        return False
    return var_desc.persistable()


def _is_parameter(persistable_var_desc, program_desc):
    # 1. firstly, param should be input of op
    input_ops = []  # op can be repeated
    for block_idx in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(block_idx)
        for op_idx in six.moves.range(block.op_size()):
            op = block.op(op_idx)
            # NOTE: parameter is the input of a certain op
            if persistable_var_desc.name() in op.input_arg_names():
                input_ops.append(op)
    # 2. secondly, param should not be output of op or be same op's output
    for block_idx in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(block_idx)
        for op_idx in six.moves.range(block.op_size()):
            op = block.op(op_idx)
            if persistable_var_desc.name() in op.output_arg_names():
                # such as batch_norm_op
                if op in input_ops:
                    continue
                else:
                    return False
    return True


def _get_persistable_vars(program_desc):
    persistable_vars = []
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        persistable_vars.extend(list(filter(_is_persistable, block.all_vars())))
    return persistable_vars


def _get_persistable_var_names(program_desc):
    """
    Get all persistable variable names in ProgramDesc.
    """
    var_names = []
    persistable_vars = _get_persistable_vars(program_desc)
    for var in persistable_vars:
        var_names.append(var.name())
    return var_names


def _get_all_var_names(program_desc):
    all_var_names = set()
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        for var in block.all_vars():
            all_var_names.add(var.name())
    return all_var_names


@switch_to_static_graph
def _append_loaded_suffix(name):
    """
    Append loaded suffix to the given variable name
    e.g. x ==> x.load_0, x.load_0 ==> x.load_0.load_0
    """
    suffix = LOADED_VAR_SUFFIX
    name = cpt.to_text(name)
    new_name = unique_name.generate_with_ignorable_key('.'.join((name, suffix)))
    return new_name


@switch_to_static_graph
def _generate_unique_var_name(prefix):
    return unique_name.generate_with_ignorable_key(prefix)


def _append_loaded_suffix_to_var(program_desc):
    suffix_varname_dict = dict()
    persistable_vars = _get_persistable_vars(program_desc)
    for var_desc in persistable_vars:
        old_name = var_desc.name()
        new_name = _append_loaded_suffix(var_desc.name())
        suffix_varname_dict[new_name] = old_name
        var_desc.set_name(new_name)
        for block_idx in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(block_idx)
            block._rename_var(cpt.to_bytes(old_name), cpt.to_bytes(new_name))
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                op._rename_input(old_name, new_name)
                op._rename_output(old_name, new_name)
    return suffix_varname_dict


@switch_to_static_graph
def _generate_unique_var_name_sync_with_main_program(prefix):
    return unique_name.generate(prefix)


def _get_loaded_var_new_old(program_desc, all_new_old_dict_all):
    new_old_dict = dict()
    persistable_vars = _get_persistable_vars(program_desc)
    for var_desc in persistable_vars:
        name_new = var_desc.name()
        new_old_dict[name_new] = all_new_old_dict_all[name_new]
    return new_old_dict


def _rename_var_program_desc(program_desc, include=None, exclude=None):
    """
    Change the name of the loaded variables.Use 'unique_name.generate' to avoid duplication.
    It is used when loading multiple program during inference.

    e.g. linear_0.tmp_3 ==> linear_0.tmp_1, x ==> x_0. For double grad, x@GRAD ==> x_0@GRAD
    If 'include' is not `None`,variables in include and the corresponding
      double grad variables (if exist) are renamed.
    If 'exclude' is not `None`,variables that are in exclude and the
      corresponding double grad variables (if exist) are not renamed.

    Args:
        program_desc(ProgramDesc):the variables in it will be modified.
        include(List):list of names of variables.
        exclude(List):list of names of variables.

    Returns:
        tuple of (dict_rename_var_new_old, dict_rename_var_old_new)
        dict_rename_var_new_old is a dict mapping from new name to old name
        dict_rename_var_old_new is a dict mapping from old name to new name
    """
    dict_rename_var_old_new = dict()
    dict_rename_var_new_old = dict()
    old_names = []
    # Store all old names
    for b_idx in six.moves.range(program_desc.num_blocks()):
        cur_block = program_desc.block(b_idx)
        for var in cur_block.all_vars():
            old_names.append(var.name())

    # Create dict_rename_var_new_old and dict_rename_var_old_new for non double
    # grad variables
    has_double_grad = False
    for b_idx in six.moves.range(program_desc.num_blocks()):
        cur_block = program_desc.block(b_idx)
        for var_idx, var in enumerate(cur_block.all_vars()):
            name_old = var.name()
            is_double_grad_var = "@GRAD" in name_old
            has_double_grad = has_double_grad or is_double_grad_var
            should_rename = (include is None or name_old in include) and (
                exclude is None or
                name_old not in exclude) and not is_double_grad_var
            if should_rename:
                temp_name = name_old.split('_')
                if len(temp_name) > 1 and temp_name[-1].isnumeric():
                    temp_name = "_".join(temp_name[:-1])
                else:
                    temp_name = name_old
                while True:
                    name_new = _generate_unique_var_name_sync_with_main_program(
                        temp_name)
                    if name_new not in old_names[:var_idx] + old_names[var_idx +
                                                                       1:]:
                        break
            else:
                name_new = name_old
            if name_old != name_new:
                cur_block._rename_var(
                    cpt.to_bytes(name_old), cpt.to_bytes(name_new))
            if not is_double_grad_var:
                dict_rename_var_old_new[name_old] = name_new
                dict_rename_var_new_old[name_new] = name_old

    # Handle double grad names
    if has_double_grad:
        double_grad_rename_dict = {}
        for name_old in dict_rename_var_old_new:
            for b_idx in six.moves.range(program_desc.num_blocks()):
                cur_block = program_desc.block(b_idx)
                for var_idx, var in enumerate(cur_block.all_vars()):
                    var_name = var.name()
                    if "@GRAD" in var_name and name_old in var_name:
                        new_var_name = var_name.replace(
                            name_old, dict_rename_var_old_new[name_old])
                        double_grad_rename_dict[var_name] = new_var_name
        for var_name in double_grad_rename_dict:
            dict_rename_var_old_new[var_name] = double_grad_rename_dict[
                var_name]
            dict_rename_var_new_old[double_grad_rename_dict[
                var_name]] = var_name

    # Rename on program desc
    for b_idx in six.moves.range(program_desc.num_blocks()):
        cur_block = program_desc.block(b_idx)
        for op_idx in six.moves.range(cur_block.op_size()):
            op = cur_block.op(op_idx)
            for input_arg_name in op.input_arg_names():
                if input_arg_name in dict_rename_var_old_new:
                    if input_arg_name != dict_rename_var_old_new[
                            input_arg_name]:
                        op._rename_input(
                            input_arg_name,
                            dict_rename_var_old_new[input_arg_name])
                        if cur_block.has_var(cpt.to_bytes(input_arg_name)):
                            cur_block._rename_var(
                                cpt.to_bytes(input_arg_name),
                                cpt.to_bytes(dict_rename_var_old_new[
                                    input_arg_name]))
            for output_arg_name in op.output_arg_names():
                if output_arg_name in dict_rename_var_old_new:
                    if output_arg_name != dict_rename_var_old_new[
                            output_arg_name]:
                        op._rename_output(
                            output_arg_name,
                            dict_rename_var_old_new[output_arg_name])
                        if cur_block.has_var(cpt.to_bytes(output_arg_name)):
                            cur_block._rename_var(
                                cpt.to_bytes(output_arg_name),
                                cpt.to_bytes(dict_rename_var_old_new[
                                    output_arg_name]))
    program_desc.flush()
    return dict_rename_var_new_old, dict_rename_var_old_new


@switch_to_static_graph
def _build_program_by_desc(program_desc):
    prog = framework.Program()
    prog.desc = program_desc
    prog.blocks = [
        framework.Block(prog, i)
        for i in six.moves.range(prog.desc.num_blocks())
    ]
    prog._sync_with_cpp()
    return prog


def _change_is_test_status(program_desc, is_test):
    # change all `is_test` attributes
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        for j in six.moves.range(block.op_size()):
            op = block.op(j)
            if op.has_attr('is_test'):
                op._set_attr('is_test', is_test)


class _ProgramHolder(object):
    """
    Holds the execution information of a Program.

    _ProgramHolder is the execution unit of TranslatedLayer, 
    if TranslatedLayer contains multiple _ProgramHolder, 
    it can execute multiple methods

    _ProgramHolder is an internal concept.
    """

    def __init__(self, program_desc):
        super(_ProgramHolder, self).__init__()

        # input, output, persistable, double_grads var info
        self._input_descs = []
        self._output_descs = []
        self._double_grad_descs = []
        self._persistable_names = []

        # execution scope
        self._inner_scope = core.Scope()

        # append suffix var name dict
        self._suffix_varname_dict = None
        # forward program
        self._infer_program_desc = self._preprocess(program_desc)
        # forward + backward program
        self._train_program_desc = self._append_backward_desc(
            self._infer_program_desc)

    @property
    def infer_program(self):
        return self._infer_program_desc

    @property
    def train_program(self):
        return self._train_program_desc

    @property
    def input_descs(self):
        return self._input_descs

    @property
    def output_descs(self):
        return self._output_descs

    @property
    def persistable_names(self):
        return self._persistable_names

    @property
    def double_grad_descs(self):
        return self._double_grad_descs

    @property
    def scope(self):
        return self._inner_scope

    def _preprocess(self, program_desc):
        # rename persistable variables of 'program_desc'
        list_persistable_var = _get_persistable_var_names(program_desc)
        rename_new_old_dict, _ = _rename_var_program_desc(program_desc,
                                                          list_persistable_var)
        # 1. Prune original program
        # remove feed, fetch and scale-1 op, remove op_callstack attr
        ops_to_remove = []
        root_block = program_desc.block(0)
        for i in six.moves.range(root_block.op_size()):
            op = root_block.op(i)
            if op.type() == 'feed':
                ops_to_remove.append(i)
                feed_var_name = cpt.to_bytes(op.input('X')[0])
                root_block._remove_var(feed_var_name)
                self._input_descs.append(
                    root_block.find_var(cpt.to_bytes(op.output('Out')[0])))
            elif op.type() == 'scale' and op.output('Out')[0].startswith(
                    'save_infer_model/scale_'):
                ops_to_remove.append(i)
                out_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(out_var_name)
                self._output_descs.append(
                    root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            elif op.type() == 'fetch':
                ops_to_remove.append(i)
                fetch_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(fetch_var_name)
                # NOTE: some old pre-train models have no extra scale_op
                if not op.input('X')[0].startswith('save_infer_model/scale_'):
                    self._output_descs.append(
                        root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            else:
                if op.has_attr("op_callstack"):
                    op.remove_attr("op_callstack")

        for op_idx in reversed(ops_to_remove):
            root_block._remove_op(op_idx, op_idx + 1)

        for i in range(program_desc.num_blocks()):
            block_desc = program_desc.block(i)
            for var_desc in block_desc.all_vars():
                if "@GRAD" in var_desc.name():
                    self._double_grad_descs.append(var_desc)

        # 2. Input processing, reverse feed vars
        self._input_descs.reverse()

        # 3. Output processing, add scale for outputs
        tmp_program = _build_program_by_desc(program_desc)
        # NOTE: [why need append scale for outputs]
        # When dealing with some more complex pre-training models, there 
        # will be situations where the pre-training model has multiple 
        # fetch outputs. In the scenario of multiple fetch outputs, 
        # there is a special case where multiple outputs of the model 
        # may be on the same branch. According to the user's subsequent 
        # use, multiple outputs may be associated with multiple branches.
        # These subsequent operations are added in TranslatedLayer is 
        # agnostic during initialization, which results in subsequent 
        # gradient accumulation operations that are required on the 
        # output node in the middle of the branch will not be performed, 
        # resulting in error, details see pull request:
        # [https://github.com/PaddlePaddle/Paddle/pull/24627]
        self._append_scale_to_output(tmp_program)

        # 4. Persistable vars processing
        # - append loaded suffix to persistable vars
        # NOTE: [why need to append suffix to persistable vars]
        # Dygraph and static graph mode use the same naming mechanism. 
        # If users want to load the model fine-tune, it is possible 
        # to add the existing Layer in the loaded model to enhance 
        # the network. For example, the original saved model has linear, 
        # and later after loading, a new linear is added. At this time, 
        # there will be a problem of duplicate names, so here is unified 
        # to add the LOADED suffix to the parameters of the model loaded
        self._suffix_varname_dict = _get_loaded_var_new_old(program_desc,
                                                            rename_new_old_dict)

        # - get persistable var
        self._persistable_names = _get_persistable_var_names(program_desc)

        return program_desc

    @switch_to_static_graph
    def _append_scale_to_output(self, program):
        # 1. append scale & save var
        scale_output_vars = []
        with framework.program_guard(program):
            for i, out in enumerate(self._output_descs):
                var = program.global_block().var(out.name())
                var = nn.scale(
                    var, 1., name="translated_layer/scale_{}".format(i))
                scale_output_vars.append(var)
        # 2. update output names & descs
        for i, var in enumerate(scale_output_vars):
            self._output_descs[i] = var.desc

    @switch_to_static_graph
    def _append_backward_desc(self, infer_program_desc):
        program_desc_copy = core.ProgramDesc(infer_program_desc)

        # 1. set all `is_test` attributes to False
        _change_is_test_status(program_desc_copy, False)

        # 2. prepare program and related var
        # NOTE: To reuse backward interfaces, build Program firstly.
        # Originally, there is no need to build a program, but need to almost
        # rewrite a series of methods for append_backward for program_desc. 
        # Therefore, in order to reuse the method of backward.py, build the program here.
        program = _build_program_by_desc(program_desc_copy)
        # 3. Add the outputs which is only used for training and not saved in
        # inference program.
        for block_idx in six.moves.range(program.num_blocks):
            block = program.block(block_idx)
            for op in block.ops:
                if op.type == "batch_norm":
                    if "ReserveSpace" not in op.output_names or len(
                            op.output("ReserveSpace")) == 0:
                        reserve_space = block.create_var(
                            name=unique_name.generate_with_ignorable_key(
                                ".".join(["reserve_space", 'tmp'])),
                            dtype=block.var(op.input("X")[0]).dtype,
                            type=core.VarDesc.VarType.LOD_TENSOR,
                            persistable=False,
                            stop_gradient=True)
                        op.desc.set_output("ReserveSpace", [reserve_space.name])

        targets = []
        for out in self._output_descs:
            targets.append(program.global_block().var(out.name()))

        # 3. append backward
        backward.gradients(targets=targets, inputs=[])
        return program.desc


# [ TranslatedLayer : Run program in imperative mode ]
# 
# DESIGN IDEA: using an special operator `RunProgram`, execute program inside operator.
#
# Op's Inputs:
#   - the input variable of the user feed
#   - the necessary parameters of the network
# Op's Outputs:
#   - the output variable of fetch
# 
# This op receives a complete program desc, internally creates scope
# and executor, executes this program. Key points:
#
# 1. Data Sharing: 
#   The varBase of the dynamic graph is not in the scope, so before the op
#   executes the program internally, create persistent variables with the
#   same name as feed, parameters, and fetch in the scope, and share the
#   LoDTensor of the op input.
# 
# 2. Forward and Backward Separation:
#   Because the dynamic graph op performs the forward and backward separately,
#   in the forward op RunProgram, we only execute the forward part of whole program,
#   and in the backward op RunProgramGrad, we execute the backward part of program.
#   We can not separate the program into forward and backward part, which will 
#   make some control flow execution logic wrong.


# NOTE: [compatible] deal with model saved by save_inference_model,
# which need get var info from program desc
def _load_persistable_vars_by_program(model_path,
                                      program_holder,
                                      params_filename=None):
    # make sure the path has been checked
    persistable_vars = _get_persistable_vars(program_holder.infer_program)
    load_var_dict = {}
    for each_var in persistable_vars:
        orig_each_name = program_holder._suffix_varname_dict[each_var.name()]
        if _is_parameter(each_var, program_holder.infer_program):
            # create output varbase
            new_var = framework.ParamBase(
                shape=each_var.shape(),
                dtype=each_var.dtype(),
                name=each_var.name(),
                type=each_var.type(),
                persistable=True)
        else:
            new_var = framework._varbase_creator(
                type=each_var.type(),
                name=each_var.name(),
                shape=each_var.shape(),
                dtype=each_var.dtype(),
                persistable=True)
        if params_filename is None:
            framework._dygraph_tracer().trace_op(
                type='load',
                inputs={},
                outputs={'Out': new_var},
                attrs={'file_path': os.path.join(model_path, orig_each_name)})
        new_var.stop_gradient = False
        load_var_dict[each_var.name()] = new_var

    if params_filename is not None:
        load_var_list = []
        dict_name_old_new = {
            v: k
            for k, v in program_holder._suffix_varname_dict.items()
        }
        for name in sorted(dict_name_old_new.keys()):
            load_var_list.append(load_var_dict[dict_name_old_new[name]])

        framework._dygraph_tracer().trace_op(
            type='load_combine',
            inputs={},
            outputs={'Out': load_var_list},
            attrs={'file_path': os.path.join(model_path, params_filename)})

        for each_var in persistable_vars:
            if not _is_parameter(each_var, program_holder.infer_program):
                continue
            param = load_var_dict[each_var.name()]
            param.stop_gradient = False

    # NOTE: [Recovery stop gradient information based on the program]
    # After loading the model, the stop_gradient information 
    # of the original variable is lost, but if a parameter does not
    # have a corresponding @GRAD variable in the backward program,
    # it can be said that it is also stop_gradient
    all_var_names = _get_all_var_names(program_holder.train_program)
    for var_name in load_var_dict:
        grad_var_name = var_name + core.grad_var_suffix()
        if grad_var_name not in all_var_names:
            load_var_dict[var_name].stop_gradient = True

    return load_var_dict


def _load_persistable_vars(model_path, var_info_path, program_holder,
                           params_filename):
    # 1. load extra var info
    with open(var_info_path, 'rb') as f:
        extra_var_info = pickle.load(f)

    # 2. construct var dict
    load_var_dict = dict()
    load_var_list = []
    inv_suffix_varname_dict = {
        value: key
        for key, value in program_holder._suffix_varname_dict.items()
    }

    # NOTE(chenweihang): we need load persistable vars based the program,
    # because the program may be pruned when `save_inference_model`, some
    # var in `extra_var_info` may have been pruned 
    for name in sorted(inv_suffix_varname_dict):
        if name not in extra_var_info:
            raise RuntimeError(
                "The model to be loaded is not complete."
                "The variable `%s` of program cannot be found in loaded model.",
                name)
        # get suffix var name, see [why need to append suffix to persistable vars]
        new_name = inv_suffix_varname_dict[name]
        # create output varbase
        if extra_var_info[name].get('trainable', None) is not None:
            # use default shape and dtype
            new_var = framework.ParamBase(
                shape=[1],  # only to pass check, this shape is not meaningful
                dtype=core.VarDesc.VarType.FP32,
                name=new_name,
                persistable=True)
        else:
            new_var = framework._varbase_creator(
                name=new_name, persistable=True)

        new_var.stop_gradient = extra_var_info[name]['stop_gradient']
        load_var_dict[new_name] = new_var
        load_var_list.append(new_var)

    # 3. load all vars
    assert params_filename is not None, "params_filename should not be None."
    var_file_path = os.path.join(model_path, params_filename)
    if not os.path.exists(var_file_path):
        if len(extra_var_info) != 0:
            raise ValueError("The model to be loaded is incomplete.")
    else:
        framework._dygraph_tracer().trace_op(
            type='load_combine',
            inputs={},
            outputs={'Out': load_var_list},
            attrs={'file_path': var_file_path})

    return load_var_dict


# NOTE(chenweihang): to adapt paddle.load to get state_dict
def _remove_varname_suffix(var_dict, program_holder):
    no_suffix_var_dict = dict()
    for var_name in var_dict:
        no_suffix_name = program_holder._suffix_varname_dict[var_name]
        no_suffix_var_dict[no_suffix_name] = var_dict[var_name]
    return no_suffix_var_dict


def _construct_program_holders(model_path, model_filename=None):
    # make sure the path has been checked
    program_holder_dict = dict()

    if model_filename is not None:
        # [compatible] if assign model_filename, only can load one program as Layer.forward
        model_filename = os.path.basename(model_filename)
        model_file_path = os.path.join(model_path, model_filename)
        model_name = model_filename[:-len(INFER_MODEL_SUFFIX)]
        #Load every file that meets the requirements in the directory model_path.
        for filename in os.listdir(model_path):
            if model_filename == filename:
                func_name = 'forward'
                model_file_path = os.path.join(model_path, model_filename)
            elif filename.endswith(INFER_MODEL_SUFFIX) and filename.startswith(
                    model_name):
                parsing_names = filename[len(model_name):-len(
                    INFER_MODEL_SUFFIX) + 1].split('.')
                if len(parsing_names) == 3 and len(parsing_names[1]) > 0:
                    func_name = parsing_names[1]
                    model_file_path = os.path.join(model_path, filename)
                else:
                    continue
            else:
                continue
            program_holder_dict[func_name] = _ProgramHolder(
                _load_program_desc(model_file_path))
    else:
        for _, _, file_names in os.walk(model_path):
            for name in file_names:
                if 'model' in name:
                    model_file_path = os.path.join(model_path, name)
                    method_name = name.strip('_')
                    if method_name == 'model':
                        method_name = 'forward'
                    else:
                        method_name.replace('model', '')
                    program_holder_dict[method_name] = _ProgramHolder(
                        _load_program_desc(model_file_path))

    return program_holder_dict


def _construct_params_and_buffers(model_path,
                                  programs,
                                  params_filename=None,
                                  append_suffix=True):
    var_info_filename = str(params_filename) + ".info"
    var_info_path = os.path.join(model_path, var_info_filename)
    params_path = os.path.join(model_path, str(params_filename))

    if os.path.exists(var_info_path):
        var_dict = _load_persistable_vars(model_path, var_info_path,
                                          programs['forward'], params_filename)
        model_name = params_filename[:-len(INFER_PARAMS_SUFFIX)]
        #Load every file that meets the requirements in the directory model_path.
        for file_name in os.listdir(model_path):
            if file_name.startswith(model_name) and file_name.endswith(
                    INFER_PARAMS_SUFFIX):
                parsing_names = file_name[len(model_name):-len(
                    INFER_PARAMS_SUFFIX) + 1].split('.')
                if len(parsing_names) == 3 and len(parsing_names[1]) > 0:
                    func_name = parsing_names[1]
                else:
                    continue
            else:
                continue
            var_info_path = os.path.join(model_path, var_info_filename)
            var_dict.update(
                _load_persistable_vars(model_path, var_info_path, programs[
                    func_name], file_name))
    elif params_filename is not None and not os.path.exists(params_path):
        # When saving XX, there is only '*.pdmodel'
        return dict()
    else:
        var_dict = _load_persistable_vars_by_program(
            model_path, programs['forward'], params_filename)

    if not append_suffix:
        var_dict = _remove_varname_suffix(var_dict, programs['forward'])

    return var_dict


def _run_dygraph(instance, input, program_holder):

    # 1. prepare inputs, outputs, attrs
    input_vars = []
    for i, value in enumerate(input):
        if not isinstance(value, (np.ndarray, core.VarBase)):
            raise TypeError(
                "The type of input in TranslatedLayer must be numpy array or Variable(VarBase), but received %s."
                % type(value))
        # NOTE: In order to unify the API, firstly convert the input to VarBase
        if isinstance(value, np.ndarray):
            var = core.VarBase(
                value=value,
                name=program_holder.input_descs[i].name(),
                persistable=False,
                place=framework._current_expected_place(),
                zero_copy=True)
        else:
            var = value
            # NOTE: we changed var name here, 
            # but it may be an important name set by user
            var.name = program_holder.input_descs[i].name()
        input_vars.append(var)
    if instance._input_args_names is None:
        instance._input_args_names = [
            ins.name() for ins in program_holder.input_descs
        ]

    persistable_vars = []
    for var_name in program_holder.persistable_names:
        dy_var_name = instance._persistable_var_name_dict[var_name]
        if dy_var_name in instance._parameters:
            persistable_vars.append(instance._parameters[dy_var_name])
        elif dy_var_name in instance._buffers:
            persistable_vars.append(instance._buffers[dy_var_name])
        else:
            raise ValueError(
                "The persistable variable %s does not exist in current TranslatedLayer."
                % var_name)

    output_vars = []
    for var_desc in program_holder.output_descs:
        var = core.VarBase(var_desc.dtype(),
                           var_desc.shape(),
                           var_desc.name(), var_desc.type(), False)
        output_vars.append(var)

    # hold forward variables
    tmp_scope_vec = core.VarBase(core.VarDesc.VarType.FP32, [],
                                 "program_out_scope",
                                 core.VarDesc.VarType.STEP_SCOPES, True)
    tmp_scope_vec.value().set_scope(program_holder.scope)

    double_grad_vars = []
    for var_desc in program_holder.double_grad_descs:
        var = core.VarBase(var_desc.dtype(),
                           var_desc.shape(),
                           var_desc.name(), var_desc.type(), False)
        double_grad_vars.append(var)
    if len(double_grad_vars) == 0:
        double_grad_vars = [
            core.VarBase(
                value=[1],
                name='Fake_var',
                place=framework._current_expected_place())
        ]

    # 2. run program by op
    trace_program = program_holder.infer_program if instance._is_test else program_holder.train_program
    end_op_index = program_holder.infer_program.block(0).op_size()
    framework._dygraph_tracer().trace_op(
        type='run_program',
        inputs={'X': input_vars,
                'Params': persistable_vars},
        outputs={
            'Out': output_vars,
            'OutScope': tmp_scope_vec,
            'DOut': double_grad_vars
        },
        attrs={
            'global_block': trace_program.block(0),
            'start_op_index': 0,
            'end_op_index': end_op_index,
            'is_test': instance._is_test,
            'program_id': _hash_with_id(trace_program)
        })
    # NOTE: [ why need set param's gradient type here ]
    # if user set sparse gradient mode, the param's gradient
    # will be SelectedRows, not LoDTensor. But tracer will just
    # set param grad VarBase by forward VarBase(LoDTensor)
    # If we don't change grad_var type here, RunProgramOp need
    # transform SelectedRows to LoDTensor forcibly, it may not
    # be user wanted result.
    for persistable_var in persistable_vars:
        grad_var_name = var.name + core.grad_var_suffix()
        grad_var = trace_program.block(0).find_var(cpt.to_bytes(grad_var_name))
        # NOTE: cannot find var desc maybe not problem, 
        # such as in batch_norm
        if grad_var is None:
            continue
        persistable_var._set_grad_type(grad_var.type())

    drop_scope_if_no_grad(instance, tmp_scope_vec)

    # 3. prepare output, keep same form with inputs
    outs = output_vars
    if len(output_vars) == 1:
        outs = output_vars[0]
    return outs


def drop_scope_if_no_grad(instance, scope_vec):
    tracer = framework._dygraph_tracer()
    if (not instance._is_test) and (not tracer._has_grad):
        scope_vec.value().get_scope().drop_kids()


def _run_static_graph(input, program_holder, trace_program):
    main_program = framework.default_main_program()
    param_var_names = _get_persistable_var_names(trace_program)
    _, dict_rename_var_old_new = _rename_var_program_desc(
        trace_program, exclude=param_var_names)
    trace_program.flush()
    output_names = [var.name() for var in program_holder.output_descs]
    # append blocks from 'trace_program'
    _append_block(main_program, trace_program, program_holder, input,
                  dict_rename_var_old_new)
    main_program._sync_with_cpp()
    outs = _get_output_from_program(main_program, program_holder,
                                    dict_rename_var_old_new)
    if len(outs) == 1:
        outs = outs[0]
    return outs


def _collect_current_and_parent_var(program, block_idx):
    '''
    Get variables in current block and its parent block.
    
    Args:
        program(Program): The program containing the current block.
        block_idx(int): index of current block.

    Returns:
        List: list of variables.
    '''
    vars = []
    if block_idx < 0:
        return vars
    for var in program.block(block_idx).vars:
        vars.append(var)
    parent_idx = program.block(block_idx).parent_idx
    if parent_idx > -1:
        vars += _collect_current_and_parent_var(program, parent_idx)
    return vars


def _append_block(dest_program,
                  src_program_desc,
                  program_holder,
                  input_variables,
                  dict_rename_var_old_new=None):
    '''
    Append Variables and Operators in 'src_program_desc' to dest_program.
    
    Args:
        dest_program(Program): Variables and Operators are appended to it.
        src_program_desc(ProgramDesc): Variables in it will be appended to 'dest_program'.
        program_holder(_ProgramHolder): program_holder of TranslatedLayer
        input_variables(list): list of input variables
        dict_rename_var_old_new(None|dict): When using '_rename_var_program_desc', 
        use it to map the name of the variable before it was modified and the new name.
    '''

    origin_block_idx = dest_program.current_block_idx
    param_var_names = _collect_current_and_parent_var(dest_program,
                                                      origin_block_idx)
    append_var_from_block_desc_static(
        dest_program.block(origin_block_idx),
        src_program_desc.block(0),
        exclude=param_var_names)

    name_inp_desc = [inp.name() for inp in program_holder.input_descs]
    input_names = [inp.name for inp in input_variables]
    if len(name_inp_desc) != len(input_names):
        raise ValueError(
            "The number of input is invalid, expected {}, but received {}.".
            format(len(name_inp_desc), len(input_names)))
    for i, out_name in enumerate(name_inp_desc):
        if dict_rename_var_old_new:
            out_name = dict_rename_var_old_new[out_name]
        dest_program.block(origin_block_idx).append_op(
            type='assign',
            inputs={'X': [input_names[i]]},
            outputs={'Out': [out_name]})

    append_ops = append_op_from_block_desc_static(
        dest_program.block(origin_block_idx), src_program_desc.block(0))
    dest_program._sync_with_cpp()

    offset_block_idx = dest_program.num_blocks - 1

    if src_program_desc.num_blocks() > 1:
        for src_block_idx in range(1, src_program_desc.num_blocks()):
            src_block = src_program_desc.block(src_block_idx)
            src_parent_idx = src_block.parent
            if src_parent_idx > 0:
                parent_idx = offset_block_idx + parent_idx
            else:
                parent_idx = origin_block_idx
            dest_block = dest_program._create_block(parent_idx=parent_idx)
            append_var_from_block_desc_static(
                dest_block, src_block, exclude=param_var_names)
            append_ops += append_op_from_block_desc_static(dest_block,
                                                           src_block)

    dest_program._sync_with_cpp()
    for op in append_ops:
        if op.has_attr('sub_block'):
            sub = op.attr('sub_block')
            if isinstance(sub, framework.core.BlockDesc):
                origin_id = sub.id
            if isinstance(sub, framework.Block):
                origin_id = sub.idx
            op._set_attr('sub_block',
                         dest_program.block(offset_block_idx + origin_id))
    dest_program._sync_with_cpp()
    dest_program.current_block_idx = origin_block_idx


def _get_output_from_program(program,
                             program_holder,
                             dict_rename_var_old_new=None):
    """
    Get output name of 'program' according to program_holder
    """
    outs = list()
    for var in program_holder.output_descs:
        for idx in range(program.num_blocks):
            vars = program.block(idx).vars
            var_name = var.name()
            if dict_rename_var_old_new:
                var_name = dict_rename_var_old_new[var_name]
            if var_name in vars:
                out = vars[var_name]
                if out not in outs:
                    outs.append(out)
    return outs


def append_op_from_block_desc_static(block, src_block_desc):
    """
    Append Operators of 'src_block_desc' to current block.

    Args:
        block(Block): append OP of  'src_block_desc' to it.
        src_block_desc(BlockDesc): append var of  'src_block_desc'

    Returns:
        List: list of the OP that are append to current block.
    """
    ops = []
    for i in range(src_block_desc.op_size()):
        ops.append(append_op_from_desc_static(block, src_block_desc.op(i)))
    return ops


def append_op_from_desc_static(block, op_desc):
    """
    Append Operators to 'block' according to 'op_desc'.

    Args:
        block(Block): append OP of  'src_block_desc' to it.
        op_desc(OpDesc): create OP according to it.

    Returns:
        Operator: OP appended to 'block'.
    """
    op_type = op_desc.type()
    op_append = block.desc.append_op()
    op_append.copy_from(op_desc)
    op = framework.Operator(
        block=block,
        desc=op_append,
        type=op_type,
        inputs=None,
        outputs=None,
        attrs=None)
    block.ops.append(op)
    return op


def append_var_from_block_desc_static(block,
                                      src_block_desc,
                                      include=None,
                                      exclude=None):
    """
    Append Variables of 'src_block_desc' to current block.
    If 'include' is not `None`,variables that are not in include are not append.
    If 'exclude' is not `None`,variables that are in exclude will are not append.

    Args:
        block(Block): append Variables of  'src_block_desc' to it.
        src_block_desc(BlockDesc): append var of  'src_block_desc'
        include(List):list of names of variables
        exclude(List):list of names of variables

    Returns:
        List: list of the variables that are append to current block.
    """
    vars_append = []
    for var_desc in src_block_desc.all_vars():
        var_desc_name = var_desc.name()
        should_append = (include is None or var_desc_name in include) and (
            exclude is None or var_desc_name not in exclude)
        if not block.has_var(var_desc_name) and should_append:
            var_type = var_desc.type()
            if var_type in [
                    core.VarDesc.VarType.SELECTED_ROWS,
                    core.VarDesc.VarType.LOD_TENSOR,
                    core.VarDesc.VarType.LOD_TENSOR_ARRAY
            ]:
                data_type = var_desc.dtype()
                var_shape = var_desc.shape()
            else:
                data_type = None
                var_shape = None
            if var_type in [
                    core.VarDesc.VarType.LOD_TENSOR,
                    core.VarDesc.VarType.LOD_TENSOR_ARRAY
            ]:
                lod_level = var_desc.lod_level()
            else:
                lod_level = None

            if var_desc.persistable():
                current_block = block.program.global_block()
            else:
                current_block = block

            vars_append.append(
                current_block.create_var(
                    name=var_desc.name(),
                    dtype=data_type,
                    type=var_type,
                    shape=var_shape,
                    lod_level=lod_level,
                    persistable=var_desc.persistable(),
                    set_need_check_feed=var_desc.need_check_feed()))
    return vars_append


class TranslatedLayer(layers.Layer):
    """
    TranslatedLayer is a ``paddle.nn.Layer`` for holding the model 
    loaded by :ref:`api_paddle_jit_load` . It can be used like a 
    general Layer object in eval or train mode.
    
    .. note:
        The TranslatedLayer objects should not be created by constructor, it only can be loaded and constructed by :ref:`api_paddle_jit_load` .

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt

            BATCH_SIZE = 16
            BATCH_NUM = 4
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            # define a random dataset
            class RandomDataset(paddle.io.Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

                @paddle.jit.to_static
                def forward(self, x):
                    return self._linear(x)

            def train(layer, loader, loss_fn, opt):
                for epoch_id in range(EPOCH_NUM):
                    for batch_id, (image, label) in enumerate(loader()):
                        out = layer(image)
                        loss = loss_fn(out, label)
                        loss.backward()
                        opt.step()
                        opt.clear_grad()
                        print("Epoch {} batch {}: loss = {}".format(
                            epoch_id, batch_id, np.mean(loss.numpy())))

            # 1. train & save model.

            # create network
            layer = LinearNet()
            loss_fn = nn.CrossEntropyLoss()
            adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)

            # train
            train(layer, loader, loss_fn, adam)

            # save
            model_path = "linear.example.model"
            paddle.jit.save(layer, model_path)

            # 2. load model as TranslatedLayer

            # load
            translated_layer = paddle.jit.load(model_path)

            # inference
            translated_layer.eval()
            x = paddle.randn([1, IMAGE_SIZE], 'float32')
            pred = translated_layer(x)

            # fine-tune
            translated_layer.train()
            adam = opt.Adam(learning_rate=0.001, parameters=translated_layer.parameters())
            train(translated_layer, loader, loss_fn, adam)

    """

    def __init__(self, programs, persistable_vars):
        super(TranslatedLayer, self).__init__()

        if not isinstance(programs, dict):
            raise TypeError(
                "TranslatedLayer need to use _ProgramHolder's dict for initialization."
            )
        if not isinstance(persistable_vars, dict):
            raise TypeError(
                "TranslatedLayer need to use persistable variable dict for initialization."
            )

        self._program_holder_dict = programs

        # NOTE(chenweihang): [ why not use var name directly? ]
        # When add parameter or buffer to Layer by follow apis,
        # the variable name can't contain `.`, beccause which may cause
        # AttributeError when access the newly added parameter or buffer
        # in the form of `self.**.**``, but the ParamBase or BarBase
        # name contains `.` originally, such as `linear_0.w_0`, so here
        # need to generate new var name for each var
        self._persistable_var_name_dict = dict()
        # the TranslatedLayer object holded var names count started from 0
        with unique_name.guard():
            for name, var in persistable_vars.items():
                if isinstance(var, framework.ParamBase):
                    dy_name = _generate_unique_var_name(PARAMETER_NAME_PREFIX)
                    self._persistable_var_name_dict[name] = dy_name
                    self.add_parameter(dy_name, var)
                elif isinstance(var, core.VarBase):
                    dy_name = _generate_unique_var_name(BUFFER_NAME_PREFIX)
                    self._persistable_var_name_dict[name] = dy_name
                    self.register_buffer(dy_name, var)
                else:
                    raise TypeError(
                        "Adding persistent variable which  to layer is not supported now"
                    )

        self._is_test = True
        self._input_args_names = None

    @staticmethod
    @framework.dygraph_only
    def _construct(model_path, configs=None):
        # 0. dir and filename check
        model_path = os.path.normpath(model_path)
        if not os.path.isdir(model_path):
            raise ValueError("There is no directory named '%s'" % model_path)
        model_filename = None
        params_filename = None
        if configs is not None:
            model_filename = configs.model_filename
            params_filename = configs.params_filename

        # 1. load program desc & construct _ProgramHolder
        programs = _construct_program_holders(model_path, model_filename)

        # 2. load layer parameters & buffers
        persistable_vars = _construct_params_and_buffers(model_path, programs,
                                                         params_filename)

        # 3. construct TranslatedLayer object
        translated_layer = TranslatedLayer(programs, persistable_vars)

        # 4. create TranslatedLayer's execution method
        for method_name, program_holder in programs.items():
            if translated_layer._input_args_names is None:
                translated_layer._input_args_names = [
                    ins.name() for ins in program_holder.input_descs
                ]
            setattr(TranslatedLayer, method_name,
                    TranslatedLayer._execution_method_creator(method_name,
                                                              program_holder))

        # 5. set TranslatedLayer's default mode to eval
        translated_layer.eval()

        return translated_layer

    @staticmethod
    def _execution_method_creator(method_name, program_holder):
        def __i_m_p_l__(self, *input):
            program_holder = self._program_holder_dict[__i_m_p_l__.__name__]
            # When using jit.save, it runs in static graph mode.
            # Run in dynamic graph mode when the model is inferring.
            if in_dygraph_mode():
                return _run_dygraph(self, input, program_holder)
            else:
                # NOTE(weixin): [ why not use 'program_holder.infer_program' directly? ]
                # When use '_run_static_graph(input, program_holder, program_holder.infer_program)',
                # because '_run_static_graph' modifies 'ProgramDesc', 'OpDesc.op_size()' will return a very large wrong number.
                # A Segmentation fault error may occur if used 'p=ProgramDesc(program_holder.infer_program)'.
                p = framework.Program._construct_from_desc(
                    core.ProgramDesc(program_holder.infer_program))
                return _run_static_graph(input, program_holder, p.desc)

        __i_m_p_l__.__name__ = method_name
        return __i_m_p_l__

    def train(self):
        self._is_test = False

    def eval(self):
        self._is_test = True

    def program(self, method_name='forward'):
        """
        Gets translated program of specified method.

        Args:
            - method_name (string): mehtod name corresponding to the program
                to be obtained. Default: 'forward'.
        
        Returns:
            Program

        Examples:
            .. code-block:: python
            
                import numpy as np
                import paddle
                import paddle.nn as nn
                import paddle.optimizer as opt

                BATCH_SIZE = 16
                BATCH_NUM = 4
                EPOCH_NUM = 4

                IMAGE_SIZE = 784
                CLASS_NUM = 10

                # define a random dataset
                class RandomDataset(paddle.io.Dataset):
                    def __init__(self, num_samples):
                        self.num_samples = num_samples

                    def __getitem__(self, idx):
                        image = np.random.random([IMAGE_SIZE]).astype('float32')
                        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                        return image, label

                    def __len__(self):
                        return self.num_samples

                class LinearNet(nn.Layer):
                    def __init__(self):
                        super(LinearNet, self).__init__()
                        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

                    @paddle.jit.to_static
                    def forward(self, x):
                        return self._linear(x)

                def train(layer, loader, loss_fn, opt):
                    for epoch_id in range(EPOCH_NUM):
                        for batch_id, (image, label) in enumerate(loader()):
                            out = layer(image)
                            loss = loss_fn(out, label)
                            loss.backward()
                            opt.step()
                            opt.clear_grad()
                            print("Epoch {} batch {}: loss = {}".format(
                                epoch_id, batch_id, np.mean(loss.numpy())))

                # create network
                layer = LinearNet()
                loss_fn = nn.CrossEntropyLoss()
                adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

                # create data loader
                dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
                loader = paddle.io.DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2)

                # train
                train(layer, loader, loss_fn, adam)

                # save
                model_path = "linear.example.model"
                paddle.jit.save(layer, model_path)

                # load
                translated_layer = paddle.jit.load(model_path)

                # get program
                program = translated_layer.program()
        """
        # 1. get program holder
        program_holder = self._get_program_holder(method_name)

        # 2. get inference program desc
        program_desc = program_holder.infer_program

        # 3. construct program
        program = _build_program_by_desc(program_desc)
        return program

    def _get_program_holder(self, method_name='forward'):
        program_holder = self._program_holder_dict.get(method_name, None)
        if program_holder is None:
            raise ValueError(
                "The method `%s` does not exist in loaded TranslatedLayer." %
                method_name)
        return program_holder

    def _input_spec(self, method_name='forward'):
        # 1. get program holder
        program_holder = self._get_program_holder(method_name)

        # 2. build input spec by input desc
        input_spec = []
        for var_desc in program_holder.input_descs:
            spec = paddle.static.InputSpec(
                shape=var_desc.shape(),
                dtype=var_desc.dtype(),
                name=var_desc.name())
            input_spec.append(spec)

        return input_spec

    def _output_spec(self, method_name='forward'):
        # 1. get program holder
        program_holder = self._get_program_holder(method_name)

        # 2. build output spec by output desc
        output_spec = []
        for var_desc in program_holder.output_descs:
            # NOTE(chenweihang): InputSpec describes a tensor, not just input. 
            # Maybe the name is not good enough. Here we use InputSpec to 
            # construct the description of Output tensor
            spec = paddle.static.InputSpec(
                shape=var_desc.shape(),
                dtype=var_desc.dtype(),
                name=var_desc.name())
            output_spec.append(spec)

        return output_spec
