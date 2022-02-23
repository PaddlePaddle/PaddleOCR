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

from __future__ import print_function

import sys
import six
import random
import os
import re
from .graphviz import GraphPreviewGenerator
from .proto import framework_pb2
from google.protobuf import text_format
from . import unique_name
from .framework import Program, default_main_program, Variable
from . import core
from . import io
from .layer_helper import LayerHelper

_vartype2str_ = [
    "UNK",
    "LoDTensor",
    "SelectedRows",
    "FeedMinibatch",
    "FetchList",
    "StepScopes",
    "LodRankTable",
    "LoDTensorArray",
    "PlaceList",
]
_dtype2str_ = [
    "bool",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
]


def repr_data_type(type):
    return _dtype2str_[type]


def repr_tensor(proto):
    return "tensor(type={}, shape={})".format(_dtype2str_[int(proto.data_type)],
                                              str(proto.dims))


reprtpl = "{ttype} {name} ({reprs})"


def repr_lodtensor(proto):
    if proto.type.type != framework_pb2.VarType.LOD_TENSOR:
        return

    level = proto.type.lod_tensor.lod_level
    reprs = repr_tensor(proto.type.lod_tensor.tensor)
    return reprtpl.format(
        ttype="LoDTensor" if level > 0 else "Tensor",
        name=proto.name,
        reprs="level=%d, %s" % (level, reprs) if level > 0 else reprs)


def repr_selected_rows(proto):
    if proto.type.type != framework_pb2.VarType.SELECTED_ROWS:
        return

    return reprtpl.format(
        ttype="SelectedRows",
        name=proto.name,
        reprs=repr_tensor(proto.type.selected_rows))


def repr_tensor_array(proto):
    if proto.type.type != framework_pb2.VarType.LOD_TENSOR_ARRAY:
        return

    return reprtpl.format(
        ttype="TensorArray",
        name=proto.name,
        reprs="level=%d, %s" % (proto.type.tensor_array.lod_level,
                                repr_tensor(proto.type.lod_tensor.tensor)))


type_handlers = [
    repr_lodtensor,
    repr_selected_rows,
    repr_tensor_array,
]


def repr_var(vardesc):
    for handler in type_handlers:
        res = handler(vardesc)
        if res:
            return res


def pprint_program_codes(program_desc):
    reprs = []
    for block_idx in range(program_desc.desc.num_blocks()):
        block_desc = program_desc.block(block_idx)
        block_repr = pprint_block_codes(block_desc)
        reprs.append(block_repr)
    return '\n'.join(reprs)


def pprint_block_codes(block_desc, show_backward=False):
    def is_op_backward(op_desc):
        if op_desc.type.endswith('_grad'): return True

        def is_var_backward(var):
            if "@GRAD" in var.parameter: return True
            for arg in var.arguments:
                if "@GRAD" in arg: return True

        for var in op_desc.inputs:
            if is_var_backward(var): return True
        for var in op_desc.outputs:
            if is_var_backward(var): return True
        return False

    def is_var_backward(var_desc):
        return "@GRAD" in var_desc.name

    if type(block_desc) is not framework_pb2.BlockDesc:
        block_desc = framework_pb2.BlockDesc.FromString(
            block_desc.desc.serialize_to_string())
    var_reprs = []
    op_reprs = []
    for var in block_desc.vars:
        if not show_backward and is_var_backward(var):
            continue
        var_reprs.append(repr_var(var))

    for op in block_desc.ops:
        if not show_backward and is_op_backward(op): continue
        op_reprs.append(repr_op(op))

    tpl = "// block-{idx}  parent-{pidx}\n// variables\n{vars}\n\n// operators\n{ops}\n"
    return tpl.format(
        idx=block_desc.idx,
        pidx=block_desc.parent_idx,
        vars='\n'.join(var_reprs),
        ops='\n'.join(op_reprs), )


def repr_attr(desc):
    tpl = "{key}={value}"
    valgetter = [
        lambda attr: attr.i,
        lambda attr: attr.f,
        lambda attr: attr.s,
        lambda attr: attr.ints,
        lambda attr: attr.floats,
        lambda attr: attr.strings,
        lambda attr: attr.b,
        lambda attr: attr.bools,
        lambda attr: attr.block_idx,
        lambda attr: attr.l,
    ]
    key = desc.name
    value = valgetter[desc.type](desc)
    if key == "dtype":
        value = repr_data_type(value)
    return tpl.format(key=key, value=str(value)), (key, value)


def _repr_op_fill_constant(optype, inputs, outputs, attrs):
    if optype == "fill_constant":
        return "{output} = {data} [shape={shape}]".format(
            output=','.join(outputs),
            data=attrs['value'],
            shape=str(attrs['shape']))


op_repr_handlers = [_repr_op_fill_constant, ]


def repr_op(opdesc):
    optype = None
    attrs = []
    attr_dict = {}
    is_target = None
    inputs = []
    outputs = []

    tpl = "{outputs} = {optype}({inputs}{is_target}) [{attrs}]"
    args2value = lambda args: args[0] if len(args) == 1 else str(list(args))
    for var in opdesc.inputs:
        key = var.parameter
        value = args2value(var.arguments)
        inputs.append("%s=%s" % (key, value))
    for var in opdesc.outputs:
        value = args2value(var.arguments)
        outputs.append(value)
    for attr in opdesc.attrs:
        attr_repr, attr_pair = repr_attr(attr)
        attrs.append(attr_repr)
        attr_dict[attr_pair[0]] = attr_pair[1]

    is_target = opdesc.is_target

    for handler in op_repr_handlers:
        res = handler(opdesc.type, inputs, outputs, attr_dict)
        if res: return res

    return tpl.format(
        outputs=', '.join(outputs),
        optype=opdesc.type,
        inputs=', '.join(inputs),
        attrs="{%s}" % ','.join(attrs),
        is_target=", is_target" if is_target else "")


def draw_block_graphviz(block, highlights=None, path="./temp.dot"):
    '''
    Generate a debug graph for block.
    Args:
        block(Block): a block.
    '''
    graph = GraphPreviewGenerator("some graph")
    # collect parameters and args
    protostr = block.desc.serialize_to_string()
    desc = framework_pb2.BlockDesc.FromString(six.binary_type(protostr))

    def need_highlight(name):
        if highlights is None: return False
        for pattern in highlights:
            assert type(pattern) is str
            if re.match(pattern, name):
                return True
        return False

    # draw parameters and args
    vars = {}
    for var in desc.vars:
        # TODO(gongwb): format the var.type
        # create var
        if var.persistable:
            varn = graph.add_param(
                var.name,
                str(var.type).replace("\n", "<br />", 1),
                highlight=need_highlight(var.name))
        else:
            varn = graph.add_arg(var.name, highlight=need_highlight(var.name))
        vars[var.name] = varn

    def add_op_link_var(op, var, op2var=False):
        for arg in var.arguments:
            if arg not in vars:
                # add missing variables as argument
                vars[arg] = graph.add_arg(arg, highlight=need_highlight(arg))
            varn = vars[arg]
            highlight = need_highlight(op.description) or need_highlight(
                varn.description)
            if op2var:
                graph.add_edge(op, varn, highlight=highlight)
            else:
                graph.add_edge(varn, op, highlight=highlight)

    for op in desc.ops:
        opn = graph.add_op(op.type, highlight=need_highlight(op.type))
        for var in op.inputs:
            add_op_link_var(opn, var, False)
        for var in op.outputs:
            add_op_link_var(opn, var, True)

    graph(path, show=False)
