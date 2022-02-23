#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import re
import functools
import warnings
import string

from six.moves import cStringIO
from ..proto import framework_pb2
from ..framework import OpProtoHolder, Variable, core, convert_np_dtype_to_dtype_, in_dygraph_mode
from ..layer_helper import LayerHelper
from ..data_feeder import check_variable_and_dtype
from paddle import _C_ops

__all__ = [
    'generate_layer_fn', 'generate_activation_fn', 'generate_inplace_fn',
    'autodoc', 'templatedoc'
]


def _convert_(name):
    """
    Formatting.

    Args:
       name: The name/alias

    This function takes in a name and converts it to a standard format of
    group1_group2. Where as per the regular expression, group1 can have
    alphabets and numbers and group2 has capital alphabets.

    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _type_to_str_(tp):
    return framework_pb2.AttrType.Name(tp)


_two_dollar_pattern_ = re.compile(r"\$\$([^\$]+)\$\$")
_single_dollar_pattern_ = re.compile(r"\$([^\$]+)\$")
_two_bang_pattern_ = re.compile(r"!!([^!]+)!!")


def escape_math(text):
    #return _two_bang_pattern_.sub(
    #    r'$$\1$$',
    #    _single_dollar_pattern_.sub(r':math:\n`\1`',
    #                                _two_dollar_pattern_.sub(r"!!\1!!", text)))
    return _two_dollar_pattern_.sub(r':math:`\1`', text)


def _generate_doc_string_(op_proto,
                          additional_args_lines=None,
                          skip_attrs_set=None):
    """
    Generate docstring by OpProto

    Args:
        op_proto (framework_pb2.OpProto): a protobuf message typed OpProto

    Returns:
        str: the document string
    """

    if not isinstance(op_proto, framework_pb2.OpProto):
        raise TypeError("OpProto should be `framework_pb2.OpProto`")

    buf = cStringIO()
    buf.write(escape_math(op_proto.comment))
    buf.write('\nArgs:\n')
    for each_input in op_proto.inputs:
        line_begin = '    {0}'.format(_convert_(each_input.name))
        buf.write(line_begin)
        buf.write(" (Tensor): ")
        buf.write(escape_math(each_input.comment))
        if each_input.duplicable:
            buf.write("  Duplicatable.")
        if each_input.dispensable:
            buf.write("  Optional.")
        buf.write('\n')

    skip_attrs = OpProtoHolder.generated_op_attr_names()
    # attr use_mkldnn and is_test also should not be visible to users.
    skip_attrs.add("use_mkldnn")
    skip_attrs.add("is_test")
    skip_attrs.add("use_cudnn")

    if skip_attrs_set:
        for t in skip_attrs_set:
            skip_attrs.add(t)

    for each_attr in op_proto.attrs:
        if each_attr.name in skip_attrs:
            continue
        buf.write('    ')
        buf.write(each_attr.name)
        buf.write(' (')
        buf.write(_type_to_str_(each_attr.type))
        buf.write('): ')
        buf.write(escape_math(each_attr.comment))
        buf.write('\n')

    if additional_args_lines is not None:
        for line in additional_args_lines:
            line = line.strip()
            buf.write('    ')
            buf.write(line)
            buf.write('\n')

    if len(op_proto.outputs) != 0:
        buf.write('\nReturns:\n')
        buf.write('    ')
        for each_opt in op_proto.outputs:
            if not each_opt.intermediate:
                break
        buf.write(_convert_(each_opt.name))
        buf.write(' (Tensor): ')
        buf.write(escape_math(each_opt.comment))

    return buf.getvalue()


def generate_layer_fn(op_type):
    """Register the Python layer for an Operator.

    Args:
       op_type: The name of the operator to be created.

    This function takes in the operator type (sigmoid, mean , average etc) and
    creates the operator functionality.

    """
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)
    not_intermediate_outputs = \
        [output for output in op_proto.outputs if not output.intermediate]
    intermediate_outputs = \
        [output for output in op_proto.outputs if output.intermediate]

    if len(not_intermediate_outputs) != 1:
        raise ValueError("Only one non intermediate output operator can be",
                         "automatically generated. {0}".format(op_type))

    if not_intermediate_outputs[0].duplicable:
        raise ValueError(
            "Only non duplicable op can be automatically generated.")

    for output in intermediate_outputs:
        if output.duplicable:
            raise ValueError("The op can be automatically generated only when ",
                             "all intermediate ops are not duplicable.")

    o_name = not_intermediate_outputs[0].name
    intermediate_output_names = [output.name for output in intermediate_outputs]

    def infer_and_check_dtype(op_proto, *args, **kwargs):
        """
        This function performs the sanity check for dtype and
        instance type.
        """
        dtype = None
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            if len(val) == 0:
                if len(args) == 0:
                    continue
                val = [args[0]]
                args = args[1:]

            for each in val:
                if not isinstance(each, Variable):
                    raise ValueError("input of {0} must be variable".format(
                        op_type))

                if dtype is None:
                    dtype = each.dtype
                elif dtype != each.dtype:
                    raise ValueError(
                        "operator {0} must input same dtype. {1} vs {2}".format(
                            op_type, dtype, each.dtype))

        if dtype is None:
            arg_dtype = kwargs.get("dtype")
            if arg_dtype:
                if not isinstance(arg_dtype, core.VarDesc.VarType):
                    dtype = convert_np_dtype_to_dtype_(arg_dtype)
                else:
                    dtype = arg_dtype
            else:
                dtype = core.VarDesc.VarType.FP32
        return dtype

    def func(*args, **kwargs):
        helper = LayerHelper(op_type, **kwargs)

        dtype = infer_and_check_dtype(op_proto, *args, **kwargs)

        inputs = dict()
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            if len(val) == 0 and len(args) != 0:
                val = args[0]
                args = args[1:]
            inputs[ipt.name] = val

        outputs = dict()
        out = kwargs.pop(_convert_(o_name), [])
        if out:
            out_var = out[0] if (isinstance(out, list) or
                                 isinstance(out, tuple)) else out
        else:
            out_var = helper.create_variable_for_type_inference(dtype=dtype)
        outputs[o_name] = [out_var]
        for name in intermediate_output_names:
            outputs[name] = [
                helper.create_variable_for_type_inference(dtype=dtype)
            ]
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=kwargs)
        return helper.append_activation(out_var)

    func.__name__ = op_type
    func.__doc__ = _generate_doc_string_(op_proto)
    return func


def generate_activation_fn(op_type):
    """Register the Python layer for an Operator without Attribute.

    Args:
       op_type: The name of the operator to be created.

    This function takes in the operator type (sigmoid, exp , tanh etc) and
    creates the operator functionality.

    """
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)

    def func(x, name=None):
        if in_dygraph_mode():
            op = getattr(_C_ops, op_type)
            return op(x)

        if op_type not in ["abs", "exp", "square"]:
            check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                     op_type)
        else:
            # abs exp square ops support dtype(int32, int64, float16, float32, float64)
            check_variable_and_dtype(
                x, 'x', ['int32', 'int64', 'float16', 'float32', 'float64'],
                op_type)

        helper = LayerHelper(op_type, **locals())

        output = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type=op_type, inputs={"X": x}, outputs={"Out": output})
        return output

    func.__name__ = op_type
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`."
        ])
    return func


def generate_inplace_fn(inplace_op_type):
    """Register the Python layer for an Inplace Operator without Attribute.

    Args:
       inplace_op_type: The name of the inplace operator to be created.

    This function takes in the inplace operator type (exp_ , ceil_ etc) and
    creates the operator functionality.
    """
    origin_op_type = inplace_op_type[:-1]

    def func(x, name=None):
        if in_dygraph_mode():
            op = getattr(_C_ops, inplace_op_type)
            return op(x)
        warnings.warn(
            "In static mode, {}() is the same as {}() and does not perform inplace operation.".
            format(inplace_op_type, origin_op_type))
        return generate_activation_fn(origin_op_type)(x, name)

    func.__name__ = inplace_op_type
    func.__doc__ = """
Inplace version of ``{0}`` API, the output Tensor will be inplaced with input ``x``.
Please refer to :ref:`api_fluid_layers_{1}`.
""".format(origin_op_type, origin_op_type)

    return func


def autodoc(comment=""):
    def __impl__(func):
        func.__doc__ = _generate_doc_string_(OpProtoHolder.instance(
        ).get_op_proto(func.__name__)) + comment
        return func

    return __impl__


def templatedoc(op_type=None):
    """
    Decorator of layer function. It will use the docstring from the layer
    function as the template. The template arguments are:

    * ${comment}: The operator comment written in CPP.
    * ${{name}_comment}: The comment of ${name} written with AddAttr, AddOutput,
        and AddInput. The ${name} is Python snake style. i.e., xxx_xxx.
    * ${{name}_type}: The type of ${name}.

    Returns:
        Decorated function.
    """

    def trim_ending_dot(msg):
        return msg.rstrip('.')

    def __impl__(func):
        if op_type is None:
            op_type_name = func.__name__
        else:
            op_type_name = op_type
        op_proto = OpProtoHolder.instance().get_op_proto(op_type_name)
        tmpl = string.Template(func.__doc__)

        comment_lines = op_proto.comment.split("\n")
        comment = ""
        for line in comment_lines:
            line = line.strip()
            if len(line) != 0:
                comment += escape_math(line)
                comment += " "
            elif len(comment) != 0:
                comment += "\n    \n    "

        args = {"comment": trim_ending_dot(comment)}
        for each_input in op_proto.inputs:
            input_name = _convert_(each_input.name)
            args["{0}_comment".format(input_name)] = trim_ending_dot(
                each_input.comment)
            args["{0}_type".format(input_name)] = "Variable"
        for each_attr in op_proto.attrs:
            input_name = _convert_(each_attr.name)
            args["{0}_comment".format(input_name)] = trim_ending_dot(
                each_attr.comment)
            args["{0}_type".format(input_name)] = _type_to_str_(each_attr.type)

        for each_opt in op_proto.outputs:
            output_name = _convert_(each_opt.name)
            args["{0}_comment".format(output_name)] = trim_ending_dot(
                each_opt.comment)
            args["{0}_type".format(output_name)] = "Variable"
        func.__doc__ = tmpl.substitute(args)
        return func

    return __impl__


def add_sample_code(func, sample_code):
    """
    Append sample code for dynamically generated functions. 

    Args:
       func: The function of the function to be append sample code to.
       sample_code: sample code session in rst format.
    """
    func.__doc__ = func.__doc__ + sample_code
