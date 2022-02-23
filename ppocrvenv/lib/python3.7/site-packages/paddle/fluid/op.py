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

import numpy as np
import six

import paddle.fluid.core as core
import paddle.fluid.proto.framework_pb2 as framework_pb2


def get_all_op_protos():
    """
    Get all registered op proto from PaddlePaddle C++ end.
    :return: A list of registered OpProto.
    """
    protostrs = core.get_all_op_protos()
    ret_values = []
    for pbstr in protostrs:
        op_proto = framework_pb2.OpProto.FromString(six.binary_type(pbstr))
        ret_values.append(op_proto)
    return ret_values


def is_str(s):
    return isinstance(s, six.string_types)


class OpDescCreationMethod(object):
    """
    Convert the user's input(only keyword arguments are supported) to OpDesc
    based on the OpProto.

    :param op_proto: The OpProto object.
    :type op_proto: op_proto_pb2.OpProto
    """

    def __init__(self, op_proto):
        if not isinstance(op_proto, framework_pb2.OpProto):
            raise TypeError(
                "Type of op_proto should be OpProto in PaddlePaddle.")
        self.__op_proto__ = op_proto

    def __call__(self, *args, **kwargs):
        """
        Convert user's input to OpDesc. Only keyword arguments are supported.
        :return: The OpDesc based on user input.
        :rtype: op_desc_pb2.OpDesc
        """
        if len(args) != 0:
            raise ValueError("Only keyword arguments are supported.")
        op_desc = framework_pb2.OpDesc()
        for input_parameter in self.__op_proto__.inputs:
            input_arguments = kwargs.get(input_parameter.name, [])
            if is_str(input_arguments):
                input_arguments = [input_arguments]

            if not input_parameter.duplicable and len(input_arguments) > 1:
                raise ValueError(
                    "Input %s expects only one input, but %d are given." %
                    (input_parameter.name, len(input_arguments)))

            ipt = op_desc.inputs.add()
            ipt.parameter = input_parameter.name
            ipt.arguments.extend(input_arguments)

        for output_parameter in self.__op_proto__.outputs:
            output_arguments = kwargs.get(output_parameter.name, [])
            if is_str(output_arguments):
                output_arguments = [output_arguments]

            if not output_parameter.duplicable and len(output_arguments) > 1:
                raise ValueError(
                    "Output %s expects only one output, but %d are given." %
                    (output_parameter.name, len(output_arguments)))

            out = op_desc.outputs.add()
            out.parameter = output_parameter.name
            out.arguments.extend(output_arguments)

        # Types
        op_desc.type = self.__op_proto__.type

        # Attrs
        for attr in self.__op_proto__.attrs:
            if attr.generated:
                continue
            user_defined_attr = kwargs.get(attr.name, None)
            if user_defined_attr is not None:
                new_attr = op_desc.attrs.add()
                new_attr.name = attr.name
                new_attr.type = attr.type
                if isinstance(user_defined_attr, np.ndarray):
                    user_defined_attr = user_defined_attr.tolist()
                if attr.type == framework_pb2.INT:
                    new_attr.i = user_defined_attr
                elif attr.type == framework_pb2.FLOAT:
                    new_attr.f = user_defined_attr
                elif attr.type == framework_pb2.LONG:
                    new_attr.l = user_defined_attr
                elif attr.type == framework_pb2.STRING:
                    new_attr.s = user_defined_attr
                elif attr.type == framework_pb2.BOOLEAN:
                    new_attr.b = user_defined_attr
                elif attr.type == framework_pb2.INTS:
                    new_attr.ints.extend(user_defined_attr)
                elif attr.type == framework_pb2.FLOATS:
                    new_attr.floats.extend(user_defined_attr)
                elif attr.type == framework_pb2.STRINGS:
                    new_attr.strings.extend(user_defined_attr)
                elif attr.type == framework_pb2.BOOLEANS:
                    new_attr.bools.extend(user_defined_attr)
                elif attr.type == framework_pb2.LONGS:
                    new_attr.longs.extend(user_defined_attr)
                else:
                    raise NotImplementedError(
                        "A not supported attribute type: %s." % (
                            str(attr.type)))

        return op_desc

    @staticmethod
    def any_is_true(generator):
        """
        Reduce a boolean array to a single boolean parameter. If any element in
        the array is True, this function will return True, otherwise False.
        """
        for flag in generator:
            if flag:
                return True
        return False


class OpInfo(object):
    def __init__(self, name, method, inputs, outputs, attrs):
        self.name = name
        self.method = method
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs


def create_op_creation_method(op_proto):
    """
    Generate op creation method for an OpProto.
    """
    method = OpDescCreationMethod(op_proto)

    def __impl__(*args, **kwargs):
        opdesc = method(*args, **kwargs)
        return core.Operator.create(opdesc.SerializeToString())

    return OpInfo(
        method=__impl__,
        name=op_proto.type,
        inputs=[(var.name, var.duplicable) for var in op_proto.inputs],
        outputs=[(var.name, var.duplicable) for var in op_proto.outputs],
        attrs=[attr.name for attr in op_proto.attrs])


class OperatorFactory(object):
    def __init__(self):
        self.op_methods = dict()

        for op_proto in get_all_op_protos():
            method = create_op_creation_method(op_proto)
            self.op_methods[method.name] = method

    def __call__(self, *args, **kwargs):
        if "type" in kwargs:
            if len(args) != 0:
                raise ValueError(
                    "Except the argument \"type\","
                    "all of the other arguments should be keyword arguments.")
            t = kwargs.pop("type")
        else:
            if len(args) != 1:
                raise ValueError(
                    "Except the argument \"type\","
                    "all of the other arguments should be keyword arguments.")
            t = args[0]

        return self.get_op_info(t).method(**kwargs)

    def types(self):
        return list(self.op_methods.keys())

    def get_op_info(self, t):
        if t not in self.op_methods:
            raise ValueError("The operator: %s is not registered." % t)
        return self.op_methods.get(t)

    def get_op_input_names(self, type):
        return [x[0] for x in self.get_op_info(type).inputs]

    def get_op_inputs(self, type):
        return self.get_op_info(type).inputs

    def get_op_output_names(self, type):
        return [x[0] for x in self.get_op_info(type).outputs]

    def get_op_outputs(self, type):
        return self.get_op_info(type).outputs

    def get_op_attr_names(self, type):
        return self.get_op_info(type).attrs


class __RecurrentOp__(object):
    __proto__ = None
    type = "recurrent"

    def __init__(self):
        # cache recurrent_op's proto
        if self.__proto__ is None:
            for op_proto in get_all_op_protos():
                if op_proto.type == self.type:
                    self.__proto__ = op_proto

    def __call__(self, *args, **kwargs):
        if self.type not in args and "type" not in kwargs:
            kwargs["type"] = self.type
        # create proto
        create_method = OpDescCreationMethod(self.__proto__)
        proto = create_method(*args, **kwargs)
        # create rnnop
        return core.RecurrentOp.create(proto.SerializeToString())


class __DynamicRecurrentOp__(object):
    __proto__ = None
    type = "dynamic_recurrent"

    def __init__(self):
        # cache recurrent_op's proto
        if self.__proto__ is None:
            for op_proto in get_all_op_protos():
                if op_proto.type == self.type:
                    self.__proto__ = op_proto

    def __call__(self, *args, **kwargs):
        if self.type not in args and "type" not in kwargs:
            kwargs["type"] = self.type
        # create proto
        create_method = OpDescCreationMethod(self.__proto__)
        proto = create_method(*args, **kwargs)
        # create rnnop
        return core.DynamicRecurrentOp.create(proto.SerializeToString())


class __CondOp__(object):
    __proto__ = None
    type = "cond"

    def __init__(self):
        # cache recurrent_op's proto
        if self.__proto__ is None:
            for op_proto in get_all_op_protos():
                if op_proto.type == self.type:
                    self.__proto__ = op_proto

    def __call__(self, *args, **kwargs):
        if self.type not in args and "type" not in kwargs:
            kwargs["type"] = self.type
        # create proto
        create_method = OpDescCreationMethod(self.__proto__)
        proto = create_method(*args, **kwargs)
        # create condop
        return core.CondOp.create(proto.SerializeToString())


Operator = OperatorFactory()  # The default global factory
RecurrentOp = __RecurrentOp__()
DynamicRecurrentOp = __DynamicRecurrentOp__()
CondOp = __CondOp__()
