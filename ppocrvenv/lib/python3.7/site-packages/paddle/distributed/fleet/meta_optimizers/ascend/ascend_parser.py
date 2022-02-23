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
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.core as core
import numpy as np
from paddle.distributed import fleet
from functools import reduce

__all__ = []

registerd_op = {## forwards
                "elementwise_add": "AddParser",
                "matmul": "MatMulParser",
                "mul": "MulParser",
                "relu": "ReluParser",
                "softmax_with_cross_entropy": "SoftmaxWithCrossEntropyParser",
                "shape": "ShapeParser",
                "fill_constant": "FillConstantParser",
                "reduce_sum": "ReduceSumParser",
                "elementwise_mul": "DotMulParser",
                "elementwise_div": "DotDivParser",
                "elementwise_pow": "DotPowParser",
                "elementwise_max": "MaxParser",
                "elementwise_min": "MinParser",
                "elementwise_sub": "DotSubParser",
                "pow": "PowParser",
                "gelu": "GeluParser",
                "sqrt": "SqrtParser",
                "log": "LogParser",
                "sum": "SumParser",
                "logical_not": "LogicalNotParser",
                "gather": "GatherParser",
                "scatter": "ScatterParser",
                "cast": "CastParser",
                "tanh": "TanhParser",
                "stack": "StackParser",
                "square": "SquareParser",
                "unsqueeze2": "UnSqueezeParser",
                "assign": "AssignParser",
                "softmax": "SoftMaxParser",
                "reshape2": "ReshapeParser",
                "transpose2": "TransposeParser",
                "layer_norm": "LayerNormParser",
                "less_than": "LessParser",
                "mean": "MeanParser",
                "scale": "ScaleParser",
                "slice": "SliceParser",
                "top_k": "TopkParser",
                "accuracy": "AccuracyParser",
                #"increment": "IncrementParser",
                "lookup_table": "LookupTableParser",
                "truncated_gaussian_random": "TruncatedNormalParser",
                "c_allgather": "AllGatherParser",
                "c_allreduce_sum": "AllReduceSumParser",
                "c_allreduce_max": "AllReduceMaxParser",
                "c_broadcast": "BroadcastParser",
                "c_reduce_scatter": "ReduceScatterParser",
                "c_send": "SendParser",
                "c_receive": "ReceiveParser",
                "uniform_random": "UniformRandomParser",
                "range": "RangeParser",
                "equal": "EqualParser",
                "expand": "ExpandParser",
                "squeeze2": "SqueezeParser",


                ## backwords
                "matmul_grad": "MatMulGradParser",
                "mul_grad": "MulGradParser",
                "relu_grad": "ReluGradParser",
                "reduce_sum_grad": "ReduceSumGradParser",
                "softmax_with_cross_entropy_grad": "SoftmaxWithCrossEntropyGradParser",
                "tanh_grad":"TanhGradParser",
                "log_grad":"LogGradParser",
                "pow_grad": "PowGradParser",
                "sqrt_grad": "SqrtGradParser",
                "gelu_grad": "GeluGradParser",
                "mean_grad": "MeanGradParser",
                'lookup_table_grad': "LookUpTableGradParser",
                "elementwise_mul_grad": "DotMulGradParser",
                "elementwise_add_grad": "DotAddGradParser",
                "elementwise_div_grad": "DotDivGradParser",
                "softmax_grad": "SoftmaxGradParser",
                "slice_grad": "SliceGradParser",
                "reshape2_grad": "ReshapeGradParser",
                "gather_grad": "GatherGradParser",
                "transpose2_grad": "TransposeGradParser",
                "layer_norm_grad": "LayerNormGradParser",

                ## opt
                "sgd": "SGDParser",
                #"adam": "AdamParser",
                }
global_cnt = -1
global_input_cnt = -1


class AscendHelper(object):
    def __init__(self):
        self.dtype2ge_map = {
            0: core.GEDataType.DT_BOOL,
            1: core.GEDataType.DT_INT16,
            2: core.GEDataType.DT_INT32,
            3: core.GEDataType.DT_INT64,
            4: core.GEDataType.DT_FLOAT16,
            5: core.GEDataType.DT_FLOAT,
            6: core.GEDataType.DT_DOUBLE
        }
        self.dtype2np_map = {
            0: "bool",
            1: "int16",
            2: "int32",
            3: "int64",
            4: "float16",
            5: "float32",
            6: "float64"
        }
        self.dtype2paddle_inv_map = {"VarType.FP32": 0, "VarType.FP16": 1}

    def dtype2ge(self, dtype):
        assert dtype in self.dtype2ge_map, "dtype[%d] is not supported %d" % (
            dtype)
        return self.dtype2ge_map[dtype]

    def dtype2np(self, index):
        assert index in self.dtype2np_map, "index[%d] is not supported %d" % (
            index)
        return self.dtype2np_map[index]


class AscendParserFactory(object):
    def __init__(self, graph, var2geop):
        self.graph = graph
        self.var2geop = var2geop

    def create_parse(self, parser_class):
        try:
            parser = globals()[parser_class](self.graph, self.var2geop)
            return parser
        except:
            raise ValueError("parser class %s does not exist" % parser_class)


class AscendParserBase(object):
    def __init__(self, graph, var2geop):
        self.graph = graph
        self.var2geop = var2geop
        self.op = None
        self.ascend_helper = AscendHelper()

    def _get_ge_input(self, input_var_name):
        assert input_var_name in self.var2geop, "var %s not created before" % (
            input_var_name)
        return self.var2geop[input_var_name]

    def update_output(self, geop_list, index_list):
        output_num = len(self.op.output_names)
        assert output_num == len(
            index_list
        ), "Parser[%s]'s output number[%d] is not equal to parameters number[%d]" % (
            self.parser_name, len(index_list), output_num)
        for output_id in range(output_num):
            arguments = self.op.output(self.op.output_names[output_id])
            if len(arguments) > 0:
                assert len(arguments) == len(
                    index_list[output_id]
                ), "Parser[%s]'s %dth argument number[%d] is not equal to paddle's number[%d]" % (
                    self.parser_name, output_id, len(index_list[output_id]),
                    len(arguments))
                for i in range(len(arguments)):
                    self.var2geop[arguments[i]] = geop_list[index_list[
                        output_id][i]]

        for geop in geop_list:
            self.graph.add_op(geop)

    def apply(self, op):
        self.op = op
        assert self.op.type == self.parser_name, "op [%s] != parser_name[%s]" % (
            self.op.type, self.parser_name)
        #print("begin to parse op %s" % (self.parser_name))
        geop_list, index_list = self._apply()
        self.update_output(geop_list, index_list)

    def _mark_as_input(self, ge_tensor):
        global global_input_cnt
        global_input_cnt += 1
        self.var2geop["geinput." + str(global_input_cnt)] = ge_tensor

    def _accumulated_op_id(self):
        global global_cnt
        global_cnt += 1
        name = "." + str(global_cnt)
        return name

    def _create_ge_tensor(self, shape, dtype, value):
        tensor_desc = core.GETensorDesc(
            core.GEShape(shape), core.GEFormat.FORMAT_ND,
            self.ascend_helper.dtype2ge(dtype))
        tensor = core.GETensor(tensor_desc)

        data = (value * np.ones((
            shape))).reshape(shape).astype(self.ascend_helper.dtype2np(dtype))
        buf = data.tobytes()
        data_8 = np.frombuffer(buf, dtype=np.uint8)
        tensor.set_data(data_8)
        return tensor

    def _get_ge_tensor(self, shape, dtype, value_list):
        tensor_desc = core.GETensorDesc(
            core.GEShape(shape), core.GEFormat.FORMAT_ND,
            self.ascend_helper.dtype2ge(dtype))
        tensor = core.GETensor(tensor_desc)

        data = np.array(value_list).reshape(shape).astype(
            self.ascend_helper.dtype2np(dtype))
        buf = data.tobytes()
        data_8 = np.frombuffer(buf, dtype=np.uint8)
        tensor.set_data(data_8)

        tensor_const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor)

        return tensor_const

    def _get_variable(self, shape, dtype, tensor):
        if dtype == "int32":
            type = core.GEDataType.DT_INT32
        elif dtype == "float32":
            type = core.GEDataType.DT_FLOAT

        var = core.GEOperatorFactory.create_operator(
            "variable" + self._accumulated_op_id(), "Variable")
        var.update_output_desc("y",
                               core.GETensorDesc(
                                   core.GEShape(shape), core.GEFormat.FORMAT_ND,
                                   type))
        assign = core.GEOperatorFactory.create_operator(
            "assign" + self._accumulated_op_id(), "Assign").set_input(
                "value", tensor).set_input("ref", var)

        return assign

    def _create_shape_tensor(self):
        tensor_desc = core.GETensorDesc(
            core.GEShape([2]), core.GEFormat.FORMAT_ND,
            core.GEDataType.DT_INT32)
        tensor = core.GETensor(tensor_desc)

        data = np.ones((2)).astype("int32").reshape([2])
        data[0] = 64
        buf = data.tobytes()
        data_8 = np.frombuffer(buf, dtype=np.uint8)
        tensor.set_data(data_8)
        return tensor

    def _get_GEtensor_shape(self, tensor):
        tensor_shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", tensor)
        tensor_shape = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", tensor_shape).set_attr_int32("dst_type", 0)
        return tensor_shape


class AddParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AddParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_add"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        add = core.GEOperatorFactory.create_operator(
            "add" + self._accumulated_op_id(),
            "Add").set_input("x1", x).set_input("x2", y)
        return [add], [[0]]


class DotSubParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotSubParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_sub"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        sub = core.GEOperatorFactory.create_operator(
            "sub" + self._accumulated_op_id(),
            "Sub").set_input("x1", x).set_input("x2", y)
        return [sub], [[0]]


class DotMulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotMulParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_mul"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        mul = core.GEOperatorFactory.create_operator(
            "dotmul" + self._accumulated_op_id(),
            "Mul").set_input("x1", x).set_input("x2", y)
        return [mul], [[0]]


class DotDivParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotDivParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_div"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        div = core.GEOperatorFactory.create_operator(
            "dotdiv" + self._accumulated_op_id(),
            "Div").set_input("x1", x).set_input("x2", y)
        return [div], [[0]]


class DotPowParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotPowParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_pow"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        pow = core.GEOperatorFactory.create_operator(
            "dotpow" + self._accumulated_op_id(),
            "Pow").set_input("x1", x).set_input("x2", y)
        return [pow], [[0]]


class LessParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LessParser, self).__init__(graph, var2geop)
        self.parser_name = "less_than"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        less_than = core.GEOperatorFactory.create_operator(
            "less_than" + self._accumulated_op_id(),
            "Less").set_input("x1", x).set_input("x2", y)
        return [less_than], [[0]]


class MaxParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MaxParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_max"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        max_out = core.GEOperatorFactory.create_operator(
            "max" + self._accumulated_op_id(),
            "Maximum").set_input("x1", x).set_input("x2", y)
        return [max_out], [[0]]


class MinParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MinParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_min"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        min_out = core.GEOperatorFactory.create_operator(
            "min" + self._accumulated_op_id(),
            "Minimum").set_input("x1", x).set_input("x2", y)
        return [min_out], [[0]]


## cal
class LogParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LogParser, self).__init__(graph, var2geop)
        self.parser_name = "log"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        log = core.GEOperatorFactory.create_operator(
            "log" + self._accumulated_op_id(), "Log").set_input("x", x)
        return [log], [[0]]


class SqrtParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SqrtParser, self).__init__(graph, var2geop)
        self.parser_name = "sqrt"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        sqrt = core.GEOperatorFactory.create_operator(
            "sqrt" + self._accumulated_op_id(), "Sqrt").set_input("x", x)
        return [sqrt], [[0]]


class PowParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(PowParser, self).__init__(graph, var2geop)
        self.parser_name = "pow"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        factor = self.op.attr("factor")
        pow_value = core.GEOperatorFactory.create_operator(
            "pow" + self._accumulated_op_id(),
            "Power").set_input("x", x).set_attr_float(
                "power", factor).set_attr_float("scale", 1.0).set_attr_float(
                    "shift", 0.0)
        return [pow_value], [[0]]


class SquareParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SquareParser, self).__init__(graph, var2geop)
        self.parser_name = "square"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        square = core.GEOperatorFactory.create_operator(
            "square" + self._accumulated_op_id(), "Square").set_input("x", x)
        return [square], [[0]]


class SumParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SumParser, self).__init__(graph, var2geop)
        self.parser_name = "sum"

    def _apply(self):
        len_list = len(self.op.input_arg_names)
        if len_list < 2:
            assert False, "the size of input list must large or equal 2"
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        sum = core.GEOperatorFactory.create_operator(
            "sum" + self._accumulated_op_id(),
            "Add").set_input("x1", x).set_input("x2", y)
        for i in range(2, len_list):
            y = self._get_ge_input(self.op.input_arg_names[i])
            sum = core.GEOperatorFactory.create_operator(
                "sum" + self._accumulated_op_id(),
                "Add").set_input("x1", sum).set_input("x2", y)
        return [sum], [[0]]


class LogicalNotParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LogicalNotParser, self).__init__(graph, var2geop)
        self.parser_name = "logical_not"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        logical_not = core.GEOperatorFactory.create_operator(
            "logical_not" + self._accumulated_op_id(),
            "LogicalNot").set_input("x", x)
        return [logical_not], [[0]]


class MeanParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MeanParser, self).__init__(graph, var2geop)
        self.parser_name = "mean"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        mean = core.GEOperatorFactory.create_operator(
            "mean" + self._accumulated_op_id(),
            "ReduceMeanD").set_input("x", x).set_attr_bool(
                "keep_dims", False).set_attr_vec_int32("axes", [])
        return [mean], [[0]]


class ReduceSumParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceSumParser, self).__init__(graph, var2geop)
        self.parser_name = "reduce_sum"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("dim")
        keep_dims = self.op.attr("keep_dim")
        reduce_all = self.op.attr("reduce_all")
        x_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        if reduce_all:
            axes = list(range(len(x_shape)))
        reduce_sum = core.GEOperatorFactory.create_operator(
            "reduce_sum" + self._accumulated_op_id(),
            "ReduceSumD").set_input("x", x, 0).set_attr_vec_int32(
                "axes", axes).set_attr_bool("keep_dims", keep_dims)
        return [reduce_sum], [[0]]


#class IncrementParser(AscendParserBase):
#    def __init__(self, graph, var2geop):
#        super(IncrementParser, self).__init__(graph, var2geop)
#        self.parser_name = "increment"
#
#    def _apply(self): 
#        x = self._get_ge_input(self.op.input_arg_names[0])
#        step = self.op.attr("step") #self._get_ge_input(self.op.input_arg_names[1])
#        print("step: ", step)
#            
#        increment = core.GEOperatorFactory.create_operator("adds" + self._accumulated_op_id(), "Adds").set_input("x", x).set_attr_float("value", step) #set_input("x2", bias)
#        
#        return [increment]


## matrix cal
class MatMulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MatMulParser, self).__init__(graph, var2geop)
        self.parser_name = "matmul"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        transpose_x = self.op.attr("transpose_X")
        transpose_y = self.op.attr("transpose_Y")

        x1_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        x2_shape = self.op.block.var(self.op.input_arg_names[1]).shape

        if len(x1_shape) > 2:
            matmul = core.GEOperatorFactory.create_operator(
                "matmul" + self._accumulated_op_id(), "BatchMatMul").set_input(
                    "x1", x).set_input("x2", y).set_attr_bool(
                        "adj_x1",
                        transpose_x).set_attr_bool("adj_x2", transpose_y)
        elif len(x1_shape) == 2:
            matmul = core.GEOperatorFactory.create_operator(
                "matmul" + self._accumulated_op_id(),
                "MatMul").set_input("x1", x).set_input("x2", y).set_attr_bool(
                    "transpose_x1", transpose_x).set_attr_bool("transpose_x2",
                                                               transpose_y)
        else:
            assert False, "not support"
        return [matmul], [[0]]


class MulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MulParser, self).__init__(graph, var2geop)
        self.parser_name = "mul"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        x_num_col_dims = self.op.attr("x_num_col_dims")
        y_num_col_dims = self.op.attr("y_num_col_dims")
        shape_x1 = self.op.block.var(self.op.input_arg_names[0]).shape
        shape_x2 = self.op.block.var(self.op.input_arg_names[1]).shape

        if x_num_col_dims == 1 and y_num_col_dims == 1:
            if len(shape_x1) == 2 and len(shape_x2) == 2:
                matmul = core.GEOperatorFactory.create_operator(
                    "mul" + self._accumulated_op_id(),
                    "MatMul").set_input("x1", x).set_input("x2", y)
            elif len(shape_x1) == 3 and len(shape_x2) == 2:
                flatten_x1 = core.GEOperatorFactory.create_operator(
                    "flatten" + self._accumulated_op_id(),
                    "Flatten").set_input("x", x)
                matmul = core.GEOperatorFactory.create_operator(
                    "mul" + self._accumulated_op_id(), "MatMul").set_input(
                        "x1", flatten_x1, 0).set_input("x2", y, 0)
            else:
                assert False, "not support"
        else:
            if len(shape_x1) == 3 and len(shape_x2) == 2:
                assert x_num_col_dims == 2, "only support 2"
                flatten_x1 = core.GEOperatorFactory.create_operator(
                    "flatten" + self._accumulated_op_id(),
                    "FlattenV2").set_input("x", x).set_attr_int32(
                        "axis", 0).set_attr_int32("end_axis", 1)
                matmul_m = core.GEOperatorFactory.create_operator(
                    "mul" + self._accumulated_op_id(), "MatMul").set_input(
                        "x1", flatten_x1, 0).set_input("x2", y, 0)
                matmul_transpose = core.GEOperatorFactory.create_operator(
                    "transpose" + self._accumulated_op_id(),
                    "TransposeD").set_input(
                        "x", matmul_m).set_attr_vec_int32("perm", [1, 0])
                tensor = self._create_ge_tensor(
                    [3], 2, [shape_x2[1], shape_x1[0], shape_x1[1]])
                const_shape = core.GEOperatorFactory.create_operator(
                    "shape" + self._accumulated_op_id(),
                    "Const").set_attr_tensor("value", tensor)
                reshape_matmul = core.GEOperatorFactory.create_operator(
                    "reshape" + self._accumulated_op_id(), "Reshape").set_input(
                        "x", matmul_transpose).set_input(
                            "shape", const_shape).set_attr_int32("axis", 0)
                matmul = core.GEOperatorFactory.create_operator(
                    "transpose" + self._accumulated_op_id(),
                    "TransposeD").set_input(
                        "x",
                        reshape_matmul).set_attr_vec_int32("perm", [1, 2, 0])
            else:
                assert False, "not support"

        return [matmul], [[0]]


class LayerNormParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LayerNormParser, self).__init__(graph, var2geop)
        self.parser_name = "layer_norm"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[2])
        scale = self._get_ge_input(self.op.input_arg_names[1])
        bias = self._get_ge_input(self.op.input_arg_names[0])
        epsilon = self.op.attr("epsilon")
        begin_norm_axis = self.op.attr("begin_norm_axis")
        x_dtype = self.op.block.var(self.op.input_arg_names[2]).dtype

        shape_tensor = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", x)
        scale_expand = core.GEOperatorFactory.create_operator(
            "broadcast_to_d" + self._accumulated_op_id(),
            "BroadcastTo").set_input("x",
                                     scale).set_input("shape", shape_tensor)
        bias_expand = core.GEOperatorFactory.create_operator(
            "broadcast_to_d" + self._accumulated_op_id(),
            "BroadcastTo").set_input("x", bias).set_input("shape", shape_tensor)
        layer_norm = core.GEOperatorFactory.create_operator(
            "layer_norm" + self._accumulated_op_id(),
            "LayerNorm").set_input("x", x).set_input(
                "gamma",
                scale_expand).set_input("beta", bias_expand).set_attr_int32(
                    "begin_norm_axis", begin_norm_axis).set_attr_int32(
                        "begin_params_axis",
                        begin_norm_axis).set_attr_float("epsilon", epsilon)

        cast_dtype = 0 if self.ascend_helper.dtype2paddle_inv_map[str(
            x_dtype)] == 0 else 1
        y = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", layer_norm, 0).set_attr_int32("dst_type", cast_dtype)
        mean = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", layer_norm, 1).set_attr_int32("dst_type", cast_dtype)
        variance = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", layer_norm, 2).set_attr_int32("dst_type", cast_dtype)
        return [y, mean, variance], [[1], [2], [0]]


## activate function
class ReluParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReluParser, self).__init__(graph, var2geop)
        self.parser_name = "relu"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        relu = core.GEOperatorFactory.create_operator(
            "relu" + self._accumulated_op_id(), "Relu").set_input("x", x)
        return [relu], [[0]]


class GeluParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(GeluParser, self).__init__(graph, var2geop)
        self.parser_name = "gelu"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        gelu = core.GEOperatorFactory.create_operator(
            "gelu" + self._accumulated_op_id(), "Gelu").set_input("x", x)
        return [gelu], [[0]]


class TanhParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TanhParser, self).__init__(graph, var2geop)
        self.parser_name = "tanh"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        tanh = core.GEOperatorFactory.create_operator(
            "tanh" + self._accumulated_op_id(), "Tanh").set_input("x", x)
        return [tanh], [[0]]


## loss function
class SoftmaxWithCrossEntropyParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxWithCrossEntropyParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_with_cross_entropy"

    def _apply(self):
        label = self._get_ge_input(self.op.input_arg_names[0])
        logits = self._get_ge_input(self.op.input_arg_names[1])
        cls_num = self.op.block.var(self.op.input_arg_names[1]).shape[1]

        softmax = core.GEOperatorFactory.create_operator(
            "softmax" + self._accumulated_op_id(),
            "SoftmaxV2").set_input("x", logits)
        label = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", label).set_attr_int32("dst_type", 3)

        tensoron = self._create_ge_tensor([1], 5, 1)
        on = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensoron)
        tensoroff = self._create_ge_tensor([1], 5, 0)
        off = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensoroff)
        self._mark_as_input(on)
        self._mark_as_input(off)
        onehot = core.GEOperatorFactory.create_operator(
            "onehot" + self._accumulated_op_id(), "OneHotD").set_input(
                "x", label).set_input("on_value", on).set_input(
                    "off_value", off).set_attr_int32("depth", cls_num)
        squeeze = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(), "Squeeze").set_input("x", onehot)

        loss_all = core.GEOperatorFactory.create_operator(
            "loss" + self._accumulated_op_id(),
            "SoftmaxCrossEntropyWithLogits").set_input(
                "features", logits).set_input("labels", squeeze)
        loss = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", loss_all, 0).set_attr_int32("dst_type", 0)
        loss_expand = core.GEOperatorFactory.create_operator(
            "unsqueeze" + self._accumulated_op_id(),
            "Unsqueeze").set_input("x", loss).set_attr_vec_int32("axes", [1])
        return [label, softmax, loss_expand], [[2], [1]]


class SoftMaxParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftMaxParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax"

    def _apply(self):
        logits = self._get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axis")

        softmax = core.GEOperatorFactory.create_operator(
            "softmax" + self._accumulated_op_id(), "SoftmaxV2").set_input(
                "x", logits).set_attr_vec_int32("axes", [axes])
        return [softmax], [[0]]


## general 
class ShapeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ShapeParser, self).__init__(graph, var2geop)
        self.parser_name = "shape"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", x)
        return [shape], [[0]]


class FillConstantParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(FillConstantParser, self).__init__(graph, var2geop)
        self.parser_name = "fill_constant"

    def _apply(self):
        shape = self.op.attr("shape")
        dtype = self.op.attr("dtype")
        value = self.op.attr("value")

        tensor = self._create_ge_tensor(shape, dtype, value)
        const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor)
        self._mark_as_input(const)
        if self.op.block.var(self.op.output('Out')[0]).persistable:
            #print("%s is Persistable in fill_constant" %
            #      (self.op.output('Out')[0]))
            var = core.GEOperatorFactory.create_operator(
                self.op.output('Out')[0], "Variable")
            var.update_output_desc("y",
                                   core.GETensorDesc(
                                       core.GEShape(shape),
                                       core.GEFormat.FORMAT_ND,
                                       core.GEDataType.DT_FLOAT))
            assign = core.GEOperatorFactory.create_operator(
                "assign" + self._accumulated_op_id(), "Assign").set_input(
                    "value", const).set_input("ref", var)
            return [const], [[0]]
        return [const], [[0]]


class TruncatedNormalParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TruncatedNormalParser, self).__init__(graph, var2geop)
        self.parser_name = "truncated_gaussian_random"

    def _apply(self):
        shape = self.op.attr("shape")
        dtype = self.op.attr("dtype")
        mean = self.op.attr("mean")
        std = self.op.attr("std")
        seed = self.op.attr("seed")

        tensor1 = self._create_ge_tensor([len(shape)], 2, shape)
        shape_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor1)
        tensor2 = self._create_ge_tensor([1], dtype, mean)
        mean_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor2)
        tensor3 = self._create_ge_tensor([1], dtype, std)
        std_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor3)
        tensor4 = self._create_ge_tensor([1], dtype, mean - 2 * std)
        min_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor4)
        tensor5 = self._create_ge_tensor([1], dtype, mean + 2 * std)
        max_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor5)

        self._mark_as_input(shape_tensor)
        self._mark_as_input(mean_tensor)
        self._mark_as_input(std_tensor)
        self._mark_as_input(min_tensor)
        self._mark_as_input(max_tensor)

        truncated_normal = core.GEOperatorFactory.create_operator(
            "truncated_normal" + self._accumulated_op_id(),
            "ParameterizedTruncatedNormal").set_input(
                "shape", shape_tensor).set_input(
                    "means", mean_tensor).set_input(
                        "stdevs", std_tensor).set_input(
                            "min", min_tensor).set_input(
                                "max", max_tensor).set_attr_int32("seed", 0)

        ## wirte the output of truncatedNormal from startup_program to main_program
        if self.op.block.var(self.op.output('Out')[0]).persistable:
            #print("%s is Persistable in truncated_normal" %
            #      (self.op.output('Out')[0]))
            var = core.GEOperatorFactory.create_operator(
                self.op.output('Out')[0], "Variable")
            var.update_output_desc("y",
                                   core.GETensorDesc(
                                       core.GEShape(shape),
                                       core.GEFormat.FORMAT_ND,
                                       core.GEDataType.DT_FLOAT))
            assign = core.GEOperatorFactory.create_operator(
                "assign" + self._accumulated_op_id(), "Assign").set_input(
                    "value", truncated_normal).set_input("ref", var)
            return [
                shape_tensor, mean_tensor, std_tensor, min_tensor, max_tensor,
                truncated_normal
            ], [[-1]]
        #else:
        #    print(
        #        "self.op.output('Out')[0] is not persistable in truncated_noraml"
        #    )
        return [truncated_normal], [[0]]


class GatherParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(GatherParser, self).__init__(graph, var2geop)
        self.parser_name = "gather"

    def _apply(self):
        index = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])
        clo = self.op.block.var(self.op.input_arg_names[1]).shape[-1]

        gather = core.GEOperatorFactory.create_operator(
            "gather" + self._accumulated_op_id(), "Gather").set_input(
                "x", x).set_input("indices", index).set_attr_bool(
                    "validate_indices", True)
        return [gather], [[0]]


class ScatterParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ScatterParser, self).__init__(graph, var2geop)
        self.parser_name = "scatter"

    def _apply(self):
        index = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])
        updates = self._get_ge_input(self.op.input_arg_names[2])
        overwrite = self.op.attr("overwrite")
        index_shape = self.op.block.var(self.op.input_arg_names[0]).shape

        if len(index_shape) == 1:
            index = core.GEOperatorFactory.create_operator(
                "unsqueeze" + self.getid(), "Unsqueeze").set_input(
                    "x", index).set_attr_vec_int32("axes", [1])
        if not overwrite:
            scatter_value = core.GEOperatorFactory.create_operator(
                "scatter" + self._accumulated_op_id(),
                "TensorScatterAdd").set_input(
                    "x", x).set_input("indices", index).set_input("updates",
                                                                  updates)
        else:
            scatter_value = core.GEOperatorFactory.create_operator(
                "scatter" + self._accumulated_op_id(),
                "TensorScatterUpdate").set_input(
                    "x", x).set_input("indices", index).set_input("updates",
                                                                  updates)
        return [x, index, updates, scatter_value], [[-1]]


class CastParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(CastParser, self).__init__(graph, var2geop)
        self.parser_name = "cast"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        dtype = self.op.attr("out_dtype")
        cast = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", x).set_attr_int32("dst_type", dtype)
        return [cast], [[0]]


class AssignParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AssignParser, self).__init__(graph, var2geop)
        self.parser_name = "assign"

    def _apply(self):
        const = self._get_ge_input(self.op.input_arg_names[0])
        var = self._get_ge_input(self.op.input_arg_names[1])
        assign = core.GEOperatorFactory.create_operator(
            "assign" + self._accumulated_op_id(), "Assign").set_input(
                "value", const).set_input("ref", var)
        return [assign], [[0]]


class ScaleParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ScaleParser, self).__init__(graph, var2geop)
        self.parser_name = "scale"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        scale = self.op.attr("scale")
        bias = self.op.attr("bias")
        bias_after_scale = self.op.attr("bias_after_scale")

        if bias_after_scale:
            scale_value = core.GEOperatorFactory.create_operator(
                "scale" + self._accumulated_op_id(), "Power").set_input(
                    "x", x).set_attr_float("power", 1.0).set_attr_float(
                        "scale", scale).set_attr_float("shift", bias)
        else:
            x_add_bias = core.GEOperatorFactory.create_operator(
                "adds" + self._accumulated_op_id(), "Adds").set_input(
                    "x", x).set_attr_float("value", bias)
            scale_value = core.GEOperatorFactory.create_operator(
                "scale" + self._accumulated_op_id(), "Power").set_input(
                    "x",
                    x_add_bias).set_attr_float("power", 1.0).set_attr_float(
                        "scale", scale).set_attr_float("shift", 0.0)
        return [scale_value], [[0]]


class SliceParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SliceParser, self).__init__(graph, var2geop)
        self.parser_name = "slice"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axes")
        starts = self.op.attr("starts")
        ends = self.op.attr("ends")

        x_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        len_shape = len(x_shape)
        axes_cor = list(range(len_shape))
        starts_cor, ends_cor = [], []
        cnt = 0
        for i in range(len_shape):
            starts_cor.append(starts[cnt] if i in axes else 0)
            if i in axes and ends[cnt] <= x_shape[i]:
                ends_cor.append(ends[cnt])
            else:
                ends_cor.append(x_shape[i])
            if i in axes:
                cnt += 1
        size = [ends_cor[i] - starts_cor[i] for i in range(len(axes_cor))]

        assert len(axes_cor) == len(starts_cor) == len(
            ends_cor), "the three fields must have same size"
        slice_value = core.GEOperatorFactory.create_operator(
            "slice" + self._accumulated_op_id(), "SliceD").set_input(
                "x", x).set_attr_vec_int32(
                    "offsets", starts_cor).set_attr_vec_int32("size", size)

        return [slice_value], [[0]]


class ReshapeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReshapeParser, self).__init__(graph, var2geop)
        self.parser_name = "reshape2"

    def _apply(self):
        org_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        assert org_shape.count(-1) == 0, "do not allow the dim is -1"
        shape = self.op.attr("shape")
        for cnt in range(len(shape)):
            if shape[cnt] == 0:
                shape[cnt] = org_shape[cnt]

        if -1 in shape:
            assert shape.count(-1) == 1, "only allow one dim is -1"
            mul_res_org = reduce(lambda x, y: x * y, org_shape)
            mul_res_refine = reduce(lambda x, y: x * y, shape) * -1
            idx = shape.index(-1)
            shape[idx] = mul_res_org // mul_res_refine

        x = self._get_ge_input(self.op.input_arg_names[0])
        tensor = self._create_ge_tensor([len(shape)], 2, shape)
        const_shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor)
        reshape = core.GEOperatorFactory.create_operator(
            "reshape" + self._accumulated_op_id(), "Reshape").set_input(
                "x",
                x).set_input("shape", const_shape).set_attr_int32("axis", 0)
        x_shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", x)

        return [x_shape, reshape], [[1], [0]]


class TransposeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TransposeParser, self).__init__(graph, var2geop)
        self.parser_name = "transpose2"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        perm = self.op.attr("axis")
        transpose = core.GEOperatorFactory.create_operator(
            "transpose" + self._accumulated_op_id(), "TransposeD").set_input(
                "x", x).set_attr_vec_int32("perm", perm)
        x_shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", x)

        return [x_shape, transpose], [[1], [0]]


class AccuracyParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AccuracyParser, self).__init__(graph, var2geop)
        self.parser_name = "accuracy"

    def _apply(self):
        pred = self._get_ge_input(self.op.input_arg_names[0])
        label = self._get_ge_input(self.op.input_arg_names[1])
        logits = self._get_ge_input(self.op.input_arg_names[2])

        pred = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", pred).set_attr_int32("dst_type", 3)
        label = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", label).set_attr_int32("dst_type", 3)
        equal = core.GEOperatorFactory.create_operator(
            "equal" + self._accumulated_op_id(), "Equal").set_input(
                "x1", pred).set_input("x2", label)
        cast = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", equal).set_attr_int32("dst_type", 0)
        acc = core.GEOperatorFactory.create_operator(
            "mean" + self._accumulated_op_id(), "ReduceMeanD").set_input(
                "x", cast).set_attr_bool("keep_dims", False).set_attr_vec_int32(
                    "axes", [])
        correct = core.GEOperatorFactory.create_operator(
            "sum" + self._accumulated_op_id(), "ReduceSumD").set_input(
                "x", cast).set_attr_bool("keep_dims", False).set_attr_vec_int32(
                    "axes", [])
        ones_tensor = core.GEOperatorFactory.create_operator(
            "oneslike" + self._accumulated_op_id(),
            "OnesLike").set_input("x", label)
        ones_tensor = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", ones_tensor).set_attr_int32("dst_type", 0)
        total = core.GEOperatorFactory.create_operator(
            "sum" + self._accumulated_op_id(), "ReduceSumD").set_input(
                "x", ones_tensor).set_attr_bool(
                    "keep_dims", False).set_attr_vec_int32("axes", [])

        return [acc, correct, total], [[0], [1], [2]]


class TopkParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TopkParser, self).__init__(graph, var2geop)
        self.parser_name = "top_k"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        k = self.op.attr("k")

        tensor = self._create_ge_tensor([1], 2, k)
        const_k = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor)
        cast_x = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(),
            "Cast").set_input("x", x).set_attr_int32("dst_type", 1)
        topk = core.GEOperatorFactory.create_operator(
            "topk" + self._accumulated_op_id(),
            "TopK").set_input("x", cast_x).set_input("k", const_k)
        value = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", topk, 0).set_attr_int32("dst_type", 0)
        index = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", topk, 1).set_attr_int32("dst_type", 0)
        return [value, index], [[1], [0]]


class LookupTableParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LookupTableParser, self).__init__(graph, var2geop)
        self.parser_name = "lookup_table"

    def _apply(self):
        ids = self._get_ge_input(self.op.input_arg_names[0])
        w = self._get_ge_input(self.op.input_arg_names[1])

        ids_squeeze = core.GEOperatorFactory.create_operator(
            "squeeze" + self._accumulated_op_id(), "Squeeze").set_input(
                "x", ids).set_attr_vec_int32("axes", [-1])
        out = core.GEOperatorFactory.create_operator(
            "lookup" + self._accumulated_op_id(), "Gather").set_input(
                "x", w).set_input("indices", ids_squeeze)
        return [out], [[0]]


class StackParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(StackParser, self).__init__(graph, var2geop)
        self.parser_name = "stack"

    def _apply(self):
        tiles = len(self.op.input_arg_names)
        data_x_lst = []
        for index in range(tiles):
            data_x_lst.append(
                self._get_ge_input(self.op.input_arg_names[index]))
        axis = self.op.attr("axis")

        data_x = data_x_lst[0]
        tensor = self._create_ge_tensor([1], 2, axis)
        tensor_axis = core.GEOperatorFactory.create_operator(
            "axis" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor)
        expand = core.GEOperatorFactory.create_operator(
            "expand" + self._accumulated_op_id(),
            "ExpandDims").set_input("x", data_x).set_input("axis", tensor_axis)

        stack = core.GEOperatorFactory.create_operator(
            "stack" + self._accumulated_op_id(),
            "TileWithAxis").set_input("x", expand).set_attr_int32(
                "axis", axis).set_attr_int32("tiles", tiles)

        return [stack], [[0]]


class UnSqueezeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(UnSqueezeParser, self).__init__(graph, var2geop)
        self.parser_name = "unsqueeze2"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr('axes')

        output = core.GEOperatorFactory.create_operator(
            "unsqueeze" + self._accumulated_op_id(),
            "Unsqueeze").set_input("x", x).set_attr_vec_int32("axes", axes)
        shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", output)
        return [shape, output], [[1], [0]]


## parallel
class AllGatherParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AllGatherParser, self).__init__(graph, var2geop)
        self.parser_name = "c_allgather"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        rank_size = self.op.attr("rank_size")
        group = self.op.attr("group")

        allgather = core.GEOperatorFactory.create_operator(
            "allgather" + self._accumulated_op_id(), "HcomAllGather").set_input(
                "x", x).set_attr_int32(
                    "rank_size", rank_size).set_attr_string("group", group)
        return [allgather], [[0]]


class AllReduceParser(AscendParserBase):
    def __init__(self, graph, var2geop, reduction):
        super(AllReduceParser, self).__init__(graph, var2geop)
        self.parser_name = "c_allreduce_" + reduction
        self.reduction = reduction

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        reduction = self.reduction
        ring_id = self.op.attr("ring_id")
        group = "hcom_group_" + str(ring_id)
        fusion = None  #self.op.attr("fusion")
        fusion_id = None  #self.op.attr("fusion_id")

        allreduce = core.GEOperatorFactory.create_operator(
            "allreduce" + self._accumulated_op_id(), "HcomAllReduce").set_input(
                "x", x).set_attr_string(
                    "reduction", reduction).set_attr_string("group", group)
        if fusion is not None:
            allreduce.set_attr_int32("fusion", fusion)

        if fusion_id is not None:
            allreduce.set_attr_int32("fusion_id", fusion_id)
        return [allreduce], [[0]]


class AllReduceSumParser(AllReduceParser):
    def __init__(self, graph, var2geop):
        super(AllReduceSumParser, self).__init__(graph, var2geop, 'sum')


class AllReduceMaxParser(AllReduceParser):
    def __init__(self, graph, var2geop):
        super(AllReduceMaxParser, self).__init__(graph, var2geop, 'max')


class BroadcastParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(BroadcastParser, self).__init__(graph, var2geop)
        self.parser_name = "c_broadcast"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        root_rank = self.op.attr("root_rank")
        group = self.op.attr("group")

        broadcast = core.GEOperatorFactory.create_operator(
            "broadcast" + self._accumulated_op_id(), "HcomBroadcast").set_input(
                "x", x).set_attr_int32(
                    "root_rank", root_rank).set_attr_string("group", group)
        return [broadcast], [[0]]


class ReduceScatterParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceScatterParser, self).__init__(graph, var2geop)
        self.parser_name = "c_reduce_scatter"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        reduction = self.op.attr("reduction")
        group = self.op.attr("group")
        rank_size = self.op.attr("rank_size")

        reduce_scatter = core.GEOperatorFactory.create_operator(
            "reducescatter" + self._accumulated_op_id(),
            "HcomReduceScatter").set_input("x", x).set_attr_string(
                "reduction", reduction).set_attr_string(
                    "group", group).set_attr_int32("rank_size", rank_size)
        return [reduce_scatter], [[0]]


class SendParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SendParser, self).__init__(graph, var2geop)
        self.parser_name = "c_send"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        sr_tag = self.op.attr("sr_tag")
        dest_rank = self.op.attr("dest_rank")
        group = self.op.attr("group")

        send = core.GEOperatorFactory.create_operator(
            "send" + self._accumulated_op_id(), "HcomSend").set_input(
                "x", x).set_attr_int32("sr_tag", sr_tag).set_attr_int32(
                    "dest_rank", dest_rank).set_attr_string("group", group)
        return [send], [[0]]


class ReceiveParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReceiveParser, self).__init__(graph, var2geop)
        self.parser_name = "c_receive"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        sr_tag = self.op.attr("sr_tag")
        src_rank = self.op.attr("src_rank")
        group = self.op.attr("group")
        shape = self.op.attr("shape")
        dtype = self.op.attr("dtype")

        receive = core.GEOperatorFactory.create_operator(
            "receive" + self._accumulated_op_id(), "HcomReceive").set_input(
                "x", x).set_attr_int32("sr_tag", sr_tag).set_attr_int32(
                    "src_rank", src_rank).set_attr_string(
                        "group", group).set_attr_vec_int32(
                            "shape", shape).set_attr_int32("dtype", dtype)
        return [receive], [[0]]


class RangeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(RangeParser, self).__init__(graph, var2geop)
        self.parser_name = "range"

    def _apply(self):
        # TODO not support range type yet
        start = self._get_ge_input(self.op.input_arg_names[0])
        end = self._get_ge_input(self.op.input_arg_names[1])
        delta = self._get_ge_input(self.op.input_arg_names[2])

        ge_range = core.GEOperatorFactory.create_operator(
            "range" + self._accumulated_op_id(), "Range")\
              .set_input("start", end)\
              .set_input("limit", start) \
              .set_input("delta", delta)

        return [ge_range], [[0]]


class UniformRandomParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(UniformRandomParser, self).__init__(graph, var2geop)
        self.parser_name = "uniform_random"

    def _apply(self):
        shape = self.op.attr("shape")

        min_v = self.op.attr("min")
        max_v = self.op.attr("max")
        seed = self.op.attr("seed")
        dtype = self.op.attr("dtype")
        assert max_v > min_v, "assert max_v > min_v, but recieved " + \
               "as max_v={}, min_v={} ".format(max_v, min_v)

        tensor1 = self._create_ge_tensor([len(shape)], 2, shape)
        shape_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor1)

        ge_ur = core.GEOperatorFactory.create_operator(
            "uniform_random" + self._accumulated_op_id(), "RandomUniform")\
            .set_input("shape", shape_tensor)\
            .set_attr_dtype("dtype", self.ascend_helper.dtype2ge(dtype))  \
            .set_attr_int32("seed", seed)\
            .set_attr_int32("seed2", seed)

        scale = max_v - min_v

        scale_value = core.GEOperatorFactory.create_operator(
            "scale" + self._accumulated_op_id(), "Power").set_input(
                "x", ge_ur).set_attr_float("power", 1.0).set_attr_float(
                    "scale", scale).set_attr_float("shift", min_v)

        return [scale_value], [[0]]


class EqualParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(EqualParser, self).__init__(graph, var2geop)
        self.parser_name = "equal"

    def _apply(self):
        data_x1 = self._get_ge_input(self.op.input_arg_names[0])
        data_x2 = self._get_ge_input(self.op.input_arg_names[1])
        equal = core.GEOperatorFactory.create_operator("equal" \
           + self._accumulated_op_id(), "Equal")\
             .set_input("x1", data_x1)\
             .set_input("x2", data_x2)
        return [equal], [[0]]


class ExpandParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ExpandParser, self).__init__(graph, var2geop)
        self.parser_name = "expand"

    def _apply(self):
        data_x1_shape = self._get_ge_input(self.op.input_arg_names[0])
        expand_times = self.op.attr('expand_times')

        tensor = self._create_ge_tensor([len(expand_times)], 2, expand_times)
        expand_tensor = core.GEOperatorFactory.\
           create_operator("const" + self._accumulated_op_id(), "Const")\
              .set_attr_tensor("value", tensor)

        assign = core.GEOperatorFactory\
           .create_operator("tile" + self._accumulated_op_id(), "Tile")\
              .set_input("x", data_x1_shape)\
              .set_input("multiples", expand_tensor)
        return [assign], [[0]]


class SqueezeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SqueezeParser, self).__init__(graph, var2geop)
        self.parser_name = "squeeze2"

    def _apply(self):
        tensor = self._get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axes")

        data_squeezed = core.GEOperatorFactory\
           .create_operator("squeeze" + self._accumulated_op_id(), "Squeeze")\
             .set_input("x", tensor)\
             .set_attr_vec_int32("axes", axes)
        shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(),
            "Shape").set_input("x", data_squeezed)
        return [shape, data_squeezed], [[1], [0]]


#****************************************************************#
#***************************            *************************#
#***************************            *************************#
#*************************** GradParser *************************#
#***************************            *************************#
#***************************            *************************#
#****************************************************************#
## grad
class ReduceSumGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceSumGradParser, self).__init__(graph, var2geop)
        self.parser_name = "reduce_sum_grad"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        input = self._get_ge_input(self.op.input_arg_names[1])

        shape_tensor = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(),
            "Shape").set_input("x", input, 0)
        tensoron = self._create_ge_tensor([1], 2, -1)
        const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensoron)
        self._mark_as_input(const)

        reduce_sum = core.GEOperatorFactory.create_operator(
            "broadcast_to_d" + self._accumulated_op_id(),
            "BroadcastTo").set_input("x", x).set_input("shape", shape_tensor)
        #reduce_sum = core.GEOperatorFactory.create_operator("expand" + self._accumulated_op_id(), "ExpandDims").set_input("x", reduce_sum).set_input("axis", const)

        return [reduce_sum], [[0]]


class MatMulGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MatMulGradParser, self).__init__(graph, var2geop)
        self.parser_name = "matmul_grad"

    def _apply(self):
        out_grad = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])
        y = self._get_ge_input(self.op.input_arg_names[2])
        transpose_x = self.op.attr("transpose_X")
        transpose_y = self.op.attr("transpose_Y")

        out_grad_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        x_shape = self.op.block.var(self.op.input_arg_names[1]).shape
        y_shape = self.op.block.var(self.op.input_arg_names[2]).shape

        if len(x_shape) > 2:
            if transpose_y:
                x_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "BatchMatMul").set_input("x1", out_grad).set_input(
                        "x2", y).set_attr_bool(
                            "adj_x1", False).set_attr_bool("adj_x2", False)
                y_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "BatchMatMul").set_input("x1", out_grad).set_input(
                        "x2", x).set_attr_bool(
                            "adj_x1", True).set_attr_bool("adj_x2", False)
            else:
                x_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "BatchMatMul").set_input("x1", out_grad).set_input(
                        "x2", y).set_attr_bool(
                            "adj_x1", False).set_attr_bool("adj_x2", True)
                y_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "BatchMatMul").set_input("x1", x).set_input(
                        "x2", out_grad).set_attr_bool(
                            "adj_x1", True).set_attr_bool("adj_x2", False)
        else:
            if transpose_y:
                x_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input("x1", out_grad).set_input(
                        "x2", y).set_attr_bool(
                            "transpose_x1", False).set_attr_bool("transpose_x2",
                                                                 False)
                y_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input("x1", out_grad).set_input(
                        "x2", x).set_attr_bool(
                            "transpose_x1", True).set_attr_bool("transpose_x2",
                                                                False)
            else:
                x_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input("x1", out_grad).set_input(
                        "x2", y).set_attr_bool(
                            "transpose_x1", False).set_attr_bool("transpose_x2",
                                                                 True)
                y_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input("x1", x).set_input(
                        "x2", out_grad).set_attr_bool(
                            "transpose_x1", True).set_attr_bool("transpose_x2",
                                                                False)

        return [x_grad, y_grad], [[0], [1]]


class MulGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MulGradParser, self).__init__(graph, var2geop)
        self.parser_name = "mul_grad"

    def _apply(self):
        out_grad = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])
        y = self._get_ge_input(self.op.input_arg_names[2])
        x_num_col_dims = self.op.attr("x_num_col_dims")
        y_num_col_dims = self.op.attr("y_num_col_dims")

        shape_out_grad = self.op.block.var(self.op.input_arg_names[0]).shape
        shape_x = self.op.block.var(self.op.input_arg_names[1]).shape
        shape_y = self.op.block.var(self.op.input_arg_names[2]).shape

        if x_num_col_dims == 1 and y_num_col_dims == 1:
            if len(shape_x) == 2 and len(shape_y) == 2:
                x_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input("x1", out_grad).set_input(
                        "x2", y).set_attr_bool(
                            "transpose_x1", False).set_attr_bool("transpose_x2",
                                                                 True)
                y_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input("x1", x).set_input(
                        "x2", out_grad).set_attr_bool(
                            "transpose_x1", True).set_attr_bool("transpose_x2",
                                                                False)
            elif len(shape_x) == 3 and len(shape_y) == 2:
                flatten_x = core.GEOperatorFactory.create_operator(
                    "flatten" + self._accumulated_op_id(),
                    "Flatten").set_input("x", x)
                x_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input(
                        "x1", out_grad).set_input("x2", y).set_attr_bool(
                            "transpose_x1",
                            False).set_attr_bool("transpose_x2", True)
                if len(shape_out_grad) == 2:
                    x_grad = core.GEOperatorFactory.create_operator(
                        "unsqueeze" + self._accumulated_op_id(),
                        "Unsqueeze").set_input("x", x_grad).set_attr_vec_int32(
                            "axes", [1])

                y_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input(
                        "x1",
                        flatten_x).set_input("x2", out_grad).set_attr_bool(
                            "transpose_x1",
                            True).set_attr_bool("transpose_x2", False)
        else:
            if len(shape_x) == 3 and len(shape_y) == 2:
                assert x_num_col_dims == 2, "only support 2"
                flatten_x = core.GEOperatorFactory.create_operator(
                    "flatten" + self._accumulated_op_id(),
                    "FlattenV2").set_input("x", x).set_attr_int32(
                        "axis", 0).set_attr_int32("end_axis", 1)
                flatten_out_grad = core.GEOperatorFactory.create_operator(
                    "flatten" + self._accumulated_op_id(),
                    "FlattenV2").set_input("x", out_grad).set_attr_int32(
                        "axis", 0).set_attr_int32("end_axis", 1)

                y_unsqueeze = core.GEOperatorFactory.create_operator(
                    "unsqueeze" + self._accumulated_op_id(),
                    "Unsqueeze").set_input("x",
                                           y).set_attr_vec_int32("axes", [0])
                y_stack = core.GEOperatorFactory.create_operator(
                    "stack" + self._accumulated_op_id(),
                    "TileWithAxis").set_input("x", y_unsqueeze).set_attr_int32(
                        "axis", 0).set_attr_int32("tiles", shape_out_grad[0])
                x_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "BatchMatMul").set_input("x1", out_grad).set_input(
                        "x2", y_stack).set_attr_bool(
                            "adj_x1", False).set_attr_bool("adj_x2", True)
                y_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "MatMul").set_input("x1", flatten_x).set_input(
                        "x2", flatten_out_grad).set_attr_bool(
                            "transpose_x1",
                            True).set_attr_bool("transpose_x2", False)

        return [x_grad, y_grad], [[0], [1]]


class ReluGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReluGradParser, self).__init__(graph, var2geop)
        self.parser_name = "relu_grad"

    def _apply(self):
        out = self._get_ge_input(self.op.input_arg_names[0])
        out_grad = self._get_ge_input(self.op.input_arg_names[1])
        relu_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(), "ReluGrad").set_input(
                "gradients", out_grad).set_input("features", out)
        return [relu_grad], [[0]]


class SoftmaxWithCrossEntropyGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxWithCrossEntropyGradParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_with_cross_entropy_grad"

    def _apply(self):
        label = self._get_ge_input(self.op.input_arg_names[0])
        loss_grad = self._get_ge_input(self.op.input_arg_names[1])
        softmax = self._get_ge_input(self.op.input_arg_names[2])
        cls_num = self.op.block.var(self.op.input_arg_names[2]).shape[1]

        label_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        loss_grad_shape = self.op.block.var(self.op.input_arg_names[1]).shape
        softmax_shape = self.op.block.var(self.op.input_arg_names[2]).shape

        tensoron = self._create_ge_tensor([1], 5, 1)
        on = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensoron)
        tensoroff = self._create_ge_tensor([1], 5, 0)
        off = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensoroff)
        self._mark_as_input(on)
        self._mark_as_input(off)

        label = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", label).set_attr_int32("dst_type", 3)
        onehot = core.GEOperatorFactory.create_operator(
            "onehot" + self._accumulated_op_id(), "OneHotD").set_input(
                "x", label).set_input("on_value", on).set_input(
                    "off_value", off).set_attr_int32("depth", cls_num)
        squeeze = core.GEOperatorFactory.create_operator(
            "suqeeze" + self._accumulated_op_id(),
            "Squeeze").set_input("x", onehot)
        sub = core.GEOperatorFactory.create_operator(
            "sub" + self._accumulated_op_id(), "Sub").set_input(
                "x1", softmax).set_input("x2", squeeze)
        grad = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(),
            "Mul").set_input("x1", loss_grad).set_input("x2", sub)

        return [on, off, label, onehot, grad], [[-1]]


class DotMulGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotMulGradParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_mul_grad"

    def _apply(self):
        out_grad = self._get_ge_input(self.op.input_arg_names[0])
        out_1 = self._get_ge_input(self.op.input_arg_names[1])
        out_2 = self._get_ge_input(self.op.input_arg_names[2])

        x_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(),
            "Mul").set_input("x1", out_grad).set_input("x2", out_2)
        y_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(),
            "Mul").set_input("x1", out_1).set_input("x2", out_grad)

        return [x_grad, y_grad], [[0], [1]]


class DotAddGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotAddGradParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_add_grad"

    def _apply(self):
        out_grad = self._get_ge_input(self.op.input_arg_names[0])
        out_1 = self._get_ge_input(self.op.input_arg_names[1])
        out_2 = self._get_ge_input(self.op.input_arg_names[2])
        out_grad_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        out_1_shape = self.op.block.var(self.op.input_arg_names[1]).shape
        out_2_shape = self.op.block.var(self.op.input_arg_names[2]).shape

        x_grad = out_grad
        cur_time_x = len(out_grad_shape) - len(out_1_shape)
        for i in range(cur_time_x):
            x_grad = core.GEOperatorFactory.create_operator(
                self.parser_name + self._accumulated_op_id(),
                "ReduceSumD").set_input("x", x_grad).set_attr_vec_int32(
                    "axes", [0]).set_attr_bool("keep_dims", False)
        for axis, size in enumerate(out_1_shape):
            if size == 1:
                x_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "ReduceSumD").set_input("x", x_grad).set_attr_vec_int32(
                        "axes", [axis]).set_attr_bool("keep_dims", True)

        y_grad = out_grad
        cur_time_y = len(out_grad_shape) - len(out_2_shape)
        for i in range(cur_time_y):
            y_grad = core.GEOperatorFactory.create_operator(
                self.parser_name + self._accumulated_op_id(),
                "ReduceSumD").set_input("x", y_grad).set_attr_vec_int32(
                    "axes", [0]).set_attr_bool("keep_dims", False)
        for axis, size in enumerate(out_2_shape):
            if size == 1:
                y_grad = core.GEOperatorFactory.create_operator(
                    self.parser_name + self._accumulated_op_id(),
                    "ReduceSumD").set_input("x", y_grad).set_attr_vec_int32(
                        "axes", [axis]).set_attr_bool("keep_dims", True)

        return [x_grad, y_grad], [[0], [1]]


class DotDivGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotDivGradParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_div_grad"

    def _apply(self):
        out = self._get_ge_input(self.op.input_arg_names[0])
        out_grad = self._get_ge_input(self.op.input_arg_names[1])
        x = self._get_ge_input(self.op.input_arg_names[2])
        y = self._get_ge_input(self.op.input_arg_names[3])

        y_power = core.GEOperatorFactory.create_operator(
            "power" + self._accumulated_op_id(), "Power").set_input(
                "x", y).set_attr_float("power", -1)

        tensor_zeros = core.GEOperatorFactory.create_operator(
            "zeroslike" + self._accumulated_op_id(),
            "ZerosLike").set_input("x", x)
        x_zero = core.GEOperatorFactory.create_operator(
            "equal" + self._accumulated_op_id(), "Equal").set_input(
                "x1", x).set_input("x2", tensor_zeros)
        x_nozero = core.GEOperatorFactory.create_operator(
            "logical_not" + self._accumulated_op_id(),
            "LogicalNot").set_input("x", x_zero)
        x_nozero_f = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", x_nozero).set_attr_int32("dst_type", 0)
        x_grad_w = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(), "Mul").set_input(
                "x1", x_nozero_f).set_input("x2", y_power)
        x_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(),
            "Mul").set_input("x1", x_grad_w).set_input("x2", out_grad)

        y_grad_w = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(), "Mul").set_input(
                "x1", out).set_input("x2", y_power)
        y_grad = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(), "Mul").set_input(
                "x1", y_grad_w).set_input("x2", out_grad)

        return [x_grad, y_grad], [[0], [1]]


class SoftmaxGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxGradParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_grad"

    def _apply(self):
        out = self._get_ge_input(self.op.input_arg_names[0])
        out_grad = self._get_ge_input(self.op.input_arg_names[1])

        x_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(),
            "SoftmaxGrad").set_input("softmax", out).set_input("grad_softmax",
                                                               out_grad)
        return [x_grad], [[0]]


class ReshapeGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReshapeGradParser, self).__init__(graph, var2geop)
        self.parser_name = "reshape2_grad"

    def _apply(self):
        out_grad = self._get_ge_input(self.op.input_arg_names[0])
        x_shape = self._get_ge_input(self.op.input_arg_names[1])
        x_shape_list = self.op.block.var(self.op.input_arg_names[1]).shape

        if x_shape_list[0] == 0:
            x_shape_delzero = x_shape_list[1:]
        tensor = self._create_ge_tensor([len(x_shape_delzero)], 2,
                                        x_shape_delzero)
        const_shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", tensor)
        x_grad = core.GEOperatorFactory.create_operator(
            "reshape" + self._accumulated_op_id(), "Reshape").set_input(
                "x", out_grad).set_input("shape", const_shape)

        return [x_grad], [[0]]


class GatherGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(GatherGradParser, self).__init__(graph, var2geop)
        self.parser_name = "gather_grad"

    def _apply(self):
        index = self._get_ge_input(self.op.input_arg_names[0])
        out_grad = self._get_ge_input(self.op.input_arg_names[1])
        x = self._get_ge_input(self.op.input_arg_names[2])

        index_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        out_grad_shape = self.op.block.var(self.op.input_arg_names[1]).shape
        x_shape = self.op.block.var(self.op.input_arg_names[2]).shape

        if len(index_shape) == 1:
            index = core.GEOperatorFactory.create_operator(
                "unsqueeze" + self._accumulated_op_id(), "Unsqueeze").set_input(
                    "x", index).set_attr_vec_int32("axes", [1])

        tensor_zeros = core.GEOperatorFactory.create_operator(
            "zeroslike" + self._accumulated_op_id(),
            "ZerosLike").set_input("x", x)
        x_grad = core.GEOperatorFactory.create_operator(
            "scatter" + self._accumulated_op_id(),
            "TensorScatterUpdate").set_input("x", tensor_zeros).set_input(
                "indices", index).set_input("updates", out_grad)

        return [tensor_zeros, x_grad], [[-1]]


class TransposeGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TransposeGradParser, self).__init__(graph, var2geop)
        self.parser_name = "transpose2_grad"

    def _apply(self):
        out_grad = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])
        perm = self.op.attr("axis")

        x_shape = self.op.block.var(self.op.input_arg_names[1]).shape[1:]
        out_grad_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        assert list(map(lambda x: out_grad_shape[x], perm)) == list(x_shape)

        x_grad = core.GEOperatorFactory.create_operator(
            "transpose" + self._accumulated_op_id(), "TransposeD").set_input(
                "x", out_grad).set_attr_vec_int32("perm", perm)

        return [x_grad], [[0]]


class LayerNormGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LayerNormGradParser, self).__init__(graph, var2geop)
        self.parser_name = "layer_norm_grad"

    def _apply(self):
        bias = self._get_ge_input(self.op.input_arg_names[0])
        mean = self._get_ge_input(self.op.input_arg_names[1])
        scale = self._get_ge_input(self.op.input_arg_names[2])
        variance = self._get_ge_input(self.op.input_arg_names[3])
        x = self._get_ge_input(self.op.input_arg_names[4])
        out_grad = self._get_ge_input(self.op.input_arg_names[5])
        x_dtype = self.op.block.var(self.op.input_arg_names[4]).dtype

        x_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(),
            "LayerNormGrad").set_input("dy", out_grad).set_input(
                "x", x).set_input("variance", variance).set_input(
                    "mean", mean).set_input("gamma", scale)

        cast_dtype = 0 if self.ascend_helper.dtype2paddle_inv_map[str(
            x_dtype)] == 0 else 1
        out_x_grad = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", x_grad, 0).set_attr_int32("dst_type", cast_dtype)
        out_scale_grad = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", x_grad, 1).set_attr_int32("dst_type", cast_dtype)
        out_bias_grad = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", x_grad, 2).set_attr_int32("dst_type", cast_dtype)

        return [out_x_grad, out_scale_grad, out_bias_grad], [[2], [1], [0]]


class TanhGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TanhGradParser, self).__init__(graph, var2geop)
        self.parser_name = 'tanh_grad'

    def _apply(self):
        y = self._get_ge_input(self.op.input_arg_names[0])
        out_grad = self._get_ge_input(self.op.input_arg_names[1])
        tanh_grad = core.GEOperatorFactory.create_operator(
            "tanh_grad" + self._accumulated_op_id(),
            "TanhGrad").set_input("y", y).set_input("dy", out_grad)

        return [tanh_grad], [[0]]


class LogGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LogGradParser, self).__init__(graph, var2geop)
        self.parser_name = 'log_grad'

    def _apply(self):
        grad = self._get_ge_input(self.op.input_arg_names[0])
        input = self._get_ge_input(self.op.input_arg_names[1])
        log_grad = core.GEOperatorFactory.create_operator(
            "log_grad" + self._accumulated_op_id(),
            "DivNoNan").set_input("x1", grad).set_input("x2", input)
        return [log_grad], [[0]]


class SqrtGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SqrtGradParser, self).__init__(graph, var2geop)
        self.parser_name = "sqrt_grad"

    def _apply(self):
        y = self._get_ge_input(self.op.input_arg_names[0])
        out_grad = self._get_ge_input(self.op.input_arg_names[1])
        sqrt_grad = core.GEOperatorFactory.create_operator(
            "sqrt_grad" + self._accumulated_op_id(),
            "SqrtGrad").set_input("y", y).set_input("dy", out_grad)
        return [sqrt_grad]


class PowGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(PowGradParser, self).__init__(graph, var2geop)
        self.parser_name = "pow_grad"

    def _apply(self):
        grad = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])
        factor = self.op.attr("factor")

        shape_tensor = self._create_shape_tensor()
        shape_tensor = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", x)
        factor_scale = self._create_ge_tensor([1], 5, factor)
        factor_scale = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(),
            "Const").set_attr_tensor("value", factor_scale)
        factor_tensor = core.GEOperatorFactory.create_operator(
            "broadcast_to_d" + self._accumulated_op_id(),
            "BroadcastTo").set_input(
                "x", factor_scale).set_input("shape", shape_tensor)

        x_power = core.GEOperatorFactory.create_operator(
            "x_power" + self._accumulated_op_id(), "Power").set_input(
                "x", x).set_attr_float("power", factor - 1)
        x_power_mul_factor = core.GEOperatorFactory.create_operator(
            "x_power_mul_factor" + self._accumulated_op_id(), "Mul").set_input(
                "x1", x).set_input("x2", factor_tensor)
        x_power_mul_factor_grad = core.GEOperatorFactory.create_operator(
            "x_power_mul_factor_grad" + self._accumulated_op_id(),
            "Mul").set_input("x1", x_power_mul_factor).set_input("x2", grad)

        return [x_power_mul_factor_grad], [[0]]


class GeluGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(GeluGradParser, self).__init__(graph, var2geop)
        self.parser_name = "gelu_grad"

    def _apply(self):
        grad = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])

        y = core.GEOperatorFactory.create_operator(
            "gelu" + self._accumulated_op_id(), "Gelu").set_input("x", x)
        gelu_grad = core.GEOperatorFactory.create_operator(
            "gelu_grad" + self._accumulated_op_id(), "GeluGrad").set_input(
                "x", x).set_input("dy", grad).set_input("y", y)

        return [gelu_grad], [[0]]


class MeanGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MeanGradParser, self).__init__(graph, var2geop)
        self.parser_name = "mean_grad"

    def _apply(self):
        grad = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])

        ones_tensor = core.GEOperatorFactory.create_operator(
            "one_tensor" + self._accumulated_op_id(),
            "OnesLike").set_input("x", x)
        sum = core.GEOperatorFactory.create_operator(
            "mean" + self._accumulated_op_id(), "ReduceSumD").set_input(
                "x", ones_tensor).set_attr_bool(
                    "keep_dims", False).set_attr_vec_int32("axes", [])
        mean = core.GEOperatorFactory.create_operator(
            "x_power" + self._accumulated_op_id(), "Power").set_input(
                "x", sum).set_attr_float("power", -1)

        mean_grad = core.GEOperatorFactory.create_operator(
            "mean_grad" + self._accumulated_op_id(),
            "Mul").set_input("x1", mean).set_input("x2", grad)

        return [mean_grad], [[0]]


class SliceGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SliceGradParser, self).__init__(graph, var2geop)
        self.parser_name = "slice_grad"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        grad = self._get_ge_input(self.op.input_arg_names[1])
        axes = self.op.attr("axes")
        starts = self.op.attr("starts")
        ends = self.op.attr("ends")

        x_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        grad_shape = self.op.block.var(self.op.input_arg_names[1]).shape

        len_shape = len(x_shape)
        axes_cor = list(range(len_shape))
        starts_cor, ends_cor = [], []
        cnt = 0
        for i in range(len_shape):
            starts_cor.append(starts[cnt] if i in axes else 0)
            if i in axes and ends[cnt] <= x_shape[i]:
                ends_cor.append(x_shape[i] - ends[cnt])
            else:
                ends_cor.append(0)
            if i in axes:
                cnt += 1

        starts_cor[0] = 0
        ends_cor[0] = 0
        paddings = [[s, e] for (s, e) in zip(starts_cor, ends_cor)]
        slice_value = core.GEOperatorFactory.create_operator(
            "slice_grad" + self._accumulated_op_id(), "PadD").set_input(
                "x", grad).set_attr_vec_vec_int64("paddings", paddings)

        return [slice_value], [[0]]


class LookUpTableGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LookUpTableGradParser, self).__init__(graph, var2geop)
        self.parser_name = "lookup_table_grad"

    def _apply(self):
        ids = self._get_ge_input(self.op.input_arg_names[0])
        grad = self._get_ge_input(self.op.input_arg_names[1])
        embedding = self._get_ge_input(self.op.input_arg_names[2])

        shape_ids = self.op.block.var(self.op.input_arg_names[0]).shape
        shape_grad = self.op.block.var(self.op.input_arg_names[1]).shape
        shape_embedding = self.op.block.var(self.op.input_arg_names[2]).shape

        ids_flatten = core.GEOperatorFactory.create_operator(
            "flatten" + self._accumulated_op_id(), "FlattenV2").set_input(
                "x",
                ids).set_attr_int32("axis", 0).set_attr_int32("end_axis", 1)
        grad_flatten = core.GEOperatorFactory.create_operator(
            "flatten" + self._accumulated_op_id(), "FlattenV2").set_input(
                "x",
                grad).set_attr_int32("axis", 0).set_attr_int32("end_axis", 1)

        tensor_zeros = core.GEOperatorFactory.create_operator(
            "zeroslike" + self._accumulated_op_id(),
            "ZerosLike").set_input("x", embedding)
        embedding_grad = core.GEOperatorFactory.create_operator(
            "scatteradd" + self._accumulated_op_id(),
            "TensorScatterAdd").set_input(
                "x", tensor_zeros).set_input("indices", ids_flatten).set_input(
                    "updates", grad_flatten)

        return [embedding_grad], [[0]]


class SGDParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SGDParser, self).__init__(graph, var2geop)
        self.parser_name = "sgd"

    def _apply(self):
        grad = self._get_ge_input(self.op.input_arg_names[0])
        lr = self._get_ge_input(self.op.input_arg_names[1])
        param = self._get_ge_input(self.op.input_arg_names[2])
        sgd = core.GEOperatorFactory.create_operator(
            "momentum" + self._accumulated_op_id(),
            "ApplyGradientDescent").set_input("var", param).set_input(
                "alpha", lr).set_input("delta", grad)
        return [sgd], [[0]]


class AdamParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AdamParser, self).__init__(graph, var2geop)
        self.parser_name = "adam"

    def _apply(self):
        beta1_power = self._get_ge_input(self.op.input_arg_names[0])
        beta2_power = self._get_ge_input(self.op.input_arg_names[1])
        grad = self._get_ge_input(self.op.input_arg_names[2])
        lr = self._get_ge_input(self.op.input_arg_names[3])
        moment1 = self._get_ge_input(self.op.input_arg_names[4])
        moment2 = self._get_ge_input(self.op.input_arg_names[5])
        param = self._get_ge_input(self.op.input_arg_names[6])
        beta1 = self.op.attr('beta1')
        beta2 = self.op.attr('beta2')
        epsilon = self.op.attr('epsilon')

        beta1 = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", self._create_ge_tensor([1], 5, beta1))
        beta2 = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", self._create_ge_tensor([1], 5, beta2))
        epsilon = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", self._create_ge_tensor([1], 5, epsilon))

        adam = core.GEOperatorFactory.create_operator(
            "adam" + self._accumulated_op_id(),
            "ApplyAdam").set_input("var", param).set_input(
                "m", moment1).set_input("v", moment2).set_input(
                    "beta1_power", beta1_power).set_input(
                        "beta2_power", beta2_power).set_input(
                            "lr", lr).set_input("beta1", beta1).set_input(
                                "beta2", beta2).set_input(
                                    "epsilon", epsilon).set_input("grad", grad)

        return [adam], [[0]]
