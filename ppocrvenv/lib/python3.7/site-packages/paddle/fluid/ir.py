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

import copy
import inspect
from os import path
import paddle
from . import core, unique_name
from .framework import _apply_pass, OpProtoHolder

try:
    from .proto import pass_desc_pb2
except ModuleNotFoundError:
    import sys
    sys.path.append(path.join(path.dirname(__file__), 'proto'))
    from .proto import pass_desc_pb2


def get_data_vars(program):
    data_vars = []
    for var_name, var in program.global_block().vars.items():
        if var.is_data:
            data_vars.append(var_name)
    return data_vars


def _update_grad_persistable(main_program):
    grad_merge_attr_name = "grad_merge_cond_name"
    op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
    has_grad_merge = False
    has_persistable_grad_var = False
    grad_vars = []
    for block_id in range(main_program.num_blocks):
        block = main_program.block(block_id)
        for op in block.ops:
            if grad_merge_attr_name in op.attr_names:
                has_grad_merge = True

            if op_role_var_attr_name not in op.attr_names:
                continue

            p_g = op.attr(op_role_var_attr_name)
            for g in p_g[1::2]:
                g_var = block._find_var_recursive(g)
                if g_var is None:
                    continue
                grad_vars.append(g_var)
                if g_var.persistable:
                    has_persistable_grad_var = True

    if has_grad_merge and has_persistable_grad_var:
        for g_var in grad_vars:
            g_var.persistable = True


def apply_build_strategy(main_program, startup_program, build_strategy,
                         pass_attrs):
    def update_attr(attrs, attr_types, name, value, typ=None):
        if name not in attrs:
            attrs[name] = value
        if typ:
            attr_types[name] = typ

    def apply_pass(name):
        attrs = dict(pass_attrs)
        attr_types = {}
        update_attr(attrs, attr_types, "nranks", 1, "size_t")
        update_attr(attrs, attr_types, "use_cuda", False, "bool")
        # TODO(zjl): how to skip fetch variables ?
        update_attr(attrs, attr_types, "mem_opt_skip_vars",
                    get_data_vars(main_program), "list[str]")
        _apply_pass(main_program, startup_program, name, attrs, attr_types)

    _update_grad_persistable(main_program)
    use_cuda = pass_attrs.get("use_cuda", False)
    build_strategy = build_strategy._copy()
    if build_strategy.sync_batch_norm:
        apply_pass("sync_batch_norm_pass")
        build_strategy.sync_batch_norm = False
    if build_strategy.fuse_relu_depthwise_conv and use_cuda:
        apply_pass("fuse_relu_depthwise_conv_pass")
        build_strategy.fuse_relu_depthwise_conv = False
    if build_strategy.fuse_bn_act_ops and use_cuda:
        apply_pass("fuse_bn_act_pass")
        build_strategy.fuse_bn_act_ops = False
    if build_strategy.fuse_bn_add_act_ops and use_cuda:
        apply_pass("fuse_bn_add_act_pass")
        build_strategy.fuse_bn_add_act_ops = False
    if build_strategy.enable_auto_fusion and use_cuda:
        apply_pass("fusion_group_pass")
        build_strategy.enable_auto_fusion = False
    if build_strategy.fuse_elewise_add_act_ops:
        apply_pass("fuse_elewise_add_act_pass")
        build_strategy.fuse_elewise_add_act_ops = False
    if build_strategy.fuse_all_optimizer_ops:
        apply_pass([
            "coalesce_grad_tensor_pass",
            "fuse_adam_op_pass",
            "fuse_sgd_op_pass",
            "fuse_momentum_op_pass",
        ])
        build_strategy.fuse_all_optimizer_ops = False
    # TODO(zjl): support fuse all reduce ops
    if build_strategy.cache_runtime_context:
        apply_pass("runtime_context_cache_pass")
        build_strategy.cache_runtime_context = False
    if build_strategy.enable_addto and use_cuda:
        # NOTE: how to get fetch vars to skip memory optimization?  
        apply_pass("inplace_addto_op_pass")
        build_strategy.enable_addto = False
    if build_strategy.enable_inplace:
        apply_pass("buffer_shared_inplace_pass")
        build_strategy.enable_inplace = False
    build_strategy._clear_finalized()
    return build_strategy


class RegisterPassHelper(object):
    _register_helpers = list()

    def __init__(self, pass_pairs, pass_type=str(), input_specs=dict()):
        self._pass_type = pass_type
        self._pass_pairs = pass_pairs
        self._input_specs = input_specs
        RegisterPassHelper._register_helpers.append(self)

    def _get_args_from_func(self, func):
        args = list()
        arg_specs = inspect.getfullargspec(func)
        for arg_name in arg_specs.args:
            input_spec = self._input_specs.get(arg_name)
            if isinstance(input_spec, paddle.static.InputSpec):
                args.append(
                    paddle.static.data(arg_name, input_spec.shape,
                                       input_spec.dtype))
            elif isinstance(input_spec, paddle.ParamAttr):
                args.append(paddle.ParamAttr(arg_name))
            else:
                args.append(paddle.static.data(arg_name, [-1]))
        return args

    def _prune_program_desc(self, program_desc):
        block_desc = program_desc.blocks[0]
        # block_desc.ClearField("vars")
        for var in [
                var for var in block_desc.vars
                if var.name not in self._input_specs
        ]:
            block_desc.vars.remove(var)
        for op_desc in block_desc.ops:
            default_attrs = core.get_op_attrs_default_value(
                paddle.compat.to_bytes(op_desc.type))
            remove_attrs = list()
            for attr in op_desc.attrs:
                # attr must not in 
                if attr.name not in [
                        "op_namescope", "op_callstack", "op_device"
                ]:
                    attr_list_fields = attr.ListFields()
                    # attr format must be: name, type, value
                    if len(attr_list_fields) == 3:
                        attr_value = attr.ListFields()[-1][-1]
                        default_attr_value = default_attrs.get(attr.name)
                        # value must not default
                        if default_attr_value != attr_value:
                            continue
                remove_attrs.append(attr)
            for attr in remove_attrs:
                op_desc.attrs.remove(attr)

    def _func_to_program_desc(self, func, program_desc, is_replace=False):
        vars = list()
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            args = self._get_args_from_func(func)
            for arg in args:
                vars.append(arg.name)
            outs = func(*args)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            for out in outs:
                if isinstance(out, PassDesc.OpHelper):
                    for out in out.Outputs().values():
                        vars.extend(out)
                elif isinstance(out, paddle.fluid.framework.Variable):
                    vars.append(out.name)
        program_desc.ParseFromString(program.desc.serialize_to_string())
        self._prune_program_desc(program_desc)
        if is_replace:
            attrs = list()
            for op in program.current_block().ops:
                if not isinstance(op, PassDesc.OpHelper):
                    continue
                attrs.extend(op._attrs.values())
            return vars, attrs
        return vars

    def SerializeMultiPassDesc(self):
        switch_static_mode = paddle.in_dynamic_mode()
        if switch_static_mode:
            paddle.enable_static()
        multi_pass_desc = pass_desc_pb2.MultiPassDesc()
        multi_pass_desc.pass_type = self._pass_type
        for (pattern, replace) in self._pass_pairs:
            pass_desc = multi_pass_desc.pass_descs.add()
            pattern_vars = self._func_to_program_desc(pattern,
                                                      pass_desc.pattern)
            replace_vars, attrs = self._func_to_program_desc(
                replace, pass_desc.replace, is_replace=True)
            for (pattern_var, replace_var) in zip(pattern_vars, replace_vars):
                var_map = pass_desc.var_maps.add()
                var_map.pattern_var = pattern_var
                var_map.replace_var = replace_var
            pattern_op_idxs = dict()
            for (idx, op) in enumerate(pass_desc.pattern.blocks[0].ops):
                op_idxs = pattern_op_idxs.get(op.type)
                if op_idxs:
                    op_idxs.append(idx)
                else:
                    pattern_op_idxs[op.type] = [idx]
            for attr in attrs:
                attr_map = pass_desc.attr_maps.add()
                attr_map.pattern_op_idx = pattern_op_idxs[
                    attr._pattern_op_type][attr._pattern_op_idx]
                attr_map.replace_op_idx = attr._replace_op_idx
                attr_map.pattern_name = attr._pattern_name
                attr_map.replace_name = attr._replace_name
        if switch_static_mode:
            paddle.disable_static()
        return multi_pass_desc.SerializeToString()


class PassDesc(object):
    class AttrHelper(object):
        def __init__(self, name, replace_op_idx):
            self._pattern_op_type = None
            self._pattern_op_idx = -1
            self._replace_op_idx = replace_op_idx
            self._pattern_name = name
            self._replace_name = name

        def ReusePattern(self, op, index=0, name=None):
            if name:
                self._pattern_name = name
            self._pattern_op_type = op
            self._pattern_op_idx = index

    class OpHelper(object):
        def __init__(self, type=None):
            self._type = type

        def __getattr__(self, name):
            op = PassDesc.OpHelper(name)
            op.Init()
            return op

        def __call__(self, *args, **kwargs):
            for (in_name, in_args) in kwargs.items():
                in_arg_names = list()
                if isinstance(in_args, (list, tuple)):
                    if len(in_args) == 0:
                        raise ValueError(
                            "Input '{}' of operator '{}' cannot be empty.".
                            format(in_name, self._type))
                else:
                    in_args = [in_args]
                for in_arg in in_args:
                    if isinstance(in_arg, PassDesc.OpHelper):
                        in_arg_names.extend(in_arg.Output())
                    else:
                        in_arg_names.append(in_arg.name)
                self._op_desc.set_input(in_name, in_arg_names)
            return self

        def Init(self):
            block = paddle.static.default_main_program().current_block()
            self._attrs = dict()
            self._op_idx = len(block.ops)
            self._op_desc = block.desc.append_op()
            self._op_desc.set_type(self._type)
            self._op_proto = OpProtoHolder.instance().op_proto_map.get(
                self._type)
            if self._op_proto is None:
                raise AttributeError(
                    "type object 'OpHelper' has no attribute '{}'".format(
                        self._type))
            block.ops.append(self)

        def Attr(self, name):
            attr = self._attrs.get(name)
            if attr:
                return attr
            attr = PassDesc.AttrHelper(name, self._op_idx)
            self._attrs[name] = attr
            return attr

        def SetAttr(self, name, value):
            self._op_desc._set_attr(name, value)

        def Output(self, name=None):
            if name:
                return self.Outputs()[name]
            return list(self.Outputs().values())[0]

        def Outputs(self):
            outputs = self._op_desc.outputs()
            if len(outputs) > 0:
                return outputs
            block = paddle.static.default_main_program().current_block()
            for output_proto in self._op_proto.outputs:
                name = unique_name.generate(self._type)
                block.create_var(name=name)
                self._op_desc.set_output(output_proto.name, [name])
            return self._op_desc.outputs()

    OP = OpHelper()


def RegisterPass(function=None, input_specs=dict()):
    """
    The function decorator of Register Pass. Decorator @RegisterPass handles
    the function and register it into a core.Pass instance. Use name of function
    as Pass type.

    Args:
        function (callable): The function with return of callable pair(s) that
            represents the pattern subgraph and the replace subgraph.
        input_specs (dict[str, InputSpec]): Dict of InputSpec to specific the shape/dtype
            information of Tensor. Some operators limit the shape and dtype of datas when
            create subgraph with Paddle APIs. So user need specify InputSpec of data to
            ensure create a correctly subgraph. Of course, this argument is not limited to
            matching subgraph. The default is dict().

    Returns:
        callables: Callable pair(s).

    Examples:
        .. code-block:: python

        import paddle
        from paddle.fluid.ir import RegisterPass

        @RegisterPass
        def multi_add_to_addn():
            def pattern(x, y, z):
                return paddle.add(paddle.add(x, y), z)
            def replace(x, y, z):
                return paddle.add_n([x, y, z])
            return pattern, replace
    """

    def _is_pass_pair(check_pair):
        if isinstance(check_pair, (list, tuple)):
            if len(check_pair) == 2:
                if all(map(inspect.isfunction, check_pair)):
                    return True
        return False

    def decorated(python_func):
        pass_type = python_func.__name__
        signature = inspect.signature(python_func)
        if len(signature.parameters) > 0:
            raise NotImplementedError(
                "Pass function with parameter is not supported now.")
        elif len(signature.parameters) == 0:
            pass_pairs = python_func()
            if _is_pass_pair(pass_pairs):
                pass_pairs = [pass_pairs]
            elif not all(map(_is_pass_pair, pass_pairs)):
                raise ValueError(
                    "Return value of Pass function must be (callable, callable)."
                )
            helper = RegisterPassHelper(pass_pairs, pass_type, input_specs)
            core.register_pass(pass_type, helper.SerializeMultiPassDesc)
        return python_func

    if inspect.isfunction(function):
        return decorated(function)

    return decorated
