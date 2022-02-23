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

import ast
import astor
import atexit
import copy
import collections
from paddle.utils import gast
import inspect
import os
import six
import tempfile
import textwrap
import numpy as np

import paddle
from paddle.fluid import unique_name
from paddle.fluid.data_feeder import convert_dtype

# Note(Aurelius): Do not forget the dot `.` to distinguish other
# module such as paddlenlp.
PADDLE_MODULE_PREFIX = 'paddle.'
DYGRAPH_MODULE_PREFIX = 'paddle.fluid.dygraph'
DYGRAPH_TO_STATIC_MODULE_PREFIX = 'paddle.fluid.dygraph.dygraph_to_static'


class BaseNodeVisitor(gast.NodeVisitor):
    """
    Implement customized NodeVisitor inherited from gast.NodeVisitor. 
    Ancestor nodes are traced to easily support more operations of currently
    visited node.
    """

    def __init__(self):
        self.ancestor_nodes = []

    def visit(self, node):
        """Visit a node."""
        self.ancestor_nodes.append(node)

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()
        return ret


# imp is deprecated in python3
from importlib.machinery import SourceFileLoader

dygraph_class_to_static_api = {
    "CosineDecay": "cosine_decay",
    "ExponentialDecay": "exponential_decay",
    "InverseTimeDecay": "inverse_time_decay",
    "NaturalExpDecay": "natural_exp_decay",
    "NoamDecay": "noam_decay",
    "PiecewiseDecay": "piecewise_decay",
    "PolynomialDecay": "polynomial_decay",
}

FOR_ITER_INDEX_PREFIX = '__for_loop_var_index'
FOR_ITER_TUPLE_PREFIX = '__for_loop_iter_tuple'
FOR_ITER_TUPLE_INDEX_PREFIX = '__for_loop_iter_tuple_index'
FOR_ITER_VAR_LEN_PREFIX = '__for_loop_var_len'
FOR_ITER_VAR_NAME_PREFIX = '__for_loop_iter_var'
FOR_ITER_ZIP_TO_LIST_PREFIX = '__for_loop_iter_zip'

# FullArgSpec is valid from Python3. Defined a Namedtuple to
# to make it available in Python2.
FullArgSpec = collections.namedtuple('FullArgSpec', [
    'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
    'annotations'
])


def getfullargspec(target):
    if hasattr(inspect, "getfullargspec"):
        return inspect.getfullargspec(target)
    else:
        argspec = inspect.getargspec(target)
        return FullArgSpec(
            args=argspec.args,
            varargs=argspec.varargs,
            varkw=argspec.keywords,
            defaults=argspec.defaults,
            kwonlyargs=[],
            kwonlydefaults=None,
            annotations={})


def parse_arg_and_kwargs(function):
    """
    Returns full argument names as list. e.g ['x', 'y', 'z']
    """
    fullargspec = getfullargspec(function)
    arg_names = fullargspec.args
    if arg_names and 'self' == arg_names[0]:
        arg_names = fullargspec.args[1:]

    # parse default kwargs
    default_kwargs = {}
    default_values = fullargspec.defaults
    if default_values:
        assert len(default_values) <= len(arg_names)
        default_kwarg_names = arg_names[-len(default_values):]
        default_kwargs = dict(zip(default_kwarg_names, default_values))

    return arg_names, default_kwargs


def parse_varargs_name(function):
    """
    Returns varargs name string of function. e.g: 'input' from `foo(x, *input)`
    """
    fullargspec = getfullargspec(function)
    varargs = fullargspec.varargs
    return varargs


def type_name(v):
    return type(v).__name__


def make_hashable(x, error_msg=None):
    """
    Makes input `x` hashable.

    For some unhashable objects, such as `dict/list/set/np.ndarray`,applying hash function by using their values.
    """
    if isinstance(x, (tuple, list, set)):
        return tuple(map(make_hashable, x))

    try:
        hash(x)
    except TypeError:
        if isinstance(x, np.ndarray):
            # Note: `tostring()` will return the binary data from np.ndarray that
            # means different value will lead to different hash code.
            return hash(x.tostring())
        elif isinstance(x, dict):
            return tuple(map(make_hashable, x.values()))

        error_msg = error_msg or "Requires a hashable object."
        raise ValueError(error_msg + " But received type: %s" % type_name(x))

    return x


def _is_api_in_module_helper(obj, module_prefix):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)


def is_api_in_module(node, module_prefix):
    assert isinstance(node, gast.Call), "Input non-Call node for is_dygraph_api"

    # Python can have gast.Call as function, for example: covert_call(func)(x)
    # We only check the most outside function
    func_node = node.func
    while isinstance(func_node, gast.Call):
        func_node = func_node.func

    func_str = astor.to_source(gast.gast_to_ast(func_node)).strip()
    try:
        # TODO(liym27):
        #  Consider a better to import modules like:
        #  source_file = inspect.getfile(dyfunc)
        #  import_statements = ImportVisitor(source_file).transform()
        #  import_str = "".join(import_statements)
        import paddle
        import paddle.fluid as fluid
        import paddle.fluid.dygraph as dygraph
        import paddle.fluid.layers as layers

        from paddle.fluid.dygraph import to_variable
        from paddle import to_tensor

        return eval("_is_api_in_module_helper({}, '{}')".format(func_str,
                                                                module_prefix))
    except NameError:
        return False


def is_dygraph_api(node):

    # Note: A api in module dygraph_to_static is not a real dygraph api.
    if is_api_in_module(node, DYGRAPH_TO_STATIC_MODULE_PREFIX):
        return False

    # TODO(liym27): A better way to determine whether it is a dygraph api.
    #  Consider the decorator @dygraph_only
    return is_api_in_module(node, DYGRAPH_MODULE_PREFIX)


def is_paddle_api(node):
    return is_api_in_module(node, PADDLE_MODULE_PREFIX)


def is_paddle_func(func):
    m = inspect.getmodule(func)
    return m is not None and m.__name__.startswith(PADDLE_MODULE_PREFIX)


# Is numpy_api cannot reuse is_api_in_module because of numpy module problem
def is_numpy_api(node):
    assert isinstance(node, gast.Call), "Input non-Call node for is_numpy_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        import numpy as np
        module_result = eval("_is_api_in_module_helper({}, '{}')".format(
            func_str, "numpy"))
        # BUG: np.random.uniform doesn't have module and cannot be analyzed
        # TODO: find a better way
        if not module_result:
            return func_str.startswith("numpy.") or func_str.startswith("np.")
    except NameError:
        return False


def is_control_flow_to_transform(node,
                                 static_analysis_visitor=None,
                                 var_name_to_type=None):
    """
    Determines whether the node is a PaddlePaddle control flow statement which needs to
    be transformed into a static graph control flow statement.
    """
    assert isinstance(node, gast.AST), \
        "The type of input node must be gast.AST, but received %s." % type(node)
    visitor = IsControlFlowVisitor(
        node, static_analysis_visitor, node_var_type_map=var_name_to_type)
    need_to_transform = visitor.transform()
    return need_to_transform


def _delete_keywords_from(node):
    assert isinstance(node, gast.Call)
    func_src = astor.to_source(gast.gast_to_ast(node.func))
    import paddle.fluid as fluid
    full_args = eval("inspect.getargspec({})".format(func_src))
    full_args_name = full_args[0]

    node.keywords = [k for k in node.keywords if k.arg in full_args_name]
    return


def to_static_api(dygraph_class):
    if dygraph_class in dygraph_class_to_static_api:
        return dygraph_class_to_static_api[dygraph_class]
    else:
        raise NotImplementedError("Paddle dygraph API {} cannot be converted "
                                  "to static graph at present.".format(
                                      dygraph_class))


def _add_keywords_to(node, dygraph_api_name):
    assert isinstance(node, gast.Call)
    if dygraph_api_name == "Linear":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"

        node.keywords.append(
            gast.keyword(
                arg="num_flatten_dims",
                value=gast.Constant(
                    value=-1, kind=None)))

    if dygraph_api_name == "BilinearTensorProduct":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"

    if dygraph_api_name == "PRelu":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "input":
                ast_keyword.arg = "x"
    return


def to_static_ast(node, class_node):
    assert isinstance(node, gast.Call)
    assert isinstance(class_node, gast.Call)
    static_api = to_static_api(class_node.func.attr)

    node.func = gast.Attribute(
        attr=static_api,
        ctx=gast.Load(),
        value=gast.Attribute(
            attr='layers',
            ctx=gast.Load(),
            value=gast.Name(
                ctx=gast.Load(), id='fluid', annotation=None,
                type_comment=None)))

    update_args_of_func(node, class_node, 'forward')

    node.args.extend(class_node.args)
    node.keywords.extend(class_node.keywords)
    _add_keywords_to(node, class_node.func.attr)
    _delete_keywords_from(node)

    gast.fix_missing_locations(node)

    return node


def update_args_of_func(node, dygraph_node, method_name):
    assert isinstance(node, gast.Call)
    if method_name not in ["__init__", "forward"]:
        raise ValueError(
            "The method name of class to update args should be '__init__' or 'forward'"
        )

    class_src = astor.to_source(gast.gast_to_ast(dygraph_node.func))
    import paddle.fluid as fluid
    if method_name == "__init__" or eval(
            "issubclass({}, fluid.dygraph.Layer)".format(class_src)):
        full_args = eval("inspect.getargspec({}.{})".format(class_src,
                                                            method_name))
        full_args_name = [
            arg_name for arg_name in full_args[0] if arg_name != "self"
        ]
    else:
        full_args_name = []
    added_keywords = []
    for idx, arg in enumerate(node.args):
        added_keywords.append(gast.keyword(arg=full_args_name[idx], value=arg))

    node.args = []
    node.keywords = added_keywords + node.keywords


def create_api_shape_node(tensor_shape_node):
    assert isinstance(tensor_shape_node,
                      (gast.Name, gast.Attribute, gast.Subscript))

    if isinstance(tensor_shape_node, gast.Name):
        api_shape_node = gast.Call(
            func=gast.parse('fluid.layers.shape').body[0].value,
            args=[tensor_shape_node],
            keywords=[])
        return api_shape_node

    if isinstance(tensor_shape_node, gast.Attribute):
        api_shape_node = gast.Call(
            func=gast.parse('fluid.layers.shape').body[0].value,
            args=[tensor_shape_node.value],
            keywords=[])
        return api_shape_node

    if isinstance(tensor_shape_node, gast.Subscript):
        result_node = copy.deepcopy(tensor_shape_node)
        result_node.value = create_api_shape_node(result_node.value)
        return result_node


def get_constant_variable_node(name, value, shape=[1], dtype='int64'):
    return gast.parse('%s = fluid.layers.fill_constant(%s, "%s", %s)' %
                      (name, str(shape), dtype, str(value)))


def get_attribute_full_name(node):
    assert isinstance(
        node,
        gast.Attribute), "Input non-Attribute node to get attribute full name"
    return astor.to_source(gast.gast_to_ast(node)).strip()


def generate_name_node(name_ids, ctx=gast.Load(), gen_tuple_if_single=False):
    """
    If name_ids is list or tuple or set with multiple strings, this function
    generates gast.Tuple of gast.Name.
    If the name_ids is single string or contains only 1 string, this function
    returns gast.Name if gen_tuple_if_single==False else returns gast.Tuple
    with only one gast.Name

    This function is used at several gast.Return statements.
    """
    if isinstance(name_ids, six.string_types):
        name_ids = [name_ids]
    if not isinstance(name_ids, (list, tuple, set)):
        raise TypeError('name_ids must be list or tuple or set, but received %s'
                        % type(type(name_ids)))
    gast_names = [
        gast.Name(
            id=name_id, ctx=ctx, annotation=None, type_comment=None)
        for name_id in name_ids
    ]
    if len(gast_names) == 1 and not gen_tuple_if_single:
        name_node = gast_names[0]
    else:
        name_node = gast.Tuple(elts=gast_names, ctx=ctx)
    return name_node


def create_funcDef_node(nodes, name, input_args, return_name_ids):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    # add return statement
    if return_name_ids:
        nodes.append(gast.Return(value=generate_name_node(return_name_ids)))
    else:
        nodes.append(gast.Return(value=None))
    func_def_node = gast.FunctionDef(
        name=name,
        args=input_args,
        body=nodes,
        decorator_list=[],
        returns=None,
        type_comment=None)
    return func_def_node


def index_in_list(array_list, item):
    try:
        return array_list.index(item)
    except ValueError:
        # Item not in array_list
        return -1


def create_assign_node(name, node):
    """
    Creates a `gast.Assign` node by given name_id as target and node as value.
    """
    targets = generate_name_node(name, ctx=gast.Store())
    assign_node = gast.Assign(targets=[targets], value=node)
    return targets, assign_node


class RenameTransformer(gast.NodeTransformer):
    def __init__(self, node):
        assert isinstance(
            node, gast.AST), "RenameTransformer only accepts gast.AST as input"
        self.root = node
        self.old_name = ""
        self.new_name = ""

    def rename(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
        self.visit(self.root)

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id == self.old_name:
            node.id = self.new_name
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        attr_full_name = get_attribute_full_name(node)
        if attr_full_name == self.old_name:
            new_name_node = gast.parse(self.new_name).body[0].value
            return new_name_node
        return node


def ast_to_func(ast_root, dyfunc, delete_on_exit=True):
    """
    Transform modified AST of decorated function into python callable object.
    TODO: If only decorate one of inner function instead of decorating the main
    function, the other inner functions are invisible for the decorated function.
    """

    def remove_if_exit(filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    source = ast_to_source_code(ast_root)
    source = _inject_import_statements() + source

    f = tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8')
    with f:
        module_name = os.path.basename(f.name[:-3])
        f.write(source)

    if delete_on_exit:
        atexit.register(lambda: remove_if_exit(f.name))
        atexit.register(lambda: remove_if_exit(f.name[:-3] + ".pyc"))

    module = SourceFileLoader(module_name, f.name).load_module()
    func_name = dyfunc.__name__
    # The 'forward' or 'another_forward' of 'TranslatedLayer' cannot be obtained
    # through 'func_name'. So set the special function name '__i_m_p_l__'.
    if hasattr(module, '__i_m_p_l__'):
        callable_func = getattr(module, '__i_m_p_l__')
        callable_func.__name__ = func_name
    elif hasattr(module, func_name):
        callable_func = getattr(module, func_name)
    else:
        raise ValueError(
            'Function: %s doesn\'t exist in the Module transformed from AST.' %
            func_name)
    # After transform dygraph function into callable_func saved in tmp file,
    # it lost the global variables from imported statements or defined in source file.
    # Recovers the necessary variables by `__globals__`.
    recover_globals_attribute(dyfunc, callable_func)

    return callable_func, f.name


def _inject_import_statements():
    import_statements = [
        "import paddle", "import paddle.fluid as fluid", "from typing import *",
        "import numpy as np"
    ]
    return '\n'.join(import_statements) + '\n'


def recover_globals_attribute(src_obj, dst_obj):
    attr_name = '__globals__'

    src_globals = getattr(src_obj, attr_name, {})
    dst_globals = getattr(dst_obj, attr_name, {})

    for k, v in six.iteritems(src_globals):
        # ignore builtin attribute.
        if not (k.startswith('__') and k.endswith('__')):
            dst_globals[k] = v


def func_to_source_code(function, dedent=True):
    """
    Transforms function into raw string of source code.
    """
    if not (inspect.isfunction(function) or inspect.ismethod(function)):
        raise TypeError(
            "The type of 'function' should be a function or method, but received {}.".
            format(type(function).__name__))
    source_code_list, _ = inspect.getsourcelines(function)
    source_code_list = [
        line for line in source_code_list if not line.lstrip().startswith('#')
    ]
    source_code = ''.join(source_code_list)
    if dedent:
        source_code = textwrap.dedent(source_code)

    return source_code


def ast_to_source_code(ast_node):
    """
    Transforms ast node into source code.
    """
    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s." %
            type(ast_node))
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)
    source_code = astor.to_source(ast_node)
    return source_code


def is_candidate_node(node):
    """
    Nodes with specified type will be dependent on tensor.
    """
    is_compare_node = isinstance(node, (gast.Compare, gast.BoolOp, gast.UnaryOp,
                                        gast.For, gast.If, gast.While))
    # TODO(Aurelius84): `.numpy()` may be an customized function,
    # and should consider a more elegant way to solve this problem.
    has_numpy_attr = ".numpy()" in ast_to_source_code(node)
    return is_compare_node or has_numpy_attr


def compare_with_none(node):
    """
    Whether the comparator of `gast.Compare` node is `None`.
    """
    if isinstance(node, gast.Compare):
        for child in [node.left, node.comparators]:
            # node.comparators is a list.
            if isinstance(child, list):
                child = child[0]
            if (isinstance(child, gast.Constant) and child.value is None) or (
                    isinstance(child, gast.Name) and child.id == 'None'):
                return True
    return False


class IsControlFlowVisitor(gast.NodeVisitor):
    """
    Judge whether the ast_node of control flow from Dygraph code dependent on paddle Tensor.
    `ast_node` can be gast.If, gast.For, gast.While, gast.If.test(gast.Compare, gast.BoolOp, gast.UnaryOp).

    If returns True,
    gast.If.test must meet at least one of the following requirements:
        1. involves at least one var whose type is Tensor.
        2. the Tensor var calls `.numpy()[]` interface or Tensor.shape is [1].
        3. involves Tensor.shape[i] and the shape[i] is unknown in compile time.
    gast.While must meet at least one of the requirements 1 to 5:
        4. has `break` statement.
        5. has `continue` statement.
    gast.For must meet at least one of the requirements 4 to 8:
        6. calls `range` function in `for` statement and the argument of range is Tensor.
        7. calls `enumerate` function in `for` statement and the argument of enumerate is Tensor.
        8. the iterable varaible in `for` statement is Tensor.
        TODO: Support non-range case

    The following examples should not be considered as control_flow_if:
        1. `if Tensor_var` or `if Tensor_var is None`
        2. if Tensor.shape[i] is determined with fixed value (not -1 or None)

    Note: pred in ConditionalBlock require variable, which means all vars should be Tensor
          or transformed into Tensor, like fill_constant(shape=[1], dtype='int32', value=Tensor.shape[i]).

    TODO: 1. need to deal with `tensor.shape[i]` which need to eval the data of shape[i],
             because reshape_op may be called before this statement.
    """

    def __init__(self,
                 ast_node,
                 static_analysis_visitor=None,
                 node_var_type_map=None):
        assert isinstance(
            ast_node, gast.AST
        ), "Type of input node should be gast.AST, but received %s." % type(
            ast_node)
        self.ast_root = ast_node
        if static_analysis_visitor is None:
            from .static_analysis import StaticAnalysisVisitor
            static_analysis_visitor = StaticAnalysisVisitor(ast_node)
        self.static_analysis_visitor = static_analysis_visitor
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        self.node_var_type_map = node_var_type_map

        self.is_control_flow_num = 0
        self._compare_node_tenor_set = set()

    def transform(self):
        node = self.ast_root
        if isinstance(node, gast.If):
            self._visit_If(node)
        elif isinstance(node, gast.For):
            self._visit_For(node)
        elif isinstance(node, gast.While):
            self._visit_While(node)
        else:
            self.visit(node)
        return self.is_control_flow_num > 0

    def _visit_If(self, node):
        assert isinstance(node, gast.If)
        self.visit(node.test)
        return

    def _visit_For(self, node):
        assert isinstance(node, gast.For)
        if isinstance(node.iter, gast.Call):
            # for in range(var[0]|var.numpy()[0]) or for in enumerate(var|var.numpy())
            if isinstance(node.iter.func, gast.Name):
                if node.iter.func.id == "range" or node.iter.func.id == "enumerate":
                    for arg in node.iter.args:
                        self.visit(arg)
                else:
                    return
            # for in var.numpy()
            elif isinstance(node.iter.func, gast.Attribute):
                if node.iter.func.attr == 'numpy':
                    self._visit_Call(node.iter)
                else:
                    return
            else:
                return
        elif isinstance(node.iter, gast.Name):
            # for in var
            self.visit(node.iter)
        else:
            return

        for child_node in gast.walk(node):
            if isinstance(child_node, (gast.Continue, gast.Break)):
                self._visit_break_continue(child_node)
        return

    def _visit_While(self, node):
        assert isinstance(node, gast.While)
        test = node.test
        self.generic_visit(test)
        for child_node in gast.walk(node):
            if isinstance(child_node, (gast.Continue, gast.Break)):
                self._visit_break_continue(child_node)
        return

    def _visit_break_continue(self, node):
        assert isinstance(node, (gast.Break, gast.Continue))
        wrapper_node = self.node_to_wrapper_map.get(node)
        if not wrapper_node:
            # Transformed node is not in node_to_wrapper_map
            return

        while wrapper_node.parent:
            parent_node = wrapper_node.parent.node
            if isinstance(parent_node, (gast.For, gast.While)):
                if parent_node is self.ast_root:
                    self.is_control_flow_num += 1
                    return
                else:
                    return

            wrapper_node = wrapper_node.parent

        return

    def visit_BoolOp(self, node):
        for i, child in enumerate(node.values):
            self.visit(child)
        return node

    def visit_Compare(self, node):
        pre_control_flow_num = self.is_control_flow_num
        if not compare_with_none(node):
            self.generic_visit(node)
            for child in gast.walk(node):
                if isinstance(child, gast.Subscript):
                    self._visit_Subscript(child)
        if self.is_control_flow_num > pre_control_flow_num:
            self._compare_node_tenor_set.add(node)
        return node

    def _visit_Subscript(self, node):
        self.generic_visit(node)
        if hasattr(node, 'value') and isinstance(node.value, gast.Call):
            self._visit_Call(node.value)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.func, gast.Attribute):
            attr_node = node.func
            if attr_node.attr == 'numpy':
                self.is_control_flow_num += 1

    def visit_Call(self, node):
        self._visit_Call(node)
        if is_paddle_api(node):
            self.is_control_flow_num += 1
        return node

    def visit_Name(self, node):
        if self._is_node_with_tensor(node, node.id):
            self.is_control_flow_num += 1
        return node

    def visit_Constant(self, node):
        if self._is_node_with_tensor(node, node.value):
            self.is_control_flow_num += 1
        return node

    def _is_node_with_tensor(self, node, name_id):
        from paddle.fluid.dygraph.dygraph_to_static.static_analysis import NodeVarType

        # Look up the node_var_type_map by name_id.
        if self.node_var_type_map:
            if name_id and isinstance(name_id, six.string_types):
                var_type = self.node_var_type_map.get(name_id, None)
                if var_type and var_type & NodeVarType.TENSOR_TYPES:
                    return True
        # if not found, look up the node_to_wrapper_map by node.
        wrapper_node = self.node_to_wrapper_map.get(node, None)
        if wrapper_node is not None:
            if wrapper_node.node_var_type & NodeVarType.TENSOR_TYPES:
                return True

        return False

    def get_compare_nodes_with_tensor(self):
        return self._compare_node_tenor_set


class NameNodeReplaceTransformer(gast.NodeTransformer):
    """
    This class replaces specified gast.Name node by replace_node.
    """

    def __init__(self, root_node, target_name, replace_node):
        assert isinstance(target_name, str)

        # NOTE(liym27):
        # Use gast.Name to replace gast.Name, otherwise, errors may occur.
        #
        # For examples:
        # If using a gast.Subscript to replace gast.Name, and the original gast.Name
        # is in the arguments of FunctionDef, an exception will be raised.
        #
        # ```
        # def func(x[i])) # x[i] can not be a argument
        #    # ...
        # ```

        assert isinstance(replace_node, gast.Name)
        self.target_name = target_name
        self.replace_node = replace_node

        self.visit(root_node)

    def visit_Name(self, node):
        if node.id == self.target_name:
            return self.replace_node
        return node


class ForLoopTuplePreTransformer(gast.NodeTransformer):
    """
    ForNodeVisitor parses 3 type statements (Here var is VarBase(Tensor) or python variable):
        1). for x in range(var[*]|var.numpy()[*])
        2). for x in var|var.numpy()
        3). for i, x in enumerate(var|var.numpy())

        We chose these 3 types because they are easier (x can be variable name iterating in var).
        However, users can write tuples in Python for loop, such as
        1). for var1, var2 in var|var.numpy()
        2). for t in enumerate(var|var.numpy())
        2). for i, (var1, var2, va3) in enumerate(var|var.numpy())

        To handle these case, this method will do the rewrite tuple pre-process:
        1). Non-enumerate case: for var1, var2 in var|var.numpy() will be re-written as:
          for FOR_ITER_TUPLE_PREFIX_x in var | var.numpy():
            var1 = FOR_ITER_TUPLE_PREFIX_x[0]
            var2 = FOR_ITER_TUPLE_PREFIX_x[1]
        2). Enumerate out tuple case: for t in enumerate(var|var.numpy) will be rewritten as:
          for FOR_ITER_TUPLE_INDEX_PREFIX_x, FOR_ITER_TUPLE_PREFIX_x in enumerate(var|var.numpy):
            t = (FOR_ITER_TUPLE_INDEX_PREFIX_x, FOR_ITER_TUPLE_PREFIX_x)
        3). Enumerate inner tuple case: for i, (var1, (var2, va3)) in enumerate(var|var.numpy()) will
        be re-written as:
          for i, FOR_ITER_TUPLE_PREFIX_x in var | var.numpy():
            var1 = FOR_ITER_TUPLE_PREFIX_x[0]
            var2 = FOR_ITER_TUPLE_PREFIX_x[1][0]
            var3 = FOR_ITER_TUPLE_PREFIX_x[1][1]
    """

    def __init__(self, wrapper_root):
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        self.visit(self.root)

    def visit_For(self, node):
        if self.is_for_enumerate_iter(node):
            if isinstance(node.target, (gast.Name, gast.Attribute)):
                # Out tuple case
                out_tuple_name = ast_to_source_code(node.target).strip()
                tuple_iter_name = unique_name.generate(
                    FOR_ITER_TUPLE_INDEX_PREFIX)
                tuple_var_name = unique_name.generate(FOR_ITER_TUPLE_PREFIX)
                node.target = gast.Tuple(
                    elts=[
                        gast.Name(
                            id=tuple_iter_name,
                            ctx=gast.Store(),
                            annotation=None,
                            type_comment=None), gast.Name(
                                id=tuple_var_name,
                                ctx=gast.Store(),
                                annotation=None,
                                type_comment=None)
                    ],
                    ctx=gast.Store())
                node.body.insert(
                    0,
                    gast.Assign(
                        targets=[
                            gast.Name(
                                id=out_tuple_name,
                                ctx=gast.Store(),
                                annotation=None,
                                type_comment=None)
                        ],
                        value=gast.Tuple(
                            elts=[
                                gast.Name(
                                    id=tuple_iter_name,
                                    ctx=gast.Load(),
                                    annotation=None,
                                    type_comment=None), gast.Name(
                                        id=tuple_var_name,
                                        ctx=gast.Load(),
                                        annotation=None,
                                        type_comment=None)
                            ],
                            ctx=gast.Load())))
            elif isinstance(node.target, (
                    gast.List,
                    gast.Tuple)) and len(node.target.elts) >= 2 and isinstance(
                        node.target.elts[1], (gast.List, gast.Tuple)):
                # Inner tuple case
                inner_tuple_name = unique_name.generate(FOR_ITER_TUPLE_PREFIX)
                origin_inner_tuple_node = node.target.elts[1]
                node.target.elts[1] = gast.Name(
                    id=inner_tuple_name,
                    ctx=gast.Store(),
                    annotation=None,
                    type_comment=None)
                node.body[0:0] = self.tuple_to_stmts(origin_inner_tuple_node,
                                                     inner_tuple_name)
        elif self.is_for_iter(node) and isinstance(node.target,
                                                   (gast.List, gast.Tuple)):
            # Non-enumrate case:
            tuple_name = unique_name.generate(FOR_ITER_TUPLE_PREFIX)
            origin_tuple_node = node.target
            node.target = gast.Name(
                id=tuple_name,
                ctx=gast.Store(),
                annotation=None,
                type_comment=None)
            node.body[0:0] = self.tuple_to_stmts(origin_tuple_node, tuple_name)
        return node

    def tuple_to_stmts(self, node, tuple_name, idx=[]):
        if not isinstance(node, (gast.Tuple, gast.List)):
            value_node_str = tuple_name
            for i in idx:
                value_node_str = value_node_str + "[{}]".format(i)

            node_str = ast_to_source_code(node).strip()
            assign_node_str = "{} = {}".format(node_str, value_node_str)
            assign_node = gast.parse(assign_node_str).body[0]
            return [assign_node]

        # isinstance(node, (gast.Tuple, gast.List))
        ret = []
        for i, element in enumerate(node.elts):
            ret += self.tuple_to_stmts(node.elts[i], tuple_name, idx + [i])
        return ret

    def is_for_iter(self, for_node):
        assert isinstance(for_node,
                          gast.For), "Input node is not gast.For node."
        if isinstance(for_node.iter, (gast.Name, gast.Attribute)):
            return True
        elif isinstance(for_node.iter, gast.Call) and isinstance(
                for_node.iter.func,
                gast.Attribute) and for_node.iter.func.attr == 'numpy':
            return True
        elif isinstance(for_node.iter, gast.Subscript):
            return True
        else:
            return False

    def is_for_enumerate_iter(self, for_node):
        assert isinstance(for_node,
                          gast.For), "Input node is not gast.For node."
        return isinstance(for_node.iter, gast.Call) and isinstance(
            for_node.iter.func,
            gast.Name) and for_node.iter.func.id == "enumerate"


class ForNodeVisitor(object):
    """
    This class parses python for statement, get transformed 3 statement components of for node
    three key statements:
        1). init_stmts: list[node], prepare nodes of for loop, may not only one
        2). cond_stmt: node, condition node to judge whether continue loop
        3). body_stmts: list[node], updated loop body, sometimes we should change
            the original statement in body, not just append new statement

    In this process, the semantics of for does not change.

    Now only can parse 3 type statements (Here var is VarBase(Tensor) or python variable):
        1). for x in range(var[*]|var.numpy()[*])
        2). for x in var|var.numpy()
        3). for i, x enumerate(var|var.numpy())
    """

    def __init__(self, for_node):
        assert isinstance(
            for_node, gast.For
        ), "Input node for the initialization of ForNodeVisitor is not gast.For node."
        # 1. original for node
        self.node = for_node

        # 2. gast.For node main parts
        self.target = for_node.target
        # NOTE: type may be Node or list[Node]
        self.iter_args = for_node.iter if self.is_for_iter(
        ) else for_node.iter.args
        self.body = for_node.body

        # 3. key shared node or names
        # - x:
        #   - for x in range(***)
        #   - for x in var|var.numpy()
        #   - for i, x enumerate(var|var.numpy())
        self.iter_var_name = self._get_iter_var_name()

        # - created index var to slice Variable: __for_loop_var_index_0
        #   - for x in var|var.numpy()
        #   - for i, x enumerate(var|var.numpy())
        self.iter_idx_name = unique_name.generate(FOR_ITER_INDEX_PREFIX)

        # - created shape var to build loop condition: __for_loop_var_len_0
        #   - for x in var|var.numpy()
        #   - for i, x enumerate(var|var.numpy())
        #   - for x in var
        self.iter_var_len_name = unique_name.generate(FOR_ITER_VAR_LEN_PREFIX)
        # - created zip to list var : __for_loop_iter_zip_0
        self.iter_zip_to_list_name = unique_name.generate(
            FOR_ITER_ZIP_TO_LIST_PREFIX)

        # - var.numpy()/var
        #   - for x in var|var.numpy()
        #   - for i, x enumerate(var|var.numpy())
        self.iter_node = self._get_iter_node()

        # - enumeate i:
        #   - for i, x enumerate(var|var.numpy())
        self.enum_idx_name = self._get_enum_idx_name()

        # - range/enumerate args length
        self.args_length = None

    def parse(self):
        self._args_check()
        if self.is_for_range_iter():
            return self._parse_for_range_stmts()
        elif self.is_for_iter():
            return self._parse_for_stmts()
        elif self.is_for_enumerate_iter():
            return self._parse_for_enumerate_stmts()
        else:
            return None

    def is_for_range_iter(self):
        return isinstance(self.node.iter, gast.Call) and isinstance(
            self.node.iter.func,
            gast.Name) and self.node.iter.func.id == "range"

    def is_for_iter(self):
        if isinstance(self.node.iter,
                      (gast.Name, gast.Attribute, gast.List, gast.Tuple)):
            return True
        elif isinstance(self.node.iter, gast.Call) and isinstance(
                self.node.iter.func,
                gast.Attribute) and self.node.iter.func.attr == 'numpy':
            return True
        elif isinstance(self.node.iter, gast.Subscript):
            return True
        else:
            return False

    def is_for_enumerate_iter(self):
        return isinstance(self.node.iter, gast.Call) and isinstance(
            self.node.iter.func,
            gast.Name) and self.node.iter.func.id == "enumerate"

    def _args_check(self):
        if self.is_for_range_iter():
            self.args_length = len(self.iter_args)
            assert self.args_length >= 1 and self.args_length <= 3, "range() function takes 1 to 3 arguments"
        elif self.is_for_enumerate_iter():
            self.args_length = len(self.iter_args)
            assert self.args_length >= 1 and self.args_length <= 2, "enumerate() function takes 1 to 2 arguments"
        else:
            self.args_length = None

    def _parse_for_range_stmts(self):
        init_stmts = []
        init_stmts.append(self._build_index_init_node())

        compare_node = self._build_compare_node()
        step_node = self._build_step_node()
        cond_stmt = self._build_cond_stmt(step_node, compare_node)

        body_stmts = self.body
        body_stmts.append(self._build_index_increase_node(step_node))

        return init_stmts, cond_stmt, body_stmts

    def _parse_for_stmts(self):
        init_stmts = []
        init_stmts.extend(self._build_iter_node())
        init_stmts.append(self._build_index_init_node())
        init_stmts.append(self._build_var_len_assign_node())

        compare_node = self._build_compare_node()
        step_node = self._build_step_node()
        cond_stmt = self._build_cond_stmt(step_node, compare_node)

        body_stmts = self.body

        # NOTE(liym27): Here add a gast.Assign, and the target of it is gast.Name.
        # In NameNodeReplaceTransformer, using gast.Name to replace gast.Name is safe.
        target_node, assign_node = self._build_assign_var_slice_node()
        body_stmts[0:0] = [assign_node]
        for body_node in body_stmts:
            NameNodeReplaceTransformer(body_node, self.iter_var_name,
                                       target_node)
        body_stmts.append(self._build_index_increase_node(step_node))

        return init_stmts, cond_stmt, body_stmts

    def _parse_for_enumerate_stmts(self):
        init_stmts = []
        init_stmts.extend(self._build_iter_node())
        init_stmts.append(self._build_index_init_node())
        init_stmts.append(self._build_var_len_assign_node())
        init_stmts.append(self._build_enum_init_node())

        compare_node = self._build_compare_node()
        step_node = self._build_step_node()
        cond_stmt = self._build_cond_stmt(step_node, compare_node)

        body_stmts = self.body

        target_node, assign_node = self._build_assign_var_slice_node()
        body_stmts[0:0] = [assign_node]
        for body_node in body_stmts:
            NameNodeReplaceTransformer(body_node, self.iter_var_name,
                                       target_node)

        body_stmts.append(self._build_index_increase_node(step_node))
        body_stmts.append(self._build_enum_increase_node())

        return init_stmts, cond_stmt, body_stmts

    def _build_index_init_node(self):
        if self.is_for_range_iter():
            if self.args_length == 1:
                index_init_value_str = '0'
            else:
                index_init_value_str = ast_to_source_code(self.iter_args[
                    0]).strip()

            index_init_var_name = self.iter_var_name
        else:
            index_init_value_str = '0'
            index_init_var_name = self.iter_idx_name

        index_init_node_source_str = "{target} = {value}".format(
            target=index_init_var_name, value=index_init_value_str)

        index_init_node = gast.parse(index_init_node_source_str).body[0]

        return index_init_node

    def _build_var_len_assign_node(self):
        # get the length of iterable variable
        if isinstance(self.iter_node, gast.Call) and isinstance(
                self.iter_node.func,
                gast.Attribute) and self.iter_node.func.attr == 'numpy':
            iter_var_name = ast_to_source_code(self.iter_node.func.value).strip(
            )
        else:
            iter_var_name = ast_to_source_code(self.iter_node).strip()

        convert_len_node_source_str = '{} = paddle.jit.dy2static.convert_len({})'.format(
            self.iter_var_len_name, iter_var_name)

        convert_len_node = gast.parse(convert_len_node_source_str).body[0]

        return convert_len_node

    def _build_iter_node(self):
        """
        Process special cases for iter_node inclue:
          - Case 1 (for zip):
            
            - for i, val in enumerate(zip(x, y))  # original code:
            
            - __for_loop_iter_zip_0 = list(zip(x, y))
            - for i, val in enumerate(__for_loop_iter_zip_0)
        """
        new_nodes = []
        if isinstance(self.iter_node, gast.Call) and isinstance(
                self.iter_node.func, gast.Name):
            if self.iter_node.func.id == 'zip':
                iter_var_name = ast_to_source_code(self.iter_node).strip()
                zip_to_list_str = "{target} = list({value})".format(
                    target=self.iter_zip_to_list_name, value=iter_var_name)
                zip_to_list_node = gast.parse(zip_to_list_str).body[0]
                new_nodes.append(zip_to_list_node)

                self.iter_node = gast.Name(
                    id=self.iter_zip_to_list_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None)

        return new_nodes

    def _build_enum_init_node(self):
        if self.is_for_enumerate_iter() and self.args_length != 1:
            init_value_str = ast_to_source_code(self.iter_args[1]).strip()
        else:
            init_value_str = '0'

        enum_init_node_source_str = "{} = {}".format(self.enum_idx_name,
                                                     init_value_str)
        enum_init_node = gast.parse(enum_init_node_source_str).body[0]
        return enum_init_node

    def _build_compare_node(self):
        if self.is_for_range_iter():
            compare_node = self.iter_args[
                0] if self.args_length == 1 else self.iter_args[1]
        else:
            compare_node = gast.Name(
                id=self.iter_var_len_name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None)
        return compare_node

    def _build_step_node(self):
        if self.is_for_range_iter():
            step_node = self.iter_args[
                2] if self.args_length == 3 else gast.Constant(
                    value=1, kind=None)
        else:
            step_node = gast.Constant(value=1, kind=None)
        return step_node

    def _build_cond_stmt(self, step_node, compare_node):
        if not isinstance(step_node, (gast.Constant, gast.UnaryOp)):
            raise NotImplementedError(
                "Dynamic-to-Static only supports the step value is a constant or negative constant in 'for-range' statements, "
                "such as '2', '-3'. But received: '{}'. Please fix code to be compatible with Dynamic-to-Static."
                .format(ast_to_source_code(step_node).strip()))

        if isinstance(step_node, gast.UnaryOp) or step_node.value < 0:
            # eg:
            # range(max, min, -2)
            # ->
            # i > min
            return gast.Compare(
                left=gast.Name(
                    id=self.iter_var_name
                    if self.is_for_range_iter() else self.iter_idx_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None),
                ops=[gast.Gt()],
                comparators=[compare_node])
        else:
            # eg:
            # range(min, max, 2)
            # ->
            # i < max
            return gast.Compare(
                left=gast.Name(
                    id=self.iter_var_name
                    if self.is_for_range_iter() else self.iter_idx_name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None),
                ops=[gast.Lt()],
                comparators=[compare_node])

    def _build_index_increase_node(self, step_node):
        return gast.AugAssign(
            target=gast.Name(
                id=self.iter_var_name
                if self.is_for_range_iter() else self.iter_idx_name,
                ctx=gast.Store(),
                annotation=None,
                type_comment=None),
            op=gast.Add(),
            value=step_node)

    def _build_assign_var_slice_node(self):
        var_slice_str = "{}[{}]".format(
            ast_to_source_code(self.iter_node).strip(), self.iter_idx_name)
        var_slice_node = gast.parse(var_slice_str).body[0].value
        new_iter_var_name = unique_name.generate(FOR_ITER_VAR_NAME_PREFIX)
        target_node, assign_node = create_assign_node(new_iter_var_name,
                                                      var_slice_node)
        return target_node, assign_node

    def _build_enum_increase_node(self):
        return gast.AugAssign(
            target=gast.Name(
                id=self.enum_idx_name,
                ctx=gast.Store(),
                annotation=None,
                type_comment=None),
            op=gast.Add(),
            value=gast.Constant(
                value=1, kind=None))

    def _get_iter_var_name(self):
        if self.is_for_range_iter():
            return self.target.id
        elif self.is_for_iter():
            return self.target.id
        elif self.is_for_enumerate_iter():
            return self.target.elts[1].id
        return None

    def _get_iter_node(self):
        if self.is_for_iter():
            return self.iter_args
        elif self.is_for_enumerate_iter():
            return self.iter_args[0]
        return None

    def _get_enum_idx_name(self):
        if self.is_for_enumerate_iter():
            return self.target.elts[0].id
        return None


class SplitAssignTransformer(gast.NodeTransformer):
    """
    This class transforms sequence assignments and multi-target assignments to normal assignments.
    """

    def __init__(self, ast_node):
        assert isinstance(ast_node, gast.AST)
        self.ast_root = ast_node

    def transform(self):
        self.visit(self.ast_root)

    def visit_Assign(self, node):
        target_nodes = node.targets
        if len(target_nodes) == 1:
            node = self._parse_sequence_assign(node)
        else:
            node = self._parse_multi_target_assign(node)
        return node

    def _parse_sequence_assign(self, node):
        """
        a, b = c, d
        ->
        a = c
        b = d
        """
        assert isinstance(node, gast.Assign)

        target_nodes = node.targets
        value_node = node.value
        if not isinstance(target_nodes[0], (gast.List, gast.Tuple)):
            return node
        if not isinstance(value_node, (gast.List, gast.Tuple)):
            return node

        targets = node.targets[0].elts
        values = node.value.elts
        if len(targets) != len(values):
            return node

        new_nodes = []
        for target, value in zip(targets, values):
            assign_node = gast.Assign(targets=[target], value=value)
            new_nodes.append(assign_node)

        return new_nodes

    def _parse_multi_target_assign(self, node):
        """
         Example 1:
         a = b = c
         ->
         b = c
         a = b

         Example 2:
         a, b = c, d = x
         ->
         c,d = x
         a = c
         b = d
         """
        assert isinstance(node, gast.Assign)

        target_nodes = node.targets
        value_node = node.value
        new_nodes = []
        for target in reversed(target_nodes):
            assign_node = gast.Assign(targets=[target], value=value_node)
            # NOTE: Because assign_node can be sequence assign statement like `a,b = c,d`,
            # it's necessary to visit this new assign_node
            parsed_node = self.visit_Assign(assign_node)
            if not isinstance(parsed_node, list):
                parsed_node = [parsed_node]

            new_nodes.extend(parsed_node)
            value_node = target

        return new_nodes


# NOTE: inspect.unwrap() exits in PY3 but not in PY2.
def unwrap(func):
    """
    Returns the object wrapped by decorators.
    """

    def _is_wrapped(f):
        return hasattr(f, '__wrapped__')

    unwrapped_f = func
    while (_is_wrapped(unwrapped_f)):
        unwrapped_f = unwrapped_f.__wrapped__

    return unwrapped_f


def input_specs_compatible(src_input_specs, desired_input_specs):
    """
    Returns True if the two input specs are compatible, otherwise False.

    args:
        src_input_spec (list or tuple[InputSpec et.al]): list/tuple of
            paddle.static.InputSpec or int/str et.al
        desired_input_specs (list or tuple[InputSpec et.al]): list/tuple of
            paddle.static.InputSpec or int/str et.al
    """
    len_specs = len(src_input_specs)
    if len_specs != len(desired_input_specs):
        # NOTE(chenweihang): if the input_spec of jit.save is a subset of
        # input_spec of to_static, also compatible
        for spec in src_input_specs:
            if spec not in desired_input_specs:
                return False
    else:
        for (src_spec, desired_spec) in zip(src_input_specs,
                                            desired_input_specs):
            if isinstance(src_spec, paddle.static.InputSpec) or isinstance(
                    desired_spec, paddle.static.InputSpec):
                if not _compatible_tensor_spec(src_spec, desired_spec):
                    return False
            else:
                if not _compatible_non_tensor_spec(src_spec, desired_spec):
                    return False

    return True


def _compatible_tensor_spec(src_spec, desired_spec):
    """
    Check whether two tensor type spec is compatible.
    """
    for spec in [src_spec, desired_spec]:
        if not isinstance(spec, paddle.static.InputSpec):
            return False
    src_shape = src_spec.shape
    other_shape = desired_spec.shape
    len_shape = len(src_shape)
    if len_shape != len(other_shape):
        return False
    for j in range(len_shape):
        if src_shape[j] is None or src_shape[j] < 0:
            continue
        if other_shape[j] is None or other_shape[j] < 0:
            continue
        if src_shape[j] != other_shape[j]:
            return False

    src_dtype = convert_dtype(src_spec.dtype)
    other_dtype = convert_dtype(desired_spec.dtype)
    if src_dtype != other_dtype:
        return False

    return True


def _compatible_non_tensor_spec(src_spec, desired_spec):
    """
    Check whether two non-tensor type spec is compatible.
    """

    def hash_value(spec):
        try:
            hash_val = make_hashable(spec)
        except:
            hash_val = None
        return hash_val

    src_hash_val = hash_value(src_spec)
    desired_hash_val = hash_value(desired_spec)

    if src_hash_val != desired_hash_val:
        return False
    else:
        return True


def slice_is_num(slice_node):
    # A slice_node.slice can be a:
    # (1) ast.Index, which is a simple number such as [1], [-2]
    # (2) ast.Slice, which is represented by bounds such as [2:-1]
    # (3) ast.Tuple, which includes the above two cases such as [2:-1, 1]
    # If slice node is case (1), return True, Otherwise, return False.
    #
    # NOTE: In (1) case, when gast>=0.4.0, gast.Index is not used, which is replaced
    # other gast node such as gast.Constant, gast.Name, gast.UnaryOp and so on.
    # Considering the compatibility of gast, here use ast note to check whether the
    # node is a num. For more details, please visit https://github.com/serge-sans-paille/gast

    assert isinstance(slice_node, gast.Subscript)
    slice_node_str = ast_to_source_code(slice_node).strip()
    ast_node = ast.parse(slice_node_str).body[0].value

    if isinstance(ast_node.slice, (ast.Tuple, ast.Slice)):
        return False

    if isinstance(ast_node.slice, ast.Index):
        return True

    return False
