# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from ..wrapped_decorator import signature_safe_contextmanager

from .layer_function_generator import autodoc, templatedoc
from .tensor import assign, cast, fill_constant
from .. import core
from ..framework import Program, Variable, Operator, in_dygraph_mode, static_only
from ..layer_helper import LayerHelper, unique_name
from .nn import logical_and, logical_not, logical_or
from .utils import assert_same_structure, map_structure, hold_mutable_vars, copy_mutable_vars
import numpy
import warnings
import six
from functools import reduce, partial
from ..data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ... import compat as cpt
from ..backward import _infer_var_data_type_shape_
from paddle import _C_ops

__all__ = [
    'While', 'Switch', 'increment', 'array_write', 'create_array', 'less_than',
    'less_equal', 'greater_than', 'greater_equal', 'equal', 'not_equal',
    'array_read', 'array_length', 'cond', 'IfElse', 'DynamicRNN', 'StaticRNN',
    'reorder_lod_tensor_by_rank', 'Print', 'Assert', 'is_empty', 'case',
    'switch_case', 'while_loop'
]


def select_output(input, outputs, mask):
    """
    **select_output**    
    This API takes in one input and multiple outputs and an integer mask. It
    selects the output specified by the mask and copy the input to selected
    output. It is useful in control flow.

    Args:
        input(Variable): The input variable
        outputs(tuple|list): The output variables
        mask(Variable): A tensor containing 1 integer number selecting which
            output to be copied with input

    Returns:
        Variable: The outputs variables
    """
    helper = LayerHelper('select_output', **locals())
    check_type(input, 'input', (Variable), 'select_output')
    check_variable_and_dtype(mask, 'mask', ['int32'], 'select_output')
    check_type(outputs, 'outputs', (list, tuple), 'select_output')

    helper.append_op(
        type='select_output',
        inputs={'X': input,
                'Mask': mask},
        outputs={'Out': outputs})
    return outputs


def select_input(inputs, mask):
    """
    **select_input**
    
    This API takes in multiple inputs and uses an integer mask to select one
    input to output. It is useful in control flow.

    Args:
        inputs(tuple|list): The input variables
        mask(Variable): A tensor containing 1 integer number selecting which
            input to output

    Returns:
        Variable: The selected input variable
    """
    helper = LayerHelper('select_input', **locals())
    check_type(inputs, 'inputs', (list, tuple), 'select_input')
    check_variable_and_dtype(mask, 'mask', ['int32'], 'select_input')

    input_dtype = inputs[0].dtype
    input_shape = inputs[0].shape
    input_type = inputs[0].type

    out = helper.create_variable(
        dtype=input_dtype, shape=input_shape, type=input_type)
    helper.append_op(
        type='select_input',
        inputs={'X': inputs,
                'Mask': mask},
        outputs={'Out': out})
    return out


def select_input_with_buildin_type(inputs, mask):
    from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import to_static_variable
    support_ret_buildin_type = (bool, float, six.integer_types)
    false_var, true_var = inputs

    if isinstance(false_var, Variable) and isinstance(true_var, Variable):
        return select_input(inputs, mask)

    elif (isinstance(false_var, (support_ret_buildin_type)) and
          isinstance(false_var, type(true_var))):
        if false_var == true_var:
            return false_var
        else:
            inputs = [
                to_static_variable(false_var), to_static_variable(true_var)
            ]
    # Deal with the situations like this: false_var is int and true_var is Variable
    elif ((isinstance(false_var, support_ret_buildin_type) and
           isinstance(true_var, Variable)) or
          (isinstance(true_var, support_ret_buildin_type) and
           isinstance(false_var, Variable))):
        inputs = [to_static_variable(false_var), to_static_variable(true_var)]
        warnings.warn(
            "Return results from different branches in cond are not same type: "
            "false_var returned by fasle_fn is '{}' and true_var of true_fn is "
            "'{}'".format(type(false_var), type(true_var)))
    else:
        raise TypeError(
            "Unsupported return type of true_fn and false_fn in cond: false_var "
            "returned by fasle_fn is '{}' and true_var of true_fn is '{}'".
            format(type(false_var), type(true_var)))

    return select_input(inputs, mask)


def split_lod_tensor(input, mask, level=0):
    """
    This function takes in an input that contains the complete lod information,
    and takes in a mask which is used to mask certain parts of the input.
    The output is the true branch and the false branch with the mask applied to
    the input at a certain level in the tensor. Mainly used in IfElse to split
    data into two parts.

    Args:
        input(Variable|tuple|list|None): The input tensor that contains complete
                                lod information needed to construct the output.
        mask(Variable|list): A bool column vector which masks the input.
        level(int): The specific lod level to split.

    Returns:
        tuple(Variable, Variable):
        The true branch of tensor as per the mask applied to input.

        The false branch of tensor as per the mask applied to input.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = fluid.layers.data(name='x', shape=[1])
          x.persistable = True

          y = fluid.layers.data(name='y', shape=[1])
          y.persistable = True

          out_true, out_false = fluid.layers.split_lod_tensor(
                input=x, mask=y, level=level)

    """
    check_type(input, 'input', (Variable, list, tuple, type(None)),
               'fluid.layers.split_lod_tensor')
    check_type(mask, 'mask', (Variable, list), 'fluid.layers.split_lod_tensor')
    check_type(level, 'level', int, 'fluid.layers.split_lod_tensor')
    helper = LayerHelper('split_lod_tensor', **locals())
    out_true = helper.create_variable_for_type_inference(dtype=input.dtype)
    out_false = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='split_lod_tensor',
        inputs={
            'X': input,
            'Mask': mask,
        },
        outputs={'OutTrue': out_true,
                 'OutFalse': out_false},
        attrs={'level': level})
    return out_true, out_false


def merge_lod_tensor(in_true, in_false, x, mask, level=0):
    """
    **merge_lod_tensor**

    This function takes in an input :math:`x`, the True branch, the False
    branch and a binary :math:`mask`. Using this information, this function
    merges the True and False branches of the tensor into a single tensor as
    output at a certain lod level indicated by :math:`level`. Used in IfElse
    to merge the output if True block and False Block.

    Args:
        in_true(Variable|tuple|list|None): The True branch to be merged.
        in_false(Variable|tuple|list|None): The False branch to be merged.
        x(Variable|tuple|list|None): The input tensor that contains complete
                            lod information needed to construct the output.
        mask(Variable|list): A bool column vector which masks the input.
        level(int): The specific lod level to merge.

    Returns:
        Variable: The merged output tensor.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = layers.data(
                      name='x', shape=[1], dtype='float32', stop_gradient=False)
          y = layers.data(
                name='y', shape=[1], dtype='bool', stop_gradient=False)

          level = 0

          out_true, out_false = layers.split_lod_tensor(
                input=x, mask=y, level=level)
          out = layers.merge_lod_tensor(
                in_true=out_true, in_false=out_false, mask=y, x=x, level=level)
    """
    helper = LayerHelper('merge_lod_tensor', **locals())
    check_type(x, 'x', (Variable, list, tuple, type(None)),
               'fluid.layers.merge_lod_tensor')
    check_type(mask, 'mask', (Variable, list), 'fluid.layers.merge_lod_tensor')
    check_type(in_true, 'in_true', (Variable, list, tuple, type(None)),
               'fluid.layers.merge_lod_tensor')
    check_type(in_false, 'in_false', (Variable, list, tuple, type(None)),
               'fluid.layers.merge_lod_tensor')
    out = helper.create_variable_for_type_inference(dtype=in_true.dtype)
    helper.append_op(
        type='merge_lod_tensor',
        inputs={'X': x,
                'Mask': mask,
                'InTrue': in_true,
                'InFalse': in_false},
        outputs={'Out': out},
        attrs={'level': level})
    return out


@static_only
def Print(input,
          first_n=-1,
          message=None,
          summarize=20,
          print_tensor_name=True,
          print_tensor_type=True,
          print_tensor_shape=True,
          print_tensor_layout=True,
          print_tensor_lod=True,
          print_phase='both'):
    '''
    :api_attr: Static Graph

    **Print operator**

    This creates a print op that will print when a tensor is accessed.

    Wraps the tensor passed in so that whenever that a tensor is accessed,
    the message `message` is printed, along with the current value of the
    tensor `t`.

    Args:
        input (Variable): A Tensor to print.
        summarize (int): Number of elements in the tensor to be print. If it's
                value is -1, then all elements in the tensor will be print.
        message (str): A string message to print as a prefix.
        first_n (int): Only log `first_n` number of times.
        print_tensor_name (bool, optional): Print the tensor name. Default: True.
        print_tensor_type (bool, optional): Print the tensor type. Defaultt: True.
        print_tensor_shape (bool, optional): Print the tensor shape. Default: True.
        print_tensor_layout (bool, optional): Print the tensor layout. Default: True.
        print_tensor_lod (bool, optional): Print the tensor lod. Default: True.
        print_phase (str): Which phase to displace, including 'forward',
                'backward' and 'both'. Default: 'both'. If set to 'backward', will 
                only print the gradients of input tensor; If set to 'both', will
                both print the input tensor itself and the gradients of input tensor.

    Returns:
        Variable: Output tensor.

    NOTES:
        The input and output are two different variables, and in the
        following process, you should use the output variable but not the input,
        otherwise, the print layer doesn't have backward.

    Examples:
        .. code-block:: python
           
           import paddle

           paddle.enable_static()
        
           x = paddle.full(shape=[2, 3], fill_value=3, dtype='int64')
           out = paddle.static.Print(x, message="The content of input layer:")

           main_program = paddle.static.default_main_program()
           exe = paddle.static.Executor(place=paddle.CPUPlace())
           res = exe.run(main_program, fetch_list=[out])
           # Variable: fill_constant_1.tmp_0
           #   - message: The content of input layer:
           #   - lod: {}
           #   - place: CPUPlace
           #   - shape: [2, 3]
           #   - layout: NCHW
           #   - dtype: long
           #   - data: [3 3 3 3 3 3]
    '''
    check_variable_and_dtype(input, 'input',
                             ['float32', 'float64', 'int32', 'int64', 'bool'],
                             'fluid.layers.Print')

    helper = LayerHelper('print' + "_" + input.name, **locals())
    output = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='print',
        inputs={'In': input},
        outputs={'Out': output},
        attrs={
            'first_n': first_n,
            'summarize': summarize,
            'message': message or "",
            'print_tensor_name': print_tensor_name,
            'print_tensor_type': print_tensor_type,
            'print_tensor_shape': print_tensor_shape,
            'print_tensor_layout': print_tensor_layout,
            'print_tensor_lod': print_tensor_lod,
            'print_phase': print_phase.upper()
        })
    return output


def Assert(cond, data=None, summarize=20, name=None):
    '''
    This API creates an op that asserts the given condition is true. If the
    condition is false, prints the tensors in data. ``summarize`` specifies the
    number of the elements in the tensors to print.

    Args:
        cond (Variable): The boolean condition tensor whose numel should be 1.
        data (list|tuple, optional): list or tuple of tensors to print when
            condition is not true. If it's ``None``, no tensor will be printed.
            The default value is ``None``.
        summarize (int, optional): Number of elements in the tensor to be
            printed. If its value is -1, then all elements in the tensor will
            be printed. The default value is 20.
        name (str, optional): The default value is ``None`` . Normally users
            don't have to set this parameter. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        Operator: the created operation.

    Raises:
        TypeError: If ``cond`` is not boolean Variable.
        TypeError: If ``data`` is not a list or tuple or ``None``.
        TypeError: If ``summarize`` is not int.
        TypeError: If ``name`` is not a string or ``None`` .
        fluid.core.EnforceNotMet: If the condition is False in running time.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            x = layers.fill_constant(shape=[2, 3], dtype='float32', value=2.0)
            condition = layers.reduce_max(x) < 1.0 # False
            layers.Assert(condition, [x], 10, "example_assert_layer")

            exe = fluid.Executor()
            try:
                exe.run(fluid.default_main_program())
                # Print x and throws paddle.fluid.core.EnforceNotMet exception
                # Example printed message for x:
                #
                # Variable: fill_constant_0.tmp_0
                #   - lod: {}
                #   - place: CPUPlace()
                #   - shape: [2, 3]
                #   - layout: NCHW
                #   - dtype: float
                #   - data: [2 2 2 2 2 2]
            except fluid.core.EnforceNotMet as e:
                print("Assert Exception Example")

    '''
    check_variable_and_dtype(cond, "cond", ["bool"], "fluid.layers.Assert")
    check_type(data, "data", (list, tuple, type(None)), "fluid.layers.Assert")
    check_type(summarize, "summarize", int, "fluid.layers.Assert")
    check_type(name, "name", (str, type(None)), "fluid.layers.Assert")

    layer_name = name if name else ('assert_' + cond.name)
    helper = LayerHelper(layer_name, **locals())

    op = helper.append_op(
        type="assert",
        inputs={"Cond": cond,
                "Data": [] if data is None else list(data)},
        attrs={"summarize": summarize})

    return op


class BlockGuard(object):
    """
    BlockGuard class.

    BlockGuard class is used to create a sub-block in a program by
    using the Python `with` keyword.
    """

    def __init__(self, main_program):
        if not isinstance(main_program, Program):
            raise TypeError("BlockGuard takes a program")
        self.main_program = main_program

    def __enter__(self):
        self.main_program._create_block()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.main_program._rollback()
        if exc_type is not None:
            return False  # re-raise exception
        return True


class BlockGuardWithCompletion(BlockGuard):
    """
    BlockGuardWithCompletion class.

    BlockGuardWithCompletion class is used to create an op with a block in a program.
    """

    def __init__(self, rnn):
        if not isinstance(rnn, StaticRNN):
            raise TypeError("BlockGuardWithCompletion takes a StaticRNN")
        super(BlockGuardWithCompletion, self).__init__(rnn.helper.main_program)
        self.rnn = rnn

    def __enter__(self):
        self.rnn.status = StaticRNN.IN_RNN_BLOCK
        return super(BlockGuardWithCompletion, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.rnn.status = StaticRNN.AFTER_RNN_BLOCK
        self.rnn._complete_op()
        return super(BlockGuardWithCompletion, self).__exit__(exc_type, exc_val,
                                                              exc_tb)


class StaticRNNMemoryLink(object):
    """
    StaticRNNMemoryLink class.

    StaticRNNMemoryLink class is used to create a link between two
    memory cells of a StaticRNN.


    NOTE: This is a internal data structure of a very low-level API.
    Please use StaticRNN instead.

    Args:
        init(Variable): the initial variable for Memory.
        pre_mem(Variable): the memory variable in previous time step.
        mem(Variable): the memory variable in current time step.
    """

    def __init__(self, init, pre_mem, mem=None):
        self.init = init
        self.pre_mem = pre_mem
        self.mem = mem


class StaticRNN(object):
    """
    :api_attr: Static Graph

    StaticRNN class.

    The StaticRNN can process a batch of sequence data. The first dimension of inputs
    represents sequence length, the length of each input sequence must be equal.
    StaticRNN will unfold sequence into time steps, user needs to define how to process
    each time step during the :code:`with` step.

    Args:
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            vocab_size, hidden_size=10000, 200
            x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            # create word sequence
            x_emb = layers.embedding(
                input=x,
                size=[vocab_size, hidden_size],
                dtype='float32',
                is_sparse=False)
            # transform batch size to dim 1
            x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            rnn = fluid.layers.StaticRNN()
            with rnn.step():
                # mark created x_emb as input, each step process a word
                word = rnn.step_input(x_emb)
                # create prev memory parameter, batch size comes from word
                prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                # use hidden to update prev
                rnn.update_memory(prev, hidden)
                # mark hidden as output 
                rnn.step_output(hidden)
            # get StaticrNN final output
            result = rnn()

    """
    BEFORE_RNN_BLOCK = 0
    IN_RNN_BLOCK = 1
    AFTER_RNN_BLOCK = 2

    def __init__(self, name=None):
        check_type(name, "name", (str, type(None)), "fluid.layers.StaticRNN")
        self.helper = LayerHelper("static_rnn", name=name)
        self.memories = {}  # memory map, from pre_mem.name --> MemoryLink
        self.inputs = []  # input variable list in current block
        self.outputs = []  # output variable list in parent block
        self.status = StaticRNN.BEFORE_RNN_BLOCK  # status flag.
        # sequence length, since it is a static RNN, sequence length are fixed.
        self.seq_len = None

    def step(self):
        """
        Define operators in each step. step is used in :code:`with` block, OP in :code:`with` block
        will be executed sequence_len times (sequence_len is the length of input)
        """
        return BlockGuardWithCompletion(self)

    def _assert_in_rnn_block_(self, method):
        if self.status != StaticRNN.IN_RNN_BLOCK:
            raise ValueError("You must invoke {0} in rnn block".format(method))

    def memory(self,
               init=None,
               shape=None,
               batch_ref=None,
               init_value=0.0,
               init_batch_dim_idx=0,
               ref_batch_dim_idx=1):
        """
        Create a memory variable for static rnn.
        If the :code:`init` is not None, :code:`memory` will be initialized by
        this Variable. If the :code:`init` is None, :code:`shape` and :code:`batch_ref`
        must be set, and this function will create a new variable with shape and batch_ref
        to initialize :code:`init` Variable.

        Args:
            init(Variable, optional): Tensor used to init memory. If it is not set,
                :code:`shape` and :code:`batch_ref` must be provided.
                Default: None.
            shape(list|tuple): When :code:`init` is None use this arg to initialize memory shape.
            NOTE the shape does not contain batch_size. Default: None.
            batch_ref(Variable, optional): When :code:`init` is None, memory's batch size will
            be set as batch_ref's ref_batch_dim_idx value. Default: None.
            init_value(float, optional): When :code:`init` is None, used to init memory's value. Default: 0.0.
            init_batch_dim_idx(int, optional): the batch_size axis of the :code:`init` Variable. Default: 0.
            ref_batch_dim_idx(int, optional): the batch_size axis of the :code:`batch_ref` Variable. Default: 1.

        Returns:
            Variable: The memory variable.

        Examples 1:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers

            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
                	dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
                	# mark created x_emb as input, each step process a word
                	word = rnn.step_input(x_emb)
                	# create prev memory parameter, batch size comes from word
                	prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                	hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                	# use hidden to update prev
                	rnn.update_memory(prev, hidden)


        Examples 2:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers
            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
                	dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])
            	boot_memory = fluid.layers.data(name='boot', shape=[hidden_size], dtype='float32', lod_level=1)
            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
            		# mark created x_emb as input, each step process a word
            		word = rnn.step_input(x_emb)
            		# init memory
            		prev = rnn.memory(init=boot_memory)
            		hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
            		# update hidden with prev
            		rnn.update_memory(prev, hidden)

        """
        self._assert_in_rnn_block_('memory')
        check_type(init, "init", (Variable, type(None)),
                   "fluid.layers.StaticRNN.memory")
        check_type(shape, "shape", (list, tuple, type(None)),
                   "fluid.layers.StaticRNN.memory")
        check_type(batch_ref, "batch_ref", (Variable, type(None)),
                   "fluid.layers.StaticRNN.memory")
        if init is None:
            if shape is None or batch_ref is None:
                raise ValueError(
                    "if init is None, memory at least need shape and batch_ref")
            parent_block = self._parent_block()
            var_name = unique_name.generate_with_ignorable_key("@".join(
                [self.helper.name, "memory_boot"]))
            boot_var = parent_block.create_var(
                name=var_name,
                shape=shape,
                dtype=batch_ref.dtype,
                persistable=False)

            parent_block.append_op(
                type="fill_constant_batch_size_like",
                inputs={'Input': [batch_ref]},
                outputs={'Out': [boot_var]},
                attrs={
                    'value': init_value,
                    'shape': boot_var.shape,
                    'dtype': boot_var.dtype,
                    'input_dim_idx': ref_batch_dim_idx,
                    'output_dim_idx': init_batch_dim_idx
                })

            return self.memory(init=boot_var)
        else:
            pre_mem = self.helper.create_variable(
                name=unique_name.generate_with_ignorable_key("@".join(
                    [self.helper.name, "mem"])),
                dtype=init.dtype,
                shape=init.shape)
            self.memories[pre_mem.name] = StaticRNNMemoryLink(
                init=init, pre_mem=pre_mem)
            return pre_mem

    def step_input(self, x):
        """
        Mark a sequence as a StaticRNN input.

        Args:
            x(Variable): The input sequence, the shape of x
                should be [seq_len, ...].

        Returns:
            Variable: The current time step data in the input sequence.

        Examples:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers

            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
                	dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
                	# mark created x_emb as input, each step process a word
                	word = rnn.step_input(x_emb)
                	# create prev memory parameter, batch size comes from word
                	prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                	hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                	# use hidden to update prev
                	rnn.update_memory(prev, hidden)

        """
        self._assert_in_rnn_block_('step_input')
        check_type(x, "x", Variable, "fluid.layers.StaticRNN.step_input")
        if self.seq_len is None:
            self.seq_len = x.shape[0]
        elif x.shape[0] != -1 and self.seq_len != x.shape[0]:
            raise ValueError("Static RNN only take fix seq_len input")

        ipt = self.helper.create_variable(
            name=x.name, dtype=x.dtype, shape=list(x.shape[1:]), type=x.type)
        self.inputs.append(ipt)
        return ipt

    def step_output(self, o):
        """
        Mark a sequence as a StaticRNN output.

        Args:
            o(Variable): The output sequence.

        Returns:
            None.

        Examples:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers

            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
               		dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
                	# mark created x_emb as input, each step process a word
               		word = rnn.step_input(x_emb)
                	# create prev memory parameter, batch size comes from word
                	prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                	hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                	# use hidden to update prev
                	rnn.update_memory(prev, hidden)
                	rnn.step_output(hidden)

            	result = rnn()

        """
        self._assert_in_rnn_block_('step_output')
        check_type(o, "o", Variable, "fluid.layers.StaticRNN.step_output")

        tmp_o = self.helper.create_variable_for_type_inference(dtype=o.dtype)
        self.helper.append_op(
            type='rnn_memory_helper',
            inputs={'X': [o]},
            outputs={'Out': tmp_o},
            attrs={'dtype': o.dtype})

        out_var = self._parent_block().create_var(
            name=tmp_o.name,
            shape=[self.seq_len] + list(tmp_o.shape),
            dtype=tmp_o.dtype)

        self.outputs.append(out_var)

    def output(self, *outputs):
        """
        Mark the StaticRNN output variables.

        Args:
            outputs: The output Tensor, can mark multiple variables as output

        Returns:
            None

        Examples:
            .. code-block:: python

            	import paddle.fluid as fluid
            	import paddle.fluid.layers as layers

            	vocab_size, hidden_size=10000, 200
            	x = fluid.data(name="x", shape=[None, 1, 1], dtype='int64')
            	# create word sequence
            	x_emb = layers.embedding(
                	input=x,
                	size=[vocab_size, hidden_size],
                	dtype='float32',
                	is_sparse=False)
            	# transform batch size to dim 1
            	x_emb = layers.transpose(x_emb, perm=[1, 0, 2])

            	rnn = fluid.layers.StaticRNN()
            	with rnn.step():
                	# mark created x_emb as input, each step process a word
                	word = rnn.step_input(x_emb)
                	# create prev memory parameter, batch size comes from word
                	prev = rnn.memory(shape=[-1, hidden_size], batch_ref = word)
                	hidden = fluid.layers.fc(input=[word, prev], size=hidden_size, act='relu')
                	# use hidden to update prev
                	rnn.update_memory(prev, hidden)
                	# mark each step's hidden and word as output
                	rnn.output(hidden, word)

            	result = rnn()
        """
        for each in outputs:
            self.step_output(each)

    def update_memory(self, mem, var):
        """
        Update the memory from :code:`mem` to :code:`var`.

        Args:
            mem(Variable): the memory variable.
            var(Variable): the plain variable generated in RNN block, used to update memory.
                           var and mem should have same dims and data type.

        Returns:
            None

        """
        check_type(mem, "mem", Variable, "fluid.layers.StaticRNN.update_memory")
        check_type(var, "var", Variable, "fluid.layers.StaticRNN.update_memory")
        self.memories[mem.name].mem = var

    def _parent_block(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)
        return parent_block

    def __call__(self, *args, **kwargs):
        if self.status != StaticRNN.AFTER_RNN_BLOCK:
            raise ValueError("RNN output can only be retrieved after rnn block")
        if len(self.outputs) == 0:
            raise ValueError("RNN has no output")
        elif len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def _complete_op(self):
        main_program = self.helper.main_program
        rnn_block = main_program.current_block()
        parent_block = self._parent_block()

        local_inputs = set()

        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    local_inputs.add(out_var_name)

        for var in self.inputs:
            local_inputs.add(var.name)
        for m in self.memories:
            local_inputs.add(m)

        # NOTE(zcd): the params have two categories of variables.
        #   - the variables that are the out of StaticRnn.
        #   - the variables that are the parameters of some layers, for example, conv2d.
        params = list()
        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in local_inputs:
                        params.append(in_var_name)

        parameters = [parent_block.var(name) for name in set(params)]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        inlinks = [parent_block.var(i.name) for i in self.inputs]
        outlinks = self.outputs

        # NOTE(zcd): the states maybe empty in some case.
        boot_memories = []
        pre_memories = []
        memories = []
        for _, mem in six.iteritems(self.memories):
            boot_memories.append(mem.init)
            pre_memories.append(mem.pre_mem.name)
            assert mem.mem is not None, "%s should be updated in every step." % (
                mem.init.name)
            mem_var = rnn_block.var(mem.mem.name)
            assert isinstance(mem_var, Variable)
            new_mem = self.helper.create_variable_for_type_inference(
                dtype=mem_var.dtype)
            rnn_block.append_op(
                type='rnn_memory_helper',
                inputs={'X': [mem_var]},
                outputs={'Out': [new_mem]},
                attrs={'dtype': mem_var.dtype})

            memories.append(new_mem.name)

        parent_block.append_op(
            type='recurrent',
            inputs={
                'inputs': inlinks,
                'initial_states': boot_memories,
                'parameters': parameters
            },
            outputs={'outputs': outlinks,
                     'step_scopes': [step_scope]},
            attrs={
                'has_states': len(pre_memories) > 0,
                'ex_states': pre_memories,
                'states': memories,
                'sub_block': rnn_block
            })


class WhileGuard(BlockGuard):
    def __init__(self, while_op):
        if not isinstance(while_op, While):
            raise TypeError("WhileGuard takes a while op")
        super(WhileGuard, self).__init__(while_op.helper.main_program)
        self.while_op = while_op

    def __enter__(self):
        self.while_op.status = While.IN_WHILE_BLOCK
        return super(WhileGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.while_op.status = While.AFTER_WHILE_BLOCK
        self.while_op._complete()
        return super(WhileGuard, self).__exit__(exc_type, exc_val, exc_tb)


def get_inputs_outputs_in_block(current_block, inner_inputs, inner_outputs,
                                helper):
    """
    Find inputs and outputs in current control flow block.
    :param current_block: Current control flow block.
    :param inner_inputs: Input var name of ops in current block.
    :param inner_outputs: Output var name of ops in current block.
    :return: inner_inputs, inner_outputs
    """

    # Step1: update inner_inputs and inner_outputs
    # NOTE: Here assumes that all variables are input or output of Ops,
    # but some variables are created without appendding a real op.
    # For example, in `arr = create_array(dtype)`, `arr` is not a output of a op.
    for op in current_block.ops:
        assert isinstance(op, Operator)
        for iname in op.input_names:
            for in_var_name in op.input(iname):
                if in_var_name not in inner_outputs:
                    inner_inputs.add(in_var_name)

        for oname in op.output_names:
            for out_var_name in op.output(oname):
                inner_outputs.add(out_var_name)

    # Step2: Remove LOD_TENSOR_ARRAY created in current control flow block.
    remove_inner_inputs = set()
    parent_block = helper.main_program.block(current_block.parent_idx)

    for in_var_name in inner_inputs:
        parent_block_var = parent_block._find_var_recursive(in_var_name)
        current_block_var = None
        if current_block.has_var(in_var_name):
            current_block_var = current_block.var(in_var_name)
        if not parent_block_var and current_block_var and \
                current_block_var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
            remove_inner_inputs.add(in_var_name)

    inner_inputs = inner_inputs - remove_inner_inputs

    return inner_inputs, inner_outputs


class While(object):
    """
    :api_attr: Static Graph
    
    while loop control flow. Repeat while body until cond is False.

    Note:
        A new OP :ref:`api_fluid_layers_while_loop` is highly recommended instead of ``While`` if the shape of parameter ``cond`` is [1].
        OP :ref:`api_fluid_layers_while_loop` is easier to use and is called with less code but does the same thing as ``While`` .

    Notice:
        Local variables created in ``While`` are similar to that created in while of C++, and cannot be referenced externally.
        As a result, they cannot be obtained through ``fetch_list`` of ``Executor``. If you would like to access the variable
        out of ``while`` , PaddlePaddle provides ``assign`` API to assign local variables to external. Please refer to example
        code 2 or refer to `issue#22724 <https://github.com/PaddlePaddle/Paddle/issues/22724>`_.

    Args:
        cond(Variable): A Tensor whose data type is bool controlling whether to continue looping.
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default value is False.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Examples 1:
          .. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np

            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)           # loop counter

            loop_len = fluid.layers.fill_constant(shape=[1],dtype='int64', value=10)    # loop length

            cond = fluid.layers.less_than(x=i, y=loop_len)
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():
                i = fluid.layers.increment(x=i, value=1, in_place=True)
                fluid.layers.less_than(x=i, y=loop_len, cond=cond)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[i])
            print(res) # [array([10])]


    Examples 2:
          .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            loop_len = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
            one = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1)
            data = fluid.data(name='data', shape=[1], dtype='float32')
            sums = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0)  # Define the variable to be obtained ouside of While, which name should be different from the variable inside the While to be obtained

            cond = fluid.layers.less_than(x=i, y=loop_len)
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():
                sums_tensor = fluid.layers.elementwise_add(x=data, y=data)
                fluid.layers.assign(sums_tensor, sums)  # Update the value of sums_tensor defined in While to the sums which defined outside of While through layers.assign
                i = fluid.layers.increment(x=i, value=1, in_place=True)
                data = fluid.layers.elementwise_add(x=data, y=one)
                fluid.layers.less_than(x=i, y=loop_len, cond=cond)

            feed_data = np.ones(1).astype('float32')
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            res = exe.run(fluid.default_main_program(), feed={'data': feed_data}, fetch_list=sums)
            print(res[0])  # [2.]    # Because the data in While does not update the value outside the While, the value of sums is [2.] after the loop
    """

    BEFORE_WHILE_BLOCK = 0
    IN_WHILE_BLOCK = 1
    AFTER_WHILE_BLOCK = 2

    def __init__(self, cond, is_test=False, name=None):
        self.helper = LayerHelper("while", name=name)
        self.status = While.BEFORE_WHILE_BLOCK
        check_variable_and_dtype(cond, 'cond', ['bool'], 'fluid.layers.While')
        if reduce(lambda a, b: a * b, cond.shape, 1) != 1:
            raise TypeError(
                "condition expected shape as [1], but given shape as {0}.".
                format(list(cond.shape)))
        self.cond_var = cond
        self.is_test = is_test

    def block(self):
        return WhileGuard(self)

    def _complete(self):
        main_program = self.helper.main_program
        while_block = main_program.current_block()
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)

        inner_outputs = {self.cond_var.name}
        x_name_list = set()
        x_name_list, inner_outputs = get_inputs_outputs_in_block(
            while_block, x_name_list, inner_outputs, self.helper)

        out_vars = []
        for inner_out_name in inner_outputs:
            inner_var = parent_block._find_var_recursive(inner_out_name)
            if inner_var:
                out_vars.append(inner_var)

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        parent_block.append_op(
            type='while',
            inputs={
                'X': [
                    parent_block._var_recursive(x_name)
                    for x_name in x_name_list
                ],
                'Condition': [self.cond_var]
            },
            outputs={'Out': out_vars,
                     'StepScopes': [step_scope]},
            attrs={'sub_block': while_block,
                   "is_test": self.is_test})


def assign_skip_lod_tensor_array(input, output):
    """
    Assign input to output, but skip the process of copying LoDTensorArray unless it's created in while_block.
    """
    if not isinstance(input, Variable) and not isinstance(input, core.VarBase):
        output = input
        return

    if input.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        main_program = input.block.program
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)
        if parent_block and not parent_block._find_var_recursive(input.name):
            assign(input, output)
    else:
        assign(input, output)


def while_loop(cond, body, loop_vars, is_test=False, name=None):
    """
    :api_attr: Static Graph

    while_loop is one of the control flows. Repeats while_loop `body` until `cond` returns False.

    Notice:
        Local variables defined in ``body`` cannot be obtained through ``fetch_list`` of ``Executor`` , variables should
        be defined outside ``body`` and placed in ``loop_vars`` for looping, then these variables can be fetched by ``fetch_list`` .

    Args:
        cond(Callable): A callable returning a boolean tensor controlling whether to continue looping. And ``cond`` takes
	    as many arguments as ``loop_vars`` .
        body(Callable): A callable returning a tuple or list of tensors or LoDTensorArrays of the same arity
            (length and structure) and types as ``loops_vars`` . And ``body`` takes as many arguments as ``loop_vars`` .
        loop_vars(list|tuple): A list or tuple of tensors or LoDTensorArrays that is passed to both ``cond`` and ``body`` .
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default value is False.
        name(str, optional): Normally there is no need for users to set this property. For more information, please
            refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        A list or tuple of Tensors or LoDTensorArrays which returned by ``body`` .

    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()

            def cond(i, ten):
                return i < ten

            def body(i, ten):
                i = i + 1
                return [i, ten]

            main_program = paddle.static.default_main_program()
            startup_program = paddle.static.default_startup_program()
            with paddle.static.program_guard(main_program, startup_program):
                i = paddle.full(shape=[1], fill_value=0, dtype='int64')     # loop counter
                ten = paddle.full(shape=[1], fill_value=10, dtype='int64')  # loop length
                i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])
                
                exe = paddle.static.Executor(paddle.CPUPlace())
                res = exe.run(main_program, feed={}, fetch_list=[i])
                print(res) # [array([10])]
    """
    helper = LayerHelper('while_loop', **locals())

    if not callable(cond):
        raise TypeError("cond in while_loop should be callable")
    if not callable(body):
        raise TypeError("body in while_loop should be callable")
    check_type(loop_vars, 'loop_vars', (list, tuple), 'fluid.layers.while_loop')
    if len(loop_vars) == 0:
        raise ValueError("loop_vars in while_loop should not be empty")

    pre_cond = cond(*loop_vars)
    check_variable_and_dtype(pre_cond, 'var of cond returned', ['bool'],
                             'fluid.layers.while_loop')
    if reduce(lambda a, b: a * b, pre_cond.shape, 1) != 1:
        raise TypeError(
            "the shape of the variable returned by cond should be [1],"
            "but given shape as {0}.".format(list(pre_cond.shape)))

    if in_dygraph_mode():
        now_cond = pre_cond.numpy()[0]
        while (now_cond):
            output_vars = body(*loop_vars)
            if not isinstance(output_vars, (list, tuple)):
                output_vars = [output_vars]
            if len(output_vars) != len(loop_vars):
                raise ValueError(
                    "body in while_loop should return the same arity "
                    "(length and structure) and types as loop_vars")
            now_cond = cond(*output_vars).numpy()[0]
            map_structure(assign_skip_lod_tensor_array, output_vars, loop_vars)
        return loop_vars

    while_loop_block = While(pre_cond, is_test, name)
    has_mutable_vars_in_loop = hold_mutable_vars(loop_vars)
    with while_loop_block.block():
        # If a variable with mutable type is included in loop_vars, like `dict/list`,
        # modifying it in the body function will cause origin variable to be modified
        # synchronously. This will raise an assignment error out of while block.
        # Here we make a copy of the mutable vars to avoid this problem.
        if has_mutable_vars_in_loop:
            new_loop_vars = copy_mutable_vars(loop_vars)
            output_vars = body(*new_loop_vars)
        else:
            output_vars = body(*loop_vars)
        if not isinstance(output_vars, (list, tuple)):
            output_vars = [output_vars]
        try:
            assert_same_structure(output_vars, loop_vars, check_types=False)
        except ValueError as e:
            raise ValueError("body in while_loop should return the same arity "
                             "(length and structure) as loop_vars: {0}".format(
                                 e))
        now_cond = cond(*output_vars)
        map_structure(assign_skip_lod_tensor_array, output_vars, loop_vars)
        assign(now_cond, pre_cond)
    return loop_vars


def lod_rank_table(x, level=0):
    """
    LoD Rank Table Operator. Given an input variable **x** and a level number
    of LoD, this layer creates a LodRankTable object. A LoDRankTable object
    contains a list of bi-element tuples. Each tuple consists of an index and
    a length, both of which are int type. Refering to specified level of LoD,
    the index is the sequence index number and the length represents the
    sequence length. Please note that the list is ranked in descending order by
    the length. The following is an example:

        .. code-block:: text

            x is a LoDTensor:
                x.lod = [[2,                1],
                         [5,             1, 1]]
                x.data = [a, b, c, d, e, f, g]

            1. set level to 0:
                Create lod rank table:
                    lod_rank_table_obj = lod_rank_table(x, level=0)

                Get:
                    lod_rank_table_obj.items() = [(0, 2), (1, 1)]

            2. set level to 1:
                Create lod rank table:
                    lod_rank_table_obj = lod_rank_table(x, level=1)

                Get:
                    lod_rank_table_obj.items() = [(0, 5), (1, 1), (2, 1)]

    Args:
        x (Variable): Input variable, a LoDTensor based which to create the lod
            rank table.
        level (int): Specify the LoD level, on which to create the lod rank
            table.

    Returns:
        Variable: The created LoDRankTable object.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[10],
                                  dtype='float32', lod_level=1)
            out = layers.lod_rank_table(x=x, level=0)
    """
    check_type(x, 'x', (Variable, list), 'lod_rank_table')
    if isinstance(x, (list)):
        for i, input_x in enumerate(x):
            check_type(input_x, 'input[' + str(i) + ']', Variable,
                       'lod_rank_table')

    helper = LayerHelper("lod_rank_table", **locals())
    table = helper.create_variable(
        type=core.VarDesc.VarType.LOD_RANK_TABLE,
        name=unique_name.generate("lod_rank_table"))
    helper.append_op(
        type='lod_rank_table',
        inputs={'X': x},
        outputs={'Out': table},
        attrs={'level': level})
    return table


@templatedoc()
def max_sequence_len(rank_table):
    """
    ${comment}

    >>> import paddle.fluid as fluid
    >>> x = fluid.layers.data(name='x', shape=[10], dtype='float32',
    >>>                       lod_level=1)
    >>> rank_table = layers.lod_rank_table(x=x, level=0)
    >>> max_seq_len = layers.max_sequence_len(rank_table)

    Args:
        rank_table(${rank_table_type}): ${rank_table_comment}.

    Returns:
        ${out_comment}.
    """
    helper = LayerHelper("max_seqence_len", **locals())
    res = helper.create_variable_for_type_inference(dtype="int64")
    helper.append_op(
        type="max_sequence_len",
        inputs={"RankTable": rank_table},
        outputs={"Out": res})
    return res


def lod_tensor_to_array(x, table):
    """
    Convert a LoDTensor to a LoDTensorArray.

    This function split a LoDTesnor to a LoDTensorArray according to its LoD
    information. LoDTensorArray is an alias of C++ std::vector<LoDTensor> in
    PaddlePaddle. The generated LoDTensorArray of this function can be further read
    or written by `read_from_array()` and `write_to_array()` operators. However,
    this function is generally an internal component of PaddlePaddle `DynamicRNN`.
    Users should not use it directly.

    Args:
        x (Variable|list): The LoDTensor to be converted to a LoDTensorArray.
        table (ParamAttr|list): The variable that stores the level of lod
                                which is ordered by sequence length in
                                descending order. It is generally generated
                                by `layers.lod_rank_table()` API.

    Returns:
        Variable: The LoDTensorArray that has been converted from the input tensor.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = fluid.layers.data(name='x', shape=[10])
          table = fluid.layers.lod_rank_table(x, level=0)
          array = fluid.layers.lod_tensor_to_array(x, table)
    """
    check_type(x, 'x', (Variable, list), 'lod_tensor_to_array')
    if isinstance(x, (list)):
        for i, input_x in enumerate(x):
            check_type(input_x, 'input[' + str(i) + ']', Variable,
                       'lod_tensor_to_array')
    check_type(table, 'table', (Variable, list), 'lod_tensor_to_array')
    if isinstance(table, (list)):
        for i, table_x in enumerate(table):
            check_type(table_x, 'table[' + str(i) + ']', Variable,
                       'lod_tensor_to_array')
    helper = LayerHelper("lod_tensor_to_array", **locals())
    array = helper.create_variable(
        name=unique_name.generate("lod_tensor_to_array"),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=x.dtype)
    helper.append_op(
        type='lod_tensor_to_array',
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': array})
    return array


def array_to_lod_tensor(x, table):
    """Convert a LoD_Tensor_Aarry to an LoDTensor.

    Args:
        x (Variable|list): The lod tensor array to be converted to a tensor.
        table (ParamAttr|list): The variable that stores the level of lod
                                which is ordered by sequence length in
                                descending order.

    Returns:
        Variable: The variable of type tensor that has been converted
                  from an array.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          x = fluid.layers.data(name='x', shape=[10])
          table = fluid.layers.lod_rank_table(x, level=0)
          array = fluid.layers.lod_tensor_to_array(x, table)
          lod_tensor = fluid.layers.array_to_lod_tensor(array, table)
    """
    check_type(x, 'x', (Variable, list), 'array_to_lod_tensor')
    if isinstance(x, (list)):
        for i, input_x in enumerate(x):
            check_type(input_x, 'input[' + str(i) + ']', Variable,
                       'array_to_lod_tensor')
    check_type(table, 'table', (Variable, list), 'array_to_lod_tensor')
    if isinstance(table, (list)):
        for i, table_x in enumerate(table):
            check_type(table_x, 'table[' + str(i) + ']', Variable,
                       'array_to_lod_tensor')

    helper = LayerHelper("array_to_lod_tensor", **locals())
    tmp = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="array_to_lod_tensor",
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': tmp})
    return tmp


def increment(x, value=1.0, in_place=True):
    """
    The OP is usually used for control flow to increment the data of :attr:`x` by an amount :attr:`value`.
    Notice that the number of elements in :attr:`x` must be equal to 1.

    Parameters:
        x (Variable): A tensor that must always contain only one element, its data type supports
            float32, float64, int32 and int64.
        value (float, optional): The amount to increment the data of :attr:`x`. Default: 1.0.
        in_place (bool, optional): Whether the OP should be performed in-place. Default: True.

    Returns:
        Variable: The elementwise-incremented tensor with the same shape and data type as :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          counter = fluid.layers.zeros(shape=[1], dtype='float32') # [0.]
          fluid.layers.increment(counter) # [1.]
    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'increment')
    helper = LayerHelper("increment", **locals())
    if not in_place:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = x
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'step': float(value)})
    return out


def array_write(x, i, array=None):
    """
    This OP writes the input ``x`` into the i-th position of the ``array``
    :ref:`api_fluid_LoDTensorArray` and returns the modified array.
    If ``array`` is none, a new LoDTensorArray will be created and returned.
    This OP is often used together with :ref:`api_fluid_layers_array_read` OP.

    Args:
        x (Variable): The input data to be written into array. It's multi-dimensional
            Tensor or LoDTensor. Data type: float32, float64, int32, int64.
        i (Variable): 1-D Tensor with shape [1], which represents the position into which
            ``x`` is written. Data type: int64.
        array (LoDTensorArray, optional): The LoDTensorArray into which ``x`` is written. 
            The default value is None, when a new LoDTensorArray will be created and returned 
            as a result.

    Returns:
        Variable: The input ``array`` after ``x`` is written into.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            tmp = fluid.layers.fill_constant(shape=[3, 2], dtype='int64', value=5)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
            # Write tmp into the position of arr with subscript 10 and return arr.
            arr = fluid.layers.array_write(tmp, i=i)

            # Now, arr is a LoDTensorArray with length 11. We can use array_read OP to read
            # the data at subscript 10 and print it out.
            item = fluid.layers.array_read(arr, i=i)
            input = fluid.layers.Print(item, message="The content of i-th LoDTensor:")
            main_program = fluid.default_main_program()
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(main_program)

            # The printed result is:
            # 1570533133    The content of i-th LoDTensor:  The place is:CPUPlace
            # Tensor[array_read_0.tmp_0]
            #    shape: [3,2,]
            #    dtype: l
            #    data: 5,5,5,5,5,5,

            # the output is 2-D Tensor with shape [3,2], which is tmp above.
            # dtype is the corresponding C++ data type, which may vary in different environments.
            # Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t, 
            #       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux, 
            #       and '__int64' on Windows. They both represent 64-bit integer variables.

    """
    if in_dygraph_mode():
        assert isinstance(
            x, Variable
        ), "The input data 'x' in array_write must be Variable in dygraph mode"
        assert isinstance(
            i, Variable
        ), "The index 'i' in array_write must be Variable in dygraph mode"
        assert i.shape == [
            1
        ], "The shape of index 'i' should be [1] in dygraph mode"
        i = i.numpy().item(0)
        if array is None:
            array = create_array(x.dtype)
        assert isinstance(
            array,
            list), "The 'array' in array_write must be a list in dygraph mode"
        assert i <= len(
            array
        ), "The index 'i' should not be greater than the length of 'array' in dygraph mode"
        if i < len(array):
            array[i] = x
        else:
            array.append(x)
        return array

    check_variable_and_dtype(i, 'i', ['int64'], 'array_write')
    check_type(x, 'x', (Variable), 'array_write')
    helper = LayerHelper('array_write', **locals())
    if array is not None:
        if not isinstance(
                array,
                Variable) or array.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
            raise TypeError(
                "array should be tensor array vairable in array_write Op")
    if array is None:
        array = helper.create_variable(
            name="{0}.out".format(helper.name),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.dtype)
    helper.append_op(
        type='write_to_array',
        inputs={'X': [x],
                'I': [i]},
        outputs={'Out': [array]})
    return array


def create_array(dtype, initialized_list=None):
    """
    This OP creates an LOD_TENSOR_ARRAY. It is used as
    the input of :ref:`api_fluid_layers_array_read` and 
    :ref:`api_fluid_layers_array_write`. Also it can be used
    with  :ref:`api_fluid_layers_While` to create RNN network.

    Args:
        dtype (str): The data type of the elements in the lod_tensor_array.
                     Support data type: float32, float64, int32, int64.
        initialized_list(list): Used to initialize as default value for created array.
                    All values in initialized list should be a Tensor.

    Returns:
        Variable: The empty lod_tensor_array. The data type of elements in Tensor is ``dtype``.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.create_array(dtype='float32') # Create a float32 LoDTensorArray.

    """
    array = []
    if initialized_list is not None:
        if not isinstance(initialized_list, (list, tuple)):
            raise TypeError(
                "Require type(initialized_list) should be list/tuple, but received {}".
                format(type(initialized_list)))
        array = list(initialized_list)

    # NOTE: Only support plain list like [x, y,...], not support nested list in static mode.
    for val in array:
        if not isinstance(val, Variable):
            raise TypeError(
                "All values in `initialized_list` should be Variable, but recevied {}.".
                format(type(val)))

    if in_dygraph_mode():
        return array

    helper = LayerHelper("array", **locals())
    tensor_array = helper.create_variable(
        name="{0}.out".format(helper.name),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=dtype)

    for val in array:
        array_write(x=val, i=array_length(tensor_array), array=tensor_array)

    return tensor_array


@templatedoc()
def less_than(x, y, force_cpu=None, cond=None, name=None):
    """

    ${comment}

    Args:
        x(Tensor): ${x_comment}.
        y(Tensor): ${y_comment}.
        force_cpu(${force_cpu_type}): ${force_cpu_comment}.
        cond(Tensor, optional): Optional output which can be any created Tensor
            that meets the requirements to store the result of *less_than*.
            if cond is None, a new Tensor will be created to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        ${out_comment}.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3, 4], dtype='float32')
            y = paddle.to_tensor([2, 2, 1, 3], dtype='float32')
            result = paddle.less_than(x, y)
            print(result) # [True, False, False, False]

    """
    check_variable_and_dtype(x, "x", ["float32", "float64", "int32", "int64"],
                             "less_than")
    check_variable_and_dtype(y, "y", ["float32", "float64", "int32", "int64"],
                             "less_than")
    if cond is not None:
        check_type(cond, "cond", Variable, "less_than")
    if force_cpu != None:
        check_type(force_cpu, "force_cpu", bool, "less_than")

    helper = LayerHelper("less_than", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()
    if force_cpu is not None:
        attrs['force_cpu'] = force_cpu

    helper.append_op(
        type='less_than',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


@templatedoc()
def less_equal(x, y, cond=None, name=None):
    """
    :alias_main: paddle.less_equal
	:alias: paddle.less_equal,paddle.tensor.less_equal,paddle.tensor.logic.less_equal
	:old_api: paddle.fluid.layers.less_equal

    This OP returns the truth value of :math:`x <= y` elementwise, which is equivalent function to the overloaded operator `<=`.

    Args:
        x(Variable): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64. 
        y(Variable): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        cond(Variable, optional): Optional output which can be any created Variable that meets the requirements to store the result of *less_equal*.
            if cond is None, a new Varibale will be created to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable, the output data type is bool: The tensor variable storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          label = fluid.layers.assign(np.array([1, 3], dtype='int32'))
          limit = fluid.layers.assign(np.array([1, 2], dtype='int32'))
          out = fluid.layers.less_equal(x=label, y=limit) #out=[True, False]
          out1 = label<= limit #out1=[True, False]

    """
    check_variable_and_dtype(x, "x", ["float32", "float64", "int32", "int64"],
                             "less_equal")
    check_variable_and_dtype(y, "y", ["float32", "float64", "int32", "int64"],
                             "less_equal")
    if cond is not None:
        check_type(cond, "cond", Variable, "less_equal")

    helper = LayerHelper("less_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='less_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


@templatedoc()
def greater_than(x, y, cond=None, name=None):
    """
    :alias_main: paddle.greater_than
	:alias: paddle.greater_than,paddle.tensor.greater_than,paddle.tensor.logic.greater_than
	:old_api: paddle.fluid.layers.greater_than

    This OP returns the truth value of :math:`x > y` elementwise, which is equivalent function to the overloaded operator `>`.

    Args:
        x(Variable): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64. 
        y(Variable): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        cond(Variable, optional): Optional output which can be any created Variable that meets the requirements to store the result of *greater_than*.
            if cond is None, a new Varibale will be created to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable, the output data type is bool: The tensor variable storing the output, the output shape is same as input :attr:`x` .

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          label = fluid.layers.assign(np.array([2, 3], dtype='int32'))
          limit = fluid.layers.assign(np.array([3, 2], dtype='int32'))
          out = fluid.layers.greater_than(x=label, y=limit) #out=[False, True]
          out1 = label > limit #out1=[False, True]
    """
    check_variable_and_dtype(x, "x", ["float32", "float64", "int32", "int64"],
                             "greater_than")
    check_variable_and_dtype(y, "y", ["float32", "float64", "int32", "int64"],
                             "greater_than")
    if cond is not None:
        check_type(cond, "cond", Variable, "greater_than")

    helper = LayerHelper("greater_than", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='greater_than',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


@templatedoc()
def greater_equal(x, y, cond=None, name=None):
    """
    :alias_main: paddle.greater_equal
	:alias: paddle.greater_equal,paddle.tensor.greater_equal,paddle.tensor.logic.greater_equal
	:old_api: paddle.fluid.layers.greater_equal

    This OP returns the truth value of :math:`x >= y` elementwise, which is equivalent function to the overloaded operator `>=`.

    Args:
        x(Variable): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64. 
        y(Variable): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        cond(Variable, optional): Optional output which can be any created Variable that meets the requirements to store the result of *greater_equal*.
            if cond is None, a new Varibale will be created to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable, the output data type is bool: The tensor variable storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          label = fluid.layers.assign(np.array([2, 2], dtype='int32'))
          limit = fluid.layers.assign(np.array([2, 3], dtype='int32'))
          out = fluid.layers.greater_equal(x=label, y=limit) #out=[True, False]
          out_1 = label >= limit #out1=[True, False]

    """
    check_variable_and_dtype(x, "x", ["float32", "float64", "int32", "int64"],
                             "greater_equal")
    check_variable_and_dtype(y, "y", ["float32", "float64", "int32", "int64"],
                             "greater_equal")
    if cond is not None:
        check_type(cond, "cond", Variable, "greater_equal")

    helper = LayerHelper("greater_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='greater_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


def equal(x, y, cond=None, name=None):
    """
    This layer returns the truth value of :math:`x == y` elementwise.

    Args:
        x(Variable): Tensor, data type is float32, float64, int32, int64.
        y(Variable): Tensor, data type is float32, float64, int32, int64.
        cond(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of *equal*.
            if cond is None, a new Varibale will be created to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: output Tensor, it's shape is the same as the input's Tensor,
        and the data type is bool.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          out_cond =fluid.data(name="input1", shape=[2], dtype='bool')
          label = fluid.layers.assign(np.array([3, 3], dtype="int32"))
          limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
          label_cond = fluid.layers.assign(np.array([1, 2], dtype="int32"))
          out1 = fluid.layers.equal(x=label,y=limit) #out1=[True, False]
          out2 = fluid.layers.equal(x=label_cond,y=limit, cond=out_cond) #out2=[False, True] out_cond=[False, True]
    """
    check_variable_and_dtype(x, "x", ["float32", "float64", "int32", "int64"],
                             "equal")
    check_variable_and_dtype(y, "y", ["float32", "float64", "int32", "int64"],
                             "equal")
    if cond is not None:
        check_type(cond, "cond", Variable, "equal")

    helper = LayerHelper("equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    helper.append_op(
        type='equal', inputs={'X': [x],
                              'Y': [y]}, outputs={'Out': [cond]})
    return cond


def not_equal(x, y, cond=None, name=None):
    """
    :alias_main: paddle.not_equal
	:alias: paddle.not_equal,paddle.tensor.not_equal,paddle.tensor.logic.not_equal
	:old_api: paddle.fluid.layers.not_equal

    This OP returns the truth value of :math:`x != y` elementwise, which is equivalent function to the overloaded operator `!=`.

    Args:
        x(Variable): First input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64. 
        y(Variable): Second input to compare which is N-D tensor. The input data type should be float32, float64, int32, int64.
        cond(Variable, optional): Optional output which can be any created Variable that meets the requirements to store the result of *not_equal*.
            if cond is None, a new Varibale will be created to store the result.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable, the output data type is bool: The tensor variable storing the output, the output shape is same as input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          
          label = fluid.layers.data(name='label', shape=[1], dtype='int64')
          limit = fluid.layers.fill_constant(shape=[1], value=1, dtype='int64')
          out = fluid.layers.not_equal(x=label, y=limit)
    """
    check_variable_and_dtype(x, "x", ["float32", "float64", "int32", "int64"],
                             "not_equal")
    check_variable_and_dtype(y, "y", ["float32", "float64", "int32", "int64"],
                             "not_equal")
    if cond is not None:
        check_type(cond, "cond", Variable, "not_equal")

    helper = LayerHelper("not_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    helper.append_op(
        type='not_equal', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [cond]})
    return cond


def array_read(array, i):
    """
    This OP is used to read data at the specified position from the input array 
    :ref:`api_fluid_LoDTensorArray` . ``array`` is the input array and ``i``
    is the specified read position. This OP is often used together with 
    :ref:`api_fluid_layers_array_write` OP.

    Case 1:
    ::
        Input:
            The shape of first three tensors are [1], and that of the last one is [1,2]:
                array = ([0.6], [0.1], [0.3], [0.4, 0.2])
            And:
                i = [3]

        Output:
            output = [0.4, 0.2]

    Args:
        array (LoDTensorArray): The input LoDTensorArray.
        i (Variable): 1-D Tensor, whose shape is [1] and dtype is int64. It represents the
            specified read position of ``array``.

    Returns:
        Variable: The LoDTensor or Tensor that is read at the specified position of ``array``.

    Examples:
        .. code-block:: python

            # First we're going to create a LoDTensorArray, then we're going to write the Tensor into
            # the specified position, and finally we're going to read the Tensor at that position.
            import paddle.fluid as fluid
            arr = fluid.layers.create_array(dtype='float32')
            tmp = fluid.layers.fill_constant(shape=[3, 2], dtype='int64', value=5)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
            # tmp is the Tensor with shape [3,2], and if we write it into the position with subscript 10
            # of the empty-array: arr, then the length of arr becomes 11.
            arr = fluid.layers.array_write(tmp, i, array=arr)
            # Read the data of the position with subscript 10.
            item = fluid.layers.array_read(arr, i)

            # You can print out the data via executor.
            input = fluid.layers.Print(item, message="The LoDTensor of the i-th position:")
            main_program = fluid.default_main_program()
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(main_program)

            # The printed result is:

            # 1569588169  The LoDTensor of the i-th position: The place is:CPUPlace
            # Tensor[array_read_0.tmp_0]
            #    shape: [3,2,]
            #    dtype: l
            #    data: 5,5,5,5,5,5,

            # the output is 2-D Tensor with shape [3,2].
            # dtype is the corresponding C++ data type, which may vary in different environments.
            # Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t, 
            #       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux, 
            #       and '__int64' on Windows. They both represent 64-bit integer variables.
    """
    if in_dygraph_mode():
        assert isinstance(
            array,
            list), "The 'array' in array_read must be list in dygraph mode"
        assert isinstance(
            i, Variable
        ), "The index 'i' in array_read must be Variable in dygraph mode"
        assert i.shape == [
            1
        ], "The shape of index 'i' should be [1] in dygraph mode"
        i = i.numpy().item(0)
        return array[i]

    check_variable_and_dtype(i, 'i', ['int64'], 'array_read')
    helper = LayerHelper('array_read', **locals())
    if not isinstance(
            array,
            Variable) or array.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        raise TypeError("array should be tensor array vairable")
    out = helper.create_variable_for_type_inference(dtype=array.dtype)
    helper.append_op(
        type='read_from_array',
        inputs={'X': [array],
                'I': [i]},
        outputs={'Out': [out]})
    return out


def shrink_memory(x, i, table):
    """
    This function creates an operator to shrink rnn memory using the RankTable
    as mentioned in the input parameter.

    NOTE: This API is very low-level API. It is used by DynamicRNN only.

    Since the Dynamic RNN uses no-padding way to implement RNN. The sequence
    will be sorted by order, and the length of valid memory will be shrink after
    each time step.

    Args:
        x(Variable): The memory object in the previous time step.
        i(Variable): The step count variable. A int scalar as LoDTensor.
        table(Variable): The RNNRankTable object.

    Returns:
        the memory variable after shrink.

    Examples:

        Since this API is very low level API. The example is not provided.
        Please reference the implementation of class DynamicRNN for detail
        usage.
    """
    helper = LayerHelper('shrink_memory', **locals())
    check_type(x, 'x', Variable, 'shrink_memory')
    check_type(i, 'i', Variable, 'shrink_memory')
    check_type(table, 'table', Variable, 'shrink_memory')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='shrink_rnn_memory',
        inputs={'X': [x],
                'I': [i],
                'RankTable': [table]},
        outputs={'Out': [out]},
        attrs={})
    return out


def array_length(array):
    """
    This OP is used to get the length of the input array :ref:`api_fluid_LoDTensorArray` .
    It can be used together with :ref:`api_fluid_layers_array_read` , :ref:`api_fluid_layers_array_write` , 
    :ref:`api_fluid_layers_While` OP to traverse, read and write LoDTensorArray.

    Args:
        array (LoDTensorArray): The input array that will be used to compute the length.

    Returns:
        Variable: 1-D Tensor with shape [1], which is the length of array. Datatype: int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            tmp = fluid.layers.zeros(shape=[10], dtype='int32')
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
            # tmp is 1-D Tensor with shape [10]. We write tmp into arr on subscript 10,
            # then the length of arr becomes 11.
            arr = fluid.layers.array_write(tmp, i=i)
            # return the length of arr
            arr_len = fluid.layers.array_length(arr)

            # You can use executor to print out the length of LoDTensorArray.
            input = fluid.layers.Print(arr_len, message="The length of LoDTensorArray:")
            main_program = fluid.default_main_program()
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(main_program)

            # The printed result is:

            # 1569576542  The length of LoDTensorArray:   The place is:CPUPlace
            # Tensor[array_length_0.tmp_0]
            #    shape: [1,]
            #    dtype: l
            #    data: 11,
            
            # 1-D Tensor with shape [1], whose value is 11. It means that the length of LoDTensorArray
            # is 11.
            # dtype is the corresponding C++ data type, which may vary in different environments.
            # Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t, 
            #       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux, 
            #       and '__int64' on Windows. They both represent 64-bit integer variables.
    """

    if in_dygraph_mode():
        assert isinstance(
            array,
            list), "The 'array' in array_write must be a list in dygraph mode"
        return len(array)

    if not isinstance(
            array,
            Variable) or array.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        raise TypeError(
            "array should be tensor array vairable in array_length Op")

    helper = LayerHelper('array_length', **locals())
    tmp = helper.create_variable_for_type_inference(dtype='int64')
    tmp.stop_gradient = True
    helper.append_op(
        type='lod_array_length', inputs={'X': [array]}, outputs={'Out': [tmp]})
    return tmp


class ConditionalBlockGuard(BlockGuard):
    """
    ConditionalBlockGuard is derived from BlockGuard. It is dedicated for
    holding a ConditionalBlock, and helping users entering and exiting the
    ConditionalBlock via Python's 'with' keyword. However, ConditionalBlockGuard
    is generally an internal component of IfElse, users should not use it directly.
    """

    def __init__(self, block):
        check_type(block, "block", ConditionalBlock, "ConditionalBlockGuard")
        super(ConditionalBlockGuard, self).__init__(block.helper.main_program)
        self.block = block

    def __enter__(self):
        return super(ConditionalBlockGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.block.complete()
        return super(ConditionalBlockGuard, self).__exit__(exc_type, exc_val,
                                                           exc_tb)


class ConditionalBlock(object):
    '''
    **ConditionalBlock**

    ConditionalBlock is an operator that bind a block to a specific condition,
    if the condition matches, the corresponding block will be executed.

    Args:
        inputs (Variable): bool conditions.
        is_scalar_condition (bool): whether the branch is controlled by a scalar.
        name(str): name of this ConditionalBlock.

    Examples:
        .. code-block:: python

             import paddle.fluid as fluid
             cond = layers.less_than(x=label, y=limit)
             true_image, false_image = layers.split_lod_tensor(
                 input=image, mask=cond)
             true_cond = layers.ConditionalBlock([true_image])

             with true_cond.block():
                 ...
             with false_cond.block():
                 ...
    '''

    def __init__(self, inputs, is_scalar_condition=False, name=None):
        for each_input in inputs:
            check_type(each_input, "input", Variable, "ConditionalBlock")
        self.inputs = inputs
        self.is_scalar_condition = is_scalar_condition
        self.helper = LayerHelper('conditional_block', name=name)

    def block(self):
        return ConditionalBlockGuard(self)

    def complete(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

        intermediate = set()
        params = set()
        params, intermediate = get_inputs_outputs_in_block(
            inside_block, params, intermediate, helper=self.helper)

        # Todo(liym27) Here assume that all params are in recursive parent block
        # but when minimize() called in control flow, some params may be in
        # conditional grad block
        param_list = [
            parent_block._var_recursive(each_name) for each_name in params
        ]

        out_list = []
        for inner_out_name in intermediate:
            inner_var = parent_block._find_var_recursive(inner_out_name)
            if inner_var:
                out_list.append(inner_var)

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)
        conditional_block_op = parent_block.append_op(
            type='conditional_block',
            inputs={
                'Cond': self.inputs,
                'Input': param_list,
            },
            outputs={'Out': out_list,
                     'Scope': [step_scope]},
            attrs={
                'sub_block': inside_block,
                'is_scalar_condition': self.is_scalar_condition
            })

        if self.need_append_conditional_block_grad(inside_block):
            self.append_conditional_block_grad(parent_block, inside_block,
                                               conditional_block_op)

    def need_append_conditional_block_grad(self, inside_block):
        grad_sub_block_idx = inside_block.backward_block_idx
        inside_block_idx = inside_block.idx

        # if inside_block have grad_block and grad_block is not itself,
        # we will append conditional block grad.
        return grad_sub_block_idx != -1 and grad_sub_block_idx != inside_block_idx

    def append_conditional_block_grad(self, parent_block, inside_block,
                                      conditional_block_op):
        '''
        Append op `conditional_block_grad` manually.
        When `optimizer.minimize/append_backward` is called in Paddle control flow,
        grad ops will be appended before appending op `conditional_block` so that
        op `conditional_block_grad` can't be appended when calling
        `optimizer.minimize/append_backward`. After appending op `conditional_block`,
        `conditional_block_grad` is appended manually.

        Args:
            parent_block (Block): The block that `conditional_block_op` blongs to.
            inside_block (Block): The sub block of `conditional_block_op`.
            conditional_block_op (Operator): The forward op conditional_block.
        '''

        grad_sub_block_idx = inside_block.backward_block_idx
        grad_sub_block = self.helper.main_program.block(grad_sub_block_idx)

        intermediate = set()
        params = set()

        for each_op in grad_sub_block.ops:
            assert isinstance(each_op, Operator)
            for iname in each_op.input_names:
                for in_var_name in each_op.input(iname):
                    if in_var_name not in intermediate:
                        params.add(in_var_name)

            for oname in each_op.output_names:
                for out_var_name in each_op.output(oname):
                    intermediate.add(out_var_name)

        param_list = []
        for inner_input_name in params:
            inner_var = parent_block._find_var_recursive(inner_input_name)
            if inner_var:
                param_list.append(cpt.to_text(inner_var.name))

        grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
            conditional_block_op.desc,
            cpt.to_text(set()), [grad_sub_block.desc])

        # append op_desc in grad_op_descs to target_block
        op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        new_op_desc = parent_block.desc.append_op()
        new_op_desc.copy_from(grad_op_desc[0])
        new_op_desc._set_attr(op_role_attr_name, backward)
        # set input and output manually
        new_op_desc.set_input('Input', param_list)
        new_op_desc.set_output('Input@GRAD',
                               [param + "@GRAD" for param in param_list])

        new_vars = set()
        for grad_var_name in new_op_desc.output_arg_names():
            if grad_sub_block.desc.has_var_recursive(
                    cpt.to_bytes(grad_var_name)
            ) or grad_var_name == core.empty_var_name():
                continue
            grad_sub_block.desc.var(cpt.to_bytes(grad_var_name))
            new_vars.add(grad_var_name)
            if grad_var_name not in op_grad_to_var:
                continue

        # infer_shape and infer_type
        new_op_desc.infer_var_type(grad_sub_block.desc)
        new_op_desc.infer_shape(grad_sub_block.desc)

        for arg in new_op_desc.output_arg_names():
            if arg in new_vars:
                _infer_var_data_type_shape_(arg, grad_sub_block)

        self.helper.main_program._sync_with_cpp()


def copy_var_to_parent_block(var, layer_helper):
    if not isinstance(var, Variable):
        return var
    prog = layer_helper.main_program
    parent_idx = prog.current_block().parent_idx
    assert parent_idx >= 0, "Got wrong parent block index when assigning var to parent scope in control_flow"
    parent_block = prog.block(parent_idx)

    if var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY \
            and parent_block._find_var_recursive(var.name):
        parent_block_var = var
    else:
        parent_block_var = parent_block.create_var(
            dtype=var.dtype, shape=var.shape, type=var.type)
        assign(var, parent_block_var)
    return parent_block_var


def cond(pred, true_fn=None, false_fn=None, name=None):
    """
    This API returns ``true_fn()`` if the predicate ``pred`` is true else
    ``false_fn()`` . Users could also set ``true_fn`` or ``false_fn`` to
    ``None`` if do nothing and this API will treat the callable simply returns
    ``None`` in this case.

    ``true_fn`` and ``false_fn`` should return same nest structure of tensors
    or both return ``None`` if user doens't like to return anything. A nest
    structure of tensors in PaddlePaddle is tensor(s), or tuple of tensors, or
    list of tensors.
    
    Note: 
        1. The tuples or lists returned by ``true_fn`` and ``false_fn`` must have
        the same shape because of dataflow model of PaddlePaddle while the
        tensors in the tuples or the lists can have different shapes.

        2. This API could be used under both static mode or dygraph mode. If it
        is in dygraph mode, the API only runs one branch based on condition.

        3. If it is in static mode, any tensors or operations created outside 
        or inside of ``true_fn`` and ``false_fn`` will be in net building
        regardless of which branch is selected at runtime. This has frequently
        surprised users who expected a lazy semantics. For example:

        .. code-block:: python

            import paddle

            a = paddle.zeros((1, 1))
            b = paddle.zeros((1, 1))
            c = a * b
            out = paddle.static.nn.cond(a < b, lambda: a + c, lambda: b * b)

        No matter whether ``a < b`` , ``c = a * b`` will be in net building and
        run. ``a + c`` and ``b * b`` will be in net building, but only one
        branch will be executed during runtime.

    Args:
        pred(Tensor): A boolean tensor whose numel should be 1. The boolean
            value determines whether to return the result of ``true_fn`` or
            ``false_fn`` .
        true_fn(callable, optional): A callable to be performed if ``pred`` is
            true. The default value is ``None`` .
        false_fn(callable, optional): A callable to be performed if ``pred`` is
            false. The default value is ``None`` .
        name(str, optional): The default value is ``None`` . Normally users
             don't have to set this parameter. For more information, please
             refer to :ref:`api_guide_Name` .

    Returns:
        Tensor|list(Tensor)|tuple(Tensor): returns ``true_fn()`` if the
        predicate ``pred`` is true else ``false_fn()`` .

    Raises:
        TypeError: if ``true_fn`` or ``false_fn`` is not callable.
        ValueError: if ``true_fn`` and ``false_fn`` don't return the same nest
            structure of tensors.

    Examples:
        .. code-block:: python

            import paddle

            #
            # pseudocode:
            # if 0.1 < 0.23:
            #     return 1, True
            # else:
            #     return 3, 2
            #

            def true_func():
                return paddle.full(shape=[1, 2], dtype='int32',
                                   fill_value=1), paddle.full(shape=[2, 3],
                                                              dtype='bool',
                                                              fill_value=True)


            def false_func():
                return paddle.full(shape=[3, 4], dtype='float32',
                                   fill_value=3), paddle.full(shape=[4, 5],
                                                              dtype='int64',
                                                              fill_value=2)


            x = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
            y = paddle.full(shape=[1], dtype='float32', fill_value=0.23)
            pred = paddle.less_than(x=x, y=y, name=None)
            ret = paddle.static.nn.cond(pred, true_func, false_func)
            # ret is a tuple containing 2 tensors
            # ret[0] = [[1 1]]
            # ret[1] = [[ True  True  True]
            #           [ True  True  True]]            

    """
    if in_dygraph_mode():
        assert isinstance(pred, Variable), "The pred in cond must be Variable"
        assert pred.numpy().size == 1, "condition input's numel should be 1"
        pred = pred.numpy()[0]
        if pred:
            if true_fn is not None:
                if not callable(true_fn):
                    raise TypeError(
                        "The true_fn in cond must be callable, but received {}".
                        format(type(true_fn).__name__))
                return true_fn()
        else:
            if false_fn is not None:
                if not callable(false_fn):
                    raise TypeError(
                        "The false_fn in cond must be callable, but received {}".
                        format(type(false_fn).__name__))
                return false_fn()
        return None

    check_variable_and_dtype(pred, "pred", ['bool'], "fluid.layers.cond")
    check_type(name, "name", (str, type(None)), "fluid.layers.cond")
    helper = LayerHelper('cond', **locals())
    true_output = None
    false_output = None
    copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)
    if true_fn is not None:
        if not callable(true_fn):
            raise TypeError(
                "The true_fn in cond must be callable, but received {}".format(
                    type(true_fn).__name__))
        true_cond_block = ConditionalBlock([pred], is_scalar_condition=True)
        with true_cond_block.block():
            origin_true_output = true_fn()
            if origin_true_output is not None:
                true_output = map_structure(copy_to_parent_func,
                                            origin_true_output)
    if false_fn is not None:
        if not callable(false_fn):
            raise TypeError(
                "The false_fn in cond must be callable, but received {}".format(
                    type(false_fn).__name__))
        false_cond_block = ConditionalBlock(
            [logical_not(pred)], is_scalar_condition=True)
        with false_cond_block.block():
            origin_false_output = false_fn()
            if origin_false_output is not None:
                false_output = map_structure(copy_to_parent_func,
                                             origin_false_output)

    if true_output is None and false_output is None:
        return None

    if true_output is None:
        raise ValueError(
            "Incompatible return values of true_fn and false_fn in cond: "
            "true_fn returns None while false_fn returns non-None")
    if false_output is None:
        raise ValueError(
            "Incompatible return values of true_fn and false_fn in cond: "
            "true_fn returns non-None while false_fn returns None")

    # Merge ture and false output if they are not None
    try:
        assert_same_structure(true_output, false_output, check_types=False)
    except ValueError as e:
        raise ValueError(
            "Incompatible return values of true_fn and false_fn in cond: {}".
            format(e))

    mask = cast(pred, dtype='int32')
    merge_func = lambda false_var, true_var : select_input_with_buildin_type([false_var, true_var], mask)
    merged_output = map_structure(merge_func, false_output, true_output)
    return merged_output


def _error_message(what, arg_name, op_name, right_value, error_value):
    error_message = "{what} of '{arg_name}' in {op_name} must be " \
        "{right_value}, but received: {error_value}.".format(
        what=what,
        arg_name=arg_name,
        op_name=op_name,
        right_value=right_value,
        error_value=error_value)

    return error_message


def case(pred_fn_pairs, default=None, name=None):
    '''
    :api_attr: Static Graph

    This operator works like an if-elif-elif-else chain.

    Args:
        pred_fn_pairs(list|tuple): A list or tuple of (pred, fn) pairs. ``pred`` is a boolean Tensor with shape [1], ``fn`` is a callable. All callables return the same structure of Tensors.
        default(callable, optional): Callable that returns a structure of Tensors.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor|list(Tensor): Tensors returned by the callable from the first pair whose pred is True,
        or Tensors returned by ``default`` if no pred in ``pred_fn_pairs`` is True and ``default`` is not None,
        or Tensors returned by the last callable in ``pred_fn_pairs``  if no pred in ``pred_fn_pairs`` is True and ``default`` is None.

    Raises:
        TypeError: If the type of ``pred_fn_pairs`` is not list or tuple.
        TypeError: If the type of elements in ``pred_fn_pairs`` is not tuple.
        TypeError: If the size of tuples in ``pred_fn_pairs`` is not 2.
        TypeError: If the first element of 2-tuple in ``pred_fn_pairs`` is not a Tensor.
        TypeError: If the second element of 2-tuple in ``pred_fn_pairs`` is not callable.
        TypeError: If ``default`` is not None but it is not callable.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()

            def fn_1():
                return paddle.full(shape=[1, 2], dtype='float32', fill_value=1)

            def fn_2():
                return paddle.full(shape=[2, 2], dtype='int32', fill_value=2)

            def fn_3():
                return paddle.full(shape=[3], dtype='int32', fill_value=3)

            main_program = paddle.static.default_startup_program()
            startup_program = paddle.static.default_main_program()

            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.full(shape=[1], dtype='float32', fill_value=0.3)
                y = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
                z = paddle.full(shape=[1], dtype='float32', fill_value=0.2)

                pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
                pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
                pred_3 = paddle.equal(x, y)      # false: 0.3 == 0.1

                # Call fn_1 because pred_1 is True
                out_1 = paddle.static.nn.case(
                    pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)

                # Argument default is None and no pred in pred_fn_pairs is True. fn_3 will be called.
                # because fn_3 is the last callable in pred_fn_pairs.
                out_2 = paddle.static.nn.case(pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)])

                exe = paddle.static.Executor(paddle.CPUPlace())
                res_1, res_2 = exe.run(main_program, fetch_list=[out_1, out_2])
                print(res_1)  # [[1. 1.]]
                print(res_2)  # [3 3 3]
    '''
    helper = LayerHelper('case', **locals())

    def _case_check_args(pred_fn_pairs, default):
        '''
        Check arguments pred_fn_pairs and default. Return canonical pre_fn_pairs and default.
        '''
        check_type(pred_fn_pairs, 'pred_fn_pairs', (list, tuple), 'case')

        for pred_fn in pred_fn_pairs:
            if not isinstance(pred_fn, tuple):
                raise TypeError(
                    _error_message("The elements' type", "pred_fn_pairs",
                                   "case", tuple, type(pred_fn)))
            if len(pred_fn) != 2:
                raise TypeError(
                    _error_message("The tuple's size", "pred_fn_pairs", "case",
                                   "2", str(len(pred_fn)) + "-tuple"))
            pred, fn = pred_fn

            if not isinstance(pred, Variable):
                raise TypeError(
                    _error_message("The pred's type", "pred_fn_pairs", "case",
                                   "boolean Variable", type(pred)))

            if not callable(fn):
                raise TypeError(
                    "The fn for {} of pred_fn_pairs in Op(case) must"
                    " be callable.".format(pred.name))

        if default is None:
            default_index = len(pred_fn_pairs) - 1  # pick the last one
            default = pred_fn_pairs[default_index][1]
            pred_fn_pairs = pred_fn_pairs[:default_index]
        elif not callable(default):
            raise TypeError("The default in Op(case) must be callable.")

        return pred_fn_pairs, default

    pred_fn_pairs, default = _case_check_args(pred_fn_pairs, default)

    false_fn = default
    for pred, true_fn in reversed(pred_fn_pairs):
        false_fn = partial(cond, pred=pred, true_fn=true_fn, false_fn=false_fn)

    final_fn = false_fn

    return final_fn()


class Switch(object):
    """
    :api_attr: Static Graph

    This class is used to implement Switch branch control function. 
    Switch branch contains several case branches and one default branch. 
    Switch control flow checks whether the case branch conditions are satisfied in turn, 
    and only executes the statement after the first case branch that satisfies the conditions. 
    If there is no case branch that satisfies the condition, 
    only the statement following the default branch is executed.

    Note:
        A new OP :ref:`api_fluid_layers_case` is highly recommended instead of ``Switch`` if the shape of parameter ``cond`` is [1].
        OP :ref:`api_fluid_layers_case` is easier to use and is called with less code but does the same thing as ``Switch`` .

    Member Functions:
        case(condition): The case branch of Switch whose parameter cond is a scalar Variable of bool type. Only if the cond of the current case branch is True and the cond of the previous case branch is False, the statement after the case branch will be executed, and the statement after the case branch will not be executed.
        
        default(): The default branch of Switch. When cond of all case branches is False, the statement after default branch is executed.

    Case and default functions can only be used inside the scope of Switch, as shown below:

    .. code-block:: python

        '''
        with fluid.layers.Switch() as switch:
            with switch.case(cond1):
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
            with switch.case(cond2):
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
            with switch.default():
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        '''

    Args:
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid

            lr = fluid.layers.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=True,
                name="learning_rate")
            zero_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=0.0)
            one_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=1.0)
            two_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=2.0)

            global_step = fluid.layers.autoincreased_step_counter(counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)

            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(global_step == zero_var):
                    fluid.layers.assign(input=one_var, output=lr)
                with switch.default():
                    fluid.layers.assign(input=two_var, output=lr)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[lr])
            print(res) # [array([1.], dtype=float32)]
    """

    def __init__(self, name=None):
        self.helper = LayerHelper('switch', name=name)
        self.inside_scope = False
        self.pre_not_conditions = []

    def case(self, condition):
        if not self.inside_scope:
            raise ValueError("case should be called inside with")

        check_variable_and_dtype(
            condition, 'condition', ['bool'],
            'the member function case of fluid.layers.Switch')

        if len(self.pre_not_conditions) == 0:
            cond_block = ConditionalBlock([condition], is_scalar_condition=True)
            not_cond = logical_not(x=condition)
            self.pre_not_conditions.append(not_cond)
        else:
            pre_cond_num = len(self.pre_not_conditions)
            pre_not_cond = self.pre_not_conditions[pre_cond_num - 1]
            new_not_cond = logical_and(
                x=pre_not_cond, y=logical_not(x=condition))
            self.pre_not_conditions.append(new_not_cond)
            cond_block = ConditionalBlock(
                [logical_and(
                    x=pre_not_cond, y=condition)],
                is_scalar_condition=True)

        return ConditionalBlockGuard(cond_block)

    def default(self):
        pre_cond_num = len(self.pre_not_conditions)
        if pre_cond_num == 0:
            raise ValueError("there should be at least one condition")
        cond_block = ConditionalBlock(
            [self.pre_not_conditions[pre_cond_num - 1]],
            is_scalar_condition=True)
        return ConditionalBlockGuard(cond_block)

    def __enter__(self):
        """
        set flag that now is inside switch.block {}
        :return:
        """
        self.inside_scope = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inside_scope = False
        if exc_type is not None:
            return False  # re-raise exception

        return True


class IfElseBlockGuard(object):
    def __init__(self, is_true, ifelse):
        if not isinstance(ifelse, IfElse):
            raise TypeError("ifelse must be an instance of IfElse class")

        if ifelse.status != IfElse.OUT_IF_ELSE_BLOCKS:
            raise ValueError("You cannot invoke IfElse.block() inside a block")

        self.is_true = is_true
        self.ie = ifelse
        if is_true:
            self.cond_block = ifelse.conditional_true_block
        else:
            self.cond_block = ifelse.conditional_false_block

        if not isinstance(self.cond_block, ConditionalBlock):
            raise TypeError("Unexpected situation")

        self.cond_block = self.cond_block.block()

    def __enter__(self):
        self.ie.status = IfElse.IN_IF_ELSE_TRUE_BLOCKS if self.is_true else IfElse.IN_IF_ELSE_FALSE_BLOCKS
        self.cond_block.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cond_block.__exit__(exc_type, exc_val, exc_tb):
            # re-raise inside exception
            return False
        if len(self.ie.output_table[1 if self.is_true else 0]) == 0:
            raise ValueError("Must set output inside block")
        self.ie.status = IfElse.OUT_IF_ELSE_BLOCKS


class IfElse(object):
    """
    :api_attr: Static Graph

    This class is used to implement IfElse branch control function. IfElse contains two blocks, true_block and false_block. IfElse will put data satisfying True or False conditions into different blocks to run.

    Cond is a 2-D Tensor with shape [N, 1] and data type bool, representing the execution conditions of the corresponding part of the input data.

    Note:
        A new OP :ref:`api_fluid_layers_cond` is highly recommended instead of ``IfElse``. if the shape of parameter ``cond`` is [1].
        OP :ref:`api_fluid_layers_cond` is easier to use and is called with less code but does the same thing as ``IfElse`` .

    IfElse OP is different from other OPs in usage, which may cause some users confusion. Here is a simple example to illustrate this OP.

    .. code-block:: python
        
        # The following code completes the function: subtract 10 from the data greater than 0 in x, add 10 to the data less than 0 in x, and sum all the data.
        import numpy as np
        import paddle.fluid as fluid

        x = fluid.layers.data(name='x', shape=[4, 1], dtype='float32', append_batch_size=False)
        y = fluid.layers.data(name='y', shape=[4, 1], dtype='float32', append_batch_size=False)

        x_d = np.array([[3], [1], [-2], [-3]]).astype(np.float32)
        y_d = np.zeros((4, 1)).astype(np.float32)
        
        # Compare the size of x, y pairs of elements, output cond, cond is shape [4, 1], data type bool 2-D tensor.
        # Based on the input data x_d, y_d, it can be inferred that the data in cond are [[true], [true], [false], [false]].
        cond = fluid.layers.greater_than(x, y)
        # Unlike other common OPs, ie below returned by the OP is an IfElse OP object
        ie = fluid.layers.IfElse(cond)

        with ie.true_block():
            # In this block, according to cond condition, the data corresponding to true dimension in X is obtained and subtracted by 10.
            out_1 = ie.input(x)
            out_1 = out_1 - 10
            ie.output(out_1)
        with ie.false_block():
            # In this block, according to cond condition, get the data of the corresponding condition in X as false dimension, and add 10
            out_1 = ie.input(x)
            out_1 = out_1 + 10
            ie.output(out_1)

        # According to cond condition, the data processed in the two blocks are merged. The output here is output, the type is List, and the element type in List is Variable.
        output = ie() #  [array([[-7.], [-9.], [ 8.], [ 7.]], dtype=float32)] 

        # Get the first Variable in the output List and add all elements.
        out = fluid.layers.reduce_sum(output[0])

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        res = exe.run(fluid.default_main_program(), feed={"x":x_d, "y":y_d}, fetch_list=[out])
        print(res)
        # [array([-1.], dtype=float32)] 

    Args:
        cond (Variable): cond is a 2-D Tensor with shape [N, 1] and data type bool, representing the corresponding execution conditions of N input data. The data type is bool.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Unlike other common OPs, the OP call returns an IfElse OP object (e.g. ie in the example), which branches the input data by calling the internal functions of the object ``true_block ()``, ``false_block ()``, ``input ()``, ``output ()``, and integrates the data processed by different branches as the overall output by calling the internal ``call ()`` function. The output type is a list, and the type of each element in the list is Variable.

    Internal Functions:
        The block is constructed by calling the ``with ie. true_block()`` function in the object, and the computational logic under condition true is put into the block. If no corresponding block is constructed, the input data in the corresponding conditional dimension is unchanged.
 
        The block is constructed by calling the ``with ie. false_block()`` function in the object, and the computational logic under condition false is put into the block. If no corresponding block is constructed, the input data in the corresponding conditional dimension is unchanged.

        ``Out = ie. input (x)`` will take out the data of the corresponding conditional dimension in X and put it into out, supporting the internal processing of multiple inputs in block.

        ``ie. output (out)`` writes the result to the output of the corresponding condition.

        There is a ``call ()`` function inside the object, that is, by calling ``output = ie ()``, all the outputs inside the block of False are fused as the whole output, the output type is a list, and the type of each element in the list is Variable.

    """
    OUT_IF_ELSE_BLOCKS = 0
    IN_IF_ELSE_TRUE_BLOCKS = 1
    IN_IF_ELSE_FALSE_BLOCKS = 2

    def __init__(self, cond, name=None):
        check_type(cond, "cond", Variable, "fluid.layers.IfElse")
        check_type(name, "name", (str, type(None)), "fluid.layers.IfElse")
        self.helper = LayerHelper('ifelse', name=name)
        self.cond = cond
        self.input_table = {}
        self.status = IfElse.OUT_IF_ELSE_BLOCKS
        self.conditional_true_block = ConditionalBlock(inputs=[self.cond])
        self.conditional_false_block = ConditionalBlock(inputs=[self.cond])
        self.output_table = ([], [])  # (true_outs, false_outs)

    def input(self, x):
        if self.status == IfElse.OUT_IF_ELSE_BLOCKS:
            raise ValueError("input must in true/false blocks")
        if id(x) not in self.input_table:
            parent_block = self._parent_block()
            out_true = parent_block.create_var(
                name=unique_name.generate_with_ignorable_key('ifelse_input' +
                                                             self.helper.name),
                dtype=x.dtype)

            out_false = parent_block.create_var(
                name=unique_name.generate_with_ignorable_key('ifelse_input' +
                                                             self.helper.name),
                dtype=x.dtype)
            parent_block.append_op(
                type='split_lod_tensor',
                inputs={
                    'X': x,
                    'Mask': self.cond,
                },
                outputs={'OutTrue': out_true,
                         'OutFalse': out_false},
                attrs={'level': 0})
            self.input_table[id(x)] = (out_true, out_false)
        else:
            out_true, out_false = self.input_table[id(x)]

        if self.status == IfElse.IN_IF_ELSE_TRUE_BLOCKS:
            return out_true
        else:
            return out_false

    def _parent_block(self):
        current_block = self.helper.main_program.current_block()
        return self.helper.main_program.block(current_block.parent_idx)

    def true_block(self):
        return IfElseBlockGuard(True, self)

    def false_block(self):
        return IfElseBlockGuard(False, self)

    def output(self, *outs):
        if self.status == self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("output can only be invoked in the sub-block")

        out_table = self.output_table[1 if self.status ==
                                      self.IN_IF_ELSE_TRUE_BLOCKS else 0]
        parent_block = self._parent_block()
        for each_out in outs:
            check_type(each_out, "each output", Variable,
                       "fluid.layers.IfElse.output")
            # create outside tensor
            outside_out = parent_block.create_var(
                name=unique_name.generate_with_ignorable_key("_".join(
                    [self.helper.name, 'output'])),
                dtype=each_out.dtype)
            out_table.append(outside_out)

            # assign local var to outside
            assign(input=each_out, output=outside_out)

    def __call__(self):
        if self.status != self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("IfElse::__call__ must be out of sub-block")
        false_len, true_len = list(map(len, self.output_table))
        if false_len == 0 and true_len == 0:
            raise ValueError("Must invoke true_block/false_block before "
                             "__call__")
        elif false_len != true_len and false_len != 0 and true_len != 0:
            raise ValueError("The output side must be same")
        elif false_len == 0 or true_len == 0:
            return self.output_table[0 if false_len != 0 else 1]

        # else none of false_len/true_len is zero
        # merge together
        rlist = []
        for false_var, true_var in zip(*self.output_table):
            rlist.append(
                merge_lod_tensor(
                    in_true=true_var,
                    in_false=false_var,
                    mask=self.cond,
                    x=self.cond,
                    level=0))
        return rlist


class DynamicRNN(object):
    """
    :api_attr: Static Graph

    **Note: the input of this class should be LoDTensor which holds the
    information of variable-length sequences. If the input is fixed-length Tensor,
    please use StaticRNN (fluid.layers.** :ref:`api_fluid_layers_StaticRNN` **) for
    better performance.**

    DynamicRNN can process a minibatch of variable-length sequences.
    The length of each sample can be different and is recorded in LoD.
    In DynamicRNN, an input sequence will be unfolded into time steps and users
    can define how to process each time step in :code:`block()` .
    The total number of time steps is determined by the longest sequence.
    DynamicRNN will not pad all sequences to the same length, instead it will
    sort the sequences internally by the sequence length in descending order.
    The input sequences will be shrank because only sequences of which the
    length is larger than the time step will participate the remaining calculation.

    If defined :code:`drnn = DynamicRNN()`, then users can call :code:`drnn()`
    to obtain the result sequences. It is a LoDTensor gained by merging all
    time steps's output. When RNN's input sequence x meets :code:`x.lod_level == 1`,
    the output LoDTensor will have the same LoD with x. The result of :code:`drnn()`
    includes RNN's outputs of all time steps, users can call
    :ref:`api_fluid_layers_sequence_last_step` to extract the data of the last time step.

    Warning:
        Currently it is not supported to set :code:`is_sparse = True` of any
        layers defined within DynamicRNN's :code:`block` function.

    Args:
        name (str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information,
            please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)
            encoder_proj = fluid.data(name='encoder_proj', shape=[None, 32], dtype='float32', lod_level=1)
            decoder_boot = fluid.data(name='boot', shape=[None, 10], dtype='float32')

            drnn = fluid.layers.DynamicRNN()
            with drnn.block():
                # Set sentence as RNN's input, each time step processes a word from the sentence
                current_word = drnn.step_input(sentence)
                # Set encode_proj as RNN's static input
                encoder_word = drnn.static_input(encoder_proj)
                # Initialize memory with boot_memory, which need reorder according to RNN's input sequences
                memory = drnn.memory(init=decoder_boot, need_reorder=True)
                fc_1 = fluid.layers.fc(input=encoder_word, size=30)
                fc_2 = fluid.layers.fc(input=current_word, size=30)
                decoder_inputs = fc_1 + fc_2
                hidden, _, _ = fluid.layers.gru_unit(input=decoder_inputs, hidden=memory, size=30)
                # Update memory with hidden
                drnn.update_memory(ex_mem=memory, new_mem=hidden)
                out = fluid.layers.fc(input=hidden, size=10, bias_attr=True, act='softmax')
                # Set hidden and out as RNN's outputs
                drnn.output(hidden, out)

            # Get RNN's result
            hidden, out = drnn()
            # Get RNN's result of the last time step
            last = fluid.layers.sequence_last_step(out)
    """
    BEFORE_RNN = 0
    IN_RNN = 1
    AFTER_RNN = 2

    def __init__(self, name=None):
        self.helper = LayerHelper('dynamic_rnn', name=name)
        self.status = DynamicRNN.BEFORE_RNN
        self.lod_rank_table = None
        self.max_seq_len = None
        self.step_idx = None
        self.zero_idx = None
        self.mem_dict = dict()
        self.output_array = []
        self.outputs = []
        self.cond = self.helper.create_variable_for_type_inference(dtype='bool')
        self.cond.stop_gradient = False
        self.while_op = While(self.cond)
        self.input_array = []
        self.mem_link = []

    def step_input(self, x, level=0):
        r"""
        This function is used to set sequence x as DynamicRNN's input.
        The maximum sequence length in x determines the number of time steps
        the RNN unit will be executed. DynamicRNN can take multiple inputs.
        When all inputs' :code:`lod_level` are 1, all inputs should hold the
        same LoD. When :code:`x.lod_level >= 2` , the input sequence will be
        unfold along specified level, and the slice of each time step is a
        LoDTensor whose lod_level is :code:`x.lod_level - level - 1` .
        In this case, the specified LoD level of multiple inputs should be the same.

        - Case 1:

        .. code-block:: text

            # input, where Si is slice data of shape [1, N]
            level = 0
            x.lod = [[2, 1, 3]]
            x.shape = [6, N]
            x.data = [[S0],
                      [S0],
                      [S1],
                      [S2],
                      [S2],
                      [S2]]

            # output
            # step 0, time step data of 3 sequences
            out.lod = [[]]
            out.shape = [3, N]
            out.data = [[S2],
                        [S0],
                        [S1]]

            # step 1, time step data of 2 sequences
            out.lod = [[]]
            out.shape = [2, N]
            out.data = [[S2],
                        [S0]]

            # step 2, time step data of 1 sequences
            out.lod = [[]]
            out.shape = [1, N]
            out.data = [[S2]]


        Args:
            x (Variable): The input LoDTensor which holds information of a
                minibatch of variable-length sequences and should meet :code:`x.lod_level >= 1` .
                When RNN has multiple inputs, the first dimension should match
                across all inputs, but other shape components may differ.
                Optional data types are: bool, float16, float32, float64, int8, int16, int32, int64, uint8.
            level (int, optional): The level of lod used to split steps.
                It should be in range :math:`[0, x.lod\_level)` . The default value is 0.

        Returns:
            Variable: The current time step in the input sequence. If there are :code:`num_sequences` \
                sequences in x whose length is larger than :code:`step_idx` , the returned Variable \
                will only hold the :code:`step_idx` -th time step of those `num_sequences` sequences. \
                The data type is the same as input. If :code:`x.lod_level == 1` , the return value is \
                a Tensor of shape :math:`\{num\_sequences, x.shape[1], ...\}` , or it will \
                be a variable-length LoDTensor.

        Raises:
            ValueError: When :code:`step_input()` is called outside :code:`block()` .
            TypeError: When x is not a Variable.

        Examples:
            ..  code-block:: python

                import paddle.fluid as fluid

                sentence = fluid.data(name='sentence', shape=[None, 1], dtype='int64', lod_level=1)
                embedding = fluid.layers.embedding(input=sentence, size=[65536, 32], is_sparse=True)

                drnn = fluid.layers.DynamicRNN()
                with drnn.block():
                    # Set embedding as RNN's input, each time step processes a word from the sentence
                    word = drnn.step_input(embedding)
                    # Initialize memory to a Tensor whose value is 0, shape=[batch_size, 200],
                    # where batch_size is the number of sequences in embedding.
                    memory = drnn.memory(shape=[200])
                    hidden = fluid.layers.fc(input=[word, memory], size=200, act='relu')
                    # Update memory to hidden
                    drnn.update_memory(ex_mem=memory, new_mem=hidden)
                    # Set hidden as RNN's output
                    drnn.output(hidden)

                # Get RNN's result
                rnn_output = drnn()
        """
        self._assert_in_rnn_block_("step_input")
        check_type(x, 'x', Variable, 'fluid.layers.DynamicRNN.step_input()')
        parent_block = self._parent_block_()
        if self.lod_rank_table is None:
            self.lod_rank_table = parent_block.create_var(
                name=unique_name.generate('lod_rank_table'),
                type=core.VarDesc.VarType.LOD_RANK_TABLE)
            self.lod_rank_table.stop_gradient = True
            parent_block.append_op(
                type='lod_rank_table',
                inputs={"X": x},
                outputs={"Out": self.lod_rank_table},
                attrs={"level": level})
            self.max_seq_len = parent_block.create_var(
                name=unique_name.generate('dynamic_rnn_max_seq_len'),
                dtype='int64')
            self.max_seq_len.stop_gradient = False
            parent_block.append_op(
                type='max_sequence_len',
                inputs={'RankTable': self.lod_rank_table},
                outputs={"Out": self.max_seq_len})
            self.cond.stop_gradient = True
            parent_block.append_op(
                type='less_than',
                inputs={'X': self.step_idx,
                        'Y': self.max_seq_len},
                outputs={'Out': self.cond},
                attrs={'force_cpu': True})

        input_array = parent_block.create_var(
            name=unique_name.generate('dynamic_rnn_input_array'),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.dtype)
        self.input_array.append((input_array, x.dtype))
        parent_block.append_op(
            type='lod_tensor_to_array',
            inputs={'X': x,
                    'RankTable': self.lod_rank_table},
            outputs={'Out': input_array})
        return array_read(array=input_array, i=self.step_idx)

    def static_input(self, x):
        r"""
        This function is used to set x as DynamicRNN's static input. It is optional.

        - Case 1, set static input with LoD

        .. code-block:: text

            # RNN's input is the same as the case listed in step_input
            # static input, where Si is slice data of shape [1, M]
            x.lod = [[3, 1, 2]]
            x.shape = [6, M]
            x.data = [[S0],
                      [S0],
                      [S0],
                      [S1],
                      [S2],
                      [S2]]

            # step 0, batch data corresponding to the 3 input sequences
            out.lod = [[2, 3, 1]]
            out.shape = [6, M]
            out.data = [[S2],
                        [S2],
                        [S0],
                        [S0],
                        [S0],
                        [S1]]

            # step 1, batch data corresponding to the 2 input sequences
            out.lod = [[2, 3]]
            out.shape = [5, M]
            out.data = [[S2],
                        [S2],
                        [S0],
                        [S0],
                        [S0]]

            # step 2, batch data corresponding to the 1 input sequences
            out.lod = [[2]]
            out.shape = [2, M]
            out.data = [[S2],
                        [S2]]


        - Case 2, set static input without LoD

        .. code-block:: text

            # RNN's input is the same as the case listed in step_input
            # static input, where Si is slice data of shape [1, M]
            x.lod = [[]]
            x.shape = [3, M]
            x.data = [[S0],
                      [S1],
                      [S2]]

            # step 0, batch data corresponding to the 3 input sequences
            out.lod = [[]]
            out.shape = [3, M]
            out.data = [[S2],
                        [S0],
                        [S1]]

            # step 1, batch data corresponding to the 2 input sequences
            out.lod = [[]]
            out.shape = [2, M]
            out.data = [[S2],
                        [S0]]

            # step 2, batch data corresponding to the 1 input sequences
            out.lod = [[]]
            out.shape = [1, M]
            out.data = [[S2]]


        Args:
            x (Variable): The static input LoDTensor which should hold the same number of sequences
                as RNN's input (the input LoDTensor set by :code:`step_input()` ). If the LoD is None,
                the input x will be treated as a minibatch with :code:`x.shape[0]` sequences of length 1.
                Optional data types are: bool, float16, float32, float64, int8, int16, int32, int64, uint8.

        Returns:
            Variable: The input LoDTensor after sorted and shrank. If there are :code:`num_sequences` \
                sequences in RNN's input LoDTensor whose length is larger than :code:`step_idx` , \
                the static input Tensor will be sorted to the same order as RNN's input and \
                will only retain data corresponding to those :code:`num_sequences` sequences. \
                The data type is the same as input. If :code:`x.lod == None` , the return value is \
                a Tensor of shape :math:`\{num\_sequences, x.shape[1], ...\}` , or it will \
                be a variable-length LoDTensor.

        Raises:
            ValueError: When :code:`static_input()` is called outside :code:`block()` .
            TypeError: When x is not a Variable.
            RuntimeError: When :code:`static_input()` is called before :code:`step_input()` .

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)
                encoder_proj = fluid.data(name='encoder_proj', shape=[None, 32], dtype='float32', lod_level=1)
                decoder_boot = fluid.data(name='boot', shape=[None, 10], dtype='float32')

                drnn = fluid.layers.DynamicRNN()
                with drnn.block():
                    # Set sentence as RNN's input, each time step processes a word from the sentence
                    current_word = drnn.step_input(sentence)
                    # Set encode_proj as RNN's static input
                    encoder_word = drnn.static_input(encoder_proj)
                    # Initialize memory with boot_memory, which need reorder according to RNN's input sequences
                    memory = drnn.memory(init=decoder_boot, need_reorder=True)
                    fc_1 = fluid.layers.fc(input=encoder_word, size=30)
                    fc_2 = fluid.layers.fc(input=current_word, size=30)
                    decoder_inputs = fc_1 + fc_2
                    hidden, _, _ = fluid.layers.gru_unit(input=decoder_inputs, hidden=memory, size=30)
                    # Update memory with hidden
                    drnn.update_memory(ex_mem=memory, new_mem=hidden)
                    out = fluid.layers.fc(input=hidden, size=10, bias_attr=True, act='softmax')
                    # Set out as RNN's output
                    drnn.output(out)

                # Get RNN's result
                rnn_output = drnn()
        """
        self._assert_in_rnn_block_("static_input")
        check_type(x, 'x', Variable, 'fluid.layers.DynamicRNN.static_input()')
        if self.lod_rank_table is None:
            raise RuntimeError(
                "static_input() must be called after step_input().")
        parent_block = self._parent_block_()
        x_reordered = parent_block.create_var(
            name=unique_name.generate("dynamic_rnn_static_input_reordered"),
            type=core.VarDesc.VarType.LOD_TENSOR,
            dtype=x.dtype)
        parent_block.append_op(
            type='reorder_lod_tensor_by_rank',
            inputs={'X': [x],
                    'RankTable': [self.lod_rank_table]},
            outputs={'Out': [x_reordered]})
        return shrink_memory(x_reordered, self.step_idx, self.lod_rank_table)

    @signature_safe_contextmanager
    def block(self):
        """
        The function is used to list the operations executed during
        each time step in RNN. The operation list will be executed :code:`max_sequence_len`
        times (where :code:`max_sequence_len` is the maximum length of RNN's input sequences).

        Raises:
            ValueError: When :code:`block()` is called multi-times.
        """
        if self.status != DynamicRNN.BEFORE_RNN:
            raise ValueError("rnn.block() can only be invoke once")
        self.step_idx = fill_constant(
            shape=[1], dtype='int64', value=0, force_cpu=True)
        self.step_idx.stop_gradient = False
        self.status = DynamicRNN.IN_RNN
        with self.while_op.block():
            yield
            increment(x=self.step_idx, value=1.0, in_place=True)

            for new_mem, mem_array in self.mem_link:
                array_write(x=new_mem, i=self.step_idx, array=mem_array)

            less_than(
                x=self.step_idx,
                y=self.max_seq_len,
                force_cpu=True,
                cond=self.cond)

        self.status = DynamicRNN.AFTER_RNN
        for each_array in self.output_array:
            self.outputs.append(
                array_to_lod_tensor(
                    x=each_array, table=self.lod_rank_table))

    def __call__(self, *args, **kwargs):
        """
        This function is used to get the output  sequences of DynamicRNN.

        Args:
            None

        Returns:
            Variable or Variable list: RNN's output sequences.

        Raises:
            ValueError: When :code:`__call__()` is called before :code:`block()` .
        """
        if self.status != DynamicRNN.AFTER_RNN:
            raise ValueError(("Output of the dynamic RNN can only be visited "
                              "outside the rnn block."))
        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def memory(self,
               init=None,
               shape=None,
               value=0.0,
               need_reorder=False,
               dtype='float32'):
        r"""
        Create a memory Variable for DynamicRNN to deliver data cross time steps.
        It can be initialized by an existing Tensor or a constant Tensor of given
        dtype and shape.

        Args:
            init (Variable, optional): LoDTensor used to initialize the memory.
                If init is not None, it should hold the same number of sequences
                as RNN's input (the input LoDTensor set by :code:`step_input()` )
                and the memory will be initialized to it. If init's LoD is None,
                it will be treated as a minibatch with :code:`init.shape[0]` sequences
                of length 1. The default value is None.
            shape (list|tuple, optional): When init is None, it is used to specify
                the memory's shape. Note that the shape does not include the batch_size.
                If setting shape to :math:`\{D_1, D_2, ...\}` , the shape of memory Tensor
                will be :math:`\{batch\_size, D_1, D_2, ...\}` , where batch_size is
                determined by RNN's input sequences. The default value is None.
            value (float, optional): When init is None, it is used as initialized value
                of memory. The default value is 0.0.
            need_reorder (bool, optional): When init is not None, it determines whether
                the memory needs to reorder like the RNN's input sequences. It should be
                set to True when the initialized memory depends on the order of input samples.
                The default value is False.
            dtype (str|numpy.dtype, optional): When init is None, it is used to set the
                data type of memory. The default value is "float32". Optional data types
                are: "float32", "float64", "int32", "int64".

        Returns:
            Variable: The memory LoDTensor after shrank.  If there are :code:`num_sequences` \
                sequences in RNN's input LoDTensor whose length is larger than :code:`step_idx` , \
                the memory Tensor also need to be shrank and will only retain data \
                corresponding to those :code:`num_sequences` sequences.

        Raises:
            ValueError: When :code:`memory()` is called outside :code:`block()` .
            TypeError: When init is set and is not a Variable.
            ValueError: When :code:`memory()` is called before :code:`step_input()` .

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)
                boot_memory = fluid.data(name='boot', shape=[None, 10], dtype='float32')

                drnn = fluid.layers.DynamicRNN()
                with drnn.block():
                    # Set sentence as RNN's input, each time step processes a word from the sentence
                    word = drnn.step_input(sentence)
                    # Initialize memory with boot_memory, which need reorder according to RNN's input sequences
                    memory = drnn.memory(init=boot_memory, need_reorder=True)
                    hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
                    # Update memory with hidden
                    drnn.update_memory(ex_mem=memory, new_mem=hidden)
                    # Set hidden as RNN's output
                    drnn.output(hidden)

                # Get RNN's result
                rnn_output = drnn()


        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)

                drnn = fluid.layers.DynamicRNN()
                with drnn.block():
                    # Set sentence as RNN's input, each time step processes a word from the sentence
                    word = drnn.step_input(sentence)
                    # Initialize memory to a Tensor whose value is 0, shape=[batch_size, 10],
                    # where batch_size is the number of sequences in sentence.
                    memory = drnn.memory(shape=[10], dtype='float32', value=0)
                    hidden = fluid.layers.fc(input=[word, memory], size=10, act='tanh')
                    # Update memory with hidden
                    drnn.update_memory(ex_mem=memory, new_mem=hidden)
                    # Set hidden as RNN's output
                    drnn.output(hidden)

                # Get RNN's result
                rnn_output = drnn()
        """
        self._assert_in_rnn_block_('memory')
        self._init_zero_idx_()
        if shape is not None:
            check_type(shape, 'shape', (list, tuple),
                       'fluid.layers.DynamicRNN.memory()')
        if init is not None:
            check_type(init, 'init', Variable,
                       'fluid.layers.DynamicRNN.memory()')
            parent_block = self._parent_block_()
            init_tensor = init
            if need_reorder == True:
                if self.lod_rank_table is None:
                    raise ValueError(
                        'If set need_reorder to True, make sure step_input be '
                        'invoked before '
                        'memory(init=init, need_reordered=True, ...).')
                init_reordered = parent_block.create_var(
                    name=unique_name.generate('dynamic_rnn_mem_init_reordered'),
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    dtype=init.dtype)
                parent_block.append_op(
                    type='reorder_lod_tensor_by_rank',
                    inputs={
                        'X': [init_tensor],
                        'RankTable': [self.lod_rank_table]
                    },
                    outputs={'Out': [init_reordered]})
                init_tensor = init_reordered
            mem_array = parent_block.create_var(
                name=unique_name.generate('dynamic_rnn_mem_array'),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=init.dtype)
            parent_block.append_op(
                type='write_to_array',
                inputs={'X': init_tensor,
                        'I': self.zero_idx},
                outputs={'Out': mem_array})
            retv = array_read(array=mem_array, i=self.step_idx)
            retv = shrink_memory(
                x=retv, i=self.step_idx, table=self.lod_rank_table)
            self.mem_dict[retv.name] = mem_array
            return retv
        else:
            if len(self.input_array) == 0:
                raise ValueError(
                    "step_input should be invoked before memory(shape=..., value=...)"
                )
            parent_block = self._parent_block_()
            init = parent_block.create_var(
                name=unique_name.generate('mem_init'), dtype=dtype)
            arr, dtype = self.input_array[0]
            in0 = parent_block.create_var(
                name=unique_name.generate('in0'), dtype=dtype)
            parent_block.append_op(
                type='read_from_array',
                inputs={'X': [arr],
                        'I': [self.zero_idx]},
                outputs={'Out': [in0]})
            parent_block.append_op(
                type='fill_constant_batch_size_like',
                inputs={'Input': [in0]},
                outputs={'Out': [init]},
                attrs={
                    'shape': [-1] + shape,
                    'value': float(value),
                    'dtype': init.dtype
                })
            return self.memory(init=init)

    def update_memory(self, ex_mem, new_mem):
        """
        Update the memory which need to be delivered across time steps.

        Args:
            ex_mem (Variable): The memory data of previous time step.
            new_mem (Variable): The new memory data produced in current time step.
                The shape and data type of ex_mem and new_mem should be the same.

        Returns:
            None
        
        Raises:
            ValueError: When :code:`update_memory()` is called outside :code:`block()` .
            TypeError: When :code:`ex_mem` or :code:`new_mem` is not a Variable.
            ValueError: When :code:`ex_mem` is defined by :code:`memory()` .
            ValueError: When :code:`update_memory()` is called before :code:`step_input()` .
        """
        self._assert_in_rnn_block_('update_memory')
        check_type(ex_mem, 'ex_mem', Variable,
                   'fluid.layers.DynamicRNN.update_memory()')
        check_type(new_mem, 'new_mem', Variable,
                   'fluid.layers.DynamicRNN.update_memory()')

        mem_array = self.mem_dict.get(ex_mem.name, None)
        if mem_array is None:
            raise ValueError("Please invoke memory before update_memory")
        if self.lod_rank_table is None:
            raise ValueError("Please invoke step_input before update_memory")

        self.mem_link.append((new_mem, mem_array))

    def output(self, *outputs):
        """
        This function is used to set :code:`outputs` as RNN's output.

        Args:
            *outputs (Variable ...): The output Tensor. DynamicRNN can mark multiple
                Variables as its output.

        Returns:
            None

        Raises:
            ValueError: When :code:`output()` is called outside :code:`block()` .
        """
        self._assert_in_rnn_block_('output')
        parent_block = self._parent_block_()
        for each in outputs:
            check_type(each, "outputs", Variable,
                       "fluid.layers.DynamicRNN.output")
            outside_array = parent_block.create_var(
                name=unique_name.generate_with_ignorable_key("_".join(
                    [self.helper.name, "output_array", each.name])),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=each.dtype)
            array_write(x=each, i=self.step_idx, array=outside_array)
            self.output_array.append(outside_array)

    def _init_zero_idx_(self):
        if self.zero_idx is None:
            parent_block = self._parent_block_()
            self.zero_idx = parent_block.create_var(
                name=unique_name.generate('zero_idx'), dtype='int64')
            parent_block.append_op(
                type='fill_constant',
                inputs={},
                outputs={'Out': [self.zero_idx]},
                attrs={
                    'shape': [1],
                    'dtype': self.zero_idx.dtype,
                    'value': float(0),
                    'force_cpu': True
                })

    def _parent_block_(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)

        return parent_block

    def _assert_in_rnn_block_(self, method):
        if self.status != DynamicRNN.IN_RNN:
            raise ValueError("{0} can only be invoked inside rnn block.".format(
                method))


def switch_case(branch_index, branch_fns, default=None, name=None):
    '''
    :api_attr: Static Graph

    This operator is like a C++ switch/case statement.

    Args:
        branch_index(Tensor): A Tensor with shape [1] to specify which branch to execute. The data type is ``int32``, ``int64`` or ``uint8``.
        branch_fns(dict|list|tuple): If it's a list or tuple, the elements in it could be pairs of (int, callable) or simple callables whose actual index will be used as the index of callable. If it's a dict, its key is a python integer and the value is a callable. All callables return the same structure of Tensors.
        default(callable, optional): Callable that returns a structure of Tensors.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor|list(Tensor): Tensors returned by the callable specified by ``branch_index`` in ``branch_fns``,
        or Tensors returned by ``default`` if ``default`` is not None and no index matches in ``branch_fns``,
        or Tensors returned by the callable with the max index in ``branch_fns`` if ``default`` is None and no index matches in ``branch_fns``.

    Raises:
        TypeError: If the type of ``branch_index`` is not Tensor.
        TypeError: If the data type of ``branch_index`` is not ``int32``, ``int64`` or ``uint8``.
        TypeError: If the type of ``branch_fns`` is not dict, list or tuple.
        TypeError: If the elements of ``branch_fns`` is not 2-tuple.
        TypeError: If the first element of 2-tuple in ``branch_fns`` is not integer.
        ValueError: If the first element of 2-tuple in ``branch_fns`` is not unique.
        TypeError: If the second element of 2-tuple in ``branch_fns`` is not callable.
        TypeError: If ``default`` is not None but it is not callable.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()

            def fn_1():
                return paddle.full(shape=[1, 2], dtype='float32', fill_value=1)

            def fn_2():
                return paddle.full(shape=[2, 2], dtype='int32', fill_value=2)

            def fn_3():
                return paddle.full(shape=[3], dtype='int32', fill_value=3)

            main_program = paddle.static.default_startup_program()
            startup_program = paddle.static.default_main_program()
            with paddle.static.program_guard(main_program, startup_program):
                index_1 = paddle.full(shape=[1], dtype='int32', fill_value=1)
                index_2 = paddle.full(shape=[1], dtype='int32', fill_value=2)

                out_1 = paddle.static.nn.switch_case(
                    branch_index=index_1,
                    branch_fns={1: fn_1, 2: fn_2},
                    default=fn_3)

                out_2 = paddle.static.nn.switch_case(
                    branch_index=index_2,
                    branch_fns=[(1, fn_1), (2, fn_2)],
                    default=fn_3)

                # Argument default is None and no index matches. fn_3 will be called because of the max index 7.
                out_3 = paddle.static.nn.switch_case(
                    branch_index=index_2,
                    branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)])

                exe = paddle.static.Executor(paddle.CPUPlace())
                res_1, res_2, res_3 = exe.run(main_program, fetch_list=[out_1, out_2, out_3])
                print(res_1)  # [[1. 1.]]
                print(res_2)  # [[2 2] [2 2]]
                print(res_3)  # [3 3 3]
    '''
    helper = LayerHelper('switch_case', **locals())

    def _check_args(branch_index, branch_fns, default):

        check_variable_and_dtype(branch_index, 'branch_index',
                                 ['uint8', 'int32', 'int64'], 'switch_case')

        if convert_dtype(branch_index.dtype) != "int64":
            branch_index = cast(branch_index, "int64")

        check_type(branch_fns, 'branch_fns', (list, tuple, dict), 'switch_case')

        branch_fns = branch_fns.items() if isinstance(branch_fns,
                                                      dict) else branch_fns

        branch_fns = list(enumerate(branch_fns)) if all(
            callable(fn) for fn in branch_fns) else branch_fns

        keys_of_fns = []
        for index_fn_pair in branch_fns:
            if not isinstance(index_fn_pair, tuple):
                raise TypeError(
                    _error_message("The elements' type", "branch_fns",
                                   "switch_case", tuple, type(branch_fns)))

            if len(index_fn_pair) != 2:
                raise TypeError(
                    _error_message("The tuple's size", "branch_fns",
                                   "switch_case", "2",
                                   str(len(index_fn_pair)) + "-tuple"))

            key, fn = index_fn_pair

            if not isinstance(key, int):
                raise TypeError(
                    _error_message("The key's type", "branch_fns",
                                   "switch_case", int, type(key)))

            if key in keys_of_fns:
                raise ValueError(
                    "The key in 'branch_fns' must be unique, but '{}' appears more than once.".
                    format(key))
            else:
                keys_of_fns.append(key)

            if not callable(fn):
                raise TypeError(
                    _error_message("The type of function for key {}".format(
                        key), "branch_fns", "switch_case", "callable", type(
                            fn)))

        if default is None:
            default = sorted(branch_fns)[-1][1]
            branch_fns = sorted(branch_fns)[:-1]
        elif not callable(default):
            raise TypeError("The default in Op(case) must be callable.")

        pred_fn_pairs = []
        for index, fn in branch_fns:
            new_index = fill_constant(shape=[1], dtype="int64", value=index)
            pred = equal(branch_index, new_index)
            pred_fn_pairs.append((pred, fn))

        return pred_fn_pairs, default

    pred_fn_pairs, default = _check_args(branch_index, branch_fns, default)
    false_fn = default
    for pred, true_fn in pred_fn_pairs:
        false_fn = partial(cond, pred=pred, true_fn=true_fn, false_fn=false_fn)

    final_fn = false_fn
    return final_fn()


@templatedoc()
def reorder_lod_tensor_by_rank(x, rank_table):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}.
        rank_table(${rank_table_type}): ${rank_table_comment}.
    
    Returns:
        out(${out_type}): ${out_comment}.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data_desc = (['input', [9], 0], ['ref', [5], 1])
          data = fluid.layers.data(name=data_desc[0][0], shape=data_desc[0][1])
          rank_data = fluid.layers.data(name=data_desc[1][0], shape=data_desc[1][1])
          table = fluid.layers.control_flow.lod_rank_table(rank_data)
          new_data = fluid.layers.reorder_lod_tensor_by_rank(
                           x=data, rank_table=table)

    """

    check_type(x, 'x', (Variable), 'reorder_lod_tensor_by_rank')
    check_type(rank_table, 'rank_table', (Variable),
               'reorder_lod_tensor_by_rank')
    if rank_table.type != core.VarDesc.VarType.LOD_RANK_TABLE:
        raise TypeError("The type of rank_table should be LOD_RANK_TABLE.")

    helper = LayerHelper('reorder_lod_tensor_by_rank', **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='reorder_lod_tensor_by_rank',
        inputs={'X': [x],
                'RankTable': [rank_table]},
        outputs={'Out': [out]})
    return out


def is_empty(x, name=None):
    """

    Test whether a Tensor is empty.

    Args:
        x (Tensor): The Tensor to be tested.
        name (str, optional): The default value is ``None`` . Normally users
                            don't have to set this parameter. For more information,
                            please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A bool scalar Tensor. True if 'x' is an empty Tensor.

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.rand(shape=[4, 32, 32], dtype='float32')
            res = paddle.is_empty(x=input)
            print("res:", res)
            # ('res:', Tensor: eager_tmp_1
            #    - place: CPUPlace
            #    - shape: [1]
            #    - layout: NCHW
            #    - dtype: bool
            #    - data: [0])

    """
    if in_dygraph_mode():
        return _C_ops.is_empty(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'is_empty')
    check_type(name, "name", (str, type(None)), "is_empty")

    helper = LayerHelper("is_empty", **locals())
    cond = helper.create_variable_for_type_inference(dtype='bool')
    cond.stop_gradient = True
    helper.append_op(
        type='is_empty', inputs={'X': [x]}, outputs={'Out': [cond]})
    return cond
