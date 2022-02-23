#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
"""
This module provides a general beam search decoder API for RNN based decoders.
The purpose of this API is to allow users to highly customize the behavior
within their RNN decoder(vanilla RNN, LSTM, attention + LSTM, future etc.),
without using the low level API such as while ops.

This API is still under active development and may change drastically.
"""

from __future__ import print_function

from ...wrapped_decorator import signature_safe_contextmanager
import numpy as np
import six

from ... import layers
from ...framework import Variable
from ... import core
from ... import framework, unique_name
from ...layer_helper import LayerHelper

__all__ = ['InitState', 'StateCell', 'TrainingDecoder', 'BeamSearchDecoder']


class _DecoderType:
    TRAINING = 1
    BEAM_SEARCH = 2


class InitState(object):
    """
    The initial hidden state object. The state objects holds a variable, and may
    use it to initialize the hidden state cell of RNN. Usually used as input to
    `StateCell` class.

    Args:
        init (Variable): The initial variable of the hidden state. If set None,
            the variable will be created as a tensor with constant value based
            on `shape` and `value` param.
        shape (tuple|list): If `init` is None, new Variable's shape. Default
            None.
        value (float): If `init` is None, new Variable's value. Default None.
        init_boot (Variable): If provided, the initial variable will be created
            with the same shape as this variable.
        need_reorder (bool): If set true, the init will be sorted by its lod
            rank within its batches. This should be used if `batch_size > 1`.
        dtype (np.dtype|core.VarDesc.VarType|str): Data type of the initial
            variable.

    Returns:
        An initialized state object.

    Examples:
        See `StateCell`.
    """

    def __init__(self,
                 init=None,
                 shape=None,
                 value=0.0,
                 init_boot=None,
                 need_reorder=False,
                 dtype='float32'):
        if init is not None:
            self._init = init
        elif init_boot is None:
            raise ValueError(
                'init_boot must be provided to infer the shape of InitState .\n')
        else:
            self._init = layers.fill_constant_batch_size_like(
                input=init_boot, value=value, shape=shape, dtype=dtype)

        self._shape = shape
        self._value = value
        self._need_reorder = need_reorder
        self._dtype = dtype

    @property
    def value(self):
        return self._init

    @property
    def need_reorder(self):
        return self._need_reorder


class _MemoryState(object):
    def __init__(self, state_name, rnn_obj, init_state):
        self._state_name = state_name  # each is a rnn.memory
        self._rnn_obj = rnn_obj
        self._state_mem = self._rnn_obj.memory(
            init=init_state.value, need_reorder=init_state.need_reorder)

    def get_state(self):
        return self._state_mem

    def update_state(self, state):
        self._rnn_obj.update_memory(self._state_mem, state)


class _ArrayState(object):
    def __init__(self, state_name, block, init_state):
        self._state_name = state_name
        self._block = block

        self._state_array = self._block.create_var(
            name=unique_name.generate('array_state_array'),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=init_state.value.dtype)

        self._counter = self._block.create_var(
            name=unique_name.generate('array_state_counter'),
            type=core.VarDesc.VarType.LOD_TENSOR,
            dtype='int64')

        # initialize counter
        self._block.append_op(
            type='fill_constant',
            inputs={},
            outputs={'Out': [self._counter]},
            attrs={
                'shape': [1],
                'dtype': self._counter.dtype,
                'value': float(0.0),
                'force_cpu': True
            })

        self._counter.stop_gradient = True

        # write initial state
        block.append_op(
            type='write_to_array',
            inputs={'X': init_state.value,
                    'I': self._counter},
            outputs={'Out': self._state_array})

    def get_state(self):
        state = layers.array_read(array=self._state_array, i=self._counter)
        return state

    def update_state(self, state):
        layers.increment(x=self._counter, value=1, in_place=True)
        layers.array_write(state, array=self._state_array, i=self._counter)


class StateCell(object):
    """
    The state cell class stores the hidden state of the RNN cell. A typical RNN
    cell has one or more hidden states, and one or more step inputs. This class
    allows you to defines the name of hidden states as well as step inputs, and
    their associated variables.

    Args:
        inputs (dict): A feeding dict of {name(str) : Variable}. It specifies
            the names of step inputs for RNN cell, and the associated variables.
            The variable could initially be None and set manually during each
            RNN step.
        states (dict): A feeding dict of {name(str) : InitState object}. It
            specifies the names of hidden states and their initialized state.
        out_state (str): A string that specifies the name of hidden state that
            will be used to compute the score in beam search process.
        name (str): The name of the RNN cell. Default None.

    Raises:
        `ValueError`: If the initial state is not an instance of InitState, or
            the out_state is not in the dict of states.

    Returns:
        StateCell: The initialized StateCell object.

    Examples:
        .. code-block:: python
          hidden_state = InitState(init=encoder_out, need_reorder=True)
          state_cell = StateCell(
              inputs={'current_word': None},
              states={'h': hidden_state},
              out_state='h')
    """

    def __init__(self, inputs, states, out_state, name=None):
        self._helper = LayerHelper('state_cell', name=name)
        self._cur_states = {}
        self._state_names = []
        for state_name, state in six.iteritems(states):
            if not isinstance(state, InitState):
                raise ValueError('state must be an InitState object.')
            self._cur_states[state_name] = state
            self._state_names.append(state_name)
        self._inputs = inputs  # inputs is place holder here
        self._cur_decoder_obj = None
        self._in_decoder = False
        self._states_holder = {}
        self._switched_decoder = False
        self._state_updater = None
        self._out_state = out_state
        if self._out_state not in self._cur_states:
            raise ValueError('out_state must be one state in states')

    def _enter_decoder(self, decoder_obj):
        if self._in_decoder == True or self._cur_decoder_obj is not None:
            raise ValueError('StateCell has already entered a decoder.')
        self._in_decoder = True
        self._cur_decoder_obj = decoder_obj
        self._switched_decoder = False

    def _leave_decoder(self, decoder_obj):
        if not self._in_decoder:
            raise ValueError('StateCell not in decoder, '
                             'invalid leaving operation.')

        if self._cur_decoder_obj != decoder_obj:
            raise ValueError('Inconsistent decoder object in StateCell.')

        self._in_decoder = False
        self._cur_decoder_obj = None
        self._switched_decoder = False

    def _switch_decoder(self):  # lazy switch
        if not self._in_decoder:
            raise ValueError('StateCell must be enter a decoder.')

        if self._switched_decoder:
            raise ValueError('StateCell already done switching.')

        for state_name in self._state_names:
            if state_name not in self._states_holder:
                state = self._cur_states[state_name]

                if not isinstance(state, InitState):
                    raise ValueError('Current type of state is %s, should be '
                                     'an InitState object.' % type(state))

                self._states_holder[state_name] = {}

                if self._cur_decoder_obj.type == _DecoderType.TRAINING:
                    self._states_holder[state_name][id(self._cur_decoder_obj)] \
                        = _MemoryState(state_name,
                                       self._cur_decoder_obj.dynamic_rnn,
                                       state)
                elif self._cur_decoder_obj.type == _DecoderType.BEAM_SEARCH:
                    self._states_holder[state_name][id(self._cur_decoder_obj)] \
                        = _ArrayState(state_name,
                                      self._cur_decoder_obj._parent_block(),
                                      state)
                else:
                    raise ValueError('Unknown decoder type, only support '
                                     '[TRAINING, BEAM_SEARCH]')

            # Read back, since current state should be LoDTensor
            self._cur_states[state_name] = \
                self._states_holder[state_name][
                    id(self._cur_decoder_obj)].get_state()

        self._switched_decoder = True

    def get_state(self, state_name):
        """
        The getter of state object. Find the state variable by its name.

        Args:
            state_name (str): A string of the state's name.

        Returns:
            The associated state object.
        """
        if self._in_decoder and not self._switched_decoder:
            self._switch_decoder()

        if state_name not in self._cur_states:
            raise ValueError(
                'Unknown state %s. Please make sure _switch_decoder() '
                'invoked.' % state_name)

        return self._cur_states[state_name]

    def get_input(self, input_name):
        """
        The getter of input variable. Find the input variable by its name.

        Args:
            input_name (str): The string of the input's name.

        Returns:
            The associated input variable.
        """
        if input_name not in self._inputs or self._inputs[input_name] is None:
            raise ValueError('Invalid input %s.' % input_name)
        return self._inputs[input_name]

    def set_state(self, state_name, state_value):
        """
        The setter of the state variable. Change the variable of the given
        `state_name`.

        Args:
            state_name (str): The name of the state to change.
            state_value (Var): The variable of the new state.
        """
        self._cur_states[state_name] = state_value

    def state_updater(self, updater):
        """
        Set up the updater to update the hidden state every RNN step. The
        behavior of updater could be customized by users. The updater should be
        a function that takes a `StateCell` object as input and update the
        hidden state within it. The hidden state could be accessed through
        `get_state` method.

        Args:
            updater (func): the updater to update the state cell.
        """
        self._state_updater = updater

        def _decorator(state_cell):
            if state_cell == self:
                raise TypeError('Updater should only accept a StateCell object '
                                'as argument.')
            updater(state_cell)

        return _decorator

    def compute_state(self, inputs):
        """
        Provide the step input of RNN cell, and compute the new hidden state
        with updater and give step input.

        Args:
            inputs (dict): A feed dict, {name(str): Variable}. name should be
            the names of step inputs for this RNN cell, and Variable should be
            the associated variables.

        Examples:
        .. code-block:: python
          state_cell.compute_state(inputs={'x': current_word})
        """
        if self._in_decoder and not self._switched_decoder:
            self._switch_decoder()

        for input_name, input_value in six.iteritems(inputs):
            if input_name not in self._inputs:
                raise ValueError('Unknown input %s. '
                                 'Please make sure %s in input '
                                 'place holder.' % (input_name, input_name))
            self._inputs[input_name] = input_value
        self._state_updater(self)

    def update_states(self):
        """
        Update and record state information after each RNN step.
        """
        if self._in_decoder and not self._switched_decoder:
            self._switched_decoder()

        for state_name, decoder_state in six.iteritems(self._states_holder):
            if id(self._cur_decoder_obj) not in decoder_state:
                raise ValueError('Unknown decoder object, please make sure '
                                 'switch_decoder been invoked.')
            decoder_state[id(self._cur_decoder_obj)].update_state(
                self._cur_states[state_name])

    def out_state(self):
        """
        Get the output state variable. This must be called after update_states.

        Returns:
            The output variable of the RNN cell.
        """
        return self._cur_states[self._out_state]


class TrainingDecoder(object):
    """
    A decoder that can only be used for training. The decoder could be
    initialized with a `StateCell` object. The computation within the RNN cell
    could be defined with decoder's block.

    Args:
        state_cell (StateCell): A StateCell object that handles the input and
            state variables.
        name (str): The name of this decoder. Default None.

    Returns:
        TrainingDecoder: The initialized TrainingDecoder object.

    Examples:
        .. code-block:: python
          decoder = TrainingDecoder(state_cell)
          with decoder.block():
              current_word = decoder.step_input(trg_embedding)
              decoder.state_cell.compute_state(inputs={'x': current_word})
              current_score = layers.fc(input=decoder.state_cell.get_state('h'),
                                        size=32,
                                        act='softmax')
              decoder.state_cell.update_states()
              decoder.output(current_score)
    """
    BEFORE_DECODER = 0
    IN_DECODER = 1
    AFTER_DECODER = 2

    def __init__(self, state_cell, name=None):
        self._helper = LayerHelper('training_decoder', name=name)
        self._status = TrainingDecoder.BEFORE_DECODER
        self._dynamic_rnn = layers.DynamicRNN()
        self._type = _DecoderType.TRAINING
        self._state_cell = state_cell
        self._state_cell._enter_decoder(self)

    @signature_safe_contextmanager
    def block(self):
        """
        Define the behavior of the decoder for each RNN time step.
        """
        if self._status != TrainingDecoder.BEFORE_DECODER:
            raise ValueError('decoder.block() can only be invoked once')
        self._status = TrainingDecoder.IN_DECODER

        with self._dynamic_rnn.block():
            yield

        self._status = TrainingDecoder.AFTER_DECODER
        self._state_cell._leave_decoder(self)

    @property
    def state_cell(self):
        self._assert_in_decoder_block('state_cell')
        return self._state_cell

    @property
    def dynamic_rnn(self):
        return self._dynamic_rnn

    @property
    def type(self):
        return self._type

    def step_input(self, x):
        """
        Set the input variable as a step input to the RNN cell. For example,
        in machine translation, each time step we read one word from the target
        sentences, then the target sentence is a step input to the RNN cell.

        Args:
            x (Variable): the variable to be used as step input.

        Returns:
            Variable: The variable as input of current step.

        Examples:
        .. code-block:: python
          current_word = decoder.step_input(trg_embedding)
        """
        self._assert_in_decoder_block('step_input')
        return self._dynamic_rnn.step_input(x)

    def static_input(self, x):
        """
        Set the input variable as a static input of RNN cell. In contrast to
        step input, this variable will be used as a whole within the RNN decode
        loop and will not be scattered into time steps.

        Args:
            x (Variable): the variable to be used as static input.

        Returns:
            Variable: The variable as input of current step.

        Examples:
        .. code-block:: python
          encoder_vec = decoder.static_input(encoded_vector)
        """
        self._assert_in_decoder_block('static_input')
        return self._dynamic_rnn.static_input(x)

    def __call__(self, *args, **kwargs):
        """
        Get the output of RNN. This API should only be invoked after RNN.block()

        Returns:
            Variable: The specified output of the RNN cell.
        """
        if self._status != TrainingDecoder.AFTER_DECODER:
            raise ValueError('Output of training decoder can only be visited '
                             'outside the block.')
        return self._dynamic_rnn(*args, **kwargs)

    def output(self, *outputs):
        """
        Set the output variable of the RNN cell.

        Args:
            *outputs (Variables): a series of variables that treated as output
                of the RNN cell.

        Examples:
        .. code-block:: python
          out = fluid.layers.fc(input=h,
                                size=32,
                                bias_attr=True,
                                act='softmax')
          decoder.output(out)
        """
        self._assert_in_decoder_block('output')
        self._dynamic_rnn.output(*outputs)

    def _assert_in_decoder_block(self, method):
        if self._status != TrainingDecoder.IN_DECODER:
            raise ValueError('%s should be invoked inside block of '
                             'TrainingDecoder object.' % method)


class BeamSearchDecoder(object):
    """
    A beam search decoder that can be used for inference. The decoder should be
    initialized with a `StateCell` object. The decode process can be defined
    within its block.

    Args:
        state_cell (StateCell): A StateCell object that handles the input and
            state variables.
        init_ids (Variable): The init beam search token ids.
        init_scores (Variable): The associated score of each id.
        target_dict_dim (int): Size of dictionary.
        word_dim (int): Word embedding dimension.
        input_var_dict (dict): A feeding dict to feed the required input
            variables to the state cell. It will be used by state_cell 's
            compute method. Default empty.
        topk_size (int): The topk size used for beam search. Default 50.
        max_len (int): The maximum allowed length of the generated sentence.
            Default 100.
        beam_size (int): The beam width of beam search decode. Default 1.
        end_id (int): The id of end token within beam search.
        name (str): The name of this decoder. Default None.

    Returns:
        BeamSearchDecoder: A initialized BeamSearchDecoder object.

    Examples:
    .. code-block:: python
      decoder = BeamSearchDecoder(
          state_cell=state_cell,
          init_ids=init_ids,
          init_scores=init_scores,
          target_dict_dim=target_dict_dim,
          word_dim=word_dim,
          init_var_dict={},
          topk_size=topk_size,
          sparse_emb=IS_SPARSE,
          max_len=max_length,
          beam_size=beam_size,
          end_id=1,
          name=None
      )
      decoder.decode()
      translation_ids, translation_scores = decoder()
    """
    BEFORE_BEAM_SEARCH_DECODER = 0
    IN_BEAM_SEARCH_DECODER = 1
    AFTER_BEAM_SEARCH_DECODER = 2

    def __init__(self,
                 state_cell,
                 init_ids,
                 init_scores,
                 target_dict_dim,
                 word_dim,
                 input_var_dict={},
                 topk_size=50,
                 sparse_emb=True,
                 max_len=100,
                 beam_size=1,
                 end_id=1,
                 name=None):
        self._helper = LayerHelper('beam_search_decoder', name=name)
        self._counter = layers.zeros(shape=[1], dtype='int64')
        self._counter.stop_gradient = True
        self._type = _DecoderType.BEAM_SEARCH
        self._max_len = layers.fill_constant(
            shape=[1], dtype='int64', value=max_len)
        self._cond = layers.less_than(
            x=self._counter,
            y=layers.fill_constant(
                shape=[1], dtype='int64', value=max_len))
        self._while_op = layers.While(self._cond)
        self._state_cell = state_cell
        self._state_cell._enter_decoder(self)
        self._status = BeamSearchDecoder.BEFORE_BEAM_SEARCH_DECODER
        self._zero_idx = layers.fill_constant(
            shape=[1], value=0, dtype='int64', force_cpu=True)
        self._array_dict = {}
        self._array_link = []
        self._ids_array = None
        self._scores_array = None
        self._beam_size = beam_size
        self._end_id = end_id

        self._init_ids = init_ids
        self._init_scores = init_scores
        self._target_dict_dim = target_dict_dim
        self._topk_size = topk_size
        self._sparse_emb = sparse_emb
        self._word_dim = word_dim
        self._input_var_dict = input_var_dict

    @signature_safe_contextmanager
    def block(self):
        """
        Define the behavior of the decoder for each RNN time step.
        """
        if self._status != BeamSearchDecoder.BEFORE_BEAM_SEARCH_DECODER:
            raise ValueError('block() can only be invoke once.')

        self._status = BeamSearchDecoder.IN_BEAM_SEARCH_DECODER

        with self._while_op.block():
            yield
            with layers.Switch() as switch:
                with switch.case(self._cond):
                    layers.increment(x=self._counter, value=1.0, in_place=True)

                    for value, array in self._array_link:
                        layers.array_write(
                            x=value, i=self._counter, array=array)

                    layers.less_than(
                        x=self._counter, y=self._max_len, cond=self._cond)

        self._status = BeamSearchDecoder.AFTER_BEAM_SEARCH_DECODER
        self._state_cell._leave_decoder(self)

    @property
    def type(self):
        return self._type

    def early_stop(self):
        """
        Stop the generation process in advance. Could be used as "break".
        """
        layers.fill_constant(
            shape=[1], value=0, dtype='bool', force_cpu=True, out=self._cond)

    def decode(self):
        """
        Set up the computation within the decoder. Then you could call the
        decoder to get the result of beam search decode. If you want to define
        a more specific decoder, you could override this function.

        Examples:
        .. code-block:: python
          decoder.decode()
          translation_ids, translation_scores = decoder()
        """
        with self.block():
            prev_ids = self.read_array(init=self._init_ids, is_ids=True)
            prev_scores = self.read_array(
                init=self._init_scores, is_scores=True)
            prev_ids_embedding = layers.embedding(
                input=prev_ids,
                size=[self._target_dict_dim, self._word_dim],
                dtype='float32',
                is_sparse=self._sparse_emb)

            feed_dict = {}
            update_dict = {}

            for init_var_name, init_var in six.iteritems(self._input_var_dict):
                if init_var_name not in self.state_cell._inputs:
                    raise ValueError('Variable ' + init_var_name +
                                     ' not found in StateCell!\n')

                read_var = self.read_array(init=init_var)
                update_dict[init_var_name] = read_var
                feed_var_expanded = layers.sequence_expand(read_var,
                                                           prev_scores)
                feed_dict[init_var_name] = feed_var_expanded

            for state_str in self._state_cell._state_names:
                prev_state = self.state_cell.get_state(state_str)
                prev_state_expanded = layers.sequence_expand(prev_state,
                                                             prev_scores)
                self.state_cell.set_state(state_str, prev_state_expanded)

            for i, input_name in enumerate(self._state_cell._inputs):
                if input_name not in feed_dict:
                    feed_dict[input_name] = prev_ids_embedding

            self.state_cell.compute_state(inputs=feed_dict)
            current_state = self.state_cell.out_state()
            current_state_with_lod = layers.lod_reset(
                x=current_state, y=prev_scores)
            scores = layers.fc(input=current_state_with_lod,
                               size=self._target_dict_dim,
                               act='softmax')
            topk_scores, topk_indices = layers.topk(scores, k=self._topk_size)
            accu_scores = layers.elementwise_add(
                x=layers.log(x=topk_scores),
                y=layers.reshape(
                    prev_scores, shape=[-1]),
                axis=0)
            selected_ids, selected_scores = layers.beam_search(
                prev_ids,
                prev_scores,
                topk_indices,
                accu_scores,
                self._beam_size,
                end_id=1,
                level=0)

            with layers.Switch() as switch:
                with switch.case(layers.is_empty(selected_ids)):
                    self.early_stop()
                with switch.default():
                    self.state_cell.update_states()
                    self.update_array(prev_ids, selected_ids)
                    self.update_array(prev_scores, selected_scores)
                    for update_name, var_to_update in six.iteritems(
                            update_dict):
                        self.update_array(var_to_update, feed_dict[update_name])

    def read_array(self, init, is_ids=False, is_scores=False):
        """
        Read an array to get the decoded ids and scores generated by previous
        RNN step. At the first step of RNN, the init variable mut be used to
        initialize the array.

        Args:
            init (Variable): The initial variable for first step usage. init
                must be provided.
            is_ids (bool): Specify whether the variable is an id.
            is_scores (bool): Specify whether the variable is a score.

        Returns:
            The associated variable generated during previous RNN steps.

        Examples:
            .. code-block:: python
              prev_ids = decoder.read_array(init=init_ids, is_ids=True)
              prev_scores = decoder.read_array(init=init_scores, is_scores=True)
        """
        self._assert_in_decoder_block('read_array')

        if is_ids and is_scores:
            raise ValueError('Shouldn\'t mark current array be ids array and'
                             'scores array at the same time.')

        if not isinstance(init, Variable):
            raise TypeError('The input argument `init` must be a Variable.')

        parent_block = self._parent_block()
        array = parent_block.create_var(
            name=unique_name.generate('beam_search_decoder_array'),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=init.dtype)
        parent_block.append_op(
            type='write_to_array',
            inputs={'X': init,
                    'I': self._zero_idx},
            outputs={'Out': array})

        if is_ids:
            self._ids_array = array
        elif is_scores:
            self._scores_array = array

        read_value = layers.array_read(array=array, i=self._counter)
        self._array_dict[read_value.name] = array
        return read_value

    def update_array(self, array, value):
        """
        Store the value generated in current step in an array for each RNN step.
        This array could be accessed by read_array method.

        Args:
            array (Variable): The array to append the new variable to.
            value (Variable): The newly generated value to be stored.
        """
        self._assert_in_decoder_block('update_array')

        if not isinstance(array, Variable):
            raise TypeError(
                'The input argument `array` of  must be a Variable.')
        if not isinstance(value, Variable):
            raise TypeError('The input argument `value` of must be a Variable.')

        array = self._array_dict.get(array.name, None)
        if array is None:
            raise ValueError('Please invoke read_array before update_array.')
        self._array_link.append((value, array))

    def __call__(self):
        """
        Run the decode process and return the final decode result.

        Returns:
            A tuple of decoded (id, score) pairs. id is a Variable that holds
            the generated tokens, and score is a Variable with the same shape
            as id, holds the score for each generated token.
        """
        if self._status != BeamSearchDecoder.AFTER_BEAM_SEARCH_DECODER:
            raise ValueError('Output of BeamSearchDecoder object can '
                             'only be visited outside the block.')
        return layers.beam_search_decode(
            ids=self._ids_array,
            scores=self._scores_array,
            beam_size=self._beam_size,
            end_id=self._end_id)

    @property
    def state_cell(self):
        self._assert_in_decoder_block('state_cell')
        return self._state_cell

    def _parent_block(self):
        """
        Getter of parent block.

        Returns:
            The parent block of decoder.
        """
        program = self._helper.main_program
        parent_block_idx = program.current_block().parent_idx
        if parent_block_idx < 0:
            raise ValueError('Invalid block with index %d.' % parent_block_idx)
        parent_block = program.block(parent_block_idx)
        return parent_block

    def _assert_in_decoder_block(self, method):
        if self._status != BeamSearchDecoder.IN_BEAM_SEARCH_DECODER:
            raise ValueError('%s should be invoked inside block of '
                             'BeamSearchDecoder object.' % method)
