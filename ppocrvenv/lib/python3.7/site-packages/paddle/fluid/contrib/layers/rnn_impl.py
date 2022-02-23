# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import layers, unique_name
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph.layer_object_helper import LayerObjectHelper
from paddle.fluid.layers.control_flow import StaticRNN

__all__ = ['BasicGRUUnit', 'basic_gru', 'BasicLSTMUnit', 'basic_lstm']


class BasicGRUUnit(Layer):
    """
    ****
    BasicGRUUnit class, using basic operators to build GRU
    The algorithm can be described as the equations below.

        .. math::
            u_t & = actGate(W_ux xu_{t} + W_uh h_{t-1} + b_u)

            r_t & = actGate(W_rx xr_{t} + W_rh h_{t-1} + b_r)

            m_t & = actNode(W_cx xm_t + W_ch dot(r_t, h_{t-1}) + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)

    Args:
        name_scope(string) : The name scope used to identify parameters and biases
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of GRU unit.
            If it is set to None or one attribute of ParamAttr, gru_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cell (actNode).
                             Default: 'fluid.layers.tanh'
        dtype(string): data type used in this unit

    Examples:

        .. code-block:: python

            import paddle.fluid.layers as layers
            from paddle.fluid.contrib.layers import BasicGRUUnit

            input_size = 128
            hidden_size = 256
            input = layers.data( name = "input", shape = [-1, input_size], dtype='float32')
            pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')

            gru_unit = BasicGRUUnit( "gru_unit", hidden_size )

            new_hidden = gru_unit( input, pre_hidden )

    """

    def __init__(self,
                 name_scope,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype='float32'):
        super(BasicGRUUnit, self).__init__(name_scope, dtype)
        # reserve old school _full_name and _helper for static graph save load
        self._full_name = unique_name.generate(name_scope + "/" +
                                               self.__class__.__name__)
        self._helper = LayerObjectHelper(self._full_name)

        self._name = name_scope
        self._hiden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._dtype = dtype

    def _build_once(self, input, pre_hidden):
        self._input_size = input.shape[-1]
        assert (self._input_size > 0)

        if self._param_attr is not None and self._param_attr.name is not None:
            gate_param_attr = copy.deepcopy(self._param_attr)
            candidate_param_attr = copy.deepcopy(self._param_attr)
            gate_param_attr.name += "_gate"
            candidate_param_attr.name += "_candidate"
        else:
            gate_param_attr = self._param_attr
            candidate_param_attr = self._param_attr

        self._gate_weight = self.create_parameter(
            attr=gate_param_attr,
            shape=[self._input_size + self._hiden_size, 2 * self._hiden_size],
            dtype=self._dtype)

        self._candidate_weight = self.create_parameter(
            attr=candidate_param_attr,
            shape=[self._input_size + self._hiden_size, self._hiden_size],
            dtype=self._dtype)

        if self._bias_attr is not None and self._bias_attr.name is not None:
            gate_bias_attr = copy.deepcopy(self._bias_attr)
            candidate_bias_attr = copy.deepcopy(self._bias_attr)
            gate_bias_attr.name += "_gate"
            candidate_bias_attr.name += "_candidate"
        else:
            gate_bias_attr = self._bias_attr
            candidate_bias_attr = self._bias_attr

        self._gate_bias = self.create_parameter(
            attr=gate_bias_attr,
            shape=[2 * self._hiden_size],
            dtype=self._dtype,
            is_bias=True)
        self._candidate_bias = self.create_parameter(
            attr=candidate_bias_attr,
            shape=[self._hiden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, pre_hidden):
        concat_input_hidden = layers.concat([input, pre_hidden], 1)

        gate_input = layers.matmul(x=concat_input_hidden, y=self._gate_weight)

        gate_input = layers.elementwise_add(gate_input, self._gate_bias)

        gate_input = self._gate_activation(gate_input)
        r, u = layers.split(gate_input, num_or_sections=2, dim=1)

        r_hidden = r * pre_hidden

        candidate = layers.matmul(
            layers.concat([input, r_hidden], 1), self._candidate_weight)
        candidate = layers.elementwise_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_hidden = u * pre_hidden + (1 - u) * c

        return new_hidden


def basic_gru(input,
              init_hidden,
              hidden_size,
              num_layers=1,
              sequence_length=None,
              dropout_prob=0.0,
              bidirectional=False,
              batch_first=True,
              param_attr=None,
              bias_attr=None,
              gate_activation=None,
              activation=None,
              dtype='float32',
              name='basic_gru'):
    r"""
    GRU implementation using basic operator, supports multiple layers and bidirectional gru.

    .. math::
            u_t & = actGate(W_ux xu_{t} + W_uh h_{t-1} + b_u)

            r_t & = actGate(W_rx xr_{t} + W_rh h_{t-1} + b_r)

            m_t & = actNode(W_cx xm_t + W_ch dot(r_t, h_{t-1}) + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)

    Args:
        input (Variable): GRU input tensor, 
                       if batch_first = False, shape should be ( seq_len x batch_size x input_size )  
                       if batch_first = True, shape should be ( batch_size x seq_len x hidden_size )
        init_hidden(Variable|None): The initial hidden state of the GRU
                       This is a tensor with shape ( num_layers x batch_size x hidden_size)
                       if is_bidirec = True, shape should be ( num_layers*2 x batch_size x hidden_size)
                       and can be reshaped to tensor with ( num_layers x 2 x batch_size x hidden_size) to use.
                       If it's None, it will be set to all 0.
        hidden_size (int): Hidden size of the GRU
        num_layers (int): The total number of layers of the GRU
        sequence_length (Variabe|None): A Tensor (shape [batch_size]) stores each real length of each instance,
                        This tensor will be convert to a mask to mask the padding ids
                        If it's None means NO padding ids
        dropout_prob(float|0.0): Dropout prob, dropout ONLY works after rnn output of each layers, 
                             NOT between time steps
        bidirectional (bool|False): If it is bidirectional
        batch_first (bool|True): The shape format of the input and output tensors. If true,
            the shape format should be :attr:`[batch_size, seq_len, hidden_size]`. If false,
            the shape format should be :attr:`[seq_len, batch_size, hidden_size]`. By default
            this function accepts input and emits output in batch-major form to be consistent
            with most of data format, though a bit less efficient because of extra transposes.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of GRU unit.
            If it is set to None or one attribute of ParamAttr, gru_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cell (actNode).
                             Default: 'fluid.layers.tanh'
        dtype(string): data type used in this unit
        name(string): name used to identify parameters and biases

    Returns:
        rnn_out(Tensor),last_hidden(Tensor)
            - rnn_out is result of GRU hidden, with shape (seq_len x batch_size x hidden_size) \
              if is_bidirec set to True, shape will be ( seq_len x batch_sze x hidden_size*2)
            - last_hidden is the hidden state of the last step of GRU \
              shape is ( num_layers x batch_size x hidden_size ) \
              if is_bidirec set to True, shape will be ( num_layers*2 x batch_size x hidden_size),
              can be reshaped to a tensor with shape( num_layers x 2 x batch_size x hidden_size)

    Examples:
        .. code-block:: python
            
            import paddle.fluid.layers as layers
            from paddle.fluid.contrib.layers import basic_gru

            batch_size = 20
            input_size = 128
            hidden_size = 256
            num_layers = 2
            dropout = 0.5
            bidirectional = True
            batch_first = False

            input = layers.data( name = "input", shape = [-1, batch_size, input_size], dtype='float32')
            pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')
            sequence_length = layers.data( name="sequence_length", shape=[-1], dtype='int32')


            rnn_out, last_hidden = basic_gru( input, pre_hidden, hidden_size, num_layers = num_layers, \
                    sequence_length = sequence_length, dropout_prob=dropout, bidirectional = bidirectional, \
                    batch_first = batch_first)

    """

    fw_unit_list = []

    for i in range(num_layers):
        new_name = name + "_layers_" + str(i)
        if param_attr is not None and param_attr.name is not None:
            layer_param_attr = copy.deepcopy(param_attr)
            layer_param_attr.name += "_fw_w_" + str(i)
        else:
            layer_param_attr = param_attr
        if bias_attr is not None and bias_attr.name is not None:
            layer_bias_attr = copy.deepcopy(bias_attr)
            layer_bias_attr.name += "_fw_b_" + str(i)
        else:
            layer_bias_attr = bias_attr
        fw_unit_list.append(
            BasicGRUUnit(new_name, hidden_size, layer_param_attr,
                         layer_bias_attr, gate_activation, activation, dtype))
    if bidirectional:
        bw_unit_list = []

        for i in range(num_layers):
            new_name = name + "_reverse_layers_" + str(i)
            if param_attr is not None and param_attr.name is not None:
                layer_param_attr = copy.deepcopy(param_attr)
                layer_param_attr.name += "_bw_w_" + str(i)
            else:
                layer_param_attr = param_attr
            if bias_attr is not None and bias_attr.name is not None:
                layer_bias_attr = copy.deepcopy(bias_attr)
                layer_bias_attr.name += "_bw_b_" + str(i)
            else:
                layer_bias_attr = bias_attr

            bw_unit_list.append(
                BasicGRUUnit(new_name, hidden_size, layer_param_attr,
                             layer_bias_attr, gate_activation, activation,
                             dtype))

    if batch_first:
        input = layers.transpose(input, [1, 0, 2])

    mask = None
    if sequence_length:
        max_seq_len = layers.shape(input)[0]
        mask = layers.sequence_mask(
            sequence_length, maxlen=max_seq_len, dtype='float32')
        mask = layers.transpose(mask, [1, 0])

    direc_num = 1
    if bidirectional:
        direc_num = 2
    if init_hidden:
        init_hidden = layers.reshape(
            init_hidden, shape=[num_layers, direc_num, -1, hidden_size])

    def get_single_direction_output(rnn_input,
                                    unit_list,
                                    mask=None,
                                    direc_index=0):
        rnn = StaticRNN()
        with rnn.step():
            step_input = rnn.step_input(rnn_input)

            if mask:
                step_mask = rnn.step_input(mask)

            for i in range(num_layers):
                if init_hidden:
                    pre_hidden = rnn.memory(init=init_hidden[i, direc_index])
                else:
                    pre_hidden = rnn.memory(
                        batch_ref=rnn_input,
                        shape=[-1, hidden_size],
                        ref_batch_dim_idx=1)

                new_hidden = unit_list[i](step_input, pre_hidden)

                if mask:
                    new_hidden = layers.elementwise_mul(
                        new_hidden, step_mask, axis=0) - layers.elementwise_mul(
                            pre_hidden, (step_mask - 1), axis=0)
                rnn.update_memory(pre_hidden, new_hidden)

                rnn.step_output(new_hidden)

                step_input = new_hidden
                if dropout_prob != None and dropout_prob > 0.0:
                    step_input = layers.dropout(
                        step_input,
                        dropout_prob=dropout_prob, )

            rnn.step_output(step_input)

        rnn_out = rnn()

        last_hidden_array = []
        rnn_output = rnn_out[-1]
        for i in range(num_layers):
            last_hidden = rnn_out[i]
            last_hidden = last_hidden[-1]
            last_hidden_array.append(last_hidden)

        last_hidden_output = layers.concat(last_hidden_array, axis=0)
        last_hidden_output = layers.reshape(
            last_hidden_output, shape=[num_layers, -1, hidden_size])

        return rnn_output, last_hidden_output
        # seq_len, batch_size, hidden_size

    fw_rnn_out, fw_last_hidden = get_single_direction_output(
        input, fw_unit_list, mask, direc_index=0)

    if bidirectional:
        bw_input = layers.reverse(input, axis=[0])
        bw_mask = None
        if mask:
            bw_mask = layers.reverse(mask, axis=[0])
        bw_rnn_out, bw_last_hidden = get_single_direction_output(
            bw_input, bw_unit_list, bw_mask, direc_index=1)

        bw_rnn_out = layers.reverse(bw_rnn_out, axis=[0])

        rnn_out = layers.concat([fw_rnn_out, bw_rnn_out], axis=2)
        last_hidden = layers.concat([fw_last_hidden, bw_last_hidden], axis=1)

        last_hidden = layers.reshape(
            last_hidden, shape=[num_layers * direc_num, -1, hidden_size])

        if batch_first:
            rnn_out = layers.transpose(rnn_out, [1, 0, 2])
        return rnn_out, last_hidden
    else:

        rnn_out = fw_rnn_out
        last_hidden = fw_last_hidden

        if batch_first:
            rnn_out = layers.transpose(rnn_out, [1, 0, 2])

        return rnn_out, last_hidden


def basic_lstm(input,
               init_hidden,
               init_cell,
               hidden_size,
               num_layers=1,
               sequence_length=None,
               dropout_prob=0.0,
               bidirectional=False,
               batch_first=True,
               param_attr=None,
               bias_attr=None,
               gate_activation=None,
               activation=None,
               forget_bias=1.0,
               dtype='float32',
               name='basic_lstm'):
    r"""
    LSTM implementation using basic operators, supports multiple layers and bidirectional LSTM.

    .. math::
           i_t &= \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_i)

           f_t &= \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_f + forget_bias )

           o_t &= \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_o)

           \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + b_c)

           c_t &= f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}

           h_t &= o_t \odot tanh(c_t)

    Args:
        input (Variable): lstm input tensor, 
                       if batch_first = False, shape should be ( seq_len x batch_size x input_size )  
                       if batch_first = True, shape should be ( batch_size x seq_len x hidden_size )
        init_hidden(Variable|None): The initial hidden state of the LSTM
                       This is a tensor with shape ( num_layers x batch_size x hidden_size)
                       if is_bidirec = True, shape should be ( num_layers*2 x batch_size x hidden_size)
                       and can be reshaped to a tensor with shape ( num_layers x 2 x batch_size x hidden_size) to use.
                       If it's None, it will be set to all 0.
        init_cell(Variable|None): The initial hidden state of the LSTM
                       This is a tensor with shape ( num_layers x batch_size x hidden_size)
                       if is_bidirec = True, shape should be ( num_layers*2 x batch_size x hidden_size)
                       and can be reshaped to a tensor with shape ( num_layers x 2 x batch_size x hidden_size) to use.
                       If it's None, it will be set to all 0.
        hidden_size (int): Hidden size of the LSTM
        num_layers (int): The total number of layers of the LSTM
        sequence_length (Variabe|None): A tensor (shape [batch_size]) stores each real length of each instance,
                        This tensor will be convert to a mask to mask the padding ids
                        If it's None means NO padding ids
        dropout_prob(float|0.0): Dropout prob, dropout ONLY work after rnn output of each layers, 
                             NOT between time steps
        bidirectional (bool|False): If it is bidirectional
        batch_first (bool|True): The shape format of the input and output tensors. If true,
            the shape format should be :attr:`[batch_size, seq_len, hidden_size]`. If false,
            the shape format should be :attr:`[seq_len, batch_size, hidden_size]`. By default
            this function accepts input and emits output in batch-major form to be consistent
            with most of data format, though a bit less efficient because of extra transposes.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of LSTM unit.
            If it is set to None or one attribute of ParamAttr, lstm_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cell (actNode).
                             Default: 'fluid.layers.tanh'
        forget_bias (float|1.0) : Forget bias used to compute the forget gate
        dtype(string): Data type used in this unit
        name(string): Name used to identify parameters and biases

    Returns:
        rnn_out(Tensor), last_hidden(Tensor), last_cell(Tensor)
            - rnn_out is the result of LSTM hidden, shape is (seq_len x batch_size x hidden_size) \
              if is_bidirec set to True, it's shape will be ( seq_len x batch_sze x hidden_size*2)
            - last_hidden is the hidden state of the last step of LSTM \
              with shape ( num_layers x batch_size x hidden_size ) \
              if is_bidirec set to True, it's shape will be ( num_layers*2 x batch_size x hidden_size),
              and can be reshaped to a tensor ( num_layers x 2 x batch_size x hidden_size)  to use.
            - last_cell is the hidden state of the last step of LSTM \
              with shape ( num_layers x batch_size x hidden_size ) \
              if is_bidirec set to True, it's shape will be ( num_layers*2 x batch_size x hidden_size),
              and can be reshaped to a tensor ( num_layers x 2 x batch_size x hidden_size)  to use.

    Examples:
        .. code-block:: python
            
            import paddle.fluid.layers as layers
            from paddle.fluid.contrib.layers import basic_lstm

            batch_size = 20
            input_size = 128
            hidden_size = 256
            num_layers = 2
            dropout = 0.5
            bidirectional = True
            batch_first = False

            input = layers.data( name = "input", shape = [-1, batch_size, input_size], dtype='float32')
            pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')
            pre_cell = layers.data( name = "pre_cell", shape=[-1, hidden_size], dtype='float32')
            sequence_length = layers.data( name="sequence_length", shape=[-1], dtype='int32')

            rnn_out, last_hidden, last_cell = basic_lstm( input, pre_hidden, pre_cell, \
                    hidden_size, num_layers = num_layers, \
                    sequence_length = sequence_length, dropout_prob=dropout, bidirectional = bidirectional, \
                    batch_first = batch_first)

    """
    fw_unit_list = []

    for i in range(num_layers):
        new_name = name + "_layers_" + str(i)
        if param_attr is not None and param_attr.name is not None:
            layer_param_attr = copy.deepcopy(param_attr)
            layer_param_attr.name += "_fw_w_" + str(i)
        else:
            layer_param_attr = param_attr
        if bias_attr is not None and bias_attr.name is not None:
            layer_bias_attr = copy.deepcopy(bias_attr)
            layer_bias_attr.name += "_fw_b_" + str(i)
        else:
            layer_bias_attr = bias_attr
        fw_unit_list.append(
            BasicLSTMUnit(
                new_name,
                hidden_size,
                param_attr=layer_param_attr,
                bias_attr=layer_bias_attr,
                gate_activation=gate_activation,
                activation=activation,
                forget_bias=forget_bias,
                dtype=dtype))
    if bidirectional:
        bw_unit_list = []

        for i in range(num_layers):
            new_name = name + "_reverse_layers_" + str(i)
            if param_attr is not None and param_attr.name is not None:
                layer_param_attr = copy.deepcopy(param_attr)
                layer_param_attr.name += "_bw_w_" + str(i)
            else:
                layer_param_attr = param_attr
            if bias_attr is not None and bias_attr.name is not None:
                layer_bias_attr = copy.deepcopy(bias_attr)
                layer_bias_attr.name += "_bw_b_" + str(i)
            else:
                layer_bias_attr = param_attr
            bw_unit_list.append(
                BasicLSTMUnit(
                    new_name,
                    hidden_size,
                    param_attr=layer_param_attr,
                    bias_attr=layer_bias_attr,
                    gate_activation=gate_activation,
                    activation=activation,
                    forget_bias=forget_bias,
                    dtype=dtype))

    if batch_first:
        input = layers.transpose(input, [1, 0, 2])

    mask = None
    if sequence_length:
        max_seq_len = layers.shape(input)[0]
        mask = layers.sequence_mask(
            sequence_length, maxlen=max_seq_len, dtype='float32')

        mask = layers.transpose(mask, [1, 0])

    direc_num = 1
    if bidirectional:
        direc_num = 2
        # convert to [num_layers, 2, batch_size, hidden_size]
    if init_hidden:
        init_hidden = layers.reshape(
            init_hidden, shape=[num_layers, direc_num, -1, hidden_size])
        init_cell = layers.reshape(
            init_cell, shape=[num_layers, direc_num, -1, hidden_size])

    # forward direction
    def get_single_direction_output(rnn_input,
                                    unit_list,
                                    mask=None,
                                    direc_index=0):
        rnn = StaticRNN()
        with rnn.step():
            step_input = rnn.step_input(rnn_input)

            if mask:
                step_mask = rnn.step_input(mask)

            for i in range(num_layers):
                if init_hidden:
                    pre_hidden = rnn.memory(init=init_hidden[i, direc_index])
                    pre_cell = rnn.memory(init=init_cell[i, direc_index])
                else:
                    pre_hidden = rnn.memory(
                        batch_ref=rnn_input, shape=[-1, hidden_size])
                    pre_cell = rnn.memory(
                        batch_ref=rnn_input, shape=[-1, hidden_size])

                new_hidden, new_cell = unit_list[i](step_input, pre_hidden,
                                                    pre_cell)

                if mask:
                    new_hidden = layers.elementwise_mul(
                        new_hidden, step_mask, axis=0) - layers.elementwise_mul(
                            pre_hidden, (step_mask - 1), axis=0)
                    new_cell = layers.elementwise_mul(
                        new_cell, step_mask, axis=0) - layers.elementwise_mul(
                            pre_cell, (step_mask - 1), axis=0)

                rnn.update_memory(pre_hidden, new_hidden)
                rnn.update_memory(pre_cell, new_cell)

                rnn.step_output(new_hidden)
                rnn.step_output(new_cell)

                step_input = new_hidden
                if dropout_prob != None and dropout_prob > 0.0:
                    step_input = layers.dropout(
                        step_input,
                        dropout_prob=dropout_prob,
                        dropout_implementation='upscale_in_train')

            rnn.step_output(step_input)

        rnn_out = rnn()

        last_hidden_array = []
        last_cell_array = []
        rnn_output = rnn_out[-1]
        for i in range(num_layers):
            last_hidden = rnn_out[i * 2]
            last_hidden = last_hidden[-1]
            last_hidden_array.append(last_hidden)
            last_cell = rnn_out[i * 2 + 1]
            last_cell = last_cell[-1]
            last_cell_array.append(last_cell)

        last_hidden_output = layers.concat(last_hidden_array, axis=0)
        last_hidden_output = layers.reshape(
            last_hidden_output, shape=[num_layers, -1, hidden_size])
        last_cell_output = layers.concat(last_cell_array, axis=0)
        last_cell_output = layers.reshape(
            last_cell_output, shape=[num_layers, -1, hidden_size])

        return rnn_output, last_hidden_output, last_cell_output
        # seq_len, batch_size, hidden_size

    fw_rnn_out, fw_last_hidden, fw_last_cell = get_single_direction_output(
        input, fw_unit_list, mask, direc_index=0)

    if bidirectional:
        bw_input = layers.reverse(input, axis=[0])
        bw_mask = None
        if mask:
            bw_mask = layers.reverse(mask, axis=[0])
        bw_rnn_out, bw_last_hidden, bw_last_cell = get_single_direction_output(
            bw_input, bw_unit_list, bw_mask, direc_index=1)

        bw_rnn_out = layers.reverse(bw_rnn_out, axis=[0])

        rnn_out = layers.concat([fw_rnn_out, bw_rnn_out], axis=2)
        last_hidden = layers.concat([fw_last_hidden, bw_last_hidden], axis=1)
        last_hidden = layers.reshape(
            last_hidden, shape=[num_layers * direc_num, -1, hidden_size])

        last_cell = layers.concat([fw_last_cell, bw_last_cell], axis=1)
        last_cell = layers.reshape(
            last_cell, shape=[num_layers * direc_num, -1, hidden_size])

        if batch_first:
            rnn_out = layers.transpose(rnn_out, [1, 0, 2])
        return rnn_out, last_hidden, last_cell
    else:

        rnn_out = fw_rnn_out
        last_hidden = fw_last_hidden
        last_cell = fw_last_cell

        if batch_first:
            rnn_out = layers.transpose(rnn_out, [1, 0, 2])

        return rnn_out, last_hidden, last_cell


class BasicLSTMUnit(Layer):
    r"""
    ****
    BasicLSTMUnit class, Using basic operator to build LSTM
    The algorithm can be described as the code below.

        .. math::

           i_t &= \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_i)

           f_t &= \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_f + forget_bias )

           o_t &= \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_o)

           \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + b_c)

           c_t &= f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}

           h_t &= o_t \odot tanh(c_t)

        - $W$ terms denote weight matrices (e.g. $W_{ix}$ is the matrix
          of weights from the input gate to the input)
        - The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
        - sigmoid is the logistic sigmoid function.
        - $i, f, o$ and $c$ are the input gate, forget gate, output gate,
          and cell activation vectors, respectively, all of which have the same size as
          the cell output activation vector $h$.
        - The :math:`\odot` is the element-wise product of the vectors.
        - :math:`tanh` is the activation functions.
        - :math:`\\tilde{c_t}` is also called candidate hidden state,
          which is computed based on the current input and the previous hidden state.

    Args:
        name_scope(string) : The name scope used to identify parameter and bias name
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of LSTM unit.
            If it is set to None or one attribute of ParamAttr, lstm_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized as zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cells (actNode).
                             Default: 'fluid.layers.tanh'
        forget_bias(float|1.0): forget bias used when computing forget gate
        dtype(string): data type used in this unit

    Examples:

        .. code-block:: python

            import paddle.fluid.layers as layers
            from paddle.fluid.contrib.layers import BasicLSTMUnit

            input_size = 128
            hidden_size = 256
            input = layers.data( name = "input", shape = [-1, input_size], dtype='float32')
            pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')
            pre_cell = layers.data( name = "pre_cell", shape=[-1, hidden_size], dtype='float32')

            lstm_unit = BasicLSTMUnit( "gru_unit", hidden_size)

            new_hidden, new_cell = lstm_unit( input, pre_hidden, pre_cell )

    """

    def __init__(self,
                 name_scope,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype='float32'):
        super(BasicLSTMUnit, self).__init__(name_scope, dtype)
        # reserve old school _full_name and _helper for static graph save load
        self._full_name = unique_name.generate(name_scope + "/" +
                                               self.__class__.__name__)
        self._helper = LayerObjectHelper(self._full_name)

        self._name = name_scope
        self._hiden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._forget_bias = layers.fill_constant(
            [1], dtype=dtype, value=forget_bias)
        self._forget_bias.stop_gradient = False
        self._dtype = dtype

    def _build_once(self, input, pre_hidden, pre_cell):
        self._input_size = input.shape[-1]
        assert (self._input_size > 0)

        self._weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._input_size + self._hiden_size, 4 * self._hiden_size],
            dtype=self._dtype)

        self._bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[4 * self._hiden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, pre_hidden, pre_cell):
        concat_input_hidden = layers.concat([input, pre_hidden], 1)
        gate_input = layers.matmul(x=concat_input_hidden, y=self._weight)

        gate_input = layers.elementwise_add(gate_input, self._bias)
        i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)
        new_cell = layers.elementwise_add(
            layers.elementwise_mul(
                pre_cell,
                layers.sigmoid(layers.elementwise_add(f, self._forget_bias))),
            layers.elementwise_mul(layers.sigmoid(i), layers.tanh(j)))
        new_hidden = layers.tanh(new_cell) * layers.sigmoid(o)

        return new_hidden, new_cell
