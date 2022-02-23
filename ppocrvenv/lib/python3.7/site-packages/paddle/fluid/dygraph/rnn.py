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

from . import Layer
from ..layers import sigmoid, tanh, concat, fill_constant, matmul, elementwise_add, elementwise_mul, split
import copy

__all__ = ['LSTMCell', 'GRUCell']


class LSTMCell(Layer):
    r"""
    LSTMCell implementation using basic operators.
    There are two LSTMCell version, the default one is compatible with CUDNN LSTM implementation.
    The algorithm can be described as the equations below.

        .. math::

            i_t &= sigmoid(W_{ix}x_{t} + W_{ih}h_{t-1} + bx_i + bh_i)

            f_t &= sigmoid(W_{fx}x_{t} + W_{fh}h_{t-1} + bx_f + bh_f)

            o_t &= sigmoid(W_{ox}x_{t} + W_{oh}h_{t-1} + bx_o + bh_o)

            \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + bx_c + bh_c)

            c_t &= f_t \\odot c_{t-1} + i_t \\odot \\tilde{c_t}

            h_t &= o_t \\odot tanh(c_t)

    The other LSTMCell version is compatible with the BasicLSTMUnit used in static graph.
    The algorithm can be described as the equations below.

        .. math::

            i_t &= sigmoid(W_{ix}x_{t} + W_{ih}h_{t-1} + b_i)

            f_t &= sigmoid(W_{fx}x_{t} + W_{fh}h_{t-1} + b_f + forget_bias )

            o_t &= sigmoid(W_{ox}x_{t} + W_{oh}h_{t-1} + b_o)

            \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + b_c)

            c_t &= f_t \\odot c_{t-1} + i_t \\odot \\tilde{c_t}

            h_t &= o_t \\odot tanh(c_t)

    Args:
        hidden_size (integer): The hidden size used in the Cell.
        input_size (integer): The input size used in the Cell.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, LSTMCell will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of LSTMCell.
            If it is set to None or one attribute of ParamAttr, LSTMCell will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized as zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cells (actNode).
                             Default: 'fluid.layers.tanh'
        forget_bias(float|1.0): forget bias used when computing forget gate. This 
            is not used in default LSTMCell implementation (CUDNN compatiable)
        use_cudnn_impl(bool|True): whether to use CUDNN compatible LSTMCell
        dtype(string): data type used in this cell
    
    Returns:
        None

    Examples:

        .. code-block:: python

            from paddle import fluid
            import paddle.fluid.core as core
            from paddle.fluid.dygraph import LSTMCell
            import numpy as np
            batch_size = 64
            input_size = 128
            hidden_size = 256
            step_input_np = np.random.uniform(-0.1, 0.1, (
                batch_size, input_size)).astype('float64')
            pre_hidden_np = np.random.uniform(-0.1, 0.1, (
                batch_size, hidden_size)).astype('float64')
            pre_cell_np = np.random.uniform(-0.1, 0.1, (
                batch_size, hidden_size)).astype('float64')
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
            else:
                place = core.CPUPlace()
            with fluid.dygraph.guard(place):
                cudnn_lstm = LSTMCell(hidden_size, input_size)
                step_input_var = fluid.dygraph.to_variable(step_input_np)
                pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)
                pre_cell_var = fluid.dygraph.to_variable(pre_cell_np)
                new_hidden, new_cell = cudnn_lstm(step_input_var, pre_hidden_var, pre_cell_var) 

    """

    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 use_cudnn_impl=True,
                 dtype='float64'):
        super(LSTMCell, self).__init__(dtype)

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype
        self._gate_activation = gate_activation or sigmoid
        self._activation = activation or tanh
        self._use_cudnn_impl = use_cudnn_impl

        if self._use_cudnn_impl:

            if self._param_attr is not None and self._param_attr.name is not None:
                weight_ih_param_attr = copy.deepcopy(self._param_attr)
                weight_hh_param_attr = copy.deepcopy(self._param_attr)
                weight_ih_param_attr.name += "_weight_ih"
                weight_hh_param_attr.name += "_weight_hh"
            else:
                weight_ih_param_attr = self._param_attr
                weight_hh_param_attr = self._param_attr

            if self._bias_attr is not None and self._bias_attr.name is not None:
                bias_ih_param_attr = copy.deepcopy(self._bias_attr)
                bias_hh_param_attr = copy.deepcopy(self._bias_attr)
                bias_ih_param_attr.name += "_bias_ih"
                bias_hh_param_attr.name += "_bias_hh"
            else:
                bias_ih_param_attr = self._bias_attr
                bias_hh_param_attr = self._bias_attr

            self._weight_ih = self.create_parameter(
                attr=weight_ih_param_attr,
                shape=[4 * self._hidden_size, self._input_size],
                dtype=self._dtype)

            self._weight_hh = self.create_parameter(
                attr=weight_hh_param_attr,
                shape=[4 * self._hidden_size, self._hidden_size],
                dtype=self._dtype)

            self._bias_ih = self.create_parameter(
                attr=bias_ih_param_attr,
                shape=[4 * self._hidden_size],
                dtype=self._dtype,
                is_bias=True)
            self._bias_hh = self.create_parameter(
                attr=bias_hh_param_attr,
                shape=[4 * self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        else:

            self._forget_bias = fill_constant(
                [1], dtype=dtype, value=forget_bias)
            self._forget_bias.stop_gradient = False

            self._weight = self.create_parameter(
                attr=self._param_attr,
                shape=[
                    self._input_size + self._hidden_size, 4 * self._hidden_size
                ],
                dtype=dtype)

            self._bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[4 * self._hidden_size],
                dtype=dtype,
                is_bias=True)

    def forward(self, input, pre_hidden, pre_cell):

        if self._use_cudnn_impl:
            igates = matmul(input, y=self._weight_ih, transpose_y=True)
            igates = elementwise_add(igates, self._bias_ih)
            hgates = matmul(pre_hidden, self._weight_hh, transpose_y=True)
            hgates = elementwise_add(hgates, self._bias_hh)

            chunked_igates = split(igates, num_or_sections=4, dim=1)
            chunked_hgates = split(hgates, num_or_sections=4, dim=1)

            ingate = elementwise_add(chunked_igates[0], chunked_hgates[0])
            ingate = self._gate_activation(ingate)

            forgetgate = elementwise_add(chunked_igates[1], chunked_hgates[1])
            forgetgate = self._gate_activation(forgetgate)

            cellgate = elementwise_add(chunked_igates[2], chunked_hgates[2])
            cellgate = self._activation(cellgate)

            outgate = elementwise_add(chunked_igates[3], chunked_hgates[3])
            outgate = self._gate_activation(outgate)

            new_cell = (forgetgate * pre_cell) + (ingate * cellgate)
            new_hidden = outgate * self._activation(new_cell)

        else:

            concat_input_hidden = concat([input, pre_hidden], 1)
            gate_input = matmul(x=concat_input_hidden, y=self._weight)

            gate_input = elementwise_add(gate_input, self._bias)
            i, j, f, o = split(gate_input, num_or_sections=4, dim=-1)
            new_cell = elementwise_add(
                elementwise_mul(pre_cell,
                                self._gate_activation(
                                    elementwise_add(f, self._forget_bias))),
                elementwise_mul(sigmoid(i), tanh(j)))
            new_hidden = self._activation(new_cell) * self._gate_activation(o)

        return new_hidden, new_cell


class GRUCell(Layer):
    r"""
    GRU implementation using basic operators.
    There are two GRUCell version, the default one is compatible with CUDNN GRU implementation.
    The algorithm can be described as the equations below.

        .. math::

            u_t & = sigmoid(W_{ux} x_{t} + b_ux + W_{uh} h_{t-1} + b_uh)

            r_t & = sigmoid(W_{rx} x_{t} + b_rx + W_{rh} h_{t-1} + b_rh)

            \\tilde{h_{t}} & = tanh(W_{cx} x_{t} + b_cx + r_t \\odot (W_{ch} h_{t-1} + b_ch))

            h_t & = u_t h_{t-1} + (1-u_t) \\tilde{h_{t}}

    The other LSTMCell version is compatible with the BasicGRUUnit used in static graph.
    The algorithm can be described as the equations below.

        .. math::

            u_t & = sigmoid(W_{ux} x_{t} + W_{uh} h_{t-1} + b_u)

            r_t & = sigmoid(W_{rx} x_{t} + W_{rh} h_{t-1} + b_r)

            \\tilde{h_{t}} & = tanh(W_{cx} x_{t} + W_{ch} \\odot(r_t, h_{t-1}) + b_m)

            h_t & = u_t h_{t-1} + (1-u_t) \\tilde{h_{t}}

    Args:
        hidden_size (integer): The hidden size used in the Cell.
        input_size (integer): The input size used in the Cell.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, GRUCell will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of GRUCell.
            If it is set to None or one attribute of ParamAttr, GRUCell will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cell (actNode).
                             Default: 'fluid.layers.tanh'
        use_cudnn_impl(bool|True): whether to use CUDNN compatible LSTMCell
        dtype(string): data type used in this cell
    
    Returns:
        None

    Examples:

        .. code-block:: python

            from paddle import fluid
            import paddle.fluid.core as core
            from paddle.fluid.dygraph import GRUCell
            import numpy as np
            batch_size = 64
            input_size = 128
            hidden_size = 256
            step_input_np = np.random.uniform(-0.1, 0.1, (
            batch_size, input_size)).astype('float64')
            pre_hidden_np = np.random.uniform(-0.1, 0.1, (
            batch_size, hidden_size)).astype('float64')
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
            else:
                place = core.CPUPlace()
            with fluid.dygraph.guard(place):
                cudnn_gru = GRUCell(hidden_size, input_size)
                step_input_var = fluid.dygraph.to_variable(step_input_np)
                pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)

    """

    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 use_cudnn_impl=True,
                 dtype='float64'):
        super(GRUCell, self).__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype
        self._gate_activation = gate_activation or sigmoid
        self._activation = activation or tanh
        self._use_cudnn_impl = use_cudnn_impl

        if self._use_cudnn_impl:

            if self._param_attr is not None and self._param_attr.name is not None:
                weight_ih_param_attr = copy.deepcopy(self._param_attr)
                weight_hh_param_attr = copy.deepcopy(self._param_attr)
                weight_ih_param_attr.name += "_weight_ih"
                weight_hh_param_attr.name += "_weight_hh"
            else:
                weight_ih_param_attr = self._param_attr
                weight_hh_param_attr = self._param_attr

            if self._bias_attr is not None and self._bias_attr.name is not None:
                bias_ih_param_attr = copy.deepcopy(self._bias_attr)
                bias_hh_param_attr = copy.deepcopy(self._bias_attr)
                bias_ih_param_attr.name += "_bias_ih"
                bias_hh_param_attr.name += "_bias_hh"
            else:
                bias_ih_param_attr = self._bias_attr
                bias_hh_param_attr = self._bias_attr

            self._weight_ih = self.create_parameter(
                attr=weight_ih_param_attr,
                shape=[3 * self._hidden_size, self._input_size],
                dtype=self._dtype)

            self._weight_hh = self.create_parameter(
                attr=weight_hh_param_attr,
                shape=[3 * self._hidden_size, self._hidden_size],
                dtype=self._dtype)

            self._bias_ih = self.create_parameter(
                attr=bias_ih_param_attr,
                shape=[3 * self._hidden_size],
                dtype=self._dtype,
                is_bias=True)
            self._bias_hh = self.create_parameter(
                attr=bias_hh_param_attr,
                shape=[3 * self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        else:

            if self._param_attr is not None and self._param_attr.name is not None:
                gate_weight_param_attr = copy.deepcopy(self._param_attr)
                candidate_weight_param_attr = copy.deepcopy(self._param_attr)
                gate_weight_param_attr.name += "_gate_weight"
                candidate_weight_param_attr.name += "_candidate_weight"
            else:
                gate_weight_param_attr = self._param_attr
                candidate_weight_param_attr = self._param_attr

            if self._bias_attr is not None and self._bias_attr.name is not None:
                gate_bias_param_attr = copy.deepcopy(self._bias_attr)
                candidate_bias_param_attr = copy.deepcopy(self._bias_attr)
                gate_bias_param_attr.name += "_gate_bias"
                candidate_bias_param_attr.name += "_candidate_bias"
            else:
                gate_bias_param_attr = self._bias_attr
                candidate_bias_param_attr = self._bias_attr

            self._gate_weight = self.create_parameter(
                attr=gate_weight_param_attr,
                shape=[
                    self._input_size + self._hidden_size, 2 * self._hidden_size
                ],
                dtype=dtype)

            self._candidate_weight = self.create_parameter(
                attr=candidate_weight_param_attr,
                shape=[
                    self._input_size + self._hidden_size, self._hidden_size
                ],
                dtype=dtype)

            self._gate_bias = self.create_parameter(
                attr=gate_bias_param_attr,
                shape=[2 * self._hidden_size],
                dtype=dtype,
                is_bias=True)
            self._candidate_bias = self.create_parameter(
                attr=candidate_bias_param_attr,
                shape=[self._hidden_size],
                dtype=dtype,
                is_bias=True)

    def forward(self, input, pre_hidden):

        if self._use_cudnn_impl:

            igates = matmul(input, y=self._weight_ih, transpose_y=True)
            igates = elementwise_add(igates, self._bias_ih)
            hgates = matmul(pre_hidden, self._weight_hh, transpose_y=True)
            hgates = elementwise_add(hgates, self._bias_hh)

            chunked_igates = split(igates, num_or_sections=3, dim=1)
            chunked_hgates = split(hgates, num_or_sections=3, dim=1)

            reset_gate = elementwise_add(chunked_igates[0], chunked_hgates[0])
            reset_gate = self._gate_activation(reset_gate)

            input_gate = elementwise_add(chunked_igates[1], chunked_hgates[1])
            input_gate = self._gate_activation(input_gate)

            _temp = reset_gate * chunked_hgates[2]
            new_gate = elementwise_add(chunked_igates[2], _temp)
            new_gate = self._activation(new_gate)

            new_hidden = (pre_hidden - new_gate) * input_gate + new_gate

        else:

            concat_input_hidden = concat([input, pre_hidden], 1)

            gate_input = matmul(x=concat_input_hidden, y=self._gate_weight)

            gate_input = elementwise_add(gate_input, self._gate_bias)
            gate_input = self._gate_activation(gate_input)
            r, u = split(gate_input, num_or_sections=2, dim=1)

            r_hidden = r * pre_hidden

            candidate = matmul(
                concat([input, r_hidden], 1), self._candidate_weight)
            candidate = elementwise_add(candidate, self._candidate_bias)

            c = self._activation(candidate)
            new_hidden = u * pre_hidden + (1 - u) * c

        return new_hidden
