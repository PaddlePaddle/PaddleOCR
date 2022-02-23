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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import weight_norm

__all__ = [
    'BoWEncoder', 'CNNEncoder', 'GRUEncoder', 'LSTMEncoder', 'RNNEncoder',
    'TCNEncoder'
]


class BoWEncoder(nn.Layer):
    r"""
    A `BoWEncoder` takes as input a sequence of vectors and returns a
    single vector, which simply sums the embeddings of a sequence across the time dimension. 
    The input to this encoder is of shape `(batch_size, num_tokens, emb_dim)`, 
    and the output is of shape `(batch_size, emb_dim)`.

    Args:
        emb_dim(int): 
            The dimension of each vector in the input sequence.

    Example:
        .. code-block::

            import paddle
            import paddle.nn as nn
            import paddlenlp as nlp

            class BoWModel(nn.Layer):
                def __init__(self,
                            vocab_size,
                            num_classes,
                            emb_dim=128,
                            padding_idx=0,
                            hidden_size=128,
                            fc_hidden_size=96):
                    super().__init__()
                    self.embedder = nn.Embedding(
                        vocab_size, emb_dim, padding_idx=padding_idx)
                    self.bow_encoder = nlp.seq2vec.BoWEncoder(emb_dim)
                    self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
                    self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
                    self.output_layer = nn.Linear(fc_hidden_size, num_classes)

                def forward(self, text):
                    # Shape: (batch_size, num_tokens, embedding_dim)
                    embedded_text = self.embedder(text)

                    # Shape: (batch_size, embedding_dim)
                    summed = self.bow_encoder(embedded_text)
                    encoded_text = paddle.tanh(summed)

                    # Shape: (batch_size, hidden_size)
                    fc1_out = paddle.tanh(self.fc1(encoded_text))
                    # Shape: (batch_size, fc_hidden_size)
                    fc2_out = paddle.tanh(self.fc2(fc1_out))
                    # Shape: (batch_size, num_classes)
                    logits = self.output_layer(fc2_out)
                    return logits

            model = BoWModel(vocab_size=100, num_classes=2)

            text = paddle.randint(low=1, high=10, shape=[1,10], dtype='int32')
            logits = model(text)
    """

    def __init__(self, emb_dim):
        super().__init__()
        self._emb_dim = emb_dim

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `BoWEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._emb_dim

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `BoWEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._emb_dim

    def forward(self, inputs, mask=None):
        r"""
        It simply sums the embeddings of a sequence across the time dimension.

        Args:
            inputs (Tensor):
                Shape as `(batch_size, num_tokens, emb_dim)` and dtype as `float32` or `float64`.
                The sequence length of the input sequence.
            mask (Tensor, optional): 
                Shape same as `inputs`. 
                Its each elements identify whether the corresponding input token is padding or not. 
                If True, not padding token. If False, padding token.
                Defaults to `None`.

        Returns:
            Tensor: 
                Returns tensor `summed`, the result vector of BagOfEmbedding.
                Its data type is same as `inputs` and its shape is `[batch_size, emb_dim]`.
        """
        if mask is not None:
            inputs = inputs * mask

        # Shape: (batch_size, embedding_dim)
        summed = inputs.sum(axis=1)
        return summed


class CNNEncoder(nn.Layer):
    r"""
    A `CNNEncoder` takes as input a sequence of vectors and returns a
    single vector, a combination of multiple convolution layers and max pooling layers.
    The input to this encoder is of shape `(batch_size, num_tokens, emb_dim)`, 
    and the output is of shape `(batch_size, ouput_dim)` or `(batch_size, len(ngram_filter_sizes) * num_filter)`.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filter. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is `len(ngram_filter_sizes) * num_filter`.  This then gets
    (optionally) projected down to a lower dimensional output, specified by `output_dim`.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to `A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification <https://arxiv.org/abs/1510.03820>`__ , 
    Zhang and Wallace 2016, particularly Figure 1.

    Args:
        emb_dim(int):
            The dimension of each vector in the input sequence.
        num_filter(int):
            This is the output dim for each convolutional layer, which is the number of "filters"
            learned by that layer.
        ngram_filter_sizes(Tuple[int], optinal):
            This specifies both the number of convolutional layers we will create and their sizes.  The
            default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
            ngrams of size 2 to 5 with some number of filters.
        conv_layer_activation(Layer, optional):
            Activation to use after the convolution layers.
            Defaults to `paddle.nn.Tanh()`.
        output_dim(int, optional):
            After doing convolutions and pooling, we'll project the collected features into a vector of
            this size.  If this value is `None`, we will just return the result of the max pooling,
            giving an output of shape `len(ngram_filter_sizes) * num_filter`.
            Defaults to `None`.

    Example:
        .. code-block::

            import paddle
            import paddle.nn as nn
            import paddlenlp as nlp

            class CNNModel(nn.Layer):
                def __init__(self,
                            vocab_size,
                            num_classes,
                            emb_dim=128,
                            padding_idx=0,
                            num_filter=128,
                            ngram_filter_sizes=(3, ),
                            fc_hidden_size=96):
                    super().__init__()
                    self.embedder = nn.Embedding(
                        vocab_size, emb_dim, padding_idx=padding_idx)
                    self.encoder = nlp.seq2vec.CNNEncoder(
                        emb_dim=emb_dim,
                        num_filter=num_filter,
                        ngram_filter_sizes=ngram_filter_sizes)
                    self.fc = nn.Linear(self.encoder.get_output_dim(), fc_hidden_size)
                    self.output_layer = nn.Linear(fc_hidden_size, num_classes)

                def forward(self, text):
                    # Shape: (batch_size, num_tokens, embedding_dim)
                    embedded_text = self.embedder(text)
                    # Shape: (batch_size, len(ngram_filter_sizes)*num_filter)
                    encoder_out = self.encoder(embedded_text)
                    encoder_out = paddle.tanh(encoder_out)
                    # Shape: (batch_size, fc_hidden_size)
                    fc_out = self.fc(encoder_out)
                    # Shape: (batch_size, num_classes)
                    logits = self.output_layer(fc_out)
                    return logits

            model = CNNModel(vocab_size=100, num_classes=2)

            text = paddle.randint(low=1, high=10, shape=[1,10], dtype='int32')
            logits = model(text)
    """

    def __init__(self,
                 emb_dim,
                 num_filter,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=nn.Tanh(),
                 output_dim=None,
                 **kwargs):
        super().__init__()
        self._emb_dim = emb_dim
        self._num_filter = num_filter
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._output_dim = output_dim

        self.convs = paddle.nn.LayerList([
            nn.Conv2D(
                in_channels=1,
                out_channels=self._num_filter,
                kernel_size=(i, self._emb_dim),
                **kwargs) for i in self._ngram_filter_sizes
        ])

        maxpool_output_dim = self._num_filter * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim,
                                              self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `CNNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._emb_dim

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `CNNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._output_dim

    def forward(self, inputs, mask=None):
        r"""
        The combination of multiple convolution layers and max pooling layers.

        Args:
            inputs (Tensor): 
                Shape as `(batch_size, num_tokens, emb_dim)` and dtype as `float32` or `float64`.
                Tensor containing the features of the input sequence. 
            mask (Tensor, optional): 
                Shape shoule be same as `inputs` and dtype as `int32`, `int64`, `float32` or `float64`. 
                Its each elements identify whether the corresponding input token is padding or not. 
                If True, not padding token. If False, padding token. 
                Defaults to `None`.

        Returns:
            Tensor:
                Returns tensor `result`.
                If output_dim is None, the result shape is of `(batch_size, output_dim)` and 
                dtype is `float`; If not, the result shape is of `(batch_size, len(ngram_filter_sizes) * num_filter)`.

        """
        if mask is not None:
            inputs = inputs * mask

        # Shape: (batch_size, 1, num_tokens, emb_dim) = (N, C, H, W)
        inputs = inputs.unsqueeze(1)

        # If output_dim is None, result shape of (batch_size, len(ngram_filter_sizes) * num_filter));
        # else, result shape of (batch_size, output_dim).
        convs_out = [
            self._activation(conv(inputs)).squeeze(3) for conv in self.convs
        ]
        maxpool_out = [
            F.adaptive_max_pool1d(
                t, output_size=1).squeeze(2) for t in convs_out
        ]
        result = paddle.concat(maxpool_out, axis=1)

        if self.projection_layer is not None:
            result = self.projection_layer(result)
        return result


class GRUEncoder(nn.Layer):
    r"""
    A GRUEncoder takes as input a sequence of vectors and returns a
    single vector, which is a combination of multiple `paddle.nn.GRU 
    <https://www.paddlepaddle.org.cn/documentation/docs/en/api
    /paddle/nn/layer/rnn/GRU_en.html>`__ subclass.
    The input to this encoder is of shape `(batch_size, num_tokens, input_size)`, 
    The output is of shape `(batch_size, hidden_size * 2)` if GRU is bidirection;
    If not, output is of shape `(batch_size, hidden_size)`.

    Paddle's GRU have two outputs: the hidden state for every time step at last layer, 
    and the hidden state at the last time step for every layer.
    If `pooling_type` is not None, we perform the pooling on the hidden state of every time 
    step at last layer to create a single vector. If None, we use the hidden state 
    of the last time step at last layer as a single output (shape of `(batch_size, hidden_size)`); 
    And if direction is bidirection, the we concat the hidden state of the last forward 
    gru and backward gru layer to create a single vector (shape of `(batch_size, hidden_size * 2)`).

    Args:
        input_size (int): 
            The number of expected features in the input (the last dimension).
        hidden_size (int): 
            The number of features in the hidden state.
        num_layers (int, optional): 
            Number of recurrent layers. 
            E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, 
            with the second GRU taking in outputs of the first GRU and computing the final results.
            Defaults to 1.
        direction (str, optional): 
            The direction of the network. It can be "forward" and "bidirect" 
            (it means bidirection network). If "bidirect", it is a birectional GRU, 
            and returns the concat output from both directions.
            Defaults to "forward".
        dropout (float, optional): 
            If non-zero, introduces a Dropout layer on the outputs of each GRU layer 
            except the last layer, with dropout probability equal to dropout.
            Defaults to 0.0.
        pooling_type (str, optional): 
            If `pooling_type` is None, then the GRUEncoder will return the hidden state of
            the last time step at last layer as a single vector.
            If pooling_type is not None, it must be one of "sum", "max" and "mean". 
            Then it will be pooled on the GRU output (the hidden state of every time 
            step at last layer) to create a single vector.
            Defaults to `None`

    Example:
        .. code-block::

            import paddle
            import paddle.nn as nn
            import paddlenlp as nlp

            class GRUModel(nn.Layer):
                def __init__(self,
                            vocab_size,
                            num_classes,
                            emb_dim=128,
                            padding_idx=0,
                            gru_hidden_size=198,
                            direction='forward',
                            gru_layers=1,
                            dropout_rate=0.0,
                            pooling_type=None,
                            fc_hidden_size=96):
                    super().__init__()
                    self.embedder = nn.Embedding(
                        num_embeddings=vocab_size,
                        embedding_dim=emb_dim,
                        padding_idx=padding_idx)
                    self.gru_encoder = nlp.seq2vec.GRUEncoder(
                        emb_dim,
                        gru_hidden_size,
                        num_layers=gru_layers,
                        direction=direction,
                        dropout=dropout_rate,
                        pooling_type=pooling_type)
                    self.fc = nn.Linear(self.gru_encoder.get_output_dim(), fc_hidden_size)
                    self.output_layer = nn.Linear(fc_hidden_size, num_classes)

                def forward(self, text, seq_len):
                    # Shape: (batch_size, num_tokens, embedding_dim)
                    embedded_text = self.embedder(text)
                    # Shape: (batch_size, num_tokens, num_directions*gru_hidden_size)
                    # num_directions = 2 if direction is 'bidirect'
                    # if not, num_directions = 1
                    text_repr = self.gru_encoder(embedded_text, sequence_length=seq_len)
                    # Shape: (batch_size, fc_hidden_size)
                    fc_out = paddle.tanh(self.fc(text_repr))
                    # Shape: (batch_size, num_classes)
                    logits = self.output_layer(fc_out)
                    return logits

            model = GRUModel(vocab_size=100, num_classes=2)

            text = paddle.randint(low=1, high=10, shape=[1,10], dtype='int32')
            seq_len = paddle.to_tensor([10])
            logits = model(text, seq_len)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.gru_layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                direction=direction,
                                dropout=dropout,
                                **kwargs)

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `GRUEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `GRUEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        if self._direction == "bidirect":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        r"""
        GRUEncoder takes the a sequence of vectors and and returns a single vector, 
        which is a combination of multiple GRU layers. The input to this 
        encoder is of shape `(batch_size, num_tokens, input_size)`, 
        The output is of shape `(batch_size, hidden_size * 2)` if GRU is bidirection;
        If not, output is of shape `(batch_size, hidden_size)`.

        Args:
            inputs (Tensor): Shape as `(batch_size, num_tokens, input_size)`.
                Tensor containing the features of the input sequence. 
            sequence_length (Tensor): Shape as `(batch_size)`.
                The sequence length of the input sequence.

        Returns:
            Tensor: Returns tensor `output`, the hidden state at the last time step for every layer.
            Its data type is `float` and its shape is `[batch_size, hidden_size]`.

        """
        encoded_text, last_hidden = self.gru_layer(
            inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If gru is not bidirection, then output is the hidden state of the last time step 
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If gru is bidirection, then output is concatenation of the forward and backward hidden state 
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size * 2)`.
            if self._direction != 'bidirect':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # The output shape is `(batch_size, hidden_size * 2)` if use bidirectional GRU, 
            # otherwise the output shape is `(batch_size, hidden_size * 2)`.
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
        return output


class LSTMEncoder(nn.Layer):
    r"""
    An LSTMEncoder takes as input a sequence of vectors and returns a
    single vector, which is a combination of multiple `paddle.nn.LSTM
    <https://www.paddlepaddle.org.cn/documentation/docs/en/api
    /paddle/nn/layer/rnn/LSTM_en.html>`__ subclass.
    The input to this encoder is of shape `(batch_size, num_tokens, input_size)`.
    The output is of shape `(batch_size, hidden_size * 2)` if LSTM is bidirection;
    If not, output is of shape `(batch_size, hidden_size)`.

    Paddle's LSTM have two outputs: the hidden state for every time step at last layer, 
    and the hidden state and cell at the last time step for every layer.
    If `pooling_type` is not None, we perform the pooling on the hidden state of every time 
    step at last layer to create a single vector. If None, we use the hidden state 
    of the last time step at last layer as a single output (shape of `(batch_size, hidden_size)`); 
    And if direction is bidirection, the we concat the hidden state of the last forward 
    lstm and backward lstm layer to create a single vector (shape of `(batch_size, hidden_size * 2)`).

    Args:
        input_size (int): 
            The number of expected features in the input (the last dimension).
        hidden_size (int): 
            The number of features in the hidden state.
        num_layers (int, optional):
            Number of recurrent layers. 
            E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, 
            with the second LSTM taking in outputs of the first LSTM and computing the final results.
            Defaults to 1.
        direction (str, optional): 
            The direction of the network. It can be "forward" or "bidirect" (it means bidirection network).
            If "bidirect", it is a birectional LSTM, and returns the concat output from both directions.
            Defaults to "forward".
        dropout (float, optional): 
            If non-zero, introduces a Dropout layer on the outputs of each LSTM layer 
            except the last layer, with dropout probability equal to dropout.
            Defaults to 0.0 .
        pooling_type (str, optional):
            If `pooling_type` is None, then the LSTMEncoder will return 
            the hidden state of the last time step at last layer as a single vector.
            If pooling_type is not None, it must be one of "sum", "max" and "mean". 
            Then it will be pooled on the LSTM output (the hidden state of every 
            time step at last layer) to create a single vector.
            Defaults to `None`.

    Example:
        .. code-block::

            import paddle
            import paddle.nn as nn
            import paddlenlp as nlp

            class LSTMModel(nn.Layer):
                def __init__(self,
                            vocab_size,
                            num_classes,
                            emb_dim=128,
                            padding_idx=0,
                            lstm_hidden_size=198,
                            direction='forward',
                            lstm_layers=1,
                            dropout_rate=0.0,
                            pooling_type=None,
                            fc_hidden_size=96):
                    super().__init__()
                    self.embedder = nn.Embedding(
                        num_embeddings=vocab_size,
                        embedding_dim=emb_dim,
                        padding_idx=padding_idx)
                    self.lstm_encoder = nlp.seq2vec.LSTMEncoder(
                        emb_dim,
                        lstm_hidden_size,
                        num_layers=lstm_layers,
                        direction=direction,
                        dropout=dropout_rate,
                        pooling_type=pooling_type)
                    self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)
                    self.output_layer = nn.Linear(fc_hidden_size, num_classes)

                def forward(self, text, seq_len):
                    # Shape: (batch_size, num_tokens, embedding_dim)
                    embedded_text = self.embedder(text)
                    # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
                    # num_directions = 2 if direction is 'bidirect'
                    # if not, num_directions = 1
                    text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
                    # Shape: (batch_size, fc_hidden_size)
                    fc_out = paddle.tanh(self.fc(text_repr))
                    # Shape: (batch_size, num_classes)
                    logits = self.output_layer(fc_out)
                    return logits

            model = LSTMModel(vocab_size=100, num_classes=2)

            text = paddle.randint(low=1, high=10, shape=[1,10], dtype='int32')
            seq_len = paddle.to_tensor([10])
            logits = model(text, seq_len)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction=direction,
            dropout=dropout,
            **kwargs)

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `LSTMEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `LSTMEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        if self._direction == "bidirect":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        r"""
        LSTMEncoder takes the a sequence of vectors and and returns a
        single vector, which is a combination of multiple LSTM layers.
        The input to this encoder is of shape `(batch_size, num_tokens, input_size)`, 
        The output is of shape `(batch_size, hidden_size * 2)` if LSTM is bidirection;
        If not, output is of shape `(batch_size, hidden_size)`.

        Args:
            inputs (Tensor): Shape as `(batch_size, num_tokens, input_size)`.
                Tensor containing the features of the input sequence. 
            sequence_length (Tensor): Shape as `(batch_size)`.
                The sequence length of the input sequence.

        Returns:
            Tensor: Returns tensor `output`, the hidden state at the last time step for every layer.
            Its data type is `float` and its shape is `[batch_size, hidden_size]`.

        """
        encoded_text, (last_hidden, last_cell) = self.lstm_layer(
            inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If lstm is not bidirection, then output is the hidden state of the last time step 
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If lstm is bidirection, then output is concatenation of the forward and backward hidden state 
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size * 2)`.
            if self._direction != 'bidirect':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # The output shape is `(batch_size, hidden_size * 2)` if use bidirectional LSTM, 
            # otherwise the output shape is `(batch_size, hidden_size * 2)`.
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
        return output


class RNNEncoder(nn.Layer):
    r"""
    A RNNEncoder takes as input a sequence of vectors and returns a
    single vector, which is a combination of multiple `paddle.nn.RNN
    <https://www.paddlepaddle.org.cn/documentation/docs/en/api
    /paddle/nn/layer/rnn/RNN_en.html>`__ subclass.
    The input to this encoder is of shape `(batch_size, num_tokens, input_size)`, 
    The output is of shape `(batch_size, hidden_size * 2)` if RNN is bidirection;
    If not, output is of shape `(batch_size, hidden_size)`.

    Paddle's RNN have two outputs: the hidden state for every time step at last layer, 
    and the hidden state at the last time step for every layer.
    If `pooling_type` is not None, we perform the pooling on the hidden state of every time 
    step at last layer to create a single vector. If None, we use the hidden state 
    of the last time step at last layer as a single output (shape of `(batch_size, hidden_size)`); 
    And if direction is bidirection, the we concat the hidden state of the last forward 
    rnn and backward rnn layer to create a single vector (shape of `(batch_size, hidden_size * 2)`).

    Args:
        input_size (int): 
            The number of expected features in the input (the last dimension).
        hidden_size (int): 
            The number of features in the hidden state.
        num_layers (int, optional): 
            Number of recurrent layers. 
            E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, 
            with the second RNN taking in outputs of the first RNN and computing the final results.
            Defaults to 1.
        direction (str, optional): 
            The direction of the network. It can be "forward" and "bidirect" 
            (it means bidirection network). If "biderect", it is a birectional RNN, 
            and returns the concat output from both directions. Defaults to "forward"
        dropout (float, optional): 
            If non-zero, introduces a Dropout layer on the outputs of each RNN layer 
            except the last layer, with dropout probability equal to dropout.
            Defaults to 0.0.
        pooling_type (str, optional):
            If `pooling_type` is None, then the RNNEncoder will return the hidden state
            of the last time step at last layer as a single vector.
            If pooling_type is not None, it must be one of "sum", "max" and "mean". 
            Then it will be pooled on the RNN output (the hidden state of every time 
            step at last layer) to create a single vector.
            Defaults to `None`.

    Example:
        .. code-block::

            import paddle
            import paddle.nn as nn
            import paddlenlp as nlp

            class RNNModel(nn.Layer):
                def __init__(self,
                            vocab_size,
                            num_classes,
                            emb_dim=128,
                            padding_idx=0,
                            rnn_hidden_size=198,
                            direction='forward',
                            rnn_layers=1,
                            dropout_rate=0.0,
                            pooling_type=None,
                            fc_hidden_size=96):
                    super().__init__()
                    self.embedder = nn.Embedding(
                        num_embeddings=vocab_size,
                        embedding_dim=emb_dim,
                        padding_idx=padding_idx)
                    self.rnn_encoder = nlp.seq2vec.RNNEncoder(
                        emb_dim,
                        rnn_hidden_size,
                        num_layers=rnn_layers,
                        direction=direction,
                        dropout=dropout_rate,
                        pooling_type=pooling_type)
                    self.fc = nn.Linear(self.rnn_encoder.get_output_dim(), fc_hidden_size)
                    self.output_layer = nn.Linear(fc_hidden_size, num_classes)

                def forward(self, text, seq_len):
                    # Shape: (batch_size, num_tokens, embedding_dim)
                    embedded_text = self.embedder(text)
                    # Shape: (batch_size, num_tokens, num_directions*rnn_hidden_size)
                    # num_directions = 2 if direction is 'bidirect'
                    # if not, num_directions = 1
                    text_repr = self.rnn_encoder(embedded_text, sequence_length=seq_len)
                    # Shape: (batch_size, fc_hidden_size)
                    fc_out = paddle.tanh(self.fc(text_repr))
                    # Shape: (batch_size, num_classes)
                    logits = self.output_layer(fc_out)
                    return logits

            model = RNNModel(vocab_size=100, num_classes=2)

            text = paddle.randint(low=1, high=10, shape=[1,10], dtype='int32')
            seq_len = paddle.to_tensor([10])
            logits = model(text, seq_len)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.rnn_layer = nn.SimpleRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction=direction,
            dropout=dropout,
            **kwargs)

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `RNNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `RNNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        if self._direction == "bidirect":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        r"""
        RNNEncoder takes the a sequence of vectors and and returns a
        single vector, which is a combination of multiple RNN layers.
        The input to this encoder is of shape `(batch_size, num_tokens, input_size)`.
        The output is of shape `(batch_size, hidden_size * 2)` if RNN is bidirection;
        If not, output is of shape `(batch_size, hidden_size)`.

        Args:
            inputs (Tensor): Shape as `(batch_size, num_tokens, input_size)`.
                Tensor containing the features of the input sequence. 
            sequence_length (Tensor): Shape as `(batch_size)`.
                The sequence length of the input sequence.

        Returns:
            Tensor: Returns tensor `output`, the hidden state at the last time step for every layer.
            Its data type is `float` and its shape is `[batch_size, hidden_size]`.

        """
        encoded_text, last_hidden = self.rnn_layer(
            inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If rnn is not bidirection, then output is the hidden state of the last time step 
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If rnn is bidirection, then output is concatenation of the forward and backward hidden state 
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size * 2)`.
            if self._direction != 'bidirect':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # The output shape is `(batch_size, hidden_size * 2)` if use bidirectional RNN, 
            # otherwise the output shape is `(batch_size, hidden_size * 2)`.
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise RuntimeError(
                    "Unexpected pooling type %s ."
                    "Pooling type must be one of sum, max and mean." %
                    self._pooling_type)
        return output


class Chomp1d(nn.Layer):
    """
    Remove the elements on the right.

    Args:
        chomp_size (int): The number of elements removed.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Layer):
    """
    The TCN block, consists of dilated causal conv, relu and residual block. 
    See the Figure 1(b) in https://arxiv.org/pdf/1803.01271.pdf for more details.

    Args:
        n_inputs ([int]): The number of channels in the input tensor.
        n_outputs ([int]): The number of filters.
        kernel_size ([int]): The filter size.
        stride ([int]): The stride size.
        dilation ([int]): The dilation size.
        padding ([int]): The size of zeros to be padded.
        dropout (float, optional): Probability of dropout the units. Defaults to 0.2.
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1D(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        # Chomp1d is used to make sure the network is causal.
        # We pad by (k-1)*d on the two sides of the input for convolution, 
        # and then use Chomp1d to remove the (k-1)*d output elements on the right.
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1D(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2)
        self.downsample = nn.Conv1D(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.set_value(
            paddle.tensor.normal(0.0, 0.01, self.conv1.weight.shape))
        self.conv2.weight.set_value(
            paddle.tensor.normal(0.0, 0.01, self.conv2.weight.shape))
        if self.downsample is not None:
            self.downsample.weight.set_value(
                paddle.tensor.normal(0.0, 0.01, self.downsample.weight.shape))

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Layer):
    r"""
    A `TCNEncoder` takes as input a sequence of vectors and returns a
    single vector, which is the last one time step in the feature map. 
    The input to this encoder is of shape `(batch_size, num_tokens, input_size)`, 
    and the output is of shape `(batch_size, num_channels[-1])` with a receptive 
    filed:
    
    .. math::
    
        receptive filed = 2 * \sum_{i=0}^{len(num\_channels)-1}2^i(kernel\_size-1).
    
    Temporal Convolutional Networks is a simple convolutional architecture. It outperforms canonical recurrent networks
    such as LSTMs in many tasks. See https://arxiv.org/pdf/1803.01271.pdf for more details.

    Args:
        input_size (int): The number of expected features in the input (the last dimension).
        num_channels (list): The number of channels in different layer. 
        kernel_size (int): The kernel size. Defaults to 2.
        dropout (float): The dropout probability. Defaults to 0.2.
    """

    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()
        self._input_size = input_size
        self._output_dim = num_channels[-1]

        layers = nn.LayerList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout))

        self.network = nn.Sequential(*layers)

    def get_input_dim(self):
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `TCNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        """
        Returns the dimension of the final vector output by this `TCNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._output_dim

    def forward(self, inputs):
        r"""
        TCNEncoder takes as input a sequence of vectors and returns a
        single vector, which is the last one time step in the feature map. 
        The input to this encoder is of shape `(batch_size, num_tokens, input_size)`, 
        and the output is of shape `(batch_size, num_channels[-1])` with a receptive 
        filed:
    
        .. math::
        
            receptive filed = 2 * \sum_{i=0}^{len(num\_channels)-1}2^i(kernel\_size-1).

        Args:
            inputs (Tensor): The input tensor with shape `[batch_size, num_tokens, input_size]`.

        Returns:
            Tensor: Returns tensor `output` with shape `[batch_size, num_channels[-1]]`.
        """
        inputs_t = inputs.transpose([0, 2, 1])
        output = self.network(inputs_t).transpose([2, 0, 1])[-1]
        return output
