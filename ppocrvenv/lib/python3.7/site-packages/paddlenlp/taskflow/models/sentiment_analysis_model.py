# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from paddlenlp.seq2vec.encoder import BoWEncoder, LSTMEncoder
from paddlenlp.transformers import SkepPretrainedModel


class BoWModel(nn.Layer):
    """
    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    Args:
        vocab_size(int): The vocab size that used to create the embedding.
        num_class(int): The num class of the classifier.
        emb_dim(int. optinal): The size of the embedding, default value is 128.
        padding_idx(int, optinal): The padding value in the embedding, the padding_idx of embedding value will 
            not be updated, the default value is 0.
        hidden_size(int, optinal): The output size of linear that after the bow, default value is 128. 
        fc_hidden_size(int, optinal): The output size of linear that after the fisrt linear, default value is 96. 
    """

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
        self.bow_encoder = BoWEncoder(emb_dim)
        self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
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


class LSTMModel(nn.Layer):
    """
    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    Args:
        vocab_size(int): The vocab size that used to create the embedding.
        num_class(int):  The num clas of the classifier.
        emb_dim(int. optinal): The size of the embedding, default value is 128.
        padding_idx(int, optinal): The padding value in the embedding, the padding_idx of embedding value will 
            not be updated, the default value is 0.
        lstm_hidden_size(int, optinal): The output size of the lstm, defalut value 198. 
        direction(string, optinal): The direction of lstm, default value is `forward`. 
        lstm_layers(string, optinal): The num of lstm layer. 
        dropout(float, optinal): The dropout rate of lstm. 
        pooling_type(float, optinal): The pooling type of lstm. Defalut value is None,
            if `pooling_type` is None, then the LSTMEncoder will return the hidden state of the last time step at last layer as a single vector.
    """

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
        self.lstm_encoder = LSTMEncoder(
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
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        return idx, probs


class SkepSequenceModel(SkepPretrainedModel):
    def __init__(self, skep, num_classes=2, dropout=None):
        super(SkepSequenceModel, self).__init__()
        self.num_classes = num_classes
        self.skep = skep  # allow skep to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.skep.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.skep.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1)
        return idx, probs
