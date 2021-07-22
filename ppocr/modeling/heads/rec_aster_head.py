# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import paddle
from paddle import nn
from paddle.nn import functional as F


class AsterHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 sDim,
                 attDim,
                 max_len_labels,
                 time_step=25,
                 beam_width=5,
                 **kwargs):
        super(AsterHead, self).__init__()
        self.num_classes = out_channels
        self.in_planes = in_channels
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels
        self.decoder = AttentionRecognitionHead(in_channels, out_channels, sDim,
                                                attDim, max_len_labels)
        self.time_step = time_step
        self.embeder = Embedding(self.time_step, in_channels)
        self.beam_width = beam_width

    def forward(self, x, targets=None, embed=None):
        return_dict = {}
        embedding_vectors = self.embeder(x)
        rec_targets, rec_lengths = targets

        if self.training:
            rec_pred = self.decoder([x, rec_targets, rec_lengths],
                                    embedding_vectors)
            return_dict['rec_pred'] = rec_pred
            return_dict['embedding_vectors'] = embedding_vectors
        else:
            rec_pred, rec_pred_scores = self.decoder.beam_search(
                x, self.beam_width, self.eos, embedding_vectors)
            return_dict['rec_pred'] = rec_pred
            return_dict['rec_pred_scores'] = rec_pred_scores
            return_dict['embedding_vectors'] = embedding_vectors

        return return_dict


class Embedding(nn.Layer):
    def __init__(self, in_timestep, in_planes, mid_dim=4096, embed_dim=300):
        super(Embedding, self).__init__()
        self.in_timestep = in_timestep
        self.in_planes = in_planes
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.eEmbed = nn.Linear(
            in_timestep * in_planes,
            self.embed_dim)  # Embed encoder output to a word-embedding like

    def forward(self, x):
        x = paddle.reshape(x, [paddle.shape(x)[0], -1])
        x = self.eEmbed(x)
        return x


class AttentionRecognitionHead(nn.Layer):
    """
  input: [b x 16 x 64 x in_planes]
  output: probability sequence: [b x T x num_classes]
  """

    def __init__(self, in_channels, out_channels, sDim, attDim, max_len_labels):
        super(AttentionRecognitionHead, self).__init__()
        self.num_classes = out_channels  # this is the output classes. So it includes the <EOS>.
        self.in_planes = in_channels
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels

        self.decoder = DecoderUnit(
            sDim=sDim, xDim=in_channels, yDim=self.num_classes, attDim=attDim)

    def forward(self, x, embed):
        x, targets, lengths = x
        batch_size = paddle.shape(x)[0]
        # Decoder
        state = self.decoder.get_initial_state(embed)
        outputs = []

        for i in range(max(lengths)):
            if i == 0:
                y_prev = paddle.full(
                    shape=[batch_size], fill_value=self.num_classes)
            else:
                y_prev = targets[:, i - 1]

            output, state = self.decoder(x, state, y_prev)
            outputs.append(output)
        outputs = paddle.concat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

    # inference stage.
    def sample(self, x):
        x, _, _ = x
        batch_size = x.size(0)
        # Decoder
        state = paddle.zeros([1, batch_size, self.sDim])

        predicted_ids, predicted_scores = [], []
        for i in range(self.max_len_labels):
            if i == 0:
                y_prev = paddle.full(
                    shape=[batch_size], fill_value=self.num_classes)
            else:
                y_prev = predicted

            output, state = self.decoder(x, state, y_prev)
            output = F.softmax(output, axis=1)
            score, predicted = output.max(1)
            predicted_ids.append(predicted.unsqueeze(1))
            predicted_scores.append(score.unsqueeze(1))
        predicted_ids = paddle.concat([predicted_ids, 1])
        predicted_scores = paddle.concat([predicted_scores, 1])
        # return predicted_ids.squeeze(), predicted_scores.squeeze()
        return predicted_ids, predicted_scores


class AttentionUnit(nn.Layer):
    def __init__(self, sDim, xDim, attDim):
        super(AttentionUnit, self).__init__()

        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim

        self.sEmbed = nn.Linear(
            sDim,
            attDim,
            weight_attr=paddle.nn.initializer.Normal(std=0.01),
            bias_attr=paddle.nn.initializer.Constant(0.0))
        self.xEmbed = nn.Linear(
            xDim,
            attDim,
            weight_attr=paddle.nn.initializer.Normal(std=0.01),
            bias_attr=paddle.nn.initializer.Constant(0.0))
        self.wEmbed = nn.Linear(
            attDim,
            1,
            weight_attr=paddle.nn.initializer.Normal(std=0.01),
            bias_attr=paddle.nn.initializer.Constant(0.0))

    def forward(self, x, sPrev):
        batch_size, T, _ = x.shape  # [b x T x xDim]
        x = paddle.reshape(x, [-1, self.xDim])  # [(b x T) x xDim]
        xProj = self.xEmbed(x)  # [(b x T) x attDim]
        xProj = paddle.reshape(xProj, [batch_size, T, -1])  # [b x T x attDim]

        sPrev = sPrev.squeeze(0)
        sProj = self.sEmbed(sPrev)  # [b x attDim]
        sProj = paddle.unsqueeze(sProj, 1)  # [b x 1 x attDim]
        sProj = paddle.expand(sProj,
                              [batch_size, T, self.attDim])  # [b x T x attDim]

        sumTanh = paddle.tanh(sProj + xProj)
        sumTanh = paddle.reshape(sumTanh, [-1, self.attDim])

        vProj = self.wEmbed(sumTanh)  # [(b x T) x 1]
        vProj = paddle.reshape(vProj, [batch_size, T])

        alpha = F.softmax(
            vProj, axis=1)  # attention weights for each sample in the minibatch

        return alpha


class DecoderUnit(nn.Layer):
    def __init__(self, sDim, xDim, yDim, attDim):
        super(DecoderUnit, self).__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.yDim = yDim
        self.attDim = attDim
        self.emdDim = attDim

        self.attention_unit = AttentionUnit(sDim, xDim, attDim)
        self.tgt_embedding = nn.Embedding(
            yDim + 1, self.emdDim, weight_attr=nn.initializer.Normal(
                std=0.01))  # the last is used for <BOS>
        self.gru = nn.GRUCell(input_size=xDim + self.emdDim, hidden_size=sDim)
        self.fc = nn.Linear(
            sDim,
            yDim,
            weight_attr=nn.initializer.Normal(std=0.01),
            bias_attr=nn.initializer.Constant(value=0))
        self.embed_fc = nn.Linear(300, self.sDim)

    def get_initial_state(self, embed, tile_times=1):
        assert embed.shape[1] == 300
        state = self.embed_fc(embed)  # N * sDim
        if tile_times != 1:
            state = state.unsqueeze(1)
            trans_state = paddle.transpose(state, perm=[1, 0, 2])
            state = paddle.tile(trans_state, repeat_times=[tile_times, 1, 1])
            trans_state = paddle.transpose(state, perm=[1, 0, 2])
            state = paddle.reshape(trans_state, shape=[-1, self.sDim])
        state = state.unsqueeze(0)  # 1 * N * sDim
        return state

    def forward(self, x, sPrev, yPrev):
        # x: feature sequence from the image decoder.
        batch_size, T, _ = x.shape
        alpha = self.attention_unit(x, sPrev)
        context = paddle.squeeze(paddle.matmul(alpha.unsqueeze(1), x), axis=1)
        yPrev = paddle.cast(yPrev, dtype="int64")
        yProj = self.tgt_embedding(yPrev)

        concat_context = paddle.concat([yProj, context], 1)
        concat_context = paddle.squeeze(concat_context, 1)
        sPrev = paddle.squeeze(sPrev, 0)
        output, state = self.gru(concat_context, sPrev)
        output = paddle.squeeze(output, axis=1)
        output = self.fc(output)
        return output, state


if __name__ == "__main__":
    model = AttentionRecognitionHead(
        num_classes=20,
        in_channels=30,
        sDim=512,
        attDim=512,
        max_len_labels=25,
        out_channels=38)

    data = paddle.ones([16, 64, 3])
    targets = paddle.ones([16, 25])
    length = paddle.to_tensor(20)
    x = [data, targets, length]
    output = model(x)
    print(output.shape)
