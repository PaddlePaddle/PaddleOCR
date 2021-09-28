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
        self.eos = self.num_classes - 1

    def forward(self, x, targets=None, embed=None):
        return_dict = {}
        embedding_vectors = self.embeder(x)

        if self.training:
            rec_targets, rec_lengths, _ = targets
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

    def beam_search(self, x, beam_width, eos, embed):
        def _inflate(tensor, times, dim):
            repeat_dims = [1] * tensor.dim()
            repeat_dims[dim] = times
            output = paddle.tile(tensor, repeat_dims)
            return output

        # https://github.com/IBM/pytorch-seq2seq/blob/fede87655ddce6c94b38886089e05321dc9802af/seq2seq/models/TopKDecoder.py
        batch_size, l, d = x.shape
        x = paddle.tile(
            paddle.transpose(
                x.unsqueeze(1), perm=[1, 0, 2, 3]), [beam_width, 1, 1, 1])
        inflated_encoder_feats = paddle.reshape(
            paddle.transpose(
                x, perm=[1, 0, 2, 3]), [-1, l, d])

        # Initialize the decoder
        state = self.decoder.get_initial_state(embed, tile_times=beam_width)

        pos_index = paddle.reshape(
            paddle.arange(batch_size) * beam_width, shape=[-1, 1])

        # Initialize the scores
        sequence_scores = paddle.full(
            shape=[batch_size * beam_width, 1], fill_value=-float('Inf'))
        index = [i * beam_width for i in range(0, batch_size)]
        sequence_scores[index] = 0.0

        # Initialize the input vector
        y_prev = paddle.full(
            shape=[batch_size * beam_width], fill_value=self.num_classes)

        # Store decisions for backtracking
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()

        for i in range(self.max_len_labels):
            output, state = self.decoder(inflated_encoder_feats, state, y_prev)
            state = paddle.unsqueeze(state, axis=0)
            log_softmax_output = paddle.nn.functional.log_softmax(
                output, axis=1)

            sequence_scores = _inflate(sequence_scores, self.num_classes, 1)
            sequence_scores += log_softmax_output
            scores, candidates = paddle.topk(
                paddle.reshape(sequence_scores, [batch_size, -1]),
                beam_width,
                axis=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            y_prev = paddle.reshape(
                candidates % self.num_classes, shape=[batch_size * beam_width])
            sequence_scores = paddle.reshape(
                scores, shape=[batch_size * beam_width, 1])

            # Update fields for next timestep
            pos_index = paddle.expand_as(pos_index, candidates)
            predecessors = paddle.cast(
                candidates / self.num_classes + pos_index, dtype='int64')
            predecessors = paddle.reshape(
                predecessors, shape=[batch_size * beam_width, 1])
            state = paddle.index_select(
                state, index=predecessors.squeeze(), axis=1)

            # Update sequence socres and erase scores for <eos> symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            y_prev = paddle.reshape(y_prev, shape=[-1, 1])
            eos_prev = paddle.full_like(y_prev, fill_value=eos)
            mask = eos_prev == y_prev
            mask = paddle.nonzero(mask)
            if mask.dim() > 0:
                sequence_scores = sequence_scores.numpy()
                mask = mask.numpy()
                sequence_scores[mask] = -float('inf')
                sequence_scores = paddle.to_tensor(sequence_scores)

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            y_prev = paddle.squeeze(y_prev)
            stored_emitted_symbols.append(y_prev)

        # Do backtracking to return the optimal values
        #====== backtrak ======#
        # Initialize return variables given different types
        p = list()
        l = [[self.max_len_labels] * beam_width for _ in range(batch_size)
             ]  # Placeholder for lengths of top-k sequences

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = paddle.topk(
            paddle.reshape(
                stored_scores[-1], shape=[batch_size, beam_width]),
            beam_width)

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * batch_size  # the number of EOS found
        # in the backward loop below for each batch
        t = self.max_len_labels - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = paddle.reshape(
            sorted_idx + pos_index.expand_as(sorted_idx),
            shape=[batch_size * beam_width])
        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = paddle.index_select(
                stored_emitted_symbols[t], index=t_predecessors, axis=0)
            t_predecessors = paddle.index_select(
                stored_predecessors[t].squeeze(), index=t_predecessors, axis=0)
            eos_indices = stored_emitted_symbols[t] == eos
            eos_indices = paddle.nonzero(eos_indices)

            if eos_indices.dim() > 0:
                for i in range(eos_indices.shape[0] - 1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / beam_width)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_width - (batch_eos_found[b_idx] %
                                              beam_width) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_width + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
                    current_symbol[res_idx] = stored_emitted_symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = stored_scores[t][idx[0], 0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)
            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(beam_width)
        for b_idx in range(batch_size):
            l[b_idx] = [
                l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx, :]
            ]

        re_sorted_idx = paddle.reshape(
            re_sorted_idx + pos_index.expand_as(re_sorted_idx),
            [batch_size * beam_width])

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        p = [
            paddle.reshape(
                paddle.index_select(step, re_sorted_idx, 0),
                shape=[batch_size, beam_width, -1]) for step in reversed(p)
        ]
        p = paddle.concat(p, -1)[:, 0, :]
        return p, paddle.ones_like(p)


class AttentionUnit(nn.Layer):
    def __init__(self, sDim, xDim, attDim):
        super(AttentionUnit, self).__init__()

        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim

        self.sEmbed = nn.Linear(sDim, attDim)
        self.xEmbed = nn.Linear(xDim, attDim)
        self.wEmbed = nn.Linear(attDim, 1)

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