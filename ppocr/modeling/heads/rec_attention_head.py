#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from .rec_seq_encoder import SequenceEncoder
import numpy as np


class AttentionPredict(object):
    def __init__(self, params):
        super(AttentionPredict, self).__init__()
        self.char_num = params['char_num']
        self.encoder = SequenceEncoder(params)
        self.decoder_size = params['Attention']['decoder_size']
        self.word_vector_dim = params['Attention']['word_vector_dim']
        self.encoder_type = params['encoder_type']
        self.max_length = params['max_text_length']

    def simple_attention(self, encoder_vec, encoder_proj, decoder_state,
                         decoder_size):
        decoder_state_proj = layers.fc(input=decoder_state,
                                       size=decoder_size,
                                       bias_attr=False,
                                       name="decoder_state_proj_fc")
        decoder_state_expand = layers.sequence_expand(
            x=decoder_state_proj, y=encoder_proj)
        concated = layers.elementwise_add(encoder_proj, decoder_state_expand)
        concated = layers.tanh(x=concated)
        attention_weights = layers.fc(input=concated,
                                      size=1,
                                      act=None,
                                      bias_attr=False,
                                      name="attention_weights_fc")
        attention_weights = layers.sequence_softmax(input=attention_weights)
        weigths_reshape = layers.reshape(x=attention_weights, shape=[-1])
        scaled = layers.elementwise_mul(
            x=encoder_vec, y=weigths_reshape, axis=0)
        context = layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    def gru_decoder_with_attention(self, target_embedding, encoder_vec,
                                   encoder_proj, decoder_boot, decoder_size,
                                   char_num):
        rnn = layers.DynamicRNN()
        with rnn.block():
            current_word = rnn.step_input(target_embedding)
            encoder_vec = rnn.static_input(encoder_vec)
            encoder_proj = rnn.static_input(encoder_proj)
            hidden_mem = rnn.memory(init=decoder_boot, need_reorder=True)
            context = self.simple_attention(encoder_vec, encoder_proj,
                                            hidden_mem, decoder_size)
            fc_1 = layers.fc(input=context,
                             size=decoder_size * 3,
                             bias_attr=False,
                             name="rnn_fc1")
            fc_2 = layers.fc(input=current_word,
                             size=decoder_size * 3,
                             bias_attr=False,
                             name="rnn_fc2")
            decoder_inputs = fc_1 + fc_2
            h, _, _ = layers.gru_unit(
                input=decoder_inputs, hidden=hidden_mem, size=decoder_size * 3)
            rnn.update_memory(hidden_mem, h)
            out = layers.fc(input=h,
                            size=char_num,
                            bias_attr=True,
                            act='softmax',
                            name="rnn_out_fc")
            rnn.output(out)
        return rnn()

    def gru_attention_infer(self, decoder_boot, max_length, char_num,
                            word_vector_dim, encoded_vector, encoded_proj,
                            decoder_size):
        init_state = decoder_boot
        beam_size = 1
        array_len = layers.fill_constant(
            shape=[1], dtype='int64', value=max_length)
        counter = layers.zeros(shape=[1], dtype='int64', force_cpu=True)

        # fill the first element with init_state
        state_array = layers.create_array('float32')
        layers.array_write(init_state, array=state_array, i=counter)

        # ids, scores as memory
        ids_array = layers.create_array('int64')
        scores_array = layers.create_array('float32')
        rois_shape = layers.shape(init_state)
        batch_size = layers.slice(
            rois_shape, axes=[0], starts=[0], ends=[1]) + 1
        lod_level = layers.range(
            start=0, end=batch_size, step=1, dtype=batch_size.dtype)

        init_ids = layers.fill_constant_batch_size_like(
            input=init_state, shape=[-1, 1], value=0, dtype='int64')
        init_ids = layers.lod_reset(init_ids, lod_level)
        init_ids = layers.lod_append(init_ids, lod_level)

        init_scores = layers.fill_constant_batch_size_like(
            input=init_state, shape=[-1, 1], value=1, dtype='float32')
        init_scores = layers.lod_reset(init_scores, init_ids)
        layers.array_write(init_ids, array=ids_array, i=counter)
        layers.array_write(init_scores, array=scores_array, i=counter)

        full_ids = fluid.layers.fill_constant_batch_size_like(
            input=init_state, shape=[-1, 1], dtype='int64', value=1)
        full_scores = fluid.layers.fill_constant_batch_size_like(
            input=init_state, shape=[-1, 1], dtype='float32', value=1)

        cond = layers.less_than(x=counter, y=array_len)
        while_op = layers.While(cond=cond)
        with while_op.block():
            pre_ids = layers.array_read(array=ids_array, i=counter)
            pre_state = layers.array_read(array=state_array, i=counter)
            pre_score = layers.array_read(array=scores_array, i=counter)
            pre_ids_emb = layers.embedding(
                input=pre_ids,
                size=[char_num, word_vector_dim],
                dtype='float32')

            context = self.simple_attention(encoded_vector, encoded_proj,
                                            pre_state, decoder_size)

            # expand the recursive_sequence_lengths of pre_state 
            # to be the same with pre_score
            pre_state_expanded = layers.sequence_expand(pre_state, pre_score)
            context_expanded = layers.sequence_expand(context, pre_score)

            fc_1 = layers.fc(input=context_expanded,
                             size=decoder_size * 3,
                             bias_attr=False,
                             name="rnn_fc1")

            fc_2 = layers.fc(input=pre_ids_emb,
                             size=decoder_size * 3,
                             bias_attr=False,
                             name="rnn_fc2")

            decoder_inputs = fc_1 + fc_2
            current_state, _, _ = layers.gru_unit(
                input=decoder_inputs,
                hidden=pre_state_expanded,
                size=decoder_size * 3)
            current_state_with_lod = layers.lod_reset(
                x=current_state, y=pre_score)
            # use score to do beam search
            current_score = layers.fc(input=current_state_with_lod,
                                      size=char_num,
                                      bias_attr=True,
                                      act='softmax',
                                      name="rnn_out_fc")
            topk_scores, topk_indices = layers.topk(current_score, k=beam_size)

            new_ids = fluid.layers.concat([full_ids, topk_indices], axis=1)
            fluid.layers.assign(new_ids, full_ids)

            new_scores = fluid.layers.concat([full_scores, topk_scores], axis=1)
            fluid.layers.assign(new_scores, full_scores)
            
            layers.increment(x=counter, value=1, in_place=True)

            # update the memories
            layers.array_write(current_state, array=state_array, i=counter)
            layers.array_write(topk_indices, array=ids_array, i=counter)
            layers.array_write(topk_scores, array=scores_array, i=counter)

            # update the break condition: 
            # up to the max length or all candidates of
            # source sentences have ended.
            length_cond = layers.less_than(x=counter, y=array_len)
            finish_cond = layers.logical_not(layers.is_empty(x=topk_indices))
            layers.logical_and(x=length_cond, y=finish_cond, out=cond)
        return full_ids, full_scores

    def __call__(self, inputs, labels=None, mode=None):
        encoder_features = self.encoder(inputs)
        char_num = self.char_num
        word_vector_dim = self.word_vector_dim
        decoder_size = self.decoder_size

        if self.encoder_type == "reshape":
            encoder_input = encoder_features
            encoded_vector = encoder_features
        else:
            encoder_input = encoder_features[1]
            encoded_vector = layers.concat(encoder_features, axis=1)
        encoded_proj = layers.fc(input=encoded_vector,
                                 size=decoder_size,
                                 bias_attr=False,
                                 name="encoded_proj_fc")
        backward_first = layers.sequence_pool(
            input=encoder_input, pool_type='first')
        decoder_boot = layers.fc(input=backward_first,
                                 size=decoder_size,
                                 bias_attr=False,
                                 act="relu",
                                 name='decoder_boot')

        if mode == "train":
            label_in = labels['label_in']
            label_out = labels['label_out']
            label_in = layers.cast(x=label_in, dtype='int64')
            trg_embedding = layers.embedding(
                input=label_in,
                size=[char_num, word_vector_dim],
                dtype='float32')
            predict = self.gru_decoder_with_attention(
                trg_embedding, encoded_vector, encoded_proj, decoder_boot,
                decoder_size, char_num)
            _, decoded_out = layers.topk(input=predict, k=1)
            decoded_out = layers.lod_reset(decoded_out, y=label_out)
            predicts = {'predict':predict, 'decoded_out':decoded_out}
        else:
            ids, predict = self.gru_attention_infer(
                decoder_boot, self.max_length, char_num, word_vector_dim,
                encoded_vector, encoded_proj, decoder_size)
            predicts = {'predict':predict, 'decoded_out':ids}
        return predicts
