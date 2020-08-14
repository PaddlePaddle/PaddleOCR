#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from paddle.fluid.param_attr import ParamAttr
#from .rec_seq_encoder import SequenceEncoder
#from ..common_functions import get_para_bias_attr
import numpy as np
from .self_attention.model import wrap_encoder
from .self_attention.model import wrap_encoder_forFeature
gradient_clip = 10



class SRNPredict(object):
    def __init__(self, params):
        super(SRNPredict, self).__init__()
        self.char_num = params['char_num']
        self.max_length = params['max_text_length']

        self.num_heads = params['num_heads']
        self.num_encoder_TUs = params['num_encoder_TUs']
        self.num_decoder_TUs = params['num_decoder_TUs']
        self.hidden_dims = params['hidden_dims']


    def pvam(self, inputs, others):

        b, c, h, w = inputs.shape
        conv_features = fluid.layers.reshape(x=inputs, shape=[-1, c, h * w])
        conv_features = fluid.layers.transpose(x=conv_features, perm=[0, 2, 1])

        #===== Transformer encoder =====
        b, t, c = conv_features.shape
        encoder_word_pos = others["encoder_word_pos"]
        gsrm_word_pos = others["gsrm_word_pos"]


        enc_inputs = [conv_features, encoder_word_pos, None]
        word_features = wrap_encoder_forFeature(src_vocab_size=-1,
                 max_length=t,
                 n_layer=self.num_encoder_TUs,
                 n_head=self.num_heads,
                 d_key= int(self.hidden_dims / self.num_heads),
                 d_value= int(self.hidden_dims / self.num_heads),
                 d_model=self.hidden_dims,
                 d_inner_hid=self.hidden_dims,
                 prepostprocess_dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 weight_sharing=True,
                 enc_inputs=enc_inputs,
                )
        fluid.clip.set_gradient_clip(fluid.clip.GradientClipByValue(gradient_clip))

        #===== Parallel Visual Attention Module =====
        b, t, c = word_features.shape

        word_features = fluid.layers.fc(word_features, c, num_flatten_dims=2) 
        word_features_ = fluid.layers.reshape(word_features, [-1, 1, t, c])
        word_features_ = fluid.layers.expand(word_features_, [1, self.max_length, 1, 1])
        word_pos_feature = fluid.layers.embedding(gsrm_word_pos, [self.max_length, c]) 
        word_pos_ = fluid.layers.reshape(word_pos_feature, [-1, self.max_length, 1, c])
        word_pos_ = fluid.layers.expand(word_pos_, [1, 1, t, 1])
        temp = fluid.layers.elementwise_add(word_features_, word_pos_, act='tanh')

        attention_weight = fluid.layers.fc(input=temp, size=1, num_flatten_dims=3, bias_attr=False) 
        attention_weight = fluid.layers.reshape(x=attention_weight, shape=[-1, self.max_length, t])    
        attention_weight = fluid.layers.softmax(input=attention_weight, axis=-1) 

        pvam_features = fluid.layers.matmul(attention_weight, word_features)#[b, max_length, c]
        
        return pvam_features
        
    def gsrm(self, pvam_features, others):

        #===== GSRM Visual-to-semantic embedding block =====
        b, t, c = pvam_features.shape
        word_out = fluid.layers.fc(input=fluid.layers.reshape(pvam_features, [-1, c]),
                                  size=self.char_num,
                                  act="softmax")
        #word_out.stop_gradient = True
        word_ids = fluid.layers.argmax(word_out, axis=1)
        word_ids.stop_gradient = True
        word_ids = fluid.layers.reshape(x=word_ids, shape=[-1, t, 1])

        #===== GSRM Semantic reasoning block =====
        """
        This module is achieved through bi-transformers, 
        ngram_feature1 is the froward one, ngram_fetaure2 is the backward one
        """
        pad_idx = self.char_num
        gsrm_word_pos = others["gsrm_word_pos"]
        gsrm_slf_attn_bias1 = others["gsrm_slf_attn_bias1"]
        gsrm_slf_attn_bias2 = others["gsrm_slf_attn_bias2"]

        def prepare_bi(word_ids):
            """
            prepare bi for gsrm
            word1 for forward; word2 for backward
            """
            word1 = fluid.layers.cast(word_ids, "float32")
            word1 = fluid.layers.pad(word1, [0, 0, 1, 0, 0, 0], pad_value=1.0 * pad_idx)
            word1 = fluid.layers.cast(word1, "int64")
            word1 = word1[:, :-1, :]
            word2 = word_ids
            return word1, word2

        word1, word2 = prepare_bi(word_ids)
        word1.stop_gradient = True
        word2.stop_gradient = True
        enc_inputs_1 = [word1, gsrm_word_pos, gsrm_slf_attn_bias1]
        enc_inputs_2 = [word2, gsrm_word_pos, gsrm_slf_attn_bias2]

        gsrm_feature1 = wrap_encoder(src_vocab_size=self.char_num + 1,
                 max_length=self.max_length,
                 n_layer=self.num_decoder_TUs,
                 n_head=self.num_heads,
                 d_key=int(self.hidden_dims / self.num_heads),
                 d_value=int(self.hidden_dims / self.num_heads),
                 d_model=self.hidden_dims,
                 d_inner_hid=self.hidden_dims,
                 prepostprocess_dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 weight_sharing=True,
                 enc_inputs=enc_inputs_1,
                )
        gsrm_feature2 = wrap_encoder(src_vocab_size=self.char_num + 1,
                 max_length=self.max_length,
                 n_layer=self.num_decoder_TUs,
                 n_head=self.num_heads,
                 d_key=int(self.hidden_dims / self.num_heads),
                 d_value=int(self.hidden_dims / self.num_heads),
                 d_model=self.hidden_dims,
                 d_inner_hid=self.hidden_dims,
                 prepostprocess_dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 weight_sharing=True,
                 enc_inputs=enc_inputs_2,
                )
        gsrm_feature2 = fluid.layers.pad(gsrm_feature2, [0, 0, 0, 1, 0, 0], pad_value=0.)
        gsrm_feature2 = gsrm_feature2[:, 1:, ]
        gsrm_features = gsrm_feature1 + gsrm_feature2

        b, t, c = gsrm_features.shape

        gsrm_out = fluid.layers.matmul(
            x=gsrm_features,
            y=fluid.default_main_program().global_block().var("src_word_emb_table"),
            transpose_y=True)
        b,t,c = gsrm_out.shape
        gsrm_out = fluid.layers.softmax(input=fluid.layers.reshape(gsrm_out, [-1, c]))

        return gsrm_features, word_out, gsrm_out

    def vsfd(self, pvam_features, gsrm_features):

        #===== Visual-Semantic Fusion Decoder Module =====
        b, t, c1 = pvam_features.shape
        b, t, c2 = gsrm_features.shape
        combine_features_ = fluid.layers.concat([pvam_features, gsrm_features], axis=2)
        img_comb_features_ = fluid.layers.reshape(x=combine_features_, shape=[-1, c1 + c2])
        img_comb_features_map = fluid.layers.fc(input=img_comb_features_, size=c1, act="sigmoid")
        img_comb_features_map = fluid.layers.reshape(x=img_comb_features_map, shape=[-1, t, c1])    
        combine_features = img_comb_features_map * pvam_features + (1.0 - img_comb_features_map) * gsrm_features    
        img_comb_features = fluid.layers.reshape(x=combine_features, shape=[-1, c1])

        fc_out = fluid.layers.fc(input=img_comb_features,
                                 size=self.char_num,
                                 act="softmax")
        return fc_out


    def __call__(self, inputs, others, mode=None):

        pvam_features = self.pvam(inputs, others)
        gsrm_features, word_out, gsrm_out = self.gsrm(pvam_features, others)
        final_out = self.vsfd(pvam_features, gsrm_features)

        _, decoded_out = fluid.layers.topk(input=final_out, k=1)
        predicts = {'predict': final_out, 'decoded_out': decoded_out, 
                    'word_out': word_out, 'gsrm_out': gsrm_out}

        return predicts








