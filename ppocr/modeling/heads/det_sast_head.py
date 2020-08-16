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

import paddle.fluid as fluid
from ..common_functions import conv_bn_layer, deconv_bn_layer
from collections import OrderedDict


class SASTHead(object):
    """
    SAST: 
        see arxiv: https://arxiv.org/abs/1908.05498
    args:
        params(dict): the super parameters for network build
    """

    def __init__(self, params):
        self.model_name = params['model_name']
        self.with_cab = params['with_cab']

    def FPN_Up_Fusion(self, blocks):
        """
        blocks{}: contain block_2, block_3, block_4, block_5, block_6, block_7 with
                1/4, 1/8, 1/16, 1/32, 1/64, 1/128 resolution.
        """
        f = [blocks['block_6'], blocks['block_5'], blocks['block_4'], blocks['block_3'], blocks['block_2']]
        num_outputs = [256, 256, 192, 192, 128]
        g = [None, None, None, None, None]
        h = [None, None, None, None, None] 
        for i in range(5):
            h[i] = conv_bn_layer(input=f[i], num_filters=num_outputs[i],
                                filter_size=1, stride=1, act=None, name='fpn_up_h'+str(i))

        for i in range(4):
            if i == 0:
                g[i] = deconv_bn_layer(input=h[i], num_filters=num_outputs[i + 1], act=None, name='fpn_up_g0')
                print("g[{}] shape: {}".format(i, g[i].shape))
            else:
                g[i] = fluid.layers.elementwise_add(x=g[i - 1], y=h[i])
                g[i] = fluid.layers.relu(g[i])
                #g[i] = conv_bn_layer(input=g[i], num_filters=num_outputs[i],
                #                    filter_size=1, stride=1, act='relu')
                g[i] = conv_bn_layer(input=g[i], num_filters=num_outputs[i],
                                    filter_size=3, stride=1, act='relu', name='fpn_up_g%d_1'%i)
                g[i] = deconv_bn_layer(input=g[i], num_filters=num_outputs[i + 1], act=None, name='fpn_up_g%d_2'%i)
                print("g[{}] shape: {}".format(i, g[i].shape))

        g[4] = fluid.layers.elementwise_add(x=g[3], y=h[4])
        g[4] = fluid.layers.relu(g[4])
        g[4] = conv_bn_layer(input=g[4], num_filters=num_outputs[4],
                            filter_size=3, stride=1, act='relu', name='fpn_up_fusion_1')
        g[4] = conv_bn_layer(input=g[4], num_filters=num_outputs[4],
                            filter_size=1, stride=1, act=None, name='fpn_up_fusion_2')
        
        return g[4]

    def FPN_Down_Fusion(self, blocks):
        """
        blocks{}: contain block_2, block_3, block_4, block_5, block_6, block_7 with
                1/4, 1/8, 1/16, 1/32, 1/64, 1/128 resolution.
        """
        f = [blocks['block_0'], blocks['block_1'], blocks['block_2']]
        num_outputs = [32, 64, 128]
        g = [None, None, None]
        h = [None, None, None] 
        for i in range(3):
            h[i] = conv_bn_layer(input=f[i], num_filters=num_outputs[i],
                                filter_size=3, stride=1, act=None, name='fpn_down_h'+str(i))
        for i in range(2):
            if i == 0:
                g[i] = conv_bn_layer(input=h[i], num_filters=num_outputs[i+1], filter_size=3, stride=2, act=None, name='fpn_down_g0')
            else:
                g[i] = fluid.layers.elementwise_add(x=g[i - 1], y=h[i])
                g[i] = fluid.layers.relu(g[i])
                g[i] = conv_bn_layer(input=g[i], num_filters=num_outputs[i], filter_size=3, stride=1, act='relu', name='fpn_down_g%d_1'%i)
                g[i] = conv_bn_layer(input=g[i], num_filters=num_outputs[i+1], filter_size=3, stride=2, act=None, name='fpn_down_g%d_2'%i)
            # print("g[{}] shape: {}".format(i, g[i].shape)) 
        g[2] = fluid.layers.elementwise_add(x=g[1], y=h[2])
        g[2] = fluid.layers.relu(g[2])
        g[2] = conv_bn_layer(input=g[2], num_filters=num_outputs[2],
                            filter_size=3, stride=1, act='relu', name='fpn_down_fusion_1')
        g[2] = conv_bn_layer(input=g[2], num_filters=num_outputs[2],
                            filter_size=1, stride=1, act=None, name='fpn_down_fusion_2')
        return g[2]

    def SAST_Header1(self, f_common):
        """Detector header."""
        #f_score
        f_score = conv_bn_layer(input=f_common, num_filters=64, filter_size=1, stride=1, act='relu', name='f_score1')
        f_score = conv_bn_layer(input=f_score, num_filters=64, filter_size=3, stride=1, act='relu', name='f_score2')
        f_score = conv_bn_layer(input=f_score, num_filters=128, filter_size=1, stride=1, act='relu', name='f_score3')
        f_score = conv_bn_layer(input=f_score, num_filters=1, filter_size=3, stride=1, name='f_score4')
        f_score = fluid.layers.sigmoid(f_score)
        # print("f_score shape: {}".format(f_score.shape))

        #f_boder
        f_border = conv_bn_layer(input=f_common, num_filters=64, filter_size=1, stride=1, act='relu', name='f_border1')
        f_border = conv_bn_layer(input=f_border, num_filters=64, filter_size=3, stride=1, act='relu', name='f_border2')
        f_border = conv_bn_layer(input=f_border, num_filters=128, filter_size=1, stride=1, act='relu', name='f_border3')
        f_border = conv_bn_layer(input=f_border, num_filters=4, filter_size=3, stride=1, name='f_border4')
        # print("f_border shape: {}".format(f_border.shape))
        
        return f_score, f_border

    def SAST_Header2(self, f_common):
        """Detector header.""" 
        #f_tvo
        f_tvo = conv_bn_layer(input=f_common, num_filters=64, filter_size=1, stride=1, act='relu', name='f_tvo1')
        f_tvo = conv_bn_layer(input=f_tvo, num_filters=64, filter_size=3, stride=1, act='relu', name='f_tvo2')
        f_tvo = conv_bn_layer(input=f_tvo, num_filters=128, filter_size=1, stride=1, act='relu', name='f_tvo3')
        f_tvo = conv_bn_layer(input=f_tvo, num_filters=8, filter_size=3, stride=1, name='f_tvo4')
        # print("f_tvo shape: {}".format(f_tvo.shape))

        #f_tco
        f_tco = conv_bn_layer(input=f_common, num_filters=64, filter_size=1, stride=1, act='relu', name='f_tco1')
        f_tco = conv_bn_layer(input=f_tco, num_filters=64, filter_size=3, stride=1, act='relu', name='f_tco2')
        f_tco = conv_bn_layer(input=f_tco, num_filters=128, filter_size=1, stride=1, act='relu', name='f_tco3')
        f_tco = conv_bn_layer(input=f_tco, num_filters=2, filter_size=3, stride=1, name='f_tco4')
        # print("f_tco shape: {}".format(f_tco.shape))
        
        return f_tvo, f_tco

    def cross_attention(self, f_common):
        """
        """
        f_shape = fluid.layers.shape(f_common)
        f_theta = conv_bn_layer(input=f_common, num_filters=128, filter_size=1, stride=1, act='relu', name='f_theta')
        f_phi = conv_bn_layer(input=f_common, num_filters=128, filter_size=1, stride=1, act='relu', name='f_phi')
        f_g = conv_bn_layer(input=f_common, num_filters=128, filter_size=1, stride=1, act='relu', name='f_g')
        ### horizon
        fh_theta = f_theta
        fh_phi = f_phi
        fh_g = f_g
        #flatten
        fh_theta = fluid.layers.transpose(fh_theta, [0, 2, 3, 1])
        fh_theta = fluid.layers.reshape(fh_theta, [f_shape[0] * f_shape[2], f_shape[3], 128])
        fh_phi = fluid.layers.transpose(fh_phi, [0, 2, 3, 1])
        fh_phi = fluid.layers.reshape(fh_phi, [f_shape[0] * f_shape[2], f_shape[3], 128])
        fh_g = fluid.layers.transpose(fh_g, [0, 2, 3, 1])
        fh_g = fluid.layers.reshape(fh_g, [f_shape[0] * f_shape[2], f_shape[3], 128])
        #correlation
        fh_attn = fluid.layers.matmul(fh_theta, fluid.layers.transpose(fh_phi, [0, 2, 1]))
        #scale
        fh_attn = fh_attn / (128 ** 0.5)
        fh_attn = fluid.layers.softmax(fh_attn)
        #weighted sum
        fh_weight = fluid.layers.matmul(fh_attn, fh_g)
        fh_weight = fluid.layers.reshape(fh_weight, [f_shape[0], f_shape[2], f_shape[3], 128])
        # print("fh_weight: {}".format(fh_weight.shape))
        fh_weight = fluid.layers.transpose(fh_weight, [0, 3, 1, 2])
        fh_weight = conv_bn_layer(input=fh_weight, num_filters=128, filter_size=1, stride=1, name='fh_weight')
        #short cut
        fh_sc = conv_bn_layer(input=f_common, num_filters=128, filter_size=1, stride=1, name='fh_sc')
        f_h = fluid.layers.relu(fh_weight + fh_sc)
        ######
        #vertical
        fv_theta = fluid.layers.transpose(f_theta, [0, 1, 3, 2])
        fv_phi = fluid.layers.transpose(f_phi, [0, 1, 3, 2])
        fv_g = fluid.layers.transpose(f_g, [0, 1, 3, 2])
        #flatten
        fv_theta = fluid.layers.transpose(fv_theta, [0, 2, 3, 1])
        fv_theta = fluid.layers.reshape(fv_theta, [f_shape[0] * f_shape[3], f_shape[2], 128])
        fv_phi = fluid.layers.transpose(fv_phi, [0, 2, 3, 1])
        fv_phi = fluid.layers.reshape(fv_phi, [f_shape[0] * f_shape[3], f_shape[2], 128])
        fv_g = fluid.layers.transpose(fv_g, [0, 2, 3, 1])
        fv_g = fluid.layers.reshape(fv_g, [f_shape[0] * f_shape[3], f_shape[2], 128])
        #correlation
        fv_attn = fluid.layers.matmul(fv_theta, fluid.layers.transpose(fv_phi, [0, 2, 1]))
        #scale
        fv_attn = fv_attn / (128 ** 0.5)
        fv_attn = fluid.layers.softmax(fv_attn)
        #weighted sum
        fv_weight = fluid.layers.matmul(fv_attn, fv_g)
        fv_weight = fluid.layers.reshape(fv_weight, [f_shape[0], f_shape[3], f_shape[2], 128])
        # print("fv_weight: {}".format(fv_weight.shape))
        fv_weight = fluid.layers.transpose(fv_weight, [0, 3, 2, 1])
        fv_weight = conv_bn_layer(input=fv_weight, num_filters=128, filter_size=1, stride=1, name='fv_weight')
        #short cut
        fv_sc = conv_bn_layer(input=f_common, num_filters=128, filter_size=1, stride=1, name='fv_sc')
        f_v = fluid.layers.relu(fv_weight + fv_sc)
        ######
        f_attn = fluid.layers.concat([f_h, f_v], axis=1)
        f_attn = conv_bn_layer(input=f_attn, num_filters=128, filter_size=1, stride=1, act='relu', name='f_attn')  
        return f_attn
        
    def __call__(self, blocks, with_cab=False):
        # for k, v in blocks.items():
        #     print(k, v.shape)

        #down fpn
        f_down = self.FPN_Down_Fusion(blocks)
        # print("f_down shape: {}".format(f_down.shape))
        #up fpn
        f_up = self.FPN_Up_Fusion(blocks)
        # print("f_up shape: {}".format(f_up.shape))
        #fusion
        f_common = fluid.layers.elementwise_add(x=f_down, y=f_up)
        f_common = fluid.layers.relu(f_common)
        # print("f_common: {}".format(f_common.shape))
        
        if self.with_cab:
            # print('enhence f_common with CAB.')
            f_common = self.cross_attention(f_common)
            
        f_score, f_border= self.SAST_Header1(f_common)
        f_tvo, f_tco = self.SAST_Header2(f_common)

        predicts = OrderedDict()
        predicts['f_score'] = f_score
        predicts['f_border'] = f_border
        predicts['f_tvo'] = f_tvo
        predicts['f_tco'] = f_tco
        return predicts