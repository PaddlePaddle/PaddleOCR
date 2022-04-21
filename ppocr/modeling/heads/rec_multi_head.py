from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

from ppocr.modeling.necks.rnn import Im2Seq, EncoderWithRNN, EncoderWithFC, SequenceEncoder, EncoderWithTransformer
from .rec_ctc_head import CTCHead
from .rec_sar_head import SARHead


class MultiHead(nn.Layer):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop('head_list')
        self.gtc_head = 'sar'
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == 'SARHead':
                # sar head
                enc_dim = self.head_list[idx][name]['enc_dim']
                max_text_length = self.head_list[idx][name]['max_text_length']
                self.sar_head = eval(name)(in_channels=in_channels, \
                    out_channels=out_channels_list['SARLabelDecode'],
                    enc_dim=enc_dim,
                    max_text_length=max_text_length)
            elif name == 'CTCHead':
                # ctc encoder
                self.encoder_reshape = Im2Seq(in_channels)
                encoder_type = self.head_list[idx][name].get('encoder_type',
                                                             'rnn')
                self.encoder = encoder_type
                if encoder_type == 'mobilevit':
                    print('using mobilevit encoder')
                    self.ctc_encoder = EncoderWithTransformer(
                        in_channels=in_channels)
                else:
                    print('using LSTM encoder')
                    hidden_size = self.head_list[idx][name].get('hidden_size',
                                                                48)
                    self.ctc_encoder = SequenceEncoder(in_channels=in_channels, \
                        encoder_type=encoder_type, hidden_size=hidden_size)
                # ctc head
                fc_decay = self.head_list[idx][name].get('fc_decay', 0.0004)
                mid_channels = self.head_list[idx][name].get('mid_channels',
                                                             None)

                self.ctc_head = eval(name)(in_channels=self.ctc_encoder.out_channels, \
                    out_channels=out_channels_list['CTCLabelDecode'], fc_decay=fc_decay, \
                    mid_channels=mid_channels)
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiHead yet'.format(name))

    def forward(self, x, targets=None):
        ctc_encoder = self.ctc_encoder(x)
        if self.encoder in ['mobilevit']:
            ctc_encoder = self.encoder_reshape(ctc_encoder)

        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = dict()
        head_out['ctc'] = ctc_out
        head_out['ctc_neck'] = ctc_encoder
        # for test
        if not self.training:
            return ctc_out

        x.stop_gradient = False
        if self.gtc_head == 'sar':
            sar_out = self.sar_head(x, targets[1:])
            head_out['sar'] = sar_out
            return head_out

        else:
            return head_out
