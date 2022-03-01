# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import os
from paddle import nn

from paddlenlp.transformers import LayoutXLMModel, LayoutXLMForTokenClassification, LayoutXLMForRelationExtraction
from paddlenlp.transformers import LayoutLMModel, LayoutLMForTokenClassification
from paddlenlp.transformers import LayoutLMv2Model, LayoutLMv2ForTokenClassification, LayoutLMv2ForRelationExtraction

__all__ = ["LayoutXLMForSer", 'LayoutLMForSer']

pretrained_model_dict = {
    LayoutXLMModel: 'layoutxlm-base-uncased',
    LayoutLMModel: 'layoutlm-base-uncased',
    LayoutLMv2Model: 'layoutlmv2-base-uncased'
}


class NLPBaseModel(nn.Layer):
    def __init__(self,
                 base_model_class,
                 model_class,
                 type='ser',
                 pretrained=True,
                 checkpoints=None,
                 **kwargs):
        super(NLPBaseModel, self).__init__()
        if checkpoints is not None:
            self.model = model_class.from_pretrained(checkpoints)
        else:
            pretrained_model_name = pretrained_model_dict[base_model_class]
            if pretrained:
                base_model = base_model_class.from_pretrained(
                    pretrained_model_name)
            else:
                base_model = base_model_class(
                    **base_model_class.pretrained_init_configuration[
                        pretrained_model_name])
            if type == 'ser':
                self.model = model_class(
                    base_model, num_classes=kwargs['num_classes'], dropout=None)
            else:
                self.model = model_class(base_model, dropout=None)
        self.out_channels = 1


class LayoutLMForSer(NLPBaseModel):
    def __init__(self, num_classes, pretrained=True, checkpoints=None,
                 **kwargs):
        super(LayoutLMForSer, self).__init__(
            LayoutLMModel,
            LayoutLMForTokenClassification,
            'ser',
            pretrained,
            checkpoints,
            num_classes=num_classes)

    def forward(self, x):
        x = self.model(
            input_ids=x[0],
            bbox=x[2],
            attention_mask=x[4],
            token_type_ids=x[5],
            position_ids=None,
            output_hidden_states=False)
        return x


class LayoutLMv2ForSer(NLPBaseModel):
    def __init__(self, num_classes, pretrained=True, checkpoints=None,
                 **kwargs):
        super(LayoutLMv2ForSer, self).__init__(
            LayoutLMv2Model,
            LayoutLMv2ForTokenClassification,
            'ser',
            pretrained,
            checkpoints,
            num_classes=num_classes)

    def forward(self, x):
        x = self.model(
            input_ids=x[0],
            bbox=x[2],
            image=x[3],
            attention_mask=x[4],
            token_type_ids=x[5],
            position_ids=None,
            head_mask=None,
            labels=None)
        return x[0]


class LayoutXLMForSer(NLPBaseModel):
    def __init__(self, num_classes, pretrained=True, checkpoints=None,
                 **kwargs):
        super(LayoutXLMForSer, self).__init__(
            LayoutXLMModel,
            LayoutXLMForTokenClassification,
            'ser',
            pretrained,
            checkpoints,
            num_classes=num_classes)

    def forward(self, x):
        x = self.model(
            input_ids=x[0],
            bbox=x[2],
            image=x[3],
            attention_mask=x[4],
            token_type_ids=x[5],
            position_ids=None,
            head_mask=None,
            labels=None)
        return x[0]


class LayoutLMv2ForRe(NLPBaseModel):
    def __init__(self, pretrained=True, checkpoints=None, **kwargs):
        super(LayoutLMv2ForRe, self).__init__(LayoutLMv2Model,
                                              LayoutLMv2ForRelationExtraction,
                                              're', pretrained, checkpoints)

    def forward(self, x):
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            labels=None,
            image=x[2],
            attention_mask=x[3],
            token_type_ids=x[4],
            position_ids=None,
            head_mask=None,
            entities=x[5],
            relations=x[6])
        return x


class LayoutXLMForRe(NLPBaseModel):
    def __init__(self, pretrained=True, checkpoints=None, **kwargs):
        super(LayoutXLMForRe, self).__init__(LayoutXLMModel,
                                             LayoutXLMForRelationExtraction,
                                             're', pretrained, checkpoints)

    def forward(self, x):
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            labels=None,
            image=x[2],
            attention_mask=x[3],
            token_type_ids=x[4],
            position_ids=None,
            head_mask=None,
            entities=x[5],
            relations=x[6])
        return x
