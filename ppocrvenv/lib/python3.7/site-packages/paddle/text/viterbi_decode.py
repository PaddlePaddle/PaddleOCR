#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ..nn import Layer
from ..fluid.framework import core, in_dygraph_mode
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type

__all__ = ['viterbi_decode', 'ViterbiDecoder']


def viterbi_decode(potentials,
                   transition_params,
                   lengths,
                   include_bos_eos_tag=True,
                   name=None):
    """
    Decode the highest scoring sequence of tags computed by transitions and potentials and get the viterbi path.
 
    Args:
        potentials (Tensor): The input tensor of unary emission. This is a 3-D
            tensor with shape of [batch_size, sequence_length, num_tags]. The data type is float32 or float64. 
        transition_params (Tensor): The input tensor of transition matrix. This is a 2-D
            tensor with shape of [num_tags, num_tags]. The data type is float32 or float64. 
        lengths (Tensor):  The input tensor of length of each sequence. This is a 1-D tensor with shape of [batch_size]. The data type is int64. 
        include_bos_eos_tag (`bool`, optional): If set to True, the last row and the last column of transitions will be considered
            as start tag, the second to last row and the second to last column of transitions will be considered as stop tag. Defaults to ``True``.
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        scores(Tensor): The output tensor containing the score for the Viterbi sequence. The shape is [batch_size]
            and the data type is float32 or float64.
        paths(Tensor): The output tensor containing the highest scoring tag indices.  The shape is [batch_size, sequence_length]
            and  the data type is int64.

    Example:
        .. code-block:: python

            import paddle
            paddle.seed(102)
            batch_size, seq_len, num_tags = 2, 4, 3
            emission = paddle.rand((batch_size, seq_len, num_tags), dtype='float32')
            length = paddle.randint(1, seq_len + 1, [batch_size])
            tags = paddle.randint(0, num_tags, [batch_size, seq_len])
            transition = paddle.rand((num_tags, num_tags), dtype='float32')
            scores, path = paddle.text.viterbi_decode(emission, transition, length, False) # scores: [3.37089300, 1.56825531], path: [[1, 0, 0], [1, 1, 0]]
    """
    if in_dygraph_mode():
        return core.ops.viterbi_decode(potentials, transition_params, lengths,
                                       'include_bos_eos_tag',
                                       include_bos_eos_tag)
    check_variable_and_dtype(potentials, 'input', ['float32', 'float64'],
                             'viterbi_decode')
    check_variable_and_dtype(transition_params, 'transitions',
                             ['float32', 'float64'], 'viterbi_decode')
    check_variable_and_dtype(lengths, 'length', 'int64', 'viterbi_decode')
    check_type(include_bos_eos_tag, 'include_tag', bool, 'viterbi_decode')
    helper = LayerHelper('viterbi_decode', **locals())
    attrs = {'include_bos_eos_tag': include_bos_eos_tag}
    scores = helper.create_variable_for_type_inference(potentials.dtype)
    path = helper.create_variable_for_type_inference('int64')
    helper.append_op(
        type='viterbi_decode',
        inputs={
            'Input': potentials,
            'Transition': transition_params,
            'Length': lengths
        },
        outputs={'Scores': scores,
                 'Path': path},
        attrs=attrs)
    return scores, path


class ViterbiDecoder(Layer):
    """ 
    Decode the highest scoring sequence of tags computed by transitions and potentials and get the viterbi path. 

    Args:
        transitions (`Tensor`): The transition matrix.  Its dtype is float32 and has a shape of `[num_tags, num_tags]`.
        include_bos_eos_tag (`bool`, optional): If set to True, the last row and the last column of transitions will be considered
            as start tag, the second to last row and the second to last column of transitions will be considered as stop tag. Defaults to ``True``.
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Shape:
        potentials (Tensor): The input tensor of unary emission. This is a 3-D tensor with shape of 
            [batch_size, sequence_length, num_tags]. The data type is float32 or float64. 
        lengths (Tensor):  The input tensor of length of each sequence. This is a 1-D tensor with shape of
            [batch_size]. The data type is int64. 

    Returns:
        scores(Tensor): The output tensor containing the score for the Viterbi sequence. The shape is [batch_size]
            and the data type is float32 or float64.
        paths(Tensor): The output tensor containing the highest scoring tag indices.  The shape is [batch_size, sequence_length]
            and the data type is int64.

    Example:
        .. code-block:: python

            import paddle
            paddle.seed(102)
            batch_size, seq_len, num_tags = 2, 4, 3
            emission = paddle.rand((batch_size, seq_len, num_tags), dtype='float32')
            length = paddle.randint(1, seq_len + 1, [batch_size])
            tags = paddle.randint(0, num_tags, [batch_size, seq_len])
            transition = paddle.rand((num_tags, num_tags), dtype='float32')
            decoder = paddle.text.ViterbiDecoder(transition, include_bos_eos_tag=False)
            scores, path = decoder(emission, length) # scores: [3.37089300, 1.56825531], path: [[1, 0, 0], [1, 1, 0]]
    """

    def __init__(self, transitions, include_bos_eos_tag=True, name=None):
        super(ViterbiDecoder, self).__init__()
        self.transitions = transitions
        self.include_bos_eos_tag = include_bos_eos_tag
        self.name = name

    def forward(self, potentials, lengths):
        return viterbi_decode(potentials, self.transitions, lengths,
                              self.include_bos_eos_tag, self.name)
