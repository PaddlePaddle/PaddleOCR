# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

from typing import List
import inspect
from abc import ABC

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.data_feeder import convert_dtype
from paddle.fluid.layers.utils import map_structure
from paddlenlp.utils.log import logger

__all__ = ["GenerationMixin"]


class BeamHypotheses:
    def __init__(self, num_beams, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, origin_len=0):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (((hyp.shape[-1] - origin_len + 5) / 6)
                                **self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len, origin_len=0):
        """
        If there are enough hypotheses and that none of the hypotheses being 
        generated can become better than the worst one in the heap, then we 
        are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / (
                (cur_len - origin_len + 5) / 6)**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class BeamSearchScorer(object):
    """
    implementing standard beam search decoding.
    """

    def __init__(self,
                 batch_size,
                 max_length,
                 num_beams,
                 length_penalty=1.0,
                 do_early_stopping=False,
                 num_beam_hyps_to_keep=1,
                 num_beam_groups=1):
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping)
            for _ in range(batch_size)
        ]
        self._done = paddle.to_tensor(
            [0 for _ in range(batch_size)], dtype='int64')

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                "`num_beams` has to be an integer strictly greater than 1, but "
                "received {}. For `num_beams` == 1, one should make use of "
                "`greedy_search` instead.".format(num_beams))

        if not isinstance(num_beam_groups, int) or (
                num_beam_groups > num_beams) or (
                    num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than "
                "`num_beams` and `num_beams` has to be divisible by "
                "`num_beam_groups`, but received num_beam_groups={}, num_beams="
                "{}.".format(num_beam_groups, num_beams))

    @property
    def is_done(self):
        return paddle.min(self._done) == 1

    def process(self,
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                origin_len=0,
                pad_token_id=None,
                eos_token_id=None):
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        next_beam_scores = paddle.zeros(
            [batch_size, self.group_size], dtype=next_scores.dtype)
        next_beam_tokens = paddle.zeros(
            [batch_size, self.group_size], dtype=next_tokens.dtype)
        next_beam_indices = paddle.zeros(
            [batch_size, self.group_size], dtype=next_indices.dtype)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx] == 1:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(
                    self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score,
                                  next_index) in enumerate(
                                      zip(next_tokens[batch_idx], next_scores[
                                          batch_idx], next_indices[batch_idx])):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (
                        next_token.numpy().item() == eos_token_id):
                    # If beam_token does not belong to top num_beams tokens,
                    # it should not be added
                    is_beam_token_worse_than_top_num_beams = (
                        beam_token_rank >= self.group_size)
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx.numpy().item()].clone(),
                        next_score.numpy().item(), origin_len)

                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token.numpy(
                    ).item()
                    next_beam_indices[batch_idx,
                                      beam_idx] = batch_beam_idx.numpy().item()
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    "At most {} tokens in `next_tokens[batch_idx]` can be equal "
                    "to `eos_token_id: {}`. Make sure `next_tokens[batch_idx]` "
                    "are corrected.".format(self.group_size, eos_token_id))

            # Check if we are done so that we can save a pad step if all(done)
            if beam_hyp.is_done(next_scores[batch_idx].max().numpy().item(),
                                cur_len, origin_len):
                self._done[batch_idx] = 1

        return {
            "next_beam_scores": next_beam_scores.reshape([-1]),
            "next_beam_tokens": next_beam_tokens.reshape([-1]),
            "next_beam_indices": next_beam_indices.reshape([-1])
        }

    def finalize(self,
                 input_ids,
                 final_beam_scores,
                 final_beam_tokens,
                 final_beam_indices,
                 origin_len=0,
                 pad_token_id=None,
                 eos_token_id=None):
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx] == 1:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].numpy().item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score, origin_len=origin_len)

        # select the best hypotheses
        sent_lengths = paddle.zeros(
            [batch_size * self.num_beam_hyps_to_keep], dtype=input_ids.dtype)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_score, best_hyp = sorted_hyps.pop()
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append([best_hyp, best_score])

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().numpy().item() + 1,
                           self.max_length)
        decoded = paddle.zeros(
            [batch_size * self.num_beam_hyps_to_keep, sent_max_len],
            dtype=input_ids.dtype)
        # shorter batches are padded if needed
        if sent_lengths.min().numpy().item() != sent_lengths.max().numpy().item(
        ):
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded[:, :] = pad_token_id
        decoded_score = paddle.zeros(
            [batch_size * self.num_beam_hyps_to_keep, 1])

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, score) in enumerate(best):
            decoded[i, :sent_lengths[i].numpy().item()] = hypo.numpy()
            decoded_score[i] = score
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i].numpy().item()] = eos_token_id
        return decoded, decoded_score


class GenerationMixin(object):
    r"""
    This class implements the interface for generation task. 
    
    It's used as the base class of `paddlenlp.transformers.PretrainedModel 
    <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.model_utils.html>`__.
    """

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no "
                             "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id,
                                              eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id))
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id
                              ).astype(paddle.get_default_dtype()) * -1e9
        else:
            attention_mask = paddle.zeros_like(
                input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    @staticmethod
    def prepare_seq_len_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id))
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            seq_len = paddle.sum(input_ids != pad_token_id,
                                 axis=1).unsqueeze(-1)
        else:
            seq_len = paddle.full(
                (input_ids.shape[0], 1), input_ids.shape[1], dtype="int64")
        return seq_len

    def get_logits_processor(self,
                             min_length=None,
                             max_length=None,
                             eos_token_id=None,
                             forced_bos_token_id=None,
                             forced_eos_token_id=None,
                             num_beams=1,
                             num_beam_groups=1,
                             diversity_rate=0.0,
                             repetition_penalty=None):
        processors = LogitsProcessorList()

        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(
                MinLengthLogitsProcessor(min_length, eos_token_id))
        if num_beam_groups > 1 and diversity_rate > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_rate=diversity_rate,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups))
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(
                RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if forced_bos_token_id is not None:
            processors.append(
                ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        # TODO
        # Add more pre_processing for distribution

        return processors

    @staticmethod
    def expand_inputs_for_generation(input_ids,
                                     expand_size,
                                     attention_mask=None,
                                     **model_kwargs):

        index = paddle.tile(
            paddle.arange(paddle.shape(input_ids)[0]).unsqueeze(-1),
            [1, expand_size]).reshape([-1])

        input_ids = paddle.gather(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.gather(attention_mask,
                                                           index)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.gather(token_type_ids,
                                                           index)

        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.gather(position_ids, index)

        if "seq_len" in model_kwargs:
            seq_len = model_kwargs["seq_len"]
            model_kwargs["seq_len"] = paddle.gather(seq_len, index)

        if "encoder_output" in model_kwargs:
            encoder_output = model_kwargs["encoder_output"]
            model_kwargs["encoder_output"] = paddle.gather(encoder_output,
                                                           index)

        return input_ids, model_kwargs

    @staticmethod
    def update_model_kwargs_for_generation(outputs,
                                           model_kwargs,
                                           is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple):
            model_kwargs["cache"] = outputs[1]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat(
                [position_ids, position_ids[:, -1].reshape((-1, 1)) + 1],
                axis=-1)

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == 'bool':
                attention_mask = paddle.cast(attention_mask, 'int64')
            attention_mask = nn.Pad2D(
                [0, 0, 0, 1], mode='replicate')(attention_mask)
            attention_mask = nn.Pad2D([0, 1, 0, 0], value=-1e9)(attention_mask)
            dtype = convert_dtype(attention_mask.dtype)
            if 'int' in dtype:
                attention_mask[:, :, -1, -1] = 1
            elif 'float' in dtype:
                attention_mask[:, :, -1, -1] = 0.0
            else:
                raise ValueError('The data type of input `attention_mask` must '
                                 'be bool, int or float')
            model_kwargs["attention_mask"] = attention_mask

        return model_kwargs

    @staticmethod
    def update_scores_for_generation(scores, next_scores, length,
                                     unfinished_flag):
        # update scores

        unfinished_scores = (scores * length + next_scores) / (length + 1)
        scores = paddle.where(unfinished_flag, unfinished_scores, scores)
        return scores

    def prepare_encoder_decoder_kwargs_for_generation(self, input_ids,
                                                      model_kwargs):
        if "encoder_output" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith(
                    "cross_attn"))
            }

            model_kwargs["encoder_output"] = encoder(input_ids,
                                                     **encoder_kwargs)

        return model_kwargs

    def prepare_decoder_input_ids_for_generation(self,
                                                 input_ids,
                                                 decoder_start_token_id=None,
                                                 bos_token_id=None):
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else getattr(
            self, 'decoder_start_token_id', None)
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else bos_token_id

        decoder_input_ids = paddle.ones(
            [input_ids.shape[0], 1], dtype="int64") * decoder_start_token_id

        return decoder_input_ids

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Implement in subclasses for custom behavior to prepare inputs in the
        # generate method.

        return {"input_ids": input_ids}

    def adjust_logits_during_generation(self, logits):
        # Implement in subclasses for custom behavior to adjust the logits in
        # the generate method.

        return logits

    def prepare_faster_entry(self, kwargs):
        return False

    def _convert_to_faster(self, kwargs):
        # try general convert
        pass

    def _build_faster(self, kwargs):
        self._faster_entry = False
        if kwargs['min_length'] != 0:
            # not support for min_length yet in the faster version
            raise AttributeError(
                "'min_length != 0' is not supported yet in the faster version")
        if kwargs['num_beam_groups'] != 1:
            # not support for group_beam_search yet in the faster version
            raise AttributeError(
                "'num_beam_groups != 1' is not supported yet in the faster version"
            )
        self.prepare_faster_entry(kwargs)

    @paddle.no_grad()
    def generate(self,
                 input_ids=None,
                 max_length=20,
                 min_length=0,
                 decode_strategy='greedy_search',
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 repetition_penalty=1.0,
                 num_beams=1,
                 num_beam_groups=1,
                 length_penalty=0.0,
                 early_stopping=False,
                 bos_token_id=None,
                 eos_token_id=None,
                 pad_token_id=None,
                 decoder_start_token_id=None,
                 forced_bos_token_id=None,
                 forced_eos_token_id=None,
                 num_return_sequences=1,
                 diversity_rate=0.0,
                 use_cache=True,
                 use_faster=False,
                 use_fp16_decoding=False,
                 **model_kwargs):
        r"""
        The interface for generation task. This method can generate sequences 
        by using decoding strategy. Currently, there are three decoding 
        strategies supported: "greedy_search", "sampling" and "beam_search".

        Args:
            input_ids (Tensor, optional): The input sequence ids for the 
                generation. It is a Tensor with shape [batch_size, sequence_length]. 
                The data type should be int32 or int64. Default to None, which 
                we will initialize it as a Tensor with shape [1, 1], filled 
                with the value `bos_token_id`.
            max_length (int, optional): The maximum length of the sequence to 
                be generated. Default to 20.
            min_length (int, optional): The minimum length of the sequence to 
                be generated. Default to 0.
            decode_strategy (str, optional): The decoding strategy in generation.
                Currently, there are three decoding strategies supported: 
                "greedy_search", "sampling" and "beam_search". Default to 
                "greedy_search".
            temperature (float, optional): The value used to module the next 
                token probabilities in the "sampling" strategy. Default to 1.0, 
                which means no effect.
            top_k (int, optional): The number of highest probability tokens to 
                keep for top-k-filtering in the "sampling" strategy. Default to 
                0, which means no effect.
            top_p (float, optional): The cumulative probability for 
                top-p-filtering in the "sampling" strategy. The value should 
                satisfy :math:`0 <= top\_p < 1`. Default to 1.0, which means no 
                effect.
            repetition_penalty (float, optional):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details. Defaults to 1.0.
            num_beams (int, optional): The number of beams in the "beam_search"
                strategy. Default to 1.
            num_beam_groups (int, optional):
                Number of groups to divide `num_beams` into in order to use DIVERSE
                BEAM SEARCH. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ 
                for more details. Default to 1.
            length_penalty (float, optional): The exponential penalty to the 
                sequence length in the "beam_search" strategy. The larger this
                param is, the more that the model would generate shorter 
                sequences. Default to 0.0, which means no penalty.
            early_stopping (bool, optional): Whether to stop searching in the 
                "beam_search" strategy when at least `num_beams` sentences are 
                finished per batch or not. Default to False.
            bos_token_id (int, optional): The id of the `bos_token`. Default to 
                None.
            eos_token_id (int, optional): The id of the `eos_token`. Default to 
                None.
            pad_token_id (int, optional): The id of the `pad_token`. Default to 
                None.
            decoder_start_token_id (int, optional): The start token id for 
                encoder-decoder models. Default to None.
            forced_bos_token_id (int, optional): The id of the token to force as 
                the first generated token. Usually use for multilingual models.
                Default to None.
            forced_eos_token_id (int, optional): The id of the token to force as 
                the last generated token. Default to None.
            num_return_sequences (int, optional): The number of returned 
                sequences for each sequence in the batch. Default to 1.
            diversity_rate (float, optional): If num_beam_groups is 1, this is the 
                diversity_rate for Diverse Siblings Search. See 
                `this paper https://arxiv.org/abs/1611.08562`__ for more details. 
                If not, this is the diversity_rate for DIVERSE BEAM SEARCH.
            use_cache: (bool, optional): Whether to use the model cache to 
                speed up decoding. Default to True.
            use_faster: (bool, optional): Whether to use faster entry of model 
                for FasterGeneration. Default to False.
            use_fp16_decoding: (bool, optional): Whether to use fp16 for decoding. 
                Only works when faster entry is avalible. Default to False.
            model_kwargs (dict): It can be used to specify additional kwargs 
                passed to the model.

        Returns:
            tuple[Tensor]: It is a tuple contains two elements: ids and scores. 
            Each element is a Tensor.

            With the fields:

            - ids (Tensor): 
                The ids of the generated sequences. It is a Tensor with shape 
                [batch_size * num_return_sequences, sequence_length]. The data 
                type is same as the input `input_ids`.
            - scores (Tensor): 
                The scores of the generated sequences. It is a Tensor with shape 
                [batch_size * num_return_sequences, 1]. The data type is float32 
                or float64, which is the same as the parameters in the model.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import (
                    UnifiedTransformerLMHeadModel, 
                    UnifiedTransformerTokenizer
                )

                paddle.seed(2)

                # Initialize the model and tokenizer
                model_name_or_path = 'unified_transformer-12L-cn-luge'
                model = UnifiedTransformerLMHeadModel.from_pretrained(model_name_or_path)
                tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name_or_path)

                # Prepare the model inputs.
                history = "早上好，今天空气质量不错。"
                inputs = tokenizer.dialogue_encode(history, task_type='chitchat', 
                    add_start_token_as_response=True, return_tensors=True)
                
            .. code-block::

                # Generate the sequence by using "greedy_search" strategy
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="greedy_search")
                print(ids.shape, scores.shape)
                # [1, 3] [1, 1]
                sequence_ids = ids.numpy().tolist()[0]
                sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                response = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                print(response)
                # 是的

            .. code-block::
            
                # Generate 2 sequences by using "sampling" strategy (top_k=5)
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="sampling",
                    top_k=5,
                    num_return_sequences=2)
                print(ids.shape, scores.shape)
                # [2, 7] [2, 1]
                response = []
                for sequence_ids in ids.numpy().tolist():
                    sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                    text = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                    response.append(text)
                print(response)
                # ['天气好,心情也好', '你也是']

            .. code-block::
            
                # Generate 2 sequences by using "beam_search" strategy (num_beams=5)
                ids, scores = model.generate(
                    input_ids=inputs['input_ids'],
                    token_type_ids=inputs['token_type_ids'],
                    position_ids=inputs['position_ids'],
                    attention_mask=inputs['attention_mask'],
                    decode_strategy="beam_search",
                    num_beams=5,
                    num_return_sequences=2)
                print(ids.shape, scores.shape)
                # [2, 3] [2, 1]
                response = []
                for sequence_ids in ids.numpy().tolist():
                    sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                    text = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                    response.append(text)
                print(response)
                # ['是的', '嗯嗯']
        """

        assert (
            decode_strategy in ["greedy_search", "sampling", "beam_search"]
        ), "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            decode_strategy)

        bos_token_id = bos_token_id if bos_token_id is not None else getattr(
            self, 'bos_token_id', None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(
            self, 'eos_token_id', None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(
            self, 'pad_token_id', None)
        forced_bos_token_id = forced_bos_token_id if forced_bos_token_id is not None else getattr(
            self, 'forced_bos_token_id', None)
        forced_eos_token_id = forced_eos_token_id if forced_eos_token_id is not None else getattr(
            self, 'forced_eos_token_id', None)
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else getattr(
            self, 'decoder_start_token_id', None)

        if getattr(self, '_faster_entry', None) is not False and use_faster:
            args = locals()
            args.pop('self')
            args.pop("__class__", None)
            model_kwargs = args.pop('model_kwargs')
            args.update(model_kwargs)
            try:
                if not hasattr(self, '_faster_entry'):
                    self._build_faster(args)
                if self._faster_entry:
                    output_ids = self._faster_entry(**args)
                    if decode_strategy == "beam_search":
                        output_ids = output_ids.transpose([1, 2, 0])
                        output_ids = output_ids[:, :
                                                num_return_sequences, :].reshape(
                                                    [-1, output_ids.shape[-1]])
                    else:
                        output_ids = output_ids.transpose([1, 0])
                    # make result and faster result oneconsistent
                    dummy_srore = None
                    return output_ids, dummy_srore

            except Exception as e:
                args['model_kwargs'] = model_kwargs
                #TODO
                # Prevent self._convert_to_faster to throw Exception
                self._convert_to_faster(args)
                logger.warning(e)
                logger.warning(
                    "FasterGeneration is not available, "
                    "and the original version would be used instead.")

        # params check
        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs[
                "attention_mask"] = self.prepare_attention_mask_for_generation(
                    input_ids, pad_token_id, eos_token_id)
        self.is_encoder_decoder = hasattr(self, 'encoder') and hasattr(
            self, 'decoder')
        if self.is_encoder_decoder:
            model_kwargs = self.prepare_encoder_decoder_kwargs_for_generation(
                input_ids, model_kwargs)
            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self.prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id, bos_token_id)

        if pad_token_id is None and eos_token_id is not None:
            print("Setting `pad_token_id` to `eos_token_id`:{} for "
                  "open-end generation.".format(eos_token_id))
            pad_token_id = eos_token_id

        model_kwargs["use_cache"] = use_cache
        max_length += input_ids.shape[-1]
        min_length += input_ids.shape[-1]

        logits_processors = self.get_logits_processor(
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty)

        if decode_strategy == 'greedy_search':
            if num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(num_return_sequences))

            return self.greedy_search(input_ids, logits_processors, max_length,
                                      pad_token_id, eos_token_id,
                                      **model_kwargs)

        elif decode_strategy == 'sampling':
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs)

            return self.sample(input_ids, logits_processors, max_length,
                               pad_token_id, eos_token_id, top_k, top_p,
                               temperature, **model_kwargs)

        elif decode_strategy == 'beam_search':
            batch_size = input_ids.shape[0]
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to "
                    "`num_beams`. But received `num_return_sequences` is {}, "
                    "`num_beams` is {}".format(num_return_sequences, num_beams))
            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` has to be bigger than 1. But received "
                    "`num_beams` is {}. If `num_beams` is 1, `decode_strategy` "
                    "should be 'greedy_search'".format(num_beams))
            if num_beam_groups > 1:
                diverse_beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences,
                    num_beam_groups=num_beam_groups)

                # interleave with `num_beams`
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_beams, **model_kwargs)

                return self.group_beam_search(input_ids, diverse_beam_scorer,
                                              logits_processors, max_length,
                                              diversity_rate, pad_token_id,
                                              eos_token_id, **model_kwargs)
            else:
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_early_stopping=early_stopping,
                    num_beam_hyps_to_keep=num_return_sequences)

                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_beams, **model_kwargs)

                return self.beam_search(
                    input_ids, beam_scorer, logits_processors, max_length,
                    diversity_rate, pad_token_id, eos_token_id, **model_kwargs)

    def greedy_search(self, input_ids, logits_processors, max_length,
                      pad_token_id, eos_token_id, **model_kwargs):
        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype='bool')
        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]
            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)
            # greedy
            probs = F.softmax(logits)
            probs = paddle.log(probs)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens,
                                           paddle.full_like(next_tokens,
                                                            pad_token_id))

            scores = self.update_scores_for_generation(
                scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(
                    unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.is_encoder_decoder)
        return input_ids[:, origin_len:], scores

    def sample(self,
               input_ids,
               logits_processors,
               max_length,
               pad_token_id,
               eos_token_id,
               top_k=None,
               top_p=None,
               temperature=None,
               min_tokens_to_keep=1,
               **model_kwargs):
        def TopKProcess(probs, top_k, min_tokens_to_keep):
            top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
            # Remove all tokens with a probability less than the last token of the top-k
            topk_probs, _ = paddle.topk(probs, k=top_k)
            probs = paddle.where(probs >= topk_probs[:, -1:], probs,
                                 paddle.full_like(probs, 0.0))
            return probs

        def TopPProcess(probs, top_p, min_tokens_to_keep):
            sorted_probs = paddle.sort(probs, descending=True)
            sorted_indices = paddle.argsort(probs, descending=True)
            cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

            # Remove tokens with cumulative probs above the top_p, But keep at
            # least min_tokens_to_keep tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Set 'min_tokens_to_keep - 1' because the first token is kept
                sorted_indices_to_remove[:, :min_tokens_to_keep - 1] = 0
            # Keep the first token
            sorted_indices_to_remove = paddle.cast(
                sorted_indices_to_remove, dtype='int64')
            sorted_indices_to_remove[:, 1:] = (
                sorted_indices_to_remove[:, :-1].clone())
            sorted_indices_to_remove[:, 0] = 0

            # Scatter sorted tensors to original indexing
            sorted_indices = sorted_indices + paddle.arange(probs.shape[
                0]).unsqueeze(-1) * probs.shape[-1]
            condition = paddle.scatter(sorted_indices_to_remove.flatten(),
                                       sorted_indices.flatten(),
                                       sorted_indices_to_remove.flatten())
            condition = paddle.cast(condition, 'bool').reshape(probs.shape)
            probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
            return probs

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype='bool')
        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)
            next_tokens = paddle.multinomial(probs)

            next_scores = paddle.index_sample(origin_probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens,
                                           paddle.full_like(next_tokens,
                                                            pad_token_id))

            scores = self.update_scores_for_generation(
                scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(
                    unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.is_encoder_decoder)
        return input_ids[:, origin_len:], scores

    def beam_search(self, input_ids, beam_scorer, logits_processors, max_length,
                    diversity_rate, pad_token_id, eos_token_id, **model_kwargs):
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size)

        beam_scores = paddle.zeros(
            (batch_size, num_beams), dtype=paddle.get_default_dtype())
        beam_scores[:, 1:] = -1e9
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)

            outputs = self(**model_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            # beam search
            # [batch_size * num_beams, vocab_size]
            next_scores = F.softmax(logits)
            next_scores = paddle.log(next_scores)
            next_scores = logits_processors(input_ids, next_scores)
            next_scores = next_scores + beam_scores.unsqueeze(-1)

            vocab_size = next_scores.shape[-1]
            if diversity_rate == 0.0:
                # reshape for beam search
                next_scores = next_scores.reshape(
                    [batch_size, num_beams * vocab_size])

                next_scores, next_tokens = paddle.topk(
                    next_scores, 2 * num_beams, axis=1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

            else:
                next_scores, next_tokens = paddle.topk(
                    next_scores, 2 * num_beams, axis=1)

                sibling_score = paddle.arange(
                    1, 2 * num_beams + 1).unsqueeze(0) * diversity_rate

                diversed_score = next_scores - sibling_score

                next_scores = next_scores.reshape(
                    [batch_size, 2 * num_beams * num_beams])
                next_tokens = next_tokens.reshape(
                    [batch_size, 2 * num_beams * num_beams])

                diversed_score = diversed_score.reshape(
                    [batch_size, 2 * num_beams * num_beams])
                diversed_score, diversed_tokens = paddle.topk(
                    diversed_score, 2 * num_beams, axis=1)

                # TODO
                # Use gather_nd() to select origan token and score
                next_scores = paddle.stack([
                    paddle.index_select(next_scores[i], diversed_tokens[i])
                    for i in range(next_scores.shape[0])
                ])
                next_tokens = paddle.stack([
                    paddle.index_select(next_tokens[i], diversed_tokens[i])
                    for i in range(next_tokens.shape[0])
                ])

                next_indices = diversed_tokens // (2 * num_beams)

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                origin_len=origin_len,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id, )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            cur_len += 1
            input_ids = paddle.concat(
                [
                    paddle.index_select(input_ids, beam_idx),
                    beam_next_tokens.unsqueeze(-1)
                ],
                axis=-1)

            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.is_encoder_decoder)
            if model_kwargs["cache"] is not None:
                # reorder the cache
                model_kwargs["cache"] = map_structure(
                    lambda x: paddle.index_select(x, beam_idx),
                    model_kwargs["cache"])

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            origin_len=origin_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id)
        return pred_ids[:, origin_len:], scores

    def group_beam_search(self, input_ids, beam_scorer, logits_processors,
                          max_length, diversity_rate, pad_token_id,
                          eos_token_id, **model_kwargs):

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups

        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size)

        beam_scores = paddle.full(
            (batch_size, num_beams), -1e9, dtype="float32")
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # predicted tokens in cur_len step
            current_tokens = paddle.zeros(
                shape=[batch_size * num_beams], dtype=input_ids.dtype)

            # indices which will form the beams in the next time step
            reordering_indices = paddle.zeros(
                shape=[batch_size * num_beams], dtype="int64")
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs)

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend([
                        batch_idx * num_beams + idx
                        for idx in range(group_start_idx, group_end_idx)
                    ])

                group_input_ids = input_ids[batch_group_indices]
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                # select outputs of beams of current group only

                logits = logits[:, -1, :]
                logits = paddle.index_select(
                    logits, paddle.to_tensor(batch_group_indices))
                logits = self.adjust_logits_during_generation(logits)

                next_scores = F.softmax(logits)
                next_scores = paddle.log(next_scores)
                vocab_size = next_scores.shape[-1]

                next_scores = logits_processors(
                    group_input_ids,
                    next_scores,
                    current_tokens=current_tokens,
                    beam_group_idx=beam_group_idx)

                next_scores = next_scores + beam_scores[
                    batch_group_indices].unsqueeze(-1)

                # reshape for beam search
                next_scores = next_scores.reshape(
                    [batch_size, group_size * vocab_size])

                next_scores, next_tokens = paddle.topk(
                    next_scores, 2 * group_size, axis=1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_scores,
                    next_tokens,
                    next_indices,
                    origin_len=origin_len,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id, )

                beam_scores[batch_group_indices] = beam_outputs[
                    "next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = paddle.concat(
                    [
                        paddle.index_select(
                            group_input_ids, index=beam_idx),
                        beam_next_tokens.unsqueeze(-1)
                    ],
                    axis=-1)
                current_tokens[batch_group_indices] = beam_next_tokens

                reordering_indices[batch_group_indices] = (
                    num_beams * (beam_idx // group_size) + group_start_idx +
                    (beam_idx % group_size))

            input_ids = paddle.concat(
                [input_ids, current_tokens.unsqueeze(-1)], axis=-1)

            cur_len += 1
            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.is_encoder_decoder)
            if model_kwargs["cache"] is not None:
                # reorder the cache
                model_kwargs["cache"] = map_structure(
                    lambda x: paddle.index_select(x, reordering_indices),
                    model_kwargs["cache"])

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            origin_len=origin_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id)
        return pred_ids[:, origin_len:], scores


class LogitsProcessorList(List):
    def __call__(self, input_ids, logits, **kwargs):
        for processor in self:
            processor_args = inspect.signature(processor.__call__).parameters
            if len(processor_args) > 2:
                assert all(
                    arg in kwargs for arg in list(processor_args.keys())[2:]
                ), f"The parameters don't match for {processor.__class__}"
                logits = processor(input_ids, logits, **kwargs)
            else:
                logits = processor(input_ids, logits)
        return logits


class LogitsProcessor(ABC):
    """
    Abstract base class for all logit processors that can be applied during 
    generation.
    """

    def __call__(self, input_ids, logits):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. "
            "Only classes inheriting this class can be called.")


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (int): The minimum length of generation sequence.
        eos_token_id (int): The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length, eos_token_id):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(
                "`min_length` should be a positive integer, but get {}".format(
                    min_length))

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(
                "`eos_token_id` should be a positive integer, but get {}".
                format(eos_token_id))

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, logits):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            logits[:, self.eos_token_id] = -float("inf")
        return logits


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty

    def __call__(self, input_ids, logits):
        score = paddle.index_sample(logits, input_ids)
        score = paddle.where(score < 0, score * self.penalty,
                             score / self.penalty)
        input_ids = input_ids + paddle.arange(logits.shape[0]).unsqueeze(
            -1) * logits.shape[-1]
        outputs = paddle.scatter(logits.flatten(),
                                 input_ids.flatten(),
                                 score.flatten()).reshape(logits.shape)
        return outputs


class HammingDiversityLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces diverse beam search. Note that this logits
    processor is only effective for `group_beam_search`. See 
    `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.

    Args:
        diversity_rate (float): This value is subtracted from a beam's score if 
            it generates a token same as any beam from other group at a particular 
            time. 
        num_beams (int): Number of beams used for group beam search. 
        num_beam_groups (int): Number of groups to divide `num_beams` into in order 
            to ensure diversity among different groups of beams. 
    """

    def __init__(self, diversity_rate, num_beams, num_beam_groups):
        if not isinstance(diversity_rate, float) or (not diversity_rate > 0.0):
            raise ValueError(
                "`diversity_rate` should be a float strictly larger than 0.")
        self._diversity_rate = diversity_rate
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError(
                "`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError(
                "`num_beam_groups` should be an integer strictly larger than 1.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(self, input_ids, scores, current_tokens, beam_group_idx):
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams,
                            self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            previous_group_tokens = current_tokens[batch_idx * self._num_beams:
                                                   batch_idx * self._num_beams +
                                                   group_start_idx]
            token_frequency = paddle.bincount(
                previous_group_tokens, minlength=vocab_size)
            scores[batch_idx * group_size:(batch_idx + 1) *
                   group_size] -= self._diversity_rate * token_frequency

        return scores


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the first generated token to be the selected `forced_bos_token`.

    Args:
        forced_bos_token_id (:obj:`int`):
            The id of the token to to be generated as the first token.
    """

    def __init__(self, forced_bos_token_id):
        self.forced_bos_token_id = forced_bos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, [
                i for i in range(num_tokens) if i != self.forced_bos_token_id
            ]] = -float("inf")
            scores[:, self.forced_bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the last generated token to be the selected `forced_eos_token`.

    Args:
        max_length (int): The maximum length of the sequence to be generated.
        forced_eos_token_id (int): The id of the token to to be generated as the last token.
    """

    def __init__(self, max_length, forced_eos_token_id):
        self.max_length = max_length
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [
                i for i in range(num_tokens) if i != self.forced_eos_token_id
            ]] = -1e9  #TODO change back to -inf after paddle.topk is fixed
            scores[:, self.forced_eos_token_id] = 0
        return scores
