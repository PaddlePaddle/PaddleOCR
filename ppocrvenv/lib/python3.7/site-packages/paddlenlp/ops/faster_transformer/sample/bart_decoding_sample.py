# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import time
from pprint import pprint

import paddle
from paddlenlp.ops import FasterBART
from paddlenlp.transformers import BartForConditionalGeneration, BartTokenizer
from paddlenlp.data import Pad
from paddlenlp.utils.log import logger


def postprocess_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def prepare_input(tokenizer, sentences, pad_id):
    word_pad = Pad(pad_id, dtype="int64")
    tokenized = tokenizer(sentences, return_length=True)
    inputs = word_pad([i["input_ids"] for i in tokenized])
    input_ids = paddle.to_tensor(inputs)
    return input_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="bart-base",
        type=str,
        help="The model name to specify the bart to use. Can be one of ['bart-base', 'bart-large',]. "
    )
    parser.add_argument(
        "--decoding_strategy",
        default='sampling',
        type=str,
        help="The decoding strategy. Can be one of [greedy_search, beam_search, sampling]"
    )
    parser.add_argument(
        "--beam_size",
        default=4,
        type=int,
        help="The parameters for beam search. ")
    parser.add_argument(
        "--top_k",
        default=4,
        type=int,
        help="The number of candidate to procedure beam search. ")
    parser.add_argument(
        "--top_p",
        default=1.0,
        type=float,
        help="The probability threshold to procedure topp sampling. ")
    parser.add_argument(
        "--max_length", default=50, type=int, help="Maximum output length. ")
    parser.add_argument(
        "--diversity_rate",
        default=0.0,
        type=float,
        help="The diversity of beam search. ")
    parser.add_argument(
        "--length_penalty",
        default=0.6,
        type=float,
        help="The power number in length penalty calculation")
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    paddle.set_device(place)

    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    logger.info('Loading the model parameters, please wait...')
    model = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path)
    # Set evaluate mode
    model.eval()
    sentences = [
        "I love that girl, but <mask> does not <mask> me.",
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
        "Drop everything now. Meet me in the pouring <mask>. Kiss me on the sidewalk.",
    ]

    bos_id = model.bart.config['bos_token_id']
    eos_id = model.bart.config['eos_token_id']
    pad_id = model.bart.config['pad_token_id']
    input_ids = prepare_input(tokenizer, sentences, pad_id)
    # Define model
    faster_bart = model

    # Set evaluate mode
    faster_bart.eval()

    with paddle.no_grad():
        for i in range(100):
            # For warmup.
            if 50 == i:
                # PaddlePaddle >= 2.2
                paddle.device.cuda.synchronize()
                start = time.perf_counter()
            finished_seq, _ = faster_bart.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                decode_strategy=args.decoding_strategy,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.beam_size,
                diversity_rate=args.diversity_rate,
                length_penalty=args.length_penalty,
                use_fp16_decoding=args.use_fp16_decoding,
                use_faster=True)

        paddle.device.cuda.synchronize()
        logger.info("Average test time for decoding is %f ms" % (
            (time.perf_counter() - start) / 50 * 1000))

        # Output
        finished_seq = finished_seq.numpy()
        for ins in finished_seq:
            generated_ids = postprocess_seq(ins, bos_id, eos_id)
            print(tokenizer.convert_ids_to_string(generated_ids))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_predict(args)
