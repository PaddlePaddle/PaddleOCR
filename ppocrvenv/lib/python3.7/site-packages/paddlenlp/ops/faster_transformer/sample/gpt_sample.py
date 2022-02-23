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
import sys
import os
import numpy as np
from attrdict import AttrDict
import argparse
import time

import paddle

import yaml
from pprint import pprint

from paddlenlp.ops import FasterGPT
from paddlenlp.transformers import GPTLMHeadModel
from paddlenlp.transformers import GPTChineseTokenizer, GPTTokenizer

from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt-cpm-large-cn": (GPTLMHeadModel, GPTChineseTokenizer),
    "gpt2-medium-en": (GPTLMHeadModel, GPTTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2-medium-en",
        type=str,
        help="The model name to specify the gpt to use. Can be one of ['gpt2-en', 'gpt2-medium-en', 'gpt-cpm-large-cn']. "
    )
    parser.add_argument(
        "--decoding_lib",
        default="../build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size. ")
    parser.add_argument(
        "--topk",
        default=4,
        type=int,
        help="The number of candidate to procedure beam search. ")
    parser.add_argument(
        "--topp",
        default=1.0,
        type=float,
        help="The probability threshold to procedure topp sampling. ")
    parser.add_argument(
        "--max_length", default=32, type=int, help="Maximum output length. ")
    parser.add_argument(
        "--start_token",
        default="<|endoftext|>",
        type=str,
        help="The start token. Defaults to <|endoftext|>. ")
    parser.add_argument(
        "--end_token",
        default="<|endoftext|>",
        type=str,
        help="The end token. Defaults to <|endoftext|>. ")
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="The temperature to set. ")
    parser.add_argument(
        "--use_fp16_decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_name_or_path]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    logger.info('Loading the model parameters, please wait...')
    model = model_class.from_pretrained(args.model_name_or_path)
    model.eval()

    bos_id = tokenizer.convert_tokens_to_ids(args.start_token)
    eos_id = tokenizer.convert_tokens_to_ids(args.end_token)

    # Define model
    gpt = model

    # Set evaluate mode
    gpt.eval()
    input_ids = np.array(
        [[bos_id] for i in range(args.batch_size * 1)]).astype("int64").reshape(
            [args.batch_size, 1])
    input_ids = paddle.to_tensor(input_ids)

    with paddle.no_grad():
        for i in range(100):
            # For warmup. 
            if 50 == i:
                paddle.device.cuda.synchronize(place)
                start = time.time()
            out_seq, _ = gpt.generate(
                input_ids,
                top_k=args.topk,
                top_p=args.topp,
                max_length=args.max_length,
                temperature=args.temperature,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                decode_strategy="sampling",
                use_fp16_decoding=args.use_fp16_decoding,
                use_faster=True)
            output_sequence = out_seq.numpy()

        paddle.device.cuda.synchronize(place)
        logger.info("Average test time for decoding is %f ms" % (
            (time.time() - start) / 50 * 1000))
        output_sequence = out_seq.numpy().tolist()
    for i in range(args.batch_size):
        print("========== Sample-%d ==========" % i)
        print(tokenizer.convert_ids_to_string(output_sequence[i]))


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_predict(args)
