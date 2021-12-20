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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import random
import time
import copy
import logging

import argparse
import paddle
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForTokenClassification
from xfun import XFUNDataset
from utils import parse_args, get_bio_label_maps, print_arguments

from ppocr.utils.logging import get_logger


def eval(args):
    logger = get_logger()
    print_arguments(args, logger)

    label2id_map, id2label_map = get_bio_label_maps(args.label_map_path)
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)
    model = LayoutXLMForTokenClassification.from_pretrained(
        args.model_name_or_path)

    eval_dataset = XFUNDataset(
        tokenizer,
        data_dir=args.eval_data_dir,
        label_path=args.eval_label_path,
        label2id_map=label2id_map,
        img_size=(224, 224),
        pad_token_label_id=pad_token_label_id,
        contains_re=False,
        add_special_ids=False,
        return_attention_mask=True,
        load_mode='all')

    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        num_workers=8,
        use_shared_memory=True,
        collate_fn=None, )

    results, _ = evaluate(args, model, tokenizer, eval_dataloader, label2id_map,
                          id2label_map, pad_token_label_id, logger)

    logger.info(results)


def evaluate(args,
             model,
             tokenizer,
             eval_dataloader,
             label2id_map,
             id2label_map,
             pad_token_label_id,
             logger,
             prefix=""):

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for idx, batch in enumerate(eval_dataloader):
        with paddle.no_grad():
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs[:2]

            tmp_eval_loss = tmp_eval_loss.mean()

            if paddle.distributed.get_rank() == 0:
                logger.info("[Eval]process: {}/{}, loss: {:.5f}".format(
                    idx, len(eval_dataloader), tmp_eval_loss.numpy()[0]))

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.numpy()
            out_label_ids = batch["labels"].numpy()
        else:
            preds = np.append(preds, logits.numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, batch["labels"].numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    # label_map = {i: label.upper() for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(id2label_map[out_label_ids[i][j]])
                preds_list[i].append(id2label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    with open(
            os.path.join(args.output_dir, "test_gt.txt"), "w",
            encoding='utf-8') as fout:
        for lbl in out_label_list:
            for l in lbl:
                fout.write(l + "\t")
            fout.write("\n")
    with open(
            os.path.join(args.output_dir, "test_pred.txt"), "w",
            encoding='utf-8') as fout:
        for lbl in preds_list:
            for l in lbl:
                fout.write(l + "\t")
            fout.write("\n")

    report = classification_report(out_label_list, preds_list)
    logger.info("\n" + report)

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    model.train()
    return results, preds_list


if __name__ == "__main__":
    args = parse_args()
    eval(args)
