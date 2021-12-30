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
from paddlenlp.transformers import LayoutLMModel, LayoutLMTokenizer, LayoutLMForTokenClassification

from xfun import XFUNDataset
from utils import parse_args, get_bio_label_maps, print_arguments, set_seed
from eval_ser import evaluate
from losses import SERLoss
from ppocr.utils.logging import get_logger

MODELS = {
    'LayoutXLM':
    (LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForTokenClassification),
    'LayoutLM':
    (LayoutLMTokenizer, LayoutLMModel, LayoutLMForTokenClassification)
}


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    rank = paddle.distributed.get_rank()
    distributed = paddle.distributed.get_world_size() > 1

    logger = get_logger(log_file=os.path.join(args.output_dir, "train.log"))
    print_arguments(args, logger)

    label2id_map, id2label_map = get_bio_label_maps(args.label_map_path)
    loss_class = SERLoss(len(label2id_map))

    pad_token_label_id = loss_class.ignore_index

    # dist mode
    if distributed:
        paddle.distributed.init_parallel_env()

    tokenizer_class, base_model_class, model_class = MODELS[args.ser_model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if not args.resume:
        base_model = base_model_class.from_pretrained(args.model_name_or_path)
        model = model_class(
            base_model, num_classes=len(label2id_map), dropout=None)
        logger.info('train from scratch')
    else:
        logger.info('resume from {}'.format(args.model_name_or_path))
        model = model_class.from_pretrained(args.model_name_or_path)

    # dist mode
    if distributed:
        model = paddle.DataParallel(model)

    train_dataset = XFUNDataset(
        tokenizer,
        data_dir=args.train_data_dir,
        label_path=args.train_label_path,
        label2id_map=label2id_map,
        img_size=(224, 224),
        pad_token_label_id=pad_token_label_id,
        contains_re=False,
        add_special_ids=False,
        return_attention_mask=True,
        load_mode='all')
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

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.per_gpu_train_batch_size, shuffle=True)

    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        use_shared_memory=True,
        collate_fn=None, )

    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        num_workers=args.num_workers,
        use_shared_memory=True,
        collate_fn=None, )

    t_total = len(train_dataloader) * args.num_train_epochs

    # build linear decay with warmup lr sch
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=args.learning_rate,
        decay_steps=t_total,
        end_lr=0.0,
        power=1.0)
    if args.warmup_steps > 0:
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            lr_scheduler,
            args.warmup_steps,
            start_lr=0,
            end_lr=args.learning_rate, )

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed) = %d",
        args.per_gpu_train_batch_size * paddle.distributed.get_world_size(), )
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    set_seed(args.seed)
    best_metrics = None

    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()

    print_step = 1
    model.train()
    for epoch_id in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            train_reader_cost += time.time() - reader_start

            if args.ser_model_type == 'LayoutLM':
                if 'image' in batch:
                    batch.pop('image')
            labels = batch.pop('labels')

            train_start = time.time()
            outputs = model(**batch)
            train_run_cost += time.time() - train_start
            if args.ser_model_type == 'LayoutXLM':
                outputs = outputs[0]
            loss = loss_class(labels, outputs, batch['attention_mask'])

            # model outputs are always tuple in ppnlp (see doc)
            loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()  # Update learning rate schedule
            optimizer.clear_grad()
            global_step += 1
            total_samples += batch['input_ids'].shape[0]

            if rank == 0 and step % print_step == 0:
                logger.info(
                    "epoch: [{}/{}], iter: [{}/{}], global_step:{}, train loss: {:.6f}, lr: {:.6f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch_id, args.num_train_epochs, step,
                           len(train_dataloader), global_step,
                           loss.numpy()[0],
                           lr_scheduler.get_lr(), train_reader_cost /
                           print_step, (train_reader_cost + train_run_cost) /
                           print_step, total_samples / print_step, total_samples
                           / (train_reader_cost + train_run_cost)))

                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0

            if rank == 0 and args.eval_steps > 0 and global_step % args.eval_steps == 0 and args.evaluate_during_training:
                # Log metrics
                # Only evaluate when single GPU otherwise metrics may not average well
                results, _ = evaluate(args, model, tokenizer, loss_class,
                                      eval_dataloader, label2id_map,
                                      id2label_map, pad_token_label_id, logger)

                if best_metrics is None or results["f1"] >= best_metrics["f1"]:
                    best_metrics = copy.deepcopy(results)
                    output_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(output_dir, exist_ok=True)
                    if distributed:
                        model._layers.save_pretrained(output_dir)
                    else:
                        model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(args,
                                os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(
                        output_dir))

                logger.info("[epoch {}/{}][iter: {}/{}] results: {}".format(
                    epoch_id, args.num_train_epochs, step,
                    len(train_dataloader), results))
                if best_metrics is not None:
                    logger.info("best metrics: {}".format(best_metrics))
            reader_start = time.time()
        if rank == 0:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "latest_model")
            os.makedirs(output_dir, exist_ok=True)
            if distributed:
                model._layers.save_pretrained(output_dir)
            else:
                model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            paddle.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to {}".format(output_dir))
    return global_step, tr_loss / global_step


if __name__ == "__main__":
    args = parse_args()
    train(args)
