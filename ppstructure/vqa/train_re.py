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
import numpy as np
import paddle

from paddlenlp.transformers import LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForRelationExtraction

from xfun import XFUNDataset
from utils import parse_args, get_bio_label_maps, print_arguments, set_seed
from data_collator import DataCollator
from eval_re import evaluate

from ppocr.utils.logging import get_logger


def train(args):
    logger = get_logger(log_file=os.path.join(args.output_dir, "train.log"))
    rank = paddle.distributed.get_rank()
    distributed = paddle.distributed.get_world_size() > 1

    print_arguments(args, logger)

    # Added here for reproducibility (even between python 2 and 3)
    set_seed(args.seed)

    label2id_map, id2label_map = get_bio_label_maps(args.label_map_path)
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    # dist mode
    if distributed:
        paddle.distributed.init_parallel_env()

    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)
    if not args.resume:
        model = LayoutXLMModel.from_pretrained(args.model_name_or_path)
        model = LayoutXLMForRelationExtraction(model, dropout=None)
        logger.info('train from scratch')
    else:
        logger.info('resume from {}'.format(args.model_name_or_path))
        model = LayoutXLMForRelationExtraction.from_pretrained(
            args.model_name_or_path)

    # dist mode
    if distributed:
        model = paddle.DataParallel(model)

    train_dataset = XFUNDataset(
        tokenizer,
        data_dir=args.train_data_dir,
        label_path=args.train_label_path,
        label2id_map=label2id_map,
        img_size=(224, 224),
        max_seq_len=args.max_seq_length,
        pad_token_label_id=pad_token_label_id,
        contains_re=True,
        add_special_ids=False,
        return_attention_mask=True,
        load_mode='all')

    eval_dataset = XFUNDataset(
        tokenizer,
        data_dir=args.eval_data_dir,
        label_path=args.eval_label_path,
        label2id_map=label2id_map,
        img_size=(224, 224),
        max_seq_len=args.max_seq_length,
        pad_token_label_id=pad_token_label_id,
        contains_re=True,
        add_special_ids=False,
        return_attention_mask=True,
        load_mode='all')

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.per_gpu_train_batch_size, shuffle=True)

    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=8,
        use_shared_memory=True,
        collate_fn=DataCollator())

    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=DataCollator())

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
    grad_clip = paddle.nn.ClipGradByNorm(clip_norm=10)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        epsilon=args.adam_epsilon,
        grad_clip=grad_clip,
        weight_decay=args.weight_decay)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = {}".format(len(train_dataset)))
    logger.info("  Num Epochs = {}".format(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = {}".format(
        args.per_gpu_train_batch_size))
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = {}".
        format(args.per_gpu_train_batch_size *
               paddle.distributed.get_world_size()))
    logger.info("  Total optimization steps = {}".format(t_total))

    global_step = 0
    model.clear_gradients()
    train_dataloader_len = len(train_dataloader)
    best_metirc = {'f1': 0}
    model.train()

    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()

    print_step = 1

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()
            outputs = model(**batch)
            train_run_cost += time.time() - train_start
            # model outputs are always tuple in ppnlp (see doc)
            loss = outputs['loss']
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # lr_scheduler.step()  # Update learning rate schedule

            global_step += 1
            total_samples += batch['image'].shape[0]

            if rank == 0 and step % print_step == 0:
                logger.info(
                    "epoch: [{}/{}], iter: [{}/{}], global_step:{}, train loss: {:.6f}, lr: {:.6f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch, args.num_train_epochs, step,
                           train_dataloader_len, global_step,
                           np.mean(loss.numpy()),
                           optimizer.get_lr(), train_reader_cost / print_step, (
                               train_reader_cost + train_run_cost) / print_step,
                           total_samples / print_step, total_samples / (
                               train_reader_cost + train_run_cost)))

                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0

            if rank == 0 and args.eval_steps > 0 and global_step % args.eval_steps == 0 and args.evaluate_during_training:
                # Log metrics
                # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(model, eval_dataloader, logger)
                if results['f1'] >= best_metirc['f1']:
                    best_metirc = results
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
                logger.info("eval results: {}".format(results))
                logger.info("best_metirc: {}".format(best_metirc))
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
    logger.info("best_metirc: {}".format(best_metirc))


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
