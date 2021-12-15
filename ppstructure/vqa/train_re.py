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
import numpy as np
import paddle

from paddlenlp.transformers import LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForRelationExtraction

from xfun import XFUNDataset
from utils import parse_args, get_bio_label_maps, print_arguments
from data_collator import DataCollator
from metric import re_score

from ppocr.utils.logging import get_logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def cal_metric(re_preds, re_labels, entities):
    gt_relations = []
    for b in range(len(re_labels)):
        rel_sent = []
        for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
            rel = {}
            rel["head_id"] = head
            rel["head"] = (entities[b]["start"][rel["head_id"]],
                           entities[b]["end"][rel["head_id"]])
            rel["head_type"] = entities[b]["label"][rel["head_id"]]

            rel["tail_id"] = tail
            rel["tail"] = (entities[b]["start"][rel["tail_id"]],
                           entities[b]["end"][rel["tail_id"]])
            rel["tail_type"] = entities[b]["label"][rel["tail_id"]]

            rel["type"] = 1
            rel_sent.append(rel)
        gt_relations.append(rel_sent)
    re_metrics = re_score(re_preds, gt_relations, mode="boundaries")
    return re_metrics


def evaluate(model, eval_dataloader, logger, prefix=""):
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = {}".format(len(eval_dataloader.dataset)))

    re_preds = []
    re_labels = []
    entities = []
    eval_loss = 0.0
    model.eval()
    for idx, batch in enumerate(eval_dataloader):
        with paddle.no_grad():
            outputs = model(**batch)
            loss = outputs['loss'].mean().item()
            if paddle.distributed.get_rank() == 0:
                logger.info("[Eval] process: {}/{}, loss: {:.5f}".format(
                    idx, len(eval_dataloader), loss))

            eval_loss += loss
        re_preds.extend(outputs['pred_relations'])
        re_labels.extend(batch['relations'])
        entities.extend(batch['entities'])
    re_metrics = cal_metric(re_preds, re_labels, entities)
    re_metrics = {
        "precision": re_metrics["ALL"]["p"],
        "recall": re_metrics["ALL"]["r"],
        "f1": re_metrics["ALL"]["f1"],
    }
    model.train()
    return re_metrics


def train(args):
    logger = get_logger(log_file=os.path.join(args.output_dir, "train.log"))
    print_arguments(args, logger)

    # Added here for reproducibility (even between python 2 and 3)
    set_seed(args.seed)

    label2id_map, id2label_map = get_bio_label_maps(args.label_map_path)
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    # dist mode
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)

    model = LayoutXLMModel.from_pretrained(args.model_name_or_path)
    model = LayoutXLMForRelationExtraction(model, dropout=None)

    # dist mode
    if paddle.distributed.get_world_size() > 1:
        model = paddle.distributed.DataParallel(model)

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
    args.train_batch_size = args.per_gpu_train_batch_size * \
                            max(1, paddle.distributed.get_world_size())
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
        format(args.train_batch_size * paddle.distributed.get_world_size()))
    logger.info("  Total optimization steps = {}".format(t_total))

    global_step = 0
    model.clear_gradients()
    train_dataloader_len = len(train_dataloader)
    best_metirc = {'f1': 0}
    model.train()

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            # model outputs are always tuple in ppnlp (see doc)
            loss = outputs['loss']
            loss = loss.mean()

            logger.info(
                "epoch: [{}/{}], iter: [{}/{}], global_step:{}, train loss: {}, lr: {}".
                format(epoch, args.num_train_epochs, step, train_dataloader_len,
                       global_step, np.mean(loss.numpy()), optimizer.get_lr()))

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # lr_scheduler.step()  # Update learning rate schedule

            global_step += 1

            if (paddle.distributed.get_rank() == 0 and args.eval_steps > 0 and
                    global_step % args.eval_steps == 0):
                # Log metrics
                if (paddle.distributed.get_rank() == 0 and args.
                        evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(model, eval_dataloader, logger)
                    if results['f1'] > best_metirc['f1']:
                        best_metirc = results
                        output_dir = os.path.join(args.output_dir,
                                                  "checkpoint-best")
                        os.makedirs(output_dir, exist_ok=True)
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        paddle.save(args,
                                    os.path.join(output_dir,
                                                 "training_args.bin"))
                        logger.info("Saving model checkpoint to {}".format(
                            output_dir))
                    logger.info("eval results: {}".format(results))
                    logger.info("best_metirc: {}".format(best_metirc))

            if (paddle.distributed.get_rank() == 0 and args.save_steps > 0 and
                    global_step % args.save_steps == 0):
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-latest")
                os.makedirs(output_dir, exist_ok=True)
                if paddle.distributed.get_rank() == 0:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(args,
                                os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(
                        output_dir))
    logger.info("best_metirc: {}".format(best_metirc))


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
