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

import paddle

from paddlenlp.transformers import LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForRelationExtraction

from xfun import XFUNDataset
from vqa_utils import parse_args, get_bio_label_maps, print_arguments
from data_collator import DataCollator
from metric import re_score

from ppocr.utils.logging import get_logger


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


def eval(args):
    logger = get_logger()
    label2id_map, id2label_map = get_bio_label_maps(args.label_map_path)
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)

    model = LayoutXLMForRelationExtraction.from_pretrained(
        args.model_name_or_path)

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

    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=DataCollator())

    results = evaluate(model, eval_dataloader, logger)
    logger.info("eval results: {}".format(results))


if __name__ == "__main__":
    args = parse_args()
    eval(args)
