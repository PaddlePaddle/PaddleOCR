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
import json
import cv2
import numpy as np
from copy import deepcopy

import paddle

# relative reference
from utils import parse_args, get_image_file_list, draw_ser_results, get_bio_label_maps
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForTokenClassification
from paddlenlp.transformers import LayoutLMModel, LayoutLMTokenizer, LayoutLMForTokenClassification

MODELS = {
    'LayoutXLM':
    (LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForTokenClassification),
    'LayoutLM':
    (LayoutLMTokenizer, LayoutLMModel, LayoutLMForTokenClassification)
}


def pad_sentences(tokenizer,
                  encoded_inputs,
                  max_seq_len=512,
                  pad_to_max_seq_len=True,
                  return_attention_mask=True,
                  return_token_type_ids=True,
                  return_overflowing_tokens=False,
                  return_special_tokens_mask=False):
    # Padding with larger size, reshape is carried out
    max_seq_len = (
        len(encoded_inputs["input_ids"]) // max_seq_len + 1) * max_seq_len

    needs_to_be_padded = pad_to_max_seq_len and \
                         max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

    if needs_to_be_padded:
        difference = max_seq_len - len(encoded_inputs["input_ids"])
        if tokenizer.padding_side == 'right':
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"]) + [0] * difference
            if return_token_type_ids:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] +
                    [tokenizer.pad_token_type_id] * difference)
            if return_special_tokens_mask:
                encoded_inputs["special_tokens_mask"] = encoded_inputs[
                    "special_tokens_mask"] + [1] * difference
            encoded_inputs["input_ids"] = encoded_inputs[
                "input_ids"] + [tokenizer.pad_token_id] * difference
            encoded_inputs["bbox"] = encoded_inputs["bbox"] + [[0, 0, 0, 0]
                                                               ] * difference
        else:
            assert False, "padding_side of tokenizer just supports [\"right\"] but got {}".format(
                tokenizer.padding_side)
    else:
        if return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                "input_ids"])

    return encoded_inputs


def split_page(encoded_inputs, max_seq_len=512):
    """
    truncate is often used in training process
    """
    for key in encoded_inputs:
        encoded_inputs[key] = paddle.to_tensor(encoded_inputs[key])
        if encoded_inputs[key].ndim <= 1:  # for input_ids, att_mask and so on
            encoded_inputs[key] = encoded_inputs[key].reshape([-1, max_seq_len])
        else:  # for bbox
            encoded_inputs[key] = encoded_inputs[key].reshape(
                [-1, max_seq_len, 4])
    return encoded_inputs


def preprocess(
        tokenizer,
        ori_img,
        ocr_info,
        img_size=(224, 224),
        pad_token_label_id=-100,
        max_seq_len=512,
        add_special_ids=False,
        return_attention_mask=True, ):
    ocr_info = deepcopy(ocr_info)
    height = ori_img.shape[0]
    width = ori_img.shape[1]

    img = cv2.resize(ori_img,
                     (224, 224)).transpose([2, 0, 1]).astype(np.float32)

    segment_offset_id = []
    words_list = []
    bbox_list = []
    input_ids_list = []
    token_type_ids_list = []

    for info in ocr_info:
        # x1, y1, x2, y2
        bbox = info["bbox"]
        bbox[0] = int(bbox[0] * 1000.0 / width)
        bbox[2] = int(bbox[2] * 1000.0 / width)
        bbox[1] = int(bbox[1] * 1000.0 / height)
        bbox[3] = int(bbox[3] * 1000.0 / height)

        text = info["text"]
        encode_res = tokenizer.encode(
            text, pad_to_max_seq_len=False, return_attention_mask=True)

        if not add_special_ids:
            # TODO: use tok.all_special_ids to remove
            encode_res["input_ids"] = encode_res["input_ids"][1:-1]
            encode_res["token_type_ids"] = encode_res["token_type_ids"][1:-1]
            encode_res["attention_mask"] = encode_res["attention_mask"][1:-1]

        input_ids_list.extend(encode_res["input_ids"])
        token_type_ids_list.extend(encode_res["token_type_ids"])
        bbox_list.extend([bbox] * len(encode_res["input_ids"]))
        words_list.append(text)
        segment_offset_id.append(len(input_ids_list))

    encoded_inputs = {
        "input_ids": input_ids_list,
        "token_type_ids": token_type_ids_list,
        "bbox": bbox_list,
        "attention_mask": [1] * len(input_ids_list),
    }

    encoded_inputs = pad_sentences(
        tokenizer,
        encoded_inputs,
        max_seq_len=max_seq_len,
        return_attention_mask=return_attention_mask)

    encoded_inputs = split_page(encoded_inputs)

    fake_bs = encoded_inputs["input_ids"].shape[0]

    encoded_inputs["image"] = paddle.to_tensor(img).unsqueeze(0).expand(
        [fake_bs] + list(img.shape))

    encoded_inputs["segment_offset_id"] = segment_offset_id

    return encoded_inputs


def postprocess(attention_mask, preds, label_map_path):
    if isinstance(preds, paddle.Tensor):
        preds = preds.numpy()
    preds = np.argmax(preds, axis=2)

    _, label_map = get_bio_label_maps(label_map_path)

    preds_list = [[] for _ in range(preds.shape[0])]

    # keep batch info
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if attention_mask[i][j] == 1:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list


def merge_preds_list_with_ocr_info(label_map_path, ocr_info, segment_offset_id,
                                   preds_list):
    # must ensure the preds_list is generated from the same image
    preds = [p for pred in preds_list for p in pred]
    label2id_map, _ = get_bio_label_maps(label_map_path)
    for key in label2id_map:
        if key.startswith("I-"):
            label2id_map[key] = label2id_map["B" + key[1:]]

    id2label_map = dict()
    for key in label2id_map:
        val = label2id_map[key]
        if key == "O":
            id2label_map[val] = key
        if key.startswith("B-") or key.startswith("I-"):
            id2label_map[val] = key[2:]
        else:
            id2label_map[val] = key

    for idx in range(len(segment_offset_id)):
        if idx == 0:
            start_id = 0
        else:
            start_id = segment_offset_id[idx - 1]

        end_id = segment_offset_id[idx]

        curr_pred = preds[start_id:end_id]
        curr_pred = [label2id_map[p] for p in curr_pred]

        if len(curr_pred) <= 0:
            pred_id = 0
        else:
            counts = np.bincount(curr_pred)
            pred_id = np.argmax(counts)
        ocr_info[idx]["pred_id"] = int(pred_id)
        ocr_info[idx]["pred"] = id2label_map[pred_id]
    return ocr_info


@paddle.no_grad()
def infer(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # init token and model
    tokenizer_class, base_model_class, model_class = MODELS[args.ser_model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)

    model.eval()

    # load ocr results json
    ocr_results = dict()
    with open(args.ocr_json_path, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            img_name, json_info = line.split("\t")
            ocr_results[os.path.basename(img_name)] = json.loads(json_info)

    # get infer img list
    infer_imgs = get_image_file_list(args.infer_imgs)

    # loop for infer
    with open(
            os.path.join(args.output_dir, "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        for idx, img_path in enumerate(infer_imgs):
            save_img_path = os.path.join(args.output_dir,
                                         os.path.basename(img_path))
            print("process: [{}/{}], save result to {}".format(
                idx, len(infer_imgs), save_img_path))

            img = cv2.imread(img_path)

            ocr_info = ocr_results[os.path.basename(img_path)]["ocr_info"]
            inputs = preprocess(
                tokenizer=tokenizer,
                ori_img=img,
                ocr_info=ocr_info,
                max_seq_len=args.max_seq_length)
            if args.ser_model_type == 'LayoutLM':
                preds = model(
                    input_ids=inputs["input_ids"],
                    bbox=inputs["bbox"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"])
            elif args.ser_model_type == 'LayoutXLM':
                preds = model(
                    input_ids=inputs["input_ids"],
                    bbox=inputs["bbox"],
                    image=inputs["image"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"])
                preds = preds[0]

            preds = postprocess(inputs["attention_mask"], preds,
                                args.label_map_path)
            ocr_info = merge_preds_list_with_ocr_info(
                args.label_map_path, ocr_info, inputs["segment_offset_id"],
                preds)

            fout.write(img_path + "\t" + json.dumps(
                {
                    "ocr_info": ocr_info,
                }, ensure_ascii=False) + "\n")

            img_res = draw_ser_results(img, ocr_info)
            cv2.imwrite(save_img_path, img_res)

    return


if __name__ == "__main__":
    args = parse_args()
    infer(args)
