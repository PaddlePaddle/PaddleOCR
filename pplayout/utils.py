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
import argparse
import cv2
import random
import numpy as np
import imghdr
from copy import deepcopy

import paddle

from PIL import Image, ImageDraw, ImageFont

from paddleocr import PaddleOCR


def get_bio_label_maps(label_map_path):
    with open(label_map_path, "r") as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    if "O" not in lines:
        lines.insert(0, "O")
    labels = []
    for line in lines:
        if line == "O":
            labels.append("O")
        else:
            labels.append("B-" + line)
            labels.append("I-" + line)
    label2id_map = {label: idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def draw_ser_results(image,
                     ocr_results,
                     font_path="../doc/fonts/simfang.ttf",
                     font_size=18):
    np.random.seed(0)
    color = (np.random.permutation(range(255)),
             np.random.permutation(range(255)),
             np.random.permutation(range(255)))
    color_map = {
        idx: (color[0][idx], color[1][idx], color[2][idx])
        for idx in range(1, 255)
    }
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    for ocr_info in ocr_results:
        if ocr_info["pred_id"] not in color_map:
            continue
        color = color_map[ocr_info["pred_id"]]

        # draw ocr results outline
        bbox = ocr_info["bbox"]
        bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
        draw.rectangle(bbox, fill=color)

        # draw ocr results
        text = "{}: {}".format(ocr_info["pred"], ocr_info["text"])
        start_y = max(0, bbox[0][1] - font_size)
        tw = font.getsize(text)[0]
        draw.rectangle(
            [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1,
                                         start_y + font_size)],
            fill=(0, 0, 255))
        draw.text(
            (bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


def build_ocr_engine(rec_model_dir, det_model_dir):
    ocr_engine = PaddleOCR(
        rec_model_dir=rec_model_dir,
        det_model_dir=det_model_dir,
        use_angle_cls=False)
    return ocr_engine


# pad sentences
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


def postprocess(attention_mask, preds, id2label_map):
    if isinstance(preds, paddle.Tensor):
        preds = preds.numpy()
    preds = np.argmax(preds, axis=2)

    preds_list = [[] for _ in range(preds.shape[0])]

    # keep batch info
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if attention_mask[i][j] == 1:
                preds_list[i].append(id2label_map[preds[i][j]])

    return preds_list


def merge_preds_list_with_ocr_info(ocr_info, segment_offset_id, preds_list,
                                   label2id_map_for_draw):
    # must ensure the preds_list is generated from the same image
    preds = [p for pred in preds_list for p in pred]

    id2label_map = dict()
    for key in label2id_map_for_draw:
        val = label2id_map_for_draw[key]
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
        curr_pred = [label2id_map_for_draw[p] for p in curr_pred]

        if len(curr_pred) <= 0:
            pred_id = 0
        else:
            counts = np.bincount(curr_pred)
            pred_id = np.argmax(counts)
        ocr_info[idx]["pred_id"] = int(pred_id)
        ocr_info[idx]["pred"] = id2label_map[int(pred_id)]
    return ocr_info


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    # yapf: disable
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,)
    parser.add_argument("--train_data_dir", default=None, type=str, required=False,)
    parser.add_argument("--train_label_path", default=None, type=str, required=False,)
    parser.add_argument("--eval_data_dir", default=None, type=str, required=False,)
    parser.add_argument("--eval_label_path", default=None, type=str, required=False,)
    parser.add_argument("--output_dir", default=None, type=str, required=True,)
    parser.add_argument("--max_seq_length", default=512, type=int,)
    parser.add_argument("--evaluate_during_training", action="store_true",)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for eval.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.",)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.",)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.",)
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.",)
    parser.add_argument("--eval_steps", type=int, default=10, help="eval every X updates steps.",)
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.",)
    parser.add_argument("--seed", type=int, default=2048, help="random seed for initialization",)

    parser.add_argument("--ocr_rec_model_dir", default=None, type=str, )
    parser.add_argument("--ocr_det_model_dir", default=None, type=str, )
    parser.add_argument("--label_map_path", default="./labels/labels_ser.txt", type=str, required=False, )
    parser.add_argument("--infer_imgs", default=None, type=str, required=False)
    parser.add_argument("--ocr_json_path", default=None, type=str, required=False, help="ocr prediction results")
    # yapf: enable
    args = parser.parse_args()
    return args
