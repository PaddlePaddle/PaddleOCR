import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import paddle

from paddlenlp.transformers import LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForRelationExtraction

from xfun import XFUNDataset
from utils import parse_args, get_bio_label_maps, draw_re_results
from data_collator import DataCollator

from ppocr.utils.logging import get_logger


def infer(args):
    os.makedirs(args.output_dir, exist_ok=True)
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
        num_workers=8,
        shuffle=False,
        collate_fn=DataCollator())

    # 读取gt的oct数据
    ocr_info_list = load_ocr(args.eval_data_dir, args.eval_label_path)

    for idx, batch in enumerate(eval_dataloader):
        ocr_info = ocr_info_list[idx]
        image_path = ocr_info['image_path']
        ocr_info = ocr_info['ocr_info']

        save_img_path = os.path.join(
            args.output_dir,
            os.path.splitext(os.path.basename(image_path))[0] + "_re.jpg")
        logger.info("[Infer] process: {}/{}, save_result to {}".format(
            idx, len(eval_dataloader), save_img_path))
        with paddle.no_grad():
            outputs = model(**batch)
        pred_relations = outputs['pred_relations']

        # 根据entity里的信息，做token解码后去过滤不要的ocr_info
        ocr_info = filter_bg_by_txt(ocr_info, batch, tokenizer)

        # 进行 relations 到 ocr信息的转换
        result = []
        used_tail_id = []
        for relations in pred_relations:
            for relation in relations:
                if relation['tail_id'] in used_tail_id:
                    continue
                if relation['head_id'] not in ocr_info or relation[
                        'tail_id'] not in ocr_info:
                    continue
                used_tail_id.append(relation['tail_id'])
                ocr_info_head = ocr_info[relation['head_id']]
                ocr_info_tail = ocr_info[relation['tail_id']]
                result.append((ocr_info_head, ocr_info_tail))

        img = cv2.imread(image_path)
        img_show = draw_re_results(img, result)
        cv2.imwrite(save_img_path, img_show)


def load_ocr(img_folder, json_path):
    import json
    d = []
    with open(json_path, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            image_name, info_str = line.split("\t")
            info_dict = json.loads(info_str)
            info_dict['image_path'] = os.path.join(img_folder, image_name)
            d.append(info_dict)
    return d


def filter_bg_by_txt(ocr_info, batch, tokenizer):
    entities = batch['entities'][0]
    input_ids = batch['input_ids'][0]

    new_info_dict = {}
    for i in range(len(entities['start'])):
        entitie_head = entities['start'][i]
        entitie_tail = entities['end'][i]
        word_input_ids = input_ids[entitie_head:entitie_tail].numpy().tolist()
        txt = tokenizer.convert_ids_to_tokens(word_input_ids)
        txt = tokenizer.convert_tokens_to_string(txt)

        for i, info in enumerate(ocr_info):
            if info['text'] == txt:
                new_info_dict[i] = info
    return new_info_dict


def post_process(pred_relations, ocr_info, img):
    result = []
    for relations in pred_relations:
        for relation in relations:
            ocr_info_head = ocr_info[relation['head_id']]
            ocr_info_tail = ocr_info[relation['tail_id']]
            result.append((ocr_info_head, ocr_info_tail))
    return result


def draw_re(result, image_path, output_folder):
    img = cv2.imread(image_path)

    from matplotlib import pyplot as plt
    for ocr_info_head, ocr_info_tail in result:
        cv2.rectangle(
            img,
            tuple(ocr_info_head['bbox'][:2]),
            tuple(ocr_info_head['bbox'][2:]), (255, 0, 0),
            thickness=2)
        cv2.rectangle(
            img,
            tuple(ocr_info_tail['bbox'][:2]),
            tuple(ocr_info_tail['bbox'][2:]), (0, 0, 255),
            thickness=2)
        center_p1 = [(ocr_info_head['bbox'][0] + ocr_info_head['bbox'][2]) // 2,
                     (ocr_info_head['bbox'][1] + ocr_info_head['bbox'][3]) // 2]
        center_p2 = [(ocr_info_tail['bbox'][0] + ocr_info_tail['bbox'][2]) // 2,
                     (ocr_info_tail['bbox'][1] + ocr_info_tail['bbox'][3]) // 2]
        cv2.line(
            img, tuple(center_p1), tuple(center_p2), (0, 255, 0), thickness=2)
    plt.imshow(img)
    plt.savefig(
        os.path.join(output_folder, os.path.basename(image_path)), dpi=600)
    # plt.show()


if __name__ == "__main__":
    args = parse_args()
    infer(args)
