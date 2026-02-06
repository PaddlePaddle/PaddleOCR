# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import ast
import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tools.infer.utility import (
    draw_ocr_box_txt,
    str2bool,
    str2int_tuple,
    init_args as infer_args,
)
import math


def init_args():
    parser = infer_args()

    # params for output
    parser.add_argument("--output", type=str, default="./output")
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default="TableAttn")
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument("--merge_no_span_structure", type=str2bool, default=True)
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        default="../ppocr/utils/dict/table_structure_dict_ch.txt",
    )
    # params for formula recognition
    parser.add_argument("--formula_algorithm", type=str, default="LaTeXOCR")
    parser.add_argument("--formula_model_dir", type=str)
    parser.add_argument(
        "--formula_char_dict_path",
        type=str,
        default="../ppocr/utils/dict/latex_ocr_tokenizer.json",
    )
    parser.add_argument("--formula_batch_num", type=int, default=1)
    # params for layout
    parser.add_argument("--layout_model_dir", type=str)
    parser.add_argument(
        "--layout_dict_path",
        type=str,
        default="../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
    )
    parser.add_argument(
        "--layout_score_threshold", type=float, default=0.5, help="Threshold of score."
    )
    parser.add_argument(
        "--layout_nms_threshold", type=float, default=0.5, help="Threshold of nms."
    )
    # params for kie
    parser.add_argument("--kie_algorithm", type=str, default="LayoutXLM")
    parser.add_argument("--ser_model_dir", type=str)
    parser.add_argument("--re_model_dir", type=str)
    parser.add_argument("--use_visual_backbone", type=str2bool, default=True)
    parser.add_argument(
        "--ser_dict_path", type=str, default="../train_data/XFUND/class_list_xfun.txt"
    )
    # need to be None or tb-yx
    parser.add_argument("--ocr_order_method", type=str, default=None)
    # params for inference
    parser.add_argument(
        "--mode",
        type=str,
        choices=["structure", "kie"],
        default="structure",
        help="structure and kie is supported",
    )
    parser.add_argument(
        "--image_orientation",
        type=bool,
        default=False,
        help="Whether to enable image orientation recognition",
    )
    parser.add_argument(
        "--layout",
        type=str2bool,
        default=True,
        help="Whether to enable layout analysis",
    )
    parser.add_argument(
        "--table",
        type=str2bool,
        default=True,
        help="In the forward, whether the table area uses table recognition",
    )
    parser.add_argument(
        "--formula",
        type=str2bool,
        default=False,
        help="Whether to enable formula recognition",
    )
    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=True,
        help="In the forward, whether the non-table area is recognition by ocr",
    )
    # param for recovery
    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=False,
        help="Whether to enable layout of recovery",
    )
    parser.add_argument(
        "--recovery_to_markdown",
        type=str2bool,
        default=False,
        help="Whether to enable layout of recovery to markdown",
    )
    parser.add_argument(
        "--use_pdf2docx_api",
        type=str2bool,
        default=False,
        help="Whether to use pdf2docx api",
    )
    parser.add_argument(
        "--invert",
        type=str2bool,
        default=False,
        help="Whether to invert image before processing",
    )
    parser.add_argument(
        "--binarize",
        type=str2bool,
        default=False,
        help="Whether to threshold binarize image before processing",
    )
    parser.add_argument(
        "--alphacolor",
        type=str2int_tuple,
        default=(255, 255, 255),
        help="Replacement color for the alpha channel, if the latter is present; R,G,B integers",
    )

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def draw_structure_result(image, result, font_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    boxes, txts, scores = [], [], []

    img_layout = image.copy()
    draw_layout = ImageDraw.Draw(img_layout)
    text_color = (255, 255, 255)
    text_background_color = (80, 127, 255)
    catid2color = {}
    font_size = 15
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    for region in result:
        if region["type"] not in catid2color:
            box_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            catid2color[region["type"]] = box_color
        else:
            box_color = catid2color[region["type"]]
        box_layout = region["bbox"]
        draw_layout.rectangle(
            [(box_layout[0], box_layout[1]), (box_layout[2], box_layout[3])],
            outline=box_color,
            width=3,
        )

        if int(PIL.__version__.split(".")[0]) < 10:
            text_w, text_h = font.getsize(region["type"])
        else:
            left, top, right, bottom = font.getbbox(region["type"])
            text_w, text_h = right - left, bottom - top

        draw_layout.rectangle(
            [
                (box_layout[0], box_layout[1]),
                (box_layout[0] + text_w, box_layout[1] + text_h),
            ],
            fill=text_background_color,
        )
        draw_layout.text(
            (box_layout[0], box_layout[1]), region["type"], fill=text_color, font=font
        )

        if region["type"] == "table" or (
            region["type"] == "equation" and "latex" in region["res"]
        ):
            pass
        else:
            for text_result in region["res"]:
                boxes.append(np.array(text_result["text_region"]))
                txts.append(text_result["text"])
                scores.append(text_result["confidence"])

                if "text_word_region" in text_result:
                    for word_region in text_result["text_word_region"]:
                        char_box = word_region
                        box_height = int(
                            math.sqrt(
                                (char_box[0][0] - char_box[3][0]) ** 2
                                + (char_box[0][1] - char_box[3][1]) ** 2
                            )
                        )
                        box_width = int(
                            math.sqrt(
                                (char_box[0][0] - char_box[1][0]) ** 2
                                + (char_box[0][1] - char_box[1][1]) ** 2
                            )
                        )
                        if box_height == 0 or box_width == 0:
                            continue
                        boxes.append(word_region)
                        txts.append("")
                        scores.append(1.0)

    im_show = draw_ocr_box_txt(
        img_layout, boxes, txts, scores, font_path=font_path, drop_score=0
    )
    return im_show


def cal_ocr_word_box(rec_str, box, rec_word_info):
    """Calculate the detection frame for each word based on the results of recognition and detection of ocr"""

    col_num, word_list, word_col_list, state_list = rec_word_info
    box = box.tolist()
    bbox_x_start = box[0][0]
    bbox_x_end = box[1][0]
    bbox_y_start = box[0][1]
    bbox_y_end = box[2][1]

    cell_width = (bbox_x_end - bbox_x_start) / col_num

    word_box_list = []
    word_box_content_list = []
    cn_width_list = []
    cn_col_list = []
    for word, word_col, state in zip(word_list, word_col_list, state_list):
        if state == "cn":
            if len(word_col) != 1:
                char_seq_length = (word_col[-1] - word_col[0] + 1) * cell_width
                char_width = char_seq_length / (len(word_col) - 1)
                cn_width_list.append(char_width)
            cn_col_list += word_col
            word_box_content_list += word
        else:
            cell_x_start = bbox_x_start + int(word_col[0] * cell_width)
            cell_x_end = bbox_x_start + int((word_col[-1] + 1) * cell_width)
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)
            word_box_content_list.append("".join(word))
    if len(cn_col_list) != 0:
        if len(cn_width_list) != 0:
            avg_char_width = np.mean(cn_width_list)
        else:
            avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_str)
        for center_idx in cn_col_list:
            center_x = (center_idx + 0.5) * cell_width
            cell_x_start = max(int(center_x - avg_char_width / 2), 0) + bbox_x_start
            cell_x_end = (
                min(int(center_x + avg_char_width / 2), bbox_x_end - bbox_x_start)
                + bbox_x_start
            )
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)

    return word_box_content_list, word_box_list
