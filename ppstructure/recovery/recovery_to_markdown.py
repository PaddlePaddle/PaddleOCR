# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import re

from ppocr.utils.logging import get_logger

logger = get_logger()


def check_merge_method(in_region):
    """Select the function to merge paragraph.

    Determine the paragraph merging method based on the positional
    relationship between the text bbox and the first line of text in the text bbox.

    Args:
        in_region: Elements with text type in the layout result.

    Returns:
        Merge the functions of paragraph, convert_text_space_head or convert_text_space_tail.
    """
    text_bbox = in_region["bbox"]
    text_x1 = text_bbox[0]
    frist_line_box = in_region["res"][0]["text_region"]
    point_1 = frist_line_box[0]
    point_2 = frist_line_box[2]
    frist_line_x1 = point_1[0]
    frist_line_height = abs(point_2[1] - point_1[1])
    x1_distance = frist_line_x1 - text_x1
    return (
        convert_text_space_head
        if x1_distance > frist_line_height
        else convert_text_space_tail
    )


def convert_text_space_head(in_region):
    """The function to merge paragraph.

    The sign of dividing paragraph is that there are two spaces at the beginning.

    Args:
        in_region: Elements with text type in the layout result.

    Returns:
        The text content of the current text box.
    """
    text = ""
    pre_x = None
    frist_line = True
    for i, res in enumerate(in_region["res"]):
        point1 = res["text_region"][0]
        point2 = res["text_region"][2]
        h = point2[1] - point1[1]

        if i == 0:
            text += res["text"]
            pre_x = point1[0]
            continue

        x1 = point1[0]
        if frist_line:
            if abs(pre_x - x1) < h:
                text += "\n\n"
                text += res["text"]
                frist_line = True
            else:
                text += res["text"]
                frist_line = False
        else:
            same_paragh = abs(pre_x - x1) < h
            if same_paragh:
                text += res["text"]
                frist_line = False
            else:
                text += "\n\n"
                text += res["text"]
                frist_line = True
        pre_x = x1
    return text


def convert_text_space_tail(in_region):
    """The function to merge paragraph.

    The symbol for dividing paragraph is a space at the end.

    Args:
        in_region: Elements with text type in the layout result.

    Returns:
        The text content of the current text box.
    """
    text = ""
    frist_line = True
    text_bbox = in_region["bbox"]
    width = text_bbox[2] - text_bbox[0]
    for i, res in enumerate(in_region["res"]):
        point1 = res["text_region"][0]
        point2 = res["text_region"][2]
        row_width = point2[0] - point1[0]
        row_height = point2[1] - point1[1]
        full_row_threshold = width - row_height
        is_full = row_width >= full_row_threshold

        if frist_line:
            text += "\n\n"
            text += res["text"]
        else:
            text += res["text"]

        frist_line = not is_full
    return text


def convert_info_markdown(res, save_folder, img_name):
    """Save the recognition result as a markdown file.

    Args:
        res: Recognition result
        save_folder: Folder to save the markdown file
        img_name: PDF file or image file name

    Returns:
        None
    """

    def replace_special_char(content):
        special_chars = ["*", "`", "~", "$"]
        for char in special_chars:
            content = content.replace(char, "\\" + char)
        return content

    markdown_string = []

    for i, region in enumerate(res):
        if not region["res"] and region["type"].lower() != "figure":
            continue
        img_idx = region["img_idx"]

        if region["type"].lower() == "figure":
            img_file_name = "{}_{}.jpg".format(region["bbox"], img_idx)
            markdown_string.append(
                f"""<div align="center">\n\t<img src="{img_name+"/"+img_file_name}">\n</div>"""
            )
        elif region["type"].lower() == "title":
            markdown_string.append(
                f"""# {region['res'][0]['text']}"""
                + "".join(
                    [" " + one_region["text"] for one_region in region["res"][1:]]
                )
            )
        elif region["type"].lower() == "table":
            markdown_string.append(region["res"]["html"])
        elif region["type"].lower() == "header" or region["type"].lower() == "footer":
            pass
        elif region["type"].lower() == "equation" and "latex" in region["res"]:
            markdown_string.append(f"""$${region["res"]["latex"]}$$""")
        elif region["type"].lower() == "text":
            merge_func = check_merge_method(region)
            # logger.warning(f"use merge method:{merge_func.__name__}")
            markdown_string.append(replace_special_char(merge_func(region)))
        else:
            string = ""
            for line in region["res"]:
                string += line["text"] + " "
            markdown_string.append(string)

    md_path = os.path.join(save_folder, "{}_ocr.md".format(img_name))
    markdown_string = "\n\n".join(markdown_string)
    markdown_string = re.sub(r"\n{3,}", "\n\n", markdown_string)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_string)
    logger.info("markdown save to {}".format(md_path))
