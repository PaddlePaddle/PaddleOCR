# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
conver table label to html
"""

import json
import argparse
from tqdm import tqdm


def save_pred_txt(key, val, tmp_file_path):
    with open(tmp_file_path, 'a+', encoding='utf-8') as f:
        f.write('{}\t{}\n'.format(key, val))


def skip_char(text, sp_char_list):
    """
    skip empty cell
    @param text: text in cell
    @param sp_char_list: style char and special code
    @return:
    """
    for sp_char in sp_char_list:
        text = text.replace(sp_char, '')
    return text


def gen_html(img):
    ''' 
    Formats HTML code from tokenized annotation of img
    '''
    html_code = img['html']['structure']['tokens'].copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], img['html']['cells'][::-1]):
        if cell['tokens']:
            text = ''.join(cell['tokens'])
            # skip empty text
            sp_char_list = ['<b>', '</b>', '\u2028', ' ', '<i>', '</i>']
            text_remove_style = skip_char(text, sp_char_list)
            if len(text_remove_style) == 0:
                continue
            html_code.insert(i + 1, text)
    html_code = ''.join(html_code)
    html_code = '<html><body><table>{}</table></body></html>'.format(html_code)
    return html_code


def load_gt_data(gt_path):
    """
    load gt
    @param gt_path:
    @return:
    """
    data_list = {}
    with open(gt_path, 'rb') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data_line = line.decode('utf-8').strip("\n")
            info = json.loads(data_line)
            data_list[info['filename']] = info
    return data_list


def convert(origin_gt_path, save_path):
    """
    gen html from label file
    @param origin_gt_path:
    @param save_path:
    @return:
    """
    data_dict = load_gt_data(origin_gt_path)
    for img_name, gt in tqdm(data_dict.items()):
        html = gen_html(gt)
        save_pred_txt(img_name, html, save_path)
    print('conver finish')


def parse_args():
    parser = argparse.ArgumentParser(description="args for paddleserving")
    parser.add_argument(
        "--ori_gt_path", type=str, required=True, help="label gt path")
    parser.add_argument(
        "--save_path", type=str, required=True, help="path to save file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    convert(args.ori_gt_path, args.save_path)
