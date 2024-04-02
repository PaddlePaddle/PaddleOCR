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

import os
import argparse


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--save_html_path", type=str, default="./default.html")
    parser.add_argument("--width", type=int, default=640)
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def draw_debug_img(args):

    html_path = args.save_html_path

    err_cnt = 0
    with open(html_path, 'w') as html:
        html.write('<html>\n<body>\n')
        html.write('<table border="1">\n')
        html.write(
            "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />"
        )
        image_list = []
        path = args.image_dir
        for i, filename in enumerate(sorted(os.listdir(path))):
            if filename.endswith("txt"): continue
            # The image path
            base = "{}/{}".format(path, filename)
            html.write("<tr>\n")
            html.write(f'<td> {filename}\n GT')
            html.write(f'<td>GT\n<img src="{base}" width={args.width}></td>')

            html.write("</tr>\n")
        html.write('<style>\n')
        html.write('span {\n')
        html.write('    color: red;\n')
        html.write('}\n')
        html.write('</style>\n')
        html.write('</table>\n')
        html.write('</html>\n</body>\n')
    print(f"The html file saved in {html_path}")
    return


if __name__ == "__main__":

    args = parse_args()

    draw_debug_img(args)
