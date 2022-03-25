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


def draw_debug_img(html_path):

    err_cnt = 0
    with open(html_path, 'w') as html:
        html.write('<html>\n<body>\n')
        html.write('<table border="1">\n')
        html.write(
            "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />"
        )
        image_list = []
        path = "./det_results/310_gt/"
        for i, filename in enumerate(sorted(os.listdir(path))):
            if filename.endswith("txt"): continue
            print(filename)
            # The image path
            base = "{}/{}".format(path, filename)
            base_2 = "../PaddleOCR/det_results/ch_PPOCRV2_infer/{}".format(
                filename)
            base_3 = "../PaddleOCR/det_results/ch_ppocr_mobile_infer/{}".format(
                filename)

            html.write("<tr>\n")
            html.write(f'<td> {filename}\n GT')
            html.write('<td>GT\n<img src="%s" width=640></td>' % (base))
            html.write('<td>PPOCRV2\n<img src="%s" width=640></td>' % (base_2))
            html.write('<td>ppocr_mobile\n<img src="%s" width=640></td>' %
                       (base_3))

            html.write("</tr>\n")
        html.write('<style>\n')
        html.write('span {\n')
        html.write('    color: red;\n')
        html.write('}\n')
        html.write('</style>\n')
        html.write('</table>\n')
        html.write('</html>\n</body>\n')
    print("ok")
    return


if __name__ == "__main__":

    html_path = "sys_visual_iou_310.html"

    draw_debug_img()
