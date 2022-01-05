# -*- coding: utf-8 -*-
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
1、安装库 pip install pymupdf
2、安装库 pip install pillow
3、直接运行
"""
import os
import fitz

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from ppocr.utils.logging import get_logger

logger = get_logger()

"""
parameter:
    pdf_file : the pdf file for convert 
    img_dir : the dir to save images

return:    
    pic_list: image list for pdf file
"""


def pdf2img(pdf_file, img_dir):
    doc = fitz.open(pdf_file)
    pdf_name = os.path.splitext(pdf_file)[0]
    pic_list = []
    for pg in range(doc.pageCount):
        page = doc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为2，这将为我们生成分辨率提高四倍的图像。
        zoom_x = 2.0
        zoom_y = 2.0
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        # 注意下边的一行，这是本的重点。原文是生成的PNG，我给改成了JPG
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
            logger.info('%s directory are created' % img_dir)
        pm.pil_save('%s/%s.jpg' % (img_dir, pg), quality=1)
        pic_list.append('%s/%s.jpg' % (img_dir, pg))
    logger.info('%s pdf file are saved under %s successfully' % (pdf_file, img_dir))
    return pic_list
