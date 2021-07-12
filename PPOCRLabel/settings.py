# -*- coding: utf-8 -*-
#
# 配置文件
# Author: caiyingyao
# Email: cyy0523xc@gmail.com
# Created Time: 2021-07-07

# 界面默认语言
default_lang = 'ch'

# 是否使用默认的ocr识别
# 默认使用paddleocr，需要按照paddleocr及相关工具
# 如果该值为False，则可以定义上游的OCR识别接口
use_default_ocr = True

"""
上游接口的返回结果格式要求:
[
    {
        'box': [x1, y1, x2, y2, x3, y3, x4, y4],     # box的4个顶点坐标
        'text': '识别到的文本',
        'score': 0.79
    }
]
"""
# ocr识别的http接口地址
# 当use_defalut_ocr为False时，该参数才有效
ocr_reg_url = 'http://192.168.1.245:20923/image/text/simple'
