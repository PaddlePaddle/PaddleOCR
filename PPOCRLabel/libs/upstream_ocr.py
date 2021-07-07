# -*- coding: utf-8 -*-
#
# 自定义上游ocr识别
# Author: caiyingyao
# Email: cyy0523xc@gmail.com
# Created Time: 2021-07-07
from typing import List, Tuple
import requests


class UpstreamOcr:
    """上游OCR识别引擎
    上游接口的返回结果要求:
    [
        {
            'box': [x1, y1, x2, y2, x3, y3, x4, y4],     # box的4个顶点坐标
            'text': '识别到的文本',
            'score': 0.79
        }
    ]
    """

    def __init__(self, url: str):
        """
        Args:
            url: 上游ocr识别接口地址
        """
        self.url = url
    
    def ocr(self, base64_img: str) -> List[Tuple[List[int], Tuple[str, float]]]:
        """图像ocr识别
        Args:
            base64_img: base64格式的图像字符串
        Returns:
            data: 结构保持和paddleocr的识别结果一致
        """
        body = {
            'image': base64_img
        }
        data = requests.post(self.url, json=body).json()
        data = [(item['box'], (item['text'], item['score'])) for item in data]
        return data
