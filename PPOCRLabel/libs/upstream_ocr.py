# -*- coding: utf-8 -*-
#
# 自定义上游ocr识别
# Author: caiyingyao
# Email: cyy0523xc@gmail.com
# Created Time: 2021-07-07
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from typing import List, Tuple, Union


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
    
    def ocr(self, img: Union[str, np.ndarray], **kwargs) -> List[Tuple[List[Tuple[int, int]], Tuple[str, float]]]:
        """图像ocr识别
        Args:
            img: opencv图像
            kwargs: 只是为了兼容接口，暂时没有用处
        Returns:
            data: 结构保持和paddleocr的识别结果一致
        """
        if type(img) == str:
            # print(img)
            # img = cv2.imread(img)   # 在windows上中文路径会报错
            img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        body = {
            'image': cv2_base64(img),
        }
        data = requests.post(self.url, json=body).json()
        data = [(fmt_box(item['box']), (item['text'], item['score'])) 
                for item in data]
        return data


def cv2_base64(img: np.ndarray, format='JPEG'):
    """将cv2格式的图像转换为base64格式
    Args:
        img numpy.ndarray cv2图像
        format str 转化后的图像格式
    Returns:
        base64字符串
    """
    out_img = Image.fromarray(img)
    output_buffer = BytesIO()
    out_img.save(output_buffer, format=format)
    binary_data = output_buffer.getvalue()
    return str(base64.b64encode(binary_data), encoding='utf8')


def fmt_box(box: List[int]) -> List[Tuple[int, int]]:
    """格式化box坐标
    Args:
        box: [x1, y1, x2, y2, ..., x4, y4]
    """
    return [(box[i*2], box[i*2+1]) for i in range(4)]
