# -*- coding: utf-8 -*-
# @Time : 2023/3/29 15:01
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : peredoc
# @Software : PyCharm
# @Dscription:

import os
from tqdm import tqdm
import requests


def get_ocr_result_peredoc(dataset_path, language):
    """
    调用青燕ocr接口,对测试集进行预测
    Args:
        dataset_path: 测试集路劲
        language: 语言

    Returns:
        ocr接口返回的预测结果列表
    """
    pred_label = {}
    error_image = []
    # 图片所在文件夹
    image_dir = os.path.join(dataset_path, "images")
    basename = os.path.basename(dataset_path)
    # 对每张图片调用接口进行测试
    for img in tqdm(os.listdir(image_dir)):
        item = {}
        img_path = os.path.join(image_dir, img)
        request_file = {'file': open(img_path, 'rb')}
        for i in range(5):
            try:
                r = requests.post(
                    f'https://r.geblab.com/api/ocr/ocr-doc?lang={language}&token=[token值]',
                    files=request_file, timeout=15)
                break
            except Exception as err:
                print(err)
        if i == 4:
            print(f"图片：{img} 识别失败")
            continue
        # 结果格式转换
        data = eval(r.content.decode('utf-8').replace('"success":true', '"success":True').replace('"success":false',
                                                                                                  '"success":False').replace(
            '":null', '":None'))
        # 结果转换为统一输出格式
        ocr_result = []
        try:
            for lines in data['data'][0]['elements']:
                for line_ in lines['textLines']:
                    for line in line_:
                        block = {}
                        points_list = [int(number) for number in line['coords']]
                        points = [
                            [points_list[0], points_list[1]],
                            [points_list[2], points_list[3]],
                            [points_list[4], points_list[5]],
                            [points_list[6], points_list[7]],
                        ]
                        block['points'] = points
                        block['transcription'] = line['content']
                        ocr_result.append(block)
        except TypeError as err:
            error_image.append(img)
            print(f"图片：{img} 识别出现错误{str(err)} data内容：{str(data)}")
        pred_label[f"{basename}/images/{img}"] = ocr_result
    return {"pred_label": pred_label, "error_image": error_image}


if __name__ == '__main__':
    path = r"C:\Users\lvjia\Pictures\dataset\test\2024\image\zh_view"
    get_ocr_result_peredoc(path, "ch")
