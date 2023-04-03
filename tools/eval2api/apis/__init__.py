# -*- coding: utf-8 -*-
# @Time : 2023/3/29 15:01
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : __init__.py
# @Software : PyCharm
# @Dscription: 调用接口对测试数据集进行预测

import os
import json
from tools.eval2api.apis.peredoc import get_ocr_result_peredoc

# OCR接口厂商字典
api_dict = {
    "peredoc": get_ocr_result_peredoc,
}


def get_pred_result(dataset, language, api, save=True):
    """
    调用接口对测试数据集进行预测
    Args:
        dataset: 测试数据集路劲
        language: 语言
        api: 应用接口厂商

    Returns:
        识别接口对测试集的预测结果
    """
    # 获取识别方法
    ocr_api = api_dict[api]
    # 调用接口识别
    data = ocr_api(dataset, language)
    # 保存预测结果
    if save is True:
        with open(os.path.join(dataset, f"label_{language}_{api}.txt"), 'w', encoding="utf8") as f:
            for image_path in data["pred_label"]:
                label = json.dumps(data["pred_label"][image_path], ensure_ascii=False)
                f.write(f"{image_path}\t{label}\n")
    # 输出识别失败的图片列表
    print("识别失败的图片列表：", data["error_image"])
    return data["pred_label"]


if __name__ == '__main__':
    dataset = r"C:\Users\lvjia\Pictures\dataset\test\2024\image\zh_view"
    get_pred_result(dataset, "ch", "peredoc")
