# -*- coding: utf-8 -*-
# @Time : 2023/3/29 14:27
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : get_gt_label
# @Software : PyCharm
# @Dscription: 读取测试数据集，返回标注信息

import os


def get_label(label_path):
    """

    Args:
        label_path: 测试数据集路劲

    Returns:

    """
    # 读取标签文件
    with open(label_path, "r", encoding="utf8") as f:
        lines = f.readlines()
    # 格式转换
    gt_label = {}
    for line in lines:
        image = line.split("\t")[0]
        label = eval(line.split("\t")[1][:-1].replace("null", "None"))
        gt_label[image] = label
    return gt_label


if __name__ == '__main__':
    path = r"C:\Users\lvjia\Pictures\dataset\test\2024\image\zh_view"
    get_label(path)
