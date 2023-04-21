# -*- coding: utf-8 -*-
# @Time : 2023/4/21 9:49
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : split_label
# @Software : PyCharm
# @Dscription: 按比例分割标签

import os
import random
import argparse

# 定义命令行解析器对象
parser = argparse.ArgumentParser(description='按比例分割标签')
# 添加命令行参数
parser.add_argument('--label_path', type=str, default="", required=True, help="标签文件路径")
parser.add_argument('--ratio', type=float, default=0.99, required=True, help="训练集比例")
# 从命令行中结构化解析参数
args = parser.parse_args()
# 解析参数
label_path = args.label_path
ratio = args.ratio

# 读取原始文件
with open(label_path, "r", encoding='utf8') as f_ori:
    labels = f_ori.readlines()

# 写入新标签文件
with open(label_path.replace(".txt", "_train.txt"), "w", encoding='utf8') as f_train:
    with open(label_path.replace(".txt", "_test.txt"), "w", encoding='utf8') as f_test:
        for label in labels:
            random_num = random.random()
            if random_num >= ratio:
                file = f_test
            else:
                file = f_train
            # 写入标签
            file.write(label)
print("数据集分割完成！")
