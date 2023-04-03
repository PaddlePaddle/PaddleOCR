# -*- coding: utf-8 -*-
# @Time : 2023/3/30 10:11
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : filter_dataset
# @Software : PyCharm
# @Dscription: 根据标注、模型预测结果对数据集进行挑选效果较好图片组成新数据集

import os
import json
import shutil


def filter(dataset_path, gt_label, imgs_metrics, output_dir, radio=0.97):
    """
    根据标注、模型预测结果对数据集进行挑选效果较好图片组成新数据集
    Args:
        dataset_path: 测试数据集路劲
        gt_label: 真实标签
        imgs_metrics: 每张图片的精度指标
        output_dir: 新数据集存储文件夹
        radio: 过滤的准确率阈值

    Returns:

    """
    count = 0
    # 创建所需文件夹
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, "images")):
        os.mkdir(os.path.join(output_dir, "images"))
    # 过滤数据集
    with open(os.path.join(output_dir, "label.txt"), 'w', encoding="utf8") as f:
        for item in imgs_metrics:
            image = item["image"]
            norm_edit_dis = item["norm_edit_dis"]
            if norm_edit_dis > radio:
                # 复制图片
                origin_image_path = os.path.join(os.path.split(dataset_path)[0], image)
                output_image_path = os.path.join(os.path.split(output_dir)[0], image)
                shutil.copy(origin_image_path, output_image_path)
                # 写入标签
                label = json.dumps(gt_label[image], ensure_ascii=False)
                f.write(f"{image}\t{label}\n")
                count += 1
    print("挑选合格图片数量: ", count)


