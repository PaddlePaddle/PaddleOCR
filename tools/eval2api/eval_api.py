# -*- coding: utf-8 -*-
# @Time : 2023/3/29 11:51
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : eval_api
# @Software : PyCharm
# @Dscription: 测试ocr接口在指定数据集上的精度

import os
import argparse
from tools.eval2api.get_gt_label import get_label
from tools.eval2api.apis import get_pred_result
from tools.eval2api.metrics import E2EMetric
from tools.eval2api.filter_dataset import filter

# 当前支持的语言及其对应厂商
language_dict = {
    "zh": {"peredoc": "ch", "hanwang": "zh", "xunfei": "ch_en"},  # 汉语
    "hi": {"peredoc": "hi", "xunfei": "hi"},                      # 印地语
    "ar": {"peredoc": "ar", "hanwang": "ar"},                     # 阿拉伯语
    "ug": {"peredoc": "ug", "hanwang": "wei", "xunfei": "ug"},    # 维吾尔语
    "vi": {"peredoc": "vi", "hanwang": "yue", "xunfei": "vi"},    # 越南语
    "ru": {"peredoc": "ru", "hanwang": "ru", "xunfei": "ru"},     # 俄语
    "kk": {"peredoc": "ka", "hanwang": "ha", "xunfei": "kka"},    # 哈萨克语
    "th": {"peredoc": "th", "hanwang": "th", "xunfei": "th"},     # 泰语
    "my": {"peredoc": "bu", "hanwang": "mn"},                     # 缅甸语
    "bo": {"peredoc": "ti", "hanwang": "bo"},                     # 藏语
    "ms": {"peredoc": "en", "hanwang": "ml", "xunfei": "ms"},     # 马来语
    "id": {"peredoc": "en", "hanwang": "yn", "xunfei": "id"},     # 印尼语
}


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def parse_args():
    """
    接收用户配置的入参
    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="测试数据集路径",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        help="所要测试的语言",
    )
    parser.add_argument(
        "-a",
        "--api",
        type=str,
        help="测试的api",
    )
    parser.add_argument(
        "-p",
        "--pred_label",
        type=str2bool,
        default=False,
        help="是否采用预测结果文件，若指定了就不会再调用接口了。",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str2bool,
        default=False,
        help="是否过滤产生新数据集",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="过滤产生的新数据集路径",
    )
    args = parser.parse_args()
    return args


def main():
    """
    执行接口精度测试的主函数
    Returns:

    """
    # 读取用户传参
    args = parse_args()
    # 读取测试数据集标注信息
    gt_path = os.path.join(args.dataset, "label.txt")
    gt_label = get_label(gt_path)
    # 调用ocr的api获取接口预测结果
    predict_file = os.path.join(args.dataset, f"label_{args.language}_{args.api}.txt")
    if args.pred_label is True and os.path.exists(predict_file):
        pred_label = get_label(predict_file)
    else:
        pred_label = get_pred_result(args.dataset, language_dict[args.language][args.api], args.api)
    # 精度统计
    acc, norm_edit_dis, imgs_metrics = E2EMetric(gt_label, pred_label)()
    # 输出测试精度
    print(f"acc: {round(acc*100, 2)}%")
    print(f"norm_edit_dis: {round(norm_edit_dis*100, 2)}%")
    # 输出每张图的准确率
    with open(os.path.join(args.dataset, "result.csv"), "w", encoding="utf8") as f:
        f.write("image,acc,norm_edit_dis\n")
        for item in imgs_metrics:
            f.write(f"{item['image']},{item['acc']},{item['norm_edit_dis']}\n")
    # 过滤产生新数据集
    if args.filter is True:
        filter(args.dataset, gt_label, imgs_metrics, args.output_dir)


if __name__ == '__main__':
    main()
