# -*- coding: utf-8 -*-
# @Time : 2022/12/6 15:27
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : download_img
# @Software : PyCharm
# @Dscription: 下载peredoc标注图片至本地，同时转换为paddleocr支持格式

import os
import json
from tqdm import tqdm
import urllib.request


def sort_points(points):
    """
    对位置框的四个点坐标进行排序,返回正常左上,右上,右下,左下的排序顺序
    :param points:
    :return:
    """
    middle_index = int(len(points) / 2)
    # 先按第一个维度排序
    points.sort(key=lambda x: x[0])
    # 对前两个元素排序
    first = points[:middle_index]
    first.sort(key=lambda x: x[1])
    # 对后两个元素排序
    last = points[middle_index:]
    last.sort(key=lambda x: x[1])
    new_points = first + last
    if len(new_points) == 4:
        out_points = [new_points[0], new_points[2], new_points[3], new_points[1]]
    elif len(new_points) == 2:
        out_points = [new_points[0], [new_points[1][0], new_points[0][1]], new_points[1],
                      [new_points[0][0], new_points[1][1]]]
    else:
        out_points = []
    return out_points


def download(origin_label_path, dataset_dir_path):
    """
    图片下载方法
    Args:
        origin_label_path: 系统导出的标注原始文件
        dataset_dir_path: 转换完成后数据集保存路径

    Returns:
        转换为PaddleOCR格式的数据集
    """
    # 定义一个字典用于存放标签结果
    results = {}
    # 创建文件夹
    if not os.path.exists(dataset_dir_path):
        os.mkdir(dataset_dir_path)
    if not os.path.exists(os.path.join(dataset_dir_path, "images")):
        os.mkdir(os.path.join(dataset_dir_path, "images"))
    # 初始化一个标签文件用于记录新转换后的标签
    with open(os.path.join(dataset_dir_path, 'label.txt'), 'w', encoding="utf8") as flabel:
        # 读取标注文件
        with open(origin_label_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            # 遍历每条标注信息
            for index, line in tqdm(enumerate(lines)):
                down_url, label = line.split("    ")
                label = label[:-1]
                # 下载并保存图片
                try:
                    data = urllib.request.urlopen(down_url, timeout=10).read()
                except Exception:
                    continue
                    print("图片下载失败")
                with open(os.path.join(dataset_dir_path, "images", str(index) + '.jpg'), 'wb') as fi:
                    fi.write(data)
                # 标签转换为paddleocr格式
                try:
                    data = eval(label.replace("null", "None"))
                except SyntaxError:
                    continue
                label_result = []
                for line in data[0]['Labels']:
                    block = {}
                    points = [[int(point['X']), int(point['Y'])] for point in line['Points']]
                    block['points'] = sort_points(points)
                    block['transcription'] = line['Value']
                    label_result.append(block)
                # 标签
                label_result = json.dumps(label_result, ensure_ascii=False)
                # 图片路径
                image_dir_name = dataset_dir_path.replace('\\', '/').split('/')[-1]
                image_path = os.path.join(image_dir_name, "images", str(index) + '.jpg').replace('\\', '/')
                flabel.write(f"{image_path}\t{label_result}\n")


if __name__ == "__main__":
    items = [
        # ["汉语", "zh_doc", "2023.03.26_21.02.04_[中文]zhongwen(doc001).txt"],
        # ["汉语", "zh_view", "2023.03.30_09.28.06_[中文]zhongwen(view001).txt"],
        # ["哈萨克语", "kk_doc", "2023.03.26_21.05.42_[哈萨克文]hayu(doc001).txt"],
        # ["哈萨克语", "kk_view", "2023.03.26_21.02.33_[哈萨克文]hayu(view001).txt"],
        # ["泰语", "th_doc", "2023.03.26_21.08.12_[泰文]taiyu(doc001).txt"],
        # ["泰语", "th_view", "2023.03.26_21.02.25_[泰文]taiyu(view001).txt"],
        # ["马来语", "ms_doc", "2023.03.26_20.59.42_[马来文]malaiyu(doc001).txt"],
        ["马来语", "ms_view", "2023.03.31_11.58.49_[马来文]malaiyu(view001).txt"],
        # ["越南语", "vi_doc", "2023.03.26_21.08.43_[越南文]yuenanyu(doc001).txt"],
        # ["越南语", "vi_view", "2023.03.30_09.28.18_[越南文]yuenanyu(view001).txt"],
        # ["印尼语", "id_doc", "2023.03.26_20.59.06_[印尼文]yinniyu(doc001).txt"],
        #["印尼语", "id_view", "2023.03.31_11.30.25_[印尼文]yinniyu(view001).txt"],
        # ["缅甸语", "my_doc", "2023.03.26_21.07.41_[缅甸文]miandianyu(doc001).txt"],
        # ["缅甸语", "my_view", "2023.03.30_15.09.38_[缅甸文]miandianu(view001).txt"],
        # ["维吾尔语", "ug_doc", "2023.03.26_21.04.51_[维吾尔文]weiyu(doc001).txt"],
        # ["维吾尔语", "ug_view", "2023.03.30_20.15.02_[维吾尔文]weiyu（view001）.txt"],
        # ["藏语", "bo_doc", "2023.03.26_21.06.09_[藏文]zangyu(doc001).txt"],
        # ["藏语", "bo_view", "2023.03.28_17.35.18_[藏文]zangyu(view001).txt"],
        #["阿拉伯语", "ar_doc", "2023.03.31_11.38.21_[阿拉伯文]ayu(doc001).txt"],
        # ["阿拉伯语", "ar_view", "2023.03.28_16.36.16_[阿拉伯文]alaboyu(view001).txt"],
        # ["俄语", "ru_doc", "2023.03.26_21.08.32_[俄罗斯文]eyu(doc001).txt"],
        # ["俄语", "ru_view", "2023.03.26_21.02.49_[俄罗斯文]eyu(view001).txt"],
        # ["印地语", "hi_doc", "2023.03.28_14.28.12_[印地文]yindiyu(doc001).txt"],
        # ["印地语", "hi_view", ""],
    ]
    path_ = r"C:\Users\lvjia\Nutstore\1\我的坚果云\工作文档\项目内容\OCR\精度测试\测试集\2023"
    output_path_ = r"C:\Users\lvjia\Pictures\dataset\test\2023\image"
    for item in items:
        print("开始下载：", item[0])
        path = os.path.join(path_, item[2])
        output_path = os.path.join(output_path_, item[1])
        download(path, output_path)
        