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
            for index_, line in tqdm(enumerate(lines)):
                index = index_ + 3000
                down_url, label = line.split("    ")
                label = label[:-1]
                # 替换为外网路径
                down_url = down_url.replace("http://192.168.100.128:4000/", "http://39.98.138.221:8600/")
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
    # items中元素含义：[语言名称，要生成的数据集文件夹名称，崔波给的txt文件名称]
    items = [
        ["乌克兰文", "uk_view", "2023.09.13_15.12.01_[乌克兰文]uk（view）.txt"]
        
    ]
    # 崔波给的txt文件所在文件夹
    path_ = r"C:\Users\lvjia\Nutstore\1\我的坚果云\工作文档\项目内容\OCR\精度测试\测试集\2023"
    # 下载后的数据集所在文件夹路径
    output_path_ = r"C:\Users\lvjia\Pictures\dataset\test\2023\image"
    for item in items:
        print("开始下载：", item[0])
        path = os.path.join(path_, item[2])
        output_path = os.path.join(output_path_, item[1])
        download(path, output_path)
        