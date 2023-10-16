# -*- coding: utf-8 -*-
# @Time : 2022/11/28 11:47
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : crop_rec_img
# @Software : PyCharm
# @Dscription: 裁剪出ocr文本识别所需图片极其对应标签,标签格式以paddleocr格式为准

import os
import cv2
import numpy as np
from tqdm import tqdm
from bidi.algorithm import get_display
from get_gt_label import get_label


class CropImage:
    """
    裁剪出ocr文本识别所需图片极其对应标签,标签格式以paddleocr格式为准
    """
    def __init__(self, input_path, output_path, bidi=False):
        """
        数据集路径
        :param input_path:
        """
        # 初始化原始数据集路径
        self.input_path = input_path
        # 输出文件夹路径
        self.output_path = output_path
        # 获取文件夹名
        self.dirname = self.output_path.replace("\\", "/").split("/")[-1]
        # 是否进行BIDI转换
        self.bidi = bidi

    @staticmethod
    def four_point_transform(img, points):
        """"""
        # 原始四边形的四个角点
        pts1 = np.float32(points)
        # 变换后的矩形的四个角点
        width = max(points[1][0]-points[0][0], points[2][0]-points[3][0])
        height = max(points[3][1]-points[0][1], points[2][1]-points[1][1])
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        # 获取变换矩阵
        H = cv2.getPerspectiveTransform(pts1, pts2)
        # 执行透视变换
        wrapped = cv2.warpPerspective(img, H, (width, height))
        return wrapped

    @staticmethod
    def sort_points(points):
        """
        对位置框的四个点坐标进行排序,返回正常左上,右上,右下,左下的排序顺序
        :param points:
        :return:
        """
        middle_index = int(len(points)/2)
        # 先按第一个维度排序
        points.sort(key=lambda x: x[0])
        # 对前两个元素排序
        first = points[:middle_index]
        first.sort(key=lambda x: x[1])
        # 对后两个元素排序
        last = points[middle_index:]
        last.sort(key=lambda x: x[1])
        new_points = first + last
        out_points = [new_points[0], new_points[2], new_points[3], new_points[1]]
        return out_points

    def __call__(self, *args, **kwargs):
        """"""
        # 图片文件夹路径
        images_path = os.path.join(self.input_path, "images")
        # 标签文件夹路径
        labels_path = os.path.join(self.input_path, "label.txt")
        # 创建输出文件夹
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(os.path.join(self.output_path, "img")):
            os.mkdir(os.path.join(self.output_path, "img"))
        # 读取标签
        labels = get_label(labels_path)
        # 初始化一个切出来的小图的计数器
        index = 0
        with open(os.path.join(self.output_path, "label.txt"), "w", encoding="utf8") as fw:
            # 遍历标注信息
            for img_path in labels:
                # 获取标签信息
                label = labels[img_path]
                img_name = img_path.split("/")[-1]
                img_path = os.path.join(images_path, img_name)
                # 读取图片
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    # 遍历每个文字块
                    for block in label:
                        # 文本内容
                        text = block["transcription"]
                        if self.bidi and text:
                            text = get_display(text)
                        # 纠正坐标点的顺序
                        if len(block["points"])<4:
                            continue
                        points = self.sort_points(block["points"])
                        # 透视变换
                        try:
                            crop_img = self.four_point_transform(img, points)
                        except cv2.error as err:
                            print("发生cv2错误:" + str(err))
                            continue
                        # 图片保存
                        cv2.imwrite("{0}/img/{1}.jpg".format(self.output_path, str(index)), crop_img)
                        # 标签写入文件
                        fw.write("{0}/img/{1}.jpg\t{2}\n".format(self.dirname, str(index), text))
                        index += 1


if __name__ == '__main__':
    items = [
        # ("ar", True),
        # ("bo", False),
        # ("hi", False),
        # ("kk", False),
        # ("ms", False),
        # ("my", False),
        # ("ru", False),
        # ("th", False),
        # ("ug", True),
        # ("vi", False),
        # ("id", False),
        # ("zh", False)
        # ("lo", False)
        # ("km", False)
        # ("ja", False), 
        # ("ko", False), 
        # ("bn", False), 
        # ("en", False), 
        # ("fa", True), 
        # ("uk", False), 
        # ("no", False), 
        # ("tr", False), 
        # ("it", False), 
        # ("pt", False), 
        # ("ro", False), 
        ("pl", False), 
        ("fa", True), 
        ("en", False), 
    ]
    for language, bidi_state in items:
        for type_ in ("view",):
            print("Start process language: ", language, "\n")
            cropper_doc = CropImage(
                r"C:\Users\lvjia\Pictures\dataset\test\2023\image\{0}_{1}".format(language, type_),
                r"C:\Users\lvjia\Pictures\dataset\test\2023\line\{0}_{1}".format(language, type_),
                bidi_state
            )
            cropper_doc()
