# -*- coding: utf-8 -*-
# @Time : 2023/3/29 16:22
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : metrics
# @Software : PyCharm
# @Dscription: 端到端OCR评价指标

from rapidfuzz.distance import Levenshtein


class E2EMetric(object):
    """
    端到端OCR评价指标
    """

    def __init__(self, truth_labels, predict_labels):
        self.truth_labels = truth_labels
        self.predict_labels = predict_labels
        self.eps = 1e-10

    @staticmethod
    def __get_iou(box1, box2):
        score_list = []
        try:
            for i in range(2):
                a = box1[0][i], box1[2][i]
                b = box2[0][i], box2[2][i]
                score = max((min(a[1], b[1]) - max(a[0], b[0])), 0) / (min((a[1] - a[0]), (b[1] - b[0])) + 0.0000001)
                score_list.append(score)
        except IndexError:
            return 0
        return score_list[0] * score_list[1]

    def __get_score(self, target, pred):
        iou_score = self.__get_iou(target['points'], pred['points'])
        if iou_score > 0.5:
            norm_edit_dis = Levenshtein.normalized_distance(pred['transcription'], target['transcription'])
            if pred['transcription'] == target['transcription']:
                correct_num = 1
            else:
                correct_num = 0
        else:
            norm_edit_dis = 1.0
            correct_num = 0
        return norm_edit_dis, correct_num

    def __call__(self, *args, **kwargs):
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        imgs_metrics = []
        # 遍历每张真实标注图片结果
        for img in self.truth_labels:
            # 记录当前指标数值
            metrics_cur = {}
            correct_num_cur = correct_num
            all_num_cur = all_num
            norm_edit_dis_cur = norm_edit_dis
            # 遍历图片中每个标注框
            for target in self.truth_labels[img]:
                norm_edit_dis_list = [1.0]
                correct_num_list = [0]
                # 遍历每个预测框
                # 同时计算标注框与预测框的correct_num和norm_edit_dis
                for pred in self.predict_labels[img]:
                    norm_edit_dis_, correct_num_ = self.__get_score(target, pred)
                    norm_edit_dis_list.append(norm_edit_dis_)
                    correct_num_list.append(correct_num_)
                # 获取最大分值当作分值
                norm_edit_dis += min(norm_edit_dis_list)
                correct_num += max(correct_num_list)
                all_num += 1
            # 计算每张图的准确率
            metrics_cur["image"] = img
            metrics_cur["acc"] = (correct_num - correct_num_cur)/ ((all_num - all_num_cur) + self.eps)
            metrics_cur["norm_edit_dis"] = 1 - (norm_edit_dis-norm_edit_dis_cur) / ((all_num - all_num_cur) + self.eps)
            imgs_metrics.append(metrics_cur)
        # 计算指标
        acc = correct_num / (all_num + self.eps)
        norm_edit_dis = 1 - norm_edit_dis / (all_num + self.eps)
        return acc, norm_edit_dis, imgs_metrics


if __name__ == '__main__':
    print('Hello World')
