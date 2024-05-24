# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from ppocr.metrics.det_metric import DetMetric


class TableStructureMetric(object):
    def __init__(self, main_indicator="acc", eps=1e-6, del_thead_tbody=False, **kwargs):
        self.main_indicator = main_indicator
        self.eps = eps
        self.del_thead_tbody = del_thead_tbody
        self.reset()

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        preds, labels = pred_label
        pred_structure_batch_list = preds["structure_batch_list"]
        gt_structure_batch_list = labels["structure_batch_list"]
        correct_num = 0
        all_num = 0
        for (pred, pred_conf), target in zip(
            pred_structure_batch_list, gt_structure_batch_list
        ):
            pred_str = "".join(pred)
            target_str = "".join(target)
            if self.del_thead_tbody:
                pred_str = (
                    pred_str.replace("<thead>", "")
                    .replace("</thead>", "")
                    .replace("<tbody>", "")
                    .replace("</tbody>", "")
                )
                target_str = (
                    target_str.replace("<thead>", "")
                    .replace("</thead>", "")
                    .replace("<tbody>", "")
                    .replace("</tbody>", "")
                )
            if pred_str == target_str:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        self.reset()
        return {"acc": acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.len_acc_num = 0
        self.token_nums = 0
        self.anys_dict = dict()


class TableMetric(object):
    def __init__(
        self,
        main_indicator="acc",
        compute_bbox_metric=False,
        box_format="xyxy",
        del_thead_tbody=False,
        **kwargs,
    ):
        """

        @param sub_metrics: configs of sub_metric
        @param main_matric: main_matric for save best_model
        @param kwargs:
        """
        self.structure_metric = TableStructureMetric(del_thead_tbody=del_thead_tbody)
        self.bbox_metric = DetMetric() if compute_bbox_metric else None
        self.main_indicator = main_indicator
        self.box_format = box_format
        self.reset()

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        self.structure_metric(pred_label)
        if self.bbox_metric is not None:
            self.bbox_metric(*self.prepare_bbox_metric_input(pred_label))

    def prepare_bbox_metric_input(self, pred_label):
        pred_bbox_batch_list = []
        gt_ignore_tags_batch_list = []
        gt_bbox_batch_list = []
        preds, labels = pred_label

        batch_num = len(preds["bbox_batch_list"])
        for batch_idx in range(batch_num):
            # pred
            pred_bbox_list = [
                self.format_box(pred_box)
                for pred_box in preds["bbox_batch_list"][batch_idx]
            ]
            pred_bbox_batch_list.append({"points": pred_bbox_list})

            # gt
            gt_bbox_list = []
            gt_ignore_tags_list = []
            for gt_box in labels["bbox_batch_list"][batch_idx]:
                gt_bbox_list.append(self.format_box(gt_box))
                gt_ignore_tags_list.append(0)
            gt_bbox_batch_list.append(gt_bbox_list)
            gt_ignore_tags_batch_list.append(gt_ignore_tags_list)

        return [
            pred_bbox_batch_list,
            [0, 0, gt_bbox_batch_list, gt_ignore_tags_batch_list],
        ]

    def get_metric(self):
        structure_metric = self.structure_metric.get_metric()
        if self.bbox_metric is None:
            return structure_metric
        bbox_metric = self.bbox_metric.get_metric()
        if self.main_indicator == self.bbox_metric.main_indicator:
            output = bbox_metric
            for sub_key in structure_metric:
                output["structure_metric_{}".format(sub_key)] = structure_metric[
                    sub_key
                ]
        else:
            output = structure_metric
            for sub_key in bbox_metric:
                output["bbox_metric_{}".format(sub_key)] = bbox_metric[sub_key]
        return output

    def reset(self):
        self.structure_metric.reset()
        if self.bbox_metric is not None:
            self.bbox_metric.reset()

    def format_box(self, box):
        if self.box_format == "xyxy":
            x1, y1, x2, y2 = box
            box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif self.box_format == "xywh":
            x, y, w, h = box
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif self.box_format == "xyxyxyxy":
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        return box
