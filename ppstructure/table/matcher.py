# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
from ppstructure.table.table_master_match import deal_eb_token, deal_bb
import html


def distance(box_1, box_2):
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


class TableMatch:
    def __init__(self, filter_ocr_result=False, use_master=False):
        self.filter_ocr_result = filter_ocr_result
        self.use_master = use_master

    def __call__(self, structure_res, dt_boxes, rec_res):
        pred_structures, pred_bboxes = structure_res
        if self.filter_ocr_result:
            dt_boxes, rec_res = self._filter_ocr_result(pred_bboxes, dt_boxes, rec_res)
        matched_index = self.match_result(dt_boxes, pred_bboxes)
        if self.use_master:
            pred_html, pred = self.get_pred_html_master(
                pred_structures, matched_index, rec_res
            )
        else:
            pred_html, pred = self.get_pred_html(
                pred_structures, matched_index, rec_res
            )
        return pred_html

    def match_result(self, dt_boxes, pred_bboxes):
        matched = {}
        for i, gt_box in enumerate(dt_boxes):
            distances = []
            for j, pred_box in enumerate(pred_bboxes):
                if len(pred_box) == 8:
                    pred_box = [
                        np.min(pred_box[0::2]),
                        np.min(pred_box[1::2]),
                        np.max(pred_box[0::2]),
                        np.max(pred_box[1::2]),
                    ]
                distances.append(
                    (distance(gt_box, pred_box), 1.0 - compute_iou(gt_box, pred_box))
                )  # compute iou and l1 distance
            sorted_distances = distances.copy()
            # select det box by iou and l1 distance
            sorted_distances = sorted(
                sorted_distances, key=lambda item: (item[1], item[0])
            )
            if distances.index(sorted_distances[0]) not in matched.keys():
                matched[distances.index(sorted_distances[0])] = [i]
            else:
                matched[distances.index(sorted_distances[0])].append(i)
        return matched

    def get_pred_html(self, pred_structures, matched_index, ocr_contents):
        end_html = []
        td_index = 0
        for tag in pred_structures:
            if "</td>" in tag:
                if "<td></td>" == tag:
                    end_html.extend("<td>")
                if td_index in matched_index.keys():
                    b_with = False
                    if (
                        "<b>" in ocr_contents[matched_index[td_index][0]]
                        and len(matched_index[td_index]) > 1
                    ):
                        b_with = True
                        end_html.extend("<b>")
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == " ":
                                content = content[1:]
                            if "<b>" in content:
                                content = content[3:]
                            if "</b>" in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if (
                                i != len(matched_index[td_index]) - 1
                                and " " != content[-1]
                            ):
                                content += " "
                        # escape content
                        content = html.escape(content)
                        end_html.extend(content)
                    if b_with:
                        end_html.extend("</b>")
                if "<td></td>" == tag:
                    end_html.append("</td>")
                else:
                    end_html.append(tag)
                td_index += 1
            else:
                end_html.append(tag)
        return "".join(end_html), end_html

    def get_pred_html_master(self, pred_structures, matched_index, ocr_contents):
        end_html = []
        td_index = 0
        for token in pred_structures:
            if "</td>" in token:
                txt = ""
                b_with = False
                if td_index in matched_index.keys():
                    if (
                        "<b>" in ocr_contents[matched_index[td_index][0]]
                        and len(matched_index[td_index]) > 1
                    ):
                        b_with = True
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == " ":
                                content = content[1:]
                            if "<b>" in content:
                                content = content[3:]
                            if "</b>" in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if (
                                i != len(matched_index[td_index]) - 1
                                and " " != content[-1]
                            ):
                                content += " "
                        txt += content
                if b_with:
                    txt = "<b>{}</b>".format(txt)
                if "<td></td>" == token:
                    token = "<td>{}</td>".format(txt)
                else:
                    token = "{}</td>".format(txt)
                td_index += 1
            token = deal_eb_token(token)
            end_html.append(token)
        html = "".join(end_html)
        html = deal_bb(html)
        return html, end_html

    def _filter_ocr_result(self, pred_bboxes, dt_boxes, rec_res):
        y1 = pred_bboxes[:, 1::2].min()
        new_dt_boxes = []
        new_rec_res = []

        for box, rec in zip(dt_boxes, rec_res):
            if np.max(box[1::2]) < y1:
                continue
            new_dt_boxes.append(box)
            new_rec_res.append(rec)
        return new_dt_boxes, new_rec_res
