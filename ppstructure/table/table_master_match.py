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
"""
This code is refer from:
https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/match.py
"""

import os
import re
import cv2
import glob
import copy
import math
import pickle
import numpy as np

from shapely.geometry import Polygon, MultiPoint
"""
Useful function in matching.
"""


def remove_empty_bboxes(bboxes):
    """
    remove [0., 0., 0., 0.] in structure master bboxes.
    len(bboxes.shape) must be 2.
    :param bboxes:
    :return:
    """
    new_bboxes = []
    for bbox in bboxes:
        if sum(bbox) == 0.:
            continue
        new_bboxes.append(bbox)
    return np.array(new_bboxes)


def xywh2xyxy(bboxes):
    if len(bboxes.shape) == 1:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[0] = bboxes[0] - bboxes[2] / 2
        new_bboxes[1] = bboxes[1] - bboxes[3] / 2
        new_bboxes[2] = bboxes[0] + bboxes[2] / 2
        new_bboxes[3] = bboxes[1] + bboxes[3] / 2
        return new_bboxes
    elif len(bboxes.shape) == 2:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
        return new_bboxes
    else:
        raise ValueError


def xyxy2xywh(bboxes):
    if len(bboxes.shape) == 1:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[0] = bboxes[0] + (bboxes[2] - bboxes[0]) / 2
        new_bboxes[1] = bboxes[1] + (bboxes[3] - bboxes[1]) / 2
        new_bboxes[2] = bboxes[2] - bboxes[0]
        new_bboxes[3] = bboxes[3] - bboxes[1]
        return new_bboxes
    elif len(bboxes.shape) == 2:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2
        new_bboxes[:, 1] = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return new_bboxes
    else:
        raise ValueError


def pickle_load(path, prefix='end2end'):
    if os.path.isfile(path):
        data = pickle.load(open(path, 'rb'))
    elif os.path.isdir(path):
        data = dict()
        search_path = os.path.join(path, '{}_*.pkl'.format(prefix))
        pkls = glob.glob(search_path)
        for pkl in pkls:
            this_data = pickle.load(open(pkl, 'rb'))
            data.update(this_data)
    else:
        raise ValueError
    return data


def convert_coord(xyxy):
    """
    Convert two points format to four points format.
    :param xyxy:
    :return:
    """
    new_bbox = np.zeros([4, 2], dtype=np.float32)
    new_bbox[0, 0], new_bbox[0, 1] = xyxy[0], xyxy[1]
    new_bbox[1, 0], new_bbox[1, 1] = xyxy[2], xyxy[1]
    new_bbox[2, 0], new_bbox[2, 1] = xyxy[2], xyxy[3]
    new_bbox[3, 0], new_bbox[3, 1] = xyxy[0], xyxy[3]
    return new_bbox


def cal_iou(bbox1, bbox2):
    bbox1_poly = Polygon(bbox1).convex_hull
    bbox2_poly = Polygon(bbox2).convex_hull
    union_poly = np.concatenate((bbox1, bbox2))

    if not bbox1_poly.intersects(bbox2_poly):
        iou = 0
    else:
        inter_area = bbox1_poly.intersection(bbox2_poly).area
        union_area = MultiPoint(union_poly).convex_hull.area
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area
    return iou


def cal_distance(p1, p2):
    delta_x = p1[0] - p2[0]
    delta_y = p1[1] - p2[1]
    d = math.sqrt((delta_x**2) + (delta_y**2))
    return d


def is_inside(center_point, corner_point):
    """
    Find if center_point inside the bbox(corner_point) or not.
    :param center_point: center point (x, y)
    :param corner_point: corner point ((x1,y1),(x2,y2))
    :return:
    """
    x_flag = False
    y_flag = False
    if (center_point[0] >= corner_point[0][0]) and (
            center_point[0] <= corner_point[1][0]):
        x_flag = True
    if (center_point[1] >= corner_point[0][1]) and (
            center_point[1] <= corner_point[1][1]):
        y_flag = True
    if x_flag and y_flag:
        return True
    else:
        return False


def find_no_match(match_list, all_end2end_nums, type='end2end'):
    """
    Find out no match end2end bbox in previous match list.
    :param match_list: matching pairs.
    :param all_end2end_nums: numbers of end2end_xywh
    :param type: 'end2end' corresponding to idx 0, 'master' corresponding to idx 1.
    :return: no match pse bbox index list
    """
    if type == 'end2end':
        idx = 0
    elif type == 'master':
        idx = 1
    else:
        raise ValueError

    no_match_indexs = []
    # m[0] is end2end index m[1] is master index
    matched_bbox_indexs = [m[idx] for m in match_list]
    for n in range(all_end2end_nums):
        if n not in matched_bbox_indexs:
            no_match_indexs.append(n)
    return no_match_indexs


def is_abs_lower_than_threshold(this_bbox, target_bbox, threshold=3):
    # only consider y axis, for grouping in row.
    delta = abs(this_bbox[1] - target_bbox[1])
    if delta < threshold:
        return True
    else:
        return False


def sort_line_bbox(g, bg):
    """
    Sorted the bbox in the same line(group)
    compare coord 'x' value, where 'y' value is closed in the same group.
    :param g: index in the same group
    :param bg: bbox in the same group
    :return:
    """

    xs = [bg_item[0] for bg_item in bg]
    xs_sorted = sorted(xs)

    g_sorted = [None] * len(xs_sorted)
    bg_sorted = [None] * len(xs_sorted)
    for g_item, bg_item in zip(g, bg):
        idx = xs_sorted.index(bg_item[0])
        bg_sorted[idx] = bg_item
        g_sorted[idx] = g_item

    return g_sorted, bg_sorted


def flatten(sorted_groups, sorted_bbox_groups):
    idxs = []
    bboxes = []
    for group, bbox_group in zip(sorted_groups, sorted_bbox_groups):
        for g, bg in zip(group, bbox_group):
            idxs.append(g)
            bboxes.append(bg)
    return idxs, bboxes


def sort_bbox(end2end_xywh_bboxes, no_match_end2end_indexes):
    """
    This function will group the render end2end bboxes in row.
    :param end2end_xywh_bboxes:
    :param no_match_end2end_indexes:
    :return:
    """
    groups = []
    bbox_groups = []
    for index, end2end_xywh_bbox in zip(no_match_end2end_indexes,
                                        end2end_xywh_bboxes):
        this_bbox = end2end_xywh_bbox
        if len(groups) == 0:
            groups.append([index])
            bbox_groups.append([this_bbox])
        else:
            flag = False
            for g, bg in zip(groups, bbox_groups):
                # this_bbox is belong to bg's row or not
                if is_abs_lower_than_threshold(this_bbox, bg[0]):
                    g.append(index)
                    bg.append(this_bbox)
                    flag = True
                    break
            if not flag:
                # this_bbox is not belong to bg's row, create a row.
                groups.append([index])
                bbox_groups.append([this_bbox])

    # sorted bboxes in a group
    tmp_groups, tmp_bbox_groups = [], []
    for g, bg in zip(groups, bbox_groups):
        g_sorted, bg_sorted = sort_line_bbox(g, bg)
        tmp_groups.append(g_sorted)
        tmp_bbox_groups.append(bg_sorted)

    # sorted groups, sort by coord y's value.
    sorted_groups = [None] * len(tmp_groups)
    sorted_bbox_groups = [None] * len(tmp_bbox_groups)
    ys = [bg[0][1] for bg in tmp_bbox_groups]
    sorted_ys = sorted(ys)
    for g, bg in zip(tmp_groups, tmp_bbox_groups):
        idx = sorted_ys.index(bg[0][1])
        sorted_groups[idx] = g
        sorted_bbox_groups[idx] = bg

    # flatten, get final result
    end2end_sorted_idx_list, end2end_sorted_bbox_list \
        = flatten(sorted_groups, sorted_bbox_groups)

    return end2end_sorted_idx_list, end2end_sorted_bbox_list, sorted_groups, sorted_bbox_groups


def get_bboxes_list(end2end_result, structure_master_result):
    """
    This function is use to convert end2end results and structure master results to
    List of xyxy bbox format and List of xywh bbox format
    :param end2end_result: bbox's format is xyxy
    :param structure_master_result: bbox's format is xywh
    :return: 4 kind list of bbox ()
    """
    # end2end
    end2end_xyxy_list = []
    end2end_xywh_list = []
    for end2end_item in end2end_result:
        src_bbox = end2end_item['bbox']
        end2end_xyxy_list.append(src_bbox)
        xywh_bbox = xyxy2xywh(src_bbox)
        end2end_xywh_list.append(xywh_bbox)
    end2end_xyxy_bboxes = np.array(end2end_xyxy_list)
    end2end_xywh_bboxes = np.array(end2end_xywh_list)

    # structure master
    src_bboxes = structure_master_result['bbox']
    src_bboxes = remove_empty_bboxes(src_bboxes)
    structure_master_xyxy_bboxes = src_bboxes
    xywh_bbox = xyxy2xywh(src_bboxes)
    structure_master_xywh_bboxes = xywh_bbox

    return end2end_xyxy_bboxes, end2end_xywh_bboxes, structure_master_xywh_bboxes, structure_master_xyxy_bboxes


def center_rule_match(end2end_xywh_bboxes, structure_master_xyxy_bboxes):
    """
    Judge end2end Bbox's center point is inside structure master Bbox or not,
    if end2end Bbox's center is in structure master Bbox, get matching pair.
    :param end2end_xywh_bboxes:
    :param structure_master_xyxy_bboxes:
    :return: match pairs list, e.g. [[0,1], [1,2], ...]
    """
    match_pairs_list = []
    for i, end2end_xywh in enumerate(end2end_xywh_bboxes):
        for j, master_xyxy in enumerate(structure_master_xyxy_bboxes):
            x_end2end, y_end2end = end2end_xywh[0], end2end_xywh[1]
            x_master1, y_master1, x_master2, y_master2 \
                = master_xyxy[0], master_xyxy[1], master_xyxy[2], master_xyxy[3]
            center_point_end2end = (x_end2end, y_end2end)
            corner_point_master = ((x_master1, y_master1),
                                   (x_master2, y_master2))
            if is_inside(center_point_end2end, corner_point_master):
                match_pairs_list.append([i, j])
    return match_pairs_list


def iou_rule_match(end2end_xyxy_bboxes, end2end_xyxy_indexes,
                   structure_master_xyxy_bboxes):
    """
    Use iou to find matching list.
    choose max iou value bbox as match pair.
    :param end2end_xyxy_bboxes:
    :param end2end_xyxy_indexes: original end2end indexes.
    :param structure_master_xyxy_bboxes:
    :return: match pairs list, e.g. [[0,1], [1,2], ...]
    """
    match_pair_list = []
    for end2end_xyxy_index, end2end_xyxy in zip(end2end_xyxy_indexes,
                                                end2end_xyxy_bboxes):
        max_iou = 0
        max_match = [None, None]
        for j, master_xyxy in enumerate(structure_master_xyxy_bboxes):
            end2end_4xy = convert_coord(end2end_xyxy)
            master_4xy = convert_coord(master_xyxy)
            iou = cal_iou(end2end_4xy, master_4xy)
            if iou > max_iou:
                max_match[0], max_match[1] = end2end_xyxy_index, j
                max_iou = iou

        if max_match[0] is None:
            # no match
            continue
        match_pair_list.append(max_match)
    return match_pair_list


def distance_rule_match(end2end_indexes, end2end_bboxes, master_indexes,
                        master_bboxes):
    """
    Get matching between no-match end2end bboxes and no-match master bboxes.
    Use min distance to match.
    This rule will only run (no-match end2end nums > 0) and (no-match master nums > 0)
    It will Return master_bboxes_nums match-pairs.
    :param end2end_indexes:
    :param end2end_bboxes:
    :param master_indexes:
    :param master_bboxes:
    :return: match_pairs list, e.g. [[0,1], [1,2], ...]
    """
    min_match_list = []
    for j, master_bbox in zip(master_indexes, master_bboxes):
        min_distance = np.inf
        min_match = [0, 0]  # i, j
        for i, end2end_bbox in zip(end2end_indexes, end2end_bboxes):
            x_end2end, y_end2end = end2end_bbox[0], end2end_bbox[1]
            x_master, y_master = master_bbox[0], master_bbox[1]
            end2end_point = (x_end2end, y_end2end)
            master_point = (x_master, y_master)
            dist = cal_distance(master_point, end2end_point)
            if dist < min_distance:
                min_match[0], min_match[1] = i, j
                min_distance = dist
        min_match_list.append(min_match)
    return min_match_list


def extra_match(no_match_end2end_indexes, master_bbox_nums):
    """
    This function will create some virtual master bboxes,
    and get match with the no match end2end indexes.
    :param no_match_end2end_indexes:
    :param master_bbox_nums:
    :return:
    """
    end_nums = len(no_match_end2end_indexes) + master_bbox_nums
    extra_match_list = []
    for i in range(master_bbox_nums, end_nums):
        end2end_index = no_match_end2end_indexes[i - master_bbox_nums]
        extra_match_list.append([end2end_index, i])
    return extra_match_list


def get_match_dict(match_list):
    """
    Convert match_list to a dict, where key is master bbox's index, value is end2end bbox index.
    :param match_list:
    :return:
    """
    match_dict = dict()
    for match_pair in match_list:
        end2end_index, master_index = match_pair[0], match_pair[1]
        if master_index not in match_dict.keys():
            match_dict[master_index] = [end2end_index]
        else:
            match_dict[master_index].append(end2end_index)
    return match_dict


def deal_successive_space(text):
    """
    deal successive space character for text
    1. Replace ' '*3 with '<space>' which is real space is text
    2. Remove ' ', which is split token, not true space
    3. Replace '<space>' with ' ', to get real text
    :param text:
    :return:
    """
    text = text.replace(' ' * 3, '<space>')
    text = text.replace(' ', '')
    text = text.replace('<space>', ' ')
    return text


def reduce_repeat_bb(text_list, break_token):
    """
    convert ['<b>Local</b>', '<b>government</b>', '<b>unit</b>'] to ['<b>Local government unit</b>']
    PS: maybe style <i>Local</i> is also exist, too. it can be processed like this.
    :param text_list:
    :param break_token:
    :return:
    """
    count = 0
    for text in text_list:
        if text.startswith('<b>'):
            count += 1
    if count == len(text_list):
        new_text_list = []
        for text in text_list:
            text = text.replace('<b>', '').replace('</b>', '')
            new_text_list.append(text)
        return ['<b>' + break_token.join(new_text_list) + '</b>']
    else:
        return text_list


def get_match_text_dict(match_dict, end2end_info, break_token=' '):
    match_text_dict = dict()
    for master_index, end2end_index_list in match_dict.items():
        text_list = [
            end2end_info[end2end_index]['text']
            for end2end_index in end2end_index_list
        ]
        text_list = reduce_repeat_bb(text_list, break_token)
        text = break_token.join(text_list)
        match_text_dict[master_index] = text
    return match_text_dict


def merge_span_token(master_token_list):
    """
    Merge the span style token (row span or col span).
    :param master_token_list:
    :return:
    """
    new_master_token_list = []
    pointer = 0
    if master_token_list[-1] != '</tbody>':
        master_token_list.append('</tbody>')
    while master_token_list[pointer] != '</tbody>':
        try:
            if master_token_list[pointer] == '<td':
                if master_token_list[pointer + 1].startswith(
                        ' colspan=') or master_token_list[
                            pointer + 1].startswith(' rowspan='):
                    """
                    example:
                    pattern <td colspan="3">
                    '<td' + 'colspan=" "' + '>' + '</td>'
                    """
                    tmp = ''.join(master_token_list[pointer:pointer + 3 + 1])
                    pointer += 4
                    new_master_token_list.append(tmp)

                elif master_token_list[pointer + 2].startswith(
                        ' colspan=') or master_token_list[
                            pointer + 2].startswith(' rowspan='):
                    """
                    example:
                    pattern <td rowspan="2" colspan="3">
                    '<td' + 'rowspan=" "' + 'colspan=" "' + '>' + '</td>'
                    """
                    tmp = ''.join(master_token_list[pointer:pointer + 4 + 1])
                    pointer += 5
                    new_master_token_list.append(tmp)

                else:
                    new_master_token_list.append(master_token_list[pointer])
                    pointer += 1
            else:
                new_master_token_list.append(master_token_list[pointer])
                pointer += 1
        except:
            print("Break in merge...")
            break
    new_master_token_list.append('</tbody>')

    return new_master_token_list


def deal_eb_token(master_token):
    """
    post process with <eb></eb>, <eb1></eb1>, ...
    emptyBboxTokenDict = {
        "[]": '<eb></eb>',
        "[' ']": '<eb1></eb1>',
        "['<b>', ' ', '</b>']": '<eb2></eb2>',
        "['\\u2028', '\\u2028']": '<eb3></eb3>',
        "['<sup>', ' ', '</sup>']": '<eb4></eb4>',
        "['<b>', '</b>']": '<eb5></eb5>',
        "['<i>', ' ', '</i>']": '<eb6></eb6>',
        "['<b>', '<i>', '</i>', '</b>']": '<eb7></eb7>',
        "['<b>', '<i>', ' ', '</i>', '</b>']": '<eb8></eb8>',
        "['<i>', '</i>']": '<eb9></eb9>',
        "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']": '<eb10></eb10>',
    }
    :param master_token:
    :return:
    """
    master_token = master_token.replace('<eb></eb>', '<td></td>')
    master_token = master_token.replace('<eb1></eb1>', '<td> </td>')
    master_token = master_token.replace('<eb2></eb2>', '<td><b> </b></td>')
    master_token = master_token.replace('<eb3></eb3>', '<td>\u2028\u2028</td>')
    master_token = master_token.replace('<eb4></eb4>', '<td><sup> </sup></td>')
    master_token = master_token.replace('<eb5></eb5>', '<td><b></b></td>')
    master_token = master_token.replace('<eb6></eb6>', '<td><i> </i></td>')
    master_token = master_token.replace('<eb7></eb7>',
                                        '<td><b><i></i></b></td>')
    master_token = master_token.replace('<eb8></eb8>',
                                        '<td><b><i> </i></b></td>')
    master_token = master_token.replace('<eb9></eb9>', '<td><i></i></td>')
    master_token = master_token.replace('<eb10></eb10>',
                                        '<td><b> \u2028 \u2028 </b></td>')
    return master_token


def insert_text_to_token(master_token_list, match_text_dict):
    """
    Insert OCR text result to structure token.
    :param master_token_list:
    :param match_text_dict:
    :return:
    """
    master_token_list = merge_span_token(master_token_list)
    merged_result_list = []
    text_count = 0
    for master_token in master_token_list:
        if master_token.startswith('<td'):
            if text_count > len(match_text_dict) - 1:
                text_count += 1
                continue
            elif text_count not in match_text_dict.keys():
                text_count += 1
                continue
            else:
                master_token = master_token.replace(
                    '><', '>{}<'.format(match_text_dict[text_count]))
                text_count += 1
        master_token = deal_eb_token(master_token)
        merged_result_list.append(master_token)

    return ''.join(merged_result_list)


def deal_isolate_span(thead_part):
    """
    Deal with isolate span cases in this function.
    It causes by wrong prediction in structure recognition model.
    eg. predict <td rowspan="2"></td> to <td></td> rowspan="2"></b></td>.
    :param thead_part:
    :return:
    """
    # 1. find out isolate span tokens.
    isolate_pattern = "<td></td> rowspan=\"(\d)+\" colspan=\"(\d)+\"></b></td>|" \
                      "<td></td> colspan=\"(\d)+\" rowspan=\"(\d)+\"></b></td>|" \
                      "<td></td> rowspan=\"(\d)+\"></b></td>|" \
                      "<td></td> colspan=\"(\d)+\"></b></td>"
    isolate_iter = re.finditer(isolate_pattern, thead_part)
    isolate_list = [i.group() for i in isolate_iter]

    # 2. find out span number, by step 1 results.
    span_pattern = " rowspan=\"(\d)+\" colspan=\"(\d)+\"|" \
                   " colspan=\"(\d)+\" rowspan=\"(\d)+\"|" \
                   " rowspan=\"(\d)+\"|" \
                   " colspan=\"(\d)+\""
    corrected_list = []
    for isolate_item in isolate_list:
        span_part = re.search(span_pattern, isolate_item)
        spanStr_in_isolateItem = span_part.group()
        # 3. merge the span number into the span token format string.
        if spanStr_in_isolateItem is not None:
            corrected_item = '<td{}></td>'.format(spanStr_in_isolateItem)
            corrected_list.append(corrected_item)
        else:
            corrected_list.append(None)

    # 4. replace original isolated token.
    for corrected_item, isolate_item in zip(corrected_list, isolate_list):
        if corrected_item is not None:
            thead_part = thead_part.replace(isolate_item, corrected_item)
        else:
            pass
    return thead_part


def deal_duplicate_bb(thead_part):
    """
    Deal duplicate <b> or </b> after replace.
    Keep one <b></b> in a <td></td> token.
    :param thead_part:
    :return:
    """
    # 1. find out <td></td> in <thead></thead>.
    td_pattern = "<td rowspan=\"(\d)+\" colspan=\"(\d)+\">(.+?)</td>|" \
                 "<td colspan=\"(\d)+\" rowspan=\"(\d)+\">(.+?)</td>|" \
                 "<td rowspan=\"(\d)+\">(.+?)</td>|" \
                 "<td colspan=\"(\d)+\">(.+?)</td>|" \
                 "<td>(.*?)</td>"
    td_iter = re.finditer(td_pattern, thead_part)
    td_list = [t.group() for t in td_iter]

    # 2. is multiply <b></b> in <td></td> or not?
    new_td_list = []
    for td_item in td_list:
        if td_item.count('<b>') > 1 or td_item.count('</b>') > 1:
            # multiply <b></b> in <td></td> case.
            # 1. remove all <b></b>
            td_item = td_item.replace('<b>', '').replace('</b>', '')
            # 2. replace <tb> -> <tb><b>, </tb> -> </b></tb>.
            td_item = td_item.replace('<td>', '<td><b>').replace('</td>',
                                                                 '</b></td>')
            new_td_list.append(td_item)
        else:
            new_td_list.append(td_item)

    # 3. replace original thead part.
    for td_item, new_td_item in zip(td_list, new_td_list):
        thead_part = thead_part.replace(td_item, new_td_item)
    return thead_part


def deal_bb(result_token):
    """
    In our opinion, <b></b> always occurs in <thead></thead> text's context.
    This function will find out all tokens in <thead></thead> and insert <b></b> by manual.
    :param result_token:
    :return:
    """
    # find out <thead></thead> parts.
    thead_pattern = '<thead>(.*?)</thead>'
    if re.search(thead_pattern, result_token) is None:
        return result_token
    thead_part = re.search(thead_pattern, result_token).group()
    origin_thead_part = copy.deepcopy(thead_part)

    # check "rowspan" or "colspan" occur in <thead></thead> parts or not .
    span_pattern = "<td rowspan=\"(\d)+\" colspan=\"(\d)+\">|<td colspan=\"(\d)+\" rowspan=\"(\d)+\">|<td rowspan=\"(\d)+\">|<td colspan=\"(\d)+\">"
    span_iter = re.finditer(span_pattern, thead_part)
    span_list = [s.group() for s in span_iter]
    has_span_in_head = True if len(span_list) > 0 else False

    if not has_span_in_head:
        # <thead></thead> not include "rowspan" or "colspan" branch 1.
        # 1. replace <td> to <td><b>, and </td> to </b></td>
        # 2. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b></b> to </b>
        thead_part = thead_part.replace('<td>', '<td><b>')\
            .replace('</td>', '</b></td>')\
            .replace('<b><b>', '<b>')\
            .replace('</b></b>', '</b>')
    else:
        # <thead></thead> include "rowspan" or "colspan" branch 2.
        # Firstly, we deal rowspan or colspan cases.
        # 1. replace > to ><b>
        # 2. replace </td> to </b></td>
        # 3. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b><b> to </b>

        # Secondly, deal ordinary cases like branch 1

        # replace ">" to "<b>"
        replaced_span_list = []
        for sp in span_list:
            replaced_span_list.append(sp.replace('>', '><b>'))
        for sp, rsp in zip(span_list, replaced_span_list):
            thead_part = thead_part.replace(sp, rsp)

        # replace "</td>" to "</b></td>"
        thead_part = thead_part.replace('</td>', '</b></td>')

        # remove duplicated <b> by re.sub
        mb_pattern = "(<b>)+"
        single_b_string = "<b>"
        thead_part = re.sub(mb_pattern, single_b_string, thead_part)

        mgb_pattern = "(</b>)+"
        single_gb_string = "</b>"
        thead_part = re.sub(mgb_pattern, single_gb_string, thead_part)

        # ordinary cases like branch 1
        thead_part = thead_part.replace('<td>', '<td><b>').replace('<b><b>',
                                                                   '<b>')

    # convert <tb><b></b></tb> back to <tb></tb>, empty cell has no <b></b>.
    # but space cell(<tb> </tb>)  is suitable for <td><b> </b></td>
    thead_part = thead_part.replace('<td><b></b></td>', '<td></td>')
    # deal with duplicated <b></b>
    thead_part = deal_duplicate_bb(thead_part)
    # deal with isolate span tokens, which causes by wrong predict by structure prediction.
    # eg.PMC5994107_011_00.png
    thead_part = deal_isolate_span(thead_part)
    # replace original result with new thead part.
    result_token = result_token.replace(origin_thead_part, thead_part)
    return result_token


class Matcher:
    def __init__(self, end2end_file, structure_master_file):
        """
        This class process the end2end results and structure recognition results.
        :param end2end_file: end2end results predict by end2end inference.
        :param structure_master_file: structure recognition results predict by structure master inference.
        """
        self.end2end_file = end2end_file
        self.structure_master_file = structure_master_file
        self.end2end_results = pickle_load(end2end_file, prefix='end2end')
        self.structure_master_results = pickle_load(
            structure_master_file, prefix='structure')

    def match(self):
        """
        Match process:
        pre-process : convert end2end and structure master results to xyxy, xywh ndnarray format.
        1. Use pseBbox is inside masterBbox judge rule
        2. Use iou between pseBbox and masterBbox rule
        3. Use min distance of center point rule
        :return:
        """
        match_results = dict()
        for idx, (file_name,
                  end2end_result) in enumerate(self.end2end_results.items()):
            match_list = []
            if file_name not in self.structure_master_results:
                continue
            structure_master_result = self.structure_master_results[file_name]
            end2end_xyxy_bboxes, end2end_xywh_bboxes, structure_master_xywh_bboxes, structure_master_xyxy_bboxes = \
                get_bboxes_list(end2end_result, structure_master_result)

            # rule 1: center rule
            center_rule_match_list = \
                center_rule_match(end2end_xywh_bboxes, structure_master_xyxy_bboxes)
            match_list.extend(center_rule_match_list)

            # rule 2: iou rule
            # firstly, find not match index in previous step.
            center_no_match_end2end_indexs = \
                find_no_match(match_list, len(end2end_xywh_bboxes), type='end2end')
            if len(center_no_match_end2end_indexs) > 0:
                center_no_match_end2end_xyxy = end2end_xyxy_bboxes[
                    center_no_match_end2end_indexs]
                # secondly, iou rule match
                iou_rule_match_list = \
                    iou_rule_match(center_no_match_end2end_xyxy, center_no_match_end2end_indexs, structure_master_xyxy_bboxes)
                match_list.extend(iou_rule_match_list)

            # rule 3: distance rule
            # match between no-match end2end bboxes and no-match master bboxes.
            # it will return master_bboxes_nums match-pairs.
            # firstly, find not match index in previous step.
            centerIou_no_match_end2end_indexs = \
                find_no_match(match_list, len(end2end_xywh_bboxes), type='end2end')
            centerIou_no_match_master_indexs = \
                find_no_match(match_list, len(structure_master_xywh_bboxes), type='master')
            if len(centerIou_no_match_master_indexs) > 0 and len(
                    centerIou_no_match_end2end_indexs) > 0:
                centerIou_no_match_end2end_xywh = end2end_xywh_bboxes[
                    centerIou_no_match_end2end_indexs]
                centerIou_no_match_master_xywh = structure_master_xywh_bboxes[
                    centerIou_no_match_master_indexs]
                distance_match_list = distance_rule_match(
                    centerIou_no_match_end2end_indexs,
                    centerIou_no_match_end2end_xywh,
                    centerIou_no_match_master_indexs,
                    centerIou_no_match_master_xywh)
                match_list.extend(distance_match_list)

            # TODO:
            # The render no-match pseBbox, insert the last
            # After step3 distance rule, a master bbox at least match one end2end bbox.
            # But end2end bbox maybe overmuch, because numbers of master bbox will cut by max length.
            # For these render end2end bboxes, we will make some virtual master bboxes, and get matching.
            # The above extra insert bboxes will be further processed in "formatOutput" function.
            # After this operation, it will increase TEDS score.
            no_match_end2end_indexes = \
                find_no_match(match_list, len(end2end_xywh_bboxes), type='end2end')
            if len(no_match_end2end_indexes) > 0:
                no_match_end2end_xywh = end2end_xywh_bboxes[
                    no_match_end2end_indexes]
                # sort the render no-match end2end bbox in row
                end2end_sorted_indexes_list, end2end_sorted_bboxes_list, sorted_groups, sorted_bboxes_groups = \
                    sort_bbox(no_match_end2end_xywh, no_match_end2end_indexes)
                # make virtual master bboxes, and get matching with the no-match end2end bboxes.
                extra_match_list = extra_match(
                    end2end_sorted_indexes_list,
                    len(structure_master_xywh_bboxes))
                match_list_add_extra_match = copy.deepcopy(match_list)
                match_list_add_extra_match.extend(extra_match_list)
            else:
                # no no-match end2end bboxes
                match_list_add_extra_match = copy.deepcopy(match_list)
                sorted_groups = []
                sorted_bboxes_groups = []

            match_result_dict = {
                'match_list': match_list,
                'match_list_add_extra_match': match_list_add_extra_match,
                'sorted_groups': sorted_groups,
                'sorted_bboxes_groups': sorted_bboxes_groups
            }

            # format output
            match_result_dict = self._format(match_result_dict, file_name)

            match_results[file_name] = match_result_dict

        return match_results

    def _format(self, match_result, file_name):
        """
        Extend the master token(insert virtual master token), and format matching result.
        :param match_result:
        :param file_name:
        :return:
        """
        end2end_info = self.end2end_results[file_name]
        master_info = self.structure_master_results[file_name]
        master_token = master_info['text']
        sorted_groups = match_result['sorted_groups']

        # creat virtual master token
        virtual_master_token_list = []
        for line_group in sorted_groups:
            tmp_list = ['<tr>']
            item_nums = len(line_group)
            for _ in range(item_nums):
                tmp_list.append('<td></td>')
            tmp_list.append('</tr>')
            virtual_master_token_list.extend(tmp_list)

        # insert virtual master token
        master_token_list = master_token.split(',')
        if master_token_list[-1] == '</tbody>':
            # complete predict(no cut by max length)
            # This situation insert virtual master token will drop TEDs score in val set.
            # So we will not extend virtual token in this situation.

            # fake extend virtual
            master_token_list[:-1].extend(virtual_master_token_list)

            # real extend virtual
            # master_token_list = master_token_list[:-1]
            # master_token_list.extend(virtual_master_token_list)
            # master_token_list.append('</tbody>')

        elif master_token_list[-1] == '<td></td>':
            master_token_list.append('</tr>')
            master_token_list.extend(virtual_master_token_list)
            master_token_list.append('</tbody>')
        else:
            master_token_list.extend(virtual_master_token_list)
            master_token_list.append('</tbody>')

        # format output
        match_result.setdefault('matched_master_token_list', master_token_list)
        return match_result

    def get_merge_result(self, match_results):
        """
        Merge the OCR result into structure token to get final results.
        :param match_results:
        :return:
        """
        merged_results = dict()

        # break_token is linefeed token, when one master bbox has multiply end2end bboxes.
        break_token = ' '

        for idx, (file_name, match_info) in enumerate(match_results.items()):
            end2end_info = self.end2end_results[file_name]
            master_token_list = match_info['matched_master_token_list']
            match_list = match_info['match_list_add_extra_match']

            match_dict = get_match_dict(match_list)
            match_text_dict = get_match_text_dict(match_dict, end2end_info,
                                                  break_token)
            merged_result = insert_text_to_token(master_token_list,
                                                 match_text_dict)
            merged_result = deal_bb(merged_result)

            merged_results[file_name] = merged_result

        return merged_results


class TableMasterMatcher(Matcher):
    def __init__(self):
        pass

    def __call__(self, structure_res, dt_boxes, rec_res, img_name=1):
        end2end_results = {img_name: []}
        for dt_box, res in zip(dt_boxes, rec_res):
            d = dict(
                bbox=np.array(dt_box),
                text=res[0], )
            end2end_results[img_name].append(d)

        self.end2end_results = end2end_results

        structure_master_result_dict = {img_name: {}}
        pred_structures, pred_bboxes = structure_res
        pred_structures = ','.join(pred_structures[3:-3])
        structure_master_result_dict[img_name]['text'] = pred_structures
        structure_master_result_dict[img_name]['bbox'] = pred_bboxes
        self.structure_master_results = structure_master_result_dict

        # match
        match_results = self.match()
        merged_results = self.get_merge_result(match_results)
        pred_html = merged_results[img_name]
        pred_html = '<html><body><table>' + pred_html + '</table></body></html>'
        return pred_html
