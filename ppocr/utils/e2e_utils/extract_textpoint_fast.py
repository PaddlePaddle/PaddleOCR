# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains various CTC decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import math

import numpy as np
from itertools import groupby
from skimage.morphology._skeletonize import thin


def get_dict(character_dict_path):
    character_str = ""
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            character_str += line
        dict_character = list(character_str)
    return dict_character


def softmax(logits):
    """
    logits: N x d
    """
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist


def get_keep_pos_idxs(labels, remove_blank=None):
    """
    Remove duplicate and get pos idxs of keep items.
    The value of keep_blank should be [None, 95].
    """
    duplicate_len_list = []
    keep_pos_idx_list = []
    keep_char_idx_list = []
    for k, v_ in groupby(labels):
        current_len = len(list(v_))
        if k != remove_blank:
            current_idx = int(sum(duplicate_len_list) + current_len // 2)
            keep_pos_idx_list.append(current_idx)
            keep_char_idx_list.append(k)
        duplicate_len_list.append(current_len)
    return keep_char_idx_list, keep_pos_idx_list


def remove_blank(labels, blank=0):
    new_labels = [x for x in labels if x != blank]
    return new_labels


def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels


def ctc_greedy_decoder(probs_seq, blank=95, keep_blank_in_idxs=True):
    """
    CTC greedy (best path) decoder.
    """
    raw_str = np.argmax(np.array(probs_seq), axis=1)
    remove_blank_in_pos = None if keep_blank_in_idxs else blank
    dedup_str, keep_idx_list = get_keep_pos_idxs(
        raw_str, remove_blank=remove_blank_in_pos)
    dst_str = remove_blank(dedup_str, blank=blank)
    return dst_str, keep_idx_list


def instance_ctc_greedy_decoder(gather_info,
                                logits_map,
                                pts_num=4,
                                point_gather_mode=None):
    _, _, C = logits_map.shape
    if point_gather_mode == 'align':
        insert_num = 0
        gather_info = np.array(gather_info)
        length = len(gather_info) - 1
        for index in range(length):
            stride_y = np.abs(gather_info[index + insert_num][0] - gather_info[
                index + 1 + insert_num][0])
            stride_x = np.abs(gather_info[index + insert_num][1] - gather_info[
                index + 1 + insert_num][1])
            max_points = int(max(stride_x, stride_y))
            stride = (gather_info[index + insert_num] -
                      gather_info[index + 1 + insert_num]) / (max_points)
            insert_num_temp = max_points - 1

            for i in range(int(insert_num_temp)):
                insert_value = gather_info[index + insert_num] - (i + 1
                                                                  ) * stride
                insert_index = index + i + 1 + insert_num
                gather_info = np.insert(
                    gather_info, insert_index, insert_value, axis=0)
            insert_num += insert_num_temp
        gather_info = gather_info.tolist()
    else:
        pass
    ys, xs = zip(*gather_info)
    logits_seq = logits_map[list(ys), list(xs)]
    probs_seq = logits_seq
    labels = np.argmax(probs_seq, axis=1)
    dst_str = [k for k, v_ in groupby(labels) if k != C - 1]
    detal = len(gather_info) // (pts_num - 1)
    keep_idx_list = [0] + [detal * (i + 1) for i in range(pts_num - 2)] + [-1]
    keep_gather_list = [gather_info[idx] for idx in keep_idx_list]
    return dst_str, keep_gather_list


def ctc_decoder_for_image(gather_info_list,
                          logits_map,
                          Lexicon_Table,
                          pts_num=6,
                          point_gather_mode=None):
    """
    CTC decoder using multiple processes.
    """
    decoder_str = []
    decoder_xys = []
    for gather_info in gather_info_list:
        if len(gather_info) < pts_num:
            continue
        dst_str, xys_list = instance_ctc_greedy_decoder(
            gather_info,
            logits_map,
            pts_num=pts_num,
            point_gather_mode=point_gather_mode)
        dst_str_readable = ''.join([Lexicon_Table[idx] for idx in dst_str])
        if len(dst_str_readable) < 2:
            continue
        decoder_str.append(dst_str_readable)
        decoder_xys.append(xys_list)
    return decoder_str, decoder_xys


def sort_with_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """

    def sort_part_with_direction(pos_list, point_direction):
        pos_list = np.array(pos_list).reshape(-1, 2)
        point_direction = np.array(point_direction).reshape(-1, 2)
        average_direction = np.mean(point_direction, axis=0, keepdims=True)
        pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
        sorted_list = pos_list[np.argsort(pos_proj_leng)].tolist()
        sorted_direction = point_direction[np.argsort(pos_proj_leng)].tolist()
        return sorted_list, sorted_direction

    pos_list = np.array(pos_list).reshape(-1, 2)
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]  # x, y
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    sorted_point, sorted_direction = sort_part_with_direction(pos_list,
                                                              point_direction)

    point_num = len(sorted_point)
    if point_num >= 16:
        middle_num = point_num // 2
        first_part_point = sorted_point[:middle_num]
        first_point_direction = sorted_direction[:middle_num]
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(
            first_part_point, first_point_direction)

        last_part_point = sorted_point[middle_num:]
        last_point_direction = sorted_direction[middle_num:]
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(
            last_part_point, last_point_direction)
        sorted_point = sorted_fist_part_point + sorted_last_part_point
        sorted_direction = sorted_fist_part_direction + sorted_last_part_direction

    return sorted_point, np.array(sorted_direction)


def add_id(pos_list, image_id=0):
    """
    Add id for gather feature, for inference.
    """
    new_list = []
    for item in pos_list:
        new_list.append((image_id, item[0], item[1]))
    return new_list


def sort_and_expand_with_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    h, w, _ = f_direction.shape
    sorted_list, point_direction = sort_with_direction(pos_list, f_direction)

    point_num = len(sorted_list)
    sub_direction_len = max(point_num // 3, 2)
    left_direction = point_direction[:sub_direction_len, :]
    right_dirction = point_direction[point_num - sub_direction_len:, :]

    left_average_direction = -np.mean(left_direction, axis=0, keepdims=True)
    left_average_len = np.linalg.norm(left_average_direction)
    left_start = np.array(sorted_list[0])
    left_step = left_average_direction / (left_average_len + 1e-6)

    right_average_direction = np.mean(right_dirction, axis=0, keepdims=True)
    right_average_len = np.linalg.norm(right_average_direction)
    right_step = right_average_direction / (right_average_len + 1e-6)
    right_start = np.array(sorted_list[-1])

    append_num = max(
        int((left_average_len + right_average_len) / 2.0 * 0.15), 1)
    left_list = []
    right_list = []
    for i in range(append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ly < h and lx < w and (ly, lx) not in left_list:
            left_list.append((ly, lx))
        ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ry < h and rx < w and (ry, rx) not in right_list:
            right_list.append((ry, rx))

    all_list = left_list[::-1] + sorted_list + right_list
    return all_list


def sort_and_expand_with_direction_v2(pos_list, f_direction, binary_tcl_map):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    binary_tcl_map: h x w
    """
    h, w, _ = f_direction.shape
    sorted_list, point_direction = sort_with_direction(pos_list, f_direction)

    point_num = len(sorted_list)
    sub_direction_len = max(point_num // 3, 2)
    left_direction = point_direction[:sub_direction_len, :]
    right_dirction = point_direction[point_num - sub_direction_len:, :]

    left_average_direction = -np.mean(left_direction, axis=0, keepdims=True)
    left_average_len = np.linalg.norm(left_average_direction)
    left_start = np.array(sorted_list[0])
    left_step = left_average_direction / (left_average_len + 1e-6)

    right_average_direction = np.mean(right_dirction, axis=0, keepdims=True)
    right_average_len = np.linalg.norm(right_average_direction)
    right_step = right_average_direction / (right_average_len + 1e-6)
    right_start = np.array(sorted_list[-1])

    append_num = max(
        int((left_average_len + right_average_len) / 2.0 * 0.15), 1)
    max_append_num = 2 * append_num

    left_list = []
    right_list = []
    for i in range(max_append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ly < h and lx < w and (ly, lx) not in left_list:
            if binary_tcl_map[ly, lx] > 0.5:
                left_list.append((ly, lx))
            else:
                break

    for i in range(max_append_num):
        ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype(
            'int32').tolist()
        if ry < h and rx < w and (ry, rx) not in right_list:
            if binary_tcl_map[ry, rx] > 0.5:
                right_list.append((ry, rx))
            else:
                break

    all_list = left_list[::-1] + sorted_list + right_list
    return all_list


def point_pair2poly(point_pair_list):
    """
    Transfer vertical point_pairs into poly point in clockwise.
    """
    point_num = len(point_pair_list) * 2
    point_list = [0] * point_num
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2)


def shrink_quad_along_width(quad, begin_width_ratio=0., end_width_ratio=1.):
    ratio_pair = np.array(
        [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
    p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
    p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
    return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])


def expand_poly_along_width(poly, shrink_ratio_of_width=0.3):
    """
    expand poly along width.
    """
    point_num = poly.shape[0]
    left_quad = np.array(
        [poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
    left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                 (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    right_quad = np.array(
        [
            poly[point_num // 2 - 2], poly[point_num // 2 - 1],
            poly[point_num // 2], poly[point_num // 2 + 1]
        ],
        dtype=np.float32)
    right_ratio = 1.0 + shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                  (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    poly[0] = left_quad_expand[0]
    poly[-1] = left_quad_expand[-1]
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2] = right_quad_expand[2]
    return poly


def restore_poly(instance_yxs_list, seq_strs, p_border, ratio_w, ratio_h, src_w,
                 src_h, valid_set):
    poly_list = []
    keep_str_list = []
    for yx_center_line, keep_str in zip(instance_yxs_list, seq_strs):
        if len(keep_str) < 2:
            print('--> too short, {}'.format(keep_str))
            continue

        offset_expand = 1.0
        if valid_set == 'totaltext':
            offset_expand = 1.2

        point_pair_list = []
        for y, x in yx_center_line:
            offset = p_border[:, y, x].reshape(2, 2) * offset_expand
            ori_yx = np.array([y, x], dtype=np.float32)
            point_pair = (ori_yx + offset)[:, ::-1] * 4.0 / np.array(
                [ratio_w, ratio_h]).reshape(-1, 2)
            point_pair_list.append(point_pair)

        detected_poly = point_pair2poly(point_pair_list)
        detected_poly = expand_poly_along_width(
            detected_poly, shrink_ratio_of_width=0.2)
        detected_poly[:, 0] = np.clip(detected_poly[:, 0], a_min=0, a_max=src_w)
        detected_poly[:, 1] = np.clip(detected_poly[:, 1], a_min=0, a_max=src_h)

        keep_str_list.append(keep_str)
        if valid_set == 'partvgg':
            middle_point = len(detected_poly) // 2
            detected_poly = detected_poly[
                [0, middle_point - 1, middle_point, -1], :]
            poly_list.append(detected_poly)
        elif valid_set == 'totaltext':
            poly_list.append(detected_poly)
        else:
            print('--> Not supported format.')
            exit(-1)
    return poly_list, keep_str_list


def generate_pivot_list_fast(p_score,
                             p_char_maps,
                             f_direction,
                             Lexicon_Table,
                             score_thresh=0.5,
                             point_gather_mode=None):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    p_score = p_score[0]
    f_direction = f_direction.transpose(1, 2, 0)
    p_tcl_map = (p_score > score_thresh) * 1.0
    skeleton_map = thin(p_tcl_map.astype(np.uint8))
    instance_count, instance_label_map = cv2.connectedComponents(
        skeleton_map.astype(np.uint8), connectivity=8)

    # get TCL Instance
    all_pos_yxs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            if len(pos_list) < 3:
                continue

            pos_list_sorted = sort_and_expand_with_direction_v2(
                pos_list, f_direction, p_tcl_map)
            all_pos_yxs.append(pos_list_sorted)

    p_char_maps = p_char_maps.transpose([1, 2, 0])
    decoded_str, keep_yxs_list = ctc_decoder_for_image(
        all_pos_yxs,
        logits_map=p_char_maps,
        Lexicon_Table=Lexicon_Table,
        point_gather_mode=point_gather_mode)
    return keep_yxs_list, decoded_str


def extract_main_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    pos_list = np.array(pos_list)
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    average_direction = np.mean(point_direction, axis=0, keepdims=True)
    average_direction = average_direction / (
        np.linalg.norm(average_direction) + 1e-6)
    return average_direction


def sort_by_direction_with_image_id_deprecated(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[id, y, x], [id, y, x], [id, y, x] ...]
    """
    pos_list_full = np.array(pos_list).reshape(-1, 3)
    pos_list = pos_list_full[:, 1:]
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]  # x, y
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    average_direction = np.mean(point_direction, axis=0, keepdims=True)
    pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
    sorted_list = pos_list_full[np.argsort(pos_proj_leng)].tolist()
    return sorted_list


def sort_by_direction_with_image_id(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """

    def sort_part_with_direction(pos_list_full, point_direction):
        pos_list_full = np.array(pos_list_full).reshape(-1, 3)
        pos_list = pos_list_full[:, 1:]
        point_direction = np.array(point_direction).reshape(-1, 2)
        average_direction = np.mean(point_direction, axis=0, keepdims=True)
        pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
        sorted_list = pos_list_full[np.argsort(pos_proj_leng)].tolist()
        sorted_direction = point_direction[np.argsort(pos_proj_leng)].tolist()
        return sorted_list, sorted_direction

    pos_list = np.array(pos_list).reshape(-1, 3)
    point_direction = f_direction[pos_list[:, 1], pos_list[:, 2]]  # x, y
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    sorted_point, sorted_direction = sort_part_with_direction(pos_list,
                                                              point_direction)

    point_num = len(sorted_point)
    if point_num >= 16:
        middle_num = point_num // 2
        first_part_point = sorted_point[:middle_num]
        first_point_direction = sorted_direction[:middle_num]
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(
            first_part_point, first_point_direction)

        last_part_point = sorted_point[middle_num:]
        last_point_direction = sorted_direction[middle_num:]
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(
            last_part_point, last_point_direction)
        sorted_point = sorted_fist_part_point + sorted_last_part_point
        sorted_direction = sorted_fist_part_direction + sorted_last_part_direction

    return sorted_point
