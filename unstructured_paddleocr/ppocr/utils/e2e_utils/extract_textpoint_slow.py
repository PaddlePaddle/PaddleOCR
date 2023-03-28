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


def point_pair2poly(point_pair_list):
    """
    Transfer vertical point_pairs into poly point in clockwise.
    """
    pair_length_list = []
    for point_pair in point_pair_list:
        pair_length = np.linalg.norm(point_pair[0] - point_pair[1])
        pair_length_list.append(pair_length)
    pair_length_list = np.array(pair_length_list)
    pair_info = (pair_length_list.max(), pair_length_list.min(),
                 pair_length_list.mean())

    point_num = len(point_pair_list) * 2
    point_list = [0] * point_num
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2), pair_info


def shrink_quad_along_width(quad, begin_width_ratio=0., end_width_ratio=1.):
    """
    Generate shrink_quad_along_width.
    """
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
    right_ratio = 1.0 + \
                  shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                  (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    poly[0] = left_quad_expand[0]
    poly[-1] = left_quad_expand[-1]
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2] = right_quad_expand[2]
    return poly


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
                                keep_blank_in_idxs=True):
    """
    gather_info: [[x, y], [x, y] ...]
    logits_map: H x W X (n_chars + 1)
    """
    _, _, C = logits_map.shape
    ys, xs = zip(*gather_info)
    logits_seq = logits_map[list(ys), list(xs)]  # n x 96
    probs_seq = softmax(logits_seq)
    dst_str, keep_idx_list = ctc_greedy_decoder(
        probs_seq, blank=C - 1, keep_blank_in_idxs=keep_blank_in_idxs)
    keep_gather_list = [gather_info[idx] for idx in keep_idx_list]
    return dst_str, keep_gather_list


def ctc_decoder_for_image(gather_info_list, logits_map,
                          keep_blank_in_idxs=True):
    """
    CTC decoder using multiple processes.
    """
    decoder_results = []
    for gather_info in gather_info_list:
        res = instance_ctc_greedy_decoder(
            gather_info, logits_map, keep_blank_in_idxs=keep_blank_in_idxs)
        decoder_results.append(res)
    return decoder_results


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

    # expand along
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

    # expand along
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


def generate_pivot_list_curved(p_score,
                               p_char_maps,
                               f_direction,
                               score_thresh=0.5,
                               is_expand=True,
                               is_backbone=False,
                               image_id=0):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    p_score = p_score[0]
    f_direction = f_direction.transpose(1, 2, 0)
    p_tcl_map = (p_score > score_thresh) * 1.0
    skeleton_map = thin(p_tcl_map)
    instance_count, instance_label_map = cv2.connectedComponents(
        skeleton_map.astype(np.uint8), connectivity=8)

    # get TCL Instance
    all_pos_yxs = []
    center_pos_yxs = []
    end_points_yxs = []
    instance_center_pos_yxs = []
    pred_strs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            ### FIX-ME, eliminate outlier
            if len(pos_list) < 3:
                continue

            if is_expand:
                pos_list_sorted = sort_and_expand_with_direction_v2(
                    pos_list, f_direction, p_tcl_map)
            else:
                pos_list_sorted, _ = sort_with_direction(pos_list, f_direction)
            all_pos_yxs.append(pos_list_sorted)

    # use decoder to filter backgroud points.
    p_char_maps = p_char_maps.transpose([1, 2, 0])
    decode_res = ctc_decoder_for_image(
        all_pos_yxs, logits_map=p_char_maps, keep_blank_in_idxs=True)
    for decoded_str, keep_yxs_list in decode_res:
        if is_backbone:
            keep_yxs_list_with_id = add_id(keep_yxs_list, image_id=image_id)
            instance_center_pos_yxs.append(keep_yxs_list_with_id)
            pred_strs.append(decoded_str)
        else:
            end_points_yxs.extend((keep_yxs_list[0], keep_yxs_list[-1]))
            center_pos_yxs.extend(keep_yxs_list)

    if is_backbone:
        return pred_strs, instance_center_pos_yxs
    else:
        return center_pos_yxs, end_points_yxs


def generate_pivot_list_horizontal(p_score,
                                   p_char_maps,
                                   f_direction,
                                   score_thresh=0.5,
                                   is_backbone=False,
                                   image_id=0):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    p_score = p_score[0]
    f_direction = f_direction.transpose(1, 2, 0)
    p_tcl_map_bi = (p_score > score_thresh) * 1.0
    instance_count, instance_label_map = cv2.connectedComponents(
        p_tcl_map_bi.astype(np.uint8), connectivity=8)

    # get TCL Instance
    all_pos_yxs = []
    center_pos_yxs = []
    end_points_yxs = []
    instance_center_pos_yxs = []

    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            ### FIX-ME, eliminate outlier
            if len(pos_list) < 5:
                continue

            # add rule here
            main_direction = extract_main_direction(pos_list,
                                                    f_direction)  # y x
            reference_directin = np.array([0, 1]).reshape([-1, 2])  # y x
            is_h_angle = abs(np.sum(
                main_direction * reference_directin)) < math.cos(math.pi / 180 *
                                                                 70)

            point_yxs = np.array(pos_list)
            max_y, max_x = np.max(point_yxs, axis=0)
            min_y, min_x = np.min(point_yxs, axis=0)
            is_h_len = (max_y - min_y) < 1.5 * (max_x - min_x)

            pos_list_final = []
            if is_h_len:
                xs = np.unique(xs)
                for x in xs:
                    ys = instance_label_map[:, x].copy().reshape((-1, ))
                    y = int(np.where(ys == instance_id)[0].mean())
                    pos_list_final.append((y, x))
            else:
                ys = np.unique(ys)
                for y in ys:
                    xs = instance_label_map[y, :].copy().reshape((-1, ))
                    x = int(np.where(xs == instance_id)[0].mean())
                    pos_list_final.append((y, x))

            pos_list_sorted, _ = sort_with_direction(pos_list_final,
                                                     f_direction)
            all_pos_yxs.append(pos_list_sorted)

    # use decoder to filter backgroud points.
    p_char_maps = p_char_maps.transpose([1, 2, 0])
    decode_res = ctc_decoder_for_image(
        all_pos_yxs, logits_map=p_char_maps, keep_blank_in_idxs=True)
    for decoded_str, keep_yxs_list in decode_res:
        if is_backbone:
            keep_yxs_list_with_id = add_id(keep_yxs_list, image_id=image_id)
            instance_center_pos_yxs.append(keep_yxs_list_with_id)
        else:
            end_points_yxs.extend((keep_yxs_list[0], keep_yxs_list[-1]))
            center_pos_yxs.extend(keep_yxs_list)

    if is_backbone:
        return instance_center_pos_yxs
    else:
        return center_pos_yxs, end_points_yxs


def generate_pivot_list_slow(p_score,
                             p_char_maps,
                             f_direction,
                             score_thresh=0.5,
                             is_backbone=False,
                             is_curved=True,
                             image_id=0):
    """
    Warp all the function together.
    """
    if is_curved:
        return generate_pivot_list_curved(
            p_score,
            p_char_maps,
            f_direction,
            score_thresh=score_thresh,
            is_expand=True,
            is_backbone=is_backbone,
            image_id=image_id)
    else:
        return generate_pivot_list_horizontal(
            p_score,
            p_char_maps,
            f_direction,
            score_thresh=score_thresh,
            is_backbone=is_backbone,
            image_id=image_id)


# for refine module
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


def generate_pivot_list_tt_inference(p_score,
                                     p_char_maps,
                                     f_direction,
                                     score_thresh=0.5,
                                     is_backbone=False,
                                     is_curved=True,
                                     image_id=0):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    p_score = p_score[0]
    f_direction = f_direction.transpose(1, 2, 0)
    p_tcl_map = (p_score > score_thresh) * 1.0
    skeleton_map = thin(p_tcl_map)
    instance_count, instance_label_map = cv2.connectedComponents(
        skeleton_map.astype(np.uint8), connectivity=8)

    # get TCL Instance
    all_pos_yxs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))
            ### FIX-ME, eliminate outlier
            if len(pos_list) < 3:
                continue
            pos_list_sorted = sort_and_expand_with_direction_v2(
                pos_list, f_direction, p_tcl_map)
            pos_list_sorted_with_id = add_id(pos_list_sorted, image_id=image_id)
            all_pos_yxs.append(pos_list_sorted_with_id)
    return all_pos_yxs
