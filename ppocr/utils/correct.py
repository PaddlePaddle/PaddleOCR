import math
import cv2
import copy
import numpy as np
from numpy.linalg import norm
from scipy.special import comb as n_over_k


def vector_slope(vec):
    assert len(vec) == 2
    return abs(vec[1] / (vec[0] + 1e-8))


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x**2, axis=axis))
    return np.sqrt(np.sum(x**2))


def split_edge_seqence(points, long_edge, n_parts):
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while cur_end > point_cumsum[cur_node + 1]:
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


def resample_contour(contour, approx_factor, num_points):
    epsilon = approx_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape((-1, 2))
    pts_num = approx.shape[0]
    e_index = [(i, (i + 1) % pts_num) for i in range(pts_num)]
    ctrl_points = split_edge_seqence(approx, e_index, num_points)
    ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)
    return ctrl_points


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    box = np.array(box)

    return box, min(bounding_box[1])


def reorder_poly_edge(points):
    """Get the respective points composing head edge, tail edge, top
    sideline and bottom sideline.

    Args:
        points (ndarray): The points composing a text polygon.

    Returns:
        head_edge (ndarray): The two points composing the head edge of text
            polygon.
        tail_edge (ndarray): The two points composing the tail edge of text
            polygon.
        top_sideline (ndarray): The points composing top curved sideline of
            text polygon.
        bot_sideline (ndarray): The points composing bottom curved sideline
            of text polygon.
    """

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2

    orientation_thr = 2.0
    head_inds, tail_inds = find_head_tail(points, orientation_thr)
    head_edge, tail_edge = points[head_inds], points[tail_inds]

    pad_points = np.vstack([points, points])
    if tail_inds[1] < 1:
        tail_inds[1] = len(points)
    sideline1 = pad_points[head_inds[1]:tail_inds[1]]
    sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
    sideline_mean_shift = np.mean(
        sideline1, axis=0) - np.mean(
            sideline2, axis=0)

    if sideline_mean_shift[1] > 0:
        top_sideline, bot_sideline = sideline2, sideline1
    else:
        top_sideline, bot_sideline = sideline1, sideline2

    return top_sideline, bot_sideline


def find_head_tail(points, orientation_thr):
    """Find the head edge and tail edge of a text polygon.

    Args:
        points (ndarray): The points composing a text polygon.
        orientation_thr (float): The threshold for distinguishing between
            head edge and tail edge among the horizontal and vertical edges
            of a quadrangle.

    Returns:
        head_inds (list): The indexes of two points composing head edge.
        tail_inds (list): The indexes of two points composing tail edge.
    """

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2
    assert isinstance(orientation_thr, float)

    if len(points) > 4:
        pad_points = np.vstack([points, points[0]])
        edge_vec = pad_points[1:] - pad_points[:-1]

        theta_sum = []
        adjacent_vec_theta = []
        for i, edge_vec1 in enumerate(edge_vec):
            adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
            adjacent_edge_vec = edge_vec[adjacent_ind]
            temp_theta_sum = np.sum(vector_angle(edge_vec1, adjacent_edge_vec))
            temp_adjacent_theta = vector_angle(adjacent_edge_vec[0],
                                               adjacent_edge_vec[1])
            theta_sum.append(temp_theta_sum)
            adjacent_vec_theta.append(temp_adjacent_theta)
        theta_sum_score = np.array(theta_sum) / np.pi
        adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
        poly_center = np.mean(points, axis=0)
        edge_dist = np.maximum(
            norm(
                pad_points[1:] - poly_center, axis=-1),
            norm(
                pad_points[:-1] - poly_center, axis=-1))
        dist_score = edge_dist / np.max(edge_dist)
        position_score = np.zeros(len(edge_vec))
        score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
        score += 0.35 * dist_score
        if len(points) % 2 == 0:
            position_score[(len(score) // 2 - 1)] += 1
            position_score[-1] += 1
        score += 0.1 * position_score
        pad_score = np.concatenate([score, score])
        score_matrix = np.zeros((len(score), len(score) - 3))
        x = np.arange(len(score) - 3) / float(len(score) - 4)
        gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
            (x - 0.5) / 0.5, 2.) / 2)
        gaussian = gaussian / np.max(gaussian)
        for i in range(len(score)):
            score_matrix[i, :] = score[i] + pad_score[(i + 2):(i + len(
                score) - 1)] * gaussian * 0.3

        head_start, tail_increment = np.unravel_index(score_matrix.argmax(),
                                                      score_matrix.shape)
        tail_start = (head_start + tail_increment + 2) % len(points)
        head_end = (head_start + 1) % len(points)
        tail_end = (tail_start + 1) % len(points)

        if head_end > tail_end:
            head_start, tail_start = tail_start, head_start
            head_end, tail_end = tail_end, head_end
        head_inds = [head_start, head_end]
        tail_inds = [tail_start, tail_end]
    else:
        if vector_slope(points[1] - points[0]) + vector_slope(points[
                3] - points[2]) < vector_slope(points[2] - points[
                    1]) + vector_slope(points[0] - points[3]):
            horizontal_edge_inds = [[0, 1], [2, 3]]
            vertical_edge_inds = [[3, 0], [1, 2]]
        else:
            horizontal_edge_inds = [[3, 0], [1, 2]]
            vertical_edge_inds = [[0, 1], [2, 3]]

        vertical_len_sum = norm(points[vertical_edge_inds[0][0]] - points[
            vertical_edge_inds[0][1]]) + norm(points[vertical_edge_inds[1][0]] -
                                              points[vertical_edge_inds[1][1]])
        horizontal_len_sum = norm(points[horizontal_edge_inds[0][0]] - points[
            horizontal_edge_inds[0][1]]) + norm(points[horizontal_edge_inds[1][
                0]] - points[horizontal_edge_inds[1][1]])

        if vertical_len_sum > horizontal_len_sum * orientation_thr:
            head_inds = horizontal_edge_inds[0]
            tail_inds = horizontal_edge_inds[1]
        else:
            head_inds = vertical_edge_inds[0]
            tail_inds = vertical_edge_inds[1]

    return head_inds, tail_inds


def vector_angle(vec1, vec2):
    if vec1.ndim > 1:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
    if vec2.ndim > 1:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
    return np.arccos(np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))


def bezier_coefficient(n, t, k):
    return t**k * (1 - t)**(n - k) * n_over_k(n, k)


def bezier_coefficients(time, point_num, ratios):
    return [[bezier_coefficient(time, ratio, num) for num in range(point_num)]
            for ratio in ratios]


def bezier2curve(bezier: np.ndarray, num_sample: int=10):
    bezier = np.asarray(bezier)
    t = np.linspace(0, 1, num_sample)
    return np.array(bezier_coefficients(3, 4, t)).dot(bezier)


def linear_interpolation(point1: np.ndarray, point2: np.ndarray,
                         number: int=2) -> np.ndarray:
    t = np.linspace(0, 1, number + 2).reshape(-1, 1)
    return point1 + (point2 - point1) * t


def curve2bezier(curve):
    curve = np.array(curve).reshape(-1, 2)
    if len(curve) == 2:
        return linear_interpolation(curve[0], curve[1])
    diff = curve[1:] - curve[:-1]
    distance = np.linalg.norm(diff, axis=-1)
    norm_distance = distance / distance.sum()
    norm_distance = np.hstack(([0], norm_distance))
    cum_norm_dis = norm_distance.cumsum()
    pseudo_inv = np.linalg.pinv(bezier_coefficients(3, 4, cum_norm_dis))
    control_points = pseudo_inv.dot(curve)
    return control_points


def resample_line(line, n):
    """Resample n points on a line.

    Args:
        line (ndarray): The points composing a line.
        n (int): The resampled points number.

    Returns:
        resampled_line (ndarray): The points composing the resampled line.
    """

    assert line.ndim == 2
    assert line.shape[0] >= 2
    assert line.shape[1] == 2
    assert isinstance(n, int)
    assert n > 0

    length_list = [norm(line[i + 1] - line[i]) for i in range(len(line) - 1)]
    total_length = sum(length_list)
    length_cumsum = np.cumsum([0.0] + length_list)
    delta_length = total_length / (float(n) + 1e-8)

    current_edge_ind = 0
    resampled_line = [line[0]]

    for i in range(1, n):
        current_line_len = i * delta_length

        while current_edge_ind + 1 < len(
                length_cumsum) and current_line_len >= length_cumsum[
                    current_edge_ind + 1]:
            current_edge_ind += 1

        current_edge_end_shift = current_line_len - length_cumsum[
            current_edge_ind]

        if current_edge_ind >= len(length_list):
            break
        end_shift_ratio = current_edge_end_shift / length_list[current_edge_ind]
        current_point = line[current_edge_ind] + (line[current_edge_ind + 1] -
                                                  line[current_edge_ind]
                                                  ) * end_shift_ratio
        resampled_line.append(current_point)
    resampled_line.append(line[-1])
    resampled_line = np.array(resampled_line)

    return resampled_line
