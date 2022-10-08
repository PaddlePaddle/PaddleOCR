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
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/postprocess/drrg_postprocessor.py
"""

import functools
import operator

import numpy as np
import paddle
from numpy.linalg import norm
import cv2


class Node:
    def __init__(self, ind):
        self.__ind = ind
        self.__links = set()

    @property
    def ind(self):
        return self.__ind

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, link_node):
        self.__links.add(link_node)
        link_node.__links.add(self)


def graph_propagation(edges, scores, text_comps, edge_len_thr=50.):
    assert edges.ndim == 2
    assert edges.shape[1] == 2
    assert edges.shape[0] == scores.shape[0]
    assert text_comps.ndim == 2
    assert isinstance(edge_len_thr, float)

    edges = np.sort(edges, axis=1)
    score_dict = {}
    for i, edge in enumerate(edges):
        if text_comps is not None:
            box1 = text_comps[edge[0], :8].reshape(4, 2)
            box2 = text_comps[edge[1], :8].reshape(4, 2)
            center1 = np.mean(box1, axis=0)
            center2 = np.mean(box2, axis=0)
            distance = norm(center1 - center2)
            if distance > edge_len_thr:
                scores[i] = 0
        if (edge[0], edge[1]) in score_dict:
            score_dict[edge[0], edge[1]] = 0.5 * (
                score_dict[edge[0], edge[1]] + scores[i])
        else:
            score_dict[edge[0], edge[1]] = scores[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((np.max(nodes) + 1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    order_inds = mapping[edges]
    vertices = [Node(node) for node in nodes]
    for ind in order_inds:
        vertices[ind[0]].add_link(vertices[ind[1]])

    return vertices, score_dict


def connected_components(nodes, score_dict, link_thr):
    assert isinstance(nodes, list)
    assert all([isinstance(node, Node) for node in nodes])
    assert isinstance(score_dict, dict)
    assert isinstance(link_thr, float)

    clusters = []
    nodes = set(nodes)
    while nodes:
        node = nodes.pop()
        cluster = {node}
        node_queue = [node]
        while node_queue:
            node = node_queue.pop(0)
            neighbors = set([
                neighbor for neighbor in node.links
                if score_dict[tuple(sorted([node.ind, neighbor.ind]))] >=
                link_thr
            ])
            neighbors.difference_update(cluster)
            nodes.difference_update(neighbors)
            cluster.update(neighbors)
            node_queue.extend(neighbors)
        clusters.append(list(cluster))
    return clusters


def clusters2labels(clusters, num_nodes):
    assert isinstance(clusters, list)
    assert all([isinstance(cluster, list) for cluster in clusters])
    assert all(
        [isinstance(node, Node) for cluster in clusters for node in cluster])
    assert isinstance(num_nodes, int)

    node_labels = np.zeros(num_nodes)
    for cluster_ind, cluster in enumerate(clusters):
        for node in cluster:
            node_labels[node.ind] = cluster_ind
    return node_labels


def remove_single(text_comps, comp_pred_labels):
    assert text_comps.ndim == 2
    assert text_comps.shape[0] == comp_pred_labels.shape[0]

    single_flags = np.zeros_like(comp_pred_labels)
    pred_labels = np.unique(comp_pred_labels)
    for label in pred_labels:
        current_label_flag = (comp_pred_labels == label)
        if np.sum(current_label_flag) == 1:
            single_flags[np.where(current_label_flag)[0][0]] = 1
    keep_ind = [i for i in range(len(comp_pred_labels)) if not single_flags[i]]
    filtered_text_comps = text_comps[keep_ind, :]
    filtered_labels = comp_pred_labels[keep_ind]

    return filtered_text_comps, filtered_labels


def norm2(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


def min_connect_path(points):
    assert isinstance(points, list)
    assert all([isinstance(point, list) for point in points])
    assert all([isinstance(coord, int) for point in points for coord in point])

    points_queue = points.copy()
    shortest_path = []
    current_edge = [[], []]

    edge_dict0 = {}
    edge_dict1 = {}
    current_edge[0] = points_queue[0]
    current_edge[1] = points_queue[0]
    points_queue.remove(points_queue[0])
    while points_queue:
        for point in points_queue:
            length0 = norm2(point, current_edge[0])
            edge_dict0[length0] = [point, current_edge[0]]
            length1 = norm2(current_edge[1], point)
            edge_dict1[length1] = [current_edge[1], point]
        key0 = min(edge_dict0.keys())
        key1 = min(edge_dict1.keys())

        if key0 <= key1:
            start = edge_dict0[key0][0]
            end = edge_dict0[key0][1]
            shortest_path.insert(0, [points.index(start), points.index(end)])
            points_queue.remove(start)
            current_edge[0] = start
        else:
            start = edge_dict1[key1][0]
            end = edge_dict1[key1][1]
            shortest_path.append([points.index(start), points.index(end)])
            points_queue.remove(end)
            current_edge[1] = end

        edge_dict0 = {}
        edge_dict1 = {}

    shortest_path = functools.reduce(operator.concat, shortest_path)
    shortest_path = sorted(set(shortest_path), key=shortest_path.index)

    return shortest_path


def in_contour(cont, point):
    x, y = point
    is_inner = cv2.pointPolygonTest(cont, (int(x), int(y)), False) > 0.5
    return is_inner


def fix_corner(top_line, bot_line, start_box, end_box):
    assert isinstance(top_line, list)
    assert all(isinstance(point, list) for point in top_line)
    assert isinstance(bot_line, list)
    assert all(isinstance(point, list) for point in bot_line)
    assert start_box.shape == end_box.shape == (4, 2)

    contour = np.array(top_line + bot_line[::-1])
    start_left_mid = (start_box[0] + start_box[3]) / 2
    start_right_mid = (start_box[1] + start_box[2]) / 2
    end_left_mid = (end_box[0] + end_box[3]) / 2
    end_right_mid = (end_box[1] + end_box[2]) / 2
    if not in_contour(contour, start_left_mid):
        top_line.insert(0, start_box[0].tolist())
        bot_line.insert(0, start_box[3].tolist())
    elif not in_contour(contour, start_right_mid):
        top_line.insert(0, start_box[1].tolist())
        bot_line.insert(0, start_box[2].tolist())
    if not in_contour(contour, end_left_mid):
        top_line.append(end_box[0].tolist())
        bot_line.append(end_box[3].tolist())
    elif not in_contour(contour, end_right_mid):
        top_line.append(end_box[1].tolist())
        bot_line.append(end_box[2].tolist())
    return top_line, bot_line


def comps2boundaries(text_comps, comp_pred_labels):
    assert text_comps.ndim == 2
    assert len(text_comps) == len(comp_pred_labels)
    boundaries = []
    if len(text_comps) < 1:
        return boundaries
    for cluster_ind in range(0, int(np.max(comp_pred_labels)) + 1):
        cluster_comp_inds = np.where(comp_pred_labels == cluster_ind)
        text_comp_boxes = text_comps[cluster_comp_inds, :8].reshape(
            (-1, 4, 2)).astype(np.int32)
        score = np.mean(text_comps[cluster_comp_inds, -1])

        if text_comp_boxes.shape[0] < 1:
            continue

        elif text_comp_boxes.shape[0] > 1:
            centers = np.mean(text_comp_boxes, axis=1).astype(np.int32).tolist()
            shortest_path = min_connect_path(centers)
            text_comp_boxes = text_comp_boxes[shortest_path]
            top_line = np.mean(
                text_comp_boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
            bot_line = np.mean(
                text_comp_boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
            top_line, bot_line = fix_corner(
                top_line, bot_line, text_comp_boxes[0], text_comp_boxes[-1])
            boundary_points = top_line + bot_line[::-1]

        else:
            top_line = text_comp_boxes[0, 0:2, :].astype(np.int32).tolist()
            bot_line = text_comp_boxes[0, 2:4:-1, :].astype(np.int32).tolist()
            boundary_points = top_line + bot_line

        boundary = [p for coord in boundary_points for p in coord] + [score]
        boundaries.append(boundary)

    return boundaries


class DRRGPostprocess(object):
    """Merge text components and construct boundaries of text instances.

    Args:
        link_thr (float): The edge score threshold.
    """

    def __init__(self, link_thr, **kwargs):
        assert isinstance(link_thr, float)
        self.link_thr = link_thr

    def __call__(self, preds, shape_list):
        """
        Args:
            edges (ndarray): The edge array of shape N * 2, each row is a node
                index pair that makes up an edge in graph.
            scores (ndarray): The edge score array of shape (N,).
            text_comps (ndarray): The text components.

        Returns:
            List[list[float]]: The predicted boundaries of text instances.
        """
        edges, scores, text_comps = preds
        if edges is not None:
            if isinstance(edges, paddle.Tensor):
                edges = edges.numpy()
            if isinstance(scores, paddle.Tensor):
                scores = scores.numpy()
            if isinstance(text_comps, paddle.Tensor):
                text_comps = text_comps.numpy()
            assert len(edges) == len(scores)
            assert text_comps.ndim == 2
            assert text_comps.shape[1] == 9

            vertices, score_dict = graph_propagation(edges, scores, text_comps)
            clusters = connected_components(vertices, score_dict, self.link_thr)
            pred_labels = clusters2labels(clusters, text_comps.shape[0])
            text_comps, pred_labels = remove_single(text_comps, pred_labels)
            boundaries = comps2boundaries(text_comps, pred_labels)
        else:
            boundaries = []

        boundaries, scores = self.resize_boundary(
            boundaries, (1 / shape_list[0, 2:]).tolist()[::-1])
        boxes_batch = [dict(points=boundaries, scores=scores)]
        return boxes_batch

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.

        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
            with size 2k+1 with k>=4.
            scale_factor(ndarray): The scale factor of size (4,).

        Returns:
            boundaries (list[list[float]]): The scaled boundaries.
        """
        boxes = []
        scores = []
        for b in boundaries:
            sz = len(b)
            scores.append(b[-1])
            b = (np.array(b[:sz - 1]) *
                 (np.tile(scale_factor[:2], int(
                     (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
            boxes.append(np.array(b).reshape([-1, 2]))
        return boxes, scores
