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
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/modules/proposal_local_graph.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from lanms import merge_quadrangle_n9 as la_nms

from ppocr.ext_op import RoIAlignRotated
from .local_graph import (euclidean_distance_matrix, feature_embedding,
                          normalize_adjacent_matrix)


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return ~canvas | input_mask


class ProposalLocalGraphs:
    def __init__(self, k_at_hops, num_adjacent_linkages, node_geo_feat_len,
                 pooling_scale, pooling_output_size, nms_thr, min_width,
                 max_width, comp_shrink_ratio, comp_w_h_ratio, comp_score_thr,
                 text_region_thr, center_region_thr, center_region_area_thr):

        assert len(k_at_hops) == 2
        assert isinstance(k_at_hops, tuple)
        assert isinstance(num_adjacent_linkages, int)
        assert isinstance(node_geo_feat_len, int)
        assert isinstance(pooling_scale, float)
        assert isinstance(pooling_output_size, tuple)
        assert isinstance(nms_thr, float)
        assert isinstance(min_width, float)
        assert isinstance(max_width, float)
        assert isinstance(comp_shrink_ratio, float)
        assert isinstance(comp_w_h_ratio, float)
        assert isinstance(comp_score_thr, float)
        assert isinstance(text_region_thr, float)
        assert isinstance(center_region_thr, float)
        assert isinstance(center_region_area_thr, int)

        self.k_at_hops = k_at_hops
        self.active_connection = num_adjacent_linkages
        self.local_graph_depth = len(self.k_at_hops)
        self.node_geo_feat_dim = node_geo_feat_len
        self.pooling = RoIAlignRotated(pooling_output_size, pooling_scale)
        self.nms_thr = nms_thr
        self.min_width = min_width
        self.max_width = max_width
        self.comp_shrink_ratio = comp_shrink_ratio
        self.comp_w_h_ratio = comp_w_h_ratio
        self.comp_score_thr = comp_score_thr
        self.text_region_thr = text_region_thr
        self.center_region_thr = center_region_thr
        self.center_region_area_thr = center_region_area_thr

    def propose_comps(self, score_map, top_height_map, bot_height_map, sin_map,
                      cos_map, comp_score_thr, min_width, max_width,
                      comp_shrink_ratio, comp_w_h_ratio):
        """Propose text components.

        Args:
            score_map (ndarray): The score map for NMS.
            top_height_map (ndarray): The predicted text height map from each
                pixel in text center region to top sideline.
            bot_height_map (ndarray): The predicted text height map from each
                pixel in text center region to bottom sideline.
            sin_map (ndarray): The predicted sin(theta) map.
            cos_map (ndarray): The predicted cos(theta) map.
            comp_score_thr (float): The score threshold of text component.
            min_width (float): The minimum width of text components.
            max_width (float): The maximum width of text components.
            comp_shrink_ratio (float): The shrink ratio of text components.
            comp_w_h_ratio (float): The width to height ratio of text
                components.

        Returns:
            text_comps (ndarray): The text components.
        """

        comp_centers = np.argwhere(score_map > comp_score_thr)
        comp_centers = comp_centers[np.argsort(comp_centers[:, 0])]
        y = comp_centers[:, 0]
        x = comp_centers[:, 1]

        top_height = top_height_map[y, x].reshape((-1, 1)) * comp_shrink_ratio
        bot_height = bot_height_map[y, x].reshape((-1, 1)) * comp_shrink_ratio
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))

        top_mid_pts = comp_centers + np.hstack(
            [top_height * sin, top_height * cos])
        bot_mid_pts = comp_centers - np.hstack(
            [bot_height * sin, bot_height * cos])

        width = (top_height + bot_height) * comp_w_h_ratio
        width = np.clip(width, min_width, max_width)
        r = width / 2

        tl = top_mid_pts[:, ::-1] - np.hstack([-r * sin, r * cos])
        tr = top_mid_pts[:, ::-1] + np.hstack([-r * sin, r * cos])
        br = bot_mid_pts[:, ::-1] + np.hstack([-r * sin, r * cos])
        bl = bot_mid_pts[:, ::-1] - np.hstack([-r * sin, r * cos])
        text_comps = np.hstack([tl, tr, br, bl]).astype(np.float32)

        score = score_map[y, x].reshape((-1, 1))
        text_comps = np.hstack([text_comps, score])

        return text_comps

    def propose_comps_and_attribs(self, text_region_map, center_region_map,
                                  top_height_map, bot_height_map, sin_map,
                                  cos_map):
        """Generate text components and attributes.

        Args:
            text_region_map (ndarray): The predicted text region probability
                map.
            center_region_map (ndarray): The predicted text center region
                probability map.
            top_height_map (ndarray): The predicted text height map from each
                pixel in text center region to top sideline.
            bot_height_map (ndarray): The predicted text height map from each
                pixel in text center region to bottom sideline.
            sin_map (ndarray): The predicted sin(theta) map.
            cos_map (ndarray): The predicted cos(theta) map.

        Returns:
            comp_attribs (ndarray): The text component attributes.
            text_comps (ndarray): The text components.
        """

        assert (text_region_map.shape == center_region_map.shape ==
                top_height_map.shape == bot_height_map.shape == sin_map.shape ==
                cos_map.shape)
        text_mask = text_region_map > self.text_region_thr
        center_region_mask = (
            center_region_map > self.center_region_thr) * text_mask

        scale = np.sqrt(1.0 / (sin_map**2 + cos_map**2 + 1e-8))
        sin_map, cos_map = sin_map * scale, cos_map * scale

        center_region_mask = fill_hole(center_region_mask)
        center_region_contours, _ = cv2.findContours(
            center_region_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

        mask_sz = center_region_map.shape
        comp_list = []
        for contour in center_region_contours:
            current_center_mask = np.zeros(mask_sz)
            cv2.drawContours(current_center_mask, [contour], -1, 1, -1)
            if current_center_mask.sum() <= self.center_region_area_thr:
                continue
            score_map = text_region_map * current_center_mask

            text_comps = self.propose_comps(
                score_map, top_height_map, bot_height_map, sin_map, cos_map,
                self.comp_score_thr, self.min_width, self.max_width,
                self.comp_shrink_ratio, self.comp_w_h_ratio)

            text_comps = la_nms(text_comps, self.nms_thr)
            text_comp_mask = np.zeros(mask_sz)
            text_comp_boxes = text_comps[:, :8].reshape(
                (-1, 4, 2)).astype(np.int32)

            cv2.drawContours(text_comp_mask, text_comp_boxes, -1, 1, -1)
            if (text_comp_mask * text_mask).sum() < text_comp_mask.sum() * 0.5:
                continue
            if text_comps.shape[-1] > 0:
                comp_list.append(text_comps)

        if len(comp_list) <= 0:
            return None, None

        text_comps = np.vstack(comp_list)
        text_comp_boxes = text_comps[:, :8].reshape((-1, 4, 2))
        centers = np.mean(text_comp_boxes, axis=1).astype(np.int32)
        x = centers[:, 0]
        y = centers[:, 1]

        scores = []
        for text_comp_box in text_comp_boxes:
            text_comp_box[:, 0] = np.clip(text_comp_box[:, 0], 0,
                                          mask_sz[1] - 1)
            text_comp_box[:, 1] = np.clip(text_comp_box[:, 1], 0,
                                          mask_sz[0] - 1)
            min_coord = np.min(text_comp_box, axis=0).astype(np.int32)
            max_coord = np.max(text_comp_box, axis=0).astype(np.int32)
            text_comp_box = text_comp_box - min_coord
            box_sz = (max_coord - min_coord + 1)
            temp_comp_mask = np.zeros((box_sz[1], box_sz[0]), dtype=np.uint8)
            cv2.fillPoly(temp_comp_mask, [text_comp_box.astype(np.int32)], 1)
            temp_region_patch = text_region_map[min_coord[1]:(max_coord[1] + 1),
                                                min_coord[0]:(max_coord[0] + 1)]
            score = cv2.mean(temp_region_patch, temp_comp_mask)[0]
            scores.append(score)
        scores = np.array(scores).reshape((-1, 1))
        text_comps = np.hstack([text_comps[:, :-1], scores])

        h = top_height_map[y, x].reshape(
            (-1, 1)) + bot_height_map[y, x].reshape((-1, 1))
        w = np.clip(h * self.comp_w_h_ratio, self.min_width, self.max_width)
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        comp_attribs = np.hstack([x, y, h, w, cos, sin])

        return comp_attribs, text_comps

    def generate_local_graphs(self, sorted_dist_inds, node_feats):
        """Generate local graphs and graph convolution network input data.

        Args:
            sorted_dist_inds (ndarray): The node indices sorted according to
                the Euclidean distance.
            node_feats (tensor): The features of nodes in graph.

        Returns:
            local_graphs_node_feats (tensor): The features of nodes in local
                graphs.
            adjacent_matrices (tensor): The adjacent matrices.
            pivots_knn_inds (tensor): The k-nearest neighbor indices in
                local graphs.
            pivots_local_graphs (tensor): The indices of nodes in local
                graphs.
        """

        assert sorted_dist_inds.ndim == 2
        assert (sorted_dist_inds.shape[0] == sorted_dist_inds.shape[1] ==
                node_feats.shape[0])

        knn_graph = sorted_dist_inds[:, 1:self.k_at_hops[0] + 1]
        pivot_local_graphs = []
        pivot_knns = []

        for pivot_ind, knn in enumerate(knn_graph):

            local_graph_neighbors = set(knn)

            for neighbor_ind in knn:
                local_graph_neighbors.update(
                    set(sorted_dist_inds[neighbor_ind, 1:self.k_at_hops[1] +
                                         1]))

            local_graph_neighbors.discard(pivot_ind)
            pivot_local_graph = list(local_graph_neighbors)
            pivot_local_graph.insert(0, pivot_ind)
            pivot_knn = [pivot_ind] + list(knn)

            pivot_local_graphs.append(pivot_local_graph)
            pivot_knns.append(pivot_knn)

        num_max_nodes = max([
            len(pivot_local_graph) for pivot_local_graph in pivot_local_graphs
        ])

        local_graphs_node_feat = []
        adjacent_matrices = []
        pivots_knn_inds = []
        pivots_local_graphs = []

        for graph_ind, pivot_knn in enumerate(pivot_knns):
            pivot_local_graph = pivot_local_graphs[graph_ind]
            num_nodes = len(pivot_local_graph)
            pivot_ind = pivot_local_graph[0]
            node2ind_map = {j: i for i, j in enumerate(pivot_local_graph)}

            knn_inds = paddle.cast(
                paddle.to_tensor([node2ind_map[i]
                                  for i in pivot_knn[1:]]), 'int64')
            pivot_feats = node_feats[pivot_ind]
            normalized_feats = node_feats[paddle.to_tensor(
                pivot_local_graph)] - pivot_feats

            adjacent_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for node in pivot_local_graph:
                neighbors = sorted_dist_inds[node, 1:self.active_connection + 1]
                for neighbor in neighbors:
                    if neighbor in pivot_local_graph:
                        adjacent_matrix[node2ind_map[node], node2ind_map[
                            neighbor]] = 1
                        adjacent_matrix[node2ind_map[neighbor], node2ind_map[
                            node]] = 1

            adjacent_matrix = normalize_adjacent_matrix(adjacent_matrix)
            pad_adjacent_matrix = paddle.zeros((num_max_nodes, num_max_nodes), )
            pad_adjacent_matrix[:num_nodes, :num_nodes] = paddle.cast(
                paddle.to_tensor(adjacent_matrix), 'float32')

            pad_normalized_feats = paddle.concat(
                [
                    normalized_feats, paddle.zeros(
                        (num_max_nodes - num_nodes, normalized_feats.shape[1]),
                    )
                ],
                axis=0)

            local_graph_nodes = paddle.to_tensor(pivot_local_graph)
            local_graph_nodes = paddle.concat(
                [
                    local_graph_nodes, paddle.zeros(
                        [num_max_nodes - num_nodes], dtype='int64')
                ],
                axis=-1)

            local_graphs_node_feat.append(pad_normalized_feats)
            adjacent_matrices.append(pad_adjacent_matrix)
            pivots_knn_inds.append(knn_inds)
            pivots_local_graphs.append(local_graph_nodes)

        local_graphs_node_feat = paddle.stack(local_graphs_node_feat, 0)
        adjacent_matrices = paddle.stack(adjacent_matrices, 0)
        pivots_knn_inds = paddle.stack(pivots_knn_inds, 0)
        pivots_local_graphs = paddle.stack(pivots_local_graphs, 0)

        return (local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
                pivots_local_graphs)

    def __call__(self, preds, feat_maps):
        """Generate local graphs and graph convolutional network input data.

        Args:
            preds (tensor): The predicted maps.
            feat_maps (tensor): The feature maps to extract content feature of
                text components.

        Returns:
            none_flag (bool): The flag showing whether the number of proposed
                text components is 0.
            local_graphs_node_feats (tensor): The features of nodes in local
                graphs.
            adjacent_matrices (tensor): The adjacent matrices.
            pivots_knn_inds (tensor): The k-nearest neighbor indices in
                local graphs.
            pivots_local_graphs (tensor): The indices of nodes in local
                graphs.
            text_comps (ndarray): The predicted text components.
        """
        if preds.ndim == 4:
            assert preds.shape[0] == 1
            preds = paddle.squeeze(preds)
        pred_text_region = F.sigmoid(preds[0]).numpy()
        pred_center_region = F.sigmoid(preds[1]).numpy()
        pred_sin_map = preds[2].numpy()
        pred_cos_map = preds[3].numpy()
        pred_top_height_map = preds[4].numpy()
        pred_bot_height_map = preds[5].numpy()

        comp_attribs, text_comps = self.propose_comps_and_attribs(
            pred_text_region, pred_center_region, pred_top_height_map,
            pred_bot_height_map, pred_sin_map, pred_cos_map)

        if comp_attribs is None or len(comp_attribs) < 2:
            none_flag = True
            return none_flag, (0, 0, 0, 0, 0)

        comp_centers = comp_attribs[:, 0:2]
        distance_matrix = euclidean_distance_matrix(comp_centers, comp_centers)

        geo_feats = feature_embedding(comp_attribs, self.node_geo_feat_dim)
        geo_feats = paddle.to_tensor(geo_feats)

        batch_id = np.zeros((comp_attribs.shape[0], 1), dtype=np.float32)
        comp_attribs = comp_attribs.astype(np.float32)
        angle = np.arccos(comp_attribs[:, -2]) * np.sign(comp_attribs[:, -1])
        angle = angle.reshape((-1, 1))
        rotated_rois = np.hstack([batch_id, comp_attribs[:, :-2], angle])
        rois = paddle.to_tensor(rotated_rois)

        content_feats = self.pooling(feat_maps, rois)
        content_feats = content_feats.reshape([content_feats.shape[0], -1])
        node_feats = paddle.concat([content_feats, geo_feats], axis=-1)

        sorted_dist_inds = np.argsort(distance_matrix, axis=1)
        (local_graphs_node_feat, adjacent_matrices, pivots_knn_inds,
         pivots_local_graphs) = self.generate_local_graphs(sorted_dist_inds,
                                                           node_feats)

        none_flag = False
        return none_flag, (local_graphs_node_feat, adjacent_matrices,
                           pivots_knn_inds, pivots_local_graphs, text_comps)
