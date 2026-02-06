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
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/modules/local_graph.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
from ppocr.ext_op import RoIAlignRotated


def normalize_adjacent_matrix(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    A = A + np.eye(A.shape[0])
    d = np.sum(A, axis=0)
    d = np.clip(d, 0, None)
    d_inv = np.power(d, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_inv = np.diag(d_inv)
    G = A.dot(d_inv).transpose().dot(d_inv)
    return G


def euclidean_distance_matrix(A, B):
    """Calculate the Euclidean distance matrix.

    Args:
        A (ndarray): The point sequence.
        B (ndarray): The point sequence with the same dimensions as A.

    returns:
        D (ndarray): The Euclidean distance matrix.
    """
    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[1]

    m = A.shape[0]
    n = B.shape[0]

    A_dots = (A * A).sum(axis=1).reshape((m, 1)) * np.ones(shape=(1, n))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(m, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    D = np.sqrt(D_squared)
    return D


def feature_embedding(input_feats, out_feat_len):
    """Embed features. This code was partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    Args:
        input_feats (ndarray): The input features of shape (N, d), where N is
            the number of nodes in graph, d is the input feature vector length.
        out_feat_len (int): The length of output feature vector.

    Returns:
        embedded_feats (ndarray): The embedded features.
    """
    assert input_feats.ndim == 2
    assert isinstance(out_feat_len, int)
    assert out_feat_len >= input_feats.shape[1]

    num_nodes = input_feats.shape[0]
    feat_dim = input_feats.shape[1]
    feat_repeat_times = out_feat_len // feat_dim
    residue_dim = out_feat_len % feat_dim

    if residue_dim > 0:
        embed_wave = np.array(
            [
                np.power(1000, 2.0 * (j // 2) / feat_repeat_times + 1)
                for j in range(feat_repeat_times + 1)
            ]
        ).reshape((feat_repeat_times + 1, 1, 1))
        repeat_feats = np.repeat(
            np.expand_dims(input_feats, axis=0), feat_repeat_times, axis=0
        )
        residue_feats = np.hstack(
            [
                input_feats[:, 0:residue_dim],
                np.zeros((num_nodes, feat_dim - residue_dim)),
            ]
        )
        residue_feats = np.expand_dims(residue_feats, axis=0)
        repeat_feats = np.concatenate([repeat_feats, residue_feats], axis=0)
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (num_nodes, -1)
        )[:, 0:out_feat_len]
    else:
        embed_wave = np.array(
            [
                np.power(1000, 2.0 * (j // 2) / feat_repeat_times)
                for j in range(feat_repeat_times)
            ]
        ).reshape((feat_repeat_times, 1, 1))
        repeat_feats = np.repeat(
            np.expand_dims(input_feats, axis=0), feat_repeat_times, axis=0
        )
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = (
            np.transpose(embedded_feats, (1, 0, 2))
            .reshape((num_nodes, -1))
            .astype(np.float32)
        )

    return embedded_feats


class LocalGraphs:
    def __init__(
        self,
        k_at_hops,
        num_adjacent_linkages,
        node_geo_feat_len,
        pooling_scale,
        pooling_output_size,
        local_graph_thr,
    ):
        assert len(k_at_hops) == 2
        assert all(isinstance(n, int) for n in k_at_hops)
        assert isinstance(num_adjacent_linkages, int)
        assert isinstance(node_geo_feat_len, int)
        assert isinstance(pooling_scale, float)
        assert all(isinstance(n, int) for n in pooling_output_size)
        assert isinstance(local_graph_thr, float)

        self.k_at_hops = k_at_hops
        self.num_adjacent_linkages = num_adjacent_linkages
        self.node_geo_feat_dim = node_geo_feat_len
        self.pooling = RoIAlignRotated(pooling_output_size, pooling_scale)
        self.local_graph_thr = local_graph_thr

    def generate_local_graphs(self, sorted_dist_inds, gt_comp_labels):
        """Generate local graphs for GCN to predict which instance a text
        component belongs to.

        Args:
            sorted_dist_inds (ndarray): The complete graph node indices, which
                is sorted according to the Euclidean distance.
            gt_comp_labels(ndarray): The ground truth labels define the
                instance to which the text components (nodes in graphs) belong.

        Returns:
            pivot_local_graphs(list[list[int]]): The list of local graph
                neighbor indices of pivots.
            pivot_knns(list[list[int]]): The list of k-nearest neighbor indices
                of pivots.
        """

        assert sorted_dist_inds.ndim == 2
        assert (
            sorted_dist_inds.shape[0]
            == sorted_dist_inds.shape[1]
            == gt_comp_labels.shape[0]
        )

        knn_graph = sorted_dist_inds[:, 1 : self.k_at_hops[0] + 1]
        pivot_local_graphs = []
        pivot_knns = []
        for pivot_ind, knn in enumerate(knn_graph):
            local_graph_neighbors = set(knn)

            for neighbor_ind in knn:
                local_graph_neighbors.update(
                    set(sorted_dist_inds[neighbor_ind, 1 : self.k_at_hops[1] + 1])
                )

            local_graph_neighbors.discard(pivot_ind)
            pivot_local_graph = list(local_graph_neighbors)
            pivot_local_graph.insert(0, pivot_ind)
            pivot_knn = [pivot_ind] + list(knn)

            if pivot_ind < 1:
                pivot_local_graphs.append(pivot_local_graph)
                pivot_knns.append(pivot_knn)
            else:
                add_flag = True
                for graph_ind, added_knn in enumerate(pivot_knns):
                    added_pivot_ind = added_knn[0]
                    added_local_graph = pivot_local_graphs[graph_ind]

                    union = len(
                        set(pivot_local_graph[1:]).union(set(added_local_graph[1:]))
                    )
                    intersect = len(
                        set(pivot_local_graph[1:]).intersection(
                            set(added_local_graph[1:])
                        )
                    )
                    local_graph_iou = intersect / (union + 1e-8)

                    if (
                        local_graph_iou > self.local_graph_thr
                        and pivot_ind in added_knn
                        and gt_comp_labels[added_pivot_ind] == gt_comp_labels[pivot_ind]
                        and gt_comp_labels[pivot_ind] != 0
                    ):
                        add_flag = False
                        break
                if add_flag:
                    pivot_local_graphs.append(pivot_local_graph)
                    pivot_knns.append(pivot_knn)

        return pivot_local_graphs, pivot_knns

    def generate_gcn_input(
        self,
        node_feat_batch,
        node_label_batch,
        local_graph_batch,
        knn_batch,
        sorted_dist_ind_batch,
    ):
        """Generate graph convolution network input data.

        Args:
            node_feat_batch (List[Tensor]): The batched graph node features.
            node_label_batch (List[ndarray]): The batched text component
                labels.
            local_graph_batch (List[List[list[int]]]): The local graph node
                indices of image batch.
            knn_batch (List[List[list[int]]]): The knn graph node indices of
                image batch.
            sorted_dist_ind_batch (list[ndarray]): The node indices sorted
                according to the Euclidean distance.

        Returns:
            local_graphs_node_feat (Tensor): The node features of graph.
            adjacent_matrices (Tensor): The adjacent matrices of local graphs.
            pivots_knn_inds (Tensor): The k-nearest neighbor indices in
                local graph.
            gt_linkage (Tensor): The surpervision signal of GCN for linkage
                prediction.
        """
        assert isinstance(node_feat_batch, list)
        assert isinstance(node_label_batch, list)
        assert isinstance(local_graph_batch, list)
        assert isinstance(knn_batch, list)
        assert isinstance(sorted_dist_ind_batch, list)

        num_max_nodes = max(
            [
                len(pivot_local_graph)
                for pivot_local_graphs in local_graph_batch
                for pivot_local_graph in pivot_local_graphs
            ]
        )

        local_graphs_node_feat = []
        adjacent_matrices = []
        pivots_knn_inds = []
        pivots_gt_linkage = []

        for batch_ind, sorted_dist_inds in enumerate(sorted_dist_ind_batch):
            node_feats = node_feat_batch[batch_ind]
            pivot_local_graphs = local_graph_batch[batch_ind]
            pivot_knns = knn_batch[batch_ind]
            node_labels = node_label_batch[batch_ind]

            for graph_ind, pivot_knn in enumerate(pivot_knns):
                pivot_local_graph = pivot_local_graphs[graph_ind]
                num_nodes = len(pivot_local_graph)
                pivot_ind = pivot_local_graph[0]
                node2ind_map = {j: i for i, j in enumerate(pivot_local_graph)}

                knn_inds = paddle.to_tensor([node2ind_map[i] for i in pivot_knn[1:]])
                pivot_feats = node_feats[pivot_ind]
                normalized_feats = (
                    node_feats[paddle.to_tensor(pivot_local_graph)] - pivot_feats
                )

                adjacent_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                for node in pivot_local_graph:
                    neighbors = sorted_dist_inds[
                        node, 1 : self.num_adjacent_linkages + 1
                    ]
                    for neighbor in neighbors:
                        if neighbor in pivot_local_graph:
                            adjacent_matrix[
                                node2ind_map[node], node2ind_map[neighbor]
                            ] = 1
                            adjacent_matrix[
                                node2ind_map[neighbor], node2ind_map[node]
                            ] = 1

                adjacent_matrix = normalize_adjacent_matrix(adjacent_matrix)
                pad_adjacent_matrix = paddle.zeros((num_max_nodes, num_max_nodes))
                pad_adjacent_matrix[:num_nodes, :num_nodes] = paddle.cast(
                    paddle.to_tensor(adjacent_matrix), "float32"
                )

                pad_normalized_feats = paddle.concat(
                    [
                        normalized_feats,
                        paddle.zeros(
                            (num_max_nodes - num_nodes, normalized_feats.shape[1])
                        ),
                    ],
                    axis=0,
                )
                local_graph_labels = node_labels[pivot_local_graph]
                knn_labels = local_graph_labels[knn_inds.numpy()]
                link_labels = (
                    (node_labels[pivot_ind] == knn_labels)
                    & (node_labels[pivot_ind] > 0)
                ).astype(np.int64)
                link_labels = paddle.to_tensor(link_labels)

                local_graphs_node_feat.append(pad_normalized_feats)
                adjacent_matrices.append(pad_adjacent_matrix)
                pivots_knn_inds.append(knn_inds)
                pivots_gt_linkage.append(link_labels)

        local_graphs_node_feat = paddle.stack(local_graphs_node_feat, 0)
        adjacent_matrices = paddle.stack(adjacent_matrices, 0)
        pivots_knn_inds = paddle.stack(pivots_knn_inds, 0)
        pivots_gt_linkage = paddle.stack(pivots_gt_linkage, 0)

        return (
            local_graphs_node_feat,
            adjacent_matrices,
            pivots_knn_inds,
            pivots_gt_linkage,
        )

    def __call__(self, feat_maps, comp_attribs):
        """Generate local graphs as GCN input.

        Args:
            feat_maps (Tensor): The feature maps to extract the content
                features of text components.
            comp_attribs (ndarray): The text component attributes.

        Returns:
            local_graphs_node_feat (Tensor): The node features of graph.
            adjacent_matrices (Tensor): The adjacent matrices of local graphs.
            pivots_knn_inds (Tensor): The k-nearest neighbor indices in local
                graph.
            gt_linkage (Tensor): The surpervision signal of GCN for linkage
                prediction.
        """

        assert isinstance(feat_maps, paddle.Tensor)
        assert comp_attribs.ndim == 3
        assert comp_attribs.shape[2] == 8

        sorted_dist_inds_batch = []
        local_graph_batch = []
        knn_batch = []
        node_feat_batch = []
        node_label_batch = []

        for batch_ind in range(comp_attribs.shape[0]):
            num_comps = int(comp_attribs[batch_ind, 0, 0])
            comp_geo_attribs = comp_attribs[batch_ind, :num_comps, 1:7]
            node_labels = comp_attribs[batch_ind, :num_comps, 7].astype(np.int32)

            comp_centers = comp_geo_attribs[:, 0:2]
            distance_matrix = euclidean_distance_matrix(comp_centers, comp_centers)

            batch_id = (
                np.zeros((comp_geo_attribs.shape[0], 1), dtype=np.float32) * batch_ind
            )
            comp_geo_attribs[:, -2] = np.clip(comp_geo_attribs[:, -2], -1, 1)
            angle = np.arccos(comp_geo_attribs[:, -2]) * np.sign(
                comp_geo_attribs[:, -1]
            )
            angle = angle.reshape((-1, 1))
            rotated_rois = np.hstack([batch_id, comp_geo_attribs[:, :-2], angle])
            rois = paddle.to_tensor(rotated_rois)
            content_feats = self.pooling(feat_maps[batch_ind].unsqueeze(0), rois)

            content_feats = content_feats.reshape([content_feats.shape[0], -1])
            geo_feats = feature_embedding(comp_geo_attribs, self.node_geo_feat_dim)
            geo_feats = paddle.to_tensor(geo_feats)
            node_feats = paddle.concat([content_feats, geo_feats], axis=-1)

            sorted_dist_inds = np.argsort(distance_matrix, axis=1)
            pivot_local_graphs, pivot_knns = self.generate_local_graphs(
                sorted_dist_inds, node_labels
            )

            node_feat_batch.append(node_feats)
            node_label_batch.append(node_labels)
            local_graph_batch.append(pivot_local_graphs)
            knn_batch.append(pivot_knns)
            sorted_dist_inds_batch.append(sorted_dist_inds)

        (node_feats, adjacent_matrices, knn_inds, gt_linkage) = self.generate_gcn_input(
            node_feat_batch,
            node_label_batch,
            local_graph_batch,
            knn_batch,
            sorted_dist_inds_batch,
        )

        return node_feats, adjacent_matrices, knn_inds, gt_linkage
