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
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/dense_heads/drrg_head.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .gcn import GCN
from .local_graph import LocalGraphs
from .proposal_local_graph import ProposalLocalGraphs


class DRRGHead(nn.Layer):
    def __init__(
        self,
        in_channels,
        k_at_hops=(8, 4),
        num_adjacent_linkages=3,
        node_geo_feat_len=120,
        pooling_scale=1.0,
        pooling_output_size=(4, 3),
        nms_thr=0.3,
        min_width=8.0,
        max_width=24.0,
        comp_shrink_ratio=1.03,
        comp_ratio=0.4,
        comp_score_thr=0.3,
        text_region_thr=0.2,
        center_region_thr=0.2,
        center_region_area_thr=50,
        local_graph_thr=0.7,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(k_at_hops, tuple)
        assert isinstance(num_adjacent_linkages, int)
        assert isinstance(node_geo_feat_len, int)
        assert isinstance(pooling_scale, float)
        assert isinstance(pooling_output_size, tuple)
        assert isinstance(comp_shrink_ratio, float)
        assert isinstance(nms_thr, float)
        assert isinstance(min_width, float)
        assert isinstance(max_width, float)
        assert isinstance(comp_ratio, float)
        assert isinstance(comp_score_thr, float)
        assert isinstance(text_region_thr, float)
        assert isinstance(center_region_thr, float)
        assert isinstance(center_region_area_thr, int)
        assert isinstance(local_graph_thr, float)

        self.in_channels = in_channels
        self.out_channels = 6
        self.downsample_ratio = 1.0
        self.k_at_hops = k_at_hops
        self.num_adjacent_linkages = num_adjacent_linkages
        self.node_geo_feat_len = node_geo_feat_len
        self.pooling_scale = pooling_scale
        self.pooling_output_size = pooling_output_size
        self.comp_shrink_ratio = comp_shrink_ratio
        self.nms_thr = nms_thr
        self.min_width = min_width
        self.max_width = max_width
        self.comp_ratio = comp_ratio
        self.comp_score_thr = comp_score_thr
        self.text_region_thr = text_region_thr
        self.center_region_thr = center_region_thr
        self.center_region_area_thr = center_region_area_thr
        self.local_graph_thr = local_graph_thr

        self.out_conv = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.graph_train = LocalGraphs(
            self.k_at_hops,
            self.num_adjacent_linkages,
            self.node_geo_feat_len,
            self.pooling_scale,
            self.pooling_output_size,
            self.local_graph_thr,
        )

        self.graph_test = ProposalLocalGraphs(
            self.k_at_hops,
            self.num_adjacent_linkages,
            self.node_geo_feat_len,
            self.pooling_scale,
            self.pooling_output_size,
            self.nms_thr,
            self.min_width,
            self.max_width,
            self.comp_shrink_ratio,
            self.comp_ratio,
            self.comp_score_thr,
            self.text_region_thr,
            self.center_region_thr,
            self.center_region_area_thr,
        )

        pool_w, pool_h = self.pooling_output_size
        node_feat_len = (pool_w * pool_h) * (
            self.in_channels + self.out_channels
        ) + self.node_geo_feat_len
        self.gcn = GCN(node_feat_len)

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs (Tensor): Shape of :math:`(N, C, H, W)`.
            gt_comp_attribs (list[ndarray]): The padded text component
                attributes. Shape: (num_component, 8).

        Returns:
            tuple: Returns (pred_maps, (gcn_pred, gt_labels)).

                - | pred_maps (Tensor): Prediction map with shape
                    :math:`(N, C_{out}, H, W)`.
                - | gcn_pred (Tensor): Prediction from GCN module, with
                    shape :math:`(N, 2)`.
                - | gt_labels (Tensor): Ground-truth label with shape
                    :math:`(N, 8)`.
        """
        if self.training:
            assert targets is not None
            gt_comp_attribs = targets[7]
            pred_maps = self.out_conv(inputs)
            feat_maps = paddle.concat([inputs, pred_maps], axis=1)
            node_feats, adjacent_matrices, knn_inds, gt_labels = self.graph_train(
                feat_maps, np.stack(gt_comp_attribs)
            )

            gcn_pred = self.gcn(node_feats, adjacent_matrices, knn_inds)

            return pred_maps, (gcn_pred, gt_labels)
        else:
            return self.single_test(inputs)

    def single_test(self, feat_maps):
        r"""
        Args:
            feat_maps (Tensor): Shape of :math:`(N, C, H, W)`.

        Returns:
            tuple: Returns (edge, score, text_comps).

                - | edge (ndarray): The edge array of shape :math:`(N, 2)`
                    where each row is a pair of text component indices
                    that makes up an edge in graph.
                - | score (ndarray): The score array of shape :math:`(N,)`,
                    corresponding to the edge above.
                - | text_comps (ndarray): The text components of shape
                    :math:`(N, 9)` where each row corresponds to one box and
                    its score: (x1, y1, x2, y2, x3, y3, x4, y4, score).
        """
        pred_maps = self.out_conv(feat_maps)
        feat_maps = paddle.concat([feat_maps, pred_maps], axis=1)

        none_flag, graph_data = self.graph_test(pred_maps, feat_maps)

        (
            local_graphs_node_feat,
            adjacent_matrices,
            pivots_knn_inds,
            pivot_local_graphs,
            text_comps,
        ) = graph_data

        if none_flag:
            return None, None, None
        gcn_pred = self.gcn(local_graphs_node_feat, adjacent_matrices, pivots_knn_inds)
        pred_labels = F.softmax(gcn_pred, axis=1)

        edges = []
        scores = []
        pivot_local_graphs = pivot_local_graphs.squeeze().numpy()

        for pivot_ind, pivot_local_graph in enumerate(pivot_local_graphs):
            pivot = pivot_local_graph[0]
            for k_ind, neighbor_ind in enumerate(pivots_knn_inds[pivot_ind]):
                neighbor = pivot_local_graph[neighbor_ind.item()]
                edges.append([pivot, neighbor])
                scores.append(
                    pred_labels[pivot_ind * pivots_knn_inds.shape[1] + k_ind, 1].item()
                )

        edges = np.asarray(edges)
        scores = np.asarray(scores)

        return edges, scores, text_comps
