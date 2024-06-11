# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# This code is refer from: https://github.com/KaiyangZhou/pytorch-center-loss

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CenterLoss(nn.Layer):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = paddle.randn(shape=[self.num_classes, self.feat_dim]).astype(
            "float64"
        )

        if center_file_path is not None:
            assert os.path.exists(
                center_file_path
            ), f"center path({center_file_path}) must exist when it is not None."
            with open(center_file_path, "rb") as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    self.centers[key] = paddle.to_tensor(char_dict[key])

    def __call__(self, predicts, batch):
        assert isinstance(predicts, (list, tuple))
        features, predicts = predicts

        feats_reshape = paddle.reshape(features, [-1, features.shape[-1]]).astype(
            "float64"
        )
        label = paddle.argmax(predicts, axis=2)
        label = paddle.reshape(label, [label.shape[0] * label.shape[1]])

        batch_size = feats_reshape.shape[0]

        # calc l2 distance between feats and centers
        square_feat = paddle.sum(paddle.square(feats_reshape), axis=1, keepdim=True)
        square_feat = paddle.expand(square_feat, [batch_size, self.num_classes])

        square_center = paddle.sum(paddle.square(self.centers), axis=1, keepdim=True)
        square_center = paddle.expand(
            square_center, [self.num_classes, batch_size]
        ).astype("float64")
        square_center = paddle.transpose(square_center, [1, 0])

        distmat = paddle.add(square_feat, square_center)
        feat_dot_center = paddle.matmul(
            feats_reshape, paddle.transpose(self.centers, [1, 0])
        )
        distmat = distmat - 2.0 * feat_dot_center

        # generate the mask
        classes = paddle.arange(self.num_classes).astype("int64")
        label = paddle.expand(
            paddle.unsqueeze(label, 1), (batch_size, self.num_classes)
        )
        mask = paddle.equal(
            paddle.expand(classes, [batch_size, self.num_classes]), label
        ).astype("float64")
        dist = paddle.multiply(distmat, mask)

        loss = paddle.sum(paddle.clip(dist, min=1e-12, max=1e12)) / batch_size
        return {"loss_center": loss}
