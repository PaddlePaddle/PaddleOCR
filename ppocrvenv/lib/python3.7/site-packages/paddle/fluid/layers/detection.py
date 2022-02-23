#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
All layers just related to the detection neural network.
"""

from __future__ import print_function

from .layer_function_generator import generate_layer_fn
from .layer_function_generator import autodoc, templatedoc
from ..layer_helper import LayerHelper
from ..framework import Variable, in_dygraph_mode, static_only
from .. import core
from .loss import softmax_with_cross_entropy
from . import tensor
from . import nn
from . import ops
from ... import compat as cpt
from ..data_feeder import check_variable_and_dtype, check_type, check_dtype
import math
import six
import numpy as np
from functools import reduce
from ..data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from paddle.utils import deprecated
from paddle import _C_ops

__all__ = [
    'prior_box',
    'density_prior_box',
    'multi_box_head',
    'bipartite_match',
    'target_assign',
    'detection_output',
    'ssd_loss',
    'rpn_target_assign',
    'retinanet_target_assign',
    'sigmoid_focal_loss',
    'anchor_generator',
    'roi_perspective_transform',
    'generate_proposal_labels',
    'generate_proposals',
    'generate_mask_labels',
    'iou_similarity',
    'box_coder',
    'polygon_box_transform',
    'yolov3_loss',
    'yolo_box',
    'box_clip',
    'multiclass_nms',
    'locality_aware_nms',
    'matrix_nms',
    'retinanet_detection_output',
    'distribute_fpn_proposals',
    'box_decoder_and_assign',
    'collect_fpn_proposals',
]


def retinanet_target_assign(bbox_pred,
                            cls_logits,
                            anchor_box,
                            anchor_var,
                            gt_boxes,
                            gt_labels,
                            is_crowd,
                            im_info,
                            num_classes=1,
                            positive_overlap=0.5,
                            negative_overlap=0.4):
    r"""
    **Target Assign Layer for the detector RetinaNet.**

    This OP finds out positive and negative samples from all anchors
    for training the detector `RetinaNet <https://arxiv.org/abs/1708.02002>`_ ,
    and assigns target labels for classification along with target locations for
    regression to each sample, then takes out the part belonging to positive and
    negative samples from category prediction( :attr:`cls_logits`) and location
    prediction( :attr:`bbox_pred`) which belong to all anchors.

    The searching principles for positive and negative samples are as followed:

    1. Anchors are assigned to ground-truth boxes when it has the highest IoU
    overlap with a ground-truth box.

    2. Anchors are assigned to ground-truth boxes when it has an IoU overlap
    higher than :attr:`positive_overlap` with any ground-truth box.

    3. Anchors are assigned to background when its IoU overlap is lower than
    :attr:`negative_overlap` for all ground-truth boxes.

    4. Anchors which do not meet the above conditions do not participate in
    the training process.

    Retinanet predicts a :math:`C`-vector for classification and a 4-vector for box
    regression for each anchor, hence the target label for each positive(or negative)
    sample is a :math:`C`-vector and the target locations for each positive sample
    is a 4-vector. As for a positive sample, if the category of its assigned
    ground-truth box is class :math:`i`, the corresponding entry in its length
    :math:`C` label vector is set to 1 and all other entries is set to 0, its box
    regression targets are computed as the offset between itself and its assigned
    ground-truth box. As for a negative sample, all entries in its length :math:`C`
    label vector are set to 0 and box regression targets are omitted because
    negative samples do not participate in the training process of location
    regression.

    After the assignment, the part belonging to positive and negative samples is
    taken out from category prediction( :attr:`cls_logits` ), and the part
    belonging to positive samples is taken out from location
    prediction( :attr:`bbox_pred` ).

    Args:
        bbox_pred(Variable): A 3-D Tensor with shape :math:`[N, M, 4]` represents
            the predicted locations of all anchors. :math:`N` is the batch size( the
            number of images in a mini-batch), :math:`M` is the number of all anchors
            of one image, and each anchor has 4 coordinate values. The data type of
            :attr:`bbox_pred` is float32 or float64.
        cls_logits(Variable): A 3-D Tensor with shape :math:`[N, M, C]` represents
            the predicted categories of all anchors. :math:`N` is the batch size,
            :math:`M` is the number of all anchors of one image, and :math:`C` is
            the number of categories (**Notice: excluding background**). The data type
            of :attr:`cls_logits` is float32 or float64.
        anchor_box(Variable): A 2-D Tensor with shape :math:`[M, 4]` represents
            the locations of all anchors. :math:`M` is the number of all anchors of
            one image, each anchor is represented as :math:`[xmin, ymin, xmax, ymax]`,
            :math:`[xmin, ymin]` is the left top coordinate of the anchor box,
            :math:`[xmax, ymax]` is the right bottom coordinate of the anchor box.
            The data type of :attr:`anchor_box` is float32 or float64. Please refer
            to the OP :ref:`api_fluid_layers_anchor_generator` 
            for the generation of :attr:`anchor_box`.
        anchor_var(Variable): A 2-D Tensor with shape :math:`[M,4]` represents the expanded 
            factors of anchor locations used in loss function. :math:`M` is number of
            all anchors of one image, each anchor possesses a 4-vector expanded factor.
            The data type of :attr:`anchor_var` is float32 or float64. Please refer
            to the OP :ref:`api_fluid_layers_anchor_generator`
            for the generation of :attr:`anchor_var`.
        gt_boxes(Variable): A 1-level 2-D LoDTensor with shape :math:`[G, 4]` represents
            locations of all ground-truth boxes. :math:`G` is the total number of
            all ground-truth boxes in a mini-batch, and each ground-truth box has 4
            coordinate values. The data type of :attr:`gt_boxes` is float32 or
            float64.
        gt_labels(variable): A 1-level 2-D LoDTensor with shape :math:`[G, 1]` represents
            categories of all ground-truth boxes, and the values are in the range of
            :math:`[1, C]`. :math:`G` is the total number of all ground-truth boxes
            in a mini-batch, and each ground-truth box has one category. The data type
            of :attr:`gt_labels` is int32.
        is_crowd(Variable): A 1-level 1-D LoDTensor with shape :math:`[G]` which
            indicates whether a ground-truth box is a crowd. If the value is 1, the
            corresponding box is a crowd, it is ignored during training. :math:`G` is
            the total number of all ground-truth boxes in a mini-batch. The data type
            of :attr:`is_crowd` is int32.
        im_info(Variable): A 2-D Tensor with shape [N, 3] represents the size
            information of input images. :math:`N` is the batch size, the size
            information of each image is a 3-vector which are the height and width
            of the network input along with the factor scaling the origin image to
            the network input. The data type of :attr:`im_info` is float32.
        num_classes(int32): The number of categories for classification, the default
            value is 1.
        positive_overlap(float32): Minimum overlap required between an anchor
            and ground-truth box for the anchor to be a positive sample, the default
            value is 0.5.
        negative_overlap(float32): Maximum overlap allowed between an anchor
            and ground-truth box for the anchor to be a negative sample, the default
            value is 0.4. :attr:`negative_overlap` should be less than or equal to
            :attr:`positive_overlap`, if not, the actual value of
            :attr:`positive_overlap` is :attr:`negative_overlap`.

    Returns:
        A tuple with 6 Variables:
        
        **predict_scores** (Variable): A 2-D Tensor with shape :math:`[F+B, C]` represents
        category prediction belonging to positive and negative samples. :math:`F`
        is the number of positive samples in a mini-batch, :math:`B` is the number
        of negative samples, and :math:`C` is the number of categories
        (**Notice: excluding background**). The data type of :attr:`predict_scores`
        is float32 or float64.

        **predict_location** (Variable): A 2-D Tensor with shape :math:`[F, 4]` represents
        location prediction belonging to positive samples. :math:`F` is the number
        of positive samples. :math:`F` is the number of positive samples, and each
        sample has 4 coordinate values. The data type of :attr:`predict_location`
        is float32 or float64.

        **target_label** (Variable): A 2-D Tensor with shape :math:`[F+B, 1]` represents
        target labels for classification belonging to positive and negative
        samples. :math:`F` is the number of positive samples, :math:`B` is the
        number of negative, and each sample has one target category. The data type
        of :attr:`target_label` is int32.

        **target_bbox** (Variable): A 2-D Tensor with shape :math:`[F, 4]` represents
        target locations for box regression belonging to positive samples.
        :math:`F` is the number of positive samples, and each sample has 4
        coordinate values. The data type of :attr:`target_bbox` is float32 or
        float64.

        **bbox_inside_weight** (Variable): A 2-D Tensor with shape :math:`[F, 4]`
        represents whether a positive sample is fake positive, if a positive
        sample is false positive, the corresponding entries in
        :attr:`bbox_inside_weight` are set 0, otherwise 1. :math:`F` is the number
        of total positive samples in a mini-batch, and each sample has 4
        coordinate values. The data type of :attr:`bbox_inside_weight` is float32
        or float64.

        **fg_num** (Variable): A 2-D Tensor with shape :math:`[N, 1]` represents the number
        of positive samples. :math:`N` is the batch size. **Notice: The number
        of positive samples is used as the denominator of later loss function,
        to avoid the condition that the denominator is zero, this OP has added 1
        to the actual number of positive samples of each image.** The data type of
        :attr:`fg_num` is int32.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          bbox_pred = fluid.data(name='bbox_pred', shape=[1, 100, 4],
                            dtype='float32')
          cls_logits = fluid.data(name='cls_logits', shape=[1, 100, 10],
                            dtype='float32')
          anchor_box = fluid.data(name='anchor_box', shape=[100, 4],
                            dtype='float32')
          anchor_var = fluid.data(name='anchor_var', shape=[100, 4],
                            dtype='float32')
          gt_boxes = fluid.data(name='gt_boxes', shape=[10, 4],
                            dtype='float32')
          gt_labels = fluid.data(name='gt_labels', shape=[10, 1],
                            dtype='int32')
          is_crowd = fluid.data(name='is_crowd', shape=[1],
                            dtype='int32')
          im_info = fluid.data(name='im_info', shape=[1, 3],
                            dtype='float32')
          score_pred, loc_pred, score_target, loc_target, bbox_inside_weight, fg_num = \\
                fluid.layers.retinanet_target_assign(bbox_pred, cls_logits, anchor_box,
                anchor_var, gt_boxes, gt_labels, is_crowd, im_info, 10)

    """

    check_variable_and_dtype(bbox_pred, 'bbox_pred', ['float32', 'float64'],
                             'retinanet_target_assign')
    check_variable_and_dtype(cls_logits, 'cls_logits', ['float32', 'float64'],
                             'retinanet_target_assign')
    check_variable_and_dtype(anchor_box, 'anchor_box', ['float32', 'float64'],
                             'retinanet_target_assign')
    check_variable_and_dtype(anchor_var, 'anchor_var', ['float32', 'float64'],
                             'retinanet_target_assign')
    check_variable_and_dtype(gt_boxes, 'gt_boxes', ['float32', 'float64'],
                             'retinanet_target_assign')
    check_variable_and_dtype(gt_labels, 'gt_labels', ['int32'],
                             'retinanet_target_assign')
    check_variable_and_dtype(is_crowd, 'is_crowd', ['int32'],
                             'retinanet_target_assign')
    check_variable_and_dtype(im_info, 'im_info', ['float32', 'float64'],
                             'retinanet_target_assign')

    helper = LayerHelper('retinanet_target_assign', **locals())
    # Assign target label to anchors
    loc_index = helper.create_variable_for_type_inference(dtype='int32')
    score_index = helper.create_variable_for_type_inference(dtype='int32')
    target_label = helper.create_variable_for_type_inference(dtype='int32')
    target_bbox = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype)
    bbox_inside_weight = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype)
    fg_num = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="retinanet_target_assign",
        inputs={
            'Anchor': anchor_box,
            'GtBoxes': gt_boxes,
            'GtLabels': gt_labels,
            'IsCrowd': is_crowd,
            'ImInfo': im_info
        },
        outputs={
            'LocationIndex': loc_index,
            'ScoreIndex': score_index,
            'TargetLabel': target_label,
            'TargetBBox': target_bbox,
            'BBoxInsideWeight': bbox_inside_weight,
            'ForegroundNumber': fg_num
        },
        attrs={
            'positive_overlap': positive_overlap,
            'negative_overlap': negative_overlap
        })

    loc_index.stop_gradient = True
    score_index.stop_gradient = True
    target_label.stop_gradient = True
    target_bbox.stop_gradient = True
    bbox_inside_weight.stop_gradient = True
    fg_num.stop_gradient = True

    cls_logits = nn.reshape(x=cls_logits, shape=(-1, num_classes))
    bbox_pred = nn.reshape(x=bbox_pred, shape=(-1, 4))
    predicted_cls_logits = nn.gather(cls_logits, score_index)
    predicted_bbox_pred = nn.gather(bbox_pred, loc_index)

    return predicted_cls_logits, predicted_bbox_pred, target_label, target_bbox, bbox_inside_weight, fg_num


def rpn_target_assign(bbox_pred,
                      cls_logits,
                      anchor_box,
                      anchor_var,
                      gt_boxes,
                      is_crowd,
                      im_info,
                      rpn_batch_size_per_im=256,
                      rpn_straddle_thresh=0.0,
                      rpn_fg_fraction=0.5,
                      rpn_positive_overlap=0.7,
                      rpn_negative_overlap=0.3,
                      use_random=True):
    """
    **Target Assign Layer for region proposal network (RPN) in Faster-RCNN detection.**

    This layer can be, for given the  Intersection-over-Union (IoU) overlap
    between anchors and ground truth boxes, to assign classification and
    regression targets to each each anchor, these target labels are used for
    train RPN. The classification targets is a binary class label (of being
    an object or not). Following the paper of Faster-RCNN, the positive labels
    are two kinds of anchors: (i) the anchor/anchors with the highest IoU
    overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap
    higher than rpn_positive_overlap(0.7) with any ground-truth box. Note
    that a single ground-truth box may assign positive labels to multiple
    anchors. A non-positive anchor is when its IoU ratio is lower than
    rpn_negative_overlap (0.3) for all ground-truth boxes. Anchors that are
    neither positive nor negative do not contribute to the training objective.
    The regression targets are the encoded ground-truth boxes associated with
    the positive anchors.

    Args:
        bbox_pred(Variable): A 3-D Tensor with shape [N, M, 4] represents the
            predicted locations of M bounding bboxes. N is the batch size,
            and each bounding box has four coordinate values and the layout
            is [xmin, ymin, xmax, ymax]. The data type can be float32 or float64.
        cls_logits(Variable): A 3-D Tensor with shape [N, M, 1] represents the
            predicted confidence predictions. N is the batch size, 1 is the
            frontground and background sigmoid, M is number of bounding boxes.
            The data type can be float32 or float64.
        anchor_box(Variable): A 2-D Tensor with shape [M, 4] holds M boxes,
            each box is represented as [xmin, ymin, xmax, ymax],
            [xmin, ymin] is the left top coordinate of the anchor box,
            if the input is image feature map, they are close to the origin
            of the coordinate system. [xmax, ymax] is the right bottom
            coordinate of the anchor box. The data type can be float32 or float64.
        anchor_var(Variable): A 2-D Tensor with shape [M,4] holds expanded 
            variances of anchors. The data type can be float32 or float64.
        gt_boxes (Variable): The ground-truth bounding boxes (bboxes) are a 2D
            LoDTensor with shape [Ng, 4], Ng is the total number of ground-truth
            bboxes of mini-batch input. The data type can be float32 or float64.
        is_crowd (Variable): A 1-D LoDTensor which indicates groud-truth is crowd.
                             The data type must be int32.
        im_info (Variable): A 2-D LoDTensor with shape [N, 3]. N is the batch size,
        3 is the height, width and scale.
        rpn_batch_size_per_im(int): Total number of RPN examples per image.
                                    The data type must be int32.
        rpn_straddle_thresh(float): Remove RPN anchors that go outside the image
            by straddle_thresh pixels. The data type must be float32.
        rpn_fg_fraction(float): Target fraction of RoI minibatch that is labeled
            foreground (i.e. class > 0), 0-th class is background. The data type must be float32.
        rpn_positive_overlap(float): Minimum overlap required between an anchor
            and ground-truth box for the (anchor, gt box) pair to be a positive
            example. The data type must be float32.
        rpn_negative_overlap(float): Maximum overlap allowed between an anchor
            and ground-truth box for the (anchor, gt box) pair to be a negative
            examples. The data type must be float32.

    Returns:
        tuple:
        A tuple(predicted_scores, predicted_location, target_label,
        target_bbox, bbox_inside_weight) is returned. The predicted_scores 
        and predicted_location is the predicted result of the RPN.
        The target_label and target_bbox is the ground truth,
        respectively. The predicted_location is a 2D Tensor with shape
        [F, 4], and the shape of target_bbox is same as the shape of
        the predicted_location, F is the number of the foreground
        anchors. The predicted_scores is a 2D Tensor with shape
        [F + B, 1], and the shape of target_label is same as the shape
        of the predicted_scores, B is the number of the background
        anchors, the F and B is depends on the input of this operator.
        Bbox_inside_weight represents whether the predicted loc is fake_fg
        or not and the shape is [F, 4].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            bbox_pred = fluid.data(name='bbox_pred', shape=[None, 4], dtype='float32')
            cls_logits = fluid.data(name='cls_logits', shape=[None, 1], dtype='float32')
            anchor_box = fluid.data(name='anchor_box', shape=[None, 4], dtype='float32')
            anchor_var = fluid.data(name='anchor_var', shape=[None, 4], dtype='float32')
            gt_boxes = fluid.data(name='gt_boxes', shape=[None, 4], dtype='float32')
            is_crowd = fluid.data(name='is_crowd', shape=[None], dtype='float32')
            im_info = fluid.data(name='im_infoss', shape=[None, 3], dtype='float32')
            loc, score, loc_target, score_target, inside_weight = fluid.layers.rpn_target_assign(
                bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info)

    """

    helper = LayerHelper('rpn_target_assign', **locals())

    check_variable_and_dtype(bbox_pred, 'bbox_pred', ['float32', 'float64'],
                             'rpn_target_assign')
    check_variable_and_dtype(cls_logits, 'cls_logits', ['float32', 'float64'],
                             'rpn_target_assign')
    check_variable_and_dtype(anchor_box, 'anchor_box', ['float32', 'float64'],
                             'rpn_target_assign')
    check_variable_and_dtype(anchor_var, 'anchor_var', ['float32', 'float64'],
                             'rpn_target_assign')
    check_variable_and_dtype(gt_boxes, 'gt_boxes', ['float32', 'float64'],
                             'rpn_target_assign')
    check_variable_and_dtype(is_crowd, 'is_crowd', ['int32'],
                             'rpn_target_assign')
    check_variable_and_dtype(im_info, 'im_info', ['float32', 'float64'],
                             'rpn_target_assign')

    # Assign target label to anchors
    loc_index = helper.create_variable_for_type_inference(dtype='int32')
    score_index = helper.create_variable_for_type_inference(dtype='int32')
    target_label = helper.create_variable_for_type_inference(dtype='int32')
    target_bbox = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype)
    bbox_inside_weight = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype)
    helper.append_op(
        type="rpn_target_assign",
        inputs={
            'Anchor': anchor_box,
            'GtBoxes': gt_boxes,
            'IsCrowd': is_crowd,
            'ImInfo': im_info
        },
        outputs={
            'LocationIndex': loc_index,
            'ScoreIndex': score_index,
            'TargetLabel': target_label,
            'TargetBBox': target_bbox,
            'BBoxInsideWeight': bbox_inside_weight
        },
        attrs={
            'rpn_batch_size_per_im': rpn_batch_size_per_im,
            'rpn_straddle_thresh': rpn_straddle_thresh,
            'rpn_positive_overlap': rpn_positive_overlap,
            'rpn_negative_overlap': rpn_negative_overlap,
            'rpn_fg_fraction': rpn_fg_fraction,
            'use_random': use_random
        })

    loc_index.stop_gradient = True
    score_index.stop_gradient = True
    target_label.stop_gradient = True
    target_bbox.stop_gradient = True
    bbox_inside_weight.stop_gradient = True

    cls_logits = nn.reshape(x=cls_logits, shape=(-1, 1))
    bbox_pred = nn.reshape(x=bbox_pred, shape=(-1, 4))
    predicted_cls_logits = nn.gather(cls_logits, score_index)
    predicted_bbox_pred = nn.gather(bbox_pred, loc_index)

    return predicted_cls_logits, predicted_bbox_pred, target_label, target_bbox, bbox_inside_weight


def sigmoid_focal_loss(x, label, fg_num, gamma=2.0, alpha=0.25):
    r"""
	:alias_main: paddle.nn.functional.sigmoid_focal_loss
	:alias: paddle.nn.functional.sigmoid_focal_loss,paddle.nn.functional.loss.sigmoid_focal_loss
	:old_api: paddle.fluid.layers.sigmoid_focal_loss

    **Sigmoid Focal Loss Operator.**

    `Focal Loss <https://arxiv.org/abs/1708.02002>`_ is used to address the foreground-background
    class imbalance existed on the training phase of many computer vision tasks. This OP computes
    the sigmoid value for each element in the input tensor :attr:`x`, after which focal loss is
    measured between the sigmoid value and target label. 

    The focal loss is given as followed:

    .. math::
  
        \\mathop{loss_{i,\\,j}}\\limits_{i\\in\\mathbb{[0,\\,N-1]},\\,j\\in\\mathbb{[0,\\,C-1]}}=\\left\\{
        \\begin{array}{rcl}
        - \\frac{1}{fg\_num} * \\alpha * {(1 - \\sigma(x_{i,\\,j}))}^{\\gamma} * \\log(\\sigma(x_{i,\\,j})) & & {(j +1) = label_{i,\\,0}} \\\\
        - \\frac{1}{fg\_num} * (1 - \\alpha) * {\sigma(x_{i,\\,j})}^{ \\gamma} * \\log(1 - \\sigma(x_{i,\\,j})) & & {(j +1)!= label_{i,\\,0}}
        \\end{array} \\right.


    We know that
    
    .. math::
        \\sigma(x_j) = \\frac{1}{1 + \\exp(-x_j)}


    Args:
        x(Variable): A 2-D tensor with shape :math:`[N, C]` represents the predicted categories of
            all samples. :math:`N` is the number of all samples responsible for optimization in
            a mini-batch, for example, samples are anchor boxes for object detection and :math:`N`
            is the total number of positive and negative samples in a mini-batch; Samples are images
            for image classification and :math:`N` is the number of images in a mini-batch. :math:`C`
            is the number of classes (**Notice: excluding background**). The data type of :attr:`x` is
            float32 or float64.
        label(Variable): A 2-D tensor with shape :math:`[N, 1]` represents the target labels for
            classification. :math:`N` is the number of all samples responsible for optimization in a
            mini-batch, each sample has one target category. The values for positive samples are in the
            range of :math:`[1, C]`, and the values for negative samples are 0. The data type of :attr:`label`
            is int32.
        fg_num(Variable): A 1-D tensor with shape [1] represents the number of positive samples in a
            mini-batch, which should be obtained before this OP. The data type of :attr:`fg_num` is int32.
        gamma(int|float): Hyper-parameter to balance the easy and hard examples. Default value is
            set to 2.0.
        alpha(int|float): Hyper-parameter to balance the positive and negative example. Default value
            is set to 0.25.

    Returns:
        Variable(the data type is float32 or float64): 
            A 2-D tensor with shape :math:`[N, C]`, which is the focal loss of each element in the input
            tensor :attr:`x`.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            
            num_classes = 10  # exclude background
            image_width = 16
            image_height = 16
            batch_size = 32
            max_iter = 20
            
            
            def gen_train_data():
                x_data = np.random.uniform(0, 255, (batch_size, 3, image_height,
                                                    image_width)).astype('float64')
                label_data = np.random.randint(0, num_classes,
                                               (batch_size, 1)).astype('int32')
                return {"x": x_data, "label": label_data}
            
            
            def get_focal_loss(pred, label, fg_num, num_classes):
                pred = fluid.layers.reshape(pred, [-1, num_classes])
                label = fluid.layers.reshape(label, [-1, 1])
                label.stop_gradient = True
                loss = fluid.layers.sigmoid_focal_loss(
                    pred, label, fg_num, gamma=2.0, alpha=0.25)
                loss = fluid.layers.reduce_sum(loss)
                return loss
            
            
            def build_model(mode='train'):
                x = fluid.data(name="x", shape=[-1, 3, -1, -1], dtype='float64')
                output = fluid.layers.pool2d(input=x, pool_type='avg', global_pooling=True)
                output = fluid.layers.fc(
                    input=output,
                    size=num_classes,
                    # Notice: size is set to be the number of target classes (excluding backgorund)
                    # because sigmoid activation will be done in the sigmoid_focal_loss op.
                    act=None)
                if mode == 'train':
                    label = fluid.data(name="label", shape=[-1, 1], dtype='int32')
                    # Obtain the fg_num needed by the sigmoid_focal_loss op:
                    # 0 in label represents background, >=1 in label represents foreground,
                    # find the elements in label which are greater or equal than 1, then
                    # computed the numbers of these elements.
                    data = fluid.layers.fill_constant(shape=[1], value=1, dtype='int32')
                    fg_label = fluid.layers.greater_equal(label, data)
                    fg_label = fluid.layers.cast(fg_label, dtype='int32')
                    fg_num = fluid.layers.reduce_sum(fg_label)
                    fg_num.stop_gradient = True
                    avg_loss = get_focal_loss(output, label, fg_num, num_classes)
                    return avg_loss
                else:
                    # During evaluating or testing phase,
                    # output of the final fc layer should be connected to a sigmoid layer.
                    pred = fluid.layers.sigmoid(output)
                    return pred
            
            
            loss = build_model('train')
            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            moment_optimizer.minimize(loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(max_iter):
                outs = exe.run(feed=gen_train_data(), fetch_list=[loss.name])
                print(outs)
    """

    check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                             'sigmoid_focal_loss')
    check_variable_and_dtype(label, 'label', ['int32'], 'sigmoid_focal_loss')
    check_variable_and_dtype(fg_num, 'fg_num', ['int32'], 'sigmoid_focal_loss')

    helper = LayerHelper("sigmoid_focal_loss", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="sigmoid_focal_loss",
        inputs={"X": x,
                "Label": label,
                "FgNum": fg_num},
        attrs={"gamma": gamma,
               'alpha': alpha},
        outputs={"Out": out})
    return out


def detection_output(loc,
                     scores,
                     prior_box,
                     prior_box_var,
                     background_label=0,
                     nms_threshold=0.3,
                     nms_top_k=400,
                     keep_top_k=200,
                     score_threshold=0.01,
                     nms_eta=1.0,
                     return_index=False):
    """

    Given the regression locations, classification confidences and prior boxes,
    calculate the detection outputs by performing following steps:

    1. Decode input bounding box predictions according to the prior boxes and
       regression locations.
    2. Get the final detection results by applying multi-class non maximum
       suppression (NMS).

    Please note, this operation doesn't clip the final output bounding boxes
    to the image window.

    Args:
        loc(Variable): A 3-D Tensor with shape [N, M, 4] represents the
            predicted locations of M bounding bboxes. Data type should be
            float32 or float64. N is the batch size,
            and each bounding box has four coordinate values and the layout
            is [xmin, ymin, xmax, ymax].
        scores(Variable): A 3-D Tensor with shape [N, M, C] represents the
            predicted confidence predictions. Data type should be float32
            or float64. N is the batch size, C is the
            class number, M is number of bounding boxes.
        prior_box(Variable): A 2-D Tensor with shape [M, 4] holds M boxes,
            each box is represented as [xmin, ymin, xmax, ymax]. Data type
            should be float32 or float64.
        prior_box_var(Variable): A 2-D Tensor with shape [M, 4] holds M group
            of variance. Data type should be float32 or float64.
        background_label(int): The index of background label,
            the background label will be ignored. If set to -1, then all
            categories will be considered. Default: 0.
        nms_threshold(float): The threshold to be used in NMS. Default: 0.3.
        nms_top_k(int): Maximum number of detections to be kept according
            to the confidences after filtering detections based on
            score_threshold and before NMS. Default: 400.
        keep_top_k(int): Number of total bboxes to be kept per image after
            NMS step. -1 means keeping all bboxes after NMS step. Default: 200.
        score_threshold(float): Threshold to filter out bounding boxes with
            low confidence score. If not provided, consider all boxes.
            Default: 0.01.
        nms_eta(float): The parameter for adaptive NMS. It works only when the
            value is less than 1.0. Default: 1.0.
        return_index(bool): Whether return selected index. Default: False

    Returns:

        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned. 

        Out (Variable): The detection outputs is a LoDTensor with shape [No, 6].
        Data type is the same as input (loc). Each row has six values:
        [label, confidence, xmin, ymin, xmax, ymax]. `No` is
        the total number of detections in this mini-batch. For each instance,
        the offsets in first dimension are called LoD, the offset number is
        N + 1, N is the batch size. The i-th image has `LoD[i + 1] - LoD[i]`
        detected results, if it is 0, the i-th image has no detected results.

        Index (Variable): Only return when return_index is True. A 2-D LoDTensor
        with shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle

            paddle.enable_static()

            pb = fluid.data(name='prior_box', shape=[10, 4], dtype='float32')
            pbv = fluid.data(name='prior_box_var', shape=[10, 4], dtype='float32')
            loc = fluid.data(name='target_box', shape=[2, 21, 4], dtype='float32')
            scores = fluid.data(name='scores', shape=[2, 21, 10], dtype='float32')
            nmsed_outs, index = fluid.layers.detection_output(scores=scores,
                                       loc=loc,
                                       prior_box=pb,
                                       prior_box_var=pbv,
                                       return_index=True)
    """
    helper = LayerHelper("detection_output", **locals())
    decoded_box = box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=loc,
        code_type='decode_center_size')
    scores = nn.softmax(input=scores)
    scores = nn.transpose(scores, perm=[0, 2, 1])
    scores.stop_gradient = True
    nmsed_outs = helper.create_variable_for_type_inference(
        dtype=decoded_box.dtype)
    if return_index:
        index = helper.create_variable_for_type_inference(dtype='int')
        helper.append_op(
            type="multiclass_nms2",
            inputs={'Scores': scores,
                    'BBoxes': decoded_box},
            outputs={'Out': nmsed_outs,
                     'Index': index},
            attrs={
                'background_label': 0,
                'nms_threshold': nms_threshold,
                'nms_top_k': nms_top_k,
                'keep_top_k': keep_top_k,
                'score_threshold': score_threshold,
                'nms_eta': 1.0,
            })
        index.stop_gradient = True
    else:
        helper.append_op(
            type="multiclass_nms",
            inputs={'Scores': scores,
                    'BBoxes': decoded_box},
            outputs={'Out': nmsed_outs},
            attrs={
                'background_label': 0,
                'nms_threshold': nms_threshold,
                'nms_top_k': nms_top_k,
                'keep_top_k': keep_top_k,
                'score_threshold': score_threshold,
                'nms_eta': 1.0,
            })
    nmsed_outs.stop_gradient = True
    if return_index:
        return nmsed_outs, index
    return nmsed_outs


@templatedoc()
def iou_similarity(x, y, box_normalized=True, name=None):
    """
	:alias_main: paddle.nn.functional.iou_similarity
	:alias: paddle.nn.functional.iou_similarity,paddle.nn.functional.loss.iou_similarity
	:old_api: paddle.fluid.layers.iou_similarity

    ${comment}

    Args:
        x (Variable): ${x_comment}.The data type is float32 or float64.
        y (Variable): ${y_comment}.The data type is float32 or float64.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
    Returns:
        Variable: ${out_comment}.The data type is same with x.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            use_gpu = False
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)

            x = fluid.data(name='x', shape=[None, 4], dtype='float32')
            y = fluid.data(name='y', shape=[None, 4], dtype='float32')
            iou = fluid.layers.iou_similarity(x=x, y=y)

            exe.run(fluid.default_startup_program())
            test_program = fluid.default_main_program().clone(for_test=True)

            [out_iou] = exe.run(test_program,
                    fetch_list=iou,
                    feed={'x': np.array([[0.5, 0.5, 2.0, 2.0],
                                         [0., 0., 1.0, 1.0]]).astype('float32'),
                          'y': np.array([[1.0, 1.0, 2.5, 2.5]]).astype('float32')})
            # out_iou is [[0.2857143],
            #             [0.       ]] with shape: [2, 1]
    """
    helper = LayerHelper("iou_similarity", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="iou_similarity",
        inputs={"X": x,
                "Y": y},
        attrs={"box_normalized": box_normalized},
        outputs={"Out": out})
    return out


@templatedoc()
def box_coder(prior_box,
              prior_box_var,
              target_box,
              code_type="encode_center_size",
              box_normalized=True,
              name=None,
              axis=0):
    r"""

    **Box Coder Layer**

    Encode/Decode the target bounding box with the priorbox information.
    
    The Encoding schema described below:

    .. math::

        ox = (tx - px) / pw / pxv

        oy = (ty - py) / ph / pyv

        ow = \log(\abs(tw / pw)) / pwv 

        oh = \log(\abs(th / ph)) / phv 

    The Decoding schema described below:
    
    .. math::
  
        ox = (pw * pxv * tx * + px) - tw / 2

        oy = (ph * pyv * ty * + py) - th / 2

        ow = \exp(pwv * tw) * pw + tw / 2

        oh = \exp(phv * th) * ph + th / 2   

    where `tx`, `ty`, `tw`, `th` denote the target box's center coordinates, 
    width and height respectively. Similarly, `px`, `py`, `pw`, `ph` denote 
    the priorbox's (anchor) center coordinates, width and height. `pxv`, 
    `pyv`, `pwv`, `phv` denote the variance of the priorbox and `ox`, `oy`, 
    `ow`, `oh` denote the encoded/decoded coordinates, width and height. 

    During Box Decoding, two modes for broadcast are supported. Say target 
    box has shape [N, M, 4], and the shape of prior box can be [N, 4] or 
    [M, 4]. Then prior box will broadcast to target box along the 
    assigned axis. 

    Args:
        prior_box(Variable): Box list prior_box is a 2-D Tensor with shape 
            [M, 4] holds M boxes and data type is float32 or float64. Each box
            is represented as [xmin, ymin, xmax, ymax], [xmin, ymin] is the 
            left top coordinate of the anchor box, if the input is image feature
            map, they are close to the origin of the coordinate system. 
            [xmax, ymax] is the right bottom coordinate of the anchor box.       
        prior_box_var(List|Variable|None): prior_box_var supports three types 
            of input. One is variable with shape [M, 4] which holds M group and 
            data type is float32 or float64. The second is list consist of 
            4 elements shared by all boxes and data type is float32 or float64. 
            Other is None and not involved in calculation. 
        target_box(Variable): This input can be a 2-D LoDTensor with shape 
            [N, 4] when code_type is 'encode_center_size'. This input also can 
            be a 3-D Tensor with shape [N, M, 4] when code_type is 
            'decode_center_size'. Each box is represented as 
            [xmin, ymin, xmax, ymax]. The data type is float32 or float64. 
            This tensor can contain LoD information to represent a batch of inputs. 
        code_type(str): The code type used with the target box. It can be
            `encode_center_size` or `decode_center_size`. `encode_center_size` 
            by default.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 
        axis(int): Which axis in PriorBox to broadcast for box decode, 
            for example, if axis is 0 and TargetBox has shape [N, M, 4] and 
            PriorBox has shape [M, 4], then PriorBox will broadcast to [N, M, 4]
            for decoding. It is only valid when code type is 
            `decode_center_size`. Set 0 by default. 

    Returns:
        Variable:

        output_box(Variable): When code_type is 'encode_center_size', the 
        output tensor of box_coder_op with shape [N, M, 4] representing the 
        result of N target boxes encoded with M Prior boxes and variances. 
        When code_type is 'decode_center_size', N represents the batch size 
        and M represents the number of decoded boxes.

    Examples:
 
        .. code-block:: python
 
            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            # For encode
            prior_box_encode = fluid.data(name='prior_box_encode',
                                  shape=[512, 4],
                                  dtype='float32')
            target_box_encode = fluid.data(name='target_box_encode',
                                   shape=[81, 4],
                                   dtype='float32')
            output_encode = fluid.layers.box_coder(prior_box=prior_box_encode,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box_encode,
                                    code_type="encode_center_size")
            # For decode
            prior_box_decode = fluid.data(name='prior_box_decode',
                                  shape=[512, 4],
                                  dtype='float32')
            target_box_decode = fluid.data(name='target_box_decode',
                                   shape=[512, 81, 4],
                                   dtype='float32')
            output_decode = fluid.layers.box_coder(prior_box=prior_box_decode,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box_decode,
                                    code_type="decode_center_size",
                                    box_normalized=False,
                                    axis=1)
    """
    check_variable_and_dtype(prior_box, 'prior_box', ['float32', 'float64'],
                             'box_coder')
    check_variable_and_dtype(target_box, 'target_box', ['float32', 'float64'],
                             'box_coder')
    helper = LayerHelper("box_coder", **locals())

    output_box = helper.create_variable_for_type_inference(
        dtype=prior_box.dtype)

    inputs = {"PriorBox": prior_box, "TargetBox": target_box}
    attrs = {
        "code_type": code_type,
        "box_normalized": box_normalized,
        "axis": axis
    }
    if isinstance(prior_box_var, Variable):
        inputs['PriorBoxVar'] = prior_box_var
    elif isinstance(prior_box_var, list):
        attrs['variance'] = prior_box_var
    else:
        raise TypeError("Input variance of box_coder must be Variable or lisz")
    helper.append_op(
        type="box_coder",
        inputs=inputs,
        attrs=attrs,
        outputs={"OutputBox": output_box})
    return output_box


@templatedoc()
def polygon_box_transform(input, name=None):
    """
    ${comment}

    Args:
        input(Variable): The input with shape [batch_size, geometry_channels, height, width].
                         A Tensor with type float32, float64.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.

    Returns:
        Variable: The output with the same shape as input. A Tensor with type float32, float64.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            input = fluid.data(name='input', shape=[4, 10, 5, 5], dtype='float32')
            out = fluid.layers.polygon_box_transform(input)
    """
    check_variable_and_dtype(input, "input", ['float32', 'float64'],
                             'polygon_box_transform')
    helper = LayerHelper("polygon_box_transform", **locals())
    output = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type="polygon_box_transform",
        inputs={"Input": input},
        attrs={},
        outputs={"Output": output})
    return output


@deprecated(since="2.0.0", update_to="paddle.vision.ops.yolo_loss")
@templatedoc(op_type="yolov3_loss")
def yolov3_loss(x,
                gt_box,
                gt_label,
                anchors,
                anchor_mask,
                class_num,
                ignore_thresh,
                downsample_ratio,
                gt_score=None,
                use_label_smooth=True,
                name=None,
                scale_x_y=1.):
    """

    ${comment}

    Args:
        x (Variable): ${x_comment}The data type is float32 or float64. 
        gt_box (Variable): groud truth boxes, should be in shape of [N, B, 4],
                          in the third dimension, x, y, w, h should be stored. 
                          x,y is the center coordinate of boxes, w, h are the
                          width and height, x, y, w, h should be divided by 
                          input image height to scale to [0, 1].
                          N is the batch number and B is the max box number in 
                          an image.The data type is float32 or float64. 
        gt_label (Variable): class id of ground truth boxes, should be in shape
                            of [N, B].The data type is int32. 
        anchors (list|tuple): ${anchors_comment}
        anchor_mask (list|tuple): ${anchor_mask_comment}
        class_num (int): ${class_num_comment}
        ignore_thresh (float): ${ignore_thresh_comment}
        downsample_ratio (int): ${downsample_ratio_comment}
        name (string): The default value is None.  Normally there is no need 
                       for user to set this property.  For more information, 
                       please refer to :ref:`api_guide_Name`
        gt_score (Variable): mixup score of ground truth boxes, should be in shape
                            of [N, B]. Default None.
        use_label_smooth (bool): ${use_label_smooth_comment}
        scale_x_y (float): ${scale_x_y_comment}

    Returns:
        Variable: A 1-D tensor with shape [N], the value of yolov3 loss

    Raises:
        TypeError: Input x of yolov3_loss must be Variable
        TypeError: Input gtbox of yolov3_loss must be Variable
        TypeError: Input gtlabel of yolov3_loss must be Variable
        TypeError: Input gtscore of yolov3_loss must be None or Variable
        TypeError: Attr anchors of yolov3_loss must be list or tuple
        TypeError: Attr class_num of yolov3_loss must be an integer
        TypeError: Attr ignore_thresh of yolov3_loss must be a float number
        TypeError: Attr use_label_smooth of yolov3_loss must be a bool value

    Examples:
      .. code-block:: python

          import paddle.fluid as fluid
          import paddle
          paddle.enable_static()
          x = fluid.data(name='x', shape=[None, 255, 13, 13], dtype='float32')
          gt_box = fluid.data(name='gt_box', shape=[None, 6, 4], dtype='float32')
          gt_label = fluid.data(name='gt_label', shape=[None, 6], dtype='int32')
          gt_score = fluid.data(name='gt_score', shape=[None, 6], dtype='float32')
          anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
          anchor_mask = [0, 1, 2]
          loss = fluid.layers.yolov3_loss(x=x, gt_box=gt_box, gt_label=gt_label,
                                          gt_score=gt_score, anchors=anchors, 
                                          anchor_mask=anchor_mask, class_num=80,
                                          ignore_thresh=0.7, downsample_ratio=32)
    """
    helper = LayerHelper('yolov3_loss', **locals())

    if not isinstance(x, Variable):
        raise TypeError("Input x of yolov3_loss must be Variable")
    if not isinstance(gt_box, Variable):
        raise TypeError("Input gtbox of yolov3_loss must be Variable")
    if not isinstance(gt_label, Variable):
        raise TypeError("Input gtlabel of yolov3_loss must be Variable")
    if gt_score is not None and not isinstance(gt_score, Variable):
        raise TypeError("Input gtscore of yolov3_loss must be Variable")
    if not isinstance(anchors, list) and not isinstance(anchors, tuple):
        raise TypeError("Attr anchors of yolov3_loss must be list or tuple")
    if not isinstance(anchor_mask, list) and not isinstance(anchor_mask, tuple):
        raise TypeError("Attr anchor_mask of yolov3_loss must be list or tuple")
    if not isinstance(class_num, int):
        raise TypeError("Attr class_num of yolov3_loss must be an integer")
    if not isinstance(ignore_thresh, float):
        raise TypeError(
            "Attr ignore_thresh of yolov3_loss must be a float number")
    if not isinstance(use_label_smooth, bool):
        raise TypeError(
            "Attr use_label_smooth of yolov3_loss must be a bool value")

    loss = helper.create_variable_for_type_inference(dtype=x.dtype)

    objectness_mask = helper.create_variable_for_type_inference(dtype='int32')
    gt_match_mask = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {
        "X": x,
        "GTBox": gt_box,
        "GTLabel": gt_label,
    }
    if gt_score is not None:
        inputs["GTScore"] = gt_score

    attrs = {
        "anchors": anchors,
        "anchor_mask": anchor_mask,
        "class_num": class_num,
        "ignore_thresh": ignore_thresh,
        "downsample_ratio": downsample_ratio,
        "use_label_smooth": use_label_smooth,
        "scale_x_y": scale_x_y,
    }

    helper.append_op(
        type='yolov3_loss',
        inputs=inputs,
        outputs={
            'Loss': loss,
            'ObjectnessMask': objectness_mask,
            'GTMatchMask': gt_match_mask
        },
        attrs=attrs)
    return loss


@deprecated(since="2.0.0", update_to="paddle.vision.ops.yolo_box")
@templatedoc(op_type="yolo_box")
def yolo_box(x,
             img_size,
             anchors,
             class_num,
             conf_thresh,
             downsample_ratio,
             clip_bbox=True,
             name=None,
             scale_x_y=1.,
             iou_aware=False,
             iou_aware_factor=0.5):
    """

    ${comment}

    Args:
        x (Variable): ${x_comment} The data type is float32 or float64. 
        img_size (Variable): ${img_size_comment} The data type is int32. 
        anchors (list|tuple): ${anchors_comment}
        class_num (int): ${class_num_comment}
        conf_thresh (float): ${conf_thresh_comment}
        downsample_ratio (int): ${downsample_ratio_comment}
        clip_bbox (bool): ${clip_bbox_comment}
        scale_x_y (float): ${scale_x_y_comment}
        name (string): The default value is None.  Normally there is no need 
                       for user to set this property.  For more information, 
                       please refer to :ref:`api_guide_Name`
        iou_aware (bool): ${iou_aware_comment}
        iou_aware_factor (float): ${iou_aware_factor_comment}

    Returns:
        Variable: A 3-D tensor with shape [N, M, 4], the coordinates of boxes,
        and a 3-D tensor with shape [N, M, :attr:`class_num`], the classification 
        scores of boxes.

    Raises:
        TypeError: Input x of yolov_box must be Variable
        TypeError: Attr anchors of yolo box must be list or tuple
        TypeError: Attr class_num of yolo box must be an integer
        TypeError: Attr conf_thresh of yolo box must be a float number

    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import paddle
        paddle.enable_static()
        x = fluid.data(name='x', shape=[None, 255, 13, 13], dtype='float32')
        img_size = fluid.data(name='img_size',shape=[None, 2],dtype='int64')
        anchors = [10, 13, 16, 30, 33, 23]
        boxes,scores = fluid.layers.yolo_box(x=x, img_size=img_size, class_num=80, anchors=anchors, 
                                        conf_thresh=0.01, downsample_ratio=32)
    """
    helper = LayerHelper('yolo_box', **locals())

    if not isinstance(x, Variable):
        raise TypeError("Input x of yolo_box must be Variable")
    if not isinstance(img_size, Variable):
        raise TypeError("Input img_size of yolo_box must be Variable")
    if not isinstance(anchors, list) and not isinstance(anchors, tuple):
        raise TypeError("Attr anchors of yolo_box must be list or tuple")
    if not isinstance(class_num, int):
        raise TypeError("Attr class_num of yolo_box must be an integer")
    if not isinstance(conf_thresh, float):
        raise TypeError("Attr ignore_thresh of yolo_box must be a float number")

    boxes = helper.create_variable_for_type_inference(dtype=x.dtype)
    scores = helper.create_variable_for_type_inference(dtype=x.dtype)

    attrs = {
        "anchors": anchors,
        "class_num": class_num,
        "conf_thresh": conf_thresh,
        "downsample_ratio": downsample_ratio,
        "clip_bbox": clip_bbox,
        "scale_x_y": scale_x_y,
        "iou_aware": iou_aware,
        "iou_aware_factor": iou_aware_factor
    }

    helper.append_op(
        type='yolo_box',
        inputs={
            "X": x,
            "ImgSize": img_size,
        },
        outputs={
            'Boxes': boxes,
            'Scores': scores,
        },
        attrs=attrs)
    return boxes, scores


@templatedoc()
def detection_map(detect_res,
                  label,
                  class_num,
                  background_label=0,
                  overlap_threshold=0.3,
                  evaluate_difficult=True,
                  has_state=None,
                  input_states=None,
                  out_states=None,
                  ap_version='integral'):
    """
    ${comment}

    Args:
        detect_res: ${detect_res_comment}
        label:  ${label_comment}
        class_num: ${class_num_comment}
        background_label: ${background_label_comment}
        overlap_threshold: ${overlap_threshold_comment}
        evaluate_difficult: ${evaluate_difficult_comment}
        has_state: ${has_state_comment}
        input_states: (tuple|None) If not None, It contains 3 elements:
            (1) pos_count ${pos_count_comment}.
            (2) true_pos ${true_pos_comment}.
            (3) false_pos ${false_pos_comment}.
        out_states: (tuple|None) If not None, it contains 3 elements.
            (1) accum_pos_count ${accum_pos_count_comment}.
            (2) accum_true_pos ${accum_true_pos_comment}.
            (3) accum_false_pos ${accum_false_pos_comment}.
        ap_version: ${ap_type_comment}

    Returns:
        ${map_comment}


    Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            from fluid.layers import detection
            detect_res = fluid.data(
                name='detect_res',
                shape=[10, 6],
                dtype='float32')
            label = fluid.data(
                name='label',
                shape=[10, 6],
                dtype='float32')

            map_out = detection.detection_map(detect_res, label, 21)
    """
    helper = LayerHelper("detection_map", **locals())

    def __create_var(type):
        return helper.create_variable_for_type_inference(dtype=type)

    map_out = __create_var('float32')
    accum_pos_count_out = out_states[
        0] if out_states is not None else __create_var('int32')
    accum_true_pos_out = out_states[
        1] if out_states is not None else __create_var('float32')
    accum_false_pos_out = out_states[
        2] if out_states is not None else __create_var('float32')

    pos_count = input_states[0] if input_states is not None else None
    true_pos = input_states[1] if input_states is not None else None
    false_pos = input_states[2] if input_states is not None else None

    helper.append_op(
        type="detection_map",
        inputs={
            'Label': label,
            'DetectRes': detect_res,
            'HasState': has_state,
            'PosCount': pos_count,
            'TruePos': true_pos,
            'FalsePos': false_pos
        },
        outputs={
            'MAP': map_out,
            'AccumPosCount': accum_pos_count_out,
            'AccumTruePos': accum_true_pos_out,
            'AccumFalsePos': accum_false_pos_out
        },
        attrs={
            'overlap_threshold': overlap_threshold,
            'evaluate_difficult': evaluate_difficult,
            'ap_type': ap_version,
            'class_num': class_num,
        })
    return map_out


def bipartite_match(dist_matrix,
                    match_type=None,
                    dist_threshold=None,
                    name=None):
    """

    This operator implements a greedy bipartite matching algorithm, which is
    used to obtain the matching with the maximum distance based on the input
    distance matrix. For input 2D matrix, the bipartite matching algorithm can
    find the matched column for each row (matched means the largest distance),
    also can find the matched row for each column. And this operator only
    calculate matched indices from column to row. For each instance,
    the number of matched indices is the column number of the input distance
    matrix. **The OP only supports CPU**.

    There are two outputs, matched indices and distance.
    A simple description, this algorithm matched the best (maximum distance)
    row entity to the column entity and the matched indices are not duplicated
    in each row of ColToRowMatchIndices. If the column entity is not matched
    any row entity, set -1 in ColToRowMatchIndices.

    NOTE: the input DistMat can be LoDTensor (with LoD) or Tensor.
    If LoDTensor with LoD, the height of ColToRowMatchIndices is batch size.
    If Tensor, the height of ColToRowMatchIndices is 1.

    NOTE: This API is a very low level API. It is used by :code:`ssd_loss`
    layer. Please consider to use :code:`ssd_loss` instead.

    Args:
        dist_matrix(Variable): This input is a 2-D LoDTensor with shape
            [K, M]. The data type is float32 or float64. It is pair-wise 
            distance matrix between the entities represented by each row and 
            each column. For example, assumed one entity is A with shape [K], 
            another entity is B with shape [M]. The dist_matrix[i][j] is the 
            distance between A[i] and B[j]. The bigger the distance is, the 
            better matching the pairs are. NOTE: This tensor can contain LoD 
            information to represent a batch of inputs. One instance of this 
            batch can contain different numbers of entities.
        match_type(str, optional): The type of matching method, should be
           'bipartite' or 'per_prediction'. None ('bipartite') by default.
        dist_threshold(float32, optional): If `match_type` is 'per_prediction',
            this threshold is to determine the extra matching bboxes based
            on the maximum distance, 0.5 by default.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.
 
    Returns:
        Tuple:

        matched_indices(Variable): A 2-D Tensor with shape [N, M]. The data
        type is int32. N is the batch size. If match_indices[i][j] is -1, it
        means B[j] does not match any entity in i-th instance.
        Otherwise, it means B[j] is matched to row
        match_indices[i][j] in i-th instance. The row number of
        i-th instance is saved in match_indices[i][j].

        matched_distance(Variable): A 2-D Tensor with shape [N, M]. The data
        type is float32. N is batch size. If match_indices[i][j] is -1,
        match_distance[i][j] is also -1.0. Otherwise, assumed
        match_distance[i][j] = d, and the row offsets of each instance
        are called LoD. Then match_distance[i][j] =
        dist_matrix[d+LoD[i]][j].

    Examples:

        >>> import paddle.fluid as fluid
        >>> x = fluid.data(name='x', shape=[None, 4], dtype='float32')
        >>> y = fluid.data(name='y', shape=[None, 4], dtype='float32')
        >>> iou = fluid.layers.iou_similarity(x=x, y=y)
        >>> matched_indices, matched_dist = fluid.layers.bipartite_match(iou)
    """
    helper = LayerHelper('bipartite_match', **locals())
    match_indices = helper.create_variable_for_type_inference(dtype='int32')
    match_distance = helper.create_variable_for_type_inference(
        dtype=dist_matrix.dtype)
    helper.append_op(
        type='bipartite_match',
        inputs={'DistMat': dist_matrix},
        attrs={
            'match_type': match_type,
            'dist_threshold': dist_threshold,
        },
        outputs={
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_distance
        })
    return match_indices, match_distance


def target_assign(input,
                  matched_indices,
                  negative_indices=None,
                  mismatch_value=None,
                  name=None):
    """

    This operator can be, for given the target bounding boxes or labels,
    to assign classification and regression targets to each prediction as well as
    weights to prediction. The weights is used to specify which prediction would
    not contribute to training loss.

    For each instance, the output `out` and`out_weight` are assigned based on
    `match_indices` and `negative_indices`.
    Assumed that the row offset for each instance in `input` is called lod,
    this operator assigns classification/regression targets by performing the
    following steps:

    1. Assigning all outputs based on `match_indices`:

    .. code-block:: text

        If id = match_indices[i][j] > 0,

            out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
            out_weight[i][j] = 1.

        Otherwise,

            out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
            out_weight[i][j] = 0.

    2. Assigning outputs based on `neg_indices` if `neg_indices` is provided:

    Assumed that i-th instance in `neg_indices` is called `neg_indice`,
    for i-th instance:

    .. code-block:: text

        for id in neg_indice:
            out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
            out_weight[i][id] = 1.0

    Args:
       input (Variable): This input is a 3D LoDTensor with shape [M, P, K].
           Data type should be int32 or float32.
       matched_indices (Variable): The input matched indices
           is 2D Tenosr<int32> with shape [N, P], If MatchIndices[i][j] is -1,
           the j-th entity of column is not matched to any entity of row in
           i-th instance.
       negative_indices (Variable, optional): The input negative example indices
           are an optional input with shape [Neg, 1] and int32 type, where Neg is
           the total number of negative example indices.
       mismatch_value (float32, optional): Fill this value to the mismatched
           location.
       name (string): The default value is None.  Normally there is no need for
           user to set this property.  For more information, please refer
           to :ref:`api_guide_Name`.

    Returns:
        tuple: A tuple(out, out_weight) is returned.

        out (Variable): a 3D Tensor with shape [N, P, K] and same data type
        with `input`, N and P is the same as they are in `matched_indices`,
        K is the same as it in input of X.

        out_weight (Variable): the weight for output with the shape of [N, P, 1].
        Data type is float32.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            x = fluid.data(
                name='x',
                shape=[4, 20, 4],
                dtype='float',
                lod_level=1)
            matched_id = fluid.data(
                name='indices',
                shape=[8, 20],
                dtype='int32')
            trg, trg_weight = fluid.layers.target_assign(
                x,
                matched_id,
                mismatch_value=0)
    """
    helper = LayerHelper('target_assign', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    out_weight = helper.create_variable_for_type_inference(dtype='float32')
    helper.append_op(
        type='target_assign',
        inputs={
            'X': input,
            'MatchIndices': matched_indices,
            'NegIndices': negative_indices
        },
        outputs={'Out': out,
                 'OutWeight': out_weight},
        attrs={'mismatch_value': mismatch_value})
    return out, out_weight


def ssd_loss(location,
             confidence,
             gt_box,
             gt_label,
             prior_box,
             prior_box_var=None,
             background_label=0,
             overlap_threshold=0.5,
             neg_pos_ratio=3.0,
             neg_overlap=0.5,
             loc_loss_weight=1.0,
             conf_loss_weight=1.0,
             match_type='per_prediction',
             mining_type='max_negative',
             normalize=True,
             sample_size=None):
    r"""
	:alias_main: paddle.nn.functional.ssd_loss
	:alias: paddle.nn.functional.ssd_loss,paddle.nn.functional.loss.ssd_loss
	:old_api: paddle.fluid.layers.ssd_loss

    **Multi-box loss layer for object detection algorithm of SSD**

    This layer is to compute detection loss for SSD given the location offset
    predictions, confidence predictions, prior boxes and ground-truth bounding
    boxes and labels, and the type of hard example mining. The returned loss
    is a weighted sum of the localization loss (or regression loss) and
    confidence loss (or classification loss) by performing the following steps:

    1. Find matched bounding box by bipartite matching algorithm.

      1.1 Compute IOU similarity between ground-truth boxes and prior boxes.

      1.2 Compute matched bounding box by bipartite matching algorithm.

    2. Compute confidence for mining hard examples

      2.1. Get the target label based on matched indices.

      2.2. Compute confidence loss.

    3. Apply hard example mining to get the negative example indices and update
       the matched indices.

    4. Assign classification and regression targets

      4.1. Encoded bbox according to the prior boxes.

      4.2. Assign regression targets.

      4.3. Assign classification targets.

    5. Compute the overall objective loss.

      5.1 Compute confidence loss.

      5.2 Compute localization loss.

      5.3 Compute the overall weighted loss.

    Args:
        location (Variable): The location predictions are a 3D Tensor with
            shape [N, Np, 4], N is the batch size, Np is total number of
            predictions for each instance. 4 is the number of coordinate values,
            the layout is [xmin, ymin, xmax, ymax].The data type is float32 or
            float64.
        confidence (Variable): The confidence predictions are a 3D Tensor
            with shape [N, Np, C], N and Np are the same as they are in
            `location`, C is the class number.The data type is float32 or
            float64.
        gt_box (Variable): The ground-truth bounding boxes (bboxes) are a 2D
            LoDTensor with shape [Ng, 4], Ng is the total number of ground-truth
            bboxes of mini-batch input.The data type is float32 or float64.
        gt_label (Variable): The ground-truth labels are a 2D LoDTensor
            with shape [Ng, 1].Ng is the total number of ground-truth bboxes of
            mini-batch input, 1 is the number of class. The data type is float32
            or float64.
        prior_box (Variable): The prior boxes are a 2D Tensor with shape [Np, 4].
            Np and 4 are the same as they are in `location`. The data type is
            float32 or float64.
        prior_box_var (Variable): The variance of prior boxes are a 2D Tensor
            with shape [Np, 4]. Np and 4 are the same as they are in `prior_box`
        background_label (int): The index of background label, 0 by default.
        overlap_threshold (float): If match_type is 'per_prediction', use
            'overlap_threshold' to determine the extra matching bboxes when finding \
            matched boxes. 0.5 by default.
        neg_pos_ratio (float): The ratio of the negative boxes to the positive
            boxes, used only when mining_type is 'max_negative', 3.0 by default.
        neg_overlap (float): The negative overlap upper bound for the unmatched
            predictions. Use only when mining_type is 'max_negative',
            0.5 by default.
        loc_loss_weight (float): Weight for localization loss, 1.0 by default.
        conf_loss_weight (float): Weight for confidence loss, 1.0 by default.
        match_type (str): The type of matching method during training, should
            be 'bipartite' or 'per_prediction', 'per_prediction' by default.
        mining_type (str): The hard example mining type, should be 'hard_example'
            or 'max_negative', now only support `max_negative`.
        normalize (bool): Whether to normalize the SSD loss by the total number
            of output locations, True by default.
        sample_size (int): The max sample size of negative box, used only when
            mining_type is 'hard_example'.

    Returns:
        Variable(Tensor):  The weighted sum of the localization loss and confidence loss, \
        with shape [N * Np, 1], N and Np are the same as they are in
        `location`.The data type is float32 or float64.

    Raises:
        ValueError: If mining_type is 'hard_example', now only support mining \
        type of `max_negative`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            pb = fluid.data(
                           name='prior_box',
                           shape=[10, 4],
                           dtype='float32')
            pbv = fluid.data(
                           name='prior_box_var',
                           shape=[10, 4],
                           dtype='float32')
            loc = fluid.data(name='target_box', shape=[10, 4], dtype='float32')
            scores = fluid.data(name='scores', shape=[10, 21], dtype='float32')
            gt_box = fluid.data(
                 name='gt_box', shape=[4], lod_level=1, dtype='float32')
            gt_label = fluid.data(
                 name='gt_label', shape=[1], lod_level=1, dtype='float32')
            loss = fluid.layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
    """

    helper = LayerHelper('ssd_loss', **locals())
    if mining_type != 'max_negative':
        raise ValueError("Only support mining_type == max_negative now.")

    num, num_prior, num_class = confidence.shape
    conf_shape = nn.shape(confidence)

    def __reshape_to_2d(var):
        return nn.flatten(x=var, axis=2)

    # 1. Find matched bounding box by prior box.
    #   1.1 Compute IOU similarity between ground-truth boxes and prior boxes.
    iou = iou_similarity(x=gt_box, y=prior_box)
    #   1.2 Compute matched bounding box by bipartite matching algorithm.
    matched_indices, matched_dist = bipartite_match(iou, match_type,
                                                    overlap_threshold)

    # 2. Compute confidence for mining hard examples
    # 2.1. Get the target label based on matched indices
    gt_label = nn.reshape(
        x=gt_label, shape=(len(gt_label.shape) - 1) * (0, ) + (-1, 1))
    gt_label.stop_gradient = True
    target_label, _ = target_assign(
        gt_label, matched_indices, mismatch_value=background_label)
    # 2.2. Compute confidence loss.
    # Reshape confidence to 2D tensor.
    confidence = __reshape_to_2d(confidence)
    target_label = tensor.cast(x=target_label, dtype='int64')
    target_label = __reshape_to_2d(target_label)
    target_label.stop_gradient = True
    conf_loss = softmax_with_cross_entropy(confidence, target_label)
    # 3. Mining hard examples
    actual_shape = nn.slice(conf_shape, axes=[0], starts=[0], ends=[2])
    actual_shape.stop_gradient = True
    # shape=(-1, 0) is set for compile-time, the correct shape is set by
    # actual_shape in runtime.
    conf_loss = nn.reshape(
        x=conf_loss, shape=(-1, 0), actual_shape=actual_shape)
    conf_loss.stop_gradient = True
    neg_indices = helper.create_variable_for_type_inference(dtype='int32')
    dtype = matched_indices.dtype
    updated_matched_indices = helper.create_variable_for_type_inference(
        dtype=dtype)
    helper.append_op(
        type='mine_hard_examples',
        inputs={
            'ClsLoss': conf_loss,
            'LocLoss': None,
            'MatchIndices': matched_indices,
            'MatchDist': matched_dist,
        },
        outputs={
            'NegIndices': neg_indices,
            'UpdatedMatchIndices': updated_matched_indices
        },
        attrs={
            'neg_pos_ratio': neg_pos_ratio,
            'neg_dist_threshold': neg_overlap,
            'mining_type': mining_type,
            'sample_size': sample_size,
        })

    # 4. Assign classification and regression targets
    # 4.1. Encoded bbox according to the prior boxes.
    encoded_bbox = box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=gt_box,
        code_type='encode_center_size')
    # 4.2. Assign regression targets
    target_bbox, target_loc_weight = target_assign(
        encoded_bbox, updated_matched_indices, mismatch_value=background_label)
    # 4.3. Assign classification targets
    target_label, target_conf_weight = target_assign(
        gt_label,
        updated_matched_indices,
        negative_indices=neg_indices,
        mismatch_value=background_label)

    # 5. Compute loss.
    # 5.1 Compute confidence loss.
    target_label = __reshape_to_2d(target_label)
    target_label = tensor.cast(x=target_label, dtype='int64')

    conf_loss = softmax_with_cross_entropy(confidence, target_label)
    target_conf_weight = __reshape_to_2d(target_conf_weight)
    conf_loss = conf_loss * target_conf_weight

    # the target_label and target_conf_weight do not have gradient.
    target_label.stop_gradient = True
    target_conf_weight.stop_gradient = True

    # 5.2 Compute regression loss.
    location = __reshape_to_2d(location)
    target_bbox = __reshape_to_2d(target_bbox)

    loc_loss = nn.smooth_l1(location, target_bbox)
    target_loc_weight = __reshape_to_2d(target_loc_weight)
    loc_loss = loc_loss * target_loc_weight

    # the target_bbox and target_loc_weight do not have gradient.
    target_bbox.stop_gradient = True
    target_loc_weight.stop_gradient = True

    # 5.3 Compute overall weighted loss.
    loss = conf_loss_weight * conf_loss + loc_loss_weight * loc_loss
    # reshape to [N, Np], N is the batch size and Np is the prior box number.
    # shape=(-1, 0) is set for compile-time, the correct shape is set by
    # actual_shape in runtime.
    loss = nn.reshape(x=loss, shape=(-1, 0), actual_shape=actual_shape)
    loss = nn.reduce_sum(loss, dim=1, keep_dim=True)
    if normalize:
        normalizer = nn.reduce_sum(target_loc_weight)
        loss = loss / normalizer

    return loss


def prior_box(input,
              image,
              min_sizes,
              max_sizes=None,
              aspect_ratios=[1.],
              variance=[0.1, 0.1, 0.2, 0.2],
              flip=False,
              clip=False,
              steps=[0.0, 0.0],
              offset=0.5,
              name=None,
              min_max_aspect_ratios_order=False):
    """

    This op generates prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
    Each position of the input produce N prior boxes, N is determined by
    the count of min_sizes, max_sizes and aspect_ratios, The size of the
    box is in range(min_size, max_size) interval, which is generated in
    sequence according to the aspect_ratios.

    Parameters:
       input(Variable): 4-D tensor(NCHW), the data type should be float32 or float64.
       image(Variable): 4-D tensor(NCHW), the input image data of PriorBoxOp,
            the data type should be float32 or float64.
       min_sizes(list|tuple|float): the min sizes of generated prior boxes.
       max_sizes(list|tuple|None): the max sizes of generated prior boxes.
            Default: None.
       aspect_ratios(list|tuple|float): the aspect ratios of generated
            prior boxes. Default: [1.].
       variance(list|tuple): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip(bool): Whether to flip aspect ratios. Default:False.
       clip(bool): Whether to clip out-of-boundary boxes. Default: False.
       step(list|tuple): Prior boxes step across width and height, If
            step[0] equals to 0.0 or step[1] equals to 0.0, the prior boxes step across
            height or weight of the input will be automatically calculated.
            Default: [0., 0.]
       offset(float): Prior boxes center offset. Default: 0.5
       min_max_aspect_ratios_order(bool): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the final
            detection results. Default: False.
       name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tuple: A tuple with two Variable (boxes, variances)

        boxes(Variable): the output prior boxes of PriorBox.
	4-D tensor, the layout is [H, W, num_priors, 4].
        H is the height of input, W is the width of input,
        num_priors is the total box count of each position of input.

        variances(Variable): the expanded variances of PriorBox.
    	4-D tensor, the layput is [H, W, num_priors, 4].
        H is the height of input, W is the width of input
        num_priors is the total box count of each position of input

    Examples:
        .. code-block:: python

	    #declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
        import paddle
        paddle.enable_static()
	    input = fluid.data(name="input", shape=[None,3,6,9])
	    image = fluid.data(name="image", shape=[None,3,9,12])
	    box, var = fluid.layers.prior_box(
                 input=input,
                 image=image,
		 min_sizes=[100.],
                 clip=True,
                 flip=True)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    # prepare a batch of data
	    input_data = np.random.rand(1,3,6,9).astype("float32")
	    image_data = np.random.rand(1,3,9,12).astype("float32")
 
	    box_out, var_out = exe.run(fluid.default_main_program(),
                feed={"input":input_data,"image":image_data},
                fetch_list=[box,var],
                return_numpy=True)
 
	    # print(box_out.shape)
	    # (6, 9, 1, 4)
	    # print(var_out.shape)
	    # (6, 9, 1, 4)

	    # imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		image = dg.to_variable(image_data)
    		box, var = fluid.layers.prior_box(
		    input=input,
		    image=image,
		    min_sizes=[100.],
		    clip=True,
		    flip=True)
		# print(box.shape)
		# [6L, 9L, 1L, 4L]
                # print(var.shape)
		# [6L, 9L, 1L, 4L]

    """
    helper = LayerHelper("prior_box", **locals())
    dtype = helper.input_dtype()
    check_variable_and_dtype(
        input, 'input', ['uint8', 'int8', 'float32', 'float64'], 'prior_box')

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if not _is_list_or_tuple_(min_sizes):
        min_sizes = [min_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not (_is_list_or_tuple_(steps) and len(steps) == 2):
        raise ValueError('steps should be a list or tuple ',
                         'with length 2, (step_width, step_height).')

    min_sizes = list(map(float, min_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    steps = list(map(float, steps))

    attrs = {
        'min_sizes': min_sizes,
        'aspect_ratios': aspect_ratios,
        'variances': variance,
        'flip': flip,
        'clip': clip,
        'step_w': steps[0],
        'step_h': steps[1],
        'offset': offset,
        'min_max_aspect_ratios_order': min_max_aspect_ratios_order
    }
    if max_sizes is not None and len(max_sizes) > 0 and max_sizes[0] > 0:
        if not _is_list_or_tuple_(max_sizes):
            max_sizes = [max_sizes]
        attrs['max_sizes'] = max_sizes

    box = helper.create_variable_for_type_inference(dtype)
    var = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prior_box",
        inputs={"Input": input,
                "Image": image},
        outputs={"Boxes": box,
                 "Variances": var},
        attrs=attrs, )
    box.stop_gradient = True
    var.stop_gradient = True
    return box, var


def density_prior_box(input,
                      image,
                      densities=None,
                      fixed_sizes=None,
                      fixed_ratios=None,
                      variance=[0.1, 0.1, 0.2, 0.2],
                      clip=False,
                      steps=[0.0, 0.0],
                      offset=0.5,
                      flatten_to_2d=False,
                      name=None):
    r"""

    This op generates density prior boxes for SSD(Single Shot MultiBox Detector) 
    algorithm. Each position of the input produce N prior boxes, N is 
    determined by the count of densities, fixed_sizes and fixed_ratios. 
    Boxes center at grid points around each input position is generated by 
    this operator, and the grid points is determined by densities and 
    the count of density prior box is determined by fixed_sizes and fixed_ratios. 
    Obviously, the number of fixed_sizes is equal to the number of densities.
    
    For densities_i in densities:
    
    .. math::

        N\_density_prior\_box = SUM(N\_fixed\_ratios * densities\_i^2)

    N_density_prior_box is the number of density_prior_box and N_fixed_ratios is the number of fixed_ratios.

    Parameters:
       input(Variable): 4-D tensor(NCHW), the data type should be float32 of float64.
       image(Variable): 4-D tensor(NCHW), the input image data of PriorBoxOp, the data type should be float32 or float64.
            the layout is NCHW.
       densities(list|tuple|None): The densities of generated density prior 
            boxes, this attribute should be a list or tuple of integers. 
            Default: None.
       fixed_sizes(list|tuple|None): The fixed sizes of generated density
            prior boxes, this attribute should a list or tuple of same 
            length with :attr:`densities`. Default: None.
       fixed_ratios(list|tuple|None): The fixed ratios of generated density
            prior boxes, if this attribute is not set and :attr:`densities`
            and :attr:`fix_sizes` is set, :attr:`aspect_ratios` will be used
            to generate density prior boxes.
       variance(list|tuple): The variances to be encoded in density prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       clip(bool): Whether to clip out of boundary boxes. Default: False.
       step(list|tuple): Prior boxes step across width and height, If
            step[0] equals 0.0 or step[1] equals 0.0, the density prior boxes step across
            height or weight of the input will be automatically calculated.
            Default: [0., 0.]
       offset(float): Prior boxes center offset. Default: 0.5
       flatten_to_2d(bool): Whether to flatten output prior boxes and variance
           to 2D shape, the second dim is 4. Default: False.
       name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        Tuple: A tuple with two Variable (boxes, variances)

        boxes: the output density prior boxes of PriorBox.
        4-D tensor, the layout is [H, W, num_priors, 4] when flatten_to_2d is False.
        2-D tensor, the layout is [H * W * num_priors, 4] when flatten_to_2d is True.
        H is the height of input, W is the width of input, and num_priors is the total box count of each position of input.

        variances: the expanded variances of PriorBox.
        4-D tensor, the layout is [H, W, num_priors, 4] when flatten_to_2d is False.
        2-D tensor, the layout is [H * W * num_priors, 4] when flatten_to_2d is True.
        H is the height of input, W is the width of input, and num_priors is the total box count of each position of input.


    Examples:

        .. code-block:: python

            #declarative mode

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            paddle.enable_static()

            input = fluid.data(name="input", shape=[None,3,6,9])
            image = fluid.data(name="image", shape=[None,3,9,12])
            box, var = fluid.layers.density_prior_box(
                 input=input,
                 image=image,
                 densities=[4, 2, 1],
                 fixed_sizes=[32.0, 64.0, 128.0],
                 fixed_ratios=[1.],
                 clip=True,
                 flatten_to_2d=True)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
 
            # prepare a batch of data
            input_data = np.random.rand(1,3,6,9).astype("float32")
            image_data = np.random.rand(1,3,9,12).astype("float32")

            box_out, var_out = exe.run(
                fluid.default_main_program(),
                feed={"input":input_data,
                      "image":image_data},
                fetch_list=[box,var],
                return_numpy=True)

            # print(box_out.shape)
            # (1134, 4)
            # print(var_out.shape)
            # (1134, 4)


            #imperative mode
            import paddle.fluid.dygraph as dg

            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                image = dg.to_variable(image_data)
                box, var = fluid.layers.density_prior_box(
                    input=input,
                    image=image,
                    densities=[4, 2, 1],
                    fixed_sizes=[32.0, 64.0, 128.0],
                    fixed_ratios=[1.],
                    clip=True)

                # print(box.shape)
                # [6L, 9L, 21L, 4L]
                # print(var.shape)
                # [6L, 9L, 21L, 4L]

    """
    helper = LayerHelper("density_prior_box", **locals())
    dtype = helper.input_dtype()
    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'density_prior_box')

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    check_type(densities, 'densities', (list, tuple), 'density_prior_box')
    check_type(fixed_sizes, 'fixed_sizes', (list, tuple), 'density_prior_box')
    check_type(fixed_ratios, 'fixed_ratios', (list, tuple), 'density_prior_box')
    if len(densities) != len(fixed_sizes):
        raise ValueError('densities and fixed_sizes length should be euqal.')

    if not (_is_list_or_tuple_(steps) and len(steps) == 2):
        raise ValueError('steps should be a list or tuple ',
                         'with length 2, (step_width, step_height).')

    densities = list(map(int, densities))
    fixed_sizes = list(map(float, fixed_sizes))
    fixed_ratios = list(map(float, fixed_ratios))
    steps = list(map(float, steps))

    attrs = {
        'variances': variance,
        'clip': clip,
        'step_w': steps[0],
        'step_h': steps[1],
        'offset': offset,
        'densities': densities,
        'fixed_sizes': fixed_sizes,
        'fixed_ratios': fixed_ratios,
        'flatten_to_2d': flatten_to_2d,
    }
    box = helper.create_variable_for_type_inference(dtype)
    var = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="density_prior_box",
        inputs={"Input": input,
                "Image": image},
        outputs={"Boxes": box,
                 "Variances": var},
        attrs=attrs, )
    box.stop_gradient = True
    var.stop_gradient = True
    return box, var


@static_only
def multi_box_head(inputs,
                   image,
                   base_size,
                   num_classes,
                   aspect_ratios,
                   min_ratio=None,
                   max_ratio=None,
                   min_sizes=None,
                   max_sizes=None,
                   steps=None,
                   step_w=None,
                   step_h=None,
                   offset=0.5,
                   variance=[0.1, 0.1, 0.2, 0.2],
                   flip=True,
                   clip=False,
                   kernel_size=1,
                   pad=0,
                   stride=1,
                   name=None,
                   min_max_aspect_ratios_order=False):
    """
	:api_attr: Static Graph

    Base on SSD ((Single Shot MultiBox Detector) algorithm, generate prior boxes,
    regression location and classification confidence on multiple input feature
    maps, then output the concatenate results. The details of this algorithm,
    please refer the section 2.2 of SSD paper `SSD: Single Shot MultiBox Detector
    <https://arxiv.org/abs/1512.02325>`_ .

    Args:
       inputs (list(Variable)|tuple(Variable)): The list of input variables,
           the format of all Variables are 4-D Tensor, layout is NCHW.
           Data type should be float32 or float64.
       image (Variable): The input image, layout is NCHW. Data type should be
           the same as inputs.
       base_size(int): the base_size is input image size. When len(inputs) > 2
           and `min_size` and `max_size` are None, the `min_size` and `max_size`
           are calculated by `baze_size`, 'min_ratio' and `max_ratio`. The
           formula is as follows:

              ..  code-block:: text

                  min_sizes = []
                  max_sizes = []
                  step = int(math.floor(((max_ratio - min_ratio)) / (num_layer - 2)))
                  for ratio in six.moves.range(min_ratio, max_ratio + 1, step):
                      min_sizes.append(base_size * ratio / 100.)
                      max_sizes.append(base_size * (ratio + step) / 100.)
                      min_sizes = [base_size * .10] + min_sizes
                      max_sizes = [base_size * .20] + max_sizes

       num_classes(int): The number of classes.
       aspect_ratios(list(float) | tuple(float)): the aspect ratios of generated
           prior boxes. The length of input and aspect_ratios must be equal.
       min_ratio(int): the min ratio of generated prior boxes.
       max_ratio(int): the max ratio of generated prior boxes.
       min_sizes(list|tuple|None): If `len(inputs) <=2`,
            min_sizes must be set up, and the length of min_sizes
            should equal to the length of inputs. Default: None.
       max_sizes(list|tuple|None): If `len(inputs) <=2`,
            max_sizes must be set up, and the length of min_sizes
            should equal to the length of inputs. Default: None.
       steps(list|tuple): If step_w and step_h are the same,
            step_w and step_h can be replaced by steps.
       step_w(list|tuple): Prior boxes step
            across width. If step_w[i] == 0.0, the prior boxes step
            across width of the inputs[i] will be automatically
            calculated. Default: None.
       step_h(list|tuple): Prior boxes step across height, If
            step_h[i] == 0.0, the prior boxes step across height of
            the inputs[i] will be automatically calculated. Default: None.
       offset(float): Prior boxes center offset. Default: 0.5
       variance(list|tuple): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip(bool): Whether to flip aspect ratios. Default:False.
       clip(bool): Whether to clip out-of-boundary boxes. Default: False.
       kernel_size(int): The kernel size of conv2d. Default: 1.
       pad(int|list|tuple): The padding of conv2d. Default:0.
       stride(int|list|tuple): The stride of conv2d. Default:1,
       name(str): The default value is None.  Normally there is no need
           for user to set this property.  For more information, please
           refer to :ref:`api_guide_Name`.
       min_max_aspect_ratios_order(bool): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the final
            detection results. Default: False.

    Returns:
        tuple: A tuple with four Variables. (mbox_loc, mbox_conf, boxes, variances)

        mbox_loc (Variable): The predicted boxes' location of the inputs. The
        layout is [N, num_priors, 4], where N is batch size, ``num_priors``
        is the number of prior boxes. Data type is the same as input.

        mbox_conf (Variable): The predicted boxes' confidence of the inputs.
        The layout is [N, num_priors, C], where ``N`` and ``num_priors`` 
        has the same meaning as above. C is the number of Classes.
        Data type is the same as input.

        boxes (Variable): the output prior boxes. The layout is [num_priors, 4].
        The meaning of num_priors is the same as above.
        Data type is the same as input.

        variances (Variable): the expanded variances for prior boxes.
        The layout is [num_priors, 4]. Data type is the same as input.

    Examples 1: set min_ratio and max_ratio:
        .. code-block:: python

          import paddle
          paddle.enable_static()

          images = paddle.static.data(name='data', shape=[None, 3, 300, 300], dtype='float32')
          conv1 = paddle.static.data(name='conv1', shape=[None, 512, 19, 19], dtype='float32')
          conv2 = paddle.static.data(name='conv2', shape=[None, 1024, 10, 10], dtype='float32')
          conv3 = paddle.static.data(name='conv3', shape=[None, 512, 5, 5], dtype='float32')
          conv4 = paddle.static.data(name='conv4', shape=[None, 256, 3, 3], dtype='float32')
          conv5 = paddle.static.data(name='conv5', shape=[None, 256, 2, 2], dtype='float32')
          conv6 = paddle.static.data(name='conv6', shape=[None, 128, 1, 1], dtype='float32')

          mbox_locs, mbox_confs, box, var = paddle.static.nn.multi_box_head(
            inputs=[conv1, conv2, conv3, conv4, conv5, conv6],
            image=images,
            num_classes=21,
            min_ratio=20,
            max_ratio=90,
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
            base_size=300,
            offset=0.5,
            flip=True,
            clip=True)

    Examples 2: set min_sizes and max_sizes:
        .. code-block:: python

          import paddle
          paddle.enable_static()

          images = paddle.static.data(name='data', shape=[None, 3, 300, 300], dtype='float32')
          conv1 = paddle.static.data(name='conv1', shape=[None, 512, 19, 19], dtype='float32')
          conv2 = paddle.static.data(name='conv2', shape=[None, 1024, 10, 10], dtype='float32')
          conv3 = paddle.static.data(name='conv3', shape=[None, 512, 5, 5], dtype='float32')
          conv4 = paddle.static.data(name='conv4', shape=[None, 256, 3, 3], dtype='float32')
          conv5 = paddle.static.data(name='conv5', shape=[None, 256, 2, 2], dtype='float32')
          conv6 = paddle.static.data(name='conv6', shape=[None, 128, 1, 1], dtype='float32')

          mbox_locs, mbox_confs, box, var = paddle.static.nn.multi_box_head(
            inputs=[conv1, conv2, conv3, conv4, conv5, conv6],
            image=images,
            num_classes=21,
            min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
            max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
            base_size=300,
            offset=0.5,
            flip=True,
            clip=True)

    """

    def _reshape_with_axis_(input, axis=1):
        out = nn.flatten(x=input, axis=axis)
        return out

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    def _is_list_or_tuple_and_equal(data, length, err_info):
        if not (_is_list_or_tuple_(data) and len(data) == length):
            raise ValueError(err_info)

    if not _is_list_or_tuple_(inputs):
        raise ValueError('inputs should be a list or tuple.')

    num_layer = len(inputs)

    if num_layer <= 2:
        assert min_sizes is not None and max_sizes is not None
        assert len(min_sizes) == num_layer and len(max_sizes) == num_layer
    elif min_sizes is None and max_sizes is None:
        min_sizes = []
        max_sizes = []
        step = int(math.floor(((max_ratio - min_ratio)) / (num_layer - 2)))
        for ratio in six.moves.range(min_ratio, max_ratio + 1, step):
            min_sizes.append(base_size * ratio / 100.)
            max_sizes.append(base_size * (ratio + step) / 100.)
        min_sizes = [base_size * .10] + min_sizes
        max_sizes = [base_size * .20] + max_sizes

    if aspect_ratios:
        _is_list_or_tuple_and_equal(
            aspect_ratios, num_layer,
            'aspect_ratios should be list or tuple, and the length of inputs '
            'and aspect_ratios should be the same.')
    if step_h is not None:
        _is_list_or_tuple_and_equal(
            step_h, num_layer,
            'step_h should be list or tuple, and the length of inputs and '
            'step_h should be the same.')
    if step_w is not None:
        _is_list_or_tuple_and_equal(
            step_w, num_layer,
            'step_w should be list or tuple, and the length of inputs and '
            'step_w should be the same.')
    if steps is not None:
        _is_list_or_tuple_and_equal(
            steps, num_layer,
            'steps should be list or tuple, and the length of inputs and '
            'step_w should be the same.')
        step_w = steps
        step_h = steps

    mbox_locs = []
    mbox_confs = []
    box_results = []
    var_results = []
    for i, input in enumerate(inputs):
        min_size = min_sizes[i]
        max_size = max_sizes[i]

        if not _is_list_or_tuple_(min_size):
            min_size = [min_size]
        if not _is_list_or_tuple_(max_size):
            max_size = [max_size]

        aspect_ratio = []
        if aspect_ratios is not None:
            aspect_ratio = aspect_ratios[i]
            if not _is_list_or_tuple_(aspect_ratio):
                aspect_ratio = [aspect_ratio]
        step = [step_w[i] if step_w else 0.0, step_h[i] if step_w else 0.0]

        box, var = prior_box(input, image, min_size, max_size, aspect_ratio,
                             variance, flip, clip, step, offset, None,
                             min_max_aspect_ratios_order)

        box_results.append(box)
        var_results.append(var)

        num_boxes = box.shape[2]

        # get loc
        num_loc_output = num_boxes * 4
        mbox_loc = nn.conv2d(
            input=input,
            num_filters=num_loc_output,
            filter_size=kernel_size,
            padding=pad,
            stride=stride)

        mbox_loc = nn.transpose(mbox_loc, perm=[0, 2, 3, 1])
        mbox_loc_flatten = nn.flatten(mbox_loc, axis=1)
        mbox_locs.append(mbox_loc_flatten)

        # get conf
        num_conf_output = num_boxes * num_classes
        conf_loc = nn.conv2d(
            input=input,
            num_filters=num_conf_output,
            filter_size=kernel_size,
            padding=pad,
            stride=stride)
        conf_loc = nn.transpose(conf_loc, perm=[0, 2, 3, 1])
        conf_loc_flatten = nn.flatten(conf_loc, axis=1)
        mbox_confs.append(conf_loc_flatten)

    if len(box_results) == 1:
        box = box_results[0]
        var = var_results[0]
        mbox_locs_concat = mbox_locs[0]
        mbox_confs_concat = mbox_confs[0]
    else:
        reshaped_boxes = []
        reshaped_vars = []
        for i in range(len(box_results)):
            reshaped_boxes.append(_reshape_with_axis_(box_results[i], axis=3))
            reshaped_vars.append(_reshape_with_axis_(var_results[i], axis=3))

        box = tensor.concat(reshaped_boxes)
        var = tensor.concat(reshaped_vars)
        mbox_locs_concat = tensor.concat(mbox_locs, axis=1)
        mbox_locs_concat = nn.reshape(mbox_locs_concat, shape=[0, -1, 4])
        mbox_confs_concat = tensor.concat(mbox_confs, axis=1)
        mbox_confs_concat = nn.reshape(
            mbox_confs_concat, shape=[0, -1, num_classes])

    box.stop_gradient = True
    var.stop_gradient = True
    return mbox_locs_concat, mbox_confs_concat, box, var


def anchor_generator(input,
                     anchor_sizes=None,
                     aspect_ratios=None,
                     variance=[0.1, 0.1, 0.2, 0.2],
                     stride=None,
                     offset=0.5,
                     name=None):
    """

    **Anchor generator operator**

    Generate anchors for Faster RCNN algorithm.
    Each position of the input produce N anchors, N =
    size(anchor_sizes) * size(aspect_ratios). The order of generated anchors
    is firstly aspect_ratios loop then anchor_sizes loop.

    Args:
       input(Variable): 4-D Tensor with shape [N,C,H,W]. The input feature map.
       anchor_sizes(float32|list|tuple, optional): The anchor sizes of generated
          anchors, given in absolute pixels e.g. [64., 128., 256., 512.].
          For instance, the anchor size of 64 means the area of this anchor 
          equals to 64**2. None by default.
       aspect_ratios(float32|list|tuple, optional): The height / width ratios 
           of generated anchors, e.g. [0.5, 1.0, 2.0]. None by default.
       variance(list|tuple, optional): The variances to be used in box 
           regression deltas. The data type is float32, [0.1, 0.1, 0.2, 0.2] by 
           default.
       stride(list|tuple, optional): The anchors stride across width and height.
           The data type is float32. e.g. [16.0, 16.0]. None by default.
       offset(float32, optional): Prior boxes center offset. 0.5 by default.
       name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and None 
           by default. 

    Returns:
        Tuple:

        Anchors(Variable): The output anchors with a layout of [H, W, num_anchors, 4].
        H is the height of input, W is the width of input,
        num_anchors is the box count of each position. 
        Each anchor is in (xmin, ymin, xmax, ymax) format an unnormalized.
 
        Variances(Variable): The expanded variances of anchors
        with a layout of [H, W, num_priors, 4].
        H is the height of input, W is the width of input
        num_anchors is the box count of each position.
        Each variance is in (xcenter, ycenter, w, h) format.


    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle

            paddle.enable_static()
            conv1 = fluid.data(name='conv1', shape=[None, 48, 16, 16], dtype='float32')
            anchor, var = fluid.layers.anchor_generator(
                input=conv1,
                anchor_sizes=[64, 128, 256, 512],
                aspect_ratios=[0.5, 1.0, 2.0],
                variance=[0.1, 0.1, 0.2, 0.2],
                stride=[16.0, 16.0],
                offset=0.5)
    """
    helper = LayerHelper("anchor_generator", **locals())
    dtype = helper.input_dtype()

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if not _is_list_or_tuple_(anchor_sizes):
        anchor_sizes = [anchor_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not (_is_list_or_tuple_(stride) and len(stride) == 2):
        raise ValueError('stride should be a list or tuple ',
                         'with length 2, (stride_width, stride_height).')

    anchor_sizes = list(map(float, anchor_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    stride = list(map(float, stride))

    attrs = {
        'anchor_sizes': anchor_sizes,
        'aspect_ratios': aspect_ratios,
        'variances': variance,
        'stride': stride,
        'offset': offset
    }

    anchor = helper.create_variable_for_type_inference(dtype)
    var = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="anchor_generator",
        inputs={"Input": input},
        outputs={"Anchors": anchor,
                 "Variances": var},
        attrs=attrs, )
    anchor.stop_gradient = True
    var.stop_gradient = True
    return anchor, var


def roi_perspective_transform(input,
                              rois,
                              transformed_height,
                              transformed_width,
                              spatial_scale=1.0,
                              name=None):
    """
    **The** `rois` **of this op should be a LoDTensor.**

    ROI perspective transform op applies perspective transform to map each roi into an 
    rectangular region. Perspective transform is a type of transformation in linear algebra.

    Parameters:
        input (Variable): 4-D Tensor, input of ROIPerspectiveTransformOp. The format of 
                          input tensor is NCHW. Where N is batch size, C is the
                          number of input channels, H is the height of the feature,
                          and W is the width of the feature. The data type is float32.
        rois (Variable):  2-D LoDTensor, ROIs (Regions of Interest) to be transformed. 
                          It should be a 2-D LoDTensor of shape (num_rois, 8). Given as 
                          [[x1, y1, x2, y2, x3, y3, x4, y4], ...], (x1, y1) is the 
                          top left coordinates, and (x2, y2) is the top right 
                          coordinates, and (x3, y3) is the bottom right coordinates, 
                          and (x4, y4) is the bottom left coordinates. The data type is the
                          same as `input` 
        transformed_height (int): The height of transformed output.
        transformed_width (int): The width of transformed output.
        spatial_scale (float): Spatial scale factor to scale ROI coords. Default: 1.0
        name(str, optional): The default value is None.  
                             Normally there is no need for user to set this property.  
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
            A tuple with three Variables. (out, mask, transform_matrix)

            out: The output of ROIPerspectiveTransformOp which is a 4-D tensor with shape
            (num_rois, channels, transformed_h, transformed_w). The data type is the same as `input`

            mask: The mask of ROIPerspectiveTransformOp which is a 4-D tensor with shape
            (num_rois, 1, transformed_h, transformed_w). The data type is int32

            transform_matrix: The transform matrix of ROIPerspectiveTransformOp which is
            a 2-D tensor with shape (num_rois, 9). The data type is the same as `input`

    Return Type:
        tuple

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.data(name='x', shape=[100, 256, 28, 28], dtype='float32')
            rois = fluid.data(name='rois', shape=[None, 8], lod_level=1, dtype='float32')
            out, mask, transform_matrix = fluid.layers.roi_perspective_transform(x, rois, 7, 7, 1.0)
    """
    check_variable_and_dtype(input, 'input', ['float32'],
                             'roi_perspective_transform')
    check_variable_and_dtype(rois, 'rois', ['float32'],
                             'roi_perspective_transform')
    check_type(transformed_height, 'transformed_height', int,
               'roi_perspective_transform')
    check_type(transformed_width, 'transformed_width', int,
               'roi_perspective_transform')
    check_type(spatial_scale, 'spatial_scale', float,
               'roi_perspective_transform')

    helper = LayerHelper('roi_perspective_transform', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype="int32")
    transform_matrix = helper.create_variable_for_type_inference(dtype)
    out2in_idx = helper.create_variable_for_type_inference(dtype="int32")
    out2in_w = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="roi_perspective_transform",
        inputs={"X": input,
                "ROIs": rois},
        outputs={
            "Out": out,
            "Out2InIdx": out2in_idx,
            "Out2InWeights": out2in_w,
            "Mask": mask,
            "TransformMatrix": transform_matrix
        },
        attrs={
            "transformed_height": transformed_height,
            "transformed_width": transformed_width,
            "spatial_scale": spatial_scale
        })
    return out, mask, transform_matrix


def generate_proposal_labels(rpn_rois,
                             gt_classes,
                             is_crowd,
                             gt_boxes,
                             im_info,
                             batch_size_per_im=256,
                             fg_fraction=0.25,
                             fg_thresh=0.25,
                             bg_thresh_hi=0.5,
                             bg_thresh_lo=0.0,
                             bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                             class_nums=None,
                             use_random=True,
                             is_cls_agnostic=False,
                             is_cascade_rcnn=False,
                             max_overlap=None,
                             return_max_overlap=False):
    """

    **Generate Proposal Labels of Faster-RCNN**

    This operator can be, for given the GenerateProposalOp output bounding boxes and groundtruth,
    to sample foreground boxes and background boxes, and compute loss target.

    RpnRois is the output boxes of RPN and was processed by generate_proposal_op, these boxes
    were combined with groundtruth boxes and sampled according to batch_size_per_im and fg_fraction,
    If an instance with a groundtruth overlap greater than fg_thresh, then it was considered as a foreground sample.
    If an instance with a groundtruth overlap greater than bg_thresh_lo and lower than bg_thresh_hi,
    then it was considered as a background sample.
    After all foreground and background boxes are chosen (so called Rois),
    then we apply random sampling to make sure
    the number of foreground boxes is no more than batch_size_per_im * fg_fraction.

    For each box in Rois, we assign the classification (class label) and regression targets (box label) to it.
    Finally BboxInsideWeights and BboxOutsideWeights are used to specify whether it would contribute to training loss.

    Args:
        rpn_rois(Variable): A 2-D LoDTensor with shape [N, 4]. N is the number of the GenerateProposalOp's output, each element is a bounding box with [xmin, ymin, xmax, ymax] format. The data type can be float32 or float64.
        gt_classes(Variable): A 2-D LoDTensor with shape [M, 1]. M is the number of groundtruth, each element is a class label of groundtruth. The data type must be int32.
        is_crowd(Variable): A 2-D LoDTensor with shape [M, 1]. M is the number of groundtruth, each element is a flag indicates whether a groundtruth is crowd. The data type must be int32.
        gt_boxes(Variable): A 2-D LoDTensor with shape [M, 4]. M is the number of groundtruth, each element is a bounding box with [xmin, ymin, xmax, ymax] format.
        im_info(Variable): A 2-D LoDTensor with shape [B, 3]. B is the number of input images, each element consists of im_height, im_width, im_scale.

        batch_size_per_im(int): Batch size of rois per images. The data type must be int32.
        fg_fraction(float): Foreground fraction in total batch_size_per_im. The data type must be float32.
        fg_thresh(float): Overlap threshold which is used to chose foreground sample. The data type must be float32.
        bg_thresh_hi(float): Overlap threshold upper bound which is used to chose background sample. The data type must be float32.
        bg_thresh_lo(float): Overlap threshold lower bound which is used to chose background sample. The data type must be float32.
        bbox_reg_weights(list|tuple): Box regression weights. The data type must be float32.
        class_nums(int): Class number. The data type must be int32.
        use_random(bool): Use random sampling to choose foreground and background boxes.
        is_cls_agnostic(bool): bbox regression use class agnostic simply which only represent fg and bg boxes.
        is_cascade_rcnn(bool): it will filter some bbox crossing the image's boundary when setting True.
        max_overlap(Variable): Maximum overlap between each proposal box and ground-truth.
        return_max_overlap(bool): Whether return the maximum overlap between each sampled RoI and ground-truth.

    Returns:
        tuple:
        A tuple with format``(rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights, max_overlap)``.

        - **rois**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 4]``. The data type is the same as ``rpn_rois``.
        - **labels_int32**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 1]``. The data type must be int32.
        - **bbox_targets**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 4 * class_num]``. The regression targets of all RoIs. The data type is the same as ``rpn_rois``.
        - **bbox_inside_weights**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 4 * class_num]``. The weights of foreground boxes' regression loss. The data type is the same as ``rpn_rois``.
        - **bbox_outside_weights**: 2-D LoDTensor with shape ``[batch_size_per_im * batch_size, 4 * class_num]``. The weights of regression loss. The data type is the same as ``rpn_rois``.
        - **max_overlap**: 1-D LoDTensor with shape ``[P]``. P is the number of output ``rois``. The maximum overlap between each sampled RoI and ground-truth.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()
            rpn_rois = fluid.data(name='rpn_rois', shape=[None, 4], dtype='float32')
            gt_classes = fluid.data(name='gt_classes', shape=[None, 1], dtype='int32')
            is_crowd = fluid.data(name='is_crowd', shape=[None, 1], dtype='int32')
            gt_boxes = fluid.data(name='gt_boxes', shape=[None, 4], dtype='float32')
            im_info = fluid.data(name='im_info', shape=[None, 3], dtype='float32')
            rois, labels, bbox, inside_weights, outside_weights = fluid.layers.generate_proposal_labels(
                           rpn_rois, gt_classes, is_crowd, gt_boxes, im_info,
                           class_nums=10)

    """

    helper = LayerHelper('generate_proposal_labels', **locals())

    check_variable_and_dtype(rpn_rois, 'rpn_rois', ['float32', 'float64'],
                             'generate_proposal_labels')
    check_variable_and_dtype(gt_classes, 'gt_classes', ['int32'],
                             'generate_proposal_labels')
    check_variable_and_dtype(is_crowd, 'is_crowd', ['int32'],
                             'generate_proposal_labels')
    if is_cascade_rcnn:
        assert max_overlap is not None, "Input max_overlap of generate_proposal_labels should not be None if is_cascade_rcnn is True"

    rois = helper.create_variable_for_type_inference(dtype=rpn_rois.dtype)
    labels_int32 = helper.create_variable_for_type_inference(
        dtype=gt_classes.dtype)
    bbox_targets = helper.create_variable_for_type_inference(
        dtype=rpn_rois.dtype)
    bbox_inside_weights = helper.create_variable_for_type_inference(
        dtype=rpn_rois.dtype)
    bbox_outside_weights = helper.create_variable_for_type_inference(
        dtype=rpn_rois.dtype)
    max_overlap_with_gt = helper.create_variable_for_type_inference(
        dtype=rpn_rois.dtype)

    inputs = {
        'RpnRois': rpn_rois,
        'GtClasses': gt_classes,
        'IsCrowd': is_crowd,
        'GtBoxes': gt_boxes,
        'ImInfo': im_info,
    }
    if max_overlap is not None:
        inputs['MaxOverlap'] = max_overlap
    helper.append_op(
        type="generate_proposal_labels",
        inputs=inputs,
        outputs={
            'Rois': rois,
            'LabelsInt32': labels_int32,
            'BboxTargets': bbox_targets,
            'BboxInsideWeights': bbox_inside_weights,
            'BboxOutsideWeights': bbox_outside_weights,
            'MaxOverlapWithGT': max_overlap_with_gt
        },
        attrs={
            'batch_size_per_im': batch_size_per_im,
            'fg_fraction': fg_fraction,
            'fg_thresh': fg_thresh,
            'bg_thresh_hi': bg_thresh_hi,
            'bg_thresh_lo': bg_thresh_lo,
            'bbox_reg_weights': bbox_reg_weights,
            'class_nums': class_nums,
            'use_random': use_random,
            'is_cls_agnostic': is_cls_agnostic,
            'is_cascade_rcnn': is_cascade_rcnn
        })

    rois.stop_gradient = True
    labels_int32.stop_gradient = True
    bbox_targets.stop_gradient = True
    bbox_inside_weights.stop_gradient = True
    bbox_outside_weights.stop_gradient = True
    max_overlap_with_gt.stop_gradient = True

    if return_max_overlap:
        return rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights, max_overlap_with_gt
    return rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights


def generate_mask_labels(im_info, gt_classes, is_crowd, gt_segms, rois,
                         labels_int32, num_classes, resolution):
    r"""

    **Generate Mask Labels for Mask-RCNN**

    This operator can be, for given the RoIs and corresponding labels,
    to sample foreground RoIs. This mask branch also has
    a :math: `K \\times M^{2}` dimensional output targets for each foreground
    RoI, which encodes K binary masks of resolution M x M, one for each of the
    K classes. This mask targets are used to compute loss of mask branch.

    Please note, the data format of groud-truth segmentation, assumed the
    segmentations are as follows. The first instance has two gt objects.
    The second instance has one gt object, this object has two gt segmentations.

        .. code-block:: python

            #[
            #  [[[229.14, 370.9, 229.14, 370.9, ...]],
            #   [[343.7, 139.85, 349.01, 138.46, ...]]], # 0-th instance
            #  [[[500.0, 390.62, ...],[115.48, 187.86, ...]]] # 1-th instance
            #]

            batch_masks = []
            for semgs in batch_semgs:
                gt_masks = []
                for semg in semgs:
                    gt_segm = []
                    for polys in semg:
                        gt_segm.append(np.array(polys).reshape(-1, 2))
                    gt_masks.append(gt_segm)
                batch_masks.append(gt_masks)
            
            
            place = fluid.CPUPlace()
            feeder = fluid.DataFeeder(place=place, feed_list=feeds)
            feeder.feed(batch_masks)

    Args:
        im_info (Variable): A 2-D Tensor with shape [N, 3] and float32
            data type. N is the batch size, each element is
            [height, width, scale] of image. Image scale is
            target_size / original_size, target_size is the size after resize,
            original_size is the original image size.
        gt_classes (Variable): A 2-D LoDTensor with shape [M, 1]. Data type
            should be int. M is the total number of ground-truth, each
            element is a class label.
        is_crowd (Variable): A 2-D LoDTensor with same shape and same data type
            as gt_classes, each element is a flag indicating whether a
            groundtruth is crowd.
        gt_segms (Variable): This input is a 2D LoDTensor with shape [S, 2] and
            float32 data type, it's LoD level is 3.
            Usually users do not needs to understand LoD,
            The users should return correct data format in reader.
            The LoD[0] represents the ground-truth objects number of
            each instance. LoD[1] represents the segmentation counts of each
            objects. LoD[2] represents the polygons number of each segmentation.
            S the total number of polygons coordinate points. Each element is
            (x, y) coordinate points.
        rois (Variable): A 2-D LoDTensor with shape [R, 4] and float32 data type
            float32. R is the total number of RoIs, each element is a bounding
            box with (xmin, ymin, xmax, ymax) format in the range of original image.
        labels_int32 (Variable): A 2-D LoDTensor in shape of [R, 1] with type
            of int32. R is the same as it in `rois`. Each element represents
            a class label of a RoI.
        num_classes (int): Class number.
        resolution (int): Resolution of mask predictions.

    Returns:
        mask_rois (Variable):  A 2D LoDTensor with shape [P, 4] and same data
        type as `rois`. P is the total number of sampled RoIs. Each element
        is a bounding box with [xmin, ymin, xmax, ymax] format in range of
        original image size.

        mask_rois_has_mask_int32 (Variable): A 2D LoDTensor with shape [P, 1]
        and int data type, each element represents the output mask RoI
        index with regard to input RoIs.

        mask_int32 (Variable): A 2D LoDTensor with shape [P, K * M * M] and int
        data type, K is the classes number and M is the resolution of mask
        predictions. Each element represents the binary mask targets.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          im_info = fluid.data(name="im_info", shape=[None, 3],
              dtype="float32")
          gt_classes = fluid.data(name="gt_classes", shape=[None, 1],
              dtype="float32", lod_level=1)
          is_crowd = fluid.data(name="is_crowd", shape=[None, 1],
              dtype="float32", lod_level=1)
          gt_masks = fluid.data(name="gt_masks", shape=[None, 2],
              dtype="float32", lod_level=3)
          # rois, roi_labels can be the output of
          # fluid.layers.generate_proposal_labels.
          rois = fluid.data(name="rois", shape=[None, 4],
              dtype="float32", lod_level=1)
          roi_labels = fluid.data(name="roi_labels", shape=[None, 1],
              dtype="int32", lod_level=1)
          mask_rois, mask_index, mask_int32 = fluid.layers.generate_mask_labels(
              im_info=im_info,
              gt_classes=gt_classes,
              is_crowd=is_crowd,
              gt_segms=gt_masks,
              rois=rois,
              labels_int32=roi_labels,
              num_classes=81,
              resolution=14)
    """

    helper = LayerHelper('generate_mask_labels', **locals())

    mask_rois = helper.create_variable_for_type_inference(dtype=rois.dtype)
    roi_has_mask_int32 = helper.create_variable_for_type_inference(
        dtype=gt_classes.dtype)
    mask_int32 = helper.create_variable_for_type_inference(
        dtype=gt_classes.dtype)

    helper.append_op(
        type="generate_mask_labels",
        inputs={
            'ImInfo': im_info,
            'GtClasses': gt_classes,
            'IsCrowd': is_crowd,
            'GtSegms': gt_segms,
            'Rois': rois,
            'LabelsInt32': labels_int32
        },
        outputs={
            'MaskRois': mask_rois,
            'RoiHasMaskInt32': roi_has_mask_int32,
            'MaskInt32': mask_int32
        },
        attrs={'num_classes': num_classes,
               'resolution': resolution})

    mask_rois.stop_gradient = True
    roi_has_mask_int32.stop_gradient = True
    mask_int32.stop_gradient = True

    return mask_rois, roi_has_mask_int32, mask_int32


def generate_proposals(scores,
                       bbox_deltas,
                       im_info,
                       anchors,
                       variances,
                       pre_nms_top_n=6000,
                       post_nms_top_n=1000,
                       nms_thresh=0.5,
                       min_size=0.1,
                       eta=1.0,
                       return_rois_num=False,
                       name=None):
    """

    **Generate proposal Faster-RCNN**

    This operation proposes RoIs according to each box with their
    probability to be a foreground object and 
    the box can be calculated by anchors. Bbox_deltais and scores
    to be an object are the output of RPN. Final proposals
    could be used to train detection net.

    For generating proposals, this operation performs following steps:

    1. Transposes and resizes scores and bbox_deltas in size of
       (H*W*A, 1) and (H*W*A, 4)
    2. Calculate box locations as proposals candidates. 
    3. Clip boxes to image
    4. Remove predicted boxes with small area. 
    5. Apply NMS to get final proposals as output.

    Args:
        scores(Variable): A 4-D Tensor with shape [N, A, H, W] represents
            the probability for each box to be an object.
            N is batch size, A is number of anchors, H and W are height and
            width of the feature map. The data type must be float32.
        bbox_deltas(Variable): A 4-D Tensor with shape [N, 4*A, H, W]
            represents the difference between predicted box location and
            anchor location. The data type must be float32.
        im_info(Variable): A 2-D Tensor with shape [N, 3] represents origin
            image information for N batch. Height and width are the input sizes 
            and scale is the ratio of network input size and original size. 
            The data type can be float32 or float64.
        anchors(Variable):   A 4-D Tensor represents the anchors with a layout
            of [H, W, A, 4]. H and W are height and width of the feature map,
            num_anchors is the box count of each position. Each anchor is
            in (xmin, ymin, xmax, ymax) format an unnormalized. The data type must be float32.
        variances(Variable): A 4-D Tensor. The expanded variances of anchors with a layout of
            [H, W, num_priors, 4]. Each variance is in
            (xcenter, ycenter, w, h) format. The data type must be float32.
        pre_nms_top_n(float): Number of total bboxes to be kept per
            image before NMS. The data type must be float32. `6000` by default.
        post_nms_top_n(float): Number of total bboxes to be kept per
            image after NMS. The data type must be float32. `1000` by default.
        nms_thresh(float): Threshold in NMS. The data type must be float32. `0.5` by default.
        min_size(float): Remove predicted boxes with either height or
            width < min_size. The data type must be float32. `0.1` by default.
        eta(float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
            `adaptive_threshold = adaptive_threshold * eta` in each iteration.
        return_rois_num(bool): When setting True, it will return a 1D Tensor with shape [N, ] that includes Rois's 
            num of each image in one batch. The N is the image's num. For example, the tensor has values [4,5] that represents
            the first image has 4 Rois, the second image has 5 Rois. It only used in rcnn model. 
            'False' by default. 
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        tuple:
        A tuple with format ``(rpn_rois, rpn_roi_probs)``.

        - **rpn_rois**: The generated RoIs. 2-D Tensor with shape ``[N, 4]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
        - **rpn_roi_probs**: The scores of generated RoIs. 2-D Tensor with shape ``[N, 1]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.

    Examples:
        .. code-block:: python
        
            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            scores = fluid.data(name='scores', shape=[None, 4, 5, 5], dtype='float32')
            bbox_deltas = fluid.data(name='bbox_deltas', shape=[None, 16, 5, 5], dtype='float32')
            im_info = fluid.data(name='im_info', shape=[None, 3], dtype='float32')
            anchors = fluid.data(name='anchors', shape=[None, 5, 4, 4], dtype='float32')
            variances = fluid.data(name='variances', shape=[None, 5, 10, 4], dtype='float32')
            rois, roi_probs = fluid.layers.generate_proposals(scores, bbox_deltas,
                         im_info, anchors, variances)

    """
    if in_dygraph_mode():
        assert return_rois_num, "return_rois_num should be True in dygraph mode."
        attrs = ('pre_nms_topN', pre_nms_top_n, 'post_nms_topN', post_nms_top_n,
                 'nms_thresh', nms_thresh, 'min_size', min_size, 'eta', eta)
        rpn_rois, rpn_roi_probs, rpn_rois_num = _C_ops.generate_proposals(
            scores, bbox_deltas, im_info, anchors, variances, *attrs)
        return rpn_rois, rpn_roi_probs, rpn_rois_num

    helper = LayerHelper('generate_proposals', **locals())

    check_variable_and_dtype(scores, 'scores', ['float32'],
                             'generate_proposals')
    check_variable_and_dtype(bbox_deltas, 'bbox_deltas', ['float32'],
                             'generate_proposals')
    check_variable_and_dtype(im_info, 'im_info', ['float32', 'float64'],
                             'generate_proposals')
    check_variable_and_dtype(anchors, 'anchors', ['float32'],
                             'generate_proposals')
    check_variable_and_dtype(variances, 'variances', ['float32'],
                             'generate_proposals')

    rpn_rois = helper.create_variable_for_type_inference(
        dtype=bbox_deltas.dtype)
    rpn_roi_probs = helper.create_variable_for_type_inference(
        dtype=scores.dtype)
    outputs = {
        'RpnRois': rpn_rois,
        'RpnRoiProbs': rpn_roi_probs,
    }
    if return_rois_num:
        rpn_rois_num = helper.create_variable_for_type_inference(dtype='int32')
        rpn_rois_num.stop_gradient = True
        outputs['RpnRoisNum'] = rpn_rois_num

    helper.append_op(
        type="generate_proposals",
        inputs={
            'Scores': scores,
            'BboxDeltas': bbox_deltas,
            'ImInfo': im_info,
            'Anchors': anchors,
            'Variances': variances
        },
        attrs={
            'pre_nms_topN': pre_nms_top_n,
            'post_nms_topN': post_nms_top_n,
            'nms_thresh': nms_thresh,
            'min_size': min_size,
            'eta': eta
        },
        outputs=outputs)
    rpn_rois.stop_gradient = True
    rpn_roi_probs.stop_gradient = True

    if return_rois_num:
        return rpn_rois, rpn_roi_probs, rpn_rois_num
    else:
        return rpn_rois, rpn_roi_probs


def box_clip(input, im_info, name=None):
    """
	
    Clip the box into the size given by im_info
    For each input box, The formula is given as follows:
        
    .. code-block:: text

        xmin = max(min(xmin, im_w - 1), 0)
        ymin = max(min(ymin, im_h - 1), 0) 
        xmax = max(min(xmax, im_w - 1), 0)
        ymax = max(min(ymax, im_h - 1), 0)
    
    where im_w and im_h are computed from im_info:
 
    .. code-block:: text

        im_h = round(height / scale)
        im_w = round(weight / scale)

    Args:
        input(Variable): The input Tensor with shape :math:`[N_1, N_2, ..., N_k, 4]`,
            the last dimension is 4 and data type is float32 or float64.
        im_info(Variable): The 2-D Tensor with shape [N, 3] with layout 
            (height, width, scale) representing the information of image. 
            Height and width are the input sizes and scale is the ratio of network input
            size and original size. The data type is float32 or float64.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 
    
    Returns:
        Variable:

        output(Variable): The clipped tensor with data type float32 or float64. 
        The shape is same as input.

        
    Examples:
        .. code-block:: python
        
            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            boxes = fluid.data(
                name='boxes', shape=[None, 8, 4], dtype='float32', lod_level=1)
            im_info = fluid.data(name='im_info', shape=[-1 ,3])
            out = fluid.layers.box_clip(
                input=boxes, im_info=im_info)
    """

    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'box_clip')
    check_variable_and_dtype(im_info, 'im_info', ['float32', 'float64'],
                             'box_clip')

    helper = LayerHelper("box_clip", **locals())
    output = helper.create_variable_for_type_inference(dtype=input.dtype)
    inputs = {"Input": input, "ImInfo": im_info}
    helper.append_op(type="box_clip", inputs=inputs, outputs={"Output": output})

    return output


def retinanet_detection_output(bboxes,
                               scores,
                               anchors,
                               im_info,
                               score_threshold=0.05,
                               nms_top_k=1000,
                               keep_top_k=100,
                               nms_threshold=0.3,
                               nms_eta=1.0):
    """
    **Detection Output Layer for the detector RetinaNet.**

    In the detector `RetinaNet <https://arxiv.org/abs/1708.02002>`_ , many 
    `FPN <https://arxiv.org/abs/1612.03144>`_ levels output the category
    and location predictions, this OP is to get the detection results by
    performing following steps:

    1. For each FPN level, decode box predictions according to the anchor
       boxes from at most :attr:`nms_top_k` top-scoring predictions after
       thresholding detector confidence at :attr:`score_threshold`.
    2. Merge top predictions from all levels and apply multi-class non 
       maximum suppression (NMS) on them to get the final detections.

    Args:
        bboxes(List): A list of Tensors from multiple FPN levels represents
            the location prediction for all anchor boxes. Each element is
            a 3-D Tensor with shape :math:`[N, Mi, 4]`, :math:`N` is the
            batch size, :math:`Mi` is the number of bounding boxes from
            :math:`i`-th FPN level and each bounding box has four coordinate
            values and the layout is [xmin, ymin, xmax, ymax]. The data type
            of each element is float32 or float64.
        scores(List): A list of Tensors from multiple FPN levels represents
            the category prediction for all anchor boxes. Each element is a
            3-D Tensor with shape :math:`[N, Mi, C]`,  :math:`N` is the batch
            size, :math:`C` is the class number (**excluding background**),
            :math:`Mi` is the number of bounding boxes from :math:`i`-th FPN
            level. The data type of each element is float32 or float64.
        anchors(List): A list of Tensors from multiple FPN levels represents
            the locations of all anchor boxes. Each element is a 2-D Tensor
            with shape :math:`[Mi, 4]`, :math:`Mi` is the number of bounding
            boxes from :math:`i`-th FPN level, and each bounding box has four
            coordinate values and the layout is [xmin, ymin, xmax, ymax].
            The data type of each element is float32 or float64.
        im_info(Variable): A 2-D Tensor with shape :math:`[N, 3]` represents the size
            information of input images. :math:`N` is the batch size, the size
            information of each image is a 3-vector which are the height and width
            of the network input along with the factor scaling the origin image to
            the network input. The data type of :attr:`im_info` is float32.
        score_threshold(float): Threshold to filter out bounding boxes
            with a confidence score before NMS, default value is set to 0.05.
        nms_top_k(int): Maximum number of detections per FPN layer to be
            kept according to the confidences before NMS, default value is set to
            1000.
        keep_top_k(int): Number of total bounding boxes to be kept per image after
            NMS step. Default value is set to 100, -1 means keeping all bounding
            boxes after NMS step.
        nms_threshold(float): The Intersection-over-Union(IoU) threshold used to 
            filter out boxes in NMS.
        nms_eta(float): The parameter for adjusting :attr:`nms_threshold` in NMS.
            Default value is set to 1., which represents the value of
            :attr:`nms_threshold` keep the same in NMS. If :attr:`nms_eta` is set
            to be lower than 1. and the value of :attr:`nms_threshold` is set to
            be higher than 0.5, everytime a bounding box is filtered out,
            the adjustment for :attr:`nms_threshold` like :attr:`nms_threshold`
            = :attr:`nms_threshold` * :attr:`nms_eta`  will not be stopped until
            the actual value of :attr:`nms_threshold` is lower than or equal to
            0.5.

    **Notice**: In some cases where the image sizes are very small, it's possible
    that there is no detection if :attr:`score_threshold` are used at all
    levels. Hence, this OP do not filter out anchors from the highest FPN level
    before NMS. And the last element in :attr:`bboxes`:, :attr:`scores` and
    :attr:`anchors` is required to be from the highest FPN level.

    Returns:
        Variable(The data type is float32 or float64):
            The detection output is a 1-level LoDTensor with shape :math:`[No, 6]`.
            Each row has six values: [label, confidence, xmin, ymin, xmax, ymax].
            :math:`No` is the total number of detections in this mini-batch.
            The :math:`i`-th image has `LoD[i + 1] - LoD[i]` detected
            results, if `LoD[i + 1] - LoD[i]` is 0, the :math:`i`-th image
            has no detected results. If all images have no detected results,
            LoD will be set to 0, and the output tensor is empty (None).

    Examples:
        .. code-block:: python

           import paddle.fluid as fluid

           bboxes_low = fluid.data(
               name='bboxes_low', shape=[1, 44, 4], dtype='float32')
           bboxes_high = fluid.data(
               name='bboxes_high', shape=[1, 11, 4], dtype='float32')
           scores_low = fluid.data(
               name='scores_low', shape=[1, 44, 10], dtype='float32')
           scores_high = fluid.data(
               name='scores_high', shape=[1, 11, 10], dtype='float32')
           anchors_low = fluid.data(
               name='anchors_low', shape=[44, 4], dtype='float32')
           anchors_high = fluid.data(
               name='anchors_high', shape=[11, 4], dtype='float32')
           im_info = fluid.data(
               name="im_info", shape=[1, 3], dtype='float32')
           nmsed_outs = fluid.layers.retinanet_detection_output(
               bboxes=[bboxes_low, bboxes_high],
               scores=[scores_low, scores_high],
               anchors=[anchors_low, anchors_high],
               im_info=im_info,
               score_threshold=0.05,
               nms_top_k=1000,
               keep_top_k=100,
               nms_threshold=0.45,
               nms_eta=1.0)
    """

    check_type(bboxes, 'bboxes', (list), 'retinanet_detection_output')
    for i, bbox in enumerate(bboxes):
        check_variable_and_dtype(bbox, 'bbox{}'.format(i),
                                 ['float32', 'float64'],
                                 'retinanet_detection_output')
    check_type(scores, 'scores', (list), 'retinanet_detection_output')
    for i, score in enumerate(scores):
        check_variable_and_dtype(score, 'score{}'.format(i),
                                 ['float32', 'float64'],
                                 'retinanet_detection_output')
    check_type(anchors, 'anchors', (list), 'retinanet_detection_output')
    for i, anchor in enumerate(anchors):
        check_variable_and_dtype(anchor, 'anchor{}'.format(i),
                                 ['float32', 'float64'],
                                 'retinanet_detection_output')
    check_variable_and_dtype(im_info, 'im_info', ['float32', 'float64'],
                             'retinanet_detection_output')

    helper = LayerHelper('retinanet_detection_output', **locals())
    output = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('scores'))
    helper.append_op(
        type="retinanet_detection_output",
        inputs={
            'BBoxes': bboxes,
            'Scores': scores,
            'Anchors': anchors,
            'ImInfo': im_info
        },
        attrs={
            'score_threshold': score_threshold,
            'nms_top_k': nms_top_k,
            'nms_threshold': nms_threshold,
            'keep_top_k': keep_top_k,
            'nms_eta': 1.,
        },
        outputs={'Out': output})
    output.stop_gradient = True
    return output


def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   nms_top_k,
                   keep_top_k,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=0,
                   name=None):
    """

    **Multiclass NMS**
    
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.

    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.

    See below for an example:

    .. code-block:: text

        if:
            box1.data = (2.0, 3.0, 7.0, 5.0) format is (xmin, ymin, xmax, ymax)
            box1.scores = (0.7, 0.2, 0.4)  which is (label0.score=0.7, label1.score=0.2, label2.cores=0.4)

            box2.data = (3.0, 4.0, 8.0, 5.0)
            box2.score = (0.3, 0.3, 0.1)

            nms_threshold = 0.3
            background_label = 0
            score_threshold = 0


        Then:
            iou = 4/11 > 0.3
            out.data = [[1, 0.3, 3.0, 4.0, 8.0, 5.0],    
                         [2, 0.4, 2.0, 3.0, 7.0, 5.0]]
                         
            Out format is (label, confidence, xmin, ymin, xmax, ymax)
    Args:
        bboxes (Variable): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is 
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           The data type is float32 or float64.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the 
                           class number. The data type is float32 or float64.   
        scores (Variable): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is 
                           number of bounding boxes. For each category there 
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.The data type is float32 or float64. 
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].The data type is float32 or float64. 
        background_label (int): The index of background label, the background 
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided, 
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        name(str): Name of the multiclass nms op. Default: None.

    Returns:
        Variable: A 2-D LoDTensor with shape [No, 6] represents the detections.
             Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
             or A 2-D LoDTensor with shape [No, 10] represents the detections.
             Each row has 10 values: 
             [label, confidence, x1, y1, x2, y2, x3, y3, x4, y4]. No is the 
             total number of detections. If there is no detected boxes for all
             images, lod will be set to {1} and Out only contains one value
             which is -1.
             (After version 1.3, when no boxes detected, the lod is changed 
             from {0} to {1}) 


    Examples:
        .. code-block:: python


            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            boxes = fluid.data(name='bboxes', shape=[None,81, 4],
                                      dtype='float32', lod_level=1)
            scores = fluid.data(name='scores', shape=[None,81],
                                      dtype='float32', lod_level=1)
            out = fluid.layers.multiclass_nms(bboxes=boxes,
                                              scores=scores,
                                              background_label=0,
                                              score_threshold=0.5,
                                              nms_top_k=400,
                                              nms_threshold=0.3,
                                              keep_top_k=200,
                                              normalized=False)
    """
    check_variable_and_dtype(bboxes, 'BBoxes', ['float32', 'float64'],
                             'multiclass_nms')
    check_variable_and_dtype(scores, 'Scores', ['float32', 'float64'],
                             'multiclass_nms')
    check_type(score_threshold, 'score_threshold', float, 'multicalss_nms')
    check_type(nms_top_k, 'nums_top_k', int, 'multiclass_nms')
    check_type(keep_top_k, 'keep_top_k', int, 'mutliclass_nms')
    check_type(nms_threshold, 'nms_threshold', float, 'multiclass_nms')
    check_type(normalized, 'normalized', bool, 'multiclass_nms')
    check_type(nms_eta, 'nms_eta', float, 'multiclass_nms')
    check_type(background_label, 'background_label', int, 'multiclass_nms')

    helper = LayerHelper('multiclass_nms', **locals())
    output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
    helper.append_op(
        type="multiclass_nms",
        inputs={'BBoxes': bboxes,
                'Scores': scores},
        attrs={
            'background_label': background_label,
            'score_threshold': score_threshold,
            'nms_top_k': nms_top_k,
            'nms_threshold': nms_threshold,
            'nms_eta': nms_eta,
            'keep_top_k': keep_top_k,
            'normalized': normalized
        },
        outputs={'Out': output})
    output.stop_gradient = True

    return output


def locality_aware_nms(bboxes,
                       scores,
                       score_threshold,
                       nms_top_k,
                       keep_top_k,
                       nms_threshold=0.3,
                       normalized=True,
                       nms_eta=1.,
                       background_label=-1,
                       name=None):
    """
    **Local Aware NMS**
    
    `Local Aware NMS <https://arxiv.org/abs/1704.03155>`_ is to do locality-aware non maximum
    suppression (LANMS) on boxes and scores.

    Firstly, this operator merge box and score according their IOU
    (intersection over union). In the NMS step, this operator greedily selects a
    subset of detection bounding boxes that have high scores larger than score_threshold,
    if providing this threshold, then selects the largest nms_top_k confidences scores
    if nms_top_k is larger than -1. Then this operator pruns away boxes that have high
    IOU overlap with already selected boxes by adaptive threshold NMS based on parameters
    of nms_threshold and nms_eta.

    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.

    Args:
        bboxes (Variable): A 3-D Tensor with shape [N, M, 4 or 8 16 24 32]
                           represents the predicted locations of M bounding
                           bboxes, N is the batch size. Each bounding box
                           has four coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           The data type is float32 or float64.
        scores (Variable): A 3-D Tensor with shape [N, C, M] represents the
                           predicted confidence predictions. N is the batch
                           size, C is the class number, M is number of bounding
                           boxes. Now only support 1 class. For each category
                           there are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension of
                           BBoxes. The data type is float32 or float64.
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: -1
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        normalized (bool): Whether detections are normalized. Default: True
        name(str): Name of the locality aware nms op, please refer to :ref:`api_guide_Name` .
                          Default: None.

    Returns:
        Variable: A 2-D LoDTensor with shape [No, 6] represents the detections.
             Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
             or A 2-D LoDTensor with shape [No, 10] represents the detections.
             Each row has 10 values:
             [label, confidence, x1, y1, x2, y2, x3, y3, x4, y4]. No is the
             total number of detections. If there is no detected boxes for all
             images, lod will be set to {1} and Out only contains one value
             which is -1.
             (After version 1.3, when no boxes detected, the lod is changed
             from {0} to {1}). The data type is float32 or float64.


    Examples:
        .. code-block:: python


            import paddle.fluid as fluid
            boxes = fluid.data(name='bboxes', shape=[None, 81, 8],
                                      dtype='float32')
            scores = fluid.data(name='scores', shape=[None, 1, 81],
                                      dtype='float32')
            out = fluid.layers.locality_aware_nms(bboxes=boxes,
                                              scores=scores,
                                              score_threshold=0.5,
                                              nms_top_k=400,
                                              nms_threshold=0.3,
                                              keep_top_k=200,
                                              normalized=False)
    """
    check_variable_and_dtype(bboxes, 'bboxes', ['float32', 'float64'],
                             'locality_aware_nms')
    check_variable_and_dtype(scores, 'scores', ['float32', 'float64'],
                             'locality_aware_nms')
    check_type(background_label, 'background_label', int, 'locality_aware_nms')
    check_type(score_threshold, 'score_threshold', float, 'locality_aware_nms')
    check_type(nms_top_k, 'nms_top_k', int, 'locality_aware_nms')
    check_type(nms_eta, 'nms_eta', float, 'locality_aware_nms')
    check_type(nms_threshold, 'nms_threshold', float, 'locality_aware_nms')
    check_type(keep_top_k, 'keep_top_k', int, 'locality_aware_nms')
    check_type(normalized, 'normalized', bool, 'locality_aware_nms')

    shape = scores.shape
    assert len(shape) == 3, "dim size of scores must be 3"
    assert shape[
        1] == 1, "locality_aware_nms only support one class, Tensor score shape must be [N, 1, M]"

    helper = LayerHelper('locality_aware_nms', **locals())

    output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
    out = {'Out': output}

    helper.append_op(
        type="locality_aware_nms",
        inputs={'BBoxes': bboxes,
                'Scores': scores},
        attrs={
            'background_label': background_label,
            'score_threshold': score_threshold,
            'nms_top_k': nms_top_k,
            'nms_threshold': nms_threshold,
            'nms_eta': nms_eta,
            'keep_top_k': keep_top_k,
            'nms_eta': nms_eta,
            'normalized': normalized
        },
        outputs={'Out': output})
    output.stop_gradient = True

    return output


def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.,
               background_label=0,
               normalized=True,
               return_index=False,
               name=None):
    """
    **Matrix NMS**

    This operator does matrix non maximum suppression (NMS).

    First selects a subset of candidate bounding boxes that have higher scores
    than score_threshold (if provided), then the top k candidate is selected if
    nms_top_k is larger than -1. Score of the remaining candidate are then
    decayed according to the Matrix NMS scheme.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.

    Args:
        bboxes (Variable): A 3-D Tensor with shape [N, M, 4] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           The data type is float32 or float64.
        scores (Variable): A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes. The data type is float32 or float64.
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score.
        post_threshold (float): Threshold to filter out bounding boxes with
                                low confidence score AFTER decaying.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        use_gaussian (bool): Use Gaussian as the decay function. Default: False
        gaussian_sigma (float): Sigma for Gaussian decay function. Default: 2.0
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        name(str): Name of the matrix nms op. Default: None.

    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, one Variable(Out) is returned.

        Out (Variable): A 2-D LoDTensor with shape [No, 6] containing the
             detection results.
             Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
             (After version 1.3, when no boxes detected, the lod is changed
             from {0} to {1})

        Index (Variable): A 2-D LoDTensor with shape [No, 1] containing the
            selected indices, which are absolute values cross batches.

    Examples:
        .. code-block:: python


            import paddle.fluid as fluid
            boxes = fluid.data(name='bboxes', shape=[None,81, 4],
                                      dtype='float32', lod_level=1)
            scores = fluid.data(name='scores', shape=[None,81],
                                      dtype='float32', lod_level=1)
            out = fluid.layers.matrix_nms(bboxes=boxes,
                                          scores=scores,
                                          background_label=0,
                                          score_threshold=0.5,
                                          post_threshold=0.1,
                                          nms_top_k=400,
                                          keep_top_k=200,
                                          normalized=False)
    """
    check_variable_and_dtype(bboxes, 'BBoxes', ['float32', 'float64'],
                             'matrix_nms')
    check_variable_and_dtype(scores, 'Scores', ['float32', 'float64'],
                             'matrix_nms')
    check_type(score_threshold, 'score_threshold', float, 'matrix_nms')
    check_type(post_threshold, 'post_threshold', float, 'matrix_nms')
    check_type(nms_top_k, 'nums_top_k', int, 'matrix_nms')
    check_type(keep_top_k, 'keep_top_k', int, 'matrix_nms')
    check_type(normalized, 'normalized', bool, 'matrix_nms')
    check_type(use_gaussian, 'use_gaussian', bool, 'matrix_nms')
    check_type(gaussian_sigma, 'gaussian_sigma', float, 'matrix_nms')
    check_type(background_label, 'background_label', int, 'matrix_nms')

    helper = LayerHelper('matrix_nms', **locals())
    output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
    index = helper.create_variable_for_type_inference(dtype='int')
    helper.append_op(
        type="matrix_nms",
        inputs={'BBoxes': bboxes,
                'Scores': scores},
        attrs={
            'background_label': background_label,
            'score_threshold': score_threshold,
            'post_threshold': post_threshold,
            'nms_top_k': nms_top_k,
            'gaussian_sigma': gaussian_sigma,
            'use_gaussian': use_gaussian,
            'keep_top_k': keep_top_k,
            'normalized': normalized
        },
        outputs={'Out': output,
                 'Index': index})
    output.stop_gradient = True

    if return_index:
        return output, index
    else:
        return output


def distribute_fpn_proposals(fpn_rois,
                             min_level,
                             max_level,
                             refer_level,
                             refer_scale,
                             rois_num=None,
                             name=None):
    r"""
	
    **This op only takes LoDTensor as input.** In Feature Pyramid Networks 
    (FPN) models, it is needed to distribute all proposals into different FPN 
    level, with respect to scale of the proposals, the referring scale and the 
    referring level. Besides, to restore the order of proposals, we return an 
    array which indicates the original index of rois in current proposals. 
    To compute FPN level for each roi, the formula is given as follows:
    
    .. math::

        roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}

        level = floor(&\log(\\frac{roi\_scale}{refer\_scale}) + refer\_level)

    where BBoxArea is a function to compute the area of each roi.

    Args:

        fpn_rois(Variable): 2-D Tensor with shape [N, 4] and data type is 
            float32 or float64. The input fpn_rois.
        min_level(int32): The lowest level of FPN layer where the proposals come 
            from.
        max_level(int32): The highest level of FPN layer where the proposals
            come from.
        refer_level(int32): The referring level of FPN layer with specified scale.
        refer_scale(int32): The referring scale of FPN layer with specified level.
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Tuple:

        multi_rois(List) : A list of 2-D LoDTensor with shape [M, 4] 
        and data type of float32 and float64. The length is 
        max_level-min_level+1. The proposals in each FPN level.

        restore_ind(Variable): A 2-D Tensor with shape [N, 1], N is 
        the number of total rois. The data type is int32. It is
        used to restore the order of fpn_rois.

        rois_num_per_level(List): A list of 1-D Tensor and each Tensor is 
        the RoIs' number in each image on the corresponding level. The shape 
        is [B] and data type of int32. B is the number of images


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            fpn_rois = fluid.data(
                name='data', shape=[None, 4], dtype='float32', lod_level=1)
            multi_rois, restore_ind = fluid.layers.distribute_fpn_proposals(
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224)
    """
    num_lvl = max_level - min_level + 1

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        attrs = ('min_level', min_level, 'max_level', max_level, 'refer_level',
                 refer_level, 'refer_scale', refer_scale)
        multi_rois, restore_ind, rois_num_per_level = _C_ops.distribute_fpn_proposals(
            fpn_rois, rois_num, num_lvl, num_lvl, *attrs)
        return multi_rois, restore_ind, rois_num_per_level

    check_variable_and_dtype(fpn_rois, 'fpn_rois', ['float32', 'float64'],
                             'distribute_fpn_proposals')
    helper = LayerHelper('distribute_fpn_proposals', **locals())
    dtype = helper.input_dtype('fpn_rois')
    multi_rois = [
        helper.create_variable_for_type_inference(dtype) for i in range(num_lvl)
    ]

    restore_ind = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {'FpnRois': fpn_rois}
    outputs = {
        'MultiFpnRois': multi_rois,
        'RestoreIndex': restore_ind,
    }

    if rois_num is not None:
        inputs['RoisNum'] = rois_num
        rois_num_per_level = [
            helper.create_variable_for_type_inference(dtype='int32')
            for i in range(num_lvl)
        ]
        outputs['MultiLevelRoIsNum'] = rois_num_per_level

    helper.append_op(
        type='distribute_fpn_proposals',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'min_level': min_level,
            'max_level': max_level,
            'refer_level': refer_level,
            'refer_scale': refer_scale
        })
    if rois_num is not None:
        return multi_rois, restore_ind, rois_num_per_level
    return multi_rois, restore_ind


@templatedoc()
def box_decoder_and_assign(prior_box,
                           prior_box_var,
                           target_box,
                           box_score,
                           box_clip,
                           name=None):
    """
	
    ${comment}
    Args:
        prior_box(${prior_box_type}): ${prior_box_comment}
        prior_box_var(${prior_box_var_type}): ${prior_box_var_comment}
        target_box(${target_box_type}): ${target_box_comment}
        box_score(${box_score_type}): ${box_score_comment}
        box_clip(${box_clip_type}): ${box_clip_comment}
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Tuple:

        decode_box(${decode_box_type}): ${decode_box_comment}

        output_assign_box(${output_assign_box_type}): ${output_assign_box_comment}


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            pb = fluid.data(
                name='prior_box', shape=[None, 4], dtype='float32')
            pbv = fluid.data(
                name='prior_box_var', shape=[4], dtype='float32')
            loc = fluid.data(
                name='target_box', shape=[None, 4*81], dtype='float32')
            scores = fluid.data(
                name='scores', shape=[None, 81], dtype='float32')
            decoded_box, output_assign_box = fluid.layers.box_decoder_and_assign(
                pb, pbv, loc, scores, 4.135)

    """
    check_variable_and_dtype(prior_box, 'prior_box', ['float32', 'float64'],
                             'box_decoder_and_assign')
    check_variable_and_dtype(target_box, 'target_box', ['float32', 'float64'],
                             'box_decoder_and_assign')
    check_variable_and_dtype(box_score, 'box_score', ['float32', 'float64'],
                             'box_decoder_and_assign')
    helper = LayerHelper("box_decoder_and_assign", **locals())

    decoded_box = helper.create_variable_for_type_inference(
        dtype=prior_box.dtype)
    output_assign_box = helper.create_variable_for_type_inference(
        dtype=prior_box.dtype)

    helper.append_op(
        type="box_decoder_and_assign",
        inputs={
            "PriorBox": prior_box,
            "PriorBoxVar": prior_box_var,
            "TargetBox": target_box,
            "BoxScore": box_score
        },
        attrs={"box_clip": box_clip},
        outputs={
            "DecodeBox": decoded_box,
            "OutputAssignBox": output_assign_box
        })
    return decoded_box, output_assign_box


def collect_fpn_proposals(multi_rois,
                          multi_scores,
                          min_level,
                          max_level,
                          post_nms_top_n,
                          rois_num_per_level=None,
                          name=None):
    """
	
    **This OP only supports LoDTensor as input**. Concat multi-level RoIs 
    (Region of Interest) and select N RoIs with respect to multi_scores. 
    This operation performs the following steps:

    1. Choose num_level RoIs and scores as input: num_level = max_level - min_level
    2. Concat multi-level RoIs and scores
    3. Sort scores and select post_nms_top_n scores
    4. Gather RoIs by selected indices from scores
    5. Re-sort RoIs by corresponding batch_id

    Args:
        multi_rois(list): List of RoIs to collect. Element in list is 2-D 
            LoDTensor with shape [N, 4] and data type is float32 or float64, 
            N is the number of RoIs.
        multi_scores(list): List of scores of RoIs to collect. Element in list 
            is 2-D LoDTensor with shape [N, 1] and data type is float32 or
            float64, N is the number of RoIs.
        min_level(int): The lowest level of FPN layer to collect
        max_level(int): The highest level of FPN layer to collect
        post_nms_top_n(int): The number of selected RoIs
        rois_num_per_level(list, optional): The List of RoIs' numbers. 
            Each element is 1-D Tensor which contains the RoIs' number of each 
            image on each level and the shape is [B] and data type is 
            int32, B is the number of images. If it is not None then return 
            a 1-D Tensor contains the output RoIs' number of each image and 
            the shape is [B]. Default: None
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.        

    Returns:
        Variable:

        fpn_rois(Variable): 2-D LoDTensor with shape [N, 4] and data type is 
        float32 or float64. Selected RoIs. 

        rois_num(Tensor): 1-D Tensor contains the RoIs's number of each 
        image. The shape is [B] and data type is int32. B is the number of 
        images. 

    Examples:
        .. code-block:: python
           
            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            multi_rois = []
            multi_scores = []
            for i in range(4):
                multi_rois.append(fluid.data(
                    name='roi_'+str(i), shape=[None, 4], dtype='float32', lod_level=1))
            for i in range(4):
                multi_scores.append(fluid.data(
                    name='score_'+str(i), shape=[None, 1], dtype='float32', lod_level=1))

            fpn_rois = fluid.layers.collect_fpn_proposals(
                multi_rois=multi_rois, 
                multi_scores=multi_scores,
                min_level=2, 
                max_level=5, 
                post_nms_top_n=2000)
    """
    num_lvl = max_level - min_level + 1
    input_rois = multi_rois[:num_lvl]
    input_scores = multi_scores[:num_lvl]

    if in_dygraph_mode():
        assert rois_num_per_level is not None, "rois_num_per_level should not be None in dygraph mode."
        attrs = ('post_nms_topN', post_nms_top_n)
        output_rois, rois_num = _C_ops.collect_fpn_proposals(
            input_rois, input_scores, rois_num_per_level, *attrs)

    check_type(multi_rois, 'multi_rois', list, 'collect_fpn_proposals')
    check_type(multi_scores, 'multi_scores', list, 'collect_fpn_proposals')
    helper = LayerHelper('collect_fpn_proposals', **locals())
    dtype = helper.input_dtype('multi_rois')
    check_dtype(dtype, 'multi_rois', ['float32', 'float64'],
                'collect_fpn_proposals')
    output_rois = helper.create_variable_for_type_inference(dtype)
    output_rois.stop_gradient = True

    inputs = {
        'MultiLevelRois': input_rois,
        'MultiLevelScores': input_scores,
    }
    outputs = {'FpnRois': output_rois}
    if rois_num_per_level is not None:
        inputs['MultiLevelRoIsNum'] = rois_num_per_level
        rois_num = helper.create_variable_for_type_inference(dtype='int32')
        rois_num.stop_gradient = True
        outputs['RoisNum'] = rois_num
    helper.append_op(
        type='collect_fpn_proposals',
        inputs=inputs,
        outputs=outputs,
        attrs={'post_nms_topN': post_nms_top_n})
    if rois_num_per_level is not None:
        return output_rois, rois_num
    return output_rois
