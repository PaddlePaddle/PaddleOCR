#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
Fluid Metrics
"""

from __future__ import print_function

import numpy as np
import copy
import warnings
import six

from .layer_helper import LayerHelper
from .initializer import Constant
from . import unique_name
from .framework import Program, Variable, program_guard
from . import layers
from .layers import detection

__all__ = [
    'MetricBase',
    'CompositeMetric',
    'Precision',
    'Recall',
    'Accuracy',
    'ChunkEvaluator',
    'EditDistance',
    'DetectionMAP',
    'Auc',
]


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


def _is_number_(var):
    return isinstance(var, int) or isinstance(var, np.int64) or isinstance(
        var, float) or (isinstance(var, np.ndarray) and var.shape == (1, ))


def _is_number_or_matrix_(var):
    return _is_number_(var) or isinstance(var, np.ndarray)


class MetricBase(object):
    """
    In many cases, we usually have to split the test data into mini-batches for evaluating 
    deep neural networks, therefore we need to collect the evaluation results of each 
    mini-batch and aggregate them into the final result. The paddle.fluid.metrics is 
    designed for a convenient way of deep neural network evaluation. 

    The paddle.fluid.metrics contains serval different evaluation metrics 
    like precision and recall, and most of them have the following functions:

    1. take the prediction result and the corresponding labels of a mini-batch as input, 
    then compute the evaluation result for the input mini-batch.

    2. aggregate the existing evaluation results as the overall performance.

    The class Metric is the base class for all classes in paddle.fluid.metrics, it defines
    the fundamental APIs for all metrics classes, including:

    1. update(preds, labels): given the prediction results (preds) and the labels (labels)
    of some mini-batch, compute the evaluation result of that mini-batch, and memorize the
    evaluation result.

    2. eval(): aggregate all existing evaluation result in the memory, and return the overall
    performance across different mini-batches.

    3. reset(): empty the memory.

    """

    def __init__(self, name):
        """
        The constructor of the metric class.

        Args:
            name(str): The name of metric instance. such as, "accuracy".
                  It can be used to distinguish different metric instances in a model.

        Returns:
            The constructed class instance.

        Return types:
            The MetricBase or its succeed classes

        """
        self._name = str(name) if name != None else self.__class__.__name__

    def __str__(self):
        return self._name

    def reset(self):
        """
        reset function empties the evaluation memory for previous mini-batches. 
        
        Args:
            None

        Returns:
            None

        Return types:
            None

        """
        states = {
            attr: value
            for attr, value in six.iteritems(self.__dict__)
            if not attr.startswith("_")
        }
        for attr, value in six.iteritems(states):
            if isinstance(value, int):
                setattr(self, attr, 0)
            elif isinstance(value, float):
                setattr(self, attr, .0)
            elif isinstance(value, (np.ndarray, np.generic)):
                setattr(self, attr, np.zeros_like(value))
            else:
                setattr(self, attr, None)

    def get_config(self):
        """
        Get the metric and current states.
        The states are the members who do not has "_" prefix.

        Args:
            None

        Returns:
            a python dict, which contains the inner states of the metric instance

        Return types:
            a python dict
        """
        states = {
            attr: value
            for attr, value in six.iteritems(self.__dict__)
            if not attr.startswith("_")
        }
        config = {}
        config.update({"name": self._name, "states": copy.deepcopy(states)})
        return config

    def update(self, preds, labels):
        """
        Given the prediction results (preds) and the labels (labels)
        of some mini-batch, compute the evaluation result of that mini-batch, 
        and memorize the evaluation result. Please notice that the update function only
        memorizes the evaluation result but would not return the score. If you want to 
        get the evaluation result, please call eval() function.

        Args:
            preds(numpy.array): the predictions of current minibatch
            labels(numpy.array): the labels of current minibatch.

        Returns:
            None

        Return types:
            None        

        """
        raise NotImplementedError(
            "Should not use it directly, please extend it.")

    def eval(self):
        """
        Aggregate all existing evaluation results in the memory, and return the overall
        performance across different mini-batches.

        Args:
            None

        Returns:
            The overall performance across different mini-batches.

        Return types:
            float|list(float)|numpy.array: the metrics via Python.
        """
        raise NotImplementedError(
            "Should not use it directly, please extend it.")


class CompositeMetric(MetricBase):
    """
    This op creates a container that contains the union of all the added metrics. 
    After the metrics added in, calling eval() method will compute all the contained metrics automatically.
    CAUTION: only metrics with the SAME argument list can be added in a CompositeMetric instance.

    Inherit from: `MetricBase <https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/metrics_cn.html#paddle.fluid.metrics.MetricBase>`_ 

    Args:
       name (str, optional): Metric name. For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            import numpy as np
            preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
                     [0.2], [0.3], [0.5], [0.8], [0.6]]
            labels = [[0], [1], [1], [1], [1],
                      [0], [0], [0], [0], [0]]
            preds = np.array(preds)
            labels = np.array(labels)
            comp = fluid.metrics.CompositeMetric()
            precision = fluid.metrics.Precision()
            recall = fluid.metrics.Recall()
            comp.add_metric(precision)
            comp.add_metric(recall)
            comp.update(preds=preds, labels=labels)
            numpy_precision, numpy_recall = comp.eval()
            print("expect precision: %.2f, got %.2f" % ( 3. / 5, numpy_precision ) )
            print("expect recall: %.2f, got %.2f" % (3. / 4, numpy_recall ) )
    """

    def __init__(self, name=None):
        super(CompositeMetric, self).__init__(name)
        self._metrics = []

    def add_metric(self, metric):
        """
        Add a new metric to container. Noted that the argument list 
        of the added one should be consistent with existed ones.  

        Args:
            metric(MetricBase): a instance of MetricBase
        """
        if not isinstance(metric, MetricBase):
            raise ValueError("SubMetric should be inherit from MetricBase.")
        self._metrics.append(metric)

    def update(self, preds, labels):
        """
        Update the metrics of this container.

        Args:
            preds(numpy.array): predicted results of current mini-batch, the shape and dtype of which should meet the requirements of the corresponded metric.
            labels(numpy.array): ground truth of current mini-batch, the shape and dtype of which should meet the requirements of the corresponded metric. 
        """
        for m in self._metrics:
            m.update(preds, labels)

    def eval(self):
        """
        Calculate the results of all metrics sequentially.

        Returns:
            list: results of all added metrics. 
            The shape and dtype of each result depend on the definition of its metric.
        """
        ans = []
        for m in self._metrics:
            ans.append(m.eval())
        return ans


class Precision(MetricBase):
    """
    Precision (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances. Refer to
    https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

    Noted that this class manages the precision score only for binary classification task.

    Args:
       name (str, optional): Metric name. For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            metric = fluid.metrics.Precision()

            # generate the preds and labels

            preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
                     [0.2], [0.3], [0.5], [0.8], [0.6]]

            labels = [[0], [1], [1], [1], [1],
                      [0], [0], [0], [0], [0]]

            preds = np.array(preds)
            labels = np.array(labels)

            metric.update(preds=preds, labels=labels)
            numpy_precision = metric.eval()

            print("expect precision: %.2f and got %.2f" % ( 3.0 / 5.0, numpy_precision))
    """

    def __init__(self, name=None):
        super(Precision, self).__init__(name)
        self.tp = 0  # true positive
        self.fp = 0  # false positive

    def update(self, preds, labels):
        """
        Update the precision based on the current mini-batch prediction results .

        Args:
            preds(numpy.ndarray): prediction results of current mini-batch, 
                                the output of two-class sigmoid function. 
                                Shape: [batch_size, 1]. Dtype: 'float64' or 'float32'.
            labels(numpy.ndarray): ground truth (labels) of current mini-batch, 
                                 the shape should keep the same as preds. 
                                 Shape: [batch_size, 1], Dtype: 'int32' or 'int64'.
        """
        if not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray.")
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        sample_num = labels.shape[0]
        preds = np.rint(preds).astype("int32")

        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if pred == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1

    def eval(self):
        """
        Calculate the final precision.

        Returns:
            float: Results of the calculated Precision. Scalar output with float dtype.
        """
        ap = self.tp + self.fp
        return float(self.tp) / ap if ap != 0 else .0


class Recall(MetricBase):
    """
    Recall (also known as sensitivity) is the fraction of
    relevant instances that have been retrieved over the
    total amount of relevant instances

    Refer to:
    https://en.wikipedia.org/wiki/Precision_and_recall

    Noted that this class manages the recall score only for binary classification task.

    Args:
       name (str, optional): Metric name. For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            metric = fluid.metrics.Recall()

            # generate the preds and labels

            preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
                     [0.2], [0.3], [0.5], [0.8], [0.6]]

            labels = [[0], [1], [1], [1], [1],
                      [0], [0], [0], [0], [0]]

            preds = np.array(preds)
            labels = np.array(labels)

            metric.update(preds=preds, labels=labels)
            numpy_recall = metric.eval()

            print("expect recall: %.2f and got %.2f" % ( 3.0 / 4.0, numpy_recall))
    """

    def __init__(self, name=None):
        super(Recall, self).__init__(name)
        self.tp = 0  # true positive
        self.fn = 0  # false negative

    def update(self, preds, labels):
        """
        Update the recall based on the current mini-batch prediction results.

        Args:
            preds(numpy.array): prediction results of current mini-batch, 
                              the output of two-class sigmoid function. 
                              Shape: [batch_size, 1]. Dtype: 'float64' or 'float32'.
            labels(numpy.array): ground truth (labels) of current mini-batch, 
                               the shape should keep the same as preds. 
                               Shape: [batch_size, 1], Dtype: 'int32' or 'int64'.
        """
        if not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray.")
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        sample_num = labels.shape[0]
        preds = np.rint(preds).astype("int32")

        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if label == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fn += 1

    def eval(self):
        """
        Calculate the final recall.

        Returns:
            float: results of the calculated Recall. Scalar output with float dtype.
        """
        recall = self.tp + self.fn
        return float(self.tp) / recall if recall != 0 else .0


class Accuracy(MetricBase):
    """
    This interface is used to calculate the mean accuracy over multiple batches.
    Accuracy object has two state: value and weight. The definition of Accuracy is available at 
    https://en.wikipedia.org/wiki/Accuracy_and_precision

    Args:
       name (str, optional): Metric name. For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            #suppose we have batch_size = 128
            batch_size=128
            accuracy_manager = fluid.metrics.Accuracy()

            #suppose the accuracy is 0.9 for the 1st batch
            batch1_acc = 0.9
            accuracy_manager.update(value = batch1_acc, weight = batch_size)
            print("expect accuracy: %.2f, get accuracy: %.2f" % (batch1_acc, accuracy_manager.eval()))

            #suppose the accuracy is 0.8 for the 2nd batch
            batch2_acc = 0.8

            accuracy_manager.update(value = batch2_acc, weight = batch_size)
            #the joint acc for batch1 and batch2 is (batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2
            print("expect accuracy: %.2f, get accuracy: %.2f" % ((batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2, accuracy_manager.eval()))

            #reset the accuracy_manager
            accuracy_manager.reset()
            #suppose the accuracy is 0.8 for the 3rd batch
            batch3_acc = 0.8
            accuracy_manager.update(value = batch3_acc, weight = batch_size)
            print("expect accuracy: %.2f, get accuracy: %.2f" % (batch3_acc, accuracy_manager.eval()))
    """

    def __init__(self, name=None):
        super(Accuracy, self).__init__(name)
        self.value = .0
        self.weight = .0

    def update(self, value, weight):
        r"""
        This function takes the minibatch states (value, weight) as input,
        to accumulate and update the corresponding status of the Accuracy object. The update method is as follows:

        .. math::
            \\\\ \\begin{array}{l}{\\text { self. value }+=\\text { value } * \\text { weight }} \\\\ {\\text { self. weight }+=\\text { weight }}\\end{array} \\\\

        Args:
            value(float|numpy.array): accuracy of one minibatch.
            weight(int|float): minibatch size.
        """
        if not _is_number_or_matrix_(value):
            raise ValueError(
                "The 'value' must be a number(int, float) or a numpy ndarray.")
        if not _is_number_(weight):
            raise ValueError("The 'weight' must be a number(int, float).")
        if _is_number_(weight) and weight < 0:
            raise ValueError("The 'weight' can not be negative")
        self.value += value * weight
        self.weight += weight

    def eval(self):
        """
        This function returns the mean accuracy (float or numpy.array) for all accumulated minibatches.

        Returns: 
            float or numpy.array: mean accuracy for all accumulated minibatches.

        """
        if self.weight == 0:
            raise ValueError("There is no data in Accuracy Metrics. \
                Please check layers.accuracy output has added to Accuracy.")
        return self.value / self.weight


class ChunkEvaluator(MetricBase):
    """
    Accumulate counter numbers output by chunk_eval from mini-batches and
    compute the precision recall and F1-score using the accumulated counter
    numbers.
    ChunkEvaluator has three states: num_infer_chunks, num_label_chunks and num_correct_chunks, 
    which correspond to the number of chunks, the number of labeled chunks, and the number of correctly identified chunks.
    For some basics of chunking, please refer to 
    `Chunking with Support Vector Machines <https://www.aclweb.org/anthology/N01-1025>`_ .
    ChunkEvalEvaluator computes the precision, recall, and F1-score of chunk detection,
    and supports IOB, IOE, IOBES and IO (also known as plain) tagging schemes.

    Args:
       name (str, optional): Metric name. For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # init the chunk-level evaluation manager
            metric = fluid.metrics.ChunkEvaluator()

            # suppose the model predict 10 chucks, while 8 ones are correct and the ground truth has 9 chucks.
            num_infer_chunks = 10
            num_label_chunks = 9 
            num_correct_chunks = 8

            metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
            numpy_precision, numpy_recall, numpy_f1 = metric.eval()

            print("precision: %.2f, recall: %.2f, f1: %.2f" % (numpy_precision, numpy_recall, numpy_f1))

            # the next batch, predicting 3 perfectly correct chucks.
            num_infer_chunks = 3
            num_label_chunks = 3
            num_correct_chunks = 3

            metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
            numpy_precision, numpy_recall, numpy_f1 = metric.eval()

            print("precision: %.2f, recall: %.2f, f1: %.2f" % (numpy_precision, numpy_recall, numpy_f1))

    """

    def __init__(self, name=None):
        super(ChunkEvaluator, self).__init__(name)
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        r"""
        This function takes (num_infer_chunks, num_label_chunks, num_correct_chunks) as input,
        to accumulate and update the corresponding status of the ChunkEvaluator object. The update method is as follows:
        
        .. math:: 
                   \\\\ \\begin{array}{l}{\\text { self. num_infer_chunks }+=\\text { num_infer_chunks }} \\\\ {\\text { self. num_Label_chunks }+=\\text { num_label_chunks }} \\\\ {\\text { self. num_correct_chunks }+=\\text { num_correct_chunks }}\\end{array} \\\\

        Args:
            num_infer_chunks(int|numpy.array): The number of chunks in Inference on the given minibatch.
            num_label_chunks(int|numpy.array): The number of chunks in Label on the given mini-batch.
            num_correct_chunks(int|float|numpy.array): The number of chunks both in Inference and Label on the
                                                  given mini-batch.
        """
        if not _is_number_or_matrix_(num_infer_chunks):
            raise ValueError(
                "The 'num_infer_chunks' must be a number(int) or a numpy ndarray."
            )
        if not _is_number_or_matrix_(num_label_chunks):
            raise ValueError(
                "The 'num_label_chunks' must be a number(int, float) or a numpy ndarray."
            )
        if not _is_number_or_matrix_(num_correct_chunks):
            raise ValueError(
                "The 'num_correct_chunks' must be a number(int, float) or a numpy ndarray."
            )
        self.num_infer_chunks += num_infer_chunks
        self.num_label_chunks += num_label_chunks
        self.num_correct_chunks += num_correct_chunks

    def eval(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns: 
            float: mean precision, recall and f1 score.

        """
        precision = float(
            self.num_correct_chunks
        ) / self.num_infer_chunks if self.num_infer_chunks else 0
        recall = float(self.num_correct_chunks
                       ) / self.num_label_chunks if self.num_label_chunks else 0
        f1_score = float(2 * precision * recall) / (
            precision + recall) if self.num_correct_chunks else 0
        return precision, recall, f1_score


class EditDistance(MetricBase):
    """
    This API is for the management of edit distances.
    Editing distance is a method to quantify the degree of dissimilarity 
    between two strings, such as words, by calculating the minimum editing 
    operand (add, delete or replace) required to convert one string into another. 
    Refer to https://en.wikipedia.org/wiki/Edit_distance.

    Args:
        name (str, optional): Metric name. For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # suppose that batch_size is 128
            batch_size = 128

            # init the edit distance manager
            distance_evaluator = fluid.metrics.EditDistance("EditDistance")

            # generate the edit distance across 128 sequence pairs, the max distance is 10 here
            edit_distances_batch0 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
            seq_num_batch0 = batch_size

            distance_evaluator.update(edit_distances_batch0, seq_num_batch0)
            avg_distance, wrong_instance_ratio = distance_evaluator.eval()
            print("the average edit distance for batch0 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))

            edit_distances_batch1 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
            seq_num_batch1 = batch_size

            distance_evaluator.update(edit_distances_batch1, seq_num_batch1)
            avg_distance, wrong_instance_ratio = distance_evaluator.eval()
            print("the average edit distance for batch0 and batch1 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))

            distance_evaluator.reset()

            edit_distances_batch2 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
            seq_num_batch2 = batch_size

            distance_evaluator.update(edit_distances_batch2, seq_num_batch2)
            avg_distance, wrong_instance_ratio = distance_evaluator.eval()
            print("the average edit distance for batch2 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))

    """

    def __init__(self, name):
        super(EditDistance, self).__init__(name)
        self.total_distance = .0
        self.seq_num = 0
        self.instance_error = 0

    def update(self, distances, seq_num):
        """
        Update the overall edit distance

        Args:
            distances(numpy.array): a (batch_size, 1) numpy.array, each element represents the edit distance between two sequences.
            seq_num(int|float): standing for the number of sequence pairs.
        """
        if not _is_numpy_(distances):
            raise ValueError("The 'distances' must be a numpy ndarray.")
        if not _is_number_(seq_num):
            raise ValueError("The 'seq_num' must be a number(int, float).")
        seq_right_count = np.sum(distances == 0)
        total_distance = np.sum(distances)
        self.seq_num += seq_num
        self.instance_error += seq_num - seq_right_count
        self.total_distance += total_distance

    def eval(self):
        """
        Return two floats:
        avg_distance: the average distance for all sequence pairs updated using the update function.
        avg_instance_error: the ratio of sequence pairs whose edit distance is not zero.
        """
        if self.seq_num == 0:
            raise ValueError(
                "There is no data in EditDistance Metric. Please check layers.edit_distance output has been added to EditDistance."
            )
        avg_distance = self.total_distance / self.seq_num
        avg_instance_error = self.instance_error / float(self.seq_num)
        return avg_distance, avg_instance_error


class Auc(MetricBase):
    """
    The auc metric is for binary classification.
    Refer to https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve.
    Please notice that the auc metric is implemented with python, which may be a little bit slow.
    If you concern the speed, please use the fluid.layers.auc instead.

    The `auc` function creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the AUC. To discretize the AUC curve, a linearly spaced set of
    thresholds is used to compute pairs of recall and precision values. The area
    under the ROC-curve is therefore computed using the height of the recall
    values by the false positive rate, while the area under the PR-curve is the
    computed using the height of the precision values by the recall.

    Args:
        name (str, optional): Metric name. For details, please refer to :ref:`api_guide_Name`. Default is None.
        curve (str): Specifies the name of the curve to be computed, 'ROC' [default] or 'PR' for the Precision-Recall-curve.

    "NOTE: only implement the ROC curve type via Python now."

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            # init the auc metric
            auc_metric = fluid.metrics.Auc("ROC")

            # suppose that batch_size is 128
            batch_num = 100
            batch_size = 128

            for batch_id in range(batch_num):

                class0_preds = np.random.random(size = (batch_size, 1))
                class1_preds = 1 - class0_preds

                preds = np.concatenate((class0_preds, class1_preds), axis=1)

                labels = np.random.randint(2, size = (batch_size, 1))
                auc_metric.update(preds = preds, labels = labels)

                # shall be some score closing to 0.5 as the preds are randomly assigned
                print("auc for iteration %d is %.2f" % (batch_id, auc_metric.eval()))
    """

    def __init__(self, name, curve='ROC', num_thresholds=4095):
        super(Auc, self).__init__(name=name)
        self._curve = curve
        self._num_thresholds = num_thresholds

        _num_pred_buckets = num_thresholds + 1
        self._stat_pos = [0] * _num_pred_buckets
        self._stat_neg = [0] * _num_pred_buckets

    def update(self, preds, labels):
        """
        Update the auc curve with the given predictions and labels.

        Args:
             preds (numpy.array): an numpy array in the shape of (batch_size, 2), preds[i][j] denotes the probability of classifying the instance i into the class j.
             labels (numpy.array): an numpy array in the shape of (batch_size, 1), labels[i] is either o or 1, representing the label of the instance i.
        """
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        if not _is_numpy_(preds):
            raise ValueError("The 'predictions' must be a numpy ndarray.")

        for i, lbl in enumerate(labels):
            value = preds[i, 1]
            bin_idx = int(value * self._num_thresholds)
            assert bin_idx <= self._num_thresholds
            if lbl:
                self._stat_pos[bin_idx] += 1.0
            else:
                self._stat_neg[bin_idx] += 1.0

    @staticmethod
    def trapezoid_area(x1, x2, y1, y2):
        return abs(x1 - x2) * (y1 + y2) / 2.0

    def eval(self):
        """
        Return the area (a float score) under auc curve

        Return:
            float: the area under auc curve
        """
        tot_pos = 0.0
        tot_neg = 0.0
        auc = 0.0

        idx = self._num_thresholds
        while idx >= 0:
            tot_pos_prev = tot_pos
            tot_neg_prev = tot_neg
            tot_pos += self._stat_pos[idx]
            tot_neg += self._stat_neg[idx]
            auc += self.trapezoid_area(tot_neg, tot_neg_prev, tot_pos,
                                       tot_pos_prev)
            idx -= 1

        return auc / tot_pos / tot_neg if tot_pos > 0.0 and tot_neg > 0.0 else 0.0


class DetectionMAP(object):
    """
    Calculate the detection mean average precision (mAP).

    The general steps are as follows:

    1. calculate the true positive and false positive according to the input
       of detection and labels.
    2. calculate mAP value, support two versions: '11 point' and 'integral'.
       11point: the 11-point interpolated average precision.
       integral: the natural integral of the precision-recall curve.

    Please get more information from the following articles:

      https://sanchom.wordpress.com/tag/average-precision/

      https://arxiv.org/abs/1512.02325

    Args:
        input (Variable): LoDTensor, The detection results, which is a LoDTensor with shape
            [M, 6]. The layout is [label, confidence, xmin, ymin, xmax, ymax].
            The data type is float32 or float64.
        gt_label (Variable): LoDTensor, The ground truth label index, which is a LoDTensor
            with shape [N, 1].The data type is float32 or float64.
        gt_box (Variable): LoDTensor, The ground truth bounding box (bbox), which is a
            LoDTensor with shape [N, 4]. The layout is [xmin, ymin, xmax, ymax].
            The data type is float32 or float64.
        gt_difficult (Variable|None): LoDTensor, Whether this ground truth is a difficult
            bounding bbox, which can be a LoDTensor [N, 1] or not set. If None,
            it means all the ground truth labels are not difficult bbox.The
            data type is int.
        class_num (int): The class number.
        background_label (int): The index of background label, the background
            label will be ignored. If set to -1, then all categories will be
            considered, 0 by default.
        overlap_threshold (float): The threshold for deciding true/false
            positive, 0.5 by default.
        evaluate_difficult (bool): Whether to consider difficult ground truth
            for evaluation, True by default. This argument does not work when
            gt_difficult is None.
        ap_version (str): The average precision calculation ways, it must be
            'integral' or '11point'. Please check
            https://sanchom.wordpress.com/tag/average-precision/ for details.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            import paddle
            paddle.enable_static()

            batch_size = None # can be any size
            image_boxs_num = 10
            bounding_bboxes_num = 21

            pb = fluid.data(name='prior_box', shape=[image_boxs_num, 4],
                       dtype='float32')

            pbv = fluid.data(name='prior_box_var', shape=[image_boxs_num, 4],
                         dtype='float32')

            loc = fluid.data(name='target_box', shape=[batch_size, bounding_bboxes_num, 4],
                        dtype='float32')

            scores = fluid.data(name='scores', shape=[batch_size, bounding_bboxes_num, image_boxs_num],
                            dtype='float32')

            nmsed_outs = fluid.layers.detection_output(scores=scores,
                loc=loc, prior_box=pb, prior_box_var=pbv)

            gt_box = fluid.data(name="gt_box", shape=[batch_size, 4], dtype="float32")
            gt_label = fluid.data(name="gt_label", shape=[batch_size, 1], dtype="float32")
            difficult = fluid.data(name="difficult", shape=[batch_size, 1], dtype="float32")

            exe = fluid.Executor(fluid.CUDAPlace(0))
            map_evaluator = fluid.metrics.DetectionMAP(nmsed_outs, gt_label, gt_box, difficult, class_num = 3)

            cur_map, accum_map = map_evaluator.get_map_var()


    """

    def __init__(self,
                 input,
                 gt_label,
                 gt_box,
                 gt_difficult=None,
                 class_num=None,
                 background_label=0,
                 overlap_threshold=0.5,
                 evaluate_difficult=True,
                 ap_version='integral'):

        self.helper = LayerHelper('map_eval')
        gt_label = layers.cast(x=gt_label, dtype=gt_box.dtype)
        if gt_difficult:
            gt_difficult = layers.cast(x=gt_difficult, dtype=gt_box.dtype)
            label = layers.concat([gt_label, gt_difficult, gt_box], axis=1)
        else:
            label = layers.concat([gt_label, gt_box], axis=1)

        # calculate mean average precision (mAP) of current mini-batch
        map = detection.detection_map(
            input,
            label,
            class_num,
            background_label,
            overlap_threshold=overlap_threshold,
            evaluate_difficult=evaluate_difficult,
            ap_version=ap_version)

        states = []
        states.append(
            self._create_state(
                dtype='int32', shape=None, suffix='accum_pos_count'))
        states.append(
            self._create_state(
                dtype='float32', shape=None, suffix='accum_true_pos'))
        states.append(
            self._create_state(
                dtype='float32', shape=None, suffix='accum_false_pos'))
        var = self._create_state(dtype='int32', shape=[1], suffix='has_state')
        self.helper.set_variable_initializer(
            var, initializer=Constant(value=int(0)))
        self.has_state = var

        # calculate accumulative mAP
        accum_map = detection.detection_map(
            input,
            label,
            class_num,
            background_label,
            overlap_threshold=overlap_threshold,
            evaluate_difficult=evaluate_difficult,
            has_state=self.has_state,
            input_states=states,
            out_states=states,
            ap_version=ap_version)

        layers.fill_constant(
            shape=self.has_state.shape,
            value=1,
            dtype=self.has_state.dtype,
            out=self.has_state)

        self.cur_map = map
        self.accum_map = accum_map

    def _create_state(self, suffix, dtype, shape):
        """
        Create state variable.
        Args:
            suffix(str): the state suffix.
            dtype(str|core.VarDesc.VarType): the state data type
            shape(tuple|list): the shape of state
        Returns: State variable
        """
        state = self.helper.create_variable(
            name="_".join([unique_name.generate(self.helper.name), suffix]),
            persistable=True,
            dtype=dtype,
            shape=shape)
        return state

    def get_map_var(self):
        """
        Returns: mAP variable of current mini-batch and
            accumulative mAP variable cross mini-batches.
        """
        return self.cur_map, self.accum_map

    def reset(self, executor, reset_program=None):
        """
        Reset metric states at the begin of each pass/user specified batch.
        Args:
            executor(Executor): a executor for executing
                the reset_program.
            reset_program(Program|None): a single Program for reset process.
                If None, will create a Program.
        """

        def _clone_var_(block, var):
            assert isinstance(var, Variable)
            return block.create_var(
                name=var.name,
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                lod_level=var.lod_level,
                persistable=var.persistable)

        if reset_program is None:
            reset_program = Program()
        with program_guard(main_program=reset_program):
            var = _clone_var_(reset_program.current_block(), self.has_state)
            layers.fill_constant(
                shape=var.shape, value=0, dtype=var.dtype, out=var)
        executor.run(reset_program)
