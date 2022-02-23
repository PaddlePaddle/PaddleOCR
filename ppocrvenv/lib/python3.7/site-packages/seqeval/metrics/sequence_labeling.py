"""Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""

from __future__ import absolute_import, division, print_function

import warnings
from collections import defaultdict
from typing import List, Optional, Type

import numpy as np

from seqeval.metrics.v1 import SCORES, _precision_recall_fscore_support
from seqeval.metrics.v1 import classification_report as cr
from seqeval.metrics.v1 import \
    precision_recall_fscore_support as precision_recall_fscore_support_v1
from seqeval.reporters import DictReporter, StringReporter
from seqeval.scheme import Token


def precision_recall_fscore_support(y_true: List[List[str]],
                                    y_pred: List[List[str]],
                                    *,
                                    average: Optional[str] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    suffix: bool = False) -> SCORES:
    """Compute precision, recall, F-measure and support for each class.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a tagger.

        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.

        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        suffix : bool, False by default.

    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]

        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]

        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]

        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.

    Examples:
        >>> from seqeval.metrics.sequence_labeling import precision_recall_fscore_support
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
        (0.5, 0.5, 0.5, 2)

        It is possible to compute per-label precisions, recalls, F1-scores and
        supports instead of averaging:

        >>> precision_recall_fscore_support(y_true, y_pred, average=None)
        (array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1, 1]))

    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """

    def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for type_name, start, end in get_entities(y_true, suffix):
            entities_true[type_name].add((start, end))
        for type_name, start, end in get_entities(y_pred, suffix):
            entities_pred[type_name].add((start, end))

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=None,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )

    return precision, recall, f_score, true_sum


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true: List[List[str]], y_pred: List[List[str]],
             *,
             average: Optional[str] = 'micro',
             suffix: bool = False,
             mode: Optional[str] = None,
             sample_weight: Optional[List[int]] = None,
             zero_division: str = 'warn',
             scheme: Optional[Type[Token]] = None):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a tagger.

        average : string, [None, 'micro' (default), 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        mode : str, [None (default), `strict`].
            if ``None``, the score is compatible with conlleval.pl. Otherwise,
            the score is calculated strictly.

        scheme : Token, [IOB2, IOE2, IOBES]

        suffix : bool, False by default.

    Returns:
        score : float or array of float, shape = [n_unique_labels].

    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred, average='micro')
        0.6666666666666666
        >>> f1_score(y_true, y_pred, average='macro')
        0.75
        >>> f1_score(y_true, y_pred, average='weighted')
        0.6666666666666666
        >>> f1_score(y_true, y_pred, average=None)
        array([0.5, 1. ])
    """
    if mode == 'strict' and scheme:
        _, _, f, _ = precision_recall_fscore_support_v1(y_true, y_pred,
                                                        average=average,
                                                        warn_for=('f-score',),
                                                        beta=1,
                                                        sample_weight=sample_weight,
                                                        zero_division=zero_division,
                                                        scheme=scheme,
                                                        suffix=suffix)
    else:
        _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     beta=1,
                                                     sample_weight=sample_weight,
                                                     zero_division=zero_division,
                                                     suffix=suffix)
    return f


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true: List[List[str]], y_pred: List[List[str]],
                    *,
                    average: Optional[str] = 'micro',
                    suffix: bool = False,
                    mode: Optional[str] = None,
                    sample_weight: Optional[List[int]] = None,
                    zero_division: str = 'warn',
                    scheme: Optional[Type[Token]] = None):
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a tagger.

        average : string, [None, 'micro' (default), 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        mode : str, [None (default), `strict`].
            if ``None``, the score is compatible with conlleval.pl. Otherwise,
            the score is calculated strictly.

        scheme : Token, [IOB2, IOE2, IOBES]

        suffix : bool, False by default.

    Returns:
        score : float or array of float, shape = [n_unique_labels].

    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred, average=None)
        array([0.5, 1. ])
        >>> precision_score(y_true, y_pred, average='micro')
        0.6666666666666666
        >>> precision_score(y_true, y_pred, average='macro')
        0.75
        >>> precision_score(y_true, y_pred, average='weighted')
        0.6666666666666666
    """
    if mode == 'strict' and scheme:
        p, _, _, _ = precision_recall_fscore_support_v1(y_true, y_pred,
                                                        average=average,
                                                        warn_for=('precision',),
                                                        sample_weight=sample_weight,
                                                        zero_division=zero_division,
                                                        scheme=scheme,
                                                        suffix=suffix)
    else:
        p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     average=average,
                                                     warn_for=('precision',),
                                                     sample_weight=sample_weight,
                                                     zero_division=zero_division,
                                                     suffix=suffix)
    return p


def recall_score(y_true: List[List[str]], y_pred: List[List[str]],
                 *,
                 average: Optional[str] = 'micro',
                 suffix: bool = False,
                 mode: Optional[str] = None,
                 sample_weight: Optional[List[int]] = None,
                 zero_division: str = 'warn',
                 scheme: Optional[Type[Token]] = None):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a tagger.

        average : string, [None, 'micro' (default), 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        mode : str, [None (default), `strict`].
            if ``None``, the score is compatible with conlleval.pl. Otherwise,
            the score is calculated strictly.

        scheme : Token, [IOB2, IOE2, IOBES]

        suffix : bool, False by default.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred, average=None)
        array([0.5, 1. ])
        >>> recall_score(y_true, y_pred, average='micro')
        0.6666666666666666
        >>> recall_score(y_true, y_pred, average='macro')
        0.75
        >>> recall_score(y_true, y_pred, average='weighted')
        0.6666666666666666
    """
    if mode == 'strict' and scheme:
        _, r, _, _ = precision_recall_fscore_support_v1(y_true, y_pred,
                                                        average=average,
                                                        warn_for=('recall',),
                                                        sample_weight=sample_weight,
                                                        zero_division=zero_division,
                                                        scheme=scheme,
                                                        suffix=suffix)
    else:
        _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     average=average,
                                                     warn_for=('recall',),
                                                     sample_weight=sample_weight,
                                                     zero_division=zero_division,
                                                     suffix=suffix)
    return r


def performance_measure(y_true, y_pred):
    """
    Compute the performance metrics: TP, FP, FN, TN

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        performance_dict : dict

    Example:
        >>> from seqeval.metrics import performance_measure
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O', 'B-PER']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O', 'B-MISC']]
        >>> performance_measure(y_true, y_pred)
        {'TP': 3, 'FP': 3, 'FN': 1, 'TN': 4}
    """
    performance_dict = dict()
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    performance_dict['TP'] = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)
                                 if ((y_t != 'O') or (y_p != 'O')))
    performance_dict['FP'] = sum(((y_t != y_p) and (y_p != 'O')) for y_t, y_p in zip(y_true, y_pred))
    performance_dict['FN'] = sum(((y_t != 'O') and (y_p == 'O'))
                                 for y_t, y_p in zip(y_true, y_pred))
    performance_dict['TN'] = sum((y_t == y_p == 'O')
                                 for y_t, y_p in zip(y_true, y_pred))

    return performance_dict


def classification_report(y_true, y_pred,
                          digits=2,
                          suffix=False,
                          output_dict=False,
                          mode=None,
                          sample_weight=None,
                          zero_division='warn',
                          scheme=None):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a classifier.

        digits : int. Number of digits for formatting output floating point values.

        output_dict : bool(default=False). If True, return output as dict else str.

        mode : str, [None (default), `strict`].
            if ``None``, the score is compatible with conlleval.pl. Otherwise,
            the score is calculated strictly.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        scheme : Token, [IOB2, IOE2, IOBES]

        suffix : bool, False by default.

    Returns:
        report : string/dict. Summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
       weighted avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    if mode == 'strict':
        return cr(y_true, y_pred,
                  digits=digits,
                  output_dict=output_dict,
                  sample_weight=sample_weight,
                  zero_division=zero_division,
                  scheme=scheme,
                  suffix=suffix
                  )

    target_names_true = {type_name for type_name, _, _ in get_entities(y_true, suffix)}
    target_names_pred = {type_name for type_name, _, _ in get_entities(y_pred, suffix)}
    target_names = sorted(target_names_true | target_names_pred)

    if output_dict:
        reporter = DictReporter()
    else:
        name_width = max(map(len, target_names))
        avg_width = len('weighted avg')
        width = max(name_width, avg_width, digits)
        reporter = StringReporter(width=width, digits=digits)

    # compute per-class scores.
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
        suffix=suffix
    )
    for row in zip(target_names, p, r, f1, s):
        reporter.write(*row)
    reporter.write_blank()

    # compute average scores.
    average_options = ('micro', 'macro', 'weighted')
    for average in average_options:
        avg_p, avg_r, avg_f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
            suffix=suffix
        )
        reporter.write('{} avg'.format(average), avg_p, avg_r, avg_f1, support)
    reporter.write_blank()

    return reporter.report()
