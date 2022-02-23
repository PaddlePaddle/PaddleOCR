"""Utilities to evaluate the clustering performance of models.

Functions named as *_score return a scalar value to maximize: the higher the
better.
"""

# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Wei LI <kuantkid@gmail.com>
#          Diego Molla <dmolla-aliod@gmail.com>
#          Arnaud Fouchet <foucheta@gmail.com>
#          Thierry Guillemot <thierry.guillemot.work@gmail.com>
#          Gregory Stupp <stuppie@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arya McCarthy <arya@jhu.edu>
#          Uwe F Mayer <uwe_f_mayer@yahoo.com>
# License: BSD 3 clause


import warnings
from math import log

import numpy as np
from scipy import sparse as sp

from ._expected_mutual_info_fast import expected_mutual_information
from ...utils.fixes import _astype_copy_false
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length


def check_clusterings(labels_true, labels_pred):
    """Check that the labels arrays are 1D and of same dimension.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        The true labels.

    labels_pred : array-like of shape (n_samples,)
        The predicted labels.
    """
    labels_true = check_array(
        labels_true,
        ensure_2d=False,
        ensure_min_samples=0,
        dtype=None,
    )

    labels_pred = check_array(
        labels_pred,
        ensure_2d=False,
        ensure_min_samples=0,
        dtype=None,
    )

    type_label = type_of_target(labels_true)
    type_pred = type_of_target(labels_pred)

    if "continuous" in (type_pred, type_label):
        msg = (
            "Clustering metrics expects discrete values but received"
            f" {type_label} values for label, and {type_pred} values "
            "for target"
        )
        warnings.warn(msg, UserWarning)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError("labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError("labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    check_consistent_length(labels_true, labels_pred)

    return labels_true, labels_pred


def _generalized_average(U, V, average_method):
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError(
            "'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'"
        )


def contingency_matrix(
    labels_true, labels_pred, *, eps=None, sparse=False, dtype=np.int64
):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    eps : float, default=None
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : bool, default=False
        If `True`, return a sparse CSR continency matrix. If `eps` is not
        `None` and `sparse` is `True` will raise ValueError.

        .. versionadded:: 0.18

    dtype : numeric type, default=np.int64
        Output dtype. Ignored if `eps` is not `None`.

        .. versionadded:: 0.24

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer unless set
        otherwise with the ``dtype`` argument. If ``eps`` is given, the dtype
        will be float.
        Will be a ``sklearn.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix(
        (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=dtype,
    )
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


# clustering measures


def pair_confusion_matrix(labels_true, labels_pred):
    """Pair confusion matrix arising from two clusterings.

    The pair confusion matrix :math:`C` computes a 2 by 2 similarity matrix
    between two clusterings by considering all pairs of samples and counting
    pairs that are assigned into the same or into different clusters under
    the true and predicted clusterings.

    Considering a pair of samples that is clustered together a positive pair,
    then as in binary classification the count of true negatives is
    :math:`C_{00}`, false negatives is :math:`C_{10}`, true positives is
    :math:`C_{11}` and false positives is :math:`C_{01}`.

    Read more in the :ref:`User Guide <pair_confusion_matrix>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.

    Returns
    -------
    C : ndarray of shape (2, 2), dtype=np.int64
        The contingency matrix.

    See Also
    --------
    rand_score: Rand Score
    adjusted_rand_score: Adjusted Rand Score
    adjusted_mutual_info_score: Adjusted Mutual Information

    Examples
    --------
    Perfectly matching labelings have all non-zero entries on the
    diagonal regardless of actual label values:

      >>> from sklearn.metrics.cluster import pair_confusion_matrix
      >>> pair_confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
      array([[8, 0],
             [0, 4]]...

    Labelings that assign all classes members to the same clusters
    are complete but may be not always pure, hence penalized, and
    have some off-diagonal non-zero entries:

      >>> pair_confusion_matrix([0, 0, 1, 2], [0, 0, 1, 1])
      array([[8, 2],
             [0, 2]]...

    Note that the matrix is not symmetric.

    References
    ----------
    .. L. Hubert and P. Arabie, Comparing Partitions, Journal of
      Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = np.int64(labels_true.shape[0])

    # Computation using the contingency data
    contingency = contingency_matrix(
        labels_true, labels_pred, sparse=True, dtype=np.int64
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data ** 2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares
    return C


def rand_score(labels_true, labels_pred):
    """Rand index.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.

    The raw RI score is:

        RI = (number of agreeing pairs) / (number of pairs)

    Read more in the :ref:`User Guide <rand_score>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.

    Returns
    -------
    RI : float
       Similarity score between 0.0 and 1.0, inclusive, 1.0 stands for
       perfect match.

    See Also
    --------
    adjusted_rand_score: Adjusted Rand Score
    adjusted_mutual_info_score: Adjusted Mutual Information

    Examples
    --------
    Perfectly matching labelings have a score of 1 even

      >>> from sklearn.metrics.cluster import rand_score
      >>> rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete but may not always be pure, hence penalized:

      >>> rand_score([0, 0, 1, 2], [0, 0, 1, 1])
      0.83...

    References
    ----------
    .. L. Hubert and P. Arabie, Comparing Partitions, Journal of
      Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    .. https://en.wikipedia.org/wiki/Simple_matching_coefficient

    .. https://en.wikipedia.org/wiki/Rand_index
    """
    contingency = pair_confusion_matrix(labels_true, labels_pred)
    numerator = contingency.diagonal().sum()
    denominator = contingency.sum()

    if numerator == denominator or denominator == 0:
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique
        # cluster. These are perfect matches hence return 1.0.
        return 1.0

    return numerator / denominator


def adjusted_rand_score(labels_true, labels_pred):
    """Rand index adjusted for chance.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.

    The raw RI score is then "adjusted for chance" into the ARI score
    using the following scheme::

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation).

    ARI is a symmetric measure::

        adjusted_rand_score(a, b) == adjusted_rand_score(b, a)

    Read more in the :ref:`User Guide <adjusted_rand_score>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate

    Returns
    -------
    ARI : float
       Similarity score between -1.0 and 1.0. Random labelings have an ARI
       close to 0.0. 1.0 stands for perfect match.

    Examples
    --------
    Perfectly matching labelings have a score of 1 even

      >>> from sklearn.metrics.cluster import adjusted_rand_score
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete but may not always be pure, hence penalized::

      >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])
      0.57...

    ARI is symmetric, so labelings that have pure clusters with members
    coming from the same classes but unnecessary splits are penalized::

      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])
      0.57...

    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::

      >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    References
    ----------
    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    .. [Steinley2004] D. Steinley, Properties of the Hubert-Arabie
      adjusted Rand index, Psychological Methods 2004

    .. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

    See Also
    --------
    adjusted_mutual_info_score : Adjusted Mutual Information.
    """
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0):
    """Compute the homogeneity and completeness and V-Measure scores at once.

    Those metrics are based on normalized conditional entropy measures of
    the clustering labeling to evaluate given the knowledge of a Ground
    Truth class labels of the same samples.

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    Both scores have positive values between 0.0 and 1.0, larger values
    being desirable.

    Those 3 metrics are independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score values in any way.

    V-Measure is furthermore symmetric: swapping ``labels_true`` and
    ``label_pred`` will give the same score. This does not hold for
    homogeneity and completeness. V-Measure is identical to
    :func:`normalized_mutual_info_score` with the arithmetic averaging
    method.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate

    beta : float, default=1.0
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.

    Returns
    -------
    homogeneity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

    completeness : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    v_measure : float
        harmonic mean of the first two

    See Also
    --------
    homogeneity_score
    completeness_score
    v_measure_score
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    if len(labels_true) == 0:
        return 1.0, 1.0, 1.0

    entropy_C = entropy(labels_true)
    entropy_K = entropy(labels_pred)

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    MI = mutual_info_score(None, None, contingency=contingency)

    homogeneity = MI / (entropy_C) if entropy_C else 1.0
    completeness = MI / (entropy_K) if entropy_K else 1.0

    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = (
            (1 + beta)
            * homogeneity
            * completeness
            / (beta * homogeneity + completeness)
        )

    return homogeneity, completeness, v_measure_score


def homogeneity_score(labels_true, labels_pred):
    """Homogeneity metric of a cluster labeling given a ground truth.

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`completeness_score` which will be different in
    general.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate

    Returns
    -------
    homogeneity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    See Also
    --------
    completeness_score
    v_measure_score

    Examples
    --------

    Perfect labelings are homogeneous::

      >>> from sklearn.metrics.cluster import homogeneity_score
      >>> homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-perfect labelings that further split classes into more clusters can be
    perfectly homogeneous::

      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2]))
      1.000000
      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3]))
      1.000000

    Clusters that include samples from different classes do not make for an
    homogeneous labeling::

      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1]))
      0.0...
      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0]))
      0.0...
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[0]


def completeness_score(labels_true, labels_pred):
    """Completeness metric of a cluster labeling given a ground truth.

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`homogeneity_score` which will be different in
    general.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate

    Returns
    -------
    completeness : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    See Also
    --------
    homogeneity_score
    v_measure_score

    Examples
    --------

    Perfect labelings are complete::

      >>> from sklearn.metrics.cluster import completeness_score
      >>> completeness_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-perfect labelings that assign all classes members to the same clusters
    are still complete::

      >>> print(completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))
      1.0
      >>> print(completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))
      0.999...

    If classes members are split across different clusters, the
    assignment cannot be complete::

      >>> print(completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))
      0.0
      >>> print(completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
      0.0
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[1]


def v_measure_score(labels_true, labels_pred, *, beta=1.0):
    """V-measure cluster labeling given a ground truth.

    This score is identical to :func:`normalized_mutual_info_score` with
    the ``'arithmetic'`` option for averaging.

    The V-measure is the harmonic mean between homogeneity and completeness::

        v = (1 + beta) * homogeneity * completeness
             / (beta * homogeneity + completeness)

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.


    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate

    beta : float, default=1.0
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.

    Returns
    -------
    v_measure : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    See Also
    --------
    homogeneity_score
    completeness_score
    normalized_mutual_info_score

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have score 1.0::

      >>> from sklearn.metrics.cluster import v_measure_score
      >>> v_measure_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> v_measure_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete be not homogeneous, hence penalized::

      >>> print("%.6f" % v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))
      0.8...
      >>> print("%.6f" % v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
      0.66...

    Labelings that have pure clusters with members coming from the same
    classes are homogeneous but un-necessary splits harms completeness
    and thus penalize V-measure as well::

      >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 0, 1, 2]))
      0.8...
      >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 1, 2, 3]))
      0.66...

    If classes members are completely split across different clusters,
    the assignment is totally incomplete, hence the V-Measure is null::

      >>> print("%.6f" % v_measure_score([0, 0, 0, 0], [0, 1, 2, 3]))
      0.0...

    Clusters that include samples from totally different classes totally
    destroy the homogeneity of the labeling, hence::

      >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 0, 0, 0]))
      0.0...
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred, beta=beta)[2]


def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    """Mutual Information between two clusterings.

    The Mutual Information is a measure of the similarity between two labels
    of the same data. Where :math:`|U_i|` is the number of the samples
    in cluster :math:`U_i` and :math:`|V_j|` is the number of the
    samples in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:

    .. math::

        MI(U,V)=\\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}
        \\log\\frac{N|U_i \\cap V_j|}{|U_i||V_j|}

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (i.e
    ``label_true``) with :math:`V` (i.e. ``label_pred``) will return the
    same score value. This can be useful to measure the agreement of two
    independent label assignments strategies on the same dataset when the
    real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    contingency : {ndarray, sparse matrix} of shape \
            (n_classes_true, n_classes_pred), default=None
        A contingency matrix given by the :func:`contingency_matrix` function.
        If value is ``None``, it will be computed, otherwise the given value is
        used, with ``labels_true`` and ``labels_pred`` ignored.

    Returns
    -------
    mi : float
       Mutual information, a non-negative value, measured in nats using the
       natural logarithm.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    See Also
    --------
    adjusted_mutual_info_score : Adjusted against chance Mutual Information.
    normalized_mutual_info_score : Normalized Mutual Information.
    """
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(
            contingency,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[int, np.int32, np.int64],
        )

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" % type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)


def adjusted_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic"
):
    """Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. It accounts for the fact that
    the MI is generally higher for two clusterings with a larger number of
    clusters, regardless of whether there is actually more information shared.
    For two clusterings :math:`U` and :math:`V`, the AMI is given as::

        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (``label_true``)
    with :math:`V` (``labels_pred``) will return the same score value. This can
    be useful to measure the agreement of two independent label assignments
    strategies on the same dataset when the real ground truth is not known.

    Be mindful that this function is an order of magnitude slower than other
    metrics, such as the Adjusted Rand Index.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    average_method : str, default='arithmetic'
        How to compute the normalizer in the denominator. Possible options
        are 'min', 'geometric', 'arithmetic', and 'max'.

        .. versionadded:: 0.20

        .. versionchanged:: 0.22
           The default value of ``average_method`` changed from 'max' to
           'arithmetic'.

    Returns
    -------
    ami: float (upperlimited by 1.0)
       The AMI returns a value of 1 when the two partitions are identical
       (ie perfectly matched). Random partitions (independent labellings) have
       an expected AMI around 0 on average hence can be negative. The value is
       in adjusted nats (based on the natural logarithm).

    See Also
    --------
    adjusted_rand_score : Adjusted Rand Index.
    mutual_info_score : Mutual Information (not adjusted for chance).

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import adjusted_mutual_info_score
      >>> adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the AMI is null::

      >>> adjusted_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0

    References
    ----------
    .. [1] `Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
       Clusterings Comparison: Variants, Properties, Normalization and
       Correction for Chance, JMLR
       <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf>`_

    .. [2] `Wikipedia entry for the Adjusted Mutual Information
       <https://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    contingency = contingency.astype(np.float64, **_astype_copy_false(contingency))
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    emi = expected_mutual_information(contingency, n_samples)
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi
    # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
    # normalizer should always be >= emi, but because of floating-point
    # representation, sometimes emi is slightly larger. Correct this
    # by preserving the sign.
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)
    ami = (mi - emi) / denominator
    return ami


def normalized_mutual_info_score(
    labels_true, labels_pred, *, average_method="arithmetic"
):
    """Normalized Mutual Information between two clusterings.

    Normalized Mutual Information (NMI) is a normalization of the Mutual
    Information (MI) score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation). In this function, mutual
    information is normalized by some generalized mean of ``H(labels_true)``
    and ``H(labels_pred))``, defined by the `average_method`.

    This measure is not adjusted for chance. Therefore
    :func:`adjusted_mutual_info_score` might be preferred.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets.

    average_method : str, default='arithmetic'
        How to compute the normalizer in the denominator. Possible options
        are 'min', 'geometric', 'arithmetic', and 'max'.

        .. versionadded:: 0.20

        .. versionchanged:: 0.22
           The default value of ``average_method`` changed from 'geometric' to
           'arithmetic'.

    Returns
    -------
    nmi : float
       Score between 0.0 and 1.0 in normalized nats (based on the natural
       logarithm). 1.0 stands for perfectly complete labeling.

    See Also
    --------
    v_measure_score : V-Measure (NMI with arithmetic mean option).
    adjusted_rand_score : Adjusted Rand Index.
    adjusted_mutual_info_score : Adjusted Mutual Information (adjusted
        against chance).

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the NMI is null::

      >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    contingency = contingency.astype(np.float64, **_astype_copy_false(contingency))
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    # Avoid 0.0 / 0.0 when either entropy is zero.
    normalizer = max(normalizer, np.finfo("float64").eps)
    nmi = mi / normalizer
    return nmi


def fowlkes_mallows_score(labels_true, labels_pred, *, sparse=False):
    """Measure the similarity of two clusterings of a set of points.

    .. versionadded:: 0.18

    The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of
    the precision and recall::

        FMI = TP / sqrt((TP + FP) * (TP + FN))

    Where ``TP`` is the number of **True Positive** (i.e. the number of pair of
    points that belongs in the same clusters in both ``labels_true`` and
    ``labels_pred``), ``FP`` is the number of **False Positive** (i.e. the
    number of pair of points that belongs in the same clusters in
    ``labels_true`` and not in ``labels_pred``) and ``FN`` is the number of
    **False Negative** (i.e the number of pair of points that belongs in the
    same clusters in ``labels_pred`` and not in ``labels_True``).

    The score ranges from 0 to 1. A high value indicates a good similarity
    between two clusters.

    Read more in the :ref:`User Guide <fowlkes_mallows_scores>`.

    Parameters
    ----------
    labels_true : int array, shape = (``n_samples``,)
        A clustering of the data into disjoint subsets.

    labels_pred : array, shape = (``n_samples``, )
        A clustering of the data into disjoint subsets.

    sparse : bool, default=False
        Compute contingency matrix internally with sparse matrix.

    Returns
    -------
    score : float
       The resulting Fowlkes-Mallows score.

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import fowlkes_mallows_score
      >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally random, hence the FMI is null::

      >>> fowlkes_mallows_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    References
    ----------
    .. [1] `E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
       hierarchical clusterings". Journal of the American Statistical
       Association
       <https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478008>`_

    .. [2] `Wikipedia entry for the Fowlkes-Mallows Index
           <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    (n_samples,) = labels_true.shape

    c = contingency_matrix(labels_true, labels_pred, sparse=True)
    c = c.astype(np.int64, **_astype_copy_false(c))
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    return np.sqrt(tk / pk) * np.sqrt(tk / qk) if tk != 0.0 else 0.0


def entropy(labels):
    """Calculates the entropy for a labeling.

    Parameters
    ----------
    labels : int array, shape = [n_samples]
        The labels

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))
