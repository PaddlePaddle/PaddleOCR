"""
This module contains the TreeGrower class.

TreeGrower builds a regression tree fitting a Newton-Raphson step, based on
the gradients and hessians of the training data.
"""
# Author: Nicolas Hug

from heapq import heappush, heappop
import numpy as np
from timeit import default_timer as time
import numbers

from .splitting import Splitter
from .histogram import HistogramBuilder
from .predictor import TreePredictor
from .utils import sum_parallel
from .common import PREDICTOR_RECORD_DTYPE
from .common import X_BITSET_INNER_DTYPE
from .common import Y_DTYPE
from .common import MonotonicConstraint
from ._bitset import set_raw_bitset_from_binned_bitset
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


EPS = np.finfo(Y_DTYPE).eps  # to avoid zero division errors


class TreeNode:
    """Tree Node class used in TreeGrower.

    This isn't used for prediction purposes, only for training (see
    TreePredictor).

    Parameters
    ----------
    depth : int
        The depth of the node, i.e. its distance from the root.
    sample_indices : ndarray of shape (n_samples_at_node,), dtype=np.uint
        The indices of the samples at the node.
    sum_gradients : float
        The sum of the gradients of the samples at the node.
    sum_hessians : float
        The sum of the hessians of the samples at the node.

    Attributes
    ----------
    depth : int
        The depth of the node, i.e. its distance from the root.
    sample_indices : ndarray of shape (n_samples_at_node,), dtype=np.uint
        The indices of the samples at the node.
    sum_gradients : float
        The sum of the gradients of the samples at the node.
    sum_hessians : float
        The sum of the hessians of the samples at the node.
    split_info : SplitInfo or None
        The result of the split evaluation.
    left_child : TreeNode or None
        The left child of the node. None for leaves.
    right_child : TreeNode or None
        The right child of the node. None for leaves.
    value : float or None
        The value of the leaf, as computed in finalize_leaf(). None for
        non-leaf nodes.
    partition_start : int
        start position of the node's sample_indices in splitter.partition.
    partition_stop : int
        stop position of the node's sample_indices in splitter.partition.
    """

    split_info = None
    left_child = None
    right_child = None
    histograms = None

    # start and stop indices of the node in the splitter.partition
    # array. Concretely,
    # self.sample_indices = view(self.splitter.partition[start:stop])
    # Please see the comments about splitter.partition and
    # splitter.split_indices for more info about this design.
    # These 2 attributes are only used in _update_raw_prediction, because we
    # need to iterate over the leaves and I don't know how to efficiently
    # store the sample_indices views because they're all of different sizes.
    partition_start = 0
    partition_stop = 0

    def __init__(self, depth, sample_indices, sum_gradients, sum_hessians, value=None):
        self.depth = depth
        self.sample_indices = sample_indices
        self.n_samples = sample_indices.shape[0]
        self.sum_gradients = sum_gradients
        self.sum_hessians = sum_hessians
        self.value = value
        self.is_leaf = False
        self.set_children_bounds(float("-inf"), float("+inf"))

    def set_children_bounds(self, lower, upper):
        """Set children values bounds to respect monotonic constraints."""

        # These are bounds for the node's *children* values, not the node's
        # value. The bounds are used in the splitter when considering potential
        # left and right child.
        self.children_lower_bound = lower
        self.children_upper_bound = upper

    def __lt__(self, other_node):
        """Comparison for priority queue.

        Nodes with high gain are higher priority than nodes with low gain.

        heapq.heappush only need the '<' operator.
        heapq.heappop take the smallest item first (smaller is higher
        priority).

        Parameters
        ----------
        other_node : TreeNode
            The node to compare with.
        """
        return self.split_info.gain > other_node.split_info.gain


class TreeGrower:
    """Tree grower class used to build a tree.

    The tree is fitted to predict the values of a Newton-Raphson step. The
    splits are considered in a best-first fashion, and the quality of a
    split is defined in splitting._split_gain.

    Parameters
    ----------
    X_binned : ndarray of shape (n_samples, n_features), dtype=np.uint8
        The binned input samples. Must be Fortran-aligned.
    gradients : ndarray of shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration ``i - 1``.
    hessians : ndarray of shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration ``i - 1``.
    max_leaf_nodes : int, default=None
        The maximum number of leaves for each tree. If None, there is no
        maximum limit.
    max_depth : int, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf.
    min_gain_to_split : float, default=0.
        The minimum gain needed to split a node. Splits with lower gain will
        be ignored.
    n_bins : int, default=256
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
    n_bins_non_missing : ndarray, dtype=np.uint32, default=None
        For each feature, gives the number of bins actually used for
        non-missing values. For features with a lot of unique values, this
        is equal to ``n_bins - 1``. If it's an int, all features are
        considered to have the same number of bins. If None, all features
        are considered to have ``n_bins - 1`` bins.
    has_missing_values : bool or ndarray, dtype=bool, default=False
        Whether each feature contains missing values (in the training data).
        If it's a bool, the same value is used for all features.
    is_categorical : ndarray of bool of shape (n_features,), default=None
        Indicates categorical features.
    monotonic_cst : array-like of shape (n_features,), dtype=int, default=None
        Indicates the monotonic constraint to enforce on each feature. -1, 1
        and 0 respectively correspond to a positive constraint, negative
        constraint and no constraint. Read more in the :ref:`User Guide
        <monotonic_cst_gbdt>`.
    l2_regularization : float, default=0.
        The L2 regularization parameter.
    min_hessian_to_split : float, default=1e-3
        The minimum sum of hessians needed in each node. Splits that result in
        at least one child having a sum of hessians less than
        ``min_hessian_to_split`` are discarded.
    shrinkage : float, default=1.
        The shrinkage parameter to apply to the leaves values, also known as
        learning rate.
    n_threads : int, default=None
        Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
        to determine the effective number of threads use, which takes cgroups CPU
        quotes into account. See the docstring of `_openmp_effective_n_threads`
        for details.
    """

    def __init__(
        self,
        X_binned,
        gradients,
        hessians,
        max_leaf_nodes=None,
        max_depth=None,
        min_samples_leaf=20,
        min_gain_to_split=0.0,
        n_bins=256,
        n_bins_non_missing=None,
        has_missing_values=False,
        is_categorical=None,
        monotonic_cst=None,
        l2_regularization=0.0,
        min_hessian_to_split=1e-3,
        shrinkage=1.0,
        n_threads=None,
    ):

        self._validate_parameters(
            X_binned,
            max_leaf_nodes,
            max_depth,
            min_samples_leaf,
            min_gain_to_split,
            l2_regularization,
            min_hessian_to_split,
        )
        n_threads = _openmp_effective_n_threads(n_threads)

        if n_bins_non_missing is None:
            n_bins_non_missing = n_bins - 1

        if isinstance(n_bins_non_missing, numbers.Integral):
            n_bins_non_missing = np.array(
                [n_bins_non_missing] * X_binned.shape[1], dtype=np.uint32
            )
        else:
            n_bins_non_missing = np.asarray(n_bins_non_missing, dtype=np.uint32)

        if isinstance(has_missing_values, bool):
            has_missing_values = [has_missing_values] * X_binned.shape[1]
        has_missing_values = np.asarray(has_missing_values, dtype=np.uint8)

        if monotonic_cst is None:
            self.with_monotonic_cst = False
            monotonic_cst = np.full(
                shape=X_binned.shape[1],
                fill_value=MonotonicConstraint.NO_CST,
                dtype=np.int8,
            )
        else:
            self.with_monotonic_cst = True
            monotonic_cst = np.asarray(monotonic_cst, dtype=np.int8)

            if monotonic_cst.shape[0] != X_binned.shape[1]:
                raise ValueError(
                    "monotonic_cst has shape {} but the input data "
                    "X has {} features.".format(
                        monotonic_cst.shape[0], X_binned.shape[1]
                    )
                )
            if np.any(monotonic_cst < -1) or np.any(monotonic_cst > 1):
                raise ValueError(
                    "monotonic_cst must be None or an array-like of -1, 0 or 1."
                )

        if is_categorical is None:
            is_categorical = np.zeros(shape=X_binned.shape[1], dtype=np.uint8)
        else:
            is_categorical = np.asarray(is_categorical, dtype=np.uint8)

        if np.any(
            np.logical_and(
                is_categorical == 1, monotonic_cst != MonotonicConstraint.NO_CST
            )
        ):
            raise ValueError("Categorical features cannot have monotonic constraints.")

        hessians_are_constant = hessians.shape[0] == 1
        self.histogram_builder = HistogramBuilder(
            X_binned, n_bins, gradients, hessians, hessians_are_constant, n_threads
        )
        missing_values_bin_idx = n_bins - 1
        self.splitter = Splitter(
            X_binned,
            n_bins_non_missing,
            missing_values_bin_idx,
            has_missing_values,
            is_categorical,
            monotonic_cst,
            l2_regularization,
            min_hessian_to_split,
            min_samples_leaf,
            min_gain_to_split,
            hessians_are_constant,
            n_threads,
        )
        self.n_bins_non_missing = n_bins_non_missing
        self.missing_values_bin_idx = missing_values_bin_idx
        self.max_leaf_nodes = max_leaf_nodes
        self.has_missing_values = has_missing_values
        self.monotonic_cst = monotonic_cst
        self.is_categorical = is_categorical
        self.l2_regularization = l2_regularization
        self.n_features = X_binned.shape[1]
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.X_binned = X_binned
        self.min_gain_to_split = min_gain_to_split
        self.shrinkage = shrinkage
        self.n_threads = n_threads
        self.splittable_nodes = []
        self.finalized_leaves = []
        self.total_find_split_time = 0.0  # time spent finding the best splits
        self.total_compute_hist_time = 0.0  # time spent computing histograms
        self.total_apply_split_time = 0.0  # time spent splitting nodes
        self.n_categorical_splits = 0
        self._intilialize_root(gradients, hessians, hessians_are_constant)
        self.n_nodes = 1

    def _validate_parameters(
        self,
        X_binned,
        max_leaf_nodes,
        max_depth,
        min_samples_leaf,
        min_gain_to_split,
        l2_regularization,
        min_hessian_to_split,
    ):
        """Validate parameters passed to __init__.

        Also validate parameters passed to splitter.
        """
        if X_binned.dtype != np.uint8:
            raise NotImplementedError("X_binned must be of type uint8.")
        if not X_binned.flags.f_contiguous:
            raise ValueError(
                "X_binned should be passed as Fortran contiguous "
                "array for maximum efficiency."
            )
        if max_leaf_nodes is not None and max_leaf_nodes <= 1:
            raise ValueError(
                "max_leaf_nodes={} should not be smaller than 2".format(max_leaf_nodes)
            )
        if max_depth is not None and max_depth < 1:
            raise ValueError(
                "max_depth={} should not be smaller than 1".format(max_depth)
            )
        if min_samples_leaf < 1:
            raise ValueError(
                "min_samples_leaf={} should not be smaller than 1".format(
                    min_samples_leaf
                )
            )
        if min_gain_to_split < 0:
            raise ValueError(
                "min_gain_to_split={} must be positive.".format(min_gain_to_split)
            )
        if l2_regularization < 0:
            raise ValueError(
                "l2_regularization={} must be positive.".format(l2_regularization)
            )
        if min_hessian_to_split < 0:
            raise ValueError(
                "min_hessian_to_split={} must be positive.".format(min_hessian_to_split)
            )

    def grow(self):
        """Grow the tree, from root to leaves."""
        while self.splittable_nodes:
            self.split_next()

        self._apply_shrinkage()

    def _apply_shrinkage(self):
        """Multiply leaves values by shrinkage parameter.

        This must be done at the very end of the growing process. If this were
        done during the growing process e.g. in finalize_leaf(), then a leaf
        would be shrunk but its sibling would potentially not be (if it's a
        non-leaf), which would lead to a wrong computation of the 'middle'
        value needed to enforce the monotonic constraints.
        """
        for leaf in self.finalized_leaves:
            leaf.value *= self.shrinkage

    def _intilialize_root(self, gradients, hessians, hessians_are_constant):
        """Initialize root node and finalize it if needed."""
        n_samples = self.X_binned.shape[0]
        depth = 0
        sum_gradients = sum_parallel(gradients, self.n_threads)
        if self.histogram_builder.hessians_are_constant:
            sum_hessians = hessians[0] * n_samples
        else:
            sum_hessians = sum_parallel(hessians, self.n_threads)
        self.root = TreeNode(
            depth=depth,
            sample_indices=self.splitter.partition,
            sum_gradients=sum_gradients,
            sum_hessians=sum_hessians,
            value=0,
        )

        self.root.partition_start = 0
        self.root.partition_stop = n_samples

        if self.root.n_samples < 2 * self.min_samples_leaf:
            # Do not even bother computing any splitting statistics.
            self._finalize_leaf(self.root)
            return
        if sum_hessians < self.splitter.min_hessian_to_split:
            self._finalize_leaf(self.root)
            return

        self.root.histograms = self.histogram_builder.compute_histograms_brute(
            self.root.sample_indices
        )
        self._compute_best_split_and_push(self.root)

    def _compute_best_split_and_push(self, node):
        """Compute the best possible split (SplitInfo) of a given node.

        Also push it in the heap of splittable nodes if gain isn't zero.
        The gain of a node is 0 if either all the leaves are pure
        (best gain = 0), or if no split would satisfy the constraints,
        (min_hessians_to_split, min_gain_to_split, min_samples_leaf)
        """

        node.split_info = self.splitter.find_node_split(
            node.n_samples,
            node.histograms,
            node.sum_gradients,
            node.sum_hessians,
            node.value,
            node.children_lower_bound,
            node.children_upper_bound,
        )

        if node.split_info.gain <= 0:  # no valid split
            self._finalize_leaf(node)
        else:
            heappush(self.splittable_nodes, node)

    def split_next(self):
        """Split the node with highest potential gain.

        Returns
        -------
        left : TreeNode
            The resulting left child.
        right : TreeNode
            The resulting right child.
        """
        # Consider the node with the highest loss reduction (a.k.a. gain)
        node = heappop(self.splittable_nodes)

        tic = time()
        (
            sample_indices_left,
            sample_indices_right,
            right_child_pos,
        ) = self.splitter.split_indices(node.split_info, node.sample_indices)
        self.total_apply_split_time += time() - tic

        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        n_leaf_nodes += 2

        left_child_node = TreeNode(
            depth,
            sample_indices_left,
            node.split_info.sum_gradient_left,
            node.split_info.sum_hessian_left,
            value=node.split_info.value_left,
        )
        right_child_node = TreeNode(
            depth,
            sample_indices_right,
            node.split_info.sum_gradient_right,
            node.split_info.sum_hessian_right,
            value=node.split_info.value_right,
        )

        node.right_child = right_child_node
        node.left_child = left_child_node

        # set start and stop indices
        left_child_node.partition_start = node.partition_start
        left_child_node.partition_stop = node.partition_start + right_child_pos
        right_child_node.partition_start = left_child_node.partition_stop
        right_child_node.partition_stop = node.partition_stop

        if not self.has_missing_values[node.split_info.feature_idx]:
            # If no missing values are encountered at fit time, then samples
            # with missing values during predict() will go to whichever child
            # has the most samples.
            node.split_info.missing_go_to_left = (
                left_child_node.n_samples > right_child_node.n_samples
            )

        self.n_nodes += 2
        self.n_categorical_splits += node.split_info.is_categorical

        if self.max_leaf_nodes is not None and n_leaf_nodes == self.max_leaf_nodes:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            self._finalize_splittable_nodes()
            return left_child_node, right_child_node

        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            return left_child_node, right_child_node

        if left_child_node.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(left_child_node)
        if right_child_node.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(right_child_node)

        if self.with_monotonic_cst:
            # Set value bounds for respecting monotonic constraints
            # See test_nodes_values() for details
            if (
                self.monotonic_cst[node.split_info.feature_idx]
                == MonotonicConstraint.NO_CST
            ):
                lower_left = lower_right = node.children_lower_bound
                upper_left = upper_right = node.children_upper_bound
            else:
                mid = (left_child_node.value + right_child_node.value) / 2
                if (
                    self.monotonic_cst[node.split_info.feature_idx]
                    == MonotonicConstraint.POS
                ):
                    lower_left, upper_left = node.children_lower_bound, mid
                    lower_right, upper_right = mid, node.children_upper_bound
                else:  # NEG
                    lower_left, upper_left = mid, node.children_upper_bound
                    lower_right, upper_right = node.children_lower_bound, mid
            left_child_node.set_children_bounds(lower_left, upper_left)
            right_child_node.set_children_bounds(lower_right, upper_right)

        # Compute histograms of children, and compute their best possible split
        # (if needed)
        should_split_left = not left_child_node.is_leaf
        should_split_right = not right_child_node.is_leaf
        if should_split_left or should_split_right:

            # We will compute the histograms of both nodes even if one of them
            # is a leaf, since computing the second histogram is very cheap
            # (using histogram subtraction).
            n_samples_left = left_child_node.sample_indices.shape[0]
            n_samples_right = right_child_node.sample_indices.shape[0]
            if n_samples_left < n_samples_right:
                smallest_child = left_child_node
                largest_child = right_child_node
            else:
                smallest_child = right_child_node
                largest_child = left_child_node

            # We use the brute O(n_samples) method on the child that has the
            # smallest number of samples, and the subtraction trick O(n_bins)
            # on the other one.
            tic = time()
            smallest_child.histograms = self.histogram_builder.compute_histograms_brute(
                smallest_child.sample_indices
            )
            largest_child.histograms = (
                self.histogram_builder.compute_histograms_subtraction(
                    node.histograms, smallest_child.histograms
                )
            )
            self.total_compute_hist_time += time() - tic

            tic = time()
            if should_split_left:
                self._compute_best_split_and_push(left_child_node)
            if should_split_right:
                self._compute_best_split_and_push(right_child_node)
            self.total_find_split_time += time() - tic

            # Release memory used by histograms as they are no longer needed
            # for leaf nodes since they won't be split.
            for child in (left_child_node, right_child_node):
                if child.is_leaf:
                    del child.histograms

        # Release memory used by histograms as they are no longer needed for
        # internal nodes once children histograms have been computed.
        del node.histograms

        return left_child_node, right_child_node

    def _finalize_leaf(self, node):
        """Make node a leaf of the tree being grown."""

        node.is_leaf = True
        self.finalized_leaves.append(node)

    def _finalize_splittable_nodes(self):
        """Transform all splittable nodes into leaves.

        Used when some constraint is met e.g. maximum number of leaves or
        maximum depth."""
        while len(self.splittable_nodes) > 0:
            node = self.splittable_nodes.pop()
            self._finalize_leaf(node)

    def make_predictor(self, binning_thresholds):
        """Make a TreePredictor object out of the current tree.

        Parameters
        ----------
        binning_thresholds : array-like of floats
            Corresponds to the bin_thresholds_ attribute of the BinMapper.
            For each feature, this stores:

            - the bin frontiers for continuous features
            - the unique raw category values for categorical features

        Returns
        -------
        A TreePredictor object.
        """
        predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
        binned_left_cat_bitsets = np.zeros(
            (self.n_categorical_splits, 8), dtype=X_BITSET_INNER_DTYPE
        )
        raw_left_cat_bitsets = np.zeros(
            (self.n_categorical_splits, 8), dtype=X_BITSET_INNER_DTYPE
        )
        _fill_predictor_arrays(
            predictor_nodes,
            binned_left_cat_bitsets,
            raw_left_cat_bitsets,
            self.root,
            binning_thresholds,
            self.n_bins_non_missing,
        )
        return TreePredictor(
            predictor_nodes, binned_left_cat_bitsets, raw_left_cat_bitsets
        )


def _fill_predictor_arrays(
    predictor_nodes,
    binned_left_cat_bitsets,
    raw_left_cat_bitsets,
    grower_node,
    binning_thresholds,
    n_bins_non_missing,
    next_free_node_idx=0,
    next_free_bitset_idx=0,
):
    """Helper used in make_predictor to set the TreePredictor fields."""
    node = predictor_nodes[next_free_node_idx]
    node["count"] = grower_node.n_samples
    node["depth"] = grower_node.depth
    if grower_node.split_info is not None:
        node["gain"] = grower_node.split_info.gain
    else:
        node["gain"] = -1

    node["value"] = grower_node.value

    if grower_node.is_leaf:
        # Leaf node
        node["is_leaf"] = True
        return next_free_node_idx + 1, next_free_bitset_idx

    split_info = grower_node.split_info
    feature_idx, bin_idx = split_info.feature_idx, split_info.bin_idx
    node["feature_idx"] = feature_idx
    node["bin_threshold"] = bin_idx
    node["missing_go_to_left"] = split_info.missing_go_to_left
    node["is_categorical"] = split_info.is_categorical

    if split_info.bin_idx == n_bins_non_missing[feature_idx] - 1:
        # Split is on the last non-missing bin: it's a "split on nans".
        # All nans go to the right, the rest go to the left.
        # Note: for categorical splits, bin_idx is 0 and we rely on the bitset
        node["num_threshold"] = np.inf
    elif split_info.is_categorical:
        categories = binning_thresholds[feature_idx]
        node["bitset_idx"] = next_free_bitset_idx
        binned_left_cat_bitsets[next_free_bitset_idx] = split_info.left_cat_bitset
        set_raw_bitset_from_binned_bitset(
            raw_left_cat_bitsets[next_free_bitset_idx],
            split_info.left_cat_bitset,
            categories,
        )
        next_free_bitset_idx += 1
    else:
        node["num_threshold"] = binning_thresholds[feature_idx][bin_idx]

    next_free_node_idx += 1

    node["left"] = next_free_node_idx
    next_free_node_idx, next_free_bitset_idx = _fill_predictor_arrays(
        predictor_nodes,
        binned_left_cat_bitsets,
        raw_left_cat_bitsets,
        grower_node.left_child,
        binning_thresholds=binning_thresholds,
        n_bins_non_missing=n_bins_non_missing,
        next_free_node_idx=next_free_node_idx,
        next_free_bitset_idx=next_free_bitset_idx,
    )

    node["right"] = next_free_node_idx
    return _fill_predictor_arrays(
        predictor_nodes,
        binned_left_cat_bitsets,
        raw_left_cat_bitsets,
        grower_node.right_child,
        binning_thresholds=binning_thresholds,
        n_bins_non_missing=n_bins_non_missing,
        next_free_node_idx=next_free_node_idx,
        next_free_bitset_idx=next_free_bitset_idx,
    )
