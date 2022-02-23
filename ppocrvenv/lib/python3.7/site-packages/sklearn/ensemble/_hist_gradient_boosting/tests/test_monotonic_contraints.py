import numpy as np
import pytest

from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common import X_BINNED_DTYPE
from sklearn.ensemble._hist_gradient_boosting.common import MonotonicConstraint
from sklearn.ensemble._hist_gradient_boosting.splitting import (
    Splitter,
    compute_node_value,
)
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

n_threads = _openmp_effective_n_threads()


def is_increasing(a):
    return (np.diff(a) >= 0.0).all()


def is_decreasing(a):
    return (np.diff(a) <= 0.0).all()


def assert_leaves_values_monotonic(predictor, monotonic_cst):
    # make sure leaves values (from left to right) are either all increasing
    # or all decreasing (or neither) depending on the monotonic constraint.
    nodes = predictor.nodes

    def get_leaves_values():
        """get leaves values from left to right"""
        values = []

        def depth_first_collect_leaf_values(node_idx):
            node = nodes[node_idx]
            if node["is_leaf"]:
                values.append(node["value"])
                return
            depth_first_collect_leaf_values(node["left"])
            depth_first_collect_leaf_values(node["right"])

        depth_first_collect_leaf_values(0)  # start at root (0)
        return values

    values = get_leaves_values()

    if monotonic_cst == MonotonicConstraint.NO_CST:
        # some increasing, some decreasing
        assert not is_increasing(values) and not is_decreasing(values)
    elif monotonic_cst == MonotonicConstraint.POS:
        # all increasing
        assert is_increasing(values)
    else:  # NEG
        # all decreasing
        assert is_decreasing(values)


def assert_children_values_monotonic(predictor, monotonic_cst):
    # Make sure siblings values respect the monotonic constraints. Left should
    # be lower (resp greater) than right child if constraint is POS (resp.
    # NEG).
    # Note that this property alone isn't enough to ensure full monotonicity,
    # since we also need to guanrantee that all the descendents of the left
    # child won't be greater (resp. lower) than the right child, or its
    # descendents. That's why we need to bound the predicted values (this is
    # tested in assert_children_values_bounded)
    nodes = predictor.nodes
    left_lower = []
    left_greater = []
    for node in nodes:
        if node["is_leaf"]:
            continue

        left_idx = node["left"]
        right_idx = node["right"]

        if nodes[left_idx]["value"] < nodes[right_idx]["value"]:
            left_lower.append(node)
        elif nodes[left_idx]["value"] > nodes[right_idx]["value"]:
            left_greater.append(node)

    if monotonic_cst == MonotonicConstraint.NO_CST:
        assert left_lower and left_greater
    elif monotonic_cst == MonotonicConstraint.POS:
        assert left_lower and not left_greater
    else:  # NEG
        assert not left_lower and left_greater


def assert_children_values_bounded(grower, monotonic_cst):
    # Make sure that the values of the children of a node are bounded by the
    # middle value between that node and its sibling (if there is a monotonic
    # constraint).
    # As a bonus, we also check that the siblings values are properly ordered
    # which is slightly redundant with assert_children_values_monotonic (but
    # this check is done on the grower nodes whereas
    # assert_children_values_monotonic is done on the predictor nodes)

    if monotonic_cst == MonotonicConstraint.NO_CST:
        return

    def recursively_check_children_node_values(node, right_sibling=None):
        if node.is_leaf:
            return
        if right_sibling is not None:
            middle = (node.value + right_sibling.value) / 2
            if monotonic_cst == MonotonicConstraint.POS:
                assert node.left_child.value <= node.right_child.value <= middle
                if not right_sibling.is_leaf:
                    assert (
                        middle
                        <= right_sibling.left_child.value
                        <= right_sibling.right_child.value
                    )
            else:  # NEG
                assert node.left_child.value >= node.right_child.value >= middle
                if not right_sibling.is_leaf:
                    assert (
                        middle
                        >= right_sibling.left_child.value
                        >= right_sibling.right_child.value
                    )

        recursively_check_children_node_values(
            node.left_child, right_sibling=node.right_child
        )
        recursively_check_children_node_values(node.right_child)

    recursively_check_children_node_values(grower.root)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "monotonic_cst",
    (
        MonotonicConstraint.NO_CST,
        MonotonicConstraint.POS,
        MonotonicConstraint.NEG,
    ),
)
def test_nodes_values(monotonic_cst, seed):
    # Build a single tree with only one feature, and make sure the nodes
    # values respect the monotonic constraints.

    # Considering the following tree with a monotonic POS constraint, we
    # should have:
    #
    #       root
    #      /    \
    #     5     10    # middle = 7.5
    #    / \   / \
    #   a  b  c  d
    #
    # a <= b and c <= d  (assert_children_values_monotonic)
    # a, b <= middle <= c, d (assert_children_values_bounded)
    # a <= b <= c <= d (assert_leaves_values_monotonic)
    #
    # The last one is a consequence of the others, but can't hurt to check

    rng = np.random.RandomState(seed)
    n_samples = 1000
    n_features = 1
    X_binned = rng.randint(0, 255, size=(n_samples, n_features), dtype=np.uint8)
    X_binned = np.asfortranarray(X_binned)

    gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)

    grower = TreeGrower(
        X_binned, gradients, hessians, monotonic_cst=[monotonic_cst], shrinkage=0.1
    )
    grower.grow()

    # grow() will shrink the leaves values at the very end. For our comparison
    # tests, we need to revert the shrinkage of the leaves, else we would
    # compare the value of a leaf (shrunk) with a node (not shrunk) and the
    # test would not be correct.
    for leave in grower.finalized_leaves:
        leave.value /= grower.shrinkage

    # We pass undefined binning_thresholds because we won't use predict anyway
    predictor = grower.make_predictor(
        binning_thresholds=np.zeros((X_binned.shape[1], X_binned.max() + 1))
    )

    # The consistency of the bounds can only be checked on the tree grower
    # as the node bounds are not copied into the predictor tree. The
    # consistency checks on the values of node children and leaves can be
    # done either on the grower tree or on the predictor tree. We only
    # do those checks on the predictor tree as the latter is derived from
    # the former.
    assert_children_values_monotonic(predictor, monotonic_cst)
    assert_children_values_bounded(grower, monotonic_cst)
    assert_leaves_values_monotonic(predictor, monotonic_cst)


@pytest.mark.parametrize("seed", range(3))
def test_predictions(seed):
    # Train a model with a POS constraint on the first feature and a NEG
    # constraint on the second feature, and make sure the constraints are
    # respected by checking the predictions.
    # test adapted from lightgbm's test_monotone_constraint(), itself inspired
    # by https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html

    rng = np.random.RandomState(seed)

    n_samples = 1000
    f_0 = rng.rand(n_samples)  # positive correlation with y
    f_1 = rng.rand(n_samples)  # negative correslation with y
    X = np.c_[f_0, f_1]
    noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)
    y = 5 * f_0 + np.sin(10 * np.pi * f_0) - 5 * f_1 - np.cos(10 * np.pi * f_1) + noise

    gbdt = HistGradientBoostingRegressor(monotonic_cst=[1, -1])
    gbdt.fit(X, y)

    linspace = np.linspace(0, 1, 100)
    sin = np.sin(linspace)
    constant = np.full_like(linspace, fill_value=0.5)

    # We now assert the predictions properly respect the constraints, on each
    # feature. When testing for a feature we need to set the other one to a
    # constant, because the monotonic constraints are only a "all else being
    # equal" type of constraints:
    # a constraint on the first feature only means that
    # x0 < x0' => f(x0, x1) < f(x0', x1)
    # while x1 stays constant.
    # The constraint does not guanrantee that
    # x0 < x0' => f(x0, x1) < f(x0', x1')

    # First feature (POS)
    # assert pred is all increasing when f_0 is all increasing
    X = np.c_[linspace, constant]
    pred = gbdt.predict(X)
    assert is_increasing(pred)
    # assert pred actually follows the variations of f_0
    X = np.c_[sin, constant]
    pred = gbdt.predict(X)
    assert np.all((np.diff(pred) >= 0) == (np.diff(sin) >= 0))

    # Second feature (NEG)
    # assert pred is all decreasing when f_1 is all increasing
    X = np.c_[constant, linspace]
    pred = gbdt.predict(X)
    assert is_decreasing(pred)
    # assert pred actually follows the inverse variations of f_1
    X = np.c_[constant, sin]
    pred = gbdt.predict(X)
    assert ((np.diff(pred) <= 0) == (np.diff(sin) >= 0)).all()


def test_input_error():
    X = [[1, 2], [2, 3], [3, 4]]
    y = [0, 1, 2]

    gbdt = HistGradientBoostingRegressor(monotonic_cst=[1, 0, -1])
    with pytest.raises(
        ValueError, match="monotonic_cst has shape 3 but the input data"
    ):
        gbdt.fit(X, y)

    for monotonic_cst in ([1, 3], [1, -3]):
        gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
        with pytest.raises(
            ValueError, match="must be None or an array-like of -1, 0 or 1"
        ):
            gbdt.fit(X, y)

    gbdt = HistGradientBoostingClassifier(monotonic_cst=[0, 1])
    with pytest.raises(
        ValueError,
        match="monotonic constraints are not supported for multiclass classification",
    ):
        gbdt.fit(X, y)


def test_bounded_value_min_gain_to_split():
    # The purpose of this test is to show that when computing the gain at a
    # given split, the value of the current node should be properly bounded to
    # respect the monotonic constraints, because it strongly interacts with
    # min_gain_to_split. We build a simple example where gradients are [1, 1,
    # 100, 1, 1] (hessians are all ones). The best split happens on the 3rd
    # bin, and depending on whether the value of the node is bounded or not,
    # the min_gain_to_split constraint is or isn't satisfied.
    l2_regularization = 0
    min_hessian_to_split = 0
    min_samples_leaf = 1
    n_bins = n_samples = 5
    X_binned = np.arange(n_samples).reshape(-1, 1).astype(X_BINNED_DTYPE)
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_hessians = np.ones(n_samples, dtype=G_H_DTYPE)
    all_gradients = np.array([1, 1, 100, 1, 1], dtype=G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    sum_hessians = all_hessians.sum()
    hessians_are_constant = False

    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    children_lower_bound, children_upper_bound = -np.inf, np.inf

    min_gain_to_split = 2000
    splitter = Splitter(
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
    )

    histograms = builder.compute_histograms_brute(sample_indices)

    # Since the gradient array is [1, 1, 100, 1, 1]
    # the max possible gain happens on the 3rd bin (or equivalently in the 2nd)
    # and is equal to about 1307, which less than min_gain_to_split = 2000, so
    # the node is considered unsplittable (gain = -1)
    current_lower_bound, current_upper_bound = -np.inf, np.inf
    value = compute_node_value(
        sum_gradients,
        sum_hessians,
        current_lower_bound,
        current_upper_bound,
        l2_regularization,
    )
    # the unbounded value is equal to -sum_gradients / sum_hessians
    assert value == pytest.approx(-104 / 5)
    split_info = splitter.find_node_split(
        n_samples,
        histograms,
        sum_gradients,
        sum_hessians,
        value,
        lower_bound=children_lower_bound,
        upper_bound=children_upper_bound,
    )
    assert split_info.gain == -1  # min_gain_to_split not respected

    # here again the max possible gain is on the 3rd bin but we now cap the
    # value of the node into [-10, inf].
    # This means the gain is now about 2430 which is more than the
    # min_gain_to_split constraint.
    current_lower_bound, current_upper_bound = -10, np.inf
    value = compute_node_value(
        sum_gradients,
        sum_hessians,
        current_lower_bound,
        current_upper_bound,
        l2_regularization,
    )
    assert value == -10
    split_info = splitter.find_node_split(
        n_samples,
        histograms,
        sum_gradients,
        sum_hessians,
        value,
        lower_bound=children_lower_bound,
        upper_bound=children_upper_bound,
    )
    assert split_info.gain > min_gain_to_split
