"""
This module contains the TreePredictor class which is used for prediction.
"""
# Author: Nicolas Hug

import numpy as np

from .common import Y_DTYPE
from ._predictor import _predict_from_raw_data
from ._predictor import _predict_from_binned_data
from ._predictor import _compute_partial_dependence


class TreePredictor:
    """Tree class used for predictions.

    Parameters
    ----------
    nodes : ndarray of PREDICTOR_RECORD_DTYPE
        The nodes of the tree.
    binned_left_cat_bitsets : ndarray of shape (n_categorical_splits, 8), \
            dtype=uint32
        Array of bitsets for binned categories used in predict_binned when a
        split is categorical.
    raw_left_cat_bitsets : ndarray of shape (n_categorical_splits, 8), \
            dtype=uint32
        Array of bitsets for raw categories used in predict when a split is
        categorical.

    """

    def __init__(self, nodes, binned_left_cat_bitsets, raw_left_cat_bitsets):
        self.nodes = nodes
        self.binned_left_cat_bitsets = binned_left_cat_bitsets
        self.raw_left_cat_bitsets = raw_left_cat_bitsets

    def get_n_leaf_nodes(self):
        """Return number of leaves."""
        return int(self.nodes["is_leaf"].sum())

    def get_max_depth(self):
        """Return maximum depth among all leaves."""
        return int(self.nodes["depth"].max())

    def predict(self, X, known_cat_bitsets, f_idx_map, n_threads):
        """Predict raw values for non-binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.

        known_cat_bitsets : ndarray of shape (n_categorical_features, 8)
            Array of bitsets of known categories, for each categorical feature.

        f_idx_map : ndarray of shape (n_features,)
            Map from original feature index to the corresponding index in the
            known_cat_bitsets array.

        n_threads : int
            Number of OpenMP threads to use.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
        out = np.empty(X.shape[0], dtype=Y_DTYPE)
        _predict_from_raw_data(
            self.nodes,
            X,
            self.raw_left_cat_bitsets,
            known_cat_bitsets,
            f_idx_map,
            n_threads,
            out,
        )
        return out

    def predict_binned(self, X, missing_values_bin_idx, n_threads):
        """Predict raw values for binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.
        missing_values_bin_idx : uint8
            Index of the bin that is used for missing values. This is the
            index of the last bin and is always equal to max_bins (as passed
            to the GBDT classes), or equivalently to n_bins - 1.
        n_threads : int
            Number of OpenMP threads to use.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
        out = np.empty(X.shape[0], dtype=Y_DTYPE)
        _predict_from_binned_data(
            self.nodes,
            X,
            self.binned_left_cat_bitsets,
            missing_values_bin_idx,
            n_threads,
            out,
        )
        return out

    def compute_partial_dependence(self, grid, target_features, out):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray, shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.
        out : ndarray, shape (n_samples)
            The value of the partial dependence function on each grid
            point.
        """
        _compute_partial_dependence(self.nodes, grid, target_features, out)
