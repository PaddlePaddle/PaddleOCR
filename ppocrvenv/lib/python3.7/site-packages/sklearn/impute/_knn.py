# Authors: Ashim Bhattarai <ashimb9@gmail.com>
#          Thomas J Fan <thomasjpfan@gmail.com>
# License: BSD 3 clause

import numpy as np

from ._base import _BaseImputer
from ..utils.validation import FLOAT_DTYPES
from ..metrics import pairwise_distances_chunked
from ..metrics.pairwise import _NAN_METRICS
from ..neighbors._base import _get_weights
from ..neighbors._base import _check_weights
from ..utils import is_scalar_nan
from ..utils._mask import _get_mask
from ..utils.validation import check_is_fitted


class KNNImputer(_BaseImputer):
    """Imputation for completing missing values using k-Nearest Neighbors.

    Each sample's missing values are imputed using the mean value from
    `n_neighbors` nearest neighbors found in the training set. Two samples are
    close if the features that neither is missing are close.

    Read more in the :ref:`User Guide <knnimpute>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to np.nan, since `pd.NA` will be converted to np.nan.

    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.

    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - callable : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    metric : {'nan_euclidean'} or callable, default='nan_euclidean'
        Distance metric for searching neighbors. Possible values:

        - 'nan_euclidean'
        - callable : a user-defined function which conforms to the definition
          of ``_pairwise_callable(X, Y, metric, **kwds)``. The function
          accepts two arrays, X and Y, and a `missing_values` keyword in
          `kwds` and returns a scalar distance value.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto the
        output of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on the
        missing indicator even if there are missing values at transform/test
        time.

    Attributes
    ----------
    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        ``None`` if add_indicator is False.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SimpleImputer : Imputation transformer for completing missing values
        with simple strategies.
    IterativeImputer : Multivariate imputer that estimates each feature
        from all the others.

    References
    ----------
    * Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
      Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing
      value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17
      no. 6, 2001 Pages 520-525.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import KNNImputer
    >>> X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
    >>> imputer = KNNImputer(n_neighbors=2)
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])
    """

    def __init__(
        self,
        *,
        missing_values=np.nan,
        n_neighbors=5,
        weights="uniform",
        metric="nan_euclidean",
        copy=True,
        add_indicator=False,
    ):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.copy = copy

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        """Helper function to impute a single column.

        Parameters
        ----------
        dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
            Distance matrix between the receivers and potential donors from
            training set. There must be at least one non-nan distance between
            a receiver and a potential donor.

        n_neighbors : int
            Number of neighbors to consider.

        fit_X_col : ndarray of shape (n_potential_donors,)
            Column of potential donors from training set.

        mask_fit_X_col : ndarray of shape (n_potential_donors,)
            Missing mask for fit_X_col.

        Returns
        -------
        imputed_values: ndarray of shape (n_receivers,)
            Imputed values for receiver.
        """
        # Get donors
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]

        # Get weight matrix from from distance matrix
        donors_dist = dist_pot_donors[
            np.arange(donors_idx.shape[0])[:, None], donors_idx
        ]

        weight_matrix = _get_weights(donors_dist, self.weights)

        # fill nans with zeros
        if weight_matrix is not None:
            weight_matrix[np.isnan(weight_matrix)] = 0.0

        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        donors = np.ma.array(donors, mask=donors_mask)

        return np.ma.average(donors, axis=1, weights=weight_matrix).data

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : array-like shape of (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            The fitted `KNNImputer` class instance.
        """
        # Check data integrity and calling arguments
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
            if self.metric not in _NAN_METRICS and not callable(self.metric):
                raise ValueError("The selected metric does not support NaN values")
        if self.n_neighbors <= 0:
            raise ValueError(
                "Expected n_neighbors > 0. Got {}".format(self.n_neighbors)
            )

        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=force_all_finite,
            copy=self.copy,
        )

        _check_weights(self.weights)
        self._fit_X = X
        self._mask_fit_X = _get_mask(self._fit_X, self.missing_values)

        super()._fit_indicator(self._mask_fit_X)

        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        X : array-like of shape (n_samples, n_output_features)
            The imputed dataset. `n_output_features` is the number of features
            that is not always missing during `fit`.
        """

        check_is_fitted(self)
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=force_all_finite,
            copy=self.copy,
            reset=False,
        )

        mask = _get_mask(X, self.missing_values)
        mask_fit_X = self._mask_fit_X
        valid_mask = ~np.all(mask_fit_X, axis=0)

        X_indicator = super()._transform_indicator(mask)

        # Removes columns where the training data is all nan
        if not np.any(mask):
            # No missing values in X
            # Remove columns where the training data is all nan
            return X[:, valid_mask]

        row_missing_idx = np.flatnonzero(mask.any(axis=1))

        non_missing_fix_X = np.logical_not(mask_fit_X)

        # Maps from indices from X to indices in dist matrix
        dist_idx_map = np.zeros(X.shape[0], dtype=int)
        dist_idx_map[row_missing_idx] = np.arange(row_missing_idx.shape[0])

        def process_chunk(dist_chunk, start):
            row_missing_chunk = row_missing_idx[start : start + len(dist_chunk)]

            # Find and impute missing by column
            for col in range(X.shape[1]):
                if not valid_mask[col]:
                    # column was all missing during training
                    continue

                col_mask = mask[row_missing_chunk, col]
                if not np.any(col_mask):
                    # column has no missing values
                    continue

                (potential_donors_idx,) = np.nonzero(non_missing_fix_X[:, col])

                # receivers_idx are indices in X
                receivers_idx = row_missing_chunk[np.flatnonzero(col_mask)]

                # distances for samples that needed imputation for column
                dist_subset = dist_chunk[dist_idx_map[receivers_idx] - start][
                    :, potential_donors_idx
                ]

                # receivers with all nan distances impute with mean
                all_nan_dist_mask = np.isnan(dist_subset).all(axis=1)
                all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]

                if all_nan_receivers_idx.size:
                    col_mean = np.ma.array(
                        self._fit_X[:, col], mask=mask_fit_X[:, col]
                    ).mean()
                    X[all_nan_receivers_idx, col] = col_mean

                    if len(all_nan_receivers_idx) == len(receivers_idx):
                        # all receivers imputed with mean
                        continue

                    # receivers with at least one defined distance
                    receivers_idx = receivers_idx[~all_nan_dist_mask]
                    dist_subset = dist_chunk[dist_idx_map[receivers_idx] - start][
                        :, potential_donors_idx
                    ]

                n_neighbors = min(self.n_neighbors, len(potential_donors_idx))
                value = self._calc_impute(
                    dist_subset,
                    n_neighbors,
                    self._fit_X[potential_donors_idx, col],
                    mask_fit_X[potential_donors_idx, col],
                )
                X[receivers_idx, col] = value

        # process in fixed-memory chunks
        gen = pairwise_distances_chunked(
            X[row_missing_idx, :],
            self._fit_X,
            metric=self.metric,
            missing_values=self.missing_values,
            force_all_finite=force_all_finite,
            reduce_func=process_chunk,
        )
        for chunk in gen:
            # process_chunk modifies X in place. No return value.
            pass

        return super()._concatenate_indicator(X[:, valid_mask], X_indicator)
