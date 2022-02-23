"""
Feature agglomeration. Base classes and functions for performing feature
agglomeration.
"""
# Author: V. Michel, A. Gramfort
# License: BSD 3 clause

import numpy as np

from ..base import TransformerMixin
from ..utils.validation import check_is_fitted
from scipy.sparse import issparse

###############################################################################
# Mixin class for feature agglomeration.


class AgglomerationTransform(TransformerMixin):
    """
    A class for feature agglomeration via the transform interface.
    """

    def transform(self, X):
        """
        Transform a new matrix using the built clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            A M by N array of M observations in N dimensions or a length
            M array of M one-dimensional observations.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_clusters) or (n_clusters,)
            The pooled values for each feature cluster.
        """
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)
        if self.pooling_func == np.mean and not issparse(X):
            size = np.bincount(self.labels_)
            n_samples = X.shape[0]
            # a fast way to compute the mean of grouped features
            nX = np.array(
                [np.bincount(self.labels_, X[i, :]) / size for i in range(n_samples)]
            )
        else:
            nX = [
                self.pooling_func(X[:, self.labels_ == l], axis=1)
                for l in np.unique(self.labels_)
            ]
            nX = np.array(nX).T
        return nX

    def inverse_transform(self, Xred):
        """
        Inverse the transformation and return a vector of size `n_features`.

        Parameters
        ----------
        Xred : array-like of shape (n_samples, n_clusters) or (n_clusters,)
            The values to be assigned to each cluster of samples.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features) or (n_features,)
            A vector of size `n_samples` with the values of `Xred` assigned to
            each of the cluster of samples.
        """
        check_is_fitted(self)

        unil, inverse = np.unique(self.labels_, return_inverse=True)
        return Xred[..., inverse]
