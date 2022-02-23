"""
Common utilities for testing clustering.

"""

import numpy as np


###############################################################################
# Generate sample data


def generate_clustered_data(
    seed=0, n_clusters=3, n_features=2, n_samples_per_cluster=20, std=0.4
):
    prng = np.random.RandomState(seed)

    # the data is voluntary shifted away from zero to check clustering
    # algorithm robustness with regards to non centered data
    means = (
        np.array(
            [
                [1, 1, 1, 0],
                [-1, -1, 0, 1],
                [1, -1, 1, 1],
                [-1, 1, 1, 0],
            ]
        )
        + 10
    )

    X = np.empty((0, n_features))
    for i in range(n_clusters):
        X = np.r_[
            X,
            means[i][:n_features] + std * prng.randn(n_samples_per_cluster, n_features),
        ]
    return X
