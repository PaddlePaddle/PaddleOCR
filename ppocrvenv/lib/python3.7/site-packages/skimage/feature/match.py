import numpy as np
from scipy.spatial.distance import cdist


def match_descriptors(descriptors1, descriptors2, metric=None, p=2,
                      max_distance=np.inf, cross_check=True, max_ratio=1.0):
    """Brute-force matching of descriptors.

    For each descriptor in the first set this matcher finds the closest
    descriptor in the second set (and vice-versa in the case of enabled
    cross-checking).

    Parameters
    ----------
    descriptors1 : (M, P) array
        Descriptors of size P about M keypoints in the first image.
    descriptors2 : (N, P) array
        Descriptors of size P about N keypoints in the second image.
    metric : {'euclidean', 'cityblock', 'minkowski', 'hamming', ...} , optional
        The metric to compute the distance between two descriptors. See
        `scipy.spatial.distance.cdist` for all possible types. The hamming
        distance should be used for binary descriptors. By default the L2-norm
        is used for all descriptors of dtype float or double and the Hamming
        distance is used for binary descriptors automatically.
    p : int, optional
        The p-norm to apply for ``metric='minkowski'``.
    max_distance : float, optional
        Maximum allowed distance between descriptors of two keypoints
        in separate images to be regarded as a match.
    cross_check : bool, optional
        If True, the matched keypoints are returned after cross checking i.e. a
        matched pair (keypoint1, keypoint2) is returned if keypoint2 is the
        best match for keypoint1 in second image and keypoint1 is the best
        match for keypoint2 in first image.
    max_ratio : float, optional
        Maximum ratio of distances between first and second closest descriptor
        in the second set of descriptors. This threshold is useful to filter
        ambiguous matches between the two descriptor sets. The choice of this
        value depends on the statistics of the chosen descriptor, e.g.,
        for SIFT descriptors a value of 0.8 is usually chosen, see
        D.G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints",
        International Journal of Computer Vision, 2004.

    Returns
    -------
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.

    """

    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor length must equal.")

    if metric is None:
        if np.issubdtype(descriptors1.dtype, bool):
            metric = 'hamming'
        else:
            metric = 'euclidean'

    kwargs = {}
    # Scipy raises an error if p is passed as an extra argument when it isn't
    # necessary for the chosen metric.
    if metric == 'minkowski':
        kwargs['p'] = p
    distances = cdist(descriptors1, descriptors2, metric=metric, **kwargs)

    indices1 = np.arange(descriptors1.shape[0])
    indices2 = np.argmin(distances, axis=1)

    if cross_check:
        matches1 = np.argmin(distances, axis=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = np.inf
        second_best_indices2 = np.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] \
            = np.finfo(np.double).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    matches = np.column_stack((indices1, indices2))

    return matches
