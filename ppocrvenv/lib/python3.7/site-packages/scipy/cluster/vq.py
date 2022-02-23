"""
K-means clustering and vector quantization (:mod:`scipy.cluster.vq`)
====================================================================

Provides routines for k-means clustering, generating code books
from k-means models and quantizing vectors by comparing them with
centroids in a code book.

.. autosummary::
   :toctree: generated/

   whiten -- Normalize a group of observations so each feature has unit variance
   vq -- Calculate code book membership of a set of observation vectors
   kmeans -- Perform k-means on a set of observation vectors forming k clusters
   kmeans2 -- A different implementation of k-means with more methods
           -- for initializing centroids

Background information
----------------------
The k-means algorithm takes as input the number of clusters to
generate, k, and a set of observation vectors to cluster. It
returns a set of centroids, one for each of the k clusters. An
observation vector is classified with the cluster number or
centroid index of the centroid closest to it.

A vector v belongs to cluster i if it is closer to centroid i than
any other centroid. If v belongs to i, we say centroid i is the
dominating centroid of v. The k-means algorithm tries to
minimize distortion, which is defined as the sum of the squared distances
between each observation vector and its dominating centroid.
The minimization is achieved by iteratively reclassifying
the observations into clusters and recalculating the centroids until
a configuration is reached in which the centroids are stable. One can
also define a maximum number of iterations.

Since vector quantization is a natural application for k-means,
information theory terminology is often used. The centroid index
or cluster index is also referred to as a "code" and the table
mapping codes to centroids and, vice versa, is often referred to as a
"code book". The result of k-means, a set of centroids, can be
used to quantize vectors. Quantization aims to find an encoding of
vectors that reduces the expected distortion.

All routines expect obs to be an M by N array, where the rows are
the observation vectors. The codebook is a k by N array, where the
ith row is the centroid of code word i. The observation vectors
and centroids have the same feature dimension.

As an example, suppose we wish to compress a 24-bit color image
(each pixel is represented by one byte for red, one for blue, and
one for green) before sending it over the web. By using a smaller
8-bit encoding, we can reduce the amount of data by two
thirds. Ideally, the colors for each of the 256 possible 8-bit
encoding values should be chosen to minimize distortion of the
color. Running k-means with k=256 generates a code book of 256
codes, which fills up all possible 8-bit sequences. Instead of
sending a 3-byte value for each pixel, the 8-bit centroid index
(or code word) of the dominating centroid is transmitted. The code
book is also sent over the wire so each 8-bit code can be
translated back to a 24-bit pixel value representation. If the
image of interest was of an ocean, we would expect many 24-bit
blues to be represented by 8-bit codes. If it was an image of a
human face, more flesh-tone colors would be represented in the
code book.

"""
import warnings
import numpy as np
from collections import deque
from scipy._lib._util import _asarray_validated, check_random_state,\
    rng_integers
from scipy.spatial.distance import cdist

from . import _vq

__docformat__ = 'restructuredtext'

__all__ = ['whiten', 'vq', 'kmeans', 'kmeans2']


class ClusterError(Exception):
    pass


def whiten(obs, check_finite=True):
    """
    Normalize a group of observations on a per feature basis.

    Before running k-means, it is beneficial to rescale each feature
    dimension of the observation set by its standard deviation (i.e. "whiten"
    it - as in "white noise" where each frequency has equal power).
    Each feature is divided by its standard deviation across all observations
    to give it unit variance.

    Parameters
    ----------
    obs : ndarray
        Each row of the array is an observation.  The
        columns are the features seen during each observation.

        >>> #         f0    f1    f2
        >>> obs = [[  1.,   1.,   1.],  #o0
        ...        [  2.,   2.,   2.],  #o1
        ...        [  3.,   3.,   3.],  #o2
        ...        [  4.,   4.,   4.]]  #o3

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    result : ndarray
        Contains the values in `obs` scaled by the standard deviation
        of each column.

    Examples
    --------
    >>> from scipy.cluster.vq import whiten
    >>> features  = np.array([[1.9, 2.3, 1.7],
    ...                       [1.5, 2.5, 2.2],
    ...                       [0.8, 0.6, 1.7,]])
    >>> whiten(features)
    array([[ 4.17944278,  2.69811351,  7.21248917],
           [ 3.29956009,  2.93273208,  9.33380951],
           [ 1.75976538,  0.7038557 ,  7.21248917]])

    """
    obs = _asarray_validated(obs, check_finite=check_finite)
    std_dev = obs.std(axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        warnings.warn("Some columns have standard deviation zero. "
                      "The values of these columns will not change.",
                      RuntimeWarning)
    return obs / std_dev


def vq(obs, code_book, check_finite=True):
    """
    Assign codes from a code book to observations.

    Assigns a code from a code book to each observation. Each
    observation vector in the 'M' by 'N' `obs` array is compared with the
    centroids in the code book and assigned the code of the closest
    centroid.

    The features in `obs` should have unit variance, which can be
    achieved by passing them through the whiten function. The code
    book can be created with the k-means algorithm or a different
    encoding algorithm.

    Parameters
    ----------
    obs : ndarray
        Each row of the 'M' x 'N' array is an observation. The columns are
        the "features" seen during each observation. The features must be
        whitened first using the whiten function or something equivalent.
    code_book : ndarray
        The code book is usually generated using the k-means algorithm.
        Each row of the array holds a different code, and the columns are
        the features of the code.

         >>> #              f0    f1    f2   f3
         >>> code_book = [
         ...             [  1.,   2.,   3.,   4.],  #c0
         ...             [  1.,   2.,   3.,   4.],  #c1
         ...             [  1.,   2.,   3.,   4.]]  #c2

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    code : ndarray
        A length M array holding the code book index for each observation.
    dist : ndarray
        The distortion (distance) between the observation and its nearest
        code.

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.cluster.vq import vq
    >>> code_book = array([[1.,1.,1.],
    ...                    [2.,2.,2.]])
    >>> features  = array([[  1.9,2.3,1.7],
    ...                    [  1.5,2.5,2.2],
    ...                    [  0.8,0.6,1.7]])
    >>> vq(features,code_book)
    (array([1, 1, 0],'i'), array([ 0.43588989,  0.73484692,  0.83066239]))

    """
    obs = _asarray_validated(obs, check_finite=check_finite)
    code_book = _asarray_validated(code_book, check_finite=check_finite)
    ct = np.common_type(obs, code_book)

    c_obs = obs.astype(ct, copy=False)
    c_code_book = code_book.astype(ct, copy=False)

    if np.issubdtype(ct, np.float64) or np.issubdtype(ct, np.float32):
        return _vq.vq(c_obs, c_code_book)
    return py_vq(obs, code_book, check_finite=False)


def py_vq(obs, code_book, check_finite=True):
    """ Python version of vq algorithm.

    The algorithm computes the Euclidean distance between each
    observation and every frame in the code_book.

    Parameters
    ----------
    obs : ndarray
        Expects a rank 2 array. Each row is one observation.
    code_book : ndarray
        Code book to use. Same format than obs. Should have same number of
        features (e.g., columns) than obs.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    code : ndarray
        code[i] gives the label of the ith obversation; its code is
        code_book[code[i]].
    mind_dist : ndarray
        min_dist[i] gives the distance between the ith observation and its
        corresponding code.

    Notes
    -----
    This function is slower than the C version but works for
    all input types. If the inputs have the wrong types for the
    C versions of the function, this one is called as a last resort.

    It is about 20 times slower than the C version.

    """
    obs = _asarray_validated(obs, check_finite=check_finite)
    code_book = _asarray_validated(code_book, check_finite=check_finite)

    if obs.ndim != code_book.ndim:
        raise ValueError("Observation and code_book should have the same rank")

    if obs.ndim == 1:
        obs = obs[:, np.newaxis]
        code_book = code_book[:, np.newaxis]

    dist = cdist(obs, code_book)
    code = dist.argmin(axis=1)
    min_dist = dist[np.arange(len(code)), code]
    return code, min_dist


# py_vq2 was equivalent to py_vq
py_vq2 = np.deprecate(py_vq, old_name='py_vq2', new_name='py_vq')


def _kmeans(obs, guess, thresh=1e-5):
    """ "raw" version of k-means.

    Returns
    -------
    code_book
        The lowest distortion codebook found.
    avg_dist
        The average distance a observation is from a code in the book.
        Lower means the code_book matches the data better.

    See Also
    --------
    kmeans : wrapper around k-means

    Examples
    --------
    Note: not whitened in this example.

    >>> from numpy import array
    >>> from scipy.cluster.vq import _kmeans
    >>> features  = array([[ 1.9,2.3],
    ...                    [ 1.5,2.5],
    ...                    [ 0.8,0.6],
    ...                    [ 0.4,1.8],
    ...                    [ 1.0,1.0]])
    >>> book = array((features[0],features[2]))
    >>> _kmeans(features,book)
    (array([[ 1.7       ,  2.4       ],
           [ 0.73333333,  1.13333333]]), 0.40563916697728591)

    """

    code_book = np.asarray(guess)
    diff = np.inf
    prev_avg_dists = deque([diff], maxlen=2)
    while diff > thresh:
        # compute membership and distances between obs and code_book
        obs_code, distort = vq(obs, code_book, check_finite=False)
        prev_avg_dists.append(distort.mean(axis=-1))
        # recalc code_book as centroids of associated obs
        code_book, has_members = _vq.update_cluster_means(obs, obs_code,
                                                          code_book.shape[0])
        code_book = code_book[has_members]
        diff = prev_avg_dists[0] - prev_avg_dists[1]

    return code_book, prev_avg_dists[1]


def kmeans(obs, k_or_guess, iter=20, thresh=1e-5, check_finite=True,
           *, seed=None):
    """
    Performs k-means on a set of observation vectors forming k clusters.

    The k-means algorithm adjusts the classification of the observations
    into clusters and updates the cluster centroids until the position of
    the centroids is stable over successive iterations. In this
    implementation of the algorithm, the stability of the centroids is
    determined by comparing the absolute value of the change in the average
    Euclidean distance between the observations and their corresponding
    centroids against a threshold. This yields
    a code book mapping centroids to codes and vice versa.

    Parameters
    ----------
    obs : ndarray
       Each row of the M by N array is an observation vector. The
       columns are the features seen during each observation.
       The features must be whitened first with the `whiten` function.

    k_or_guess : int or ndarray
       The number of centroids to generate. A code is assigned to
       each centroid, which is also the row index of the centroid
       in the code_book matrix generated.

       The initial k centroids are chosen by randomly selecting
       observations from the observation matrix. Alternatively,
       passing a k by N array specifies the initial k centroids.

    iter : int, optional
       The number of times to run k-means, returning the codebook
       with the lowest distortion. This argument is ignored if
       initial centroids are specified with an array for the
       ``k_or_guess`` parameter. This parameter does not represent the
       number of iterations of the k-means algorithm.

    thresh : float, optional
       Terminates the k-means algorithm if the change in
       distortion since the last k-means iteration is less than
       or equal to threshold.

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        Seed for initializing the pseudo-random number generator.
        If `seed` is None (or `numpy.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        The default is None.

    Returns
    -------
    codebook : ndarray
       A k by N array of k centroids. The ith centroid
       codebook[i] is represented with the code i. The centroids
       and codes generated represent the lowest distortion seen,
       not necessarily the globally minimal distortion.
       Note that the number of centroids is not necessarily the same as the
       ``k_or_guess`` parameter, because centroids assigned to no observations
       are removed during iterations.

    distortion : float
       The mean (non-squared) Euclidean distance between the observations
       passed and the centroids generated. Note the difference to the standard
       definition of distortion in the context of the k-means algorithm, which
       is the sum of the squared distances.

    See Also
    --------
    kmeans2 : a different implementation of k-means clustering
       with more methods for generating initial centroids but without
       using a distortion change threshold as a stopping criterion.

    whiten : must be called prior to passing an observation matrix
       to kmeans.

    Notes
    -----
    For more functionalities or optimal performance, you can use
    `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
    `This <https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html#comparison-of-high-performance-implementations>`_
    is a benchmark result of several implementations.

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.cluster.vq import vq, kmeans, whiten
    >>> import matplotlib.pyplot as plt
    >>> features  = array([[ 1.9,2.3],
    ...                    [ 1.5,2.5],
    ...                    [ 0.8,0.6],
    ...                    [ 0.4,1.8],
    ...                    [ 0.1,0.1],
    ...                    [ 0.2,1.8],
    ...                    [ 2.0,0.5],
    ...                    [ 0.3,1.5],
    ...                    [ 1.0,1.0]])
    >>> whitened = whiten(features)
    >>> book = np.array((whitened[0],whitened[2]))
    >>> kmeans(whitened,book)
    (array([[ 2.3110306 ,  2.86287398],    # random
           [ 0.93218041,  1.24398691]]), 0.85684700941625547)

    >>> codes = 3
    >>> kmeans(whitened,codes)
    (array([[ 2.3110306 ,  2.86287398],    # random
           [ 1.32544402,  0.65607529],
           [ 0.40782893,  2.02786907]]), 0.5196582527686241)

    >>> # Create 50 datapoints in two clusters a and b
    >>> pts = 50
    >>> rng = np.random.default_rng()
    >>> a = rng.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
    >>> b = rng.multivariate_normal([30, 10],
    ...                             [[10, 2], [2, 1]],
    ...                             size=pts)
    >>> features = np.concatenate((a, b))
    >>> # Whiten data
    >>> whitened = whiten(features)
    >>> # Find 2 clusters in the data
    >>> codebook, distortion = kmeans(whitened, 2)
    >>> # Plot whitened data and cluster centers in red
    >>> plt.scatter(whitened[:, 0], whitened[:, 1])
    >>> plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    >>> plt.show()

    """
    obs = _asarray_validated(obs, check_finite=check_finite)
    if iter < 1:
        raise ValueError("iter must be at least 1, got %s" % iter)

    # Determine whether a count (scalar) or an initial guess (array) was passed.
    if not np.isscalar(k_or_guess):
        guess = _asarray_validated(k_or_guess, check_finite=check_finite)
        if guess.size < 1:
            raise ValueError("Asked for 0 clusters. Initial book was %s" %
                             guess)
        return _kmeans(obs, guess, thresh=thresh)

    # k_or_guess is a scalar, now verify that it's an integer
    k = int(k_or_guess)
    if k != k_or_guess:
        raise ValueError("If k_or_guess is a scalar, it must be an integer.")
    if k < 1:
        raise ValueError("Asked for %d clusters." % k)

    rng = check_random_state(seed)

    # initialize best distance value to a large value
    best_dist = np.inf
    for i in range(iter):
        # the initial code book is randomly selected from observations
        guess = _kpoints(obs, k, rng)
        book, dist = _kmeans(obs, guess, thresh=thresh)
        if dist < best_dist:
            best_book = book
            best_dist = dist
    return best_book, best_dist


def _kpoints(data, k, rng):
    """Pick k points at random in data (one row = one observation).

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    rng : `numpy.random.Generator` or `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """
    idx = rng.choice(data.shape[0], size=k, replace=False)
    return data[idx]


def _krandinit(data, k, rng):
    """Returns k samples of a random variable whose parameters depend on data.

    More precisely, it returns k observations sampled from a Gaussian random
    variable whose mean and covariances are the ones estimated from the data.

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    rng : `numpy.random.Generator` or `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """
    mu = data.mean(axis=0)

    if data.ndim == 1:
        cov = np.cov(data)
        x = rng.standard_normal(size=k)
        x *= np.sqrt(cov)
    elif data.shape[1] > data.shape[0]:
        # initialize when the covariance matrix is rank deficient
        _, s, vh = np.linalg.svd(data - mu, full_matrices=False)
        x = rng.standard_normal(size=(k, s.size))
        sVh = s[:, None] * vh / np.sqrt(data.shape[0] - 1)
        x = x.dot(sVh)
    else:
        cov = np.atleast_2d(np.cov(data, rowvar=False))

        # k rows, d cols (one row = one obs)
        # Generate k sample of a random variable ~ Gaussian(mu, cov)
        x = rng.standard_normal(size=(k, mu.size))
        x = x.dot(np.linalg.cholesky(cov).T)

    x += mu
    return x


def _kpp(data, k, rng):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    rng : `numpy.random.Generator` or `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    """

    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = np.ndarray((k, dims))

    for i in range(k):
        if i == 0:
            init[i, :] = data[rng_integers(rng, data.shape[0])]

        else:
            D2 = cdist(init[:i,:], data, metric='sqeuclidean').min(axis=0)
            probs = D2/D2.sum()
            cumprobs = probs.cumsum()
            r = rng.uniform()
            init[i, :] = data[np.searchsorted(cumprobs, r)]

    return init


_valid_init_meth = {'random': _krandinit, 'points': _kpoints, '++': _kpp}


def _missing_warn():
    """Print a warning when called."""
    warnings.warn("One of the clusters is empty. "
                  "Re-run kmeans with a different initialization.")


def _missing_raise():
    """Raise a ClusterError when called."""
    raise ClusterError("One of the clusters is empty. "
                       "Re-run kmeans with a different initialization.")


_valid_miss_meth = {'warn': _missing_warn, 'raise': _missing_raise}


def kmeans2(data, k, iter=10, thresh=1e-5, minit='random',
            missing='warn', check_finite=True, *, seed=None):
    """
    Classify a set of observations into k clusters using the k-means algorithm.

    The algorithm attempts to minimize the Euclidean distance between
    observations and centroids. Several initialization methods are
    included.

    Parameters
    ----------
    data : ndarray
        A 'M' by 'N' array of 'M' observations in 'N' dimensions or a length
        'M' array of 'M' 1-D observations.
    k : int or ndarray
        The number of clusters to form as well as the number of
        centroids to generate. If `minit` initialization string is
        'matrix', or if a ndarray is given instead, it is
        interpreted as initial cluster to use instead.
    iter : int, optional
        Number of iterations of the k-means algorithm to run. Note
        that this differs in meaning from the iters parameter to
        the kmeans function.
    thresh : float, optional
        (not used yet)
    minit : str, optional
        Method for initialization. Available methods are 'random',
        'points', '++' and 'matrix':

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        'points': choose k observations (rows) at random from data for
        the initial centroids.

        '++': choose k observations accordingly to the kmeans++ method
        (careful seeding)

        'matrix': interpret the k parameter as a k by M (or length k
        array for 1-D data) array of initial centroids.
    missing : str, optional
        Method to deal with empty clusters. Available methods are
        'warn' and 'raise':

        'warn': give a warning and continue.

        'raise': raise an ClusterError and terminate the algorithm.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        Seed for initializing the pseudo-random number generator.
        If `seed` is None (or `numpy.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        The default is None.

    Returns
    -------
    centroid : ndarray
        A 'k' by 'N' array of centroids found at the last iteration of
        k-means.
    label : ndarray
        label[i] is the code or index of the centroid the
        ith observation is closest to.

    See Also
    --------
    kmeans

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.

    Examples
    --------
    >>> from scipy.cluster.vq import kmeans2
    >>> import matplotlib.pyplot as plt

    Create z, an array with shape (100, 2) containing a mixture of samples
    from three multivariate normal distributions.

    >>> rng = np.random.default_rng()
    >>> a = rng.multivariate_normal([0, 6], [[2, 1], [1, 1.5]], size=45)
    >>> b = rng.multivariate_normal([2, 0], [[1, -1], [-1, 3]], size=30)
    >>> c = rng.multivariate_normal([6, 4], [[5, 0], [0, 1.2]], size=25)
    >>> z = np.concatenate((a, b, c))
    >>> rng.shuffle(z)

    Compute three clusters.

    >>> centroid, label = kmeans2(z, 3, minit='points')
    >>> centroid
    array([[ 2.22274463, -0.61666946],  # may vary
           [ 0.54069047,  5.86541444],
           [ 6.73846769,  4.01991898]])

    How many points are in each cluster?

    >>> counts = np.bincount(label)
    >>> counts
    array([29, 51, 20])  # may vary

    Plot the clusters.

    >>> w0 = z[label == 0]
    >>> w1 = z[label == 1]
    >>> w2 = z[label == 2]
    >>> plt.plot(w0[:, 0], w0[:, 1], 'o', alpha=0.5, label='cluster 0')
    >>> plt.plot(w1[:, 0], w1[:, 1], 'd', alpha=0.5, label='cluster 1')
    >>> plt.plot(w2[:, 0], w2[:, 1], 's', alpha=0.5, label='cluster 2')
    >>> plt.plot(centroid[:, 0], centroid[:, 1], 'k*', label='centroids')
    >>> plt.axis('equal')
    >>> plt.legend(shadow=True)
    >>> plt.show()

    """
    if int(iter) < 1:
        raise ValueError("Invalid iter (%s), "
                         "must be a positive integer." % iter)
    try:
        miss_meth = _valid_miss_meth[missing]
    except KeyError as e:
        raise ValueError("Unknown missing method %r" % (missing,)) from e

    data = _asarray_validated(data, check_finite=check_finite)
    if data.ndim == 1:
        d = 1
    elif data.ndim == 2:
        d = data.shape[1]
    else:
        raise ValueError("Input of rank > 2 is not supported.")

    if data.size < 1:
        raise ValueError("Empty input is not supported.")

    # If k is not a single value, it should be compatible with data's shape
    if minit == 'matrix' or not np.isscalar(k):
        code_book = np.array(k, copy=True)
        if data.ndim != code_book.ndim:
            raise ValueError("k array doesn't match data rank")
        nc = len(code_book)
        if data.ndim > 1 and code_book.shape[1] != d:
            raise ValueError("k array doesn't match data dimension")
    else:
        nc = int(k)

        if nc < 1:
            raise ValueError("Cannot ask kmeans2 for %d clusters"
                             " (k was %s)" % (nc, k))
        elif nc != k:
            warnings.warn("k was not an integer, was converted.")

        try:
            init_meth = _valid_init_meth[minit]
        except KeyError as e:
            raise ValueError("Unknown init method %r" % (minit,)) from e
        else:
            rng = check_random_state(seed)
            code_book = init_meth(data, k, rng)

    for i in range(iter):
        # Compute the nearest neighbor for each obs using the current code book
        label = vq(data, code_book)[0]
        # Update the code book by computing centroids
        new_code_book, has_members = _vq.update_cluster_means(data, label, nc)
        if not has_members.all():
            miss_meth()
            # Set the empty clusters to their previous positions
            new_code_book[~has_members] = code_book[~has_members]
        code_book = new_code_book

    return code_book, label
