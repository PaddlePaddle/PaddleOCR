import numpy as np
from scipy._lib._util import check_random_state
from scipy.special import ndtr, ndtri
from scipy._lib._util import rng_integers
from dataclasses import make_dataclass
from ._common import ConfidenceInterval
from scipy.stats.stats import _broadcast_concatenate


def _vectorize_statistic(statistic):
    """Vectorize an n-sample statistic"""
    # This is a little cleaner than np.nditer at the expense of some data
    # copying: concatenate samples together, then use np.apply_along_axis
    def stat_nd(*data, axis=0):
        lengths = [sample.shape[axis] for sample in data]
        split_indices = np.cumsum(lengths)[:-1]
        z = _broadcast_concatenate(data, axis)

        def stat_1d(z):
            data = np.split(z, split_indices)
            return statistic(*data)

        return np.apply_along_axis(stat_1d, axis, z)[()]
    return stat_nd


def _jackknife_resample(sample, batch=None):
    """Jackknife resample the sample. Only one-sample stats for now."""
    n = sample.shape[-1]
    batch_nominal = batch or n

    for k in range(0, n, batch_nominal):
        # col_start:col_end are the observations to remove
        batch_actual = min(batch_nominal, n-k)

        # jackknife - each row leaves out one observation
        j = np.ones((batch_actual, n), dtype=bool)
        np.fill_diagonal(j[:, k:k+batch_actual], False)
        i = np.arange(n)
        i = np.broadcast_to(i, (batch_actual, n))
        i = i[j].reshape((batch_actual, n-1))

        resamples = sample[..., i]
        yield resamples


def _bootstrap_resample(sample, n_resamples=None, random_state=None):
    """Bootstrap resample the sample."""
    n = sample.shape[-1]

    # bootstrap - each row is a random resample of original observations
    i = rng_integers(random_state, 0, n, (n_resamples, n))

    resamples = sample[..., i]
    return resamples


def _percentile_of_score(a, score, axis):
    """Vectorized, simplified `scipy.stats.percentileofscore`.

    Unlike `stats.percentileofscore`, the percentile returned is a fraction
    in [0, 1].
    """
    B = a.shape[axis]
    return (a < score).sum(axis=axis) / B


def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    # the difference between _percentile_along_axis and np.percentile is that
    # np.percentile gets _all_ the qs for each axis slice, whereas
    # _percentile_along_axis gets the q corresponding with each axis slice
    shape = theta_hat_b.shape[:-1]
    alpha = np.broadcast_to(alpha, shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        theta_hat_b_i = theta_hat_b[indices]
        percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]  # return scalar instead of 0d array


def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
    """Bias-corrected and accelerated interval."""
    # closely follows [2] "BCa Bootstrap CIs"
    sample = data[0]  # only works with 1 sample statistics right now

    # calculate z0_hat
    theta_hat = statistic(sample, axis=axis)[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)
    z0_hat = ndtri(percentile)

    # calculate a_hat
    theta_hat_i = []  # would be better to fill pre-allocated array
    for jackknife_sample in _jackknife_resample(sample, batch):
        theta_hat_i.append(statistic(jackknife_sample, axis=-1))
    theta_hat_i = np.concatenate(theta_hat_i, axis=-1)
    theta_hat_dot = theta_hat_i.mean(axis=-1, keepdims=True)
    num = ((theta_hat_dot - theta_hat_i)**3).sum(axis=-1)
    den = 6*((theta_hat_dot - theta_hat_i)**2).sum(axis=-1)**(3/2)
    a_hat = num / den

    # calculate alpha_1, alpha_2
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1/(1 - a_hat*num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2/(1 - a_hat*num2))
    return alpha_1, alpha_2


def _bootstrap_iv(data, statistic, vectorized, paired, axis, confidence_level,
                  n_resamples, batch, method, random_state):
    """Input validation and standardization for `bootstrap`."""

    if vectorized not in {True, False}:
        raise ValueError("`vectorized` must be `True` or `False`.")

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    n_samples = 0
    try:
        n_samples = len(data)
    except TypeError:
        raise ValueError("`data` must be a sequence of samples.")

    if n_samples == 0:
        raise ValueError("`data` must contain at least one sample.")

    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis_int] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    if paired not in {True, False}:
        raise ValueError("`paired` must be `True` or `False`.")

    if paired:
        n = data_iv[0].shape[-1]
        for sample in data_iv[1:]:
            if sample.shape[-1] != n:
                message = ("When `paired is True`, all samples must have the "
                           "same length along `axis`")
                raise ValueError(message)

        # to generate the bootstrap distribution for paired-sample statistics,
        # resample the indices of the observations
        def statistic(i, axis=-1, data=data_iv, unpaired_statistic=statistic):
            data = [sample[..., i] for sample in data]
            return unpaired_statistic(*data, axis=axis)

        data_iv = [np.arange(n)]

    confidence_level_float = float(confidence_level)

    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    methods = {'percentile', 'basic', 'bca'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f"`method` must be in {methods}")

    message = "`method = 'BCa' is only available for one-sample statistics"
    if not paired and n_samples > 1 and method == 'bca':
        raise ValueError(message)

    random_state = check_random_state(random_state)

    return (data_iv, statistic, vectorized, paired, axis_int,
            confidence_level_float, n_resamples_int, batch_iv,
            method, random_state)


fields = ['confidence_interval', 'standard_error']
BootstrapResult = make_dataclass("BootstrapResult", fields)


def bootstrap(data, statistic, *, vectorized=True, paired=False, axis=0,
              confidence_level=0.95, n_resamples=9999, batch=None,
              method='BCa', random_state=None):
    r"""
    Compute a two-sided bootstrap confidence interval of a statistic.

    When `method` is ``'percentile'``, a bootstrap confidence interval is
    computed according to the following procedure.

    1. Resample the data: for each sample in `data` and for each of
       `n_resamples`, take a random sample of the original sample
       (with replacement) of the same size as the original sample.

    2. Compute the bootstrap distribution of the statistic: for each set of
       resamples, compute the test statistic.

    3. Determine the confidence interval: find the interval of the bootstrap
       distribution that is

       - symmetric about the median and
       - contains `confidence_level` of the resampled statistic values.

    While the ``'percentile'`` method is the most intuitive, it is rarely
    used in practice. Two more common methods are available, ``'basic'``
    ('reverse percentile') and ``'BCa'`` ('bias-corrected and accelerated');
    they differ in how step 3 is performed.

    If the samples in `data` are  taken at random from their respective
    distributions :math:`n` times, the confidence interval returned by
    `bootstrap` will contain the true value of the statistic for those
    distributions approximately `confidence_level`:math:`\, \times \, n` times.

    Parameters
    ----------
    data : sequence of array-like
         Each element of data is a sample from an underlying distribution.
    statistic : callable
        Statistic for which the confidence interval is to be calculated.
        `statistic` must be a callable that accepts ``len(data)`` samples
        as separate arguments and returns the resulting statistic.
        If `vectorized` is set ``True``,
        `statistic` must also accept a keyword argument `axis` and be
        vectorized to compute the statistic along the provided `axis`.
    vectorized : bool, default: ``True``
        If `vectorized` is set ``False``, `statistic` will not be passed
        keyword argument `axis`, and is assumed to calculate the statistic
        only for 1D samples.
    paired : bool, default: ``False``
        Whether the statistic treats corresponding elements of the samples
        in `data` as paired.
    axis : int, default: ``0``
        The axis of the samples in `data` along which the `statistic` is
        calculated.
    confidence_level : float, default: ``0.95``
        The confidence level of the confidence interval.
    n_resamples : int, default: ``9999``
        The number of resamples performed to form the bootstrap distribution
        of the statistic.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        `statistic`. Memory usage is O(`batch`*``n``), where ``n`` is the
        sample size. Default is ``None``, in which case ``batch = n_resamples``
        (or ``batch = max(n_resamples, n)`` for ``method='BCa'``).
    method : {'percentile', 'basic', 'bca'}, default: ``'BCa'``
        Whether to return the 'percentile' bootstrap confidence interval
        (``'percentile'``), the 'reverse' or the bias-corrected and accelerated
        bootstrap confidence interval (``'BCa'``).
        Note that only ``'percentile'`` and ``'basic'`` support multi-sample
        statistics at this time.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is ``None`` (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

        Pseudorandom number generator state used to generate resamples.

    Returns
    -------
    res : BootstrapResult
        An object with attributes:

        confidence_interval : ConfidenceInterval
            The bootstrap confidence interval as an instance of
            `collections.namedtuple` with attributes `low` and `high`.
        standard_error : float or ndarray
            The bootstrap standard error, that is, the sample standard
            deviation of the bootstrap distribution

    References
    ----------
    .. [1] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap,
       Chapman & Hall/CRC, Boca Raton, FL, USA (1993)
    .. [2] Nathaniel E. Helwig, "Bootstrap Confidence Intervals",
       http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
    .. [3] Bootstrapping (statistics), Wikipedia,
       https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

    Examples
    --------
    Suppose we have sampled data from an unknown distribution.

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> from scipy.stats import norm
    >>> dist = norm(loc=2, scale=4)  # our "unknown" distribution
    >>> data = dist.rvs(size=100, random_state=rng)

    We are interested int the standard deviation of the distribution.

    >>> std_true = dist.std()      # the true value of the statistic
    >>> print(std_true)
    4.0
    >>> std_sample = np.std(data)  # the sample statistic
    >>> print(std_sample)
    3.9460644295563863

    We can calculate a 90% confidence interval of the statistic using
    `bootstrap`.

    >>> from scipy.stats import bootstrap
    >>> data = (data,)  # samples must be in a sequence
    >>> res = bootstrap(data, np.std, confidence_level=0.9,
    ...                 random_state=rng)
    >>> print(res.confidence_interval)
    ConfidenceInterval(low=3.57655333533867, high=4.382043696342881)

    If we sample from the distribution 1000 times and form a bootstrap
    confidence interval for each sample, the confidence interval
    contains the true value of the statistic approximately 900 times.

    >>> n_trials = 1000
    >>> ci_contains_true_std = 0
    >>> for i in range(n_trials):
    ...    data = (dist.rvs(size=100, random_state=rng),)
    ...    ci = bootstrap(data, np.std, confidence_level=0.9, n_resamples=1000,
    ...                   random_state=rng).confidence_interval
    ...    if ci[0] < std_true < ci[1]:
    ...        ci_contains_true_std += 1
    >>> print(ci_contains_true_std)
    875

    Rather than writing a loop, we can also determine the confidence intervals
    for all 1000 samples at once.

    >>> data = (dist.rvs(size=(n_trials, 100), random_state=rng),)
    >>> res = bootstrap(data, np.std, axis=-1, confidence_level=0.9,
    ...                 n_resamples=1000, random_state=rng)
    >>> ci_l, ci_u = res.confidence_interval

    Here, `ci_l` and `ci_u` contain the confidence interval for each of the
    ``n_trials = 1000`` samples.

    >>> print(ci_l[995:])
    [3.77729695 3.75090233 3.45829131 3.34078217 3.48072829]
    >>> print(ci_u[995:])
    [4.88316666 4.86924034 4.32032996 4.2822427  4.59360598]

    And again, approximately 90% contain the true value, ``std_true = 4``.

    >>> print(np.sum((ci_l < std_true) & (std_true < ci_u)))
    900

    `bootstrap` can also be used to estimate confidence intervals of
    multi-sample statistics, including those calculated by hypothesis
    tests. `scipy.stats.mood` perform's Mood's test for equal scale parameters,
    and it returns two outputs: a statistic, and a p-value. To get a
    confidence interval for the test statistic, we first wrap
    `scipy.stats.mood` in a function that accepts two sample arguments,
    accepts an `axis` keyword argument, and returns only the statistic.

    >>> from scipy.stats import mood
    >>> def my_statistic(sample1, sample2, axis):
    ...     statistic, _ = mood(sample1, sample2, axis=-1)
    ...     return statistic

    Here, we use the 'percentile' method with the default 95% confidence level.

    >>> sample1 = norm.rvs(scale=1, size=100, random_state=rng)
    >>> sample2 = norm.rvs(scale=2, size=100, random_state=rng)
    >>> data = (sample1, sample2)
    >>> res = bootstrap(data, my_statistic, method='basic', random_state=rng)
    >>> print(mood(sample1, sample2)[0])  # element 0 is the statistic
    -5.521109549096542
    >>> print(res.confidence_interval)
    ConfidenceInterval(low=-7.255994487314675, high=-4.016202624747605)

    The bootstrap estimate of the standard error is also available.

    >>> print(res.standard_error)
    0.8344963846318795

    Paired-sample statistics work, too. For example, consider the Pearson
    correlation coefficient.

    >>> from scipy.stats import pearsonr
    >>> n = 100
    >>> x = np.linspace(0, 10, n)
    >>> y = x + rng.uniform(size=n)
    >>> print(pearsonr(x, y)[0])  # element 0 is the statistic
    0.9962357936065914

    We wrap `pearsonr` so that it returns only the statistic.

    >>> def my_statistic(x, y):
    ...     return pearsonr(x, y)[0]

    We call `bootstrap` using ``paired=True``.
    Also, since ``my_statistic`` isn't vectorized to calculate the statistic
    along a given axis, we pass in ``vectorized=False``.

    >>> res = bootstrap((x, y), my_statistic, vectorized=False, paired=True,
    ...                 random_state=rng)
    >>> print(res.confidence_interval)
    ConfidenceInterval(low=0.9950085825848624, high=0.9971212407917498)

    """
    # Input validation
    args = _bootstrap_iv(data, statistic, vectorized, paired, axis,
                         confidence_level, n_resamples, batch, method,
                         random_state)
    data, statistic, vectorized, paired, axis = args[:5]
    confidence_level, n_resamples, batch, method, random_state = args[5:]

    theta_hat_b = []

    batch_nominal = batch or n_resamples

    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples-k)
        # Generate resamples
        resampled_data = []
        for sample in data:
            resample = _bootstrap_resample(sample, n_resamples=batch_actual,
                                           random_state=random_state)
            resampled_data.append(resample)

        # Compute bootstrap distribution of statistic
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    alpha = (1 - confidence_level)/2
    if method == 'bca':
        interval = _bca_interval(data, statistic, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = statistic(*data, axis=-1)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1))
