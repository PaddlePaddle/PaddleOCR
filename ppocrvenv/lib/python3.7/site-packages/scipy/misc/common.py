"""
Functions which are common and require SciPy Base and Level 1 SciPy
(special, linalg)
"""

from numpy import arange, newaxis, hstack, prod, array, frombuffer, load

__all__ = ['central_diff_weights', 'derivative', 'ascent', 'face',
           'electrocardiogram']


def central_diff_weights(Np, ndiv=1):
    """
    Return weights for an Np-point central derivative.

    Assumes equally-spaced function points.

    If weights are in the vector w, then
    derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)

    Parameters
    ----------
    Np : int
        Number of points for the central derivative.
    ndiv : int, optional
        Number of divisions. Default is 1.

    Returns
    -------
    w : ndarray
        Weights for an Np-point central derivative. Its size is `Np`.

    Notes
    -----
    Can be inaccurate for a large number of points.

    Examples
    --------
    We can calculate a derivative value of a function.

    >>> from scipy.misc import central_diff_weights
    >>> def f(x):
    ...     return 2 * x**2 + 3
    >>> x = 3.0 # derivative point
    >>> h = 0.1 # differential step
    >>> Np = 3 # point number for central derivative
    >>> weights = central_diff_weights(Np) # weights for first derivative
    >>> vals = [f(x + (i - Np/2) * h) for i in range(Np)]
    >>> sum(w * v for (w, v) in zip(weights, vals))/h
    11.79999999999998

    This value is close to the analytical solution:
    f'(x) = 4x, so f'(3) = 12

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Finite_difference

    """
    if Np < ndiv + 1:
        raise ValueError("Number of points must be at least the derivative order + 1.")
    if Np % 2 == 0:
        raise ValueError("The number of points must be odd.")
    from scipy import linalg
    ho = Np >> 1
    x = arange(-ho,ho+1.0)
    x = x[:,newaxis]
    X = x**0.0
    for k in range(1,Np):
        X = hstack([X,x**k])
    w = prod(arange(1,ndiv+1),axis=0)*linalg.inv(X)[ndiv]
    return w


def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    """
    Find the nth derivative of a function at a point.

    Given a function, use a central difference formula with spacing `dx` to
    compute the nth derivative at `x0`.

    Parameters
    ----------
    func : function
        Input function.
    x0 : float
        The point at which the nth derivative is found.
    dx : float, optional
        Spacing.
    n : int, optional
        Order of the derivative. Default is 1.
    args : tuple, optional
        Arguments
    order : int, optional
        Number of points to use, must be odd.

    Notes
    -----
    Decreasing the step size too small can result in round-off error.

    Examples
    --------
    >>> from scipy.misc import derivative
    >>> def f(x):
    ...     return x**3 + x**2
    >>> derivative(f, 1.0, dx=1e-6)
    4.9999999999217337

    """
    if order < n + 1:
        raise ValueError("'order' (the number of points used to compute the derivative), "
                         "must be at least the derivative order 'n' + 1.")
    if order % 2 == 0:
        raise ValueError("'order' (the number of points used to compute the derivative) "
                         "must be odd.")
    # pre-computed for n=1 and 2 and low-order for speed.
    if n == 1:
        if order == 3:
            weights = array([-1,0,1])/2.0
        elif order == 5:
            weights = array([1,-8,0,8,-1])/12.0
        elif order == 7:
            weights = array([-1,9,-45,0,45,-9,1])/60.0
        elif order == 9:
            weights = array([3,-32,168,-672,0,672,-168,32,-3])/840.0
        else:
            weights = central_diff_weights(order,1)
    elif n == 2:
        if order == 3:
            weights = array([1,-2.0,1])
        elif order == 5:
            weights = array([-1,16,-30,16,-1])/12.0
        elif order == 7:
            weights = array([2,-27,270,-490,270,-27,2])/180.0
        elif order == 9:
            weights = array([-9,128,-1008,8064,-14350,8064,-1008,128,-9])/5040.0
        else:
            weights = central_diff_weights(order,2)
    else:
        weights = central_diff_weights(order, n)
    val = 0.0
    ho = order >> 1
    for k in range(order):
        val += weights[k]*func(x0+(k-ho)*dx,*args)
    return val / prod((dx,)*n,axis=0)


def ascent():
    """
    Get an 8-bit grayscale bit-depth, 512 x 512 derived image for easy use in demos

    The image is derived from accent-to-the-top.jpg at
    http://www.public-domain-image.com/people-public-domain-images-pictures/

    Parameters
    ----------
    None

    Returns
    -------
    ascent : ndarray
       convenient image to use for testing and demonstration

    Examples
    --------
    >>> import scipy.misc
    >>> ascent = scipy.misc.ascent()
    >>> ascent.shape
    (512, 512)
    >>> ascent.max()
    255

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(ascent)
    >>> plt.show()

    """
    import pickle
    import os
    fname = os.path.join(os.path.dirname(__file__),'ascent.dat')
    with open(fname, 'rb') as f:
        ascent = array(pickle.load(f))
    return ascent


def face(gray=False):
    """
    Get a 1024 x 768, color image of a raccoon face.

    raccoon-procyon-lotor.jpg at http://www.public-domain-image.com

    Parameters
    ----------
    gray : bool, optional
        If True return 8-bit grey-scale image, otherwise return a color image

    Returns
    -------
    face : ndarray
        image of a racoon face

    Examples
    --------
    >>> import scipy.misc
    >>> face = scipy.misc.face()
    >>> face.shape
    (768, 1024, 3)
    >>> face.max()
    255
    >>> face.dtype
    dtype('uint8')

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(face)
    >>> plt.show()

    """
    import bz2
    import os
    with open(os.path.join(os.path.dirname(__file__), 'face.dat'), 'rb') as f:
        rawdata = f.read()
    data = bz2.decompress(rawdata)
    face = frombuffer(data, dtype='uint8')
    face.shape = (768, 1024, 3)
    if gray is True:
        face = (0.21 * face[:,:,0] + 0.71 * face[:,:,1] + 0.07 * face[:,:,2]).astype('uint8')
    return face


def electrocardiogram():
    """
    Load an electrocardiogram as an example for a 1-D signal.

    The returned signal is a 5 minute long electrocardiogram (ECG), a medical
    recording of the heart's electrical activity, sampled at 360 Hz.

    Returns
    -------
    ecg : ndarray
        The electrocardiogram in millivolt (mV) sampled at 360 Hz.

    Notes
    -----
    The provided signal is an excerpt (19:35 to 24:35) from the `record 208`_
    (lead MLII) provided by the MIT-BIH Arrhythmia Database [1]_ on
    PhysioNet [2]_. The excerpt includes noise induced artifacts, typical
    heartbeats as well as pathological changes.

    .. _record 208: https://physionet.org/physiobank/database/html/mitdbdir/records.htm#208

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
           IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
           (PMID: 11446209); :doi:`10.13026/C2F305`
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank,
           PhysioToolkit, and PhysioNet: Components of a New Research Resource
           for Complex Physiologic Signals. Circulation 101(23):e215-e220;
           :doi:`10.1161/01.CIR.101.23.e215`

    Examples
    --------
    >>> from scipy.misc import electrocardiogram
    >>> ecg = electrocardiogram()
    >>> ecg
    array([-0.245, -0.215, -0.185, ..., -0.405, -0.395, -0.385])
    >>> ecg.shape, ecg.mean(), ecg.std()
    ((108000,), -0.16510875, 0.5992473991177294)

    As stated the signal features several areas with a different morphology.
    E.g., the first few seconds show the electrical activity of a heart in
    normal sinus rhythm as seen below.

    >>> import matplotlib.pyplot as plt
    >>> fs = 360
    >>> time = np.arange(ecg.size) / fs
    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(9, 10.2)
    >>> plt.ylim(-1, 1.5)
    >>> plt.show()

    After second 16, however, the first premature ventricular contractions, also
    called extrasystoles, appear. These have a different morphology compared to
    typical heartbeats. The difference can easily be observed in the following
    plot.

    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(46.5, 50)
    >>> plt.ylim(-2, 1.5)
    >>> plt.show()

    At several points large artifacts disturb the recording, e.g.:

    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(207, 215)
    >>> plt.ylim(-2, 3.5)
    >>> plt.show()

    Finally, examining the power spectrum reveals that most of the biosignal is
    made up of lower frequencies. At 60 Hz the noise induced by the mains
    electricity can be clearly observed.

    >>> from scipy.signal import welch
    >>> f, Pxx = welch(ecg, fs=fs, nperseg=2048, scaling="spectrum")
    >>> plt.semilogy(f, Pxx)
    >>> plt.xlabel("Frequency in Hz")
    >>> plt.ylabel("Power spectrum of the ECG in mV**2")
    >>> plt.xlim(f[[0, -1]])
    >>> plt.show()
    """
    import os
    file_path = os.path.join(os.path.dirname(__file__), "ecg.dat")
    with load(file_path) as file:
        ecg = file["ecg"].astype(int)  # np.uint16 -> int
    # Convert raw output of ADC to mV: (ecg - adc_zero) / adc_gain
    ecg = (ecg - 1024) / 200.0
    return ecg
