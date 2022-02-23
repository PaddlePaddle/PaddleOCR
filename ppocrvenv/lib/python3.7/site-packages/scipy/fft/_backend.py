import scipy._lib.uarray as ua
from . import _pocketfft


class _ScipyBackend:
    """The default backend for fft calculations

    Notes
    -----
    We use the domain ``numpy.scipy`` rather than ``scipy`` because in the
    future, ``uarray`` will treat the domain as a hierarchy. This means the user
    can install a single backend for ``numpy`` and have it implement
    ``numpy.scipy.fft`` as well.
    """
    __ua_domain__ = "numpy.scipy.fft"

    @staticmethod
    def __ua_function__(method, args, kwargs):
        fn = getattr(_pocketfft, method.__name__, None)

        if fn is None:
            return NotImplemented
        return fn(*args, **kwargs)


_named_backends = {
    'scipy': _ScipyBackend,
}


def _backend_from_arg(backend):
    """Maps strings to known backends and validates the backend"""

    if isinstance(backend, str):
        try:
            backend = _named_backends[backend]
        except KeyError as e:
            raise ValueError('Unknown backend {}'.format(backend)) from e

    if backend.__ua_domain__ != 'numpy.scipy.fft':
        raise ValueError('Backend does not implement "numpy.scipy.fft"')

    return backend


def set_global_backend(backend):
    """Sets the global fft backend

    The global backend has higher priority than registered backends, but lower
    priority than context-specific backends set with `set_backend`.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.

    Raises
    ------
    ValueError: If the backend does not implement ``numpy.scipy.fft``.

    Notes
    -----
    This will overwrite the previously set global backend, which, by default, is
    the SciPy implementation.

    Examples
    --------
    We can set the global fft backend:

    >>> from scipy.fft import fft, set_global_backend
    >>> set_global_backend("scipy")  # Sets global backend. "scipy" is the default backend.
    >>> fft([1])  # Calls the global backend
    array([1.+0.j])
    """
    backend = _backend_from_arg(backend)
    ua.set_global_backend(backend)


def register_backend(backend):
    """
    Register a backend for permanent use.

    Registered backends have the lowest priority and will be tried after the
    global backend.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.

    Raises
    ------
    ValueError: If the backend does not implement ``numpy.scipy.fft``.

    Examples
    --------
    We can register a new fft backend:

    >>> from scipy.fft import fft, register_backend, set_global_backend
    >>> class NoopBackend:  # Define an invalid Backend
    ...     __ua_domain__ = "numpy.scipy.fft"
    ...     def __ua_function__(self, func, args, kwargs):
    ...          return NotImplemented
    >>> set_global_backend(NoopBackend())  # Set the invalid backend as global
    >>> register_backend("scipy")  # Register a new backend
    >>> fft([1])  # The registered backend is called because the global backend returns `NotImplemented`
    array([1.+0.j])
    >>> set_global_backend("scipy")  # Restore global backend to default

    """
    backend = _backend_from_arg(backend)
    ua.register_backend(backend)


def set_backend(backend, coerce=False, only=False):
    """Context manager to set the backend within a fixed scope.

    Upon entering the ``with`` statement, the given backend will be added to
    the list of available backends with the highest priority. Upon exit, the
    backend is reset to the state before entering the scope.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.
    coerce : bool, optional
        Whether to allow expensive conversions for the ``x`` parameter. e.g.,
        copying a NumPy array to the GPU for a CuPy backend. Implies ``only``.
    only : bool, optional
        If only is ``True`` and this backend returns ``NotImplemented``, then a
        BackendNotImplemented error will be raised immediately. Ignoring any
        lower priority backends.

    Examples
    --------
    >>> import scipy.fft as fft
    >>> with fft.set_backend('scipy', only=True):
    ...     fft.fft([1])  # Always calls the scipy implementation
    array([1.+0.j])
    """
    backend = _backend_from_arg(backend)
    return ua.set_backend(backend, coerce=coerce, only=only)


def skip_backend(backend):
    """Context manager to skip a backend within a fixed scope.

    Within the context of a ``with`` statement, the given backend will not be
    called. This covers backends registered both locally and globally. Upon
    exit, the backend will again be considered.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to skip.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.

    Examples
    --------
    >>> import scipy.fft as fft
    >>> fft.fft([1])  # Calls default SciPy backend
    array([1.+0.j])
    >>> with fft.skip_backend('scipy'):  # We explicitly skip the SciPy backend
    ...     fft.fft([1])                 # leaving no implementation available
    Traceback (most recent call last):
        ...
    BackendNotImplementedError: No selected backends had an implementation ...
    """
    backend = _backend_from_arg(backend)
    return ua.skip_backend(backend)


set_global_backend('scipy')
