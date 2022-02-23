"""
Low-level LAPACK functions (:mod:`scipy.linalg.lapack`)
=======================================================

This module contains low-level functions from the LAPACK library.

The `*gegv` family of routines have been removed from LAPACK 3.6.0
and have been deprecated in SciPy 0.17.0. They will be removed in
a future release.

.. versionadded:: 0.12.0

.. note::

    The common ``overwrite_<>`` option in many routines, allows the
    input arrays to be overwritten to avoid extra memory allocation.
    However this requires the array to satisfy two conditions
    which are memory order and the data type to match exactly the
    order and the type expected by the routine.

    As an example, if you pass a double precision float array to any
    ``S....`` routine which expects single precision arguments, f2py
    will create an intermediate array to match the argument types and
    overwriting will be performed on that intermediate array.

    Similarly, if a C-contiguous array is passed, f2py will pass a
    FORTRAN-contiguous array internally. Please make sure that these
    details are satisfied. More information can be found in the f2py
    documentation.

.. warning::

   These functions do little to no error checking.
   It is possible to cause crashes by mis-using them,
   so prefer using the higher-level routines in `scipy.linalg`.

Finding functions
-----------------

.. autosummary::
   :toctree: generated/

   get_lapack_funcs

All functions
-------------

.. autosummary::
   :toctree: generated/

   sgbsv
   dgbsv
   cgbsv
   zgbsv

   sgbtrf
   dgbtrf
   cgbtrf
   zgbtrf

   sgbtrs
   dgbtrs
   cgbtrs
   zgbtrs

   sgebal
   dgebal
   cgebal
   zgebal

   sgecon
   dgecon
   cgecon
   zgecon

   sgeequ
   dgeequ
   cgeequ
   zgeequ

   sgeequb
   dgeequb
   cgeequb
   zgeequb

   sgees
   dgees
   cgees
   zgees

   sgeev
   dgeev
   cgeev
   zgeev

   sgeev_lwork
   dgeev_lwork
   cgeev_lwork
   zgeev_lwork

   sgegv
   dgegv
   cgegv
   zgegv

   sgehrd
   dgehrd
   cgehrd
   zgehrd

   sgehrd_lwork
   dgehrd_lwork
   cgehrd_lwork
   zgehrd_lwork

   sgejsv
   dgejsv

   sgels
   dgels
   cgels
   zgels

   sgels_lwork
   dgels_lwork
   cgels_lwork
   zgels_lwork

   sgelsd
   dgelsd
   cgelsd
   zgelsd

   sgelsd_lwork
   dgelsd_lwork
   cgelsd_lwork
   zgelsd_lwork

   sgelss
   dgelss
   cgelss
   zgelss

   sgelss_lwork
   dgelss_lwork
   cgelss_lwork
   zgelss_lwork

   sgelsy
   dgelsy
   cgelsy
   zgelsy

   sgelsy_lwork
   dgelsy_lwork
   cgelsy_lwork
   zgelsy_lwork

   sgeqp3
   dgeqp3
   cgeqp3
   zgeqp3

   sgeqrf
   dgeqrf
   cgeqrf
   zgeqrf

   sgeqrf_lwork
   dgeqrf_lwork
   cgeqrf_lwork
   zgeqrf_lwork

   sgeqrfp
   dgeqrfp
   cgeqrfp
   zgeqrfp

   sgeqrfp_lwork
   dgeqrfp_lwork
   cgeqrfp_lwork
   zgeqrfp_lwork

   sgerqf
   dgerqf
   cgerqf
   zgerqf

   sgesdd
   dgesdd
   cgesdd
   zgesdd

   sgesdd_lwork
   dgesdd_lwork
   cgesdd_lwork
   zgesdd_lwork

   sgesv
   dgesv
   cgesv
   zgesv

   sgesvd
   dgesvd
   cgesvd
   zgesvd

   sgesvd_lwork
   dgesvd_lwork
   cgesvd_lwork
   zgesvd_lwork

   sgesvx
   dgesvx
   cgesvx
   zgesvx

   sgetrf
   dgetrf
   cgetrf
   zgetrf

   sgetc2
   dgetc2
   cgetc2
   zgetc2

   sgetri
   dgetri
   cgetri
   zgetri

   sgetri_lwork
   dgetri_lwork
   cgetri_lwork
   zgetri_lwork

   sgetrs
   dgetrs
   cgetrs
   zgetrs

   sgesc2
   dgesc2
   cgesc2
   zgesc2

   sgges
   dgges
   cgges
   zgges

   sggev
   dggev
   cggev
   zggev

   sgglse
   dgglse
   cgglse
   zgglse

   sgglse_lwork
   dgglse_lwork
   cgglse_lwork
   zgglse_lwork

   sgtsv
   dgtsv
   cgtsv
   zgtsv

   sgtsvx
   dgtsvx
   cgtsvx
   zgtsvx

   chbevd
   zhbevd

   chbevx
   zhbevx

   checon
   zhecon

   cheequb
   zheequb

   cheev
   zheev

   cheev_lwork
   zheev_lwork

   cheevd
   zheevd

   cheevd_lwork
   zheevd_lwork

   cheevr
   zheevr

   cheevr_lwork
   zheevr_lwork

   cheevx
   zheevx

   cheevx_lwork
   zheevx_lwork

   chegst
   zhegst

   chegv
   zhegv

   chegv_lwork
   zhegv_lwork

   chegvd
   zhegvd

   chegvx
   zhegvx

   chegvx_lwork
   zhegvx_lwork

   chesv
   zhesv

   chesv_lwork
   zhesv_lwork

   chesvx
   zhesvx

   chesvx_lwork
   zhesvx_lwork

   chetrd
   zhetrd

   chetrd_lwork
   zhetrd_lwork

   chetrf
   zhetrf

   chetrf_lwork
   zhetrf_lwork

   chfrk
   zhfrk

   slamch
   dlamch

   slange
   dlange
   clange
   zlange

   slarf
   dlarf
   clarf
   zlarf

   slarfg
   dlarfg
   clarfg
   zlarfg

   slartg
   dlartg
   clartg
   zlartg

   slasd4
   dlasd4

   slaswp
   dlaswp
   claswp
   zlaswp

   slauum
   dlauum
   clauum
   zlauum

   sorcsd
   dorcsd
   sorcsd_lwork
   dorcsd_lwork

   sorghr
   dorghr
   sorghr_lwork
   dorghr_lwork

   sorgqr
   dorgqr

   sorgrq
   dorgrq

   sormqr
   dormqr

   sormrz
   dormrz

   sormrz_lwork
   dormrz_lwork

   spbsv
   dpbsv
   cpbsv
   zpbsv

   spbtrf
   dpbtrf
   cpbtrf
   zpbtrf

   spbtrs
   dpbtrs
   cpbtrs
   zpbtrs

   spftrf
   dpftrf
   cpftrf
   zpftrf

   spftri
   dpftri
   cpftri
   zpftri

   spftrs
   dpftrs
   cpftrs
   zpftrs

   spocon
   dpocon
   cpocon
   zpocon

   spstrf
   dpstrf
   cpstrf
   zpstrf

   spstf2
   dpstf2
   cpstf2
   zpstf2

   sposv
   dposv
   cposv
   zposv

   sposvx
   dposvx
   cposvx
   zposvx

   spotrf
   dpotrf
   cpotrf
   zpotrf

   spotri
   dpotri
   cpotri
   zpotri

   spotrs
   dpotrs
   cpotrs
   zpotrs

   sppcon
   dppcon
   cppcon
   zppcon

   sppsv
   dppsv
   cppsv
   zppsv

   spptrf
   dpptrf
   cpptrf
   zpptrf

   spptri
   dpptri
   cpptri
   zpptri

   spptrs
   dpptrs
   cpptrs
   zpptrs

   sptsv
   dptsv
   cptsv
   zptsv

   sptsvx
   dptsvx
   cptsvx
   zptsvx

   spttrf
   dpttrf
   cpttrf
   zpttrf

   spttrs
   dpttrs
   cpttrs
   zpttrs

   spteqr
   dpteqr
   cpteqr
   zpteqr

   crot
   zrot

   ssbev
   dsbev

   ssbevd
   dsbevd

   ssbevx
   dsbevx

   ssfrk
   dsfrk

   sstebz
   dstebz

   sstein
   dstein

   sstemr
   dstemr

   sstemr_lwork
   dstemr_lwork

   ssterf
   dsterf

   sstev
   dstev

   ssycon
   dsycon
   csycon
   zsycon

   ssyconv
   dsyconv
   csyconv
   zsyconv

   ssyequb
   dsyequb
   csyequb
   zsyequb

   ssyev
   dsyev

   ssyev_lwork
   dsyev_lwork

   ssyevd
   dsyevd

   ssyevd_lwork
   dsyevd_lwork

   ssyevr
   dsyevr

   ssyevr_lwork
   dsyevr_lwork

   ssyevx
   dsyevx

   ssyevx_lwork
   dsyevx_lwork

   ssygst
   dsygst

   ssygv
   dsygv

   ssygv_lwork
   dsygv_lwork

   ssygvd
   dsygvd

   ssygvx
   dsygvx

   ssygvx_lwork
   dsygvx_lwork

   ssysv
   dsysv
   csysv
   zsysv

   ssysv_lwork
   dsysv_lwork
   csysv_lwork
   zsysv_lwork

   ssysvx
   dsysvx
   csysvx
   zsysvx

   ssysvx_lwork
   dsysvx_lwork
   csysvx_lwork
   zsysvx_lwork

   ssytf2
   dsytf2
   csytf2
   zsytf2

   ssytrd
   dsytrd

   ssytrd_lwork
   dsytrd_lwork

   ssytrf
   dsytrf
   csytrf
   zsytrf

   ssytrf_lwork
   dsytrf_lwork
   csytrf_lwork
   zsytrf_lwork

   stbtrs
   dtbtrs
   ctbtrs
   ztbtrs

   stfsm
   dtfsm
   ctfsm
   ztfsm

   stfttp
   dtfttp
   ctfttp
   ztfttp

   stfttr
   dtfttr
   ctfttr
   ztfttr

   stgexc
   dtgexc
   ctgexc
   ztgexc

   stgsen
   dtgsen
   ctgsen
   ztgsen

   stgsen_lwork
   dtgsen_lwork
   ctgsen_lwork
   ztgsen_lwork

   stpttf
   dtpttf
   ctpttf
   ztpttf

   stpttr
   dtpttr
   ctpttr
   ztpttr

   strsyl
   dtrsyl
   ctrsyl
   ztrsyl

   strtri
   dtrtri
   ctrtri
   ztrtri

   strtrs
   dtrtrs
   ctrtrs
   ztrtrs

   strttf
   dtrttf
   ctrttf
   ztrttf

   strttp
   dtrttp
   ctrttp
   ztrttp

   stzrzf
   dtzrzf
   ctzrzf
   ztzrzf

   stzrzf_lwork
   dtzrzf_lwork
   ctzrzf_lwork
   ztzrzf_lwork

   cunghr
   zunghr

   cunghr_lwork
   zunghr_lwork

   cungqr
   zungqr

   cungrq
   zungrq

   cunmqr
   zunmqr

   sgeqrt
   dgeqrt
   cgeqrt
   zgeqrt

   sgemqrt
   dgemqrt
   cgemqrt
   zgemqrt

   sgttrf
   dgttrf
   cgttrf
   zgttrf

   sgttrs
   dgttrs
   cgttrs
   zgttrs

   stpqrt
   dtpqrt
   ctpqrt
   ztpqrt

   stpmqrt
   dtpmqrt
   ctpmqrt
   ztpmqrt

   cuncsd
   zuncsd

   cuncsd_lwork
   zuncsd_lwork

   cunmrz
   zunmrz

   cunmrz_lwork
   zunmrz_lwork

   ilaver

"""
#
# Author: Pearu Peterson, March 2002
#

import numpy as _np
from .blas import _get_funcs, _memoize_get_funcs
from scipy.linalg import _flapack
from re import compile as regex_compile
try:
    from scipy.linalg import _clapack
except ImportError:
    _clapack = None

try:
    from scipy.linalg import _flapack_64
    HAS_ILP64 = True
except ImportError:
    HAS_ILP64 = False
    _flapack_64 = None

# Backward compatibility
from scipy._lib._util import DeprecatedImport as _DeprecatedImport
clapack = _DeprecatedImport("scipy.linalg.blas.clapack", "scipy.linalg.lapack")
flapack = _DeprecatedImport("scipy.linalg.blas.flapack", "scipy.linalg.lapack")

# Expose all functions (only flapack --- clapack is an implementation detail)
empty_module = None
from scipy.linalg._flapack import *
del empty_module

__all__ = ['get_lapack_funcs']

_dep_message = """The `*gegv` family of routines has been deprecated in
LAPACK 3.6.0 in favor of the `*ggev` family of routines.
The corresponding wrappers will be removed from SciPy in
a future release."""

cgegv = _np.deprecate(cgegv, old_name='cgegv', message=_dep_message)
dgegv = _np.deprecate(dgegv, old_name='dgegv', message=_dep_message)
sgegv = _np.deprecate(sgegv, old_name='sgegv', message=_dep_message)
zgegv = _np.deprecate(zgegv, old_name='zgegv', message=_dep_message)

# Modify _flapack in this scope so the deprecation warnings apply to
# functions returned by get_lapack_funcs.
_flapack.cgegv = cgegv
_flapack.dgegv = dgegv
_flapack.sgegv = sgegv
_flapack.zgegv = zgegv

# some convenience alias for complex functions
_lapack_alias = {
    'corghr': 'cunghr', 'zorghr': 'zunghr',
    'corghr_lwork': 'cunghr_lwork', 'zorghr_lwork': 'zunghr_lwork',
    'corgqr': 'cungqr', 'zorgqr': 'zungqr',
    'cormqr': 'cunmqr', 'zormqr': 'zunmqr',
    'corgrq': 'cungrq', 'zorgrq': 'zungrq',
}


# Place guards against docstring rendering issues with special characters
p1 = regex_compile(r'with bounds (?P<b>.*?)( and (?P<s>.*?) storage){0,1}\n')
p2 = regex_compile(r'Default: (?P<d>.*?)\n')


def backtickrepl(m):
    if m.group('s'):
        return ('with bounds ``{}`` with ``{}`` storage\n'
                ''.format(m.group('b'), m.group('s')))
    else:
        return 'with bounds ``{}``\n'.format(m.group('b'))


for routine in [ssyevr, dsyevr, cheevr, zheevr,
                ssyevx, dsyevx, cheevx, zheevx,
                ssygvd, dsygvd, chegvd, zhegvd]:
    if routine.__doc__:
        routine.__doc__ = p1.sub(backtickrepl, routine.__doc__)
        routine.__doc__ = p2.sub('Default ``\\1``\n', routine.__doc__)
    else:
        continue

del regex_compile, p1, p2, backtickrepl


@_memoize_get_funcs
def get_lapack_funcs(names, arrays=(), dtype=None, ilp64=False):
    """Return available LAPACK function objects from names.

    Arrays are used to determine the optimal prefix of LAPACK routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of LAPACK functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of LAPACK
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    ilp64 : {True, False, 'preferred'}, optional
        Whether to return ILP64 routine variant.
        Choosing 'preferred' returns ILP64 routine if available, and
        otherwise the 32-bit routine. Default: False

    Returns
    -------
    funcs : list
        List containing the found function(s).

    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In LAPACK, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the NumPy
    types {float32, float64, complex64, complex128} respectively, and
    are stored in attribute ``typecode`` of the returned functions.

    Examples
    --------
    Suppose we would like to use '?lange' routine which computes the selected
    norm of an array. We pass our array in order to get the correct 'lange'
    flavor.

    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()
    >>> a = rng.random((3,2))
    >>> x_lange = LA.get_lapack_funcs('lange', (a,))
    >>> x_lange.typecode
    'd'
    >>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
    >>> x_lange.typecode
    'z'

    Several LAPACK routines work best when its internal WORK array has
    the optimal size (big enough for fast computation and small enough to
    avoid waste of memory). This size is determined also by a dedicated query
    to the function which is often wrapped as a standalone function and
    commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()
    >>> a = rng.random((1000, 1000))
    >>> b = rng.random((1000, 1)) * 1j
    >>> # We pick up zsysv and zsysv_lwork due to b array
    ... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
    >>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
    >>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))

    """
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for 'ilp64'")

    if not ilp64:
        return _get_funcs(names, arrays, dtype,
                          "LAPACK", _flapack, _clapack,
                          "flapack", "clapack", _lapack_alias,
                          ilp64=False)
    else:
        if not HAS_ILP64:
            raise RuntimeError("LAPACK ILP64 routine requested, but Scipy "
                               "compiled only with 32-bit BLAS")
        return _get_funcs(names, arrays, dtype,
                          "LAPACK", _flapack_64, None,
                          "flapack_64", None, _lapack_alias,
                          ilp64=True)


_int32_max = _np.iinfo(_np.int32).max
_int64_max = _np.iinfo(_np.int64).max


def _compute_lwork(routine, *args, **kwargs):
    """
    Round floating-point lwork returned by lapack to integer.

    Several LAPACK routines compute optimal values for LWORK, which
    they return in a floating-point variable. However, for large
    values of LWORK, single-precision floating point is not sufficient
    to hold the exact value --- some LAPACK versions (<= 3.5.0 at
    least) truncate the returned integer to single precision and in
    some cases this can be smaller than the required value.

    Examples
    --------
    >>> from scipy.linalg import lapack
    >>> n = 5000
    >>> s_r, s_lw = lapack.get_lapack_funcs(('sysvx', 'sysvx_lwork'))
    >>> lwork = lapack._compute_lwork(s_lw, n)
    >>> lwork
    32000

    """
    dtype = getattr(routine, 'dtype', None)
    int_dtype = getattr(routine, 'int_dtype', None)
    ret = routine(*args, **kwargs)
    if ret[-1] != 0:
        raise ValueError("Internal work array size computation failed: "
                         "%d" % (ret[-1],))

    if len(ret) == 2:
        return _check_work_float(ret[0].real, dtype, int_dtype)
    else:
        return tuple(_check_work_float(x.real, dtype, int_dtype)
                     for x in ret[:-1])


def _check_work_float(value, dtype, int_dtype):
    """
    Convert LAPACK-returned work array size float to integer,
    carefully for single-precision types.
    """

    if dtype == _np.float32 or dtype == _np.complex64:
        # Single-precision routine -- take next fp value to work
        # around possible truncation in LAPACK code
        value = _np.nextafter(value, _np.inf, dtype=_np.float32)

    value = int(value)
    if int_dtype.itemsize == 4:
        if value < 0 or value > _int32_max:
            raise ValueError("Too large work array required -- computation "
                             "cannot be performed with standard 32-bit"
                             " LAPACK.")
    elif int_dtype.itemsize == 8:
        if value < 0 or value > _int64_max:
            raise ValueError("Too large work array required -- computation"
                             " cannot be performed with standard 64-bit"
                             " LAPACK.")
    return value
