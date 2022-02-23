import sys
from typing import overload, Optional, Any, Union, Tuple, SupportsFloat

import numpy as np
from numpy.typing import ArrayLike

if sys.version_info >= (3, 8):
    from typing import Literal, Protocol, SupportsIndex
else:
    from typing_extensions import Literal, Protocol

# Anything that can be parsed by `np.float64.__init__` and is thus
# compatible with `ndarray.__setitem__` (for a float64 array)
if sys.version_info >= (3, 8):
    _FloatValue = Union[None, str, bytes, SupportsFloat, SupportsIndex]
else:
    _FloatValue = Union[None, str, bytes, SupportsFloat]

class _MetricCallback1(Protocol):
    def __call__(
        self, __XA: np.ndarray, __XB: np.ndarray
    ) -> _FloatValue: ...

class _MetricCallback2(Protocol):
    def __call__(
        self, __XA: np.ndarray, __XB: np.ndarray, **kwargs: Any
    ) -> _FloatValue: ...

# TODO: Use a single protocol with a parameter specification variable
# once available (PEP 612)
_MetricCallback = Union[_MetricCallback1, _MetricCallback2]

_MetricKind = Literal[
    'braycurtis',
    'canberra',
    'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch',
    'cityblock', 'cblock', 'cb', 'c',
    'correlation', 'co',
    'cosine', 'cos',
    'dice',
    'euclidean', 'euclid', 'eu', 'e',
    'matching', 'hamming', 'hamm', 'ha', 'h',
    'minkowski', 'mi', 'm', 'pnorm',
    'jaccard', 'jacc', 'ja', 'j',
    'jensenshannon', 'js',
    'kulsinski',
    'mahalanobis', 'mahal', 'mah',
    'rogerstanimoto',
    'russellrao',
    'seuclidean', 'se', 's',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean', 'sqe', 'sqeuclid',
    # NOTE: deprecated
    # 'wminkowski', 'wmi', 'wm', 'wpnorm',
    'yule',
]

# Function annotations

def braycurtis(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> np.float64: ...

def canberra(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> np.float64: ...

# TODO: Add `metric`-specific overloads
@overload
def cdist(
    XA: ArrayLike,
    XB: ArrayLike,
    metric: _MetricKind = ...,
    *,
    out: Optional[np.ndarray] = ...,
    p: float = ...,
    w: Optional[ArrayLike] = ...,
    V: Optional[ArrayLike] = ...,
    VI: Optional[ArrayLike] = ...,
) -> np.ndarray: ...
@overload
def cdist(
    XA: ArrayLike,
    XB: ArrayLike,
    metric: _MetricCallback,
    *,
    out: Optional[np.ndarray] = ...,
    **kwargs: Any,
) -> np.ndarray: ...

# TODO: Wait for dtype support; the return type is
# dependent on the input arrays dtype
def chebyshev(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> Any: ...

# TODO: Wait for dtype support; the return type is
# dependent on the input arrays dtype
def cityblock(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> Any: ...

def correlation(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ..., centered: bool = ...
) -> np.float64: ...

def cosine(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> np.float64: ...

def dice(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> float: ...

def directed_hausdorff(
    u: ArrayLike, v: ArrayLike, seed: Optional[int] = ...
) -> Tuple[float, int, int]: ...

def euclidean(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> float: ...

def hamming(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> np.float64: ...

def is_valid_dm(
    D: ArrayLike,
    tol: float = ...,
    throw: bool = ...,
    name: Optional[str] = ...,
    warning: bool = ...,
) -> bool: ...

def is_valid_y(
    y: ArrayLike,
    warning: bool = ...,
    throw: bool = ...,
    name: Optional[str] = ...,
) -> bool: ...

def jaccard(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> np.float64: ...

def jensenshannon(
    p: ArrayLike, q: ArrayLike, base: Optional[float] = ...
) -> np.float64: ...

def kulsinski(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> np.float64: ...

def mahalanobis(
    u: ArrayLike, v: ArrayLike, VI: ArrayLike
) -> np.float64: ...

# NOTE: deprecated
# def matching(u, v, w=None): ...

def minkowski(
    u: ArrayLike, v: ArrayLike, p: float = ..., w: Optional[ArrayLike] = ...
) -> float: ...

def num_obs_dm(d: ArrayLike) -> int: ...

def num_obs_y(Y: ArrayLike) -> int: ...

# TODO: Add `metric`-specific overloads
@overload
def pdist(
    X: ArrayLike,
    metric: _MetricKind = ...,
    *,
    out: Optional[np.ndarray] = ...,
    p: float = ...,
    w: Optional[ArrayLike] = ...,
    V: Optional[ArrayLike] = ...,
    VI: Optional[ArrayLike] = ...,
) -> np.ndarray: ...
@overload
def pdist(
    X: ArrayLike,
    metric: _MetricCallback,
    *,
    out: Optional[np.ndarray] = ...,
    **kwargs: Any,
) -> np.ndarray: ...

def seuclidean(
    u: ArrayLike, v: ArrayLike, V: ArrayLike
) -> float: ...

def sokalmichener(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> float: ...

def sokalsneath(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> np.float64: ...

def sqeuclidean(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> np.float64: ...

def squareform(
    X: ArrayLike,
    force: Literal["no", "tomatrix", "tovector"] = ...,
    checks: bool = ...,
) -> np.ndarray: ...

def rogerstanimoto(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> float: ...

def russellrao(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> float: ...

# NOTE: deprecated
# def wminkowski(u, v, p, w): ...

def yule(
    u: ArrayLike, v: ArrayLike, w: Optional[ArrayLike] = ...
) -> float: ...
