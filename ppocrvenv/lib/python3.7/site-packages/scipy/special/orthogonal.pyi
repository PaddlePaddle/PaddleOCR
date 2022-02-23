from typing import (
    Any,
    Callable,
    List,
    Optional,
    overload,
    Tuple,
    Union,
)
from typing_extensions import Literal

import numpy

_IntegerType = Union[int, numpy.integer]
_FloatingType = Union[float, numpy.floating]
_PointsAndWeights = Tuple[numpy.ndarray, numpy.ndarray]
_PointsAndWeightsAndMu = Tuple[numpy.ndarray, numpy.ndarray, float]

__all__ = [
    'legendre',
    'chebyt',
    'chebyu',
    'chebyc',
    'chebys',
    'jacobi',
    'laguerre',
    'genlaguerre',
    'hermite',
    'hermitenorm',
    'gegenbauer',
    'sh_legendre',
    'sh_chebyt',
    'sh_chebyu',
    'sh_jacobi',
    'roots_legendre',
    'roots_chebyt',
    'roots_chebyu',
    'roots_chebyc',
    'roots_chebys',
    'roots_jacobi',
    'roots_laguerre',
    'roots_genlaguerre',
    'roots_hermite',
    'roots_hermitenorm',
    'roots_gegenbauer',
    'roots_sh_legendre',
    'roots_sh_chebyt',
    'roots_sh_chebyu',
    'roots_sh_jacobi',
]

@overload
def roots_jacobi(
        n: _IntegerType,
        alpha: _FloatingType,
        beta: _FloatingType,
) -> _PointsAndWeights: ...
@overload
def roots_jacobi(
        n: _IntegerType,
        alpha: _FloatingType,
        beta: _FloatingType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_jacobi(
        n: _IntegerType,
        alpha: _FloatingType,
        beta: _FloatingType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_sh_jacobi(
        n: _IntegerType,
        p1: _FloatingType,
        q1: _FloatingType,
) -> _PointsAndWeights: ...
@overload
def roots_sh_jacobi(
        n: _IntegerType,
        p1: _FloatingType,
        q1: _FloatingType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_sh_jacobi(
        n: _IntegerType,
        p1: _FloatingType,
        q1: _FloatingType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_genlaguerre(
        n: _IntegerType,
        alpha: _FloatingType,
) -> _PointsAndWeights: ...
@overload
def roots_genlaguerre(
        n: _IntegerType,
        alpha: _FloatingType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_genlaguerre(
        n: _IntegerType,
        alpha: _FloatingType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_laguerre(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_laguerre(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_laguerre(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_hermite(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_hermite(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_hermite(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_hermitenorm(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_hermitenorm(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_hermitenorm(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_gegenbauer(
        n: _IntegerType,
        alpha: _FloatingType,
) -> _PointsAndWeights: ...
@overload
def roots_gegenbauer(
        n: _IntegerType,
        alpha: _FloatingType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_gegenbauer(
        n: _IntegerType,
        alpha: _FloatingType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_chebyt(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_chebyt(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_chebyt(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_chebyu(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_chebyu(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_chebyu(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_chebyc(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_chebyc(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_chebyc(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_chebys(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_chebys(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_chebys(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_sh_chebyt(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_sh_chebyt(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_sh_chebyt(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_sh_chebyu(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_sh_chebyu(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_sh_chebyu(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_legendre(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_legendre(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_legendre(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

@overload
def roots_sh_legendre(n: _IntegerType) -> _PointsAndWeights: ...
@overload
def roots_sh_legendre(
        n: _IntegerType,
        mu: Literal[False],
) -> _PointsAndWeights: ...
@overload
def roots_sh_legendre(
        n: _IntegerType,
        mu: Literal[True],
) -> _PointsAndWeightsAndMu: ...

class orthopoly1d(numpy.poly1d):
    def __init__(
            self,
            roots: Any,  # TODO: ArrayLike
            weights: Optional[Any],  # TODO: ArrayLike
            hn: float = ...,
            kn: float = ...,
            wfunc = Optional[Callable[[float], float]],
            limits = Optional[Tuple[float, float]],
            monic: bool = ...,
            eval_func: numpy.ufunc = ...,
    ) -> None: ...
    @property
    def limits(self) -> Tuple[float, float]: ...
    def weight_func(self, x: float) -> float: ...
    # TODO: ArrayLike
    def __call__(self, x: Any) -> Any: ...

def legendre(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def chebyt(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def chebyu(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def chebyc(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def chebys(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def jacobi(
        n: _IntegerType,
        alpha: _FloatingType,
        beta: _FloatingType,
        monic: bool = ...,
) -> orthopoly1d: ...
def laguerre(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def genlaguerre(
        n: _IntegerType,
        alpha: _FloatingType,
        monic: bool = ...,
) -> orthopoly1d: ...
def hermite(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def hermitenorm(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def gegenbauer(
        n: _IntegerType,
        alpha: _FloatingType,
        monic: bool = ...,
) -> orthopoly1d: ...
def sh_legendre(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def sh_chebyt(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def sh_chebyu(n: _IntegerType, monic: bool = ...) -> orthopoly1d: ...
def sh_jacobi(
        n: _IntegerType,
        p: _FloatingType,
        q: _FloatingType,
        monic: bool = ...,
) -> orthopoly1d: ...

# These functions are not public, but still need stubs because they
# get checked in the tests.
def _roots_hermite_asy(n: _IntegerType) -> _PointsAndWeights: ...
