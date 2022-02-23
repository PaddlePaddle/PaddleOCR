from typing import List

__all__: List[str]

def nanmin(a, axis=..., out=..., keepdims=...): ...
def nanmax(a, axis=..., out=..., keepdims=...): ...
def nanargmin(a, axis=...): ...
def nanargmax(a, axis=...): ...
def nansum(a, axis=..., dtype=..., out=..., keepdims=...): ...
def nanprod(a, axis=..., dtype=..., out=..., keepdims=...): ...
def nancumsum(a, axis=..., dtype=..., out=...): ...
def nancumprod(a, axis=..., dtype=..., out=...): ...
def nanmean(a, axis=..., dtype=..., out=..., keepdims=...): ...
def nanmedian(
    a,
    axis=...,
    out=...,
    overwrite_input=...,
    keepdims=...,
): ...
def nanpercentile(
    a,
    q,
    axis=...,
    out=...,
    overwrite_input=...,
    interpolation=...,
    keepdims=...,
): ...
def nanquantile(
    a,
    q,
    axis=...,
    out=...,
    overwrite_input=...,
    interpolation=...,
    keepdims=...,
): ...
def nanvar(
    a,
    axis=...,
    dtype=...,
    out=...,
    ddof=...,
    keepdims=...,
): ...
def nanstd(
    a,
    axis=...,
    dtype=...,
    out=...,
    ddof=...,
    keepdims=...,
): ...
