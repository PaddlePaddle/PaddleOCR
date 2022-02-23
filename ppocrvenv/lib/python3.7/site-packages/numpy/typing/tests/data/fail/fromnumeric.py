"""Tests for :mod:`numpy.core.fromnumeric`."""

import numpy as np

A = np.array(True, ndmin=2, dtype=bool)
A.setflags(write=False)

a = np.bool_(True)

np.take(a, None)  # E: incompatible type
np.take(a, axis=1.0)  # E: incompatible type
np.take(A, out=1)  # E: incompatible type
np.take(A, mode="bob")  # E: incompatible type

np.reshape(a, None)  # E: Argument 2 to "reshape" has incompatible type
np.reshape(A, 1, order="bob")  # E: Argument "order" to "reshape" has incompatible type

np.choose(a, None)  # E: incompatible type
np.choose(a, out=1.0)  # E: incompatible type
np.choose(A, mode="bob")  # E: incompatible type

np.repeat(a, None)  # E: Argument 2 to "repeat" has incompatible type
np.repeat(A, 1, axis=1.0)  # E: Argument "axis" to "repeat" has incompatible type

np.swapaxes(A, None, 1)  # E: Argument 2 to "swapaxes" has incompatible type
np.swapaxes(A, 1, [0])  # E: Argument 3 to "swapaxes" has incompatible type

np.transpose(A, axes=1.0)  # E: Argument "axes" to "transpose" has incompatible type

np.partition(a, None)  # E: Argument 2 to "partition" has incompatible type
np.partition(
    a, 0, axis="bob"  # E: Argument "axis" to "partition" has incompatible type
)
np.partition(
    A, 0, kind="bob"  # E: Argument "kind" to "partition" has incompatible type
)
np.partition(
    A, 0, order=range(5)  # E: Argument "order" to "partition" has incompatible type
)

np.argpartition(
    a, None  # E: incompatible type
)
np.argpartition(
    a, 0, axis="bob"  # E: incompatible type
)
np.argpartition(
    A, 0, kind="bob"  # E: incompatible type
)
np.argpartition(
    A, 0, order=range(5)  # E: Argument "order" to "argpartition" has incompatible type
)

np.sort(A, axis="bob")  # E: Argument "axis" to "sort" has incompatible type
np.sort(A, kind="bob")  # E: Argument "kind" to "sort" has incompatible type
np.sort(A, order=range(5))  # E: Argument "order" to "sort" has incompatible type

np.argsort(A, axis="bob")  # E: Argument "axis" to "argsort" has incompatible type
np.argsort(A, kind="bob")  # E: Argument "kind" to "argsort" has incompatible type
np.argsort(A, order=range(5))  # E: Argument "order" to "argsort" has incompatible type

np.argmax(A, axis="bob")  # E: No overload variant of "argmax" matches argument type
np.argmax(A, kind="bob")  # E: No overload variant of "argmax" matches argument type

np.argmin(A, axis="bob")  # E: No overload variant of "argmin" matches argument type
np.argmin(A, kind="bob")  # E: No overload variant of "argmin" matches argument type

np.searchsorted(  # E: No overload variant of "searchsorted" matches argument type
    A[0], 0, side="bob"
)
np.searchsorted(  # E: No overload variant of "searchsorted" matches argument type
    A[0], 0, sorter=1.0
)

np.resize(A, 1.0)  # E: Argument 2 to "resize" has incompatible type

np.squeeze(A, 1.0)  # E: No overload variant of "squeeze" matches argument type

np.diagonal(A, offset=None)  # E: Argument "offset" to "diagonal" has incompatible type
np.diagonal(A, axis1="bob")  # E: Argument "axis1" to "diagonal" has incompatible type
np.diagonal(A, axis2=[])  # E: Argument "axis2" to "diagonal" has incompatible type

np.trace(A, offset=None)  # E: Argument "offset" to "trace" has incompatible type
np.trace(A, axis1="bob")  # E: Argument "axis1" to "trace" has incompatible type
np.trace(A, axis2=[])  # E: Argument "axis2" to "trace" has incompatible type

np.ravel(a, order="bob")  # E: Argument "order" to "ravel" has incompatible type

np.compress(
    [True], A, axis=1.0  # E: Argument "axis" to "compress" has incompatible type
)

np.clip(a, 1, 2, out=1)  # E: No overload variant of "clip" matches argument type
np.clip(1, None, None)  # E: No overload variant of "clip" matches argument type

np.sum(a, axis=1.0)  # E: incompatible type
np.sum(a, keepdims=1.0)  # E: incompatible type
np.sum(a, initial=[1])  # E: incompatible type

np.all(a, axis=1.0)  # E: No overload variant
np.all(a, keepdims=1.0)  # E: No overload variant
np.all(a, out=1.0)  # E: No overload variant

np.any(a, axis=1.0)  # E: No overload variant
np.any(a, keepdims=1.0)  # E: No overload variant
np.any(a, out=1.0)  # E: No overload variant

np.cumsum(a, axis=1.0)  # E: incompatible type
np.cumsum(a, dtype=1.0)  # E: incompatible type
np.cumsum(a, out=1.0)  # E: incompatible type

np.ptp(a, axis=1.0)  # E: incompatible type
np.ptp(a, keepdims=1.0)  # E: incompatible type
np.ptp(a, out=1.0)  # E: incompatible type

np.amax(a, axis=1.0)  # E: incompatible type
np.amax(a, keepdims=1.0)  # E: incompatible type
np.amax(a, out=1.0)  # E: incompatible type
np.amax(a, initial=[1.0])  # E: incompatible type
np.amax(a, where=[1.0])  # E: incompatible type

np.amin(a, axis=1.0)  # E: incompatible type
np.amin(a, keepdims=1.0)  # E: incompatible type
np.amin(a, out=1.0)  # E: incompatible type
np.amin(a, initial=[1.0])  # E: incompatible type
np.amin(a, where=[1.0])  # E: incompatible type

np.prod(a, axis=1.0)  # E: incompatible type
np.prod(a, out=False)  # E: incompatible type
np.prod(a, keepdims=1.0)  # E: incompatible type
np.prod(a, initial=int)  # E: incompatible type
np.prod(a, where=1.0)  # E: incompatible type

np.cumprod(a, axis=1.0)  # E: Argument "axis" to "cumprod" has incompatible type
np.cumprod(a, out=False)  # E: Argument "out" to "cumprod" has incompatible type

np.size(a, axis=1.0)  # E: Argument "axis" to "size" has incompatible type

np.around(a, decimals=1.0)  # E: incompatible type
np.around(a, out=type)  # E: incompatible type

np.mean(a, axis=1.0)  # E: incompatible type
np.mean(a, out=False)  # E: incompatible type
np.mean(a, keepdims=1.0)  # E: incompatible type

np.std(a, axis=1.0)  # E: incompatible type
np.std(a, out=False)  # E: incompatible type
np.std(a, ddof='test')  # E: incompatible type
np.std(a, keepdims=1.0)  # E: incompatible type

np.var(a, axis=1.0)  # E: incompatible type
np.var(a, out=False)  # E: incompatible type
np.var(a, ddof='test')  # E: incompatible type
np.var(a, keepdims=1.0)  # E: incompatible type
