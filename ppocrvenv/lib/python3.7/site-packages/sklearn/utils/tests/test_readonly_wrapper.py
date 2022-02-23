import numpy as np

import pytest

from sklearn.utils._readonly_array_wrapper import ReadonlyArrayWrapper, _test_sum
from sklearn.utils._testing import create_memmap_backed_data


def _readonly_array_copy(x):
    """Return a copy of x with flag writeable set to False."""
    y = x.copy()
    y.flags["WRITEABLE"] = False
    return y


def _create_memmap_backed_data(data):
    return create_memmap_backed_data(
        data, mmap_mode="r", return_folder=False, aligned=True
    )


@pytest.mark.parametrize("readonly", [_readonly_array_copy, _create_memmap_backed_data])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_readonly_array_wrapper(readonly, dtype):
    """Test that ReadonlyWrapper allows working with fused-typed."""
    x = np.arange(10).astype(dtype)
    sum_origin = _test_sum(x)

    # ReadonlyArrayWrapper works with writable buffers
    sum_writable = _test_sum(ReadonlyArrayWrapper(x))
    assert sum_writable == pytest.approx(sum_origin, rel=1e-11)

    # Now, check on readonly buffers
    x_readonly = readonly(x)

    with pytest.raises(ValueError, match="buffer source array is read-only"):
        _test_sum(x_readonly)

    x_readonly = ReadonlyArrayWrapper(x_readonly)
    sum_readonly = _test_sum(x_readonly)
    assert sum_readonly == pytest.approx(sum_origin, rel=1e-11)
