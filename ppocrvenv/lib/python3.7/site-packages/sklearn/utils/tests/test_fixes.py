# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Justin Vincent
#          Lars Buitinck
# License: BSD 3 clause

import math

import numpy as np
import pytest
import scipy.stats

from sklearn.utils._testing import assert_array_equal

from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.fixes import _object_dtype_isnan
from sklearn.utils.fixes import loguniform
from sklearn.utils.fixes import linspace, parse_version, np_version


@pytest.mark.parametrize("joblib_version", ("0.11", "0.12.0"))
def test_joblib_parallel_args(monkeypatch, joblib_version):
    import joblib

    monkeypatch.setattr(joblib, "__version__", joblib_version)

    if joblib_version == "0.12.0":
        # arguments are simply passed through
        assert _joblib_parallel_args(prefer="threads") == {"prefer": "threads"}
        assert _joblib_parallel_args(prefer="processes", require=None) == {
            "prefer": "processes",
            "require": None,
        }
        assert _joblib_parallel_args(non_existing=1) == {"non_existing": 1}
    elif joblib_version == "0.11":
        # arguments are mapped to the corresponding backend
        assert _joblib_parallel_args(prefer="threads") == {"backend": "threading"}
        assert _joblib_parallel_args(prefer="processes") == {
            "backend": "multiprocessing"
        }
        with pytest.raises(ValueError):
            _joblib_parallel_args(prefer="invalid")
        assert _joblib_parallel_args(prefer="processes", require="sharedmem") == {
            "backend": "threading"
        }
        with pytest.raises(ValueError):
            _joblib_parallel_args(require="invalid")
        with pytest.raises(NotImplementedError):
            _joblib_parallel_args(verbose=True)
    else:
        raise ValueError


@pytest.mark.parametrize("dtype, val", ([object, 1], [object, "a"], [float, 1]))
def test_object_dtype_isnan(dtype, val):
    X = np.array([[val, np.nan], [np.nan, val]], dtype=dtype)

    expected_mask = np.array([[False, True], [True, False]])

    mask = _object_dtype_isnan(X)

    assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize("low,high,base", [(-1, 0, 10), (0, 2, np.exp(1)), (-1, 1, 2)])
def test_loguniform(low, high, base):
    rv = loguniform(base ** low, base ** high)
    assert isinstance(rv, scipy.stats._distn_infrastructure.rv_frozen)
    rvs = rv.rvs(size=2000, random_state=0)

    # Test the basics; right bounds, right size
    assert (base ** low <= rvs).all() and (rvs <= base ** high).all()
    assert len(rvs) == 2000

    # Test that it's actually (fairly) uniform
    log_rvs = np.array([math.log(x, base) for x in rvs])
    counts, _ = np.histogram(log_rvs)
    assert counts.mean() == 200
    assert np.abs(counts - counts.mean()).max() <= 40

    # Test that random_state works
    assert loguniform(base ** low, base ** high).rvs(random_state=0) == loguniform(
        base ** low, base ** high
    ).rvs(random_state=0)


def test_linspace():
    """Test that linespace works like np.linespace as of numpy version 1.16."""
    start, stop = 0, 10
    num = 6
    out = linspace(start=start, stop=stop, num=num, endpoint=True)
    assert_array_equal(out, np.array([0.0, 2, 4, 6, 8, 10]))

    start, stop = [0, 100], [10, 1100]
    num = 6
    out = linspace(start=start, stop=stop, num=num, endpoint=True)
    res = np.c_[[0.0, 2, 4, 6, 8, 10], [100, 300, 500, 700, 900, 1100]]
    assert_array_equal(out, res)

    out2 = linspace(start=start, stop=stop, num=num, endpoint=True, axis=1)
    assert_array_equal(out2, out.T)

    out, step = linspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=True,
        retstep=True,
    )
    assert_array_equal(out, res)
    assert_array_equal(step, [2, 200])

    if np_version < parse_version("1.16"):
        with pytest.raises(ValueError):
            linspace(start=[0, 1], stop=10)
    else:
        linspace(start=[0, 1], stop=10)
