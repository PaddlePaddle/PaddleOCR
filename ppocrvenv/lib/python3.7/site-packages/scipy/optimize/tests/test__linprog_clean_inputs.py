"""
Unit test for Linear Programming via Simplex Algorithm.
"""
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from copy import deepcopy
from datetime import date


def test_aliasing():
    """
    Test for ensuring that no objects referred to by `lp` attributes,
    `c`, `A_ub`, `b_ub`, `A_eq`, `b_eq`, `bounds`, have been modified
    by `_clean_inputs` as a side effect.
    """
    lp = _LPProblem(
        c=1,
        A_ub=[[1]],
        b_ub=[1],
        A_eq=[[1]],
        b_eq=[1],
        bounds=(-np.inf, np.inf)
    )
    lp_copy = deepcopy(lp)

    _clean_inputs(lp)

    assert_(lp.c == lp_copy.c, "c modified by _clean_inputs")
    assert_(lp.A_ub == lp_copy.A_ub, "A_ub modified by _clean_inputs")
    assert_(lp.b_ub == lp_copy.b_ub, "b_ub modified by _clean_inputs")
    assert_(lp.A_eq == lp_copy.A_eq, "A_eq modified by _clean_inputs")
    assert_(lp.b_eq == lp_copy.b_eq, "b_eq modified by _clean_inputs")
    assert_(lp.bounds == lp_copy.bounds, "bounds modified by _clean_inputs")


def test_aliasing2():
    """
    Similar purpose as `test_aliasing` above.
    """
    lp = _LPProblem(
        c=np.array([1, 1]),
        A_ub=np.array([[1, 1], [2, 2]]),
        b_ub=np.array([[1], [1]]),
        A_eq=np.array([[1, 1]]),
        b_eq=np.array([1]),
        bounds=[(-np.inf, np.inf), (None, 1)]
    )
    lp_copy = deepcopy(lp)

    _clean_inputs(lp)

    assert_allclose(lp.c, lp_copy.c, err_msg="c modified by _clean_inputs")
    assert_allclose(lp.A_ub, lp_copy.A_ub, err_msg="A_ub modified by _clean_inputs")
    assert_allclose(lp.b_ub, lp_copy.b_ub, err_msg="b_ub modified by _clean_inputs")
    assert_allclose(lp.A_eq, lp_copy.A_eq, err_msg="A_eq modified by _clean_inputs")
    assert_allclose(lp.b_eq, lp_copy.b_eq, err_msg="b_eq modified by _clean_inputs")
    assert_(lp.bounds == lp_copy.bounds, "bounds modified by _clean_inputs")


def test_missing_inputs():
    c = [1, 2]
    A_ub = np.array([[1, 1], [2, 2]])
    b_ub = np.array([1, 1])
    A_eq = np.array([[1, 1], [2, 2]])
    b_eq = np.array([1, 1])

    assert_raises(TypeError, _clean_inputs)
    assert_raises(TypeError, _clean_inputs, _LPProblem(c=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=A_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=A_ub, b_ub=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, b_ub=b_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=None, b_ub=b_ub))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=A_eq))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=A_eq, b_eq=None))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, b_eq=b_eq))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=None, b_eq=b_eq))


def test_too_many_dimensions():
    cb = [1, 2, 3, 4]
    A = np.random.rand(4, 4)
    bad2D = [[1, 2], [3, 4]]
    bad3D = np.random.rand(4, 4, 4)
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=bad2D, A_ub=A, b_ub=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_ub=bad3D, b_ub=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_ub=A, b_ub=bad2D))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_eq=bad3D, b_eq=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_eq=A, b_eq=bad2D))


def test_too_few_dimensions():
    bad = np.random.rand(4, 4).ravel()
    cb = np.random.rand(4)
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_ub=bad, b_ub=cb))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=cb, A_eq=bad, b_eq=cb))


def test_inconsistent_dimensions():
    m = 2
    n = 4
    c = [1, 2, 3, 4]

    Agood = np.random.rand(m, n)
    Abad = np.random.rand(m, n + 1)
    bgood = np.random.rand(m)
    bbad = np.random.rand(m + 1)
    boundsbad = [(0, 1)] * (n + 1)
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=Abad, b_ub=bgood))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_ub=Agood, b_ub=bbad))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=Abad, b_eq=bgood))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, A_eq=Agood, b_eq=bbad))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, bounds=boundsbad))
    assert_raises(ValueError, _clean_inputs, _LPProblem(c=c, bounds=[[1, 2], [2, 3], [3, 4], [4, 5, 6]]))


def test_type_errors():
    lp = _LPProblem(
        c=[1, 2],
        A_ub=np.array([[1, 1], [2, 2]]),
        b_ub=np.array([1, 1]),
        A_eq=np.array([[1, 1], [2, 2]]),
        b_eq=np.array([1, 1]),
        bounds=[(0, 1)]
    )
    bad = "hello"

    assert_raises(TypeError, _clean_inputs, lp._replace(c=bad))
    assert_raises(TypeError, _clean_inputs, lp._replace(A_ub=bad))
    assert_raises(TypeError, _clean_inputs, lp._replace(b_ub=bad))
    assert_raises(TypeError, _clean_inputs, lp._replace(A_eq=bad))
    assert_raises(TypeError, _clean_inputs, lp._replace(b_eq=bad))

    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=bad))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds="hi"))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=["hi"]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[("hi")]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, "")]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2), (1, "")]))
    assert_raises(TypeError, _clean_inputs, lp._replace(bounds=[(1, date(2020, 2, 29))]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[[[1, 2]]]))


def test_non_finite_errors():
    lp = _LPProblem(
        c=[1, 2],
        A_ub=np.array([[1, 1], [2, 2]]),
        b_ub=np.array([1, 1]),
        A_eq=np.array([[1, 1], [2, 2]]),
        b_eq=np.array([1, 1]),
        bounds=[(0, 1)]
    )
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[0, None]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[np.inf, 0]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[0, -np.inf]))
    assert_raises(ValueError, _clean_inputs, lp._replace(c=[np.nan, 0]))

    assert_raises(ValueError, _clean_inputs, lp._replace(A_ub=[[1, 2], [None, 1]]))
    assert_raises(ValueError, _clean_inputs, lp._replace(b_ub=[np.inf, 1]))
    assert_raises(ValueError, _clean_inputs, lp._replace(A_eq=[[1, 2], [1, -np.inf]]))
    assert_raises(ValueError, _clean_inputs, lp._replace(b_eq=[1, np.nan]))


def test__clean_inputs1():
    lp = _LPProblem(
        c=[1, 2],
        A_ub=[[1, 1], [2, 2]],
        b_ub=[1, 1],
        A_eq=[[1, 1], [2, 2]],
        b_eq=[1, 1],
        bounds=None
    )

    lp_cleaned = _clean_inputs(lp)

    assert_allclose(lp_cleaned.c, np.array(lp.c))
    assert_allclose(lp_cleaned.A_ub, np.array(lp.A_ub))
    assert_allclose(lp_cleaned.b_ub, np.array(lp.b_ub))
    assert_allclose(lp_cleaned.A_eq, np.array(lp.A_eq))
    assert_allclose(lp_cleaned.b_eq, np.array(lp.b_eq))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)

    assert_(lp_cleaned.c.shape == (2,), "")
    assert_(lp_cleaned.A_ub.shape == (2, 2), "")
    assert_(lp_cleaned.b_ub.shape == (2,), "")
    assert_(lp_cleaned.A_eq.shape == (2, 2), "")
    assert_(lp_cleaned.b_eq.shape == (2,), "")


def test__clean_inputs2():
    lp = _LPProblem(
        c=1,
        A_ub=[[1]],
        b_ub=1,
        A_eq=[[1]],
        b_eq=1,
        bounds=(0, 1)
    )

    lp_cleaned = _clean_inputs(lp)

    assert_allclose(lp_cleaned.c, np.array(lp.c))
    assert_allclose(lp_cleaned.A_ub, np.array(lp.A_ub))
    assert_allclose(lp_cleaned.b_ub, np.array(lp.b_ub))
    assert_allclose(lp_cleaned.A_eq, np.array(lp.A_eq))
    assert_allclose(lp_cleaned.b_eq, np.array(lp.b_eq))
    assert_equal(lp_cleaned.bounds, [(0, 1)])

    assert_(lp_cleaned.c.shape == (1,), "")
    assert_(lp_cleaned.A_ub.shape == (1, 1), "")
    assert_(lp_cleaned.b_ub.shape == (1,), "")
    assert_(lp_cleaned.A_eq.shape == (1, 1), "")
    assert_(lp_cleaned.b_eq.shape == (1,), "")


def test__clean_inputs3():
    lp = _LPProblem(
        c=[[1, 2]],
        A_ub=np.random.rand(2, 2),
        b_ub=[[1], [2]],
        A_eq=np.random.rand(2, 2),
        b_eq=[[1], [2]],
        bounds=[(0, 1)]
    )

    lp_cleaned = _clean_inputs(lp)

    assert_allclose(lp_cleaned.c, np.array([1, 2]))
    assert_allclose(lp_cleaned.b_ub, np.array([1, 2]))
    assert_allclose(lp_cleaned.b_eq, np.array([1, 2]))
    assert_equal(lp_cleaned.bounds, [(0, 1)] * 2)

    assert_(lp_cleaned.c.shape == (2,), "")
    assert_(lp_cleaned.b_ub.shape == (2,), "")
    assert_(lp_cleaned.b_eq.shape == (2,), "")


def test_bad_bounds():
    lp = _LPProblem(c=[1, 2])

    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=(1, 2, 2)))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2, 2)]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2), (1, 2, 2)]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2), (1, 2), (1, 2)]))

    lp = _LPProblem(c=[1, 2, 3, 4])

    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2, 3, 4), (1, 2, 3, 4)]))


def test_good_bounds():
    lp = _LPProblem(c=[1, 2])

    lp_cleaned = _clean_inputs(lp)  # lp.bounds is None by default
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[]))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[[]]))
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 2)

    lp_cleaned = _clean_inputs(lp._replace(bounds=(1, 2)))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 2)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, 2)]))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 2)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, None)]))
    assert_equal(lp_cleaned.bounds, [(1, np.inf)] * 2)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, 1)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, 1)] * 2)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, None), (-np.inf, None)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, np.inf)] * 2)

    lp = _LPProblem(c=[1, 2, 3, 4])

    lp_cleaned = _clean_inputs(lp)  # lp.bounds is None by default
    assert_equal(lp_cleaned.bounds, [(0, np.inf)] * 4)

    lp_cleaned = _clean_inputs(lp._replace(bounds=(1, 2)))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 4)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, 2)]))
    assert_equal(lp_cleaned.bounds, [(1, 2)] * 4)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[(1, None)]))
    assert_equal(lp_cleaned.bounds, [(1, np.inf)] * 4)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, 1)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, 1)] * 4)

    lp_cleaned = _clean_inputs(lp._replace(bounds=[(None, None), (-np.inf, None), (None, np.inf), (-np.inf, np.inf)]))
    assert_equal(lp_cleaned.bounds, [(-np.inf, np.inf)] * 4)
