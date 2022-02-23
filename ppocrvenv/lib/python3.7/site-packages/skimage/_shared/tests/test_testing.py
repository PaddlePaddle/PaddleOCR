""" Testing decorators module
"""

from numpy.testing import assert_equal
from skimage._shared.testing import doctest_skip_parser, test_parallel
from skimage._shared import testing

from skimage._shared._warnings import expected_warnings
from warnings import warn


def test_skipper():
    def f():
        pass

    class c():

        def __init__(self):
            self.me = "I think, therefore..."

    docstring = \
        """ Header

            >>> something # skip if not HAVE_AMODULE
            >>> something + else
            >>> a = 1 # skip if not HAVE_BMODULE
            >>> something2   # skip if HAVE_AMODULE
        """
    f.__doc__ = docstring
    c.__doc__ = docstring

    global HAVE_AMODULE, HAVE_BMODULE
    HAVE_AMODULE = False
    HAVE_BMODULE = True

    f2 = doctest_skip_parser(f)
    c2 = doctest_skip_parser(c)
    assert f is f2
    assert c is c2

    expected = \
        """ Header

            >>> something # doctest: +SKIP
            >>> something + else
            >>> a = 1
            >>> something2
        """
    assert_equal(f2.__doc__, expected)
    assert_equal(c2.__doc__, expected)

    HAVE_AMODULE = True
    HAVE_BMODULE = False
    f.__doc__ = docstring
    c.__doc__ = docstring
    f2 = doctest_skip_parser(f)
    c2 = doctest_skip_parser(c)

    assert f is f2
    expected = \
        """ Header

            >>> something
            >>> something + else
            >>> a = 1 # doctest: +SKIP
            >>> something2   # doctest: +SKIP
        """
    assert_equal(f2.__doc__, expected)
    assert_equal(c2.__doc__, expected)

    del HAVE_AMODULE
    f.__doc__ = docstring
    c.__doc__ = docstring
    with testing.raises(NameError):
        doctest_skip_parser(f)
    with testing.raises(NameError):
        doctest_skip_parser(c)


def test_test_parallel():
    state = []

    @test_parallel()
    def change_state1():
        state.append(None)
    change_state1()
    assert len(state) == 2

    @test_parallel(num_threads=1)
    def change_state2():
        state.append(None)
    change_state2()
    assert len(state) == 3

    @test_parallel(num_threads=3)
    def change_state3():
        state.append(None)
    change_state3()
    assert len(state) == 6


def test_parallel_warning():
    @test_parallel()
    def change_state_warns_fails():
        warn("Test warning for test parallel", stacklevel=2)

    with expected_warnings(['Test warning for test parallel']):
        change_state_warns_fails()

    @test_parallel(warnings_matching=['Test warning for test parallel'])
    def change_state_warns_passes():
        warn("Test warning for test parallel", stacklevel=2)

    change_state_warns_passes()


def test_expected_warnings_noop():
    # This will ensure the line beolow it behaves like a no-op
    with expected_warnings(['Expected warnings test']):

        # This should behave as a no-op
        with expected_warnings(None):
            warn('Expected warnings test')

